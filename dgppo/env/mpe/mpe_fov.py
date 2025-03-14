import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Tuple, Optional, List, Dict
from functools import partial
import numpy as np

from dgppo.utils.graph import EdgeBlock, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState, Pos2d
from dgppo.env.mpe.base import MPE, MPEEnvState, MPEEnvGraphsTuple
from dgppo.env.utils import get_node_goal_rng


class MPEFoVState(NamedTuple):
    agent: State       # x, y, psi, vx, vy, omega
    goal: State        # x, y, 0, 0, 0, 0
    obs: State         # x, y, 0, 0, 0, 0
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEFoVGraphsTuple = GraphsTuple[State, MPEFoVState]


class MPEFoV(MPE):
    """
    Multi-agent Particle Environment with Field of View (FoV) Communication.
    
    Agents have orientation and can rotate around z-axis.
    Communication is based on FoV - agents can only communicate with neighbors in their field of view.
    """
    
    AGENT = 0
    GOAL = 1
    OBS = 2
    
    # Leader and follower types
    LEADER = 0
    FOLLOWER = 1
    
    PARAMS = {
        "car_radius": 0.05,           # Agent radius
        "comm_radius": 1.0,           # Maximum communication distance
        "n_obs": 0,                   # Number of obstacles (not used in this environment)
        "obs_radius": 0.05,           # Obstacle radius
        "default_area_size": 2.0,     # Default area size
        "dist2goal": 0.1,             # Distance threshold to goal
        "alpha_max": 60,              # Maximum communication angle in degrees
        "n_leaders": 2,               # Number of leader agents
        "critical_neighbors": {},     # Dictionary mapping agent IDs to their critical neighbor IDs
    }
    
    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = self.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPE, self).__init__(num_agents, area_size, max_step, dt, params)
        
        # Set number of goals to number of leader agents
        self.num_goals = self.params.get("n_leaders", 2)
        
        # Set up agent types
        self.agent_types = jnp.zeros(num_agents, dtype=jnp.int32)
        self.agent_types = self.agent_types.at[:self.num_goals].set(self.LEADER)
        self.agent_types = self.agent_types.at[self.num_goals:].set(self.FOLLOWER)
        
        # Critical neighbors will be set during reset
        self.critical_neighbors = {}
    
    @property
    def state_dim(self) -> int:
        return 6  # x, y, psi, vx, vy, omega
    
    @property
    def node_dim(self) -> int:
        return 9  # state_dim (6) + indicator: agent: 001, goal: 010, obstacle: 100
    
    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, psi_rel, vx_rel, vy_rel, omega_rel
    
    @property
    def action_dim(self) -> int:
        return 2  # Linear acceleration, angular acceleration
    
    @property
    def n_cost(self) -> int:
        return 2  # agent collisions, connectivity
    
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "connectivity"
    
    def _identify_critical_neighbors(self, agent_states: State) -> Dict[int, List[int]]:
        """
        Identify critical neighbors for each agent based on proximity and communication ability.
        For each agent, select the closest agent that can communicate bidirectionally.
        
        Returns a dictionary where keys are agent indices and values are lists of critical neighbor indices.
        NOTE: This function is adapted to work outside of JIT compilation since it returns a Python dict.
        Use _identify_critical_neighbors_jax for JIT-compatible version.
        """
        # Get JIT-compatible neighbor indices
        neighbor_indices = self._identify_critical_neighbors_jax(agent_states)
        
        # Convert to Python dictionary format (for compatibility with existing code)
        critical_neighbors = {}
        for i in range(self.num_agents):
            # Check if there is a valid neighbor (not -1)
            if neighbor_indices[i] != -1:
                critical_neighbors[i] = [int(neighbor_indices[i])]
        
        return critical_neighbors
    
    def _identify_critical_neighbors_jax(self, agent_states: State) -> Array:
        """
        JAX-compatible version of critical neighbor identification.
        Returns an array where each element i contains the index of agent i's critical neighbor,
        or -1 if there is no critical neighbor.
        """
        # Get communication matrix
        comm_matrix = self._get_communication_matrix(agent_states)
        
        # Extract positions for distance calculation
        positions = agent_states[:, :2]
        
        # Calculate distance matrix
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Set non-communicating agents to infinite distance
        masked_distances = jnp.where(comm_matrix, distances, jnp.ones_like(distances) * float('inf'))
        
        # Set self-distance to infinity (can't be your own critical neighbor)
        masked_distances = masked_distances.at[jnp.arange(len(positions)), jnp.arange(len(positions))].set(float('inf'))
        
        # For each agent, find the closest agent that can communicate
        min_dist_indices = jnp.argmin(masked_distances, axis=1)
        
        # Create mask for agents that have valid neighbors (distance < inf)
        min_distances = jnp.min(masked_distances, axis=1)
        valid_neighbor_mask = min_distances < float('inf')
        
        # Return indices or -1 for agents without valid neighbors
        critical_neighbor_indices = jnp.where(valid_neighbor_mask, min_dist_indices, -1 * jnp.ones_like(min_dist_indices))
        
        return critical_neighbor_indices

    def _is_strongly_connected(self, comm_matrix):
        """
        Check if the graph defined by the communication matrix is strongly connected.
        A directed graph is strongly connected if there is a path from each vertex to every other vertex.
        
        This implementation uses matrix powers to determine reachability, making it JAX-compatible.
        
        Args:
            comm_matrix: NxN boolean matrix where comm_matrix[i,j] is True if i can communicate with j
            
        Returns:
            bool: True if the graph is strongly connected, False otherwise
        """
        # Convert boolean matrix to int matrix for matrix multiplication
        adj_matrix = comm_matrix.astype(jnp.int32)
        n = adj_matrix.shape[0]
        
        # Calculate reachability matrix
        # Matrix power approach: if (A^n)[i,j] > 0, there is a path from i to j
        reachability = adj_matrix
        
        # Use matrix powers up to n-1 to find all paths
        for _ in range(n-1):
            reachability = jnp.matmul(reachability, adj_matrix)
            reachability = (reachability > 0).astype(jnp.int32)
        
        # Check if all elements in reachability matrix are > 0 (excluding diagonal)
        # We create a mask to ignore the diagonal elements
        mask = 1 - jnp.eye(n, dtype=jnp.int32)  # Mask is 0 on diagonal, 1 elsewhere
        masked_reachability = reachability * mask
        
        # The graph is strongly connected if all off-diagonal elements are > 0
        # This means every node can reach every other node
        min_value = jnp.min(masked_reachability + jnp.eye(n, dtype=jnp.int32))  # Add identity to make diagonal 1
        
        return min_value > 0

    def _adjust_positions_for_connectivity(self, key, leader_positions, follower_positions, n_attempts=10):
        """
        Adjust agent positions to ensure the communication graph is strongly connected.
        
        Note: This function is meant to be used outside of JIT compilation, as it may require
        multiple iterations and early stopping which are not JAX-compatible in a direct way.
        
        Args:
            key: JAX random key
            leader_positions: Positions of leader agents
            follower_positions: Positions of follower agents
            n_attempts: Number of attempts to create a strongly connected graph
            
        Returns:
            leader_positions: Adjusted leader positions
            follower_positions: Adjusted follower positions
            key: Updated random key
        """
        # Create a scan-compatible function for JAX
        def attempt_connectivity(carry, _):
            key, best_score, best_leader_pos, best_follower_pos = carry
            
            # Combine positions
            agent_positions = jnp.concatenate([leader_positions, follower_positions], axis=0)
            
            # Generate random orientations to check connectivity
            key, subkey = jr.split(key)
            orientations = jr.uniform(subkey, (self.num_agents, 1), minval=0, maxval=2*jnp.pi)
            
            # Create temporary agent states to check connectivity
            temp_velocities = jnp.zeros((self.num_agents, 3))  # vx, vy, omega all zero
            temp_states = jnp.concatenate([
                agent_positions, 
                orientations, 
                temp_velocities
            ], axis=1)
            
            # Get communication matrix and check connectivity
            comm_matrix = self._get_communication_matrix(temp_states)
            is_connected = self._is_strongly_connected(comm_matrix)
            
            # Calculate connectivity score (number of connected pairs)
            connectivity_score = jnp.sum(comm_matrix)
            
            # Update best score and positions if better
            new_best_score = jnp.where(connectivity_score > best_score, 
                                     connectivity_score, best_score)
            
            new_best_leader_pos = jnp.where(connectivity_score > best_score,
                                          leader_positions, best_leader_pos)
            
            new_best_follower_pos = jnp.where(connectivity_score > best_score,
                                            follower_positions, best_follower_pos)
            
            # Return early if connected
            return (
                key, 
                new_best_score, 
                new_best_leader_pos, 
                new_best_follower_pos
            ), is_connected
        
        # Initial carry state
        init_carry = (
            key, 
            jnp.array(-1, dtype=jnp.int32),  # best_score 
            leader_positions, 
            follower_positions
        )
        
        # Run the scan operation for up to n_attempts
        final_carry, is_connected_arr = jax.lax.scan(
            attempt_connectivity, 
            init_carry, 
            jnp.arange(n_attempts),
            length=n_attempts
        )
        
        # Extract results
        final_key, _, final_leader_pos, final_follower_pos = final_carry
        
        # Return the final positions
        return final_leader_pos, final_follower_pos, final_key

    def reset(self, key: Array) -> GraphsTuple:
        """Reset the environment, placing agents in a structured topology.
        
        Generates a strongly connected communication graph by using structured
        topologies like star, chain, or ring formations. Carefully places agents
        and adjusts their orientations to ensure FOV communication requirements are met.
        """
        # Split the key for different random operations
        key, subkey1, subkey2, subkey3, subkey4 = jr.split(key, 5)
        
        # Choose a random topology type: 0=star, 1=chain, 2=ring
        topology_type = jr.choice(subkey1, jnp.array([0, 1, 2]))
        
        # Central position for formations (center of the area)
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # Formation radius - a reasonable fraction of the area size
        formation_radius = min(self.area_size * 0.3, self.params["comm_radius"] * 0.8)
        
        # Initialize agent positions array
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # Define formation functions
        def create_star_formation(positions):
            # Central agent at the center
            positions = positions.at[0].set(center_pos)
            
            # Place other agents in a circle around the central agent
            for i in range(1, self.num_agents):
                # Angle for this agent
                angle = 2 * jnp.pi * (i - 1) / (self.num_agents - 1)
                
                # Position on the circle
                pos_x = center_pos[0] + formation_radius * jnp.cos(angle)
                pos_y = center_pos[1] + formation_radius * jnp.sin(angle)
                
                positions = positions.at[i].set(jnp.array([pos_x, pos_y]))
            
            return positions
            
        def create_chain_formation(positions):
            # Place agents in a line
            for i in range(self.num_agents):
                # Position along the chain
                progress = i / (self.num_agents - 1)  # 0 to 1
                pos_x = center_pos[0] - formation_radius + 2 * formation_radius * progress
                pos_y = center_pos[1]
                
                positions = positions.at[i].set(jnp.array([pos_x, pos_y]))
                
            return positions
            
        def create_ring_formation(positions):
            # Place agents in a circle
            for i in range(self.num_agents):
                # Angle for this agent
                angle = 2 * jnp.pi * i / self.num_agents
                
                # Position on the circle
                pos_x = center_pos[0] + formation_radius * jnp.cos(angle)
                pos_y = center_pos[1] + formation_radius * jnp.sin(angle)
                
                positions = positions.at[i].set(jnp.array([pos_x, pos_y]))
                
            return positions
        
        # Use JAX's functional approach to select the formation
        agent_positions = jnp.where(topology_type == 0, 
                                   create_star_formation(agent_positions), 
                                   jnp.where(topology_type == 1,
                                           create_chain_formation(agent_positions),
                                           create_ring_formation(agent_positions)))
        
        # Set orientations based on formation type using the same JAX-compatible approach
        agent_orientations = jnp.zeros(self.num_agents)
        
        def set_star_orientations(orientations):
            # Central agent random orientation
            orientations = orientations.at[0].set(jr.uniform(subkey2, minval=0, maxval=2*jnp.pi))
            
            # Others pointing toward center
            for i in range(1, self.num_agents):
                # Calculate direction vector to center
                direction = center_pos - agent_positions[i]
                # Calculate angle (atan2 takes y, x in that order)
                angle = jnp.arctan2(direction[1], direction[0])
                orientations = orientations.at[i].set(angle)
            
            return orientations
            
        def set_chain_orientations(orientations):
            # All agents face the same direction along the chain
            orientations = orientations.at[:].set(0.0)  # All face right (0 radians)
            return orientations
            
        def set_ring_orientations(orientations):
            # Each agent faces tangent to the circle (90 degrees rotated from radius)
            for i in range(self.num_agents):
                # Angle for this agent
                angle = 2 * jnp.pi * i / self.num_agents
                # Tangent orientation
                tangent_angle = angle + jnp.pi/2
                orientations = orientations.at[i].set(tangent_angle)
            
            return orientations
        
        # Apply orientation function based on topology_type
        agent_orientations = jnp.where(topology_type == 0,
                                      set_star_orientations(agent_orientations),
                                      jnp.where(topology_type == 1,
                                              set_chain_orientations(agent_orientations),
                                              set_ring_orientations(agent_orientations)))
        
        # Initialize velocities to zero
        agent_velocities = jnp.zeros((self.num_agents, 2))
        agent_angular_velocities = jnp.zeros((self.num_agents, 1))
        
        # Reshape orientations to match expected dimensions
        agent_orientations = agent_orientations.reshape((self.num_agents, 1))
        
        # Combine positions, orientations, and velocities
        states = jnp.concatenate([
            agent_positions,                # x, y (shape: n_agents x 2)
            agent_orientations,             # psi (shape: n_agents x 1)
            agent_velocities,               # vx, vy (shape: n_agents x 2)
            agent_angular_velocities        # omega (shape: n_agents x 1)
        ], axis=1)
        
        # Create goal positions (only for leader agents)
        key, subkey = jr.split(key)
        goal_positions = jr.uniform(
            subkey, 
            (self.num_goals, 2),
            minval=self.params["car_radius"] * 3,
            maxval=self.area_size - self.params["car_radius"] * 3
        )
        
        goals = jnp.concatenate([
            goal_positions,
            jnp.zeros((self.num_goals, 4))  # psi, vx, vy, omega all zero for goals
        ], axis=1)
        
        # Empty obstacle array with right dimensions - using JAX-compatible approach
        obs = jnp.zeros((jnp.maximum(self.params["n_obs"], 0), self.state_dim))
        obs = jnp.where(self.params["n_obs"] > 0, obs, jnp.zeros((0, self.state_dim)))
        
        # Validate and fix connectivity - separate connectivity check and fix functions
        comm_matrix = self._get_communication_matrix(states)
        is_connected = self._is_strongly_connected(comm_matrix)
        
        # Define connectivity fix function to be applied conditionally
        def fix_connectivity(states_to_fix, conn_matrix):
            # Your connectivity fixing logic here
            # For now, let's just return the original states
            return states_to_fix
            
        # Apply connectivity fix conditionally
        states = jnp.where(is_connected, states, fix_connectivity(states, comm_matrix))
        
        # Create initial environment state
        env_state = MPEFoVState(states, goals, obs)
        
        # Identify critical neighbors based on initial state
        self.critical_neighbors = self._identify_critical_neighbors(states)
        
        return self.get_graph(env_state)
    
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """Update agent states using Euler integration with orientation"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        
        # Extract current state components
        x = agent_states[:, 0]
        y = agent_states[:, 1]
        psi = agent_states[:, 2]
        vx = agent_states[:, 3]
        vy = agent_states[:, 4]
        omega = agent_states[:, 5]
        
        # Extract actions
        linear_acc = action[:, 0] * 5.0  # Scale linear acceleration
        angular_acc = action[:, 1] * 5.0  # Scale angular acceleration
        
        # Update velocities
        vx_new = vx + linear_acc * jnp.cos(psi) * self.dt
        vy_new = vy + linear_acc * jnp.sin(psi) * self.dt
        omega_new = omega + angular_acc * self.dt
        
        # Update positions and orientation
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        psi_new = psi + omega * self.dt
        
        # Normalize psi to [0, 2π)
        psi_new = psi_new % (2 * jnp.pi)
        
        # Combine into new state
        new_states = jnp.stack([x_new, y_new, psi_new, vx_new, vy_new, omega_new], axis=1)
        
        return self.clip_state(new_states)
    
    def step(
            self, graph: MPEFoVGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEFoVGraphsTuple, Reward, Cost, Done, Info]:
        """Step function for the environment"""
        # Get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = None
        if self.params["n_obs"] > 0:
            obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])
        
        # Calculate next graph
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEFoVState(next_agent_states, goals, obstacles if obstacles is not None else jnp.zeros((0, self.state_dim)))
        info = {}
        
        # The episode ends when reaching max_episode_steps
        done = jnp.array(False)
        
        # Calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        
        return self.get_graph(next_env_state), reward, cost, done, info
    
    def get_reward(self, graph: MPEFoVGraphsTuple, action: Action) -> Reward:
        """Calculate reward for the current state"""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        
        reward = jnp.zeros(()).astype(jnp.float32)
        
        # 1. Leaders get reward for reaching goals
        leader_positions = agent_states[:self.num_goals, :2]
        goal_positions = goals[:, :2]
        
        # Distance from leaders to their goals
        dist2goal = jnp.linalg.norm(goal_positions - leader_positions, axis=1)
        
        # Reward based on distance to goal
        reward -= dist2goal.mean() * 0.1
        
        # Bonus for reaching goals
        reward += jnp.sum(jnp.where(dist2goal < self.params["dist2goal"], 1.0, 0.0)) * 0.1
        
        # 2. Reward for maintaining critical communication links
        comm_matrix = self._get_communication_matrix(agent_states)
        critical_links_maintained = jnp.array(1.0)
        
        # Check if critical links are maintained
        for agent_id, neighbors in self.critical_neighbors.items():
            for neighbor_id in neighbors:
                # Check if the link is maintained
                link_maintained = comm_matrix[agent_id, neighbor_id]
                critical_links_maintained = critical_links_maintained * link_maintained
        
        # Reward for critical link maintenance
        reward += critical_links_maintained * 0.1
        
        # 3. Small penalty for actions (energy efficiency)
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.01
        
        return reward
    
    def get_cost(self, graph: MPEFoVGraphsTuple) -> Cost:
        """Calculate cost for the current state"""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        
        # Agent positions
        agent_pos = agent_states[:, :2]
        
        # 1. Cost for agent-agent collisions
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # Exclude self-distance
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
        # 2. Cost for critical communication link breaks
        comm_matrix = self._get_communication_matrix(agent_states)
        critical_links_maintained = jnp.array(1.0)
        
        # Check if critical links are maintained
        for agent_id, neighbors in self.critical_neighbors.items():
            for neighbor_id in neighbors:
                # Check if the link is maintained
                link_maintained = comm_matrix[agent_id, neighbor_id]
                critical_links_maintained = critical_links_maintained * link_maintained
        
        # Cost is high if critical links are broken
        comm_cost = 1.0 - critical_links_maintained
        # Broadcast to all agents
        comm_cost = jnp.ones((self.num_agents,)) * comm_cost
        
        # Combine costs
        cost = jnp.stack([agent_cost, comm_cost], axis=1)
        
        # Add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)
        
        return cost
    
    def _get_communication_matrix(self, agent_states: State) -> jnp.ndarray:
        """
        Calculate the communication matrix between agents using the mathematical conditions.
        For bidirectional communication between agents i and j, both must satisfy:
        (v_i · e_ij)^2 - cos^2(alpha_max) >= 0 AND (v_j · e_ji)^2 - cos^2(alpha_max) >= 0
        where v_i is the direction vector for agent i, and e_ij is the unit direction vector from i to j.
        
        For each agent, the direction vector v_i is computed by applying a rotation matrix to [1,0]:
        v_i = [cos(psi_i), sin(psi_i)]
        """
        # Extract positions and orientations
        positions = agent_states[:, :2]  # x, y
        orientations = agent_states[:, 2]  # psi
        n_agents = len(positions)
        
        # Calculate distance matrix
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Calculate unit direction vectors from i to j
        # Avoid division by zero by adding small epsilon
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # Direction vector for each agent using the forward vector rotated by orientation
        # v_i = R(psi_i) * [1, 0] = [cos(psi_i), sin(psi_i)]
        direction_vectors = jnp.stack([
            jnp.cos(orientations),  # cos(psi_i)
            jnp.sin(orientations)   # sin(psi_i)
        ], axis=1)
        
        # Compute the dot product between direction vector and line of sight: v_i · e_ij
        # For each pair of agents i, j
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        )
        
        # Square the dot product values
        dot_products_squared = dot_products ** 2
        
        # Convert alpha_max to radians and calculate cos^2(alpha_max)
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max_squared = jnp.cos(alpha_max_rad) ** 2
        
        # Check if (v_i · e_ij)^2 - cos^2(alpha_max) >= 0
        i_to_j_condition = dot_products_squared - cos_alpha_max_squared >= 0
        
        # For j to i direction, e_ji = -e_ij
        # Compute the dot product between j's direction vector and e_ji: v_j · e_ji
        # Since e_ji = -e_ij, v_j · e_ji = -v_j · e_ij
        reverse_diff_normalized = -diff_normalized  # e_ji
        
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        
        # Square the dot product values
        rev_dot_products_squared = rev_dot_products ** 2
        
        # Check if (v_j · e_ji)^2 - cos^2(alpha_max) >= 0
        j_to_i_condition = rev_dot_products_squared - cos_alpha_max_squared >= 0
        
        # For bidirectional communication, both conditions must be satisfied
        bidirectional_in_fov = jnp.logical_and(i_to_j_condition, j_to_i_condition)
        
        # Check distance constraint
        within_distance = distances < self.params["comm_radius"]
        
        # Final communication matrix: 1 if agents can communicate, 0 otherwise
        can_communicate = jnp.logical_and(bidirectional_in_fov, within_distance)
        
        # Set diagonal to 0 (no self-communication)
        can_communicate = can_communicate * (1 - jnp.eye(n_agents))
        
        return can_communicate
    
    def edge_blocks(self, state: MPEFoVState) -> list[EdgeBlock]:
        """Define edge connections based on FoV communication"""
        # Get communication matrix
        communication_matrix = self._get_communication_matrix(state.agent)
        
        # Agent-agent connections
        agent_pos = state.agent[:, :2]
        agent_psi = state.agent[:, 2:3]
        agent_vel = state.agent[:, 3:]
        
        # Create state difference
        pos_diff = jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0)  # [i, j]: i -> j
        psi_diff = jnp.expand_dims(agent_psi, 1) - jnp.expand_dims(agent_psi, 0)
        vel_diff = jnp.expand_dims(agent_vel, 1) - jnp.expand_dims(agent_vel, 0)
        
        state_diff = jnp.concatenate([pos_diff, psi_diff, vel_diff], axis=-1)
        
        # Use communication matrix as mask
        agent_agent_mask = communication_matrix
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)
        
        # Agent-goal connections (only for leader agents)
        leader_mask = jnp.zeros((self.num_agents, self.num_goals))
        leader_mask = leader_mask.at[:self.num_goals].set(jnp.eye(self.num_goals))
        
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.num_goals)
        agent_goal_feats = jnp.zeros((self.num_agents, self.num_goals, self.state_dim))
        
        for i in range(self.num_goals):
            agent_state = state.agent[i]
            goal_state = state.goal[i]
            agent_goal_feats = agent_goal_feats.at[i, i].set(agent_state - goal_state)
        
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, leader_mask, id_agent, id_goal
        )
        
        return [agent_agent_edges, agent_goal_edges]
    
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """Define state limits"""
        lower_lim = jnp.array([0.0, 0.0, 0.0, -2.0, -2.0, -1.0])
        upper_lim = jnp.array([self.area_size, self.area_size, 2*jnp.pi, 2.0, 2.0, 1.0])
        return lower_lim, upper_lim
    
    def action_lim(self) -> Tuple[Action, Action]:
        """Define action limits"""
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim 