import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Tuple, Optional, List, Dict
from functools import partial
import numpy as np
import pathlib

from dgppo.utils.graph import EdgeBlock, GraphsTuple, GetGraph
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState, Pos2d
from dgppo.env.mpe.base import MPE, MPEEnvState, MPEEnvGraphsTuple
from dgppo.env.utils import get_node_goal_rng
from dgppo.env.plot import render_mpe
from dgppo.trainer.data import Rollout


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
        
        # Initial neighbors will be set during reset
        self.neighbor_indices = jnp.ones((self.num_agents,), dtype=jnp.int32) * -1  # Initialize with -1 (no neighbors)
    
    @property
    def state_dim(self) -> int:
        return 6  # x, y, psi, vx, vy, omega
    
    @property
    def node_dim(self) -> int:
        return 10  # state_dim (6) + indicator: leader: 0001, follower: 0010, goal: 0100, obstacle: 1000
    
    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, psi_rel, vx_rel, vy_rel, omega_rel
    
    @property
    def action_dim(self) -> int:
        return 3  # x acceleration, y acceleration, angular acceleration
    
    @property
    def n_cost(self) -> int:
        return 4  # agent collisions, fov condition 1, fov condition 2, distance condition
    
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "viewing angle (i→j)", "viewing angle (j→i)", "distance"
    
    def _identify_initial_neighbors(self, agent_states: State) -> Array:
        """
        JAX-compatible version of initial neighbor identification.
        Returns an array where each element i contains the index of agent i's initial neighbor,
        or -1 if there is no initial neighbor.
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
        
        # Set self-distance to infinity (can't be your own initial neighbor)
        masked_distances = masked_distances.at[jnp.arange(len(positions)), jnp.arange(len(positions))].set(float('inf'))
        
        # For each agent, find the closest agent that can communicate
        min_dist_indices = jnp.argmin(masked_distances, axis=1)
        
        # Create mask for agents that have valid neighbors (distance < inf)
        min_distances = jnp.min(masked_distances, axis=1)
        valid_neighbor_mask = min_distances < float('inf')
        
        # Return indices or -1 for agents without valid neighbors
        initial_neighbor_indices = jnp.where(valid_neighbor_mask, min_dist_indices, -1 * jnp.ones_like(min_dist_indices))
        
        return initial_neighbor_indices

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
        """Reset the environment, placing agents in a chain topology.
        
        Generates a strongly connected communication graph by placing the agents
        in a chain formation where each agent can communicate with its neighbors.
        Carefully adjusts orientations to ensure FOV communication requirements are met.
        """
        # Split the key for different random operations
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        
        # Central position for formations (center of the area)
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # Formation radius - a reasonable fraction of the area size
        formation_radius = min(self.area_size * 0.3, self.params["comm_radius"] * 0.8)
        
        # Initialize agent positions array
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # Always create a chain formation
        # Place agents in a line
        for i in range(self.num_agents):
            # Position along the chain
            progress = i / (self.num_agents - 1)  # 0 to 1
            pos_x = center_pos[0] - formation_radius + 2 * formation_radius * progress
            pos_y = center_pos[1]
            
            agent_positions = agent_positions.at[i].set(jnp.array([pos_x, pos_y]))
        
        # Add some small random noise to y position to avoid perfect line
        # This helps with numerical stability in some calculations
        key, noise_key = jr.split(key)
        y_noise = jr.uniform(noise_key, (self.num_agents, 1), minval=-0.05, maxval=0.05)
        agent_positions = agent_positions.at[:, 1].add(y_noise[:, 0])
        
        # Set orientations to face neighbors to ensure communication
        agent_orientations = jnp.zeros(self.num_agents)
        
        # For chain topology, each agent faces towards the next agent, except the last one
        # which faces toward the previous agent
        for i in range(self.num_agents-1):
            # Calculate direction vector to next agent
            direction = agent_positions[i+1] - agent_positions[i]
            # Calculate angle (atan2 takes y, x in that order)
            angle = jnp.arctan2(direction[1], direction[0])
            agent_orientations = agent_orientations.at[i].set(angle)
        
        # Last agent faces back
        if self.num_agents > 1:
            direction = agent_positions[self.num_agents-2] - agent_positions[self.num_agents-1]
            angle = jnp.arctan2(direction[1], direction[0])
            agent_orientations = agent_orientations.at[self.num_agents-1].set(angle)
        
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
        
        # Empty obstacle array with right dimensions - using static shapes for JAX compatibility
        n_obs = int(self.params["n_obs"])  # Convert to static int value
        if n_obs > 0:
            obs = jnp.zeros((n_obs, self.state_dim))
        else:
            obs = jnp.zeros((0, self.state_dim))
        
        # Check connectivity to ensure a strongly connected graph
        comm_matrix = self._get_communication_matrix(states)
        is_connected = self._is_strongly_connected(comm_matrix)
        
        # Create initial environment state
        env_state = MPEFoVState(states, goals, obs)
        
        # Use JIT-compatible version for reset to set initial neighbors
        # Here we define the chain links: each agent i is linked to agent i+1
        # except the last agent which is linked to none or to the first agent to form a cycle
        initial_links = jnp.arange(1, self.num_agents)  # [1, 2, ..., n-1]
        initial_links = jnp.pad(initial_links, (0, 1), constant_values=-1)  # padding last agent
        
        # If we want a cycle, connect the last agent back to the first
        form_cycle = True
        if form_cycle and self.num_agents > 2:
            initial_links = initial_links.at[self.num_agents-1].set(0)
        
        self.neighbor_indices = initial_links
        
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
        x_acc = action[:, 0] * 5.0  # Scale x acceleration
        y_acc = action[:, 1] * 5.0  # Scale y acceleration
        angular_acc = action[:, 2] * 5.0  # Scale angular acceleration
        
        # Update velocities
        vx_new = vx + x_acc * self.dt
        vy_new = vy + y_acc * self.dt
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
    
    def agent_step_euler_without_clip(self, agent_states: AgentState, action: Action) -> AgentState:
        """Update agent states using Euler integration without clipping"""
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
        x_acc = action[:, 0] * 5.0  # Scale x acceleration
        y_acc = action[:, 1] * 5.0  # Scale y acceleration
        angular_acc = action[:, 2] * 5.0  # Scale angular acceleration
        
        # Update velocities
        vx_new = vx + x_acc * self.dt
        vy_new = vy + y_acc * self.dt
        omega_new = omega + angular_acc * self.dt
        
        # Update positions and orientation
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        psi_new = psi + omega * self.dt
        
        # Normalize psi to [0, 2π)
        psi_new = psi_new % (2 * jnp.pi)
        
        # Combine into new state
        new_states = jnp.stack([x_new, y_new, psi_new, vx_new, vy_new, omega_new], axis=1)
        
        return new_states
    
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
        
        # Directly apply Euler integration with clipping, without tracking out-of-bounds
        # This matches the approach used in MPESpread
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
        
        # 1. All leaders try to reach their goals
        leader_positions = agent_states[:self.num_goals, :2]
        goal_positions = goals[:, :2]
        
        # Distance from leaders to their goals
        leader_goal_dist = jnp.linalg.norm(goal_positions - leader_positions, axis=1)
        
        # Reward based on distance to goal (negative reward proportional to distance)
        reward -= leader_goal_dist.mean() * 0.1
        
        # Bonus for leaders reaching goals
        reward += jnp.sum(jnp.where(leader_goal_dist < self.params["dist2goal"], 1.0, 0.0)) * 0.1
        
        # Small penalty for actions (energy efficiency)
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001
        
        return reward
    
    def get_cost(self, graph: MPEFoVGraphsTuple) -> Cost:
        """Calculate cost for the current state using the three FOV communication conditions"""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        
        # Agent positions
        agent_pos = agent_states[:, :2]
        
        # 1. Cost for agent-agent collisions
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # Exclude self-distance
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
        # 2. Cost for initial communication link breaks
        # 移除未使用的comm_matrix
        
        # Get current connectivity conditions
        positions = agent_states[:, :2]  # x, y
        orientations = agent_states[:, 2]  # psi
        
        # Calculate distance matrix
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Calculate unit direction vectors from i to j
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # Direction vector for each agent: v_i = [cos(psi_i), sin(psi_i)]
        direction_vectors = jnp.stack([
            jnp.cos(orientations),  # cos(psi_i)
            jnp.sin(orientations)   # sin(psi_i)
        ], axis=1)
        
        # Compute dot products: v_i · e_ij
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        )
        
        # Square the dot product values: (v_i · e_ij)^2
        dot_products_squared = dot_products ** 2
        
        # Convert alpha_max to radians and calculate cos^2(alpha_max)
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max_squared = jnp.cos(alpha_max_rad) ** 2
        
        # Calculate condition 1: (v_i · e_ij)^2 - cos^2(alpha_max) >= 0
        condition1 = dot_products_squared - cos_alpha_max_squared
        
        # For j to i direction
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        rev_dot_products_squared = rev_dot_products ** 2
        
        # Calculate condition 2: (v_j · e_ji)^2 - cos^2(alpha_max) >= 0
        condition2 = rev_dot_products_squared - cos_alpha_max_squared
        
        # Calculate condition 3: d_max^2 - d_ij^2 >= 0
        d_max_squared = self.params["comm_radius"] ** 2
        distances_squared = distances ** 2
        condition3 = d_max_squared - distances_squared
        
        # 为每个智能体计算与其初始邻居之间的每个条件成本
        def process_agent(i, acc):
            neighbor_idx = self.neighbor_indices[i]
            # Only process valid neighbor indices (-1 means no initial neighbor)
            is_valid_neighbor = neighbor_idx != -1
            
            # Calculate individual condition costs
            cost_condition1 = condition1[i, neighbor_idx]
            cost_condition2 = condition2[i, neighbor_idx]
            cost_condition3 = condition3[i, neighbor_idx]
            
            # Apply valid neighbor mask
            cost_condition1 = jnp.where(is_valid_neighbor, cost_condition1, 0.0)
            cost_condition2 = jnp.where(is_valid_neighbor, cost_condition2, 0.0)
            cost_condition3 = jnp.where(is_valid_neighbor, cost_condition3, 0.0)
            
            # Calculate total for averaging
            total_cost = cost_condition1 + cost_condition2 + cost_condition3
            
            # 返回所有条件的单独成本和总成本
            return acc[0] + cost_condition1, acc[1] + cost_condition2, acc[2] + cost_condition3, acc[3] + total_cost
        
        # 计算所有智能体的连接成本总和
        init_acc = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        cost1_sum, cost2_sum, cost3_sum, total_cost_sum = jax.lax.fori_loop(
            0, self.num_agents, process_agent, init_acc
        )
        
        # Count number of valid initial links (where neighbor_indices != -1)
        n_initial_links = jnp.sum(self.neighbor_indices != -1)
        # Ensure we don't divide by zero
        n_initial_links = jnp.maximum(n_initial_links, 1)
        
        # 计算每个条件的平均成本
        avg_cost1 = cost1_sum / n_initial_links
        avg_cost2 = cost2_sum / n_initial_links
        avg_cost3 = cost3_sum / n_initial_links
        
        # Broadcast to all agents
        condition1_cost = jnp.ones((self.num_agents,)) * avg_cost1
        condition2_cost = jnp.ones((self.num_agents,)) * avg_cost2
        condition3_cost = jnp.ones((self.num_agents,)) * avg_cost3
        
        # Combine all costs into a single matrix [n_agents, 4]
        cost = jnp.stack([
            agent_cost,         # 智能体碰撞成本
            condition1_cost,    # 视野角度条件1成本（i到j方向）
            condition2_cost,    # 视野角度条件2成本（j到i方向）
            condition3_cost     # 通信距离条件成本
        ], axis=1)
        
        # Add margin as required by safe RL:
        # - If cost <= 0 (safe), subtract eps to make it more negative (more safe)
        # - If cost > 0 (unsafe), add eps to make it more positive (more unsafe)
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        
        # Limit minimum cost to -1.0 (not too negative)
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
        lower_lim = jnp.ones(3) * -1.0
        upper_lim = jnp.ones(3)
        return lower_lim, upper_lim
    
    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        """Render a video of the FOV environment with visualization of field of view"""
        if viz_opts is None:
            viz_opts = {}
            
        # Add FOV-specific visualization options
        viz_opts.update({
            "show_fov": True,
            "fov_alpha": 0.2,
            "fov_angle": self.params["alpha_max"]
        })
        
        # Call the base MPE render_video method with FOV-specific parameters
        render_mpe(
            rollout=rollout, 
            video_path=video_path, 
            side_length=self.area_size, 
            dim=2, 
            n_agent=self.num_agents,
            n_obs=self.params['n_obs'], 
            r=self.params["car_radius"], 
            obs_r=self.params['obs_radius'],
            cost_components=self.cost_components, 
            Ta_is_unsafe=Ta_is_unsafe, 
            viz_opts=viz_opts,
            n_goal=self.num_goals, 
            dpi=dpi, 
            agent_types=self.agent_types,  # Pass agent types to distinguish leaders and followers
            include_orientation=True,  # Flag to indicate agents have orientation
            **kwargs
        )
    
    def get_graph(self, env_state: MPEFoVState) -> MPEFoVGraphsTuple:
        """
        创建GraphsTuple，区分leader和follower
        """
        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + self.params["n_obs"], self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[
                    self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(env_state.goal)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, :self.state_dim].set(env_state.obs)

        # indicators - distinguish leaders and followers
        # Leader: 0001
        node_feats = node_feats.at[:self.num_goals, 9].set(1.0)
        # Follower: 0010
        node_feats = node_feats.at[self.num_goals:self.num_agents, 8].set(1.0)
        # Goal: 0100
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, 7].set(1.0)
        # Obstacle: 1000
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, 6].set(1.0)

        # node type
        node_type = -jnp.ones((self.num_agents + self.num_goals + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(MPE.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(MPE.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[self.num_agents + self.num_goals:].set(MPE.OBS)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if self.params["n_obs"] > 0:
            states = jnp.concatenate([states, env_state.obs], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded() 