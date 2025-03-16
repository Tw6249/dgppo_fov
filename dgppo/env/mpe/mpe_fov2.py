import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Tuple, Optional, List, Dict
from functools import partial
import numpy as np
import pathlib

from dgppo.utils.graph import EdgeBlock, GraphsTuple, GetGraph
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState, Pos2d
from dgppo.env.mpe.base import MPE, MPEEnvState
from dgppo.env.mpe.mpe_fov import MPEFoV, MPEFoVState
from dgppo.env.utils import get_node_goal_rng
from dgppo.env.plot import render_mpe
from dgppo.trainer.data import Rollout


class MPEFoV2State(NamedTuple):
    agent: State            # x, y, psi, vx, vy, omega
    goal: State             # x, y, 0, 0, 0, 0
    obs: State              # x, y, 0, 0, 0, 0
    initial_neighbors: Array  # [n_agents] array of indices of initial neighbor for each agent
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEFoV2GraphsTuple = GraphsTuple[State, MPEFoV2State]


class MPEFoV2(MPEFoV):
    """
    Improved Multi-agent Particle Environment with Field of View (FoV) Communication.
    
    Key Improvements:
    1. Initial neighbor relationships are explicitly encoded in edge features
    2. All agents share the leaders' goal rewards
    3. Cost function properly tracks communication constraints for initial links
    """
    
    PARAMS = {
        "car_radius": 0.05,           # Agent radius
        "comm_radius": 0.8,           # Maximum communication distance (reduced from 1.0 for more stable training)
        "n_obs": 0,                   # Number of obstacles (not used in this environment)
        "obs_radius": 0.05,           # Obstacle radius
        "default_area_size": 2.0,     # Default area size
        "dist2goal": 0.1,             # Distance threshold to goal
        "alpha_max": 45,              # Maximum communication angle in degrees (reduced from 60 for easier satisfaction)
        "n_leaders": 2,               # Number of leader agents
        "energy_penalty_weight": 0.0005  # Weight for energy efficiency penalty (reduced for better balance)
    }
    
    @property
    def node_dim(self) -> int:
        return 10  # state_dim (6) + indicator: leader(0001), follower(0010), goal(0100), obstacle(1000)
    
    @property
    def edge_dim(self) -> int:
        return 7  # x_rel, y_rel, psi_rel, vx_rel, vy_rel, omega_rel, is_initial_edge(1)
    
    def reset(self, key: Array) -> GraphsTuple:
        """Reset the environment with improved agent positioning for better learning."""
        
        # Split the key for different random operations
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        
        # Central position for formations (center of the area)
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # Formation radius - adaptive to comm_radius and area_size
        formation_radius = min(self.area_size * 0.25, self.params["comm_radius"] * 0.7)
        
        # Initialize agent positions array
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # Create a more robust formation - circular for better connectivity
        for i in range(self.num_agents):
            # Position along the circle for more even spacing
            angle = 2 * jnp.pi * i / self.num_agents
            pos_x = center_pos[0] + formation_radius * jnp.cos(angle)
            pos_y = center_pos[1] + formation_radius * jnp.sin(angle)
            
            agent_positions = agent_positions.at[i].set(jnp.array([pos_x, pos_y]))
        
        # Add small random noise to positions to break symmetry
        key, noise_key = jr.split(key)
        position_noise = jr.uniform(noise_key, (self.num_agents, 2), minval=-0.03, maxval=0.03)
        agent_positions = agent_positions + position_noise
        
        # Set orientations to initially face center for better connectivity
        agent_orientations = jnp.zeros(self.num_agents)
        
        # Each agent faces toward adjacent agents or slightly toward the center
        for i in range(self.num_agents):
            # Calculate vector to center
            vec_to_center = center_pos - agent_positions[i]
            # Calculate vector to next agent (circular arrangement)
            next_idx = (i + 1) % self.num_agents
            vec_to_next = agent_positions[next_idx] - agent_positions[i]
            
            # Weighted combination of directions (70% to next agent, 30% to center)
            combined_vec = 0.7 * vec_to_next + 0.3 * vec_to_center
            
            # Calculate angle
            angle = jnp.arctan2(combined_vec[1], combined_vec[0])
            agent_orientations = agent_orientations.at[i].set(angle)
        
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
        
        # Create goal positions in a more structured way
        key, subkey = jr.split(key)
        
        # Place goals on opposite side of formation for clear task
        goal_angles = jr.uniform(subkey, (self.num_goals,), minval=0, maxval=2*jnp.pi)
        goal_distances = jr.uniform(
            subkey, 
            (self.num_goals,),
            minval=self.area_size * 0.3,
            maxval=self.area_size * 0.45
        )
        
        goal_positions = jnp.zeros((self.num_goals, 2))
        for i in range(self.num_goals):
            goal_x = center_pos[0] + goal_distances[i] * jnp.cos(goal_angles[i])
            goal_y = center_pos[1] + goal_distances[i] * jnp.sin(goal_angles[i])
            # Ensure goals are within boundaries
            goal_x = jnp.clip(goal_x, self.params["car_radius"]*3, self.area_size - self.params["car_radius"]*3)
            goal_y = jnp.clip(goal_y, self.params["car_radius"]*3, self.area_size - self.params["car_radius"]*3)
            goal_positions = goal_positions.at[i].set(jnp.array([goal_x, goal_y]))
        
        goals = jnp.concatenate([
            goal_positions,
            jnp.zeros((self.num_goals, 4))  # psi, vx, vy, omega all zero for goals
        ], axis=1)
        
        # Empty obstacle array with right dimensions
        n_obs = int(self.params["n_obs"])
        if n_obs > 0:
            obs = jnp.zeros((n_obs, self.state_dim))
        else:
            obs = jnp.zeros((0, self.state_dim))
        
        # Define initial communication links in circular topology
        # Each agent i is connected to agents i-1 and i+1 (circular arrangement)
        initial_links = jnp.zeros(self.num_agents, dtype=jnp.int32)
        for i in range(self.num_agents):
            initial_links = initial_links.at[i].set((i + 1) % self.num_agents)  # Connect to next agent
        
        # Create initial environment state with explicit neighbor tracking
        env_state = MPEFoV2State(states, goals, obs, initial_links)
        
        return self.get_graph(env_state)
    
    def step(
            self, graph: MPEFoV2GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEFoV2GraphsTuple, Reward, Cost, Done, Info]:
        """Step function for the environment with enhanced state representation"""
        # Get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = None
        if self.params["n_obs"] > 0:
            obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])
        
        # Extract initial neighbor relationships from environment state
        env_state = graph.env_states
        initial_neighbors = env_state.initial_neighbors
        
        # Calculate next graph
        action = self.clip_action(action)
        
        # Apply Euler integration with clipping
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEFoV2State(
            next_agent_states, 
            goals, 
            obstacles if obstacles is not None else jnp.zeros((0, self.state_dim)),
            initial_neighbors
        )
        info = {}
        
        # The episode ends when reaching max_episode_steps
        done = jnp.array(False)
        
        # Calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        
        return self.get_graph(next_env_state), reward, cost, done, info
    
    def get_reward(self, graph: MPEFoV2GraphsTuple, action: Action) -> Reward:
        """
        Enhanced reward function with gradual goal rewards and balanced energy penalties.
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        
        reward = jnp.zeros(()).astype(jnp.float32)
        
        # 1. Leaders try to reach their goals - shared by all agents
        leader_positions = agent_states[:self.num_goals, :2]
        goal_positions = goals[:, :2]
        
        # Distance from leaders to their goals
        leader_goal_dist = jnp.linalg.norm(goal_positions - leader_positions, axis=1)
        
        # Enhanced gradual reward based on distance to goal
        # Small negative reward proportional to distance
        progress_reward = -0.05 * leader_goal_dist.mean()
        
        # Larger rewards for getting closer to goals (gradually increasing)
        close_reward = jnp.sum(jnp.exp(-2.0 * leader_goal_dist)) * 0.2
        
        # Bonus reward for reaching goals
        goal_reached_reward = jnp.sum(jnp.where(leader_goal_dist < self.params["dist2goal"], 0.5, 0.0))
        
        # Combine leader rewards
        leader_reward = progress_reward + close_reward + goal_reached_reward
        
        # 2. Energy efficiency penalty - scaled down for balance
        energy_penalty = -jnp.mean(jnp.linalg.norm(action, axis=1) ** 2) * self.params["energy_penalty_weight"] * 0.5
        
        # Combine rewards
        reward = leader_reward + energy_penalty
        
        return reward
    
    def get_cost(self, graph: MPEFoV2GraphsTuple) -> Cost:
        """
        Enhanced cost function for FOV environment with explicit tracking of 
        initial communication links.
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        env_state = graph.env_states
        initial_neighbors = env_state.initial_neighbors
        
        # Agent positions
        agent_pos = agent_states[:, :2]
        
        # 1. Cost for agent-agent collisions
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # Exclude self-distance
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
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
        
        # Convert alpha_max to radians
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # Simplified visibility check - instead of squared formulation
        # 1: Check if angles are within FoV for i->j
        condition1 = dot_products - cos_alpha_max
        
        # 2: Check for j->i direction
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        condition2 = rev_dot_products - cos_alpha_max
        
        # 3: Check for maximum communication distance
        condition3 = self.params["comm_radius"] - distances
        
        # For each agent, calculate link maintenance costs for initial neighbors
        def process_agent_cost(i, acc):
            neighbor_idx = initial_neighbors[i]
            # Only process valid neighbor indices (-1 means no initial neighbor)
            is_valid_neighbor = neighbor_idx != -1
            
            # Calculate individual condition costs
            cost_condition1 = jnp.where(
                is_valid_neighbor,
                -condition1[i, neighbor_idx],  # Negative to make it a cost when constraint is violated
                0.0
            )
            
            cost_condition2 = jnp.where(
                is_valid_neighbor,
                -condition2[i, neighbor_idx],
                0.0
            )
            
            cost_condition3 = jnp.where(
                is_valid_neighbor,
                -condition3[i, neighbor_idx],
                0.0
            )
            
            # Return individual condition costs
            return acc[0] + cost_condition1, acc[1] + cost_condition2, acc[2] + cost_condition3
        
        # Calculate total communication costs
        init_acc = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        cost1_sum, cost2_sum, cost3_sum = jax.lax.fori_loop(
            0, self.num_agents, process_agent_cost, init_acc
        )
        
        # Count number of valid initial links (where initial_neighbors != -1)
        n_initial_links = jnp.sum(initial_neighbors != -1)
        # Ensure we don't divide by zero
        n_initial_links = jnp.maximum(n_initial_links, 1)
        
        # Calculate average costs
        avg_cost1 = cost1_sum / n_initial_links
        avg_cost2 = cost2_sum / n_initial_links
        avg_cost3 = cost3_sum / n_initial_links
        
        # Broadcast to all agents
        condition1_cost = jnp.ones(self.num_agents) * avg_cost1
        condition2_cost = jnp.ones(self.num_agents) * avg_cost2
        condition3_cost = jnp.ones(self.num_agents) * avg_cost3
        
        # Combine all costs into a single matrix [n_agents, 4]
        cost = jnp.stack([
            agent_cost,         # Agent collision cost
            condition1_cost,    # Viewing angle condition 1 cost (i to j)
            condition2_cost,    # Viewing angle condition 2 cost (j to i)
            condition3_cost     # Communication distance condition cost
        ], axis=1)
        
        # Add margin as required by safe RL
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        
        # Limit minimum cost to -1.0 and maximum to 1.0 (add upper bound)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        
        return cost
    
    def edge_blocks(self, state: MPEFoV2State) -> list[EdgeBlock]:
        """
        Edge blocks construction with explicit marking of initial communication links.
        """
        # Get communication matrix based on FoV constraints
        communication_matrix = self._get_communication_matrix(state.agent)
        
        # Get initial neighbor information for prioritizing connections
        initial_neighbors = state.initial_neighbors
        
        # Create initial neighbor mask for identifying initial links
        initial_neighbor_mask = jnp.zeros((self.num_agents, self.num_agents), dtype=bool)
        
        # For each agent i with valid initial neighbor j, set initial_neighbor_mask[i,j] = True
        def set_initial_neighbor(i, mask):
            j = initial_neighbors[i]
            is_valid = j != -1
            return jnp.where(
                is_valid,
                mask.at[i, j].set(True),
                mask
            )
        
        initial_neighbor_mask = jax.lax.fori_loop(
            0, self.num_agents, set_initial_neighbor, initial_neighbor_mask
        )
        
        # Enhance communication matrix to prioritize initial links
        # Make sure initial links are always included in communication
        enhanced_comm_matrix = jnp.logical_or(communication_matrix, initial_neighbor_mask)
        
        # Agent-agent connections
        agent_pos = state.agent[:, :2]
        agent_psi = state.agent[:, 2:3]
        agent_vel = state.agent[:, 3:]
        
        # Create state difference
        pos_diff = jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0)  # [i, j]: i -> j
        psi_diff = jnp.expand_dims(agent_psi, 1) - jnp.expand_dims(agent_psi, 0)
        vel_diff = jnp.expand_dims(agent_vel, 1) - jnp.expand_dims(agent_vel, 0)
        
        # Create basic state difference features
        state_diff = jnp.concatenate([pos_diff, psi_diff, vel_diff], axis=-1)
        
        # Add feature dimension indicating if this is an initial communication link
        is_initial_link = initial_neighbor_mask.astype(jnp.float32)
        is_initial_link = jnp.expand_dims(is_initial_link, -1)
        
        # Concatenate to form enhanced edge features with initial link information
        enhanced_state_diff = jnp.concatenate([state_diff, is_initial_link], axis=-1)
        
        # Use enhanced communication matrix as mask for agent-agent edges
        agent_agent_mask = enhanced_comm_matrix
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(enhanced_state_diff, agent_agent_mask, id_agent, id_agent)
        
        # Agent-goal connections (only for leader agents)
        leader_mask = jnp.zeros((self.num_agents, self.num_goals))
        leader_mask = leader_mask.at[:self.num_goals].set(jnp.eye(self.num_goals))
        
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.num_goals)
        
        # Create basic agent-goal features
        agent_goal_feats = jnp.zeros((self.num_agents, self.num_goals, self.state_dim))
        
        for i in range(self.num_goals):
            agent_state = state.agent[i]
            goal_state = state.goal[i]
            agent_goal_feats = agent_goal_feats.at[i, i].set(agent_state - goal_state)
        
        # Add dummy dimension for is_initial_link (always False for agent-goal edges)
        agent_goal_is_initial = jnp.zeros((self.num_agents, self.num_goals, 1))
        
        # Combine features
        enhanced_agent_goal_feats = jnp.concatenate([agent_goal_feats, agent_goal_is_initial], axis=-1)
        
        agent_goal_edges = EdgeBlock(
            enhanced_agent_goal_feats, leader_mask, id_agent, id_goal
        )
        
        return [agent_agent_edges, agent_goal_edges]
    
    def get_graph(self, env_state: MPEFoV2State) -> MPEFoV2GraphsTuple:
        """
        Simplified graph creation with cleaner node and edge features.
        """
        # Create node features
        node_feats = jnp.zeros((self.num_agents + self.num_goals + self.params["n_obs"], self.node_dim))
        
        # Set state components
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[
                    self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(env_state.goal)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, :self.state_dim].set(env_state.obs)

        # Set node type indicators
        # Leader: 0001
        node_feats = node_feats.at[:self.num_goals, 9].set(1.0)
        # Follower: 0010
        node_feats = node_feats.at[self.num_goals:self.num_agents, 8].set(1.0)
        # Goal: 0100
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, 7].set(1.0)
        # Obstacle: 1000
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, 6].set(1.0)

        # Node type
        node_type = -jnp.ones((self.num_agents + self.num_goals + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(self.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(self.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[self.num_agents + self.num_goals:].set(self.OBS)

        # Edges
        edge_blocks = self.edge_blocks(env_state)

        # Create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if self.params["n_obs"] > 0:
            states = jnp.concatenate([states, env_state.obs], axis=0)
            
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()
        
    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        """Render a video with enhanced visualization of initial communication links"""
        if viz_opts is None:
            viz_opts = {}
            
        # Add FOV-specific visualization options
        viz_opts.update({
            "show_fov": True,
            "fov_alpha": 0.2,
            "fov_angle": self.params["alpha_max"],
            "show_initial_links": True  # New option to visualize initial links
        })
        
        # Call the base render_video method with enhanced options
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
            agent_types=self.agent_types,
            include_orientation=True,
            **kwargs
        ) 