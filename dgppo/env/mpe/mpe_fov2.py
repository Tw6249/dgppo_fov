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
        "comm_radius": 1.0,           # Maximum communication distance
        "n_obs": 0,                   # Number of obstacles (not used in this environment)
        "obs_radius": 0.05,           # Obstacle radius
        "default_area_size": 2.0,     # Default area size
        "dist2goal": 0.1,             # Distance threshold to goal
        "alpha_max": 60,              # Maximum communication angle in degrees
        "n_leaders": 2,               # Number of leader agents
        "energy_penalty_weight": 0.001  # Weight for energy efficiency penalty
    }
    
    @property
    def node_dim(self) -> int:
        return 10  # state_dim (6) + indicator: leader(0001), follower(0010), goal(0100), obstacle(1000)
    
    @property
    def edge_dim(self) -> int:
        return 7  # x_rel, y_rel, psi_rel, vx_rel, vy_rel, omega_rel, is_initial_edge(1)
    
    def reset(self, key: Array) -> GraphsTuple:
        """Reset the environment, placing agents in a chain topology with explicit neighbor tracking."""
        
        # Split the key for different random operations
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        
        # Central position for formations (center of the area)
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # Formation radius - a reasonable fraction of the area size
        formation_radius = min(self.area_size * 0.3, self.params["comm_radius"] * 0.8)
        
        # Initialize agent positions array
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # Create a chain formation
        for i in range(self.num_agents):
            # Position along the chain
            progress = i / (self.num_agents - 1) if self.num_agents > 1 else 0.5
            pos_x = center_pos[0] - formation_radius + 2 * formation_radius * progress
            pos_y = center_pos[1]
            
            agent_positions = agent_positions.at[i].set(jnp.array([pos_x, pos_y]))
        
        # Add some small random noise to y position to avoid perfect line
        key, noise_key = jr.split(key)
        y_noise = jr.uniform(noise_key, (self.num_agents, 1), minval=-0.05, maxval=0.05)
        agent_positions = agent_positions.at[:, 1].add(y_noise[:, 0])
        
        # Set orientations to face neighbors to ensure communication
        agent_orientations = jnp.zeros(self.num_agents)
        
        # For chain topology, each agent faces towards the next agent, except the last one
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
        
        # Empty obstacle array with right dimensions
        n_obs = int(self.params["n_obs"])
        if n_obs > 0:
            obs = jnp.zeros((n_obs, self.state_dim))
        else:
            obs = jnp.zeros((0, self.state_dim))
        
        # Explicitly define initial neighbors in a chain topology
        # Each agent i is connected to agent i+1, last agent connected to none (or first to form a cycle)
        initial_links = jnp.arange(1, self.num_agents)  # [1, 2, ..., n-1]
        initial_links = jnp.pad(initial_links, (0, 1), constant_values=-1)  # padding last agent
        
        # If we want a cycle, connect the last agent back to the first
        form_cycle = True
        if form_cycle and self.num_agents > 2:
            initial_links = initial_links.at[self.num_agents-1].set(0)

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
        env_state = graph.global_state
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
        Simplified reward function, where all agents share leader goal rewards.
        Communication maintenance is handled by the cost function.
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        
        reward = jnp.zeros(()).astype(jnp.float32)
        
        # 1. Leaders try to reach their goals - shared by all agents
        leader_positions = agent_states[:self.num_goals, :2]
        goal_positions = goals[:, :2]
        
        # Distance from leaders to their goals
        leader_goal_dist = jnp.linalg.norm(goal_positions - leader_positions, axis=1)
        
        # Reward based on distance to goal (negative reward proportional to distance)
        leader_reward = -leader_goal_dist.mean() * 0.1
        
        # Bonus for leaders reaching goals
        leader_reward += jnp.sum(jnp.where(leader_goal_dist < self.params["dist2goal"], 1.0, 0.0)) * 0.1
        
        # 2. Energy efficiency penalty for all agents
        energy_penalty = -jnp.mean(jnp.linalg.norm(action, axis=1) ** 2) * self.params["energy_penalty_weight"]
        
        # Combine rewards
        reward = leader_reward + energy_penalty
        
        return reward
    
    def get_cost(self, graph: MPEFoV2GraphsTuple) -> Cost:
        """
        Enhanced cost function for FOV environment with explicit tracking of 
        initial communication links.
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        env_state = graph.global_state
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
        
        # Limit minimum cost to -1.0
        cost = jnp.clip(cost, a_min=-1.0)
        
        return cost
    
    def edge_blocks(self, state: MPEFoV2State) -> list[EdgeBlock]:
        """
        Enhanced edge blocks construction with explicit marking of initial communication links.
        """
        # Get communication matrix based on FoV constraints
        communication_matrix = self._get_communication_matrix(state.agent)
        
        # Get initial neighbor information
        initial_neighbors = state.initial_neighbors
        
        # Create initial link matrix (for visualization and feature enhancement)
        initial_link_matrix = jnp.zeros((self.num_agents, self.num_agents), dtype=bool)
        
        # For each agent i with valid initial neighbor j, set initial_link_matrix[i,j] = True
        def set_initial_link(i, matrix):
            j = initial_neighbors[i]
            is_valid = j != -1
            # Only set if neighbor index is valid
            return jnp.where(
                is_valid,
                matrix.at[i, j].set(True),
                matrix
            )
        
        initial_link_matrix = jax.lax.fori_loop(
            0, self.num_agents, set_initial_link, initial_link_matrix
        )
        
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
        is_initial_link = initial_link_matrix.astype(jnp.float32)
        is_initial_link = jnp.expand_dims(is_initial_link, -1)
        
        # Concatenate to form enhanced edge features 
        enhanced_state_diff = jnp.concatenate([state_diff, is_initial_link], axis=-1)
        
        # Use communication matrix as mask for agent-agent edges
        agent_agent_mask = communication_matrix
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
        Graph creation with simplified node features (no sender/receiver indicators)
        but with edge features that mark initial communication links.
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