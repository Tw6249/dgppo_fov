import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
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
    agent: State            # x, y, psi, vx, vy, omega, cos(psi), sin(psi)
    goal: State             # x, y, 0, 0, 0, 0, 0, 0
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEFoV2GraphsTuple = GraphsTuple[State, MPEFoV2State]


class MPEFoV2(MPEFoV):
    """
    Improved Multi-agent Particle Environment with Field of View (FoV) Communication.
    
    Key Improvements:
    1. All agents share the leaders' goal rewards
    2. Cost function properly tracks communication constraints
    3. Enhanced node features with explicit orientation encoding
    """
    
    PARAMS = {
        "car_radius": 0.05,           # Agent radius
        "comm_radius": 0.8,           # Maximum communication distance (reduced from 1.0 for more stable training)
        "default_area_size": 2.0,     # Default area size
        "dist2goal": 0.1,             # Distance threshold to goal
        "alpha_max": 45,              # Maximum communication angle in degrees (reduced from 60 for easier satisfaction)
        "n_leaders": 2,               # Number of leader agents
        "energy_penalty_weight": 0.0005  # Weight for energy efficiency penalty (reduced for better balance)
    }
    
    @property
    def node_dim(self) -> int:
        return 12  # state_dim (8) + indicator: leader(0001), follower(0010), goal(0100)
    
    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, psi_rel, vx_rel, vy_rel, omega_rel
    
    @property
    def state_dim(self) -> int:
        return 8  # x, y, psi, vx, vy, omega, cos(psi), sin(psi)
    
    def reset(self, key: Array) -> GraphsTuple:
        """Reset the environment with agents in a chain topology."""
        
        # Split the key for different random operations
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        
        # Central position for formations (center of the area)
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # Calculate appropriate spacing between agents
        # Use a fraction of communication radius to ensure connectivity
        spacing = self.params["comm_radius"] * 0.7
        
        # Calculate total chain length
        chain_length = spacing * (self.num_agents - 1)
        
        # Determine start position to center the chain
        start_x = center_pos[0] - chain_length / 2
        start_y = center_pos[1]
        
        # Create random permutation for agent ordering
        key, perm_key = jr.split(key)
        agent_order = jr.permutation(perm_key, jnp.arange(self.num_agents))
        
        # Initialize agent positions array
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # Place agents in a chain with random ordering
        for i in range(self.num_agents):
            # Position along the chain
            pos_x = start_x + i * spacing
            pos_y = start_y
            
            # Set position for the agent at the permuted index
            agent_idx = agent_order[i]
            agent_positions = agent_positions.at[agent_idx].set(jnp.array([pos_x, pos_y]))
        
        # Add small random noise to y position to avoid perfect alignment
        key, noise_key = jr.split(key)
        y_noise = jr.uniform(noise_key, (self.num_agents, 1), minval=-0.05, maxval=0.05)
        agent_positions = agent_positions + jnp.concatenate([jnp.zeros((self.num_agents, 1)), y_noise], axis=1)
        
        # Set orientations to face the next agent in the chain
        agent_orientations = jnp.zeros(self.num_agents)
        
        # For each agent, calculate orientation to face the next agent in the chain
        # This ensures good visibility for communication
        for i in range(self.num_agents):
            # Find the nearest agent to the right in the chain
            # Calculate distances to all other agents
            dists = jnp.linalg.norm(agent_positions - agent_positions[i], axis=1)
            # Mask out self and agents to the left (with smaller x coordinates)
            mask = (agent_positions[:, 0] > agent_positions[i, 0]) & (dists > 0)
            
            # If there's an agent to the right, face it
            # Use lax.cond instead of if statement for JAX compatibility
            has_right_agent = jnp.any(mask).astype(jnp.int32)
            
            def true_branch(_):
                # Find the nearest agent to the right
                masked_dists = jnp.where(mask, dists, jnp.ones_like(dists) * 1e6)
                nearest_idx = jnp.argmin(masked_dists)
                
                # Calculate direction vector to the nearest agent
                direction = agent_positions[nearest_idx] - agent_positions[i]
                # Calculate angle
                angle = jnp.arctan2(direction[1], direction[0])
                return angle
            
            def false_branch(_):
                # If no agent to the right, face the last agent in the chain
                # Find the agent with the largest x coordinate
                rightmost_idx = jnp.argmax(agent_positions[:, 0])
                
                # Use another lax.cond to handle the case where i is the rightmost agent
                is_rightmost = (i == rightmost_idx).astype(jnp.int32)
                
                def not_rightmost(_):
                    direction = agent_positions[rightmost_idx] - agent_positions[i]
                    return jnp.arctan2(direction[1], direction[0])
                
                def is_rightmost_agent(_):
                    # If this is the rightmost agent, face left (toward the chain)
                    return jnp.array(jnp.pi)
                
                return lax.cond(is_rightmost, is_rightmost_agent, not_rightmost, operand=None)
            
            new_orientation = lax.cond(has_right_agent, true_branch, false_branch, operand=None)
            agent_orientations = agent_orientations.at[i].set(new_orientation)
        
        # Initialize velocities to zero
        agent_velocities = jnp.zeros((self.num_agents, 2))
        agent_angular_velocities = jnp.zeros((self.num_agents, 1))
        
        # Reshape orientations to match expected dimensions
        agent_orientations = agent_orientations.reshape((self.num_agents, 1))
        
        # Calculate cos(psi) and sin(psi)
        cos_psi = jnp.cos(agent_orientations)
        sin_psi = jnp.sin(agent_orientations)
        
        # Combine positions, orientations, velocities, and orientation features
        states = jnp.concatenate([
            agent_positions,                # x, y (shape: n_agents x 2)
            agent_orientations,             # psi (shape: n_agents x 1)
            agent_velocities,               # vx, vy (shape: n_agents x 2)
            agent_angular_velocities,       # omega (shape: n_agents x 1)
            cos_psi,                        # cos(psi) (shape: n_agents x 1)
            sin_psi                         # sin(psi) (shape: n_agents x 1)
        ], axis=1)
        
        # Create goal positions in a more structured way
        key, subkey = jr.split(key)
        
        # Place goals at random positions around the area
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
            jnp.zeros((self.num_goals, 6))  # psi, vx, vy, omega, cos(psi), sin(psi) all zero for goals
        ], axis=1)
        
        # Create initial environment state
        env_state = MPEFoV2State(states, goals)
        
        return self.get_graph(env_state)
    
    def step(
            self, graph: MPEFoV2GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEFoV2GraphsTuple, Reward, Cost, Done, Info]:
        """Step function for the environment with enhanced state representation"""
        # Get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        
        # Calculate next graph
        action = self.clip_action(action)
        
        # Apply Euler integration with clipping
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEFoV2State(
            next_agent_states, 
            goals
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
        
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        
        # Agent positions
        agent_pos = agent_states[:, :2]
        
        # 1. Cost for agent-agent collisions
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # Exclude self-distance
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
        # Get current connectivity conditions
        positions = agent_states[:, :2]  # x, y
        
        # Use precomputed cos(psi) and sin(psi) from the state
        cos_psi = agent_states[:, 6]  # cos(psi)
        sin_psi = agent_states[:, 7]  # sin(psi)
        direction_vectors = jnp.stack([cos_psi, sin_psi], axis=1)
        
        # Calculate distance matrix
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Calculate unit direction vectors from i to j
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # Compute dot products: v_i · e_ij
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        )
        
        # Convert alpha_max to radians
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # Simplified visibility check - instead of squared formulation
        # 1: Check if angles are within FoV for i->j, absolute value
        condition1 = jnp.abs(dot_products) - cos_alpha_max
        
        # 2: Check for j->i direction, absolute value
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        condition2 = jnp.abs(rev_dot_products) - cos_alpha_max
        
        # 3: Check for maximum communication distance
        condition3 = self.params["comm_radius"] - distances
        
        # Calculate average communication costs across all agent pairs
        # Mask out self-connections
        mask = 1.0 - jnp.eye(self.num_agents)
        
        # Calculate average costs for each condition
        # For each condition, we compute the average negative value (violations)
        # across all agent pairs, excluding self-connections
        
        # Condition 1: viewing angle i->j
        neg_condition1 = jnp.where(condition1 < 0, -condition1, 0.0) * mask
        avg_cost1 = jnp.sum(neg_condition1) / jnp.maximum(jnp.sum(mask), 1.0)
        
        # Condition 2: viewing angle j->i
        neg_condition2 = jnp.where(condition2 < 0, -condition2, 0.0) * mask
        avg_cost2 = jnp.sum(neg_condition2) / jnp.maximum(jnp.sum(mask), 1.0)
        
        # Condition 3: distance constraint
        neg_condition3 = jnp.where(condition3 < 0, -condition3, 0.0) * mask
        avg_cost3 = jnp.sum(neg_condition3) / jnp.maximum(jnp.sum(mask), 1.0)
        
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
        Simplified edge blocks construction with basic state differences.
        """
        # Get communication matrix based on FoV constraints
        communication_matrix = self._get_communication_matrix(state.agent)
        
        # Agent-agent connections
        agent_pos = state.agent[:, :2]         # x, y
        agent_psi = state.agent[:, 2:3]        # psi
        agent_vel = state.agent[:, 3:6]        # vx, vy, omega
        
        # Create state difference
        pos_diff = jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0)  # [i, j]: i -> j
        psi_diff = jnp.expand_dims(agent_psi, 1) - jnp.expand_dims(agent_psi, 0)
        vel_diff = jnp.expand_dims(agent_vel, 1) - jnp.expand_dims(agent_vel, 0)
        
        # Create basic state difference features
        state_diff = jnp.concatenate([pos_diff, psi_diff, vel_diff], axis=-1)
        
        # Use communication matrix as mask for agent-agent edges
        agent_agent_mask = communication_matrix
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)
        
        # Agent-goal connections (only for leader agents)
        leader_mask = jnp.zeros((self.num_agents, self.num_goals))
        leader_mask = leader_mask.at[:self.num_goals].set(jnp.eye(self.num_goals))
        
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.num_goals)
        
        # Create basic agent-goal features
        agent_goal_feats = jnp.zeros((self.num_agents, self.num_goals, 6))  # Only use the basic state diff (6 dims)
        
        for i in range(self.num_goals):
            agent_state = state.agent[i, :6]  # Only use the basic state (x, y, psi, vx, vy, omega)
            goal_state = state.goal[i, :6]    # Only use the basic state
            agent_goal_feats = agent_goal_feats.at[i, i].set(agent_state - goal_state)
        
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, leader_mask, id_agent, id_goal
        )
        
        return [agent_agent_edges, agent_goal_edges]
    
    def get_graph(self, env_state: MPEFoV2State) -> MPEFoV2GraphsTuple:
        """
        Enhanced graph creation with explicit orientation encoding in state features.
        """
        # Create node features
        node_feats = jnp.zeros((self.num_agents + self.num_goals, self.node_dim))
        
        # Get agent states with explicit orientation encoding
        agent_states = env_state.agent
        
        # Set state components with enhanced orientation features
        # First 8 dimensions are state: x, y, psi, vx, vy, omega, cos(psi), sin(psi)
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(agent_states)
        
        # Set goal states
        node_feats = node_feats.at[
                    self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(env_state.goal)

        # Set node type indicators (positions 8-11)
        # Leader: 0001
        node_feats = node_feats.at[:self.num_goals, 11].set(1.0)
        # Follower: 0010
        node_feats = node_feats.at[self.num_goals:self.num_agents, 10].set(1.0)
        # Goal: 0100
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, 9].set(1.0)

        # Node type
        node_type = -jnp.ones((self.num_agents + self.num_goals,), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(self.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(self.GOAL)

        # Edges
        edge_blocks = self.edge_blocks(env_state)

        # Create graph - keep original states for compatibility
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
            
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
        """Render a video with FOV visualization"""
        if viz_opts is None:
            viz_opts = {}
            
        # Add FOV-specific visualization options
        viz_opts.update({
            "show_fov": True,
            "fov_alpha": 0.2,
            "fov_angle": self.params["alpha_max"]
        })
        
        # Call the base render_video method with enhanced options
        render_mpe(
            rollout=rollout, 
            video_path=video_path, 
            side_length=self.area_size, 
            dim=2, 
            n_agent=self.num_agents,
            r=self.params["car_radius"], 
            cost_components=self.cost_components, 
            Ta_is_unsafe=Ta_is_unsafe, 
            viz_opts=viz_opts,
            n_goal=self.num_goals, 
            dpi=dpi, 
            agent_types=self.agent_types,
            include_orientation=True,
            **kwargs
        ) 
    
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """Apply Euler integration with clipping and update orientation features."""
        # Call the parent class method to get the basic state update
        next_states = super().agent_step_euler(agent_states, action)
        
        # Extract the updated orientation
        orientations = next_states[:, 2:3]
        
        # Calculate cos(psi) and sin(psi) for the updated orientation
        cos_psi = jnp.cos(orientations)
        sin_psi = jnp.sin(orientations)
        
        # Append cos(psi) and sin(psi) to the state
        next_states_with_features = jnp.concatenate([
            next_states,  # Original updated state (x, y, psi, vx, vy, omega)
            cos_psi,      # cos(psi)
            sin_psi       # sin(psi)
        ], axis=1)
        
        return next_states_with_features
    
    def _get_communication_matrix(self, agent_states: State) -> jnp.ndarray:
        """
        Calculate communication matrix based on FoV constraints.
        
        Args:
            agent_states: Agent states with shape [n_agents, state_dim]
                state_dim includes: x, y, psi, vx, vy, omega, cos(psi), sin(psi)
                
        Returns:
            communication_matrix: Binary matrix indicating which agents can communicate
        """
        # Extract positions and orientations
        positions = agent_states[:, :2]  # x, y
        
        # We can use either the raw orientation or the precomputed cos/sin values
        # Here we'll use the precomputed values for efficiency
        cos_psi = agent_states[:, 6:7]  # cos(psi)
        sin_psi = agent_states[:, 7:8]  # sin(psi)
        direction_vectors = jnp.concatenate([cos_psi, sin_psi], axis=1)
        
        # Calculate distance matrix
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Calculate unit direction vectors from i to j
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # Compute dot products: v_i · e_ij
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        )
        
        # Convert alpha_max to radians
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # Check if angles are within FoV for i->j
        condition1 = dot_products >= cos_alpha_max
        
        # Check for j->i direction
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        condition2 = rev_dot_products >= cos_alpha_max
        
        # Check for maximum communication distance
        condition3 = distances <= self.params["comm_radius"]
        
        # Combine all conditions
        # Both agents must be within each other's FoV and within communication range
        communication_matrix = jnp.logical_and(
            jnp.logical_and(condition1, condition2),
            condition3
        ).astype(jnp.float32)
        
        # Exclude self-connections
        communication_matrix = communication_matrix * (1.0 - jnp.eye(self.num_agents))
        
        return communication_matrix