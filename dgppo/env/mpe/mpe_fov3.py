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
from dgppo.env.mpe.mpe_fov2 import MPEFoV2, MPEFoV2State, MPEFoV2GraphsTuple
from dgppo.env.utils import get_node_goal_rng
from dgppo.env.plot import render_mpe
from dgppo.trainer.data import Rollout


class MPEFoV3(MPEFoV2):
    """
    Multi-agent Particle Environment with Field of View (FoV) Communication without connectivity constraints.
    
    This environment is based on MPEFoV2 but removes all connectivity constraints (condition1, condition2, condition3).
    Only collision avoidance constraints are maintained.
    """
    
    PARAMS = {
        "car_radius": 0.05,           # Agent radius
        "comm_radius": 0.8,           # Maximum communication distance (not used for constraints)
        "default_area_size": 2.0,     # Default area size
        "dist2goal": 0.1,             # Distance threshold to goal
        "alpha_max": 45,              # Maximum communication angle in degrees (not used for constraints)
        "n_leaders": 2,               # Number of leader agents
        "energy_penalty_weight": 0.0005  # Weight for energy efficiency penalty
    }
    
    def get_cost(self, graph: MPEFoV2GraphsTuple) -> Cost:
        """
        Modified cost function that only considers agent-agent collisions.
        All connectivity constraints (condition1, condition2, condition3) are removed.
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        
        # Agent positions
        agent_pos = agent_states[:, :2]
        
        # Cost for agent-agent collisions
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # Exclude self-distance
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
        # Create a cost matrix with only collision cost
        # We still use a 4-dimensional cost to maintain compatibility with the original implementation
        # but the last 3 dimensions are set to zero (no connectivity constraints)
        cost = jnp.zeros((self.num_agents, 4))
        cost = cost.at[:, 0].set(agent_cost)
        
        # Add margin as required by safe RL
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        
        # Limit minimum cost to -1.0 and maximum to 1.0 (add upper bound)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        
        return cost
    
    def _get_communication_matrix(self, agent_states: State) -> jnp.ndarray:
        """
        Modified communication matrix that allows all agents to communicate with each other.
        This effectively removes the FoV constraints from the communication graph.
        
        Args:
            agent_states: Agent states with shape [n_agents, state_dim]
                
        Returns:
            communication_matrix: Binary matrix indicating which agents can communicate
        """
        # Create a fully connected communication matrix (except self-connections)
        communication_matrix = jnp.ones((self.num_agents, self.num_agents)) - jnp.eye(self.num_agents)
        
        return communication_matrix 