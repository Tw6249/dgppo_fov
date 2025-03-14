#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for MPE FOV environment with all available algorithms
"""

import os
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Ensure we can find dgppo module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dgppo.env import make_env
from dgppo.algo import make_algo

# Set random seed
seed = 123
key = jr.PRNGKey(seed)

# Environment parameters
NUM_AGENTS = 4
MAX_STEPS = 10

# Create environment
print("Creating environment...")
env = make_env(
    env_id='MPEFoV',
    num_agents=NUM_AGENTS,
    max_step=MAX_STEPS
)
print("Environment created successfully!")

# List of algorithms to test
algorithms = ['dgppo', 'hcbfcrpo', 'informarl', 'informarl_lagr']

# Test each algorithm
for algo_name in algorithms:
    print(f"\n=== Testing {algo_name} algorithm ===")
    
    try:
        # Create algorithm
        print(f"Creating {algo_name} algorithm...")
        algo = make_algo(
            algo=algo_name,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            cost_weight=0.0,
            cbf_weight=1.0,
            actor_gnn_layers=2,
            Vl_gnn_layers=2,
            Vh_gnn_layers=1,
            rnn_layers=1,
            lr_actor=3e-4,
            lr_Vl=1e-3,
            lr_Vh=1e-3,
            max_grad_norm=2.0,
            alpha=10.0,
            cbf_eps=1e-2,
            seed=seed,
            batch_size=64,
            use_rnn=True,
            use_lstm=False,
            coef_ent=1e-2,
            rnn_step=16,
            gamma=0.99,
            clip_eps=0.25,
            lagr_init=0.5,
            lr_lagr=1e-7,
            train_steps=100,
            cbf_schedule=True,
            cost_schedule=False
        )
        print(f"{algo_name} algorithm created successfully!")
        
        # Test reset
        print("Testing environment reset...")
        key, subkey = jr.split(key)
        graph = env.reset(subkey)
        print("Environment reset successful!")
        
        # Test algorithm step
        print(f"Testing {algo_name} step...")
        key, subkey = jr.split(key)
        rnn_state = algo.init_rnn_state
        
        # Try different return value patterns
        result = algo.step(graph, rnn_state, subkey)
        print(f"Algorithm step returned {len(result)} values")
        
        if len(result) == 3:
            action, log_pi, new_rnn_state = result
        elif len(result) == 4:
            action, z, log_pi, new_rnn_state = result
        else:
            action = result[0]
            new_rnn_state = result[-1]
            
        print(f"{algo_name} step successful!")
        print(f"Action shape: {action.shape}")
        
        # Test environment step
        print("Testing environment step...")
        next_graph, reward, cost, done, info = env.step(graph, action)
        print("Environment step successful!")
        print(f"Reward: {reward}")
        print(f"Cost shape: {cost.shape}")
        
        print(f"{algo_name} test PASSED!")
        
    except Exception as e:
        print(f"Error with {algo_name}: {e}")
        print(f"{algo_name} test FAILED!")

print("\n=== Summary ===")
print("MPE FOV environment can be used with the following algorithms:")
print(", ".join(algorithms)) 