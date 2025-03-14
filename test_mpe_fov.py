#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for MPE FOV environment
Please activate dgppo environment before use
"""

import os
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Arrow
import matplotlib.animation as animation

# Ensure we can find dgppo module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgppo.env import make_env
from dgppo.env.mpe.mpe_fov import MPEFoV

print("Testing MPE FOV Environment...")

# Set random seed
key = jr.PRNGKey(42)

# Environment parameters
NUM_AGENTS = 8
MAX_STEPS = 50
PARAMS = {
    "n_leaders": 2,           # Number of leader agents
    "alpha_max": 60,          # Maximum communication angle (degrees)
    "comm_radius": 0.8,       # Maximum communication distance
    "car_radius": 0.05,       # Agent radius
    "default_area_size": 5.0, # Area size
}

# Create environment
print("Creating environment...")
env = make_env(
    env_id='MPEFoV',
    num_agents=NUM_AGENTS,
    max_step=MAX_STEPS
)

# Override default parameters
for k, v in PARAMS.items():
    env.params[k] = v

# Reset environment
print("Resetting environment...")
key, subkey = jr.split(key)
graph = env.reset(subkey)

# Extract environment state
agent_states = graph.type_states(type_idx=0, n_type=NUM_AGENTS)
goal_states = graph.type_states(type_idx=1, n_type=env.num_goals)

# Get communication matrix
comm_matrix = env._get_communication_matrix(agent_states)

# Print critical neighbor information
print("\n===== Network Information =====")
print("Critical Neighbors:")
for agent_id, neighbors in env.critical_neighbors.items():
    print(f"Agent {agent_id} -> {neighbors}")

# Print communication matrix
print("\nCommunication Matrix (1=can communicate, 0=cannot communicate):")
print(np.array(comm_matrix))

# Check and print graph connectivity status
is_connected = env._is_strongly_connected(comm_matrix)
print(f"\nIs graph strongly connected: {'Yes' if is_connected else 'No'}")

# Calculate in-degree and out-degree for each agent
out_degrees = np.sum(comm_matrix, axis=1)
in_degrees = np.sum(comm_matrix, axis=0)
print("\nAgent Connectivity:")
for i in range(NUM_AGENTS):
    print(f"Agent {i}: in-degree={in_degrees[i]}, out-degree={out_degrees[i]}")

# Visualization function
def visualize_agents(agent_states, goal_states, communication_matrix=None, title="MPE Field of View Environment"):
    """Visualize agents, goals and communication links"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis range
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Extract positions and orientations
    positions = agent_states[:, :2]
    orientations = agent_states[:, 2]
    
    # Draw goals
    for i in range(len(goal_states)):
        goal_pos = goal_states[i, :2]
        ax.add_patch(Circle(goal_pos, radius=0.08, color='green', alpha=0.7))
        ax.text(goal_pos[0], goal_pos[1], f"G{i}", ha='center', va='center')
    
    # Field of view angle (degrees) - full sector angle is twice alpha_max
    fov_angle = env.params["alpha_max"] * 2
    
    # Draw communication links
    if communication_matrix is not None:
        # First draw general communication links (thin yellow lines)
        for i in range(len(positions)):
            for j in range(len(positions)):
                if communication_matrix[i, j]:
                    ax.plot([positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            'y-', alpha=0.4, linewidth=1, zorder=1,
                            label="_nolegend_")
        
        # Then draw critical communication links (thick blue lines)
        for agent_id, neighbors in env.critical_neighbors.items():
            for neighbor_id in neighbors:
                if communication_matrix[agent_id, neighbor_id]:
                    # Critical links with thick lines and different color
                    ax.plot([positions[agent_id, 0], positions[neighbor_id, 0]],
                            [positions[agent_id, 1], positions[neighbor_id, 1]],
                            'c-', alpha=1.0, linewidth=2.5, zorder=2,
                            label="_nolegend_")
                else:
                    # If critical link is broken, mark with red dashed line
                    ax.plot([positions[agent_id, 0], positions[neighbor_id, 0]],
                            [positions[agent_id, 1], positions[neighbor_id, 1]],
                            'r--', alpha=0.7, linewidth=2, zorder=2,
                            label="_nolegend_")
    
    # Draw agents
    leader_count = env.num_goals
    for i, (pos, ori) in enumerate(zip(positions, orientations)):
        # Draw agent circle
        is_leader = i < leader_count
        color = 'red' if is_leader else 'blue'
        ax.add_patch(Circle(pos, radius=env.params["car_radius"], color=color, zorder=3))
        
        # Draw orientation arrow
        dx = 0.1 * np.cos(ori)
        dy = 0.1 * np.sin(ori)
        ax.arrow(pos[0], pos[1], dx, dy, head_width=0.04, head_length=0.05, 
                 fc=color, ec=color, zorder=4)
        
        # Draw front and back field of view
        # Forward field - sector centered on current orientation
        front_wedge = Wedge(pos, env.params["comm_radius"], 
                           ori * 180/np.pi - fov_angle/2, 
                           ori * 180/np.pi + fov_angle/2, 
                           alpha=0.1, color=color, zorder=0)
        ax.add_patch(front_wedge)
        
        # Backward field - sector centered on opposite orientation
        back_ori = (ori + np.pi) % (2 * np.pi)
        back_wedge = Wedge(pos, env.params["comm_radius"], 
                          back_ori * 180/np.pi - fov_angle/2, 
                          back_ori * 180/np.pi + fov_angle/2, 
                          alpha=0.1, color=color, zorder=0)
        ax.add_patch(back_wedge)
        
        # Add agent label
        label = f"L{i}" if is_leader else f"F{i}"
        ax.text(pos[0], pos[1], label, ha='center', va='center', color='white', 
                fontweight='bold', zorder=5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=1, alpha=0.4, label='Regular Communication'),
        Line2D([0], [0], color='cyan', lw=2.5, alpha=1.0, label='Critical Communication'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, alpha=0.7, label='Broken Critical Link'),
        Circle((0, 0), radius=0.05, color='red', label='Leader'),
        Circle((0, 0), radius=0.05, color='blue', label='Follower'),
        Circle((0, 0), radius=0.05, color='green', alpha=0.7, label='Goal')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig, ax

# Display initial state
print("\nCreating initial state visualization...")
fig, ax = visualize_agents(agent_states, goal_states, comm_matrix, "Initial State")
plt.savefig("initial_state.png")
plt.close()

# Execute random action and observe results
print(f"\nSimulating {MAX_STEPS} steps...")
frames = []
states_history = [agent_states]
comm_history = [comm_matrix]
connectivity_history = [is_connected]

# Function to calculate connectivity metrics
def calculate_connectivity_metrics(comm_matrix):
    # Calculate in-degree and out-degree
    out_degrees = np.sum(comm_matrix, axis=1)
    in_degrees = np.sum(comm_matrix, axis=0)
    
    # Calculate average degree
    avg_degree = np.mean(out_degrees)
    
    # Calculate whether strongly connected
    is_strongly_connected = env._is_strongly_connected(comm_matrix)
    
    # Calculate number of connections
    num_connections = np.sum(comm_matrix)
    
    return {
        'avg_degree': avg_degree,
        'is_strongly_connected': is_strongly_connected,
        'num_connections': num_connections,
        'out_degrees': out_degrees,
        'in_degrees': in_degrees
    }

connectivity_metrics_history = [calculate_connectivity_metrics(comm_matrix)]

for step in range(MAX_STEPS):
    # Random action, note linear acceleration and angular acceleration
    key, subkey = jr.split(key)
    random_action = jr.uniform(subkey, (NUM_AGENTS, 2), minval=-0.5, maxval=0.5)
    
    # Step environment
    next_graph, reward, cost, done, info = env.step(graph, random_action)
    graph = next_graph
    
    # Extract new state
    agent_states = graph.type_states(type_idx=0, n_type=NUM_AGENTS)
    comm_matrix = env._get_communication_matrix(agent_states)
    
    # Update critical neighbors - update every 10 steps
    if (step + 1) % 10 == 0:
        env.critical_neighbors = env._identify_critical_neighbors(agent_states)
        print(f"\nStep {step+1} - Update critical neighbors:")
        for agent_id, neighbors in env.critical_neighbors.items():
            print(f"Agent {agent_id} -> {neighbors}")
    
    # Calculate connectivity and save
    is_connected = env._is_strongly_connected(comm_matrix)
    connectivity_metrics = calculate_connectivity_metrics(comm_matrix)
    
    # Save history
    states_history.append(agent_states)
    comm_history.append(comm_matrix)
    connectivity_history.append(is_connected)
    connectivity_metrics_history.append(connectivity_metrics)
    
    # Save every 10 steps visualization snapshot
    if (step + 1) % 10 == 0:
        fig, ax = visualize_agents(agent_states, goal_states, comm_matrix, 
                                  f"Step {step+1}")
        plt.savefig(f"step_{step+1}.png")
        plt.close()
    
    # Save frame for animation
    frames.append(agent_states)

# Create connectivity metrics plot over time
print("\nCreating connectivity metrics plot...")
steps = list(range(len(connectivity_metrics_history)))

# Plot connectivity metrics over time
plt.figure(figsize=(12, 8))

# Plot average degree
plt.subplot(2, 2, 1)
avg_degrees = [m['avg_degree'] for m in connectivity_metrics_history]
plt.plot(steps, avg_degrees, 'b-', linewidth=2)
plt.title('Average Degree')
plt.xlabel('Step')
plt.ylabel('Average Degree')
plt.grid(True)

# Plot number of connections
plt.subplot(2, 2, 2)
num_connections = [m['num_connections'] for m in connectivity_metrics_history]
plt.plot(steps, num_connections, 'g-', linewidth=2)
plt.title('Number of Connections')
plt.xlabel('Step')
plt.ylabel('Connections')
plt.grid(True)

# Plot strongly connected status
plt.subplot(2, 2, 3)
is_connected_values = [1 if c else 0 for c in connectivity_history]
plt.plot(steps, is_connected_values, 'r-', linewidth=2)
plt.title('Strong Connectivity')
plt.xlabel('Step')
plt.ylabel('Connected (1=Yes, 0=No)')
plt.yticks([0, 1], ['No', 'Yes'])
plt.grid(True)

# Plot out-degree for each agent
plt.subplot(2, 2, 4)
for i in range(NUM_AGENTS):
    plt.plot(steps, [m['out_degrees'][i] for m in connectivity_metrics_history], 
             label=f"Agent {i}")
plt.title('Agent Out-Degree Over Time')
plt.xlabel('Step')
plt.ylabel('Out Degree')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('connectivity_metrics.png')
plt.close()

# Create animation
print("\n=== Creating Animation ===")
fig = plt.figure(figsize=(10, 10))

# Animation function
def animate(frame_idx):
    """Create animation frame"""
    frame = frames[frame_idx]
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_aspect('equal')
    
    # Add connectivity information to title
    is_connected = connectivity_history[frame_idx]
    connected_status = "Connected" if is_connected else "Not Connected"
    ax.set_title(f"Step {frame_idx} - {connected_status}")
    
    # Get communication matrix for this frame
    comm_matrix = comm_history[frame_idx]
    
    # Extract positions and orientations
    positions = frame[:, :2]
    orientations = frame[:, 2]
    
    # Draw goals
    for i in range(len(goal_states)):
        goal_pos = goal_states[i, :2]
        ax.add_patch(Circle(goal_pos, radius=0.08, color='green', alpha=0.7))
        ax.text(goal_pos[0], goal_pos[1], f"G{i}", ha='center', va='center')
    
    # Field of view angle (degrees)
    fov_angle = env.params["alpha_max"] * 2
    
    # Draw communication links
    if comm_matrix is not None:
        # First draw general communication links (thin yellow lines)
        for i in range(len(positions)):
            for j in range(len(positions)):
                if comm_matrix[i, j]:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           'y-', alpha=0.4, linewidth=1, zorder=1)
        
        # Then draw critical links if available for this frame
        if frame_idx < len(env.critical_neighbors):
            for agent_id, neighbors in env.critical_neighbors.items():
                for neighbor_id in neighbors:
                    if comm_matrix[agent_id, neighbor_id]:
                        # Critical links with thick cyan lines
                        ax.plot([positions[agent_id, 0], positions[neighbor_id, 0]],
                               [positions[agent_id, 1], positions[neighbor_id, 1]],
                               'c-', alpha=1.0, linewidth=2.5, zorder=2)
                    else:
                        # Broken critical links with red dashed lines
                        ax.plot([positions[agent_id, 0], positions[neighbor_id, 0]],
                               [positions[agent_id, 1], positions[neighbor_id, 1]],
                               'r--', alpha=0.7, linewidth=2, zorder=2)
    
    # Draw agents
    leader_count = env.num_goals
    for i, (pos, ori) in enumerate(zip(positions, orientations)):
        # Draw agent circle
        is_leader = i < leader_count
        color = 'red' if is_leader else 'blue'
        ax.add_patch(Circle(pos, radius=env.params["car_radius"], color=color, zorder=3))
        
        # Draw orientation arrow
        dx = 0.1 * np.cos(ori)
        dy = 0.1 * np.sin(ori)
        ax.arrow(pos[0], pos[1], dx, dy, head_width=0.04, head_length=0.05, 
                fc=color, ec=color, zorder=4)
        
        # Draw front and back field of view
        front_wedge = Wedge(pos, env.params["comm_radius"], 
                           ori * 180/np.pi - fov_angle/2, 
                           ori * 180/np.pi + fov_angle/2, 
                           alpha=0.1, color=color, zorder=0)
        ax.add_patch(front_wedge)
        
        back_ori = (ori + np.pi) % (2 * np.pi)
        back_wedge = Wedge(pos, env.params["comm_radius"], 
                          back_ori * 180/np.pi - fov_angle/2, 
                          back_ori * 180/np.pi + fov_angle/2, 
                          alpha=0.1, color=color, zorder=0)
        ax.add_patch(back_wedge)
        
        # Add agent label
        label = f"L{i}" if is_leader else f"F{i}"
        ax.text(pos[0], pos[1], label, ha='center', va='center', color='white', 
                fontweight='bold', zorder=5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=1, alpha=0.4, label='Regular Communication'),
        Line2D([0], [0], color='cyan', lw=2.5, alpha=1.0, label='Critical Communication'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, alpha=0.7, label='Broken Critical Link'),
        Circle((0, 0), radius=0.05, color='red', label='Leader'),
        Circle((0, 0), radius=0.05, color='blue', label='Follower'),
        Circle((0, 0), radius=0.05, color='green', alpha=0.7, label='Goal')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    return ax

# Create animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), interval=200, blit=False
)

# Save animation as MP4
print("Saving animation as MP4...")
ani.save('mpe_fov_simulation.mp4', writer='ffmpeg', fps=5)

# Save as GIF (more compatible but larger file)
print("Saving animation as GIF...")
ani.save('mpe_fov_simulation.gif', writer='pillow', fps=5)

# Final results
print("\n=== Simulation Complete ===")
print("Visualization files:")
print(f"- Initial state: initial_state.png")
print(f"- Snapshot every 10 steps: step_10.png, step_20.png, ...")
print(f"- Connectivity metrics: connectivity_metrics.png")
print(f"- Complete animation: mpe_fov_simulation.mp4 or mpe_fov_simulation.gif") 