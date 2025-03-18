import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple, List, Optional

from dgppo.env.mpe.mpe_fov2 import MPEFoV2
from dgppo.utils.graph import GraphsTuple

def simple_policy(graph: GraphsTuple, key) -> jnp.ndarray:
    """一个简单的随机策略，产生随机动作"""
    # 获取代理数量（假设第一个节点类型是代理）
    n_agents = graph.env_states.n_agent
    key, subkey = jr.split(key)
    return jr.uniform(subkey, (n_agents, 3), minval=-0.5, maxval=0.5), key

def run_simulation(env: MPEFoV2, n_steps: int = 100) -> Tuple[List, List, List, List, List]:
    """运行模拟并返回收集的数据"""
    # 初始化
    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    graph = env.reset(subkey)
    
    # 收集数据
    graphs = [graph]
    rewards = []
    costs = []
    actions = []
    dones = []
    
    # 运行模拟
    for _ in range(n_steps):
        action, key = simple_policy(graph, key)
        next_graph, reward, cost, done, _ = env.step(graph, action)
        
        # 保存数据
        graphs.append(next_graph)
        rewards.append(reward)
        costs.append(cost)
        actions.append(action)
        dones.append(done)
        
        # 更新当前图
        graph = next_graph
    
    return graphs, rewards, costs, actions, dones

def render_mpe_fov2(
    graphs: List[GraphsTuple],
    costs: np.ndarray,
    rewards: np.ndarray,
    video_path: pathlib.Path,
    env: MPEFoV2,
    is_unsafe: Optional[List[np.ndarray]] = None,
    viz_opts: Optional[Dict] = None,
    dpi: int = 100
):
    """自定义渲染函数，直接使用图形列表"""
    if viz_opts is None:
        viz_opts = {}
    
    # 设置默认可视化选项
    viz_opts.setdefault("show_fov", True)
    viz_opts.setdefault("fov_alpha", 0.2)
    viz_opts.setdefault("fov_angle", env.params["alpha_max"])
    viz_opts.setdefault("comm_radius", env.params["comm_radius"])
    
    # 设置颜色
    leader_color = "#FF6B6B"  # 领导者颜色：红色
    follower_color = "#4ECDC4"  # 跟随者颜色：青色
    goal_color = "#2fdd00"  # 目标颜色：绿色
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    
    # 设置坐标轴范围
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 创建图例
    leader_patch = patches.Circle((0, 0), radius=env.params["car_radius"], color=leader_color, label='Leader')
    follower_patch = patches.Circle((0, 0), radius=env.params["car_radius"], color=follower_color, label='Follower')
    goal_patch = patches.Circle((0, 0), radius=env.params["car_radius"], color=goal_color, label='Goal')
    ax.legend(handles=[leader_patch, follower_patch, goal_patch], loc='upper right')
    
    # 创建文本对象显示奖励和成本
    reward_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment='top')
    cost_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, verticalalignment='top')
    
    # 创建代理和目标的图形对象
    n_agents = env.num_agents
    n_goals = env.num_goals
    n_leaders = env.params["n_leaders"]
    
    # 代理圆圈
    agent_circles = []
    for i in range(n_agents):
        if i < n_leaders:
            color = leader_color
        else:
            color = follower_color
        circle = plt.Circle((0, 0), radius=env.params["car_radius"], color=color, zorder=10)
        agent_circles.append(circle)
        ax.add_patch(circle)
    
    # 目标圆圈
    goal_circles = []
    for i in range(n_goals):
        circle = plt.Circle((0, 0), radius=env.params["car_radius"], color=goal_color, zorder=5)
        goal_circles.append(circle)
        ax.add_patch(circle)
    
    # 视野扇形
    fov_patches = []
    if viz_opts["show_fov"]:
        for i in range(n_agents):
            # 创建视野扇形
            fov = patches.Wedge(
                (0, 0), 
                viz_opts["comm_radius"], 
                0, 0, 
                alpha=viz_opts["fov_alpha"],
                color='gray', 
                zorder=1
            )
            fov_patches.append(fov)
            ax.add_patch(fov)
    
    # 通信线
    comm_lines = []
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=0.5, zorder=2)
                comm_lines.append((i, j, line))
    
    # 更新函数
    def update(frame):
        graph = graphs[frame]
        agent_states = graph.type_states(type_idx=0, n_type=n_agents)
        goal_states = graph.type_states(type_idx=1, n_type=n_goals)
        
        # 更新代理位置和方向
        for i in range(n_agents):
            pos = agent_states[i, :2]
            agent_circles[i].center = (pos[0], pos[1])
            
            # 如果有不安全状态标记，更新颜色
            if is_unsafe is not None and is_unsafe[frame][i]:
                agent_circles[i].set_edgecolor('red')
                agent_circles[i].set_linewidth(2)
            else:
                agent_circles[i].set_edgecolor('none')
                agent_circles[i].set_linewidth(0)
        
        # 更新目标位置
        for i in range(n_goals):
            pos = goal_states[i, :2]
            goal_circles[i].center = (pos[0], pos[1])
        
        # 更新视野扇形
        if viz_opts["show_fov"]:
            for i in range(n_agents):
                pos = agent_states[i, :2]
                psi = agent_states[i, 2]  # 方向角
                
                # 计算视野扇形的起始和结束角度（度）
                start_angle = np.degrees(psi) - viz_opts["fov_angle"]
                end_angle = np.degrees(psi) + viz_opts["fov_angle"]
                
                # 更新视野扇形
                fov_patches[i].set_center(pos)
                fov_patches[i].set_theta1(start_angle)
                fov_patches[i].set_theta2(end_angle)
        
        # 更新通信线
        # 获取通信矩阵
        comm_matrix = env._get_communication_matrix(agent_states)
        
        for i, j, line in comm_lines:
            if comm_matrix[i, j] > 0:
                line.set_data([agent_states[i, 0], agent_states[j, 0]], 
                              [agent_states[i, 1], agent_states[j, 1]])
            else:
                line.set_data([], [])
        
        # 更新奖励和成本文本
        reward_text.set_text(f"Reward: {rewards[frame]:.4f}")
        
        # 计算平均成本
        avg_cost = np.mean(costs[frame])
        cost_text.set_text(f"Avg Cost: {avg_cost:.4f}")
        
        # 返回所有更新的对象
        artists = agent_circles + goal_circles + fov_patches + [reward_text, cost_text]
        artists += [line for _, _, line in comm_lines]
        return artists
    
    # 创建动画
    frames = len(graphs) - 1  # 减去1因为最后一帧是下一个状态
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # 保存视频
    ani.save(video_path, writer='ffmpeg', fps=20)
    plt.close(fig)

def main():
    # 创建输出目录
    output_dir = pathlib.Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化环境
    num_agents = 8
    n_leaders = 2  # 领导者数量
    env_params = {
        "n_leaders": n_leaders,
        "alpha_max": 45,  # 视野角度，度
        "comm_radius": 0.8,  # 通信范围
        "car_radius": 0.05,  # 代理半径
        "dist2goal": 0.1,  # 到达目标的距离阈值
        "energy_penalty_weight": 0.0005  # 能量惩罚权重
    }
    
    env = MPEFoV2(num_agents=num_agents, params=env_params)
    
    # 运行模拟
    print("运行模拟...")
    graphs, rewards, costs, actions, dones = run_simulation(env, n_steps=100)
    
    # 转换为numpy数组以便存储
    np_rewards = jnp.stack(rewards).astype(np.float32)
    np_costs = jnp.stack(costs).astype(np.float32)
    
    # 检测不安全状态（当任何成本为正时）
    is_unsafe = [np.any(cost > 0, axis=1) for cost in np_costs]
    
    # 渲染视频
    print("渲染视频...")
    video_path = output_dir / "mpe_fov2_test.mp4"
    
    # 设置可视化选项
    viz_opts = {
        "show_fov": True,  # 显示视野
        "fov_alpha": 0.2,  # 视野透明度
        "fov_angle": env_params["alpha_max"],  # 视野角度
        "comm_radius": env_params["comm_radius"]  # 通信范围
    }
    
    # 使用自定义渲染函数
    render_mpe_fov2(
        graphs=graphs,
        costs=np_costs,
        rewards=np_rewards,
        video_path=video_path,
        env=env,
        is_unsafe=is_unsafe,
        viz_opts=viz_opts,
        dpi=100
    )
    
    print(f"视频已保存到: {video_path}")

if __name__ == "__main__":
    main() 