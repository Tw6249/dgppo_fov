import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pathlib
from typing import Dict, Tuple, List

from dgppo.env.mpe.mpe_fov import MPEFoV
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple
from simple_render_mpe_fov import render_mpe_fov

def simple_policy(graph: GraphsTuple, key) -> jnp.ndarray:
    """一个简单的随机策略，产生随机动作"""
    # 获取代理数量（假设第一个节点类型是代理）
    # 这里改用env_states属性中的n_agent
    n_agents = graph.env_states.n_agent
    key, subkey = jr.split(key)
    return jr.uniform(subkey, (n_agents, 3), minval=-0.5, maxval=0.5), key

def run_simulation(env: MPEFoV, n_steps: int = 100) -> Tuple[List, List, List, List, List]:
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

def main():
    # 创建输出目录
    output_dir = pathlib.Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化环境
    num_agents = 8
    n_leaders = 4  # 领导者数量
    env_params = {
        "n_leaders": n_leaders,
        "alpha_max": 60,  # 视野角度，度
        "comm_radius": 1.0,  # 通信范围
        "car_radius": 0.05,  # 代理半径
        "n_obs": 0,  # 障碍物数量
        "obs_radius": 0.05,  # 障碍物半径
        "dist2goal": 0.1,  # 到达目标的距离阈值
    }
    
    env = MPEFoV(num_agents=num_agents, params=env_params)
    
    # 运行模拟
    print("运行模拟...")
    graphs, rewards, costs, actions, dones = run_simulation(env, n_steps=100)
    
    # 转换为numpy数组以便存储
    np_rewards = jnp.stack(rewards).astype(np.float32)
    np_costs = jnp.stack(costs).astype(np.float32)
    
    # 设置可视化选项
    viz_opts = {
        "show_fov": True,  # 显示视野
        "fov_alpha": 0.2,  # 视野透明度
        "fov_angle": env_params["alpha_max"],  # 视野角度
        "comm_radius": env_params["comm_radius"]  # 通信范围
    }
    
    # 检测不安全状态（当任何成本为正时）
    is_unsafe = [np.any(cost > 0, axis=1) for cost in np_costs]
    
    # 渲染视频
    print("渲染视频...")
    video_path = output_dir / "mpe_fov_test.mp4"
    
    # 使用简化版渲染函数
    render_mpe_fov(
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