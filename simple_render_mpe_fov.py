import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import pathlib
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict

from dgppo.env.mpe.mpe_fov import MPEFoV
from dgppo.utils.graph import GraphsTuple


def save_animation(anim, filename):
    """Save animation to file"""
    print(f"Saving animation to {filename}...")
    anim.save(filename, dpi=100, writer='ffmpeg')
    print(f"Animation saved to {filename}")


def render_mpe_fov(
        graphs: List[GraphsTuple],
        costs: List[jnp.ndarray],
        rewards: List[float],
        video_path: pathlib.Path,
        env: MPEFoV,
        is_unsafe: Optional[List[np.ndarray]] = None,
        viz_opts: Optional[Dict] = None,
        dpi: int = 100
):
    """
    简化版的MPEFoV环境渲染函数，专门用于可视化代理的视野和方向
    
    参数:
        graphs: 图形列表，每个时间步的环境状态
        costs: 成本列表，每个时间步的成本
        rewards: 奖励列表，每个时间步的奖励
        video_path: 视频保存路径
        env: MPEFoV环境实例
        is_unsafe: 不安全状态标记列表，可选
        viz_opts: 可视化选项，可选
        dpi: 图像分辨率，默认100
    """
    if viz_opts is None:
        viz_opts = {}
    
    # 设置图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    ax.set_xlim(0., env.area_size)
    ax.set_ylim(0., env.area_size)
    ax.set(aspect="equal")
    plt.axis("off")
    
    # 获取初始图
    graph0 = graphs[0]
    
    # 定义颜色
    leader_color = "#FF6B6B"  # 领导者颜色：红色
    follower_color = "#4ECDC4"  # 跟随者颜色：青色
    goal_color = "#2fdd00"  # 目标颜色：绿色
    edge_color = "0.2"  # 边颜色：灰色
    
    # 从环境中获取代理类型
    agent_types = env.agent_types
    n_agents = env.num_agents
    n_leaders = env.num_goals
    r = env.params["car_radius"]
    
    # 获取代理和目标的位置
    agent_states = graph0.env_states.agent
    goal_states = graph0.env_states.goal
    
    agent_pos = np.array(agent_states[:, :2])
    goal_pos = np.array(goal_states[:, :2])
    
    # 设置代理颜色
    agent_colors = []
    for i in range(n_agents):
        if agent_types[i] == 0:  # LEADER
            agent_colors.append(leader_color)
        else:  # FOLLOWER
            agent_colors.append(follower_color)
    
    # 绘制代理
    agent_circs = []
    for i in range(n_agents):
        circ = plt.Circle(agent_pos[i], r, color=agent_colors[i], linewidth=0)
        agent_circs.append(circ)
        ax.add_patch(circ)
    
    # 绘制目标
    goal_circs = []
    for i in range(n_leaders):
        circ = plt.Circle(goal_pos[i], r, color=goal_color, linewidth=0)
        goal_circs.append(circ)
        ax.add_patch(circ)
    
    # 绘制方向箭头
    direction_arrows = []
    orientations = np.array(agent_states[:, 2])
    arrow_length = r * 2.0
    
    for i in range(n_agents):
        dx = arrow_length * np.cos(orientations[i])
        dy = arrow_length * np.sin(orientations[i])
        arrow = ax.arrow(agent_pos[i, 0], agent_pos[i, 1], dx, dy,
                         head_width=r*0.7, head_length=r*0.7,
                         fc='k', ec='k', zorder=7, alpha=0.7)
        direction_arrows.append(arrow)
    
    # 绘制视野 - 前后两个扇形
    forward_fov_wedges = []
    backward_fov_wedges = []
    if viz_opts.get("show_fov", False):
        fov_angle = viz_opts.get("fov_angle", 60)  # 从环境参数获取的视野角度
        # 注意：环境中的alpha_max定义的是单侧视野角度，两侧共2*alpha_max
        # 这里的fov_angle代表单侧角度，总视野角度为2*fov_angle
        fov_alpha = viz_opts.get("fov_alpha", 0.2)  # 透明度
        comm_radius = env.params["comm_radius"]  # 通信范围
        
        fov_angle_rad = np.deg2rad(fov_angle)
        
        for i in range(n_agents):
            # 前方视野 - 角度范围为 [-fov_angle_rad, +fov_angle_rad]，总宽度为2*fov_angle_rad
            theta1_forward = orientations[i] - fov_angle_rad
            theta2_forward = orientations[i] + fov_angle_rad
            wedge_forward = mpatches.Wedge(agent_pos[i], comm_radius,
                                         np.rad2deg(theta1_forward), np.rad2deg(theta2_forward),
                                         alpha=fov_alpha, color=agent_colors[i], zorder=2)
            forward_fov_wedges.append(wedge_forward)
            ax.add_patch(wedge_forward)
            
            # 后方视野 (方向相反) - 角度范围为 [π-fov_angle_rad, π+fov_angle_rad]，总宽度为2*fov_angle_rad
            theta1_backward = orientations[i] + np.pi - fov_angle_rad
            theta2_backward = orientations[i] + np.pi + fov_angle_rad
            wedge_backward = mpatches.Wedge(agent_pos[i], comm_radius,
                                          np.rad2deg(theta1_backward), np.rad2deg(theta2_backward),
                                          alpha=fov_alpha, color=agent_colors[i], zorder=2)
            backward_fov_wedges.append(wedge_backward)
            ax.add_patch(wedge_backward)
    
    # 绘制通信连接（通信矩阵）
    comm_matrix = env._get_communication_matrix(agent_states)
    comm_matrix_np = np.array(comm_matrix)
    
    # 添加注释说明通信条件
    """
    通信矩阵表示哪些智能体之间可以通信，它基于以下三个条件：
    1. 视野角度条件1: (v_i · e_ij)^2 - cos^2(alpha_max) >= 0
       - 智能体i的朝向与i到j的方向之间的夹角在视野范围内
    2. 视野角度条件2: (v_j · e_ji)^2 - cos^2(alpha_max) >= 0
       - 智能体j的朝向与j到i的方向之间的夹角在视野范围内
    3. 距离条件: d_max^2 - d_ij^2 >= 0
       - 智能体i和j之间的距离在通信范围内
    
    只有同时满足所有三个条件，两个智能体之间才能建立通信连接。
    comm_matrix[i,j] = True 表示智能体i可以与智能体j通信。
    """
    
    # 通信连线容器
    edge_collection = None
    
    # 初始连线 - 只有当两个智能体满足所有通信条件时才绘制连线
    edge_lines = []
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and comm_matrix_np[i, j]:
                edge_lines.append(np.array([agent_pos[i], agent_pos[j]]))
    
    if edge_lines:
        edge_collection = LineCollection(edge_lines, colors=edge_color, 
                                       linewidths=1, alpha=0.5, zorder=3)
        ax.add_collection(edge_collection)
    
    # 添加文本信息
    text_font_opts = dict(
        size=16,
        color="k",
        family="sans-serif",
        weight="normal",
        transform=ax.transAxes,
    )
    
    cost_text = ax.text(0.02, 1.00, "Cost: 0.0\nReward: 0.0", va="bottom", **text_font_opts)
    
    # 安全状态文本
    safe_text = []
    if is_unsafe is not None:
        safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
    
    # 时间步文本
    time_text = ax.text(0.99, 1.04, "Step=0", va="bottom", ha="right", **text_font_opts)
    
    # 代理标签
    label_font_opts = dict(
        size=20,
        color="k",
        family="sans-serif",
        weight="normal",
        ha="center",
        va="center",
        zorder=7,
    )
    
    agent_labels = [ax.text(agent_pos[i, 0], agent_pos[i, 1], f"{i}", **label_font_opts) 
                   for i in range(n_agents)]
    
    def update(frame):
        # 获取当前帧的图和代理状态
        graph = graphs[frame]
        agent_states = graph.env_states.agent
        
        # 更新代理位置和方向
        agent_pos = np.array(agent_states[:, :2])
        orientations = np.array(agent_states[:, 2])
        
        # 更新代理圆形
        for i in range(n_agents):
            agent_circs[i].set_center(agent_pos[i])
            agent_labels[i].set_position(agent_pos[i])
        
        # 更新方向箭头
        for i in range(n_agents):
            # 移除旧箭头
            if direction_arrows[i] in ax.get_children():
                direction_arrows[i].remove()
            
            # 创建新箭头
            dx = arrow_length * np.cos(orientations[i])
            dy = arrow_length * np.sin(orientations[i])
            direction_arrows[i] = ax.arrow(agent_pos[i, 0], agent_pos[i, 1], dx, dy,
                                         head_width=r*0.7, head_length=r*0.7,
                                         fc='k', ec='k', zorder=7, alpha=0.7)
        
        # 更新视野楔形
        if viz_opts.get("show_fov", False):
            fov_angle = viz_opts.get("fov_angle", 60)
            # 注意：环境中的alpha_max定义的是单侧视野角度，两侧共2*alpha_max
            fov_alpha = viz_opts.get("fov_alpha", 0.2)
            comm_radius = env.params["comm_radius"]
            fov_angle_rad = np.deg2rad(fov_angle)
            
            for i in range(n_agents):
                # 移除前方旧楔形
                if forward_fov_wedges[i] in ax.get_children():
                    forward_fov_wedges[i].remove()
                
                # 移除后方旧楔形
                if backward_fov_wedges[i] in ax.get_children():
                    backward_fov_wedges[i].remove()
                
                # 创建前方新楔形 - 总视野为2*fov_angle_rad
                theta1_forward = orientations[i] - fov_angle_rad
                theta2_forward = orientations[i] + fov_angle_rad
                forward_fov_wedges[i] = mpatches.Wedge(agent_pos[i], comm_radius,
                                                     np.rad2deg(theta1_forward), np.rad2deg(theta2_forward),
                                                     alpha=fov_alpha, color=agent_colors[i], zorder=2)
                ax.add_patch(forward_fov_wedges[i])
                
                # 创建后方新楔形 - 总视野为2*fov_angle_rad
                theta1_backward = orientations[i] + np.pi - fov_angle_rad
                theta2_backward = orientations[i] + np.pi + fov_angle_rad
                backward_fov_wedges[i] = mpatches.Wedge(agent_pos[i], comm_radius,
                                                      np.rad2deg(theta1_backward), np.rad2deg(theta2_backward),
                                                      alpha=fov_alpha, color=agent_colors[i], zorder=2)
                ax.add_patch(backward_fov_wedges[i])
        
        # 更新通信连接
        comm_matrix = env._get_communication_matrix(agent_states)
        comm_matrix_np = np.array(comm_matrix)
        
        # 清除所有旧连线 - 确保通过遍历所有collections彻底清除
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.remove()
        
        # 创建新连线 - 只画满足所有条件的连接
        edge_lines = []
        for i in range(n_agents):
            for j in range(n_agents):
                # 只考虑不同智能体之间且满足通信条件的连接
                if i != j and comm_matrix_np[i, j]:
                    edge_lines.append(np.array([agent_pos[i], agent_pos[j]]))
        
        # 添加新连线
        new_edge_collection = None
        if edge_lines:
            new_edge_collection = LineCollection(edge_lines, colors=edge_color, 
                                              linewidths=1, alpha=0.5, zorder=3)
            ax.add_collection(new_edge_collection)
        
        # 更新文本信息
        if frame < len(costs):
            all_costs = f"Cost:\n"
            all_costs += f"    agent collisions: {costs[frame][:, 0].max():5.4f}\n"
            all_costs += f"    viewing angle (i→j): {costs[frame][:, 1].max():5.4f}\n"
            all_costs += f"    viewing angle (j→i): {costs[frame][:, 2].max():5.4f}\n"
            all_costs += f"    distance: {costs[frame][:, 3].max():5.4f}\n"
            all_costs += f"Reward: {rewards[frame]:5.4f}"
            cost_text.set_text(all_costs)
        
        if is_unsafe is not None and frame < len(is_unsafe):
            unsafe_idx = np.where(is_unsafe[frame])[0]
            safe_text[0].set_text(f"Unsafe: {unsafe_idx}")
        
        time_text.set_text(f"Step={frame:04d}")
        
        # 返回所有更新的元素
        artists = [*agent_circs, *goal_circs, *direction_arrows, *agent_labels, 
                  cost_text, time_text]
        if safe_text:
            artists.extend(safe_text)
            
        # 添加所有视野楔形
        if viz_opts.get("show_fov", False):
            artists.extend(forward_fov_wedges)
            artists.extend(backward_fov_wedges)
            
        # 添加边缘集合（如果存在）
        if new_edge_collection:
            artists.append(new_edge_collection)
        
        return artists
    
    # 创建动画
    print("创建动画...")
    anim = FuncAnimation(fig, update, frames=len(graphs), interval=33, blit=True)
    
    # 保存动画
    save_animation(anim, video_path)
    plt.close() 