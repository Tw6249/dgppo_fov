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
from dgppo.env.utils import get_node_goal_rng
from dgppo.env.plot import render_mpe
from dgppo.trainer.data import Rollout


class MPEFoV4State(NamedTuple):
    agent: State            # x, y, psi, vx, vy, omega, cos(psi), sin(psi)
    goal: State             # x, y, 0, 0, 0, 0, 0, 0
    initial_connectivity: Array  # 初始连接关系矩阵
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEFoV4GraphsTuple = GraphsTuple[State, MPEFoV4State]


class MPEFoV4(MPE):
    """
    具有视场(FoV)通信约束的多智能体粒子环境
    
    主要特点:
    1. 简化的奖励函数，移除渐进奖励和能量惩罚
    2. 每个智能体的连接成本基于自身连接条件，而非平均成本
    3. 每个智能体只维护与初始连接的邻居的连通性
    4. 增强的节点特征，包含显式方向编码
    """
    
    PARAMS = {
        "car_radius": 0.05,           # 智能体半径
        "comm_radius": 0.8,           # 最大通信距离
        "default_area_size": 2.0,     # 默认区域大小
        "dist2goal": 0.1,             # 到达目标的距离阈值
        "alpha_max": 45,              # 最大通信角度（度）
        "n_leaders": 2,               # 领导者智能体数量
    }
    
    @property
    def node_dim(self) -> int:
        return 11  # 状态维度 (7) + 指示符: 领导者(0001), 跟随者(0010), 目标(0100)
    
    @property
    def edge_dim(self) -> int:
        return 9  # 相对位置(2), 相对速度(3), 通信约束特征(4)
    
    @property
    def state_dim(self) -> int:
        return 7  # x, y, vx, vy, omega, cos(psi), sin(psi)
    
    def reset(self, key: Array) -> GraphsTuple:
        """重置环境，使智能体处于链式拓扑结构"""
        
        # 分割随机数键以用于不同的随机操作
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        
        # 编队的中心位置（区域中心）
        center_pos = jnp.array([self.area_size/2, self.area_size/2])
        
        # 计算智能体之间的适当间距
        # 使用通信半径的一部分确保连接性
        spacing = self.params["comm_radius"] * 0.7
        
        # 计算链的总长度
        chain_length = spacing * (self.num_agents - 1)
        
        # 确定起始位置以使链居中
        start_x = center_pos[0] - chain_length / 2
        start_y = center_pos[1]
        
        # 为智能体排序创建随机排列
        key, perm_key = jr.split(key)
        agent_order = jr.permutation(perm_key, jnp.arange(self.num_agents))
        
        # 初始化智能体位置数组
        agent_positions = jnp.zeros((self.num_agents, 2))
        
        # 将智能体按随机顺序放置在链中
        for i in range(self.num_agents):
            # 沿链的位置
            pos_x = start_x + i * spacing
            pos_y = start_y
            
            # 为置换索引处的智能体设置位置
            agent_idx = agent_order[i]
            agent_positions = agent_positions.at[agent_idx].set(jnp.array([pos_x, pos_y]))
        
        # 向y位置添加小随机噪声以避免完美对齐
        key, noise_key = jr.split(key)
        y_noise = jr.uniform(noise_key, (self.num_agents, 1), minval=-0.05, maxval=0.05)
        agent_positions = agent_positions + jnp.concatenate([jnp.zeros((self.num_agents, 1)), y_noise], axis=1)
        
        # 设置方向朝向链中的下一个智能体
        agent_orientations = jnp.zeros(self.num_agents)
        
        # 对于每个智能体，计算朝向链中下一个智能体的方向
        # 这确保了良好的通信可见性
        for i in range(self.num_agents):
            # 找到链中右侧最近的智能体
            # 计算到所有其他智能体的距离
            dists = jnp.linalg.norm(agent_positions - agent_positions[i], axis=1)
            # 排除自身和左侧（x坐标更小）的智能体
            mask = (agent_positions[:, 0] > agent_positions[i, 0]) & (dists > 0)
            
            # 如果右侧有智能体，则朝向它
            # 使用lax.cond代替if语句以兼容JAX
            has_right_agent = jnp.any(mask).astype(jnp.int32)
            
            def true_branch(_):
                # 找到右侧最近的智能体
                masked_dists = jnp.where(mask, dists, jnp.ones_like(dists) * 1e6)
                nearest_idx = jnp.argmin(masked_dists)
                
                # 计算朝向最近智能体的方向向量
                direction = agent_positions[nearest_idx] - agent_positions[i]
                # 计算角度
                angle = jnp.arctan2(direction[1], direction[0])
                return angle
            
            def false_branch(_):
                # 如果右侧没有智能体，则朝向链中最后一个智能体
                # 找到x坐标最大的智能体
                rightmost_idx = jnp.argmax(agent_positions[:, 0])
                
                # 使用另一个lax.cond处理i是最右侧智能体的情况
                is_rightmost = (i == rightmost_idx).astype(jnp.int32)
                
                def not_rightmost(_):
                    direction = agent_positions[rightmost_idx] - agent_positions[i]
                    return jnp.arctan2(direction[1], direction[0])
                
                def is_rightmost_agent(_):
                    # 如果这是最右侧的智能体，则朝向左侧（朝向链）
                    return jnp.array(jnp.pi)
                
                return lax.cond(is_rightmost, is_rightmost_agent, not_rightmost, operand=None)
            
            new_orientation = lax.cond(has_right_agent, true_branch, false_branch, operand=None)
            agent_orientations = agent_orientations.at[i].set(new_orientation)
        
        # 初始化速度为零
        agent_velocities = jnp.zeros((self.num_agents, 2))
        agent_angular_velocities = jnp.zeros((self.num_agents, 1))
        
        # 调整方向以匹配预期维度
        agent_orientations = agent_orientations.reshape((self.num_agents, 1))
        
        # 计算cos(psi)和sin(psi)
        cos_psi = jnp.cos(agent_orientations)
        sin_psi = jnp.sin(agent_orientations)
        
        # 组合位置、方向、速度和方向特征
        states = jnp.concatenate([
            agent_positions,                # x, y (形状: n_agents x 2)
            agent_velocities,               # vx, vy (形状: n_agents x 2)
            agent_angular_velocities,       # omega (形状: n_agents x 1)
            cos_psi,                        # cos(psi) (形状: n_agents x 1)
            sin_psi                         # sin(psi) (形状: n_agents x 1)
        ], axis=1)
        
        # 以更结构化的方式创建目标位置
        key, subkey = jr.split(key)
        
        # 在区域周围随机放置目标
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
            # 确保目标在边界内
            goal_x = jnp.clip(goal_x, self.params["car_radius"]*3, self.area_size - self.params["car_radius"]*3)
            goal_y = jnp.clip(goal_y, self.params["car_radius"]*3, self.area_size - self.params["car_radius"]*3)
            goal_positions = goal_positions.at[i].set(jnp.array([goal_x, goal_y]))
        
        goals = jnp.concatenate([
            goal_positions,
            jnp.zeros((self.num_goals, 3)),  # vx, vy, omega 对目标全为零
            jnp.ones((self.num_goals, 1)),   # cos(psi) = 1.0
            jnp.zeros((self.num_goals, 1))   # sin(psi) = 0.0
        ], axis=1)
        
        # 计算初始连接关系 - 用于后续成本计算
        initial_connectivity = self._get_communication_matrix(states)
        
        # 创建初始环境状态
        env_state = MPEFoV4State(states, goals, initial_connectivity)
        
        return self.get_graph(env_state)
    
    def step(
            self, graph: MPEFoV4GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEFoV4GraphsTuple, Reward, Cost, Done, Info]:
        """环境的步进函数，使用增强的状态表示"""
        # 从图中获取信息
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        initial_connectivity = graph.state.initial_connectivity
        
        # 计算下一个图
        action = self.clip_action(action)
        
        # 应用欧拉积分并裁剪
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEFoV4State(
            next_agent_states, 
            goals,
            initial_connectivity
        )
        info = {}
        
        # 当达到max_episode_steps时回合结束
        done = jnp.array(False)
        
        # 计算奖励和成本
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        
        return self.get_graph(next_env_state), reward, cost, done, info
    
    def get_reward(self, graph: MPEFoV4GraphsTuple, action: Action) -> Reward:
        """
        简化的奖励函数，没有渐进奖励和能量惩罚
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        
        reward = jnp.zeros(()).astype(jnp.float32)
        
        # 领导者尝试达到目标 - 所有智能体共享
        leader_positions = agent_states[:self.num_goals, :2]
        goal_positions = goals[:, :2]
        
        # 领导者到目标的距离
        leader_goal_dist = jnp.linalg.norm(goal_positions - leader_positions, axis=1)
        
        # 达到目标奖励
        goal_reached_reward = jnp.sum(jnp.where(leader_goal_dist < self.params["dist2goal"], 1.0, 0.0))
        
        # 未达到目标的惩罚
        not_reaching_goal_penalty = -jnp.where(leader_goal_dist > self.params["dist2goal"], 1.0, 0.0).mean() * 0.001
        
        # 合并奖励
        reward = goal_reached_reward + not_reaching_goal_penalty
        
        return reward
    
    def get_cost(self, graph: MPEFoV4GraphsTuple) -> Cost:
        """
        成本函数，每个智能体成本仅基于自身连接条件，并且只考虑初始连接的邻居
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        initial_connectivity = graph.state.initial_connectivity
        
        # 智能体位置
        agent_pos = agent_states[:, :2]
        
        # 1. 智能体-智能体碰撞成本
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6  # 排除自身距离
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self.params["car_radius"] * 2 - min_dist
        
        # 获取当前连接条件
        positions = agent_states[:, :2]  # x, y
        
        # 使用状态中预计算的cos(psi)和sin(psi)
        cos_psi = agent_states[:, 5]  # cos(psi)
        sin_psi = agent_states[:, 6]  # sin(psi)
        direction_vectors = jnp.stack([cos_psi, sin_psi], axis=1)
        
        # 计算距离矩阵
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # 计算从i到j的单位方向向量
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # 计算点积: v_i · e_ij
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        ) 
        
        # 将alpha_max转换为弧度
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # 1: 检查角度是否在i->j的视场内
        angle_cost_i_to_j = cos_alpha_max - jnp.abs(dot_products)
        
        # 2: 检查j->i方向
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        angle_cost_j_to_i = cos_alpha_max - jnp.abs(rev_dot_products)
        
        # 3: 检查最大通信距离
        distance_cost = self.params["comm_radius"] - distances
        
        # 只考虑初始连接的邻居
        # 初始连接矩阵作为掩码
        connectivity_mask = initial_connectivity # 形状：[n_agents, n_agents]
        
        # 计算每个智能体的个体连接成本
        # 对每个智能体，计算其与初始连接邻居的连接成本
        
        # 对每个智能体，所有邻居的角度成本i->j
        angle_cost_i_to_j = angle_cost_i_to_j * connectivity_mask # 形状：[n_agents, n_agents]
        individual_angle_cost_i_to_j = jnp.sum(angle_cost_i_to_j, axis=1) / jnp.maximum(jnp.sum(connectivity_mask, axis=1), 1.0)
        
        # 对每个智能体，所有邻居的角度成本j->i
        angle_cost_j_to_i = angle_cost_j_to_i * connectivity_mask
        individual_angle_cost_j_to_i = jnp.sum(angle_cost_j_to_i, axis=0) / jnp.maximum(jnp.sum(connectivity_mask, axis=0), 1.0)
        
        # 对每个智能体，所有邻居的距离成本
        distance_cost = distance_cost * connectivity_mask
        individual_distance_cost = jnp.sum(distance_cost, axis=1) / jnp.maximum(jnp.sum(connectivity_mask, axis=1), 1.0)
        
        # 组合所有成本到单个矩阵 [n_agents, 4]
        cost = jnp.stack([
            agent_cost,                      # 智能体碰撞成本
            individual_angle_cost_i_to_j,    # 视角条件1成本 (i到j)
            individual_angle_cost_j_to_i,    # 视角条件2成本 (j到i)
            individual_distance_cost         # 通信距离条件成本
        ], axis=1)
        
        # 添加安全RL所需的边界
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        
        # 限制最小成本为-1.0，最大为1.0
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        
        return cost
    
    def edge_blocks(self, state: MPEFoV4State) -> list[EdgeBlock]:
        """
        增强的边块构造，包含通信约束特征
        """
        # 基于视场约束获取通信矩阵
        communication_matrix = self._get_communication_matrix(state.agent)
        
        # 智能体-智能体连接
        agent_pos = state.agent[:, :2]         # x, y
        agent_vel = state.agent[:, 2:5]        # vx, vy, omega
        agent_dir = state.agent[:, 5:7]        # cos(psi), sin(psi)
        
        # 计算距离矩阵
        diff = jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # 计算从i到j的单位方向向量
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # 计算角度裕度所需的点积
        dot_products_i_to_j = jnp.sum(
            jnp.expand_dims(agent_dir, 1) * diff_normalized, 
            axis=2
        )
        
        # 计算反向方向的点积
        reverse_diff_normalized = -diff_normalized  # e_ji
        dot_products_j_to_i = jnp.sum(
            jnp.expand_dims(agent_dir, 0) * reverse_diff_normalized,
            axis=2
        )
        
        # 将alpha_max转换为弧度
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # 计算通信约束特征
        # 1. 距离裕度: 1.0 - dist_ij/comm_radius
        dist_margin = 1.0 - distances / self.params["comm_radius"]
        
        # 2. 角度裕度 i->j: cos(view_angle)/cos(alpha_max) - 1.0
        angle_margin_i_to_j = dot_products_i_to_j / cos_alpha_max - 1.0
        
        # 3. 角度裕度 j->i: cos(view_angle)/cos(alpha_max) - 1.0
        angle_margin_j_to_i = dot_products_j_to_i / cos_alpha_max - 1.0
        
        # 4. 初始邻居指示器 (one-hot)
        is_initial_neighbor = state.initial_connectivity
        
        # 创建相对位置和速度特征
        pos_diff = diff  # 相对位置
        vel_diff = jnp.expand_dims(agent_vel, 1) - jnp.expand_dims(agent_vel, 0)  # 相对速度
        
        # 创建通信约束特征
        comm_features = jnp.stack([
            dist_margin,
            angle_margin_i_to_j,
            angle_margin_j_to_i,
            is_initial_neighbor
        ], axis=-1)
        
        # 组合所有边特征
        edge_features = jnp.concatenate([
            pos_diff,          # 相对位置 (2维)
            vel_diff,          # 相对速度 (3维)
            comm_features      # 通信约束特征 (4维)
        ], axis=-1)
        
        # 使用通信矩阵作为智能体-智能体边的掩码
        agent_agent_mask = communication_matrix
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_features, agent_agent_mask, id_agent, id_agent)
        
        # 智能体-目标连接（仅领导者智能体）
        leader_mask = jnp.zeros((self.num_agents, self.num_goals))
        leader_mask = leader_mask.at[:self.num_goals].set(jnp.eye(self.num_goals))
        
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.num_goals)
        
        # 创建基本智能体-目标特征
        agent_goal_feats = jnp.zeros((self.num_agents, self.num_goals, self.edge_dim))
        
        for i in range(self.num_goals):
            # 仅使用相对位置和速度
            agent_pos_i = state.agent[i, :2]  # x, y
            agent_vel_i = state.agent[i, 2:5]  # vx, vy, omega
            
            goal_pos_i = state.goal[i, :2]  # x, y
            goal_vel_i = state.goal[i, 2:5]  # vx, vy, omega (都是0)
            
            # 计算相对位置和速度
            pos_diff_i = agent_pos_i - goal_pos_i
            vel_diff_i = agent_vel_i - goal_vel_i
            
            # 填充边缘特征的前5个维度（相对位置和速度），其余填0
            feat = jnp.concatenate([
                pos_diff_i,  # 相对位置 (2维)
                vel_diff_i,  # 相对速度 (3维)
                jnp.zeros(4)  # 通信约束特征对智能体-目标边无意义，填0
            ])
            
            agent_goal_feats = agent_goal_feats.at[i, i].set(feat)
        
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, leader_mask, id_agent, id_goal
        )
        
        return [agent_agent_edges, agent_goal_edges]
    
    def get_graph(self, env_state: MPEFoV4State) -> MPEFoV4GraphsTuple:
        """
        增强的图创建，状态特征中具有显式方向编码
        """
        # 创建节点特征
        node_feats = jnp.zeros((self.num_agents + self.num_goals, self.node_dim))
        
        # 获取具有显式方向编码的智能体状态
        agent_states = env_state.agent
        
        # 设置状态组件，带有增强的方向特征
        # 前7个维度是状态: x, y, vx, vy, omega, cos(psi), sin(psi)
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(agent_states)
        
        # 设置目标状态
        node_feats = node_feats.at[
                    self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(env_state.goal)

        # 设置节点类型指示符 (位置7-10)
        # 领导者: 0001
        node_feats = node_feats.at[:self.num_goals, 10].set(1.0)
        # 跟随者: 0010
        node_feats = node_feats.at[self.num_goals:self.num_agents, 9].set(1.0)
        # 目标: 0100
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, 8].set(1.0)

        # 节点类型
        node_type = -jnp.ones((self.num_agents + self.num_goals,), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(self.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(self.GOAL)

        # 边
        edge_blocks = self.edge_blocks(env_state)

        # 创建图 - 保留原始状态以兼容
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
        """渲染带有FOV可视化的视频"""
        if viz_opts is None:
            viz_opts = {}
            
        # 添加FOV特定的可视化选项
        viz_opts.update({
            "show_fov": True,
            "fov_alpha": 0.2,
            "fov_angle": self.params["alpha_max"]
        })
        
        # 使用增强选项调用基本render_video方法
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
        """应用欧拉积分并更新方向特征"""
        # 解压缩状态
        x = agent_states[:, 0:1]  # x position
        y = agent_states[:, 1:2]  # y position
        vx = agent_states[:, 2:3]  # x velocity
        vy = agent_states[:, 3:4]  # y velocity
        omega = agent_states[:, 4:5]  # angular velocity
        cos_psi = agent_states[:, 5:6]  # cos(psi)
        sin_psi = agent_states[:, 6:7]  # sin(psi)
        
        # 从cos(psi)和sin(psi)中恢复psi，用于计算
        psi = jnp.arctan2(sin_psi, cos_psi)
        
        # 解压缩动作
        ax = action[:, 0:1]  # x acceleration
        ay = action[:, 1:2]  # y acceleration
        alpha = action[:, 2:3]  # angular acceleration
        
        # 应用欧拉积分
        dt = 0.1  # 时间步长
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt
        omega_next = omega + alpha * dt
        
        x_next = x + vx * dt
        y_next = y + vy * dt
        psi_next = psi + omega * dt
        
        # 限制位置在环境边界内
        x_next = jnp.clip(x_next, 0.0, self.area_size)
        y_next = jnp.clip(y_next, 0.0, self.area_size)
        
        # 将psi规范化到[-pi, pi]
        psi_next = jnp.mod(psi_next + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # 计算cos(psi)和sin(psi)
        cos_psi_next = jnp.cos(psi_next)
        sin_psi_next = jnp.sin(psi_next)
        
        # 组合更新后的状态
        next_states = jnp.concatenate([
            x_next, y_next, vx_next, vy_next, omega_next, cos_psi_next, sin_psi_next
        ], axis=1)
        
        return next_states
    
    def clip_action(self, action: Action) -> Action:
        """裁剪动作到合理范围"""
        # 最大加速度和角加速度限制
        a_max = 1.0
        alpha_max = 1.5
        
        # 裁剪线性加速度
        a_norm = jnp.linalg.norm(action[:, :2], axis=1, keepdims=True)
        a_norm_clipped = jnp.minimum(a_norm, a_max)
        scale = jnp.where(a_norm > 0, a_norm_clipped / jnp.maximum(a_norm, 1e-6), 1.0)
        a_clipped = action[:, :2] * scale
        
        # 裁剪角加速度
        alpha_clipped = jnp.clip(action[:, 2:3], -alpha_max, alpha_max)
        
        # 组合裁剪后的动作
        action_clipped = jnp.concatenate([a_clipped, alpha_clipped], axis=1)
        
        return action_clipped
    
    def _get_communication_matrix(self, agent_states: State) -> jnp.ndarray:
        """
        基于视场约束计算通信矩阵
        
        参数:
            agent_states: 形状为[n_agents, state_dim]的智能体状态
                state_dim包括: x, y, vx, vy, omega, cos(psi), sin(psi)
                
        返回:
            communication_matrix: 指示哪些智能体可以通信的二元矩阵
        """
        # 提取位置和方向
        positions = agent_states[:, :2]  # x, y
        
        # 使用预计算的cos/sin值以提高效率
        cos_psi = agent_states[:, 5:6]  # cos(psi)
        sin_psi = agent_states[:, 6:7]  # sin(psi)
        direction_vectors = jnp.concatenate([cos_psi, sin_psi], axis=1)
        
        # 计算距离矩阵
        diff = jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0)  # [i, j]: i -> j
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # 计算从i到j的单位方向向量
        norms = jnp.maximum(distances, 1e-10)
        diff_normalized = diff / jnp.expand_dims(norms, -1)  # e_ij
        
        # 计算点积: v_i · e_ij
        dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 1) * diff_normalized, 
            axis=2
        )
        
        # 将alpha_max转换为弧度
        alpha_max_rad = jnp.deg2rad(self.params["alpha_max"])
        cos_alpha_max = jnp.cos(alpha_max_rad)
        
        # 检查角度是否在i->j的视场内
        condition1 = dot_products >= cos_alpha_max
        
        # 检查j->i方向
        reverse_diff_normalized = -diff_normalized  # e_ji
        rev_dot_products = jnp.sum(
            jnp.expand_dims(direction_vectors, 0) * reverse_diff_normalized,
            axis=2
        )
        condition2 = rev_dot_products >= cos_alpha_max
        
        # 检查最大通信距离
        condition3 = distances <= self.params["comm_radius"]
        
        # 组合所有条件
        # 两个智能体必须在彼此的视场内并且在通信范围内
        communication_matrix = jnp.logical_and(
            jnp.logical_and(condition1, condition2),
            condition3
        ).astype(jnp.float32)
        
        # 排除自连接
        communication_matrix = communication_matrix * (1.0 - jnp.eye(self.num_agents))
        
        return communication_matrix
    
    @property
    def cost_components(self) -> List[str]:
        """定义成本组件名称"""
        return [
            "agent_collision",       # 智能体碰撞成本
            "fov_i_to_j",            # 视场条件1 (i到j)
            "fov_j_to_i",            # 视场条件2 (j到i)
            "comm_distance"          # 通信距离条件
        ]
    
    @property
    def agent_types(self) -> List[str]:
        """定义智能体类型"""
        types = []
        for i in range(self.num_goals):
            types.append("leader")
        for i in range(self.num_goals, self.num_agents):
            types.append("follower")
        return types 