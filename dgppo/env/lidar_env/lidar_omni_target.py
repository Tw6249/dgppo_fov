import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax

from typing import Tuple, Optional

from dgppo.utils.graph import EdgeBlock, GraphsTuple
from dgppo.utils.typing import Action, Array, State, AgentState, Cost
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.env.utils import get_node_goal_rng, inside_obstacles
from dgppo.utils.utils import jax_vmap

# --- 辅助函数 ---
@jax.jit
def rotation_matrix_transpose(cos_psi: float, sin_psi: float) -> Array:
    """计算旋转矩阵的转置 R_i^T"""
    # R_i = [[cos, -sin], [sin, cos]]
    # R_i^T = [[cos, sin], [-sin, cos]]
    # 输入: cos_psi (标量), sin_psi (标量)
    # 输出: Array, 形状 (2, 2)
    return jnp.array([[cos_psi, sin_psi], [-sin_psi, cos_psi]])

# Vmap 版本的批量计算
# 输入: cos_psi (向量, N), sin_psi (向量, N)
# 输出: Array, 形状 (N, 2, 2)
batch_rotation_matrix_transpose = jax.vmap(rotation_matrix_transpose)
# ---------------

class LidarOmniTarget(LidarEnv):
    """
    麦轮车（全向轮）智能体激光雷达环境 - 加入了分离的 FoV 约束

    该环境模拟具有全向移动能力的机器人，类似于装有麦克纳姆轮或麦轮的移动平台。
    这种机器人可以在不改变朝向的情况下向任意方向移动，同时也可以原地旋转。
    与典型的差速机器人或自行车模型不同，全向轮机器人没有非完整约束。

    新增特性：
    - 每个机器人 i 需要将机器人 i+1 保持在其 FoV 内。
    - FoV 由角度、最大距离和最小距离（碰撞避免）定义。
    - 引入了新的、分离的代价值来惩罚 FoV 违规 (角度、最大距离、最小距离)。
    - reset 方法尝试生成满足 FoV 约束的初始链状配置。
    - 边特征增加了"关键边"维度。
    """

    # --- 更新后的参数 ---
    PARAMS = {
        "car_radius": 0.05,         # 车辆半径 (r)
        "comm_radius": 0.5,         # 通信半径
        "n_rays": 32,               # 激光雷达射线数量
        "obs_len_range": [0.1, 0.3],# 障碍物尺寸范围
        "n_obs": 3,                 # 障碍物数量
        "default_area_size": 1.5,   # 默认环境大小
        "dist2goal": 0.01,          # 到目标的距离阈值
        "top_k_rays": 8,            # 激光雷达选择的重要射线数量
        "max_angular_vel": 100.0,     # 最大角速度
        "rotation_penalty": 0.001,  # 旋转惩罚因子

        # --- FoV 参数 ---
        "fov_angle_deg": 60.0,      # FoV 半顶角 (beta)，单位度
        "max_sensor_range": 0.5,    # FoV 最大感知距离 (r_max)
        "min_safe_distance": 0.2,  # FoV 最小安全距离 (D), 要求 D > 2*car_radius
        # -----------------
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        """
        初始化麦轮车环境

        参数:
            num_agents: 智能体数量
            area_size: 环境空间大小
            max_step: 最大步数
            dt: 时间步长
            params: 环境参数，可覆盖默认参数
        """
        area_size = LidarOmniTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarOmniTarget, self).__init__(num_agents, area_size, max_step, dt, params)

        # --- 计算并存储 FoV 相关常量 ---
        self._fov_beta_rad = jnp.deg2rad(self.params["fov_angle_deg"]) # FoV 半角 (弧度)
        self._cos_fov_beta = jnp.cos(self._fov_beta_rad)             # FoV 半角的余弦
        self._r_max = self.params["max_sensor_range"]               # 最大感知距离
        self._min_safe_dist_D = self.params["min_safe_distance"]    # FoV 最小安全距离
        # 确保参数有效性
        assert self._min_safe_dist_D > 2 * self.params["car_radius"], "最小安全距离 D 必须大于 2 倍车辆半径 (2*r)"
        assert self._min_safe_dist_D < self._r_max, "最小安全距离 D 必须小于最大感知距离 r_max"
        # --------------------------------

        # --- 更新成本维度 ---
        # Dim 0: 智能体间碰撞 (所有对, 2*r - min_dist)
        # Dim 1: 智能体-障碍物碰撞 (r - min_dist_obs)
        # Dim 2: FoV 角度约束 (关键对 i -> i+1, h_angle)
        # Dim 3: FoV 最大距离约束 (关键对 i -> i+1, h_range)
        # Dim 4: FoV 最小距离约束 (关键对 i -> i+1, h_coll_fov = D - dist)
        self._n_cost = 5
        # -------------------

    @property
    def n_cost(self) -> int:
        """定义成本维度数量"""
        return self._n_cost

    @property
    def cost_components(self) -> Tuple[str, ...]:
        """定义各成本分量的名称"""
        return "agent collisions", "obs collisions", "fov angle", "fov max range", "fov min distance"

    @property
    def state_dim(self) -> int:
        """定义智能体状态维度"""
        # x, y, cos(theta), sin(theta), vx, vy, omega
        # 输出: 整数 7
        return 7

    @property
    def node_dim(self) -> int:
        """定义节点特征维度"""
        # 状态维度(7) + 节点类型指示器(3): agent: 001, goal: 010, obstacle: 100
        # 输出: 整数 10
        return 10

    @property
    def edge_dim(self) -> int:
        """边特征维度：相对位置、速度、朝向差异 + 是否关键边 + 局部坐标系特征"""
        # 原始 7: x_rel, y_rel, cos_rel, sin_rel, vx_rel, vy_rel, omega_rel
        # +1: is_critical_edge (i -> i+1)
        # +2: norm_p_j_i (局部相对距离), i_x_j (局部前向分量)
        # 输出: 整数 10
        return 10

    @property
    def action_dim(self) -> int:
        """定义动作空间维度"""
        # ax, ay, alpha(角加速度)
        # 输出: 整数 3
        return 3

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """
        使用欧拉积分法更新智能体状态

        参数:
            agent_states: 当前智能体状态, 形状 (N, state_dim)
            action: 智能体动作, 形状 (N, action_dim)

        返回:
            更新后的智能体状态, 形状 (N, state_dim)
        """
        # ... (内部逻辑保持不变, 注释已更新) ...
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # 提取当前状态
        pos = agent_states[:, :2]          # 位置 (x,y), 形状 (N, 2)
        orientation = agent_states[:, 2:4] # 朝向 [cos(θ), sin(θ)], 形状 (N, 2)
        vel = agent_states[:, 4:6]         # 线速度 (vx, vy), 形状 (N, 2)
        omega = agent_states[:, 6:7]       # 角速度 (ω), 形状 (N, 1)

        # 分解动作
        acc = action[:, :2] * 10.0         # 线加速度, 形状 (N, 2)
        alpha = action[:, 2:3] * 5.0       # 角加速度, 形状 (N, 1)

        # 计算状态变化率
        pos_dot = vel                      # 位置导数 = 速度, 形状 (N, 2)

        # 角度更新
        delta_theta = omega * self.dt      # 角度变化量, 形状 (N, 1)
        theta = jnp.arctan2(orientation[:, 1], orientation[:, 0]) # 当前角度, 形状 (N,)
        new_theta = theta + delta_theta[:, 0] # 新角度, 形状 (N,)
        # 计算新朝向的 cos 和 sin
        new_cos = jnp.cos(new_theta)       # 形状 (N,)
        new_sin = jnp.sin(new_theta)       # 形状 (N,)
        new_orientation = jnp.stack([new_cos, new_sin], axis=1) # 新朝向, 形状 (N, 2)

        # 速度更新 (全向轮可以在任意方向加速)
        vel_dot = acc                      # 速度导数 = 加速度, 形状 (N, 2)
        omega_dot = alpha                  # 角速度导数 = 角加速度, 形状 (N, 1)

        # 使用欧拉积分法更新状态
        new_pos = pos + pos_dot * self.dt         # 新位置, 形状 (N, 2)
        new_vel = vel + vel_dot * self.dt         # 新速度, 形状 (N, 2)
        new_omega = omega + omega_dot * self.dt   # 新角速度, 形状 (N, 1)

        # 组合为新状态
        # new_state 形状: (N, 2+2+2+1) = (N, 7)
        new_state = jnp.concatenate([new_pos, new_orientation, new_vel, new_omega], axis=1)

        # 确保不超出状态限制
        return self.clip_state(new_state)

    def reset(self, key: Array) -> GraphsTuple:
        """
        重置环境，生成新的初始状态，确保满足 FoV 约束的链状拓扑。

        参数:
            key: JAX 随机密钥

        返回:
            GraphsTuple: 表示环境初始状态的图结构
        """
        # --- 创建障碍物 ---
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        
        if n_rng_obs == 0:
            obstacles = None
        else:
            obstacle_key, key = jr.split(key, 2)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (n_rng_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1]
            )
            
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # --- 生成智能体和目标的位置 ---
        node_goal_key, key = jr.split(key, 2)
        
        # 使用更大的最小距离来增加满足FoV约束的可能性
        min_distance = jnp.maximum(
            2.2 * self.params["car_radius"],
            self._min_safe_dist_D
        )
        
        # 使用JAX的get_node_goal_rng生成位置
        agent_pos, goal_pos = get_node_goal_rng(
            node_goal_key, self.area_size, 2, self.num_agents,
            min_distance, obstacles
        )
        
        # --- 设置智能体朝向 ---
        # 初始化朝向数组
        theta_states = jnp.zeros((self.num_agents, 2))
        
        # 仅在多个智能体时需要链状拓扑
        if self.num_agents > 1:
            # 为智能体0到N-2计算朝向，指向下一个智能体
            for i in range(self.num_agents - 1):
                # 计算从智能体i到i+1的单位向量
                delta_p = agent_pos[i+1] - agent_pos[i]
                norm = jnp.linalg.norm(delta_p) + 1e-8
                orientation_i = delta_p / norm
                # 更新朝向
                theta_states = theta_states.at[i].set(orientation_i)
            
            # 最后一个智能体随机朝向
            theta_key, key = jr.split(key, 2)
            last_theta = jr.uniform(theta_key, (1,), minval=0, maxval=2 * np.pi)[0]
            last_orientation = jnp.array([jnp.cos(last_theta), jnp.sin(last_theta)])
            theta_states = theta_states.at[self.num_agents - 1].set(last_orientation)
        elif self.num_agents == 1:
            # 单智能体随机朝向
            theta_key, key = jr.split(key, 2)
            theta = jr.uniform(theta_key, (1,), minval=0, maxval=2 * np.pi)[0]
            orientation = jnp.array([jnp.cos(theta), jnp.sin(theta)])
            theta_states = theta_states.at[0].set(orientation)
        
        # --- 构建最终状态 ---
        states = jnp.concatenate([
            agent_pos,
            theta_states,
            jnp.zeros((self.num_agents, 3), dtype=agent_pos.dtype)
        ], axis=1)
        
        goals = jnp.concatenate([
            goal_pos,
            jnp.zeros((self.num_goals, self.state_dim - 2), dtype=goal_pos.dtype)
        ], axis=1)
        
        # 创建环境状态对象
        env_states = LidarEnvState(states, goals, obstacles)
        
        # 获取激光雷达数据
        lidar_data = self.get_lidar_data(states, obstacles)
        
        # 返回图结构
        return self.get_graph(env_states, lidar_data)

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> float:
        """
        计算奖励函数

        参数:
            graph: 当前环境状态的图表示
            action: 智能体采取的动作, 形状 (N, action_dim)

        返回:
            标量奖励值
        """
        # ... (内部逻辑保持不变, 注释已更新) ...
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents) # 形状 (N, state_dim)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)         # 形状 (N, state_dim)
        reward = jnp.zeros(()).astype(jnp.float32)                          # 标量

        # 目标距离惩罚
        agent_pos = agent_states[:, :2] # 形状 (N, 2)
        goal_pos = goals[:, :2]         # 形状 (N, 2)
        # dist2goal 形状 (N,)
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= (dist2goal.mean()) * 0.01 # 平均距离惩罚

        # 未到达目标惩罚
        # jnp.where(...) 形状 (N,)
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # 动作惩罚
        # jnp.linalg.norm(...)**2 形状 (N,)
        reward -= (jnp.linalg.norm(action[:, :2], axis=1) ** 2).mean() * 0.0001  # 线加速度惩罚
        # jnp.abs(...)**2 形状 (N,)
        reward -= (jnp.abs(action[:, 2]) ** 2).mean() * self._params["rotation_penalty"] # 角加速度惩罚

        # 角速度惩罚
        omega = agent_states[:, 6] # 形状 (N,)
        # jnp.abs(...)**2 形状 (N,)
        reward -= (jnp.abs(omega) ** 2).mean() * self._params["rotation_penalty"] * 0.5

        return reward

    def state2feat(self, state: State) -> Array:
        """将状态转换为特征表示 (保持不变)"""
        # 输入: state, 形状 (state_dim,)
        # 输出: state, 形状 (state_dim,)
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Array] = None) -> list[EdgeBlock]:
        """
        构建图神经网络的边块 - 加入关键边特征和局部坐标系特征

        参数:
            state: 当前环境状态 (LidarEnvState)
            lidar_data: 激光雷达数据, 形状 (num_hits, 2) 或 None

        返回:
            包含 EdgeBlock 对象的列表
        """
        agent_states = state.agent # 形状 (N, state_dim)
        agent_pos = agent_states[:, :2] # 形状 (N, 2)
        agent_ori = agent_states[:, 2:4] # 形状 (N, 2), [cos(psi), sin(psi)]

        # --- 1. 智能体-智能体 边 ---
        # pos_diff[i, j] = pos_i - pos_j, 形状 (N, N, 2)
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        # edge_feats_rel[i, j] = feat_i - feat_j, 形状 (N, N, state_dim)
        edge_feats_rel_state = (jax_vmap(self.state2feat)(agent_states)[:, None, :] -
                                jax_vmap(self.state2feat)(agent_states)[None, :, :])
        # 仅提取与 edge_dim - 3 (即 7) 相关的状态特征 (例如，忽略类型指示符)
        # 假设 state2feat 直接返回 state，前 7 个是需要的部分
        edge_feats_rel = edge_feats_rel_state[:, :, :7] # 形状 (N, N, 7)

        # 计算局部坐标系下的相对位置
        # 注意这里需要调整，pos_diff[i,j] = pos_i - pos_j，但我们需要 p_j - p_i
        # 因此需要对 pos_diff 取负值
        global_pos_diff = -pos_diff # 形状 (N, N, 2), 现在 [i,j] = p_j - p_i

        # 对每个智能体对计算旋转矩阵转置
        # cos_psi_i 形状 (N,), sin_psi_i 形状 (N,)
        cos_psi_i = agent_ori[:, 0]
        sin_psi_i = agent_ori[:, 1]
        # R_i_T 形状 (N, 2, 2)
        R_i_T = batch_rotation_matrix_transpose(cos_psi_i, sin_psi_i)
        
        # 计算局部坐标系下的相对位置 p_j^i = R_i^T * (p_j - p_i)
        # 需要为每个发送者i (第一维)广播旋转矩阵
        # R_i_T_expanded 形状 (N, 1, 2, 2)
        R_i_T_expanded = R_i_T[:, None, :, :]
        # global_pos_diff_expanded 形状 (N, N, 2, 1)
        global_pos_diff_expanded = global_pos_diff[:, :, :, None]
        # p_j_i 形状 (N, N, 2, 1)
        p_j_i_expanded = jnp.matmul(R_i_T_expanded, global_pos_diff_expanded)
        # p_j_i 形状 (N, N, 2)
        p_j_i = p_j_i_expanded[:, :, :, 0]
        
        # 提取局部坐标系特征
        # i_x_j 形状 (N, N, 1) - 局部坐标系x分量(前向分量)
        i_x_j = p_j_i[:, :, 0:1]
        # 计算局部坐标系下的距离 ||p_j^i||
        # norm_p_j_i 形状 (N, N, 1)
        norm_p_j_i = jnp.linalg.norm(p_j_i, axis=-1, keepdims=True)

        # dist[i, j] = ||pos_i - pos_j||, 形状 (N, N)
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        # dist_no_self 形状 (N, N)
        dist_no_self = dist + jnp.eye(self.num_agents) * (self._params["comm_radius"] + 1)
        # agent_agent_mask 形状 (N, N), bool 类型
        agent_agent_mask = jnp.less(dist_no_self, self._params["comm_radius"])

        # 创建 "是否关键边" 特征
        # is_critical_feat 形状 (N, N, 1)
        is_critical_feat = jnp.zeros((self.num_agents, self.num_agents, 1))
        if self.num_agents > 1:
            idx_sender = jnp.arange(self.num_agents - 1)      # 形状 (N-1,)
            idx_receiver = jnp.arange(1, self.num_agents)    # 形状 (N-1,)
            is_critical_feat = is_critical_feat.at[idx_sender, idx_receiver, 0].set(1.0)

        # 拼接特征: 原始特征 + 关键边特征 + 局部坐标系特征 
        # edge_feats_combined 形状 (N, N, 10)
        edge_feats_combined = jnp.concatenate([
            edge_feats_rel,       # 原始边特征 (7维)
            is_critical_feat,     # 关键边标记 (1维) 
            norm_p_j_i,           # 局部距离 (1维)
            i_x_j                 # 局部前向分量 (1维)
        ], axis=-1)
        assert edge_feats_combined.shape[-1] == self.edge_dim

        # 创建边块
        id_agent = jnp.arange(self.num_agents) # 形状 (N,)
        agent_agent_edges = EdgeBlock(edge_feats_combined, agent_agent_mask, id_agent, id_agent)

        # --- 2. 智能体-目标 边 ---
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent] # 形状 (state_dim,)
            goal_state_i = state.goal[i_agent]   # 形状 (state_dim,)
            # agent_goal_feats_i 形状 (state_dim,)
            agent_goal_feats_i_full = self.state2feat(agent_state_i) - self.state2feat(goal_state_i)
            # 取前 7 个特征
            agent_goal_feats_i = agent_goal_feats_i_full[:7] # 形状 (7,)
            # 填充零以匹配 edge_dim (目标边不是关键边，没有局部坐标系特征)
            # padding 形状 (1, 1, 3)
            padding = jnp.zeros((1, 1, self.edge_dim - agent_goal_feats_i.shape[-1]))
            # agent_goal_feats_padded 形状 (1, 1, 10)
            agent_goal_feats_padded = jnp.concatenate([agent_goal_feats_i[None, None, :], padding], axis=-1)

            sender_idx = jnp.array([i_agent]) # 形状 (1,)
            receiver_idx = jnp.array([i_agent + self.num_agents]) # 形状 (1,)
            mask = jnp.ones((1, 1), dtype=bool) # 形状 (1, 1)

            agent_goal_edges.append(EdgeBlock(agent_goal_feats_padded, mask, sender_idx, receiver_idx))


        # --- 3. 智能体-障碍物 边 ---
        agent_obs_edges = []
        n_hits_expected = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None and n_hits_expected > 0:
            if lidar_data.size == 0:
                 lidar_data = jnp.zeros((0, 2)) # lidar_data 形状 (0, 2)

            obs_start_idx = self.num_agents + self.num_goals
            n_actual_hits = lidar_data.shape[0] # 实际 hit 点数
            # id_obs 形状 (n_actual_hits,)
            id_obs = jnp.arange(obs_start_idx, obs_start_idx + n_actual_hits)

            for i in range(self.num_agents):
                hit_start_idx = i * self._params["top_k_rays"]
                hit_end_idx = (i + 1) * self._params["top_k_rays"]
                actual_start = min(hit_start_idx, n_actual_hits)
                actual_end = min(hit_end_idx, n_actual_hits)
                n_hits_for_agent = actual_end - actual_start # 当前智能体的 hit 数

                if n_hits_for_agent > 0:
                    # id_hits_indices 形状 (n_hits_for_agent,)
                    id_hits_indices = jnp.arange(actual_start, actual_end)
                    # id_obs_nodes 形状 (n_hits_for_agent,)
                    id_obs_nodes = id_obs[id_hits_indices]
                    # lidar_points 形状 (n_hits_for_agent, 2)
                    lidar_points = lidar_data[id_hits_indices, :]
                    # lidar_feats_pos 形状 (n_hits_for_agent, 2)
                    lidar_feats_pos = agent_pos[i, :] - lidar_points
                    # lidar_dist 形状 (n_hits_for_agent,)
                    lidar_dist = jnp.linalg.norm(lidar_feats_pos, axis=-1)

                    # 填充零以匹配 edge_dim
                    # padding_obs 形状 (n_hits_for_agent, 8)
                    padding_obs = jnp.zeros((n_hits_for_agent, self.edge_dim - lidar_feats_pos.shape[1]))
                    # lidar_feats_padded 形状 (n_hits_for_agent, 10)
                    lidar_feats_padded = jnp.concatenate([lidar_feats_pos, padding_obs], axis=-1)

                    # 创建掩码
                    # active_lidar 形状 (n_hits_for_agent,)
                    active_lidar = jnp.less(lidar_dist, self._params["comm_radius"])
                    # agent_obs_mask 形状 (1, n_hits_for_agent)
                    agent_obs_mask = jnp.ones((1, n_hits_for_agent), dtype=bool)
                    agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar[None, :])

                    sender_idx = id_agent[i][None] # 形状 (1,)
                    # lidar_feats_padded[None, :, :] 形状 (1, n_hits_for_agent, 10)
                    agent_obs_edges.append(
                        EdgeBlock(lidar_feats_padded[None, :, :],
                                  agent_obs_mask,
                                  sender_idx,
                                  id_obs_nodes) # receiver 形状 (n_hits_for_agent,)
                    )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """定义状态的上下限"""
        # lower_lim 形状 (state_dim,)
        lower_lim = jnp.array([0., 0., -1., -1., -2., -2., -self._params["max_angular_vel"]])
        # upper_lim 形状 (state_dim,)
        upper_lim = jnp.array([self.area_size, self.area_size, 1., 1., 2., 2., self._params["max_angular_vel"]])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """定义动作的上下限"""
        # 初始化限制
        lower_lim = jnp.ones(self.action_dim) * -1
        upper_lim = jnp.ones(self.action_dim) * 1
        
        # 修改角加速度（第3个维度）的限制为 ±1000
        lower_lim = lower_lim.at[2].set(-1000)
        upper_lim = upper_lim.at[2].set(1000)
        
        return lower_lim, upper_lim

    def get_cost(self, graph: GraphsTuple) -> Cost:
        """
        计算环境中的代价，包括碰撞和分离的 FoV 约束。

        计算五种类型的代价:
        Dim 0: 智能体间碰撞 (所有对): `2*r - min_{k!=i} ||p_k - p_i||`
        Dim 1: 智能体与障碍物碰撞: `r - min_dist_obs`
        Dim 2: FoV 角度约束 (关键对 i->i+1): `h_angle = cos(beta) * ||p_j^i|| - i_x_j`
        Dim 3: FoV 最大距离约束 (关键对 i->i+1): `h_range = ||p_j^i|| - r_max`
        Dim 4: FoV 最小距离约束 (关键对 i->i+1): `h_coll_fov = D - ||p_j^i||`

        代价为负值或零表示安全，正值表示违规。

        参数:
            graph: 当前环境状态的图表示

        返回:
            Cost: 形状为 (N, 5) 的数组，表示每个智能体的五种代价
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents) # 形状 (N, state_dim)
        agent_pos = agent_states[:, :2]    # 形状 (N, 2)
        agent_ori = agent_states[:, 2:4]   # 形状 (N, 2), [cos(psi), sin(psi)]

        N = self.num_agents # 智能体数量

        # --- 计算距离矩阵 ---
        # dist_mat[i, j] = ||pos_i - pos_j||, 形状 (N, N)
        dist_mat = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)

        # --- 1. 智能体间碰撞 (Cost 0) ---
        # 适用于所有智能体对 (i, k) where k != i
        # 忽略自身距离 (对角线)
        dist_mat_no_self = dist_mat + jnp.eye(N) * 1e6 # 形状 (N, N)
        # 找到每个智能体 i 到其他所有智能体 k (k!=i) 的最小距离
        min_dist_agent = jnp.min(dist_mat_no_self, axis=1) # 形状 (N,)
        # 计算碰撞成本: 2*r - 最小距离 (如果距离 < 2r，则为正)
        # agent_coll_cost 形状 (N,)
        agent_coll_cost: Array = self.params["car_radius"] * 2 - min_dist_agent

        # --- 2. 智能体与障碍物碰撞 (Cost 1) ---
        # obs_coll_cost 形状 (N,)
        if self.params['n_obs'] == 0 or self._params["top_k_rays"] == 0:
            obs_coll_cost = jnp.zeros((N,)).astype(jnp.float32)
        else:
            num_obstacle_nodes = graph.nodes.shape[0] - N - self.num_goals # 障碍物节点数
            if num_obstacle_nodes > 0:
                 # obs_pos_flat 形状 (num_obstacle_nodes, 2)
                 obs_pos_flat = graph.type_states(type_idx=2, n_type=num_obstacle_nodes)[:, :2]
                 # dist_obs_flat[i, k] = ||agent_pos_i - obs_pos_k||, 形状 (N, num_obstacle_nodes)
                 dist_obs_flat = jnp.linalg.norm(obs_pos_flat[None, :, :] - agent_pos[:, None, :], axis=-1)
                 # min_dist_obs 形状 (N,)
                 min_dist_obs = dist_obs_flat.min(axis=1)
                 # 计算碰撞成本: r - 最小距离 (如果距离 < r，则为正)
                 obs_coll_cost: Array = self.params["car_radius"] - min_dist_obs
            else: # 没有障碍物节点
                obs_coll_cost = jnp.zeros((N,)).astype(jnp.float32)

        # --- 3, 4, 5. FoV 约束 (Cost 2, 3, 4) ---
        # 仅为关键对 (i, j=i+1) 计算，其中 i = 0 to N-2
        # Agent N-1 没有 FoV 目标，其 FoV 相关成本设为安全值 (-1.0)
        safe_value = -1.0 # 定义一个表示安全的成本值
        # 初始化 FoV 相关成本向量，形状 (N,)
        fov_angle_cost = jnp.full((N,), safe_value)
        fov_range_cost = jnp.full((N,), safe_value)
        fov_coll_cost = jnp.full((N,), safe_value)

        if N > 1:
            # 获取发送者 i (0 to N-2) 和接收者 j=i+1 (1 to N-1) 的状态
            states_i = agent_states[:-1] # 形状 (N-1, state_dim)
            states_j = agent_states[1:]  # 形状 (N-1, state_dim)

            pos_i = states_i[:, :2]    # 形状 (N-1, 2)
            ori_i = states_i[:, 2:4]   # 形状 (N-1, 2)
            pos_j = states_j[:, :2]    # 形状 (N-1, 2)

            # 计算全局坐标系中的相对位置: p_j - p_i
            # delta_p_global 形状 (N-1, 2)
            delta_p_global = pos_j - pos_i

            # 计算智能体 i 局部坐标系中的相对位置: p_j^i = R_i^T * (p_j - p_i)
            # R_i_T 形状 (N-1, 2, 2)
            R_i_T = batch_rotation_matrix_transpose(ori_i[:, 0], ori_i[:, 1])
            # p_j_i 形状 (N-1, 2)
            p_j_i = jnp.squeeze(R_i_T @ delta_p_global[:, :, None], axis=-1)

            # 提取局部坐标系分量
            i_x_j = p_j_i[:, 0] # 形状 (N-1,)

            # 计算相对距离 ||p_j^i||
            # norm_p_j_i 形状 (N-1,)
            norm_p_j_i = jnp.linalg.norm(p_j_i, axis=-1)
            # norm_p_j_i_safe 形状 (N-1,)
            norm_p_j_i_safe = norm_p_j_i + 1e-8 # 防止数值问题

            # 计算 FoV 障碍函数 h(p_j^i) <= 0 表示安全
            # h_angle: 正值表示违规
            # h_angle 形状 (N-1,)
            h_angle = self._cos_fov_beta * norm_p_j_i_safe - i_x_j

            # h_range: 正值表示违规
            # h_range 形状 (N-1,)
            h_range = norm_p_j_i - self._r_max

            # h_coll_fov: 正值表示违规
            # h_coll_fov 形状 (N-1,)
            h_coll_fov = self._min_safe_dist_D - norm_p_j_i

            # 将计算得到的 FoV 成本填充到对应智能体 i 的成本向量中
            fov_angle_cost = fov_angle_cost.at[:-1].set(h_angle)
            fov_range_cost = fov_range_cost.at[:-1].set(h_range)
            fov_coll_cost = fov_coll_cost.at[:-1].set(h_coll_fov)

        # --- 组合所有成本 ---
        # 堆叠: agent_coll, obs_coll, fov_angle, fov_range, fov_coll_fov
        # cost 形状 (N, 5)
        cost = jnp.stack([agent_coll_cost, obs_coll_cost, fov_angle_cost, fov_range_cost, fov_coll_cost], axis=1)

        assert cost.shape == (N, self.n_cost) # 确认成本形状

        # --- 应用边距 (margin) ---
        eps = 0.1 # 使用一个较小的边距值
        # cost_with_margin 形状 (N, 5)
        cost_with_margin = jnp.where(cost <= 0.0, cost - eps, cost + eps)

        # --- 裁剪成本 ---
        # clipped_cost 形状 (N, 5)
        clipped_cost = jnp.clip(cost_with_margin, a_min=-1.0, a_max=1.0)

        return clipped_cost