import jax.numpy as jnp
import numpy as np
import jax.random as jr

from typing import Tuple, Optional

from dgppo.utils.graph import EdgeBlock, GraphsTuple
from dgppo.utils.typing import Action, Array, State, AgentState, Cost
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.env.utils import get_node_goal_rng, inside_obstacles
from dgppo.utils.utils import jax_vmap


class LidarOmniTarget(LidarEnv):
    """
    麦轮车（全向轮）智能体激光雷达环境
    
    该环境模拟具有全向移动能力的机器人，类似于装有麦克纳姆轮或麦轮的移动平台。
    这种机器人可以在不改变朝向的情况下向任意方向移动，同时也可以原地旋转。
    与典型的差速机器人或自行车模型不同，全向轮机器人没有非完整约束。
    """

    PARAMS = {
        "car_radius": 0.05,  # 车辆半径
        "comm_radius": 0.5,  # 通信半径
        "n_rays": 32,  # 激光雷达射线数量
        "obs_len_range": [0.1, 0.3],  # 障碍物尺寸范围
        "n_obs": 3,  # 障碍物数量
        "default_area_size": 1.5,  # 默认环境大小
        "dist2goal": 0.01,  # 到目标的距离阈值
        "top_k_rays": 8,  # 激光雷达选择的重要射线数量
        "max_angular_vel": 1.0,  # 最大角速度
        "rotation_penalty": 0.001,  # 旋转惩罚因子
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

    @property
    def state_dim(self) -> int:
        """
        定义智能体状态维度
        
        麦轮车状态包含:
        - 位置坐标(x,y)
        - 朝向(cos(θ),sin(θ))
        - 线速度(vx,vy)
        - 角速度(ω)
        """
        return 7  # x, y, cos(theta), sin(theta), vx, vy, omega

    @property
    def node_dim(self) -> int:
        """
        定义节点特征维度
        
        节点特征包含:
        - 状态维度(7)
        - 节点类型指示器(3): agent: 001, goal: 010, obstacle: 100
        """
        return 10  # state dim (7) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        """边特征维度：相对位置、速度和朝向差异"""
        return 7  # x_rel, y_rel, cos_rel, sin_rel, vx_rel, vy_rel, omega_rel

    @property
    def action_dim(self) -> int:
        """
        定义动作空间维度
        
        麦轮车动作包含:
        - x方向加速度
        - y方向加速度
        - 角加速度
        """
        return 3  # ax, ay, alpha(角加速度)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """
        使用欧拉积分法更新智能体状态
        
        麦轮车的运动更新包括:
        1. 位置更新：根据当前速度和时间步长
        2. 朝向更新：根据当前角速度
        3. 速度更新：根据施加的加速度
        
        参数:
            agent_states: 当前智能体状态
            action: 智能体动作(ax, ay, alpha)
            
        返回:
            更新后的智能体状态
        """
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        
        # 提取当前状态
        pos = agent_states[:, :2]  # 位置
        orientation = agent_states[:, 2:4]  # 朝向[cos(θ), sin(θ)]
        vel = agent_states[:, 4:6]  # 线速度
        omega = agent_states[:, 6:7]  # 角速度
        
        # 分解动作
        acc = action[:, :2] * 10.0  # 线加速度
        alpha = action[:, 2:3] * 5.0  # 角加速度
        
        # 计算状态变化率
        pos_dot = vel
        
        # 角度增量
        delta_theta = omega * self.dt
        # 当前朝向角度(使用arctan2恢复角度)
        theta = jnp.arctan2(orientation[:, 1], orientation[:, 0])
        # 更新角度
        new_theta = theta + delta_theta[:, 0]
        # 计算新的朝向
        new_cos = jnp.cos(new_theta)
        new_sin = jnp.sin(new_theta)
        new_orientation = jnp.stack([new_cos, new_sin], axis=1)
        
        # 速度更新(全向轮车辆可以在任意方向加速，不受朝向限制)
        vel_dot = acc
        omega_dot = alpha
        
        # 使用欧拉积分法更新状态
        new_pos = pos + pos_dot * self.dt
        new_vel = vel + vel_dot * self.dt
        new_omega = omega + omega_dot * self.dt
        
        # 组合为新状态
        new_state = jnp.concatenate([new_pos, new_orientation, new_vel, new_omega], axis=1)
        
        # 确保不超出状态限制
        return self.clip_state(new_state)

    def reset(self, key: Array) -> GraphsTuple:
        """
        重置环境，生成新的初始状态
        
        过程包括:
        1. 生成随机障碍物
        2. 生成智能体和目标的初始位置
        3. 为智能体分配随机朝向
        4. 获取激光雷达数据
        """
        # 随机生成障碍物
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
                (self._params["n_obs"], 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # 随机生成智能体和目标位置
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        
        # 生成随机朝向角
        theta_key, key = jr.split(key, 2)
        thetas = jr.uniform(theta_key, (self.num_agents,), minval=0, maxval=2 * np.pi)
        theta_states = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
        
        # 合并位置和朝向，并添加零速度初始状态
        states = jnp.concatenate([
            states,  # 位置(x,y)
            theta_states,  # 朝向(cos(θ),sin(θ))
            jnp.zeros((self.num_agents, 3), dtype=states.dtype)  # 速度(vx,vy,ω)
        ], axis=1)
        
        # 为目标添加零填充，使其维度与状态匹配
        goals = jnp.concatenate([
            goals,  # 位置(x,y)
            jnp.zeros((self.num_goals, 5), dtype=goals.dtype)  # 填充零
        ], axis=1)
        
        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)
        env_states = LidarEnvState(states, goals, obstacles)

        # 获取激光雷达数据
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> float:
        """
        计算奖励函数
        
        奖励包含:
        1. 到目标距离惩罚
        2. 未到达目标惩罚
        3. 动作幅度惩罚
        4. 角速度惩罚(鼓励平滑旋转)
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        # 目标距离惩罚
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= (dist2goal.mean()) * 0.01

        # 未到达目标惩罚
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # 动作惩罚(线加速度和角加速度)
        reward -= (jnp.linalg.norm(action[:, :2], axis=1) ** 2).mean() * 0.0001  # 线加速度惩罚
        reward -= (jnp.abs(action[:, 2]) ** 2).mean() * self._params["rotation_penalty"]  # 角加速度惩罚

        # 角速度惩罚(鼓励平滑旋转)
        omega = agent_states[:, 6]
        reward -= (jnp.abs(omega) ** 2).mean() * self._params["rotation_penalty"] * 0.5

        return reward

    def state2feat(self, state: State) -> Array:
        """将状态转换为特征表示"""
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Array] = None) -> list[EdgeBlock]:
        """
        构建图神经网络的边块
        
        创建三种边关系:
        1. 智能体-智能体: 当两个智能体距离小于通信半径时连接
        2. 智能体-目标: 每个智能体与其对应的目标连接
        3. 智能体-障碍物: 基于激光雷达探测结果
        """
        # 智能体-智能体连接
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # 智能体-目标连接
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = self.state2feat(agent_state_i) - self.state2feat(goal_state_i)
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # 智能体-障碍物连接(基于激光雷达)
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        定义状态的上下限
        
        限制:
        - 位置在环境边界内
        - 朝向(cos,sin)在[-1,1]范围内
        - 速度和角速度有合理上限
        """
        lower_lim = jnp.array([0., 0., -1., -1., -2., -2., -self._params["max_angular_vel"]])
        upper_lim = jnp.array([self.area_size, self.area_size, 1., 1., 2., 2., self._params["max_angular_vel"]])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """定义动作的上下限"""
        lower_lim = jnp.ones(3) * -1.0
        upper_lim = jnp.ones(3)
        return lower_lim, upper_lim
        
    def get_cost(self, graph: GraphsTuple) -> Cost:
        """
        计算环境中的代价（碰撞惩罚）
        
        该方法与基类LidarEnv中的get_cost方法完全相同，只是被覆盖定义在LidarOmniTarget中，
        以便将来可以根据全向轮车辆的特性进行定制修改。
        
        计算两种类型的代价:
        1. 智能体间碰撞: 2*car_radius - 最小距离
        2. 智能体与障碍物碰撞: car_radius - 最小距离
        
        代价为负值表示安全，正值表示发生碰撞
        
        参数:
            graph: 环境的图表示
            
        返回:
            形状为(num_agents, 2)的数组，表示每个智能体的两种代价
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost 