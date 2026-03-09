import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class DummyPbar:
    """
    作用：提供一个在禁用 tqdm 时使用的静默进度条替身。
    输入：无。
    输出：DummyPbar 类。
    """

    def __init__(self, total=0, desc=""):
        """
        作用：初始化静默进度条需要的最小状态。
        输入：total: 总步数；desc: 描述文本。
        输出：无。
        """
        self.total = total
        self.desc = desc

    def __enter__(self):
        """
        作用：支持像真实进度条一样被 with 语句管理。
        输入：无。
        输出：self。
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        作用：退出上下文时保持异常继续向外传播。
        输入：exc_type/exc/tb: 异常信息。
        输出：False。
        """
        return False

    def update(self, n=1):
        """
        作用：在静默模式下忽略进度更新。
        输入：n: 本次增加的步数。
        输出：None。
        """
        return None

    def set_postfix(self, *args, **kwargs):
        """
        作用：在静默模式下忽略 postfix 更新。
        输入：*args/**kwargs: 与 tqdm 兼容的参数。
        输出：None。
        """
        return None


def make_pbar(total, desc):
    """
    作用：根据环境变量返回真实 tqdm 或静默进度条。
    输入：total: 总步数；desc: 描述文本。
    输出：tqdm 实例或 DummyPbar 实例。
    """
    force_tqdm = os.getenv("FORCE_TQDM", "0") == "1"
    if (os.getenv("DISABLE_TQDM", "0") == "1") and (not force_tqdm):
        return DummyPbar(total=total, desc=desc)
    return tqdm(total=total, desc=desc)


class RunningMeanStd:
    """
    作用：维护标量序列的运行均值与方差，用于归一化 reward return。
    输入：无。
    输出：RunningMeanStd 类。
    """

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, x):
        """
        作用：用一批样本更新运行均值和方差。
        输入：x: 数值数组。
        输出：无。
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return
        batch_mean = float(arr.mean())
        batch_var = float(arr.var())
        batch_count = float(arr.size)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.mean = float(new_mean)
        self.var = float(max(m2 / total_count, 1e-8))
        self.count = float(total_count)

    def state_dict(self):
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, state):
        if not state:
            return
        self.mean = float(state.get("mean", self.mean))
        self.var = float(max(state.get("var", self.var), 1e-8))
        self.count = float(max(state.get("count", self.count), 1e-4))


class ValueNet(torch.nn.Module):
    """
    作用：定义输出标量状态价值的 critic 网络。
    输入：无。
    输出：ValueNet 类。
    """

    def __init__(self, state_dim, hidden_dim):
        """
        作用：构建两层全连接 critic。
        输入：state_dim: 状态维度；hidden_dim: 隐层维度。
        输出：无。
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        作用：前向计算一批状态的价值估计。
        输入：x: 状态张量。
        输出：价值张量。
        """
        return self.fc2(F.relu(self.fc1(x)))


class PolicyNetContinuous(torch.nn.Module):
    """
    作用：定义连续动作 actor，为每个动作维度输出均值和标准差。
    输入：无。
    输出：PolicyNetContinuous 类。
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        std_offset=0.02,
        min_std=0.015,
        max_std=0.35,
        init_std_bias=-2.2,
    ):
        """
        作用：构建 actor 网络并统一动作边界表示。
        输入：state_dim/hidden_dim/action_dim: 网络维度；action_bound: 动作边界。
        输出：无。
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.std_offset = float(std_offset)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        action_bound_arr = np.array(action_bound, dtype=np.float32)
        if action_bound_arr.ndim == 0:
            action_bound_arr = np.full((action_dim,), float(action_bound_arr), dtype=np.float32)
        self.register_buffer("action_bound", torch.tensor(action_bound_arr, dtype=torch.float32).view(1, -1))

        torch.nn.init.normal_(self.fc1.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.fc_mu.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.fc_std.weight, 0.0, 0.02)
        torch.nn.init.constant_(self.fc_mu.bias, 0.0)
        torch.nn.init.constant_(self.fc_std.bias, float(init_std_bias))

    def forward(self, x, explore_decay=1.0):
        """
        作用：前向计算 squash 前动作分布的均值和探索尺度。
        输入：x: 状态张量；explore_decay: 探索衰减系数。
        输出：mu 和 std 张量。
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        std = torch.clamp(
            (F.softplus(self.fc_std(h)) + self.std_offset) * explore_decay,
            self.min_std,
            self.max_std,
        )
        return mu, std


class PPOContinuous:
    """
    作用：实现面向连续 residual/regen/coast 动作空间的 PPO 智能体。
    输入：无。
    输出：PPOContinuous 类。
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
        action_bound=10.0,
        minibatch_size=64,
        target_kl=0.01,
        init_explore_decay=0.90,
        decay_rate=0.993,
        min_explore=0.08,
        max_explore=1.20,
        entropy_coef=0.0003,
        entropy_coef_min=0.00005,
        entropy_coef_max=0.0025,
        entropy_adapt_rate=0.08,
        entropy_kl_low_ratio=0.65,
        entropy_kl_high_ratio=1.35,
        explore_expand=1.06,
        explore_shrink=0.94,
        std_offset=0.02,
        min_std=0.015,
        max_std=0.35,
        init_std_bias=-2.2,
        base_state_dim=None,
        constraint_names=None,
        constraint_critic_lr=None,
        use_return_rms=1.0,
        return_rms_eps=1e-4,
        return_rms_clip=5.0,
    ):
        """
        作用：初始化 actor、critic 以及 PPO 训练超参数。
        输入：状态维度、动作维度、学习率、折扣因子等 PPO 配置。
        输出：无。
        """
        self.actor = PolicyNetContinuous(
            state_dim,
            hidden_dim,
            action_dim,
            action_bound,
            std_offset=std_offset,
            min_std=min_std,
            max_std=max_std,
            init_std_bias=init_std_bias,
        ).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.constraint_names = tuple(
            constraint_names
            if constraint_names is not None
            else ("speed", "distance", "smoothness", "net_energy", "projection", "underspeed", "window_energy")
        )
        self.cost_critics = torch.nn.ModuleDict(
            {name: ValueNet(state_dim, hidden_dim) for name in self.constraint_names}
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critics.parameters(),
            lr=(critic_lr if constraint_critic_lr is None else constraint_critic_lr),
        )

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.action_dim = action_dim
        self.base_state_dim = int(state_dim if base_state_dim is None else base_state_dim)
        action_bound_arr = np.array(action_bound, dtype=np.float32)
        if action_bound_arr.ndim == 0:
            action_bound_arr = np.full((action_dim,), float(action_bound_arr), dtype=np.float32)
        self.action_bound = torch.tensor(action_bound_arr, dtype=torch.float32, device=device).view(1, -1)

        self.explore_decay = float(init_explore_decay)
        self.decay_rate = float(decay_rate)
        self.min_explore = float(min_explore)
        self.max_explore = float(max_explore)
        self.entropy_coef = float(entropy_coef)
        self.entropy_coef_min = float(entropy_coef_min)
        self.entropy_coef_max = float(entropy_coef_max)
        self.entropy_adapt_rate = float(entropy_adapt_rate)
        self.entropy_kl_low_ratio = float(entropy_kl_low_ratio)
        self.entropy_kl_high_ratio = float(entropy_kl_high_ratio)
        self.explore_expand = float(explore_expand)
        self.explore_shrink = float(explore_shrink)
        self.squash_eps = 1e-6
        self.context_action_hint_dim = 7
        self.use_return_rms = bool(use_return_rms)
        self.return_rms_clip = float(return_rms_clip)
        self.reward_return_rms = RunningMeanStd(epsilon=return_rms_eps)

    def get_auxiliary_state_dict(self):
        """
        作用：导出 reward critic 之外的训练附加状态，供 checkpoint 使用。
        输入：无。
        输出：包含多约束 critic 权重的字典。
        """
        return {
            "cost_critics": self.cost_critics.state_dict(),
            "adaptive_policy_state": {
                "explore_decay": float(self.explore_decay),
                "entropy_coef": float(self.entropy_coef),
            },
            "reward_return_rms": self.reward_return_rms.state_dict(),
        }

    def load_auxiliary_state_dict(self, state):
        """
        作用：恢复 reward critic 之外的训练附加状态。
        输入：state: 由 get_auxiliary_state_dict 生成的状态字典。
        输出：无。
        """
        if not state:
            return
        cost_state = state.get("cost_critics")
        if cost_state:
            self.cost_critics.load_state_dict(cost_state, strict=False)
        adaptive_state = state.get("adaptive_policy_state")
        if adaptive_state:
            self.explore_decay = float(adaptive_state.get("explore_decay", self.explore_decay))
            self.entropy_coef = float(adaptive_state.get("entropy_coef", self.entropy_coef))
        self.reward_return_rms.load_state_dict(state.get("reward_return_rms"))

    def _reward_value_mean_std(self, device):
        mean = torch.tensor(self.reward_return_rms.mean, dtype=torch.float32, device=device)
        std = torch.tensor(np.sqrt(max(self.reward_return_rms.var, 1e-8)), dtype=torch.float32, device=device)
        return mean, std

    def _denormalize_reward_value(self, value_pred):
        if not self.use_return_rms:
            return value_pred
        mean, std = self._reward_value_mean_std(value_pred.device)
        return value_pred * std + mean

    def _normalize_reward_return(self, reward_return):
        if not self.use_return_rms:
            return reward_return
        mean, std = self._reward_value_mean_std(reward_return.device)
        normalized = (reward_return - mean) / torch.clamp(std, min=1e-6)
        if self.return_rms_clip > 0.0:
            normalized = torch.clamp(normalized, -self.return_rms_clip, self.return_rms_clip)
        return normalized

    def _adaptive_exploration_update(self, mean_kl):
        """
        作用：基于目标 KL 自适应调节动作标准差尺度和熵正则系数。
        输入：mean_kl: 本轮 PPO 更新的平均近似 KL。
        输出：无。
        """
        if mean_kl <= 0.0 or self.target_kl <= 0.0:
            return
        low_kl = self.target_kl * self.entropy_kl_low_ratio
        high_kl = self.target_kl * self.entropy_kl_high_ratio
        if mean_kl < low_kl:
            self.explore_decay = min(self.max_explore, self.explore_decay * self.explore_expand)
            self.entropy_coef = min(
                self.entropy_coef_max,
                self.entropy_coef * (1.0 + self.entropy_adapt_rate),
            )
        elif mean_kl > high_kl:
            self.explore_decay = max(self.min_explore, self.explore_decay * self.explore_shrink)
            self.entropy_coef = max(
                self.entropy_coef_min,
                self.entropy_coef * (1.0 - self.entropy_adapt_rate),
            )
        else:
            mid_kl = 0.5 * (low_kl + high_kl)
            adapt_mag = 0.25 * self.entropy_adapt_rate
            if mean_kl < mid_kl:
                self.explore_decay = min(self.max_explore, self.explore_decay * (1.0 + adapt_mag))
                self.entropy_coef = min(self.entropy_coef_max, self.entropy_coef * (1.0 + adapt_mag))
            else:
                self.explore_decay = max(self.min_explore, self.explore_decay * (1.0 - adapt_mag))
                self.entropy_coef = max(self.entropy_coef_min, self.entropy_coef * (1.0 - adapt_mag))

    def _latest_base_state(self, state_tensor):
        """
        作用：从可能堆叠的观测中提取最近一帧基础状态。
        输入：state_tensor: 单帧或堆叠后的状态张量。
        输出：最近一帧基础状态张量。
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.size(1) <= self.base_state_dim:
            return state_tensor[:, -self.base_state_dim :]
        return state_tensor[:, -self.base_state_dim :]

    def _action_parameterization(self, states):
        """
        作用：根据状态中的动作语义提示构建当前动作的中心与尺度。
        输入：states: 状态张量。
        输出：action_center 与 action_scale。
        """
        latest = self._latest_base_state(states)
        batch_size = latest.size(0)
        center = torch.zeros((batch_size, self.action_dim), dtype=latest.dtype, device=latest.device)
        scale = self.action_bound.expand(batch_size, -1).clone()
        if latest.size(1) < (24 + self.context_action_hint_dim):
            return center, scale

        residual_scale_hint = torch.clamp(latest[:, 25], 0.35, 1.0)
        regen_center_hint = torch.clamp(latest[:, 27], -1.0 + 0.04, 1.0 - 0.04)
        regen_range_hint = torch.clamp(latest[:, 28], 0.12, 0.65)
        coast_center_hint = torch.clamp(latest[:, 29], -1.0 + 0.04, 1.0 - 0.04)
        coast_range_hint = torch.clamp(latest[:, 30], 0.12, 0.65)
        residual_scale = torch.clamp(
            self.action_bound[:, 0] * residual_scale_hint.unsqueeze(1),
            0.28 * self.action_bound[:, 0],
            self.action_bound[:, 0],
        )
        regen_room = torch.clamp(
            torch.minimum((1.0 - self.squash_eps) - regen_center_hint, (1.0 - self.squash_eps) + regen_center_hint),
            min=0.05,
        )
        coast_room = torch.clamp(
            torch.minimum((1.0 - self.squash_eps) - coast_center_hint, (1.0 - self.squash_eps) + coast_center_hint),
            min=0.05,
        )

        center[:, 1] = regen_center_hint
        center[:, 2] = coast_center_hint
        scale[:, 0] = residual_scale.squeeze(1)
        scale[:, 1] = torch.minimum(regen_range_hint, regen_room)
        scale[:, 2] = torch.minimum(coast_range_hint, coast_room)
        return center, scale

    def _parameterize_action(self, raw_action, states):
        """
        作用：把无界动作映射到状态相关的动作语义空间。
        输入：raw_action: squash 前动作张量；states: 对应状态张量。
        输出：实际动作、动作中心、动作尺度与 tanh 后的中间值。
        """
        center, scale = self._action_parameterization(states)
        squashed = torch.tanh(raw_action)
        action = center + scale * squashed
        return action, center, scale, squashed

    def _inverse_squash_action(self, action, states=None):
        """
        作用：把已执行动作反推回参数化前空间，便于兼容旧格式轨迹。
        输入：action: 已执行动作张量；states: 对应状态张量。
        输出：对应的 squash 前动作张量。
        """
        if states is None:
            center = torch.zeros_like(action)
            scale = self.action_bound.expand_as(action)
        else:
            center, scale = self._action_parameterization(states)
        scaled = torch.clamp((action - center) / torch.clamp(scale, min=self.squash_eps), -1.0 + self.squash_eps, 1.0 - self.squash_eps)
        return 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))

    def _squashed_log_prob(self, mu, std, raw_action, states):
        """
        作用：计算状态相关动作参数化下高斯策略在给定原始动作上的对数概率。
        输入：mu/std: 原始高斯分布参数；raw_action: 参数化前动作样本；states: 对应状态。
        输出：按动作维度求和后的对数概率张量。
        """
        dist = torch.distributions.Normal(mu, std)
        _, _, scale, squashed = self._parameterize_action(raw_action, states)
        log_det = torch.log(torch.clamp(scale * (1.0 - squashed.pow(2)), min=self.squash_eps))
        return dist.log_prob(raw_action).sum(dim=1, keepdim=True) - log_det.sum(dim=1, keepdim=True)

    def take_action(self, state, deterministic=False, return_raw=False):
        """
        作用：为单个环境状态采样动作或返回确定性动作。
        输入：state: 当前状态；deterministic: 是否确定性；return_raw: 是否返回 squash 前动作。
        输出：动作列表，或动作与原始动作的二元组。
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu, sigma = self.actor(state_t, self.explore_decay)
            if deterministic:
                raw_action = mu
            else:
                raw_action = torch.distributions.Normal(mu, sigma).sample()
            action, _, _, _ = self._parameterize_action(raw_action, state_t)
        action_list = action.squeeze(0).cpu().numpy().astype(np.float32).tolist()
        if return_raw:
            raw_list = raw_action.squeeze(0).cpu().numpy().astype(np.float32).tolist()
            return action_list, raw_list
        return action_list

    def update(self, transition_dict):
        """
        作用：使用收集到的一段轨迹执行一次 PPO 参数更新。
        输入：transition_dict: 轨迹数据字典。
        输出：无。
        """
        states = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float32, device=self.device)
        raw_actions_src = transition_dict.get("raw_actions")
        if raw_actions_src is None:
            action_tensor = torch.tensor(
                np.array(transition_dict["actions"]),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, self.action_dim)
            raw_actions = self._inverse_squash_action(action_tensor, states=states)
        else:
            raw_actions = torch.tensor(
                np.array(raw_actions_src),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, self.action_dim)
        rewards = torch.tensor(np.array(transition_dict["rewards"]), dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float32, device=self.device)
        terminateds_src = transition_dict.get("terminateds", transition_dict.get("dones"))
        terminateds = torch.tensor(np.array(terminateds_src), dtype=torch.float32, device=self.device).view(-1, 1)
        constraint_costs_src = transition_dict.get("constraint_costs", {})
        constraint_lambdas_src = transition_dict.get("constraint_lambdas", {})

        with torch.no_grad():
            reward_values = self._denormalize_reward_value(self.critic(states))
            next_reward_values = self._denormalize_reward_value(self.critic(next_states))
            td_target = rewards + self.gamma * next_reward_values * (1 - terminateds)
            td_delta = td_target - reward_values
            reward_advantage = compute_advantage(self.gamma, self.lmbda, td_delta, terminateds)
            reward_returns = reward_advantage + reward_values
            if self.use_return_rms:
                self.reward_return_rms.update(reward_returns.detach().cpu().numpy())
            reward_returns = self._normalize_reward_return(reward_returns)
            cost_advantages = {}
            cost_returns = {}
            for name in self.constraint_names:
                cost_arr = np.array(
                    constraint_costs_src.get(name, np.zeros((states.size(0),), dtype=np.float32)),
                    dtype=np.float32,
                )
                cost_tensor = torch.tensor(cost_arr, dtype=torch.float32, device=self.device).view(-1, 1)
                critic = self.cost_critics[name]
                cost_td_target = cost_tensor + self.gamma * critic(next_states) * (1 - terminateds)
                cost_td_delta = cost_td_target - critic(states)
                cost_adv = compute_advantage(self.gamma, self.lmbda, cost_td_delta, terminateds)
                cost_returns[name] = cost_adv + critic(states)
                cost_advantages[name] = cost_adv
            combined_advantage = reward_advantage.clone()
            for name in self.constraint_names:
                lambda_val = max(0.0, float(constraint_lambdas_src.get(name, 0.0)))
                combined_advantage = combined_advantage - lambda_val * cost_advantages[name]
            combined_advantage = (combined_advantage - combined_advantage.mean()) / (combined_advantage.std() + 1e-8)
            old_mu, old_std = self.actor(states, self.explore_decay)
            old_log_probs = self._squashed_log_prob(old_mu, old_std, raw_actions, states)

        data_size = states.size(0)
        batch_size = min(self.minibatch_size, data_size)
        all_kl_vals = []

        for _ in range(self.epochs):
            perm = torch.randperm(data_size, device=self.device)
            kl_vals = []

            for start in range(0, data_size, batch_size):
                idx = perm[start : start + batch_size]
                b_states = states[idx]
                b_raw_actions = raw_actions[idx]
                b_adv = combined_advantage[idx]
                b_old_log_probs = old_log_probs[idx]
                b_reward_returns = reward_returns[idx]
                b_cost_returns = {name: cost_returns[name][idx] for name in self.constraint_names}

                mu, std = self.actor(b_states, self.explore_decay)
                dist = torch.distributions.Normal(mu, std)
                log_probs = self._squashed_log_prob(mu, std, b_raw_actions, b_states)
                entropy = dist.entropy().sum(dim=1, keepdim=True)

                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * b_adv

                actor_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy.mean())
                reward_critic_loss = F.mse_loss(self.critic(b_states), b_reward_returns.detach())
                cost_critic_loss = 0.0
                for name in self.constraint_names:
                    cost_critic_loss = cost_critic_loss + F.mse_loss(
                        self.cost_critics[name](b_states),
                        b_cost_returns[name].detach(),
                    )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                reward_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()

                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critics.parameters(), 1.0)
                self.cost_critic_optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - b_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    kl_vals.append(approx_kl)
                    all_kl_vals.append(approx_kl)

            if kl_vals and float(np.mean(kl_vals)) > self.target_kl:
                break

        mean_kl = float(np.mean(all_kl_vals)) if len(all_kl_vals) > 0 else 0.0
        self._adaptive_exploration_update(mean_kl)


def compute_advantage(gamma, lmbda, td_delta, terminateds):
    """
    作用：按终止标记计算 GAE 优势。
    输入：gamma/lmbda: GAE 参数；td_delta: TD 误差；terminateds: 终止标记。
    输出：advantage 张量。
    """
    advantage = torch.zeros_like(td_delta)
    gae = torch.zeros(1, device=td_delta.device, dtype=td_delta.dtype)
    for t in reversed(range(td_delta.shape[0])):
        gae = gamma * lmbda * gae * (1.0 - terminateds[t]) + td_delta[t]
        advantage[t] = gae
    return advantage
