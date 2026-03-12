import os
from collections import deque

import numpy as np

MPS_TO_KMH = 3.6


def build_random_road_profiles(
    horizon,
    max_speed,
    rng,
    slope_segments=(2, 4),
    curve_segments=(2, 5),
    grade_range=(-0.06, 0.08),
    slope_length_range=(18, 55),
    curve_width_range=(10, 30),
    curve_drop_range=(0.12, 0.28),
    min_curve_speed_ratio=0.54,
):
    """
    作用：生成训练或评估道路使用的随机坡度与弯道限速曲线。
    输入：horizon/max_speed/rng: 道路长度、限速与随机源；其余参数为段数范围。
    输出：slope 与 curve_speed_limit 两个数组。
    """
    slope = np.zeros(horizon + 1, dtype=np.float32)
    n_slope = int(rng.integers(slope_segments[0], slope_segments[1] + 1))
    for _ in range(n_slope):
        start = int(rng.integers(0, horizon - 20))
        length = int(rng.integers(slope_length_range[0], slope_length_range[1]))
        end = min(horizon, start + length)
        grade = float(rng.uniform(grade_range[0], grade_range[1]))
        slope[start:end] = grade
    slope = np.convolve(slope, np.ones(9, dtype=np.float32) / 9.0, mode="same").astype(np.float32)
    slope = np.clip(slope, min(-0.10, float(grade_range[0]) - 0.02), max(0.10, float(grade_range[1]) + 0.02))

    curve_speed_limit = np.full(horizon + 1, max_speed, dtype=np.float32)
    n_curve = int(rng.integers(curve_segments[0], curve_segments[1] + 1))
    min_curve_speed = float(min_curve_speed_ratio) * max_speed
    max_drop_low = float(curve_drop_range[0]) * max_speed
    max_drop_high = float(curve_drop_range[1]) * max_speed
    for _ in range(n_curve):
        center = int(rng.integers(20, horizon - 20))
        width = int(rng.integers(curve_width_range[0], curve_width_range[1]))
        max_drop = float(rng.uniform(max_drop_low, max_drop_high))
        idx = np.arange(horizon + 1)
        dip = max_drop * np.exp(-0.5 * ((idx - center) / max(width, 1)) ** 2)
        curve_speed_limit = np.minimum(curve_speed_limit, max_speed - np.array(dip, dtype=np.float32))
    curve_speed_limit = np.clip(curve_speed_limit, min_curve_speed, max_speed).astype(np.float32)
    return slope, curve_speed_limit


class StraightRoadScenario:
    """
    作用：封装简化道路、车辆动力学和驾驶员参考生成逻辑。
    输入：无。
    输出：StraightRoadScenario 类。
    """

    STYLE_TO_IDX = {"eco": 0, "normal": 1, "sport": 2}
    _MOTOR_MAP_CACHE = {}

    def __init__(
        self,
        horizon=220,
        dt=0.1,
        slope_profile=None,
        curve_speed_limit=None,
        scenario_name="road",
        driver_style="normal",
        motor_map_csv=None,
    ):
        """
        作用：初始化道路参数、车辆动力学和驾驶风格控制参数。
        输入：道路长度、步长、坡度/弯道曲线、驾驶风格、电机图等。
        输出：无。
        """
        self.horizon = horizon
        self.dt = dt
        self.scenario_name = scenario_name

        self.max_torque = 80.0
        self.max_speed = 80.0 / MPS_TO_KMH
        self.max_soc = 1.0
        self.min_soc = 0.2
        # 使用简单执行器动力学来模拟更真实的扭矩响应。
        self.max_torque_rate = 420.0  # 牛米/秒
        self.torque_time_const = 0.08  # 秒

        self.torque_to_acc = 0.042
        self.drag_linear = 0.018
        self.drag_quad = 0.0018

        self.base_soc_factor = 0.0009
        self.regen_eff = 0.62
        self.grade_acc_gain = 13.5
        self.curve_drag_gain = 0.12
        self.rpm_per_mps = 430.0
        self.max_rpm = self.max_speed * self.rpm_per_mps
        self.driver_style = driver_style if driver_style in self.STYLE_TO_IDX else "normal"
        self.driver_style_idx = int(self.STYLE_TO_IDX[self.driver_style])
        self.driver_style_onehot = np.eye(3, dtype=np.float32)[self.driver_style_idx]
        self.driver_pid_gains = {
            "eco": {"kp": 13.0, "ki": 1.8, "kd": 0.9, "ff": 0.92, "bias": -0.3},
            "normal": {"kp": 14.5, "ki": 2.2, "kd": 1.0, "ff": 1.00, "bias": 0.0},
            # 激进风格：使用更高的比例/微分增益和正偏置。
            "sport": {"kp": 18.6, "ki": 2.9, "kd": 1.5, "ff": 1.12, "bias": 0.8},
        }
        self.driver_pid_int_clip = 12.0
        self.driver_pid_deadband = 0.03
        self.driver_cmd_rate_limit = 260.0
        # 风格相关的闭环行为参数。
        self.driver_pid_behavior = {
            "eco": {"int_clip": 10.0, "deadband": 0.040, "cmd_rate_limit": 220.0},
            "normal": {"int_clip": 12.0, "deadband": 0.030, "cmd_rate_limit": 260.0},
            "sport": {"int_clip": 16.0, "deadband": 0.018, "cmd_rate_limit": 360.0},
        }
        self.driver_regen_gain_on_brake = {"eco": 0.95, "normal": 0.92, "sport": 0.86}
        self.motor_map_csv = motor_map_csv
        self.motor_map_loaded = False
        self.motor_map_source = "model"
        self.map_rpm_axis = None
        self.map_torque_axis = None
        self.map_eff_grid = None

        if slope_profile is None:
            self.slope_profile = np.zeros(self.horizon + 1, dtype=np.float32)
        else:
            self.slope_profile = np.array(slope_profile, dtype=np.float32)
        if curve_speed_limit is None:
            self.curve_speed_limit = np.full(self.horizon + 1, self.max_speed, dtype=np.float32)
        else:
            self.curve_speed_limit = np.array(curve_speed_limit, dtype=np.float32)
        self._maybe_load_motor_map()

    def _maybe_load_motor_map(self):
        """
        作用：在提供外部电机效率图时加载并缓存该图。
        输入：无。
        输出：无。
        """
        if not self.motor_map_csv:
            return
        csv_path = self.motor_map_csv
        if not os.path.exists(csv_path):
            return
        cache_key = os.path.abspath(csv_path)
        cached = self._MOTOR_MAP_CACHE.get(cache_key)
        if cached is None:
            loaded = self._load_motor_map_csv(csv_path)
            if loaded is None:
                return
            self._MOTOR_MAP_CACHE[cache_key] = loaded
            cached = loaded
        self.map_rpm_axis, self.map_torque_axis, self.map_eff_grid = cached
        self.motor_map_loaded = True
        self.motor_map_source = f"csv:{os.path.basename(csv_path)}"

    @staticmethod
    def _load_motor_map_csv(csv_path):
        """
        作用：把稀疏长表或 pivot 形式的电机效率 CSV 解析为稠密查表网格。
        输入：csv_path: CSV 文件路径。
        输出：rpm 轴、torque 轴和效率网格，或 None。
        """
        def build_grid(rpm, torque, eff):
            valid = np.isfinite(rpm) & np.isfinite(torque) & np.isfinite(eff)
            rpm = rpm[valid]
            torque = torque[valid]
            eff = eff[valid]
            if rpm.size < 6:
                return None

            if np.nanmax(np.abs(eff)) > 1.5:
                eff = eff / 100.0

            rpm_axis = np.unique(rpm)
            torque_axis = np.unique(torque)
            if rpm_axis.size < 3 or torque_axis.size < 3:
                return None

            grid = np.full((torque_axis.size, rpm_axis.size), np.nan, dtype=np.float32)
            counts = np.zeros_like(grid, dtype=np.float32)
            rpm_idx = np.searchsorted(rpm_axis, rpm)
            tq_idx = np.searchsorted(torque_axis, torque)
            for i in range(eff.size):
                r = rpm_idx[i]
                t = tq_idx[i]
                if 0 <= r < rpm_axis.size and 0 <= t < torque_axis.size:
                    grid[t, r] = np.nan_to_num(grid[t, r], nan=0.0) + eff[i]
                    counts[t, r] += 1.0
            mask = counts > 0
            grid[mask] = grid[mask] / counts[mask]
            if not np.any(mask):
                return None

            row_mean = np.nanmean(np.where(mask, grid, np.nan), axis=1)
            col_mean = np.nanmean(np.where(mask, grid, np.nan), axis=0)
            global_mean = float(np.nanmean(np.where(mask, grid, np.nan)))
            for ti in range(grid.shape[0]):
                for ri in range(grid.shape[1]):
                    if not mask[ti, ri]:
                        v = row_mean[ti]
                        if not np.isfinite(v):
                            v = col_mean[ri]
                        if not np.isfinite(v):
                            v = global_mean
                        grid[ti, ri] = v

            grid = np.clip(grid, 0.60, 0.99).astype(np.float32)
            return rpm_axis.astype(np.float32), torque_axis.astype(np.float32), grid

        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)
        except Exception:
            data = None
        if data is not None and getattr(data, "dtype", None) is not None and data.size != 0:
            names = [str(n).lower() for n in data.dtype.names]
            required = {"rpm", "torque", "efficiency"}
            if required.issubset(set(names)):
                rpm = np.array(data[data.dtype.names[names.index("rpm")]], dtype=np.float32).reshape(-1)
                torque = np.array(data[data.dtype.names[names.index("torque")]], dtype=np.float32).reshape(-1)
                eff = np.array(data[data.dtype.names[names.index("efficiency")]], dtype=np.float32).reshape(-1)
                built = build_grid(rpm, torque, eff)
                if built is not None:
                    return built

        try:
            raw = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32, filling_values=np.nan)
        except Exception:
            return None
        if raw is None or getattr(raw, "shape", None) is None or raw.ndim != 2:
            return None
        if raw.shape[0] < 4 or raw.shape[1] < 4:
            return None

        rpm_axis = np.array(raw[0, 1:], dtype=np.float32).reshape(-1)
        torque_axis = np.array(raw[1:, 0], dtype=np.float32).reshape(-1)
        grid = np.array(raw[1:, 1:], dtype=np.float32)
        if grid.shape != (torque_axis.size, rpm_axis.size):
            return None
        rpm_mesh = np.tile(rpm_axis.reshape(1, -1), (torque_axis.size, 1)).reshape(-1)
        torque_mesh = np.tile(torque_axis.reshape(-1, 1), (1, rpm_axis.size)).reshape(-1)
        eff = grid.reshape(-1)
        built = build_grid(rpm_mesh, torque_mesh, eff)
        if built is None:
            return None
        return built

    @staticmethod
    def _interp_bilinear(x_axis, y_axis, z_grid, x, y):
        """
        作用：在二维网格上执行双线性插值。
        输入：x_axis/y_axis/z_grid: 网格定义；x/y: 查询点。
        输出：插值后的标量结果。
        """
        x = float(np.clip(x, x_axis[0], x_axis[-1]))
        y = float(np.clip(y, y_axis[0], y_axis[-1]))
        ix = int(np.searchsorted(x_axis, x, side="right") - 1)
        iy = int(np.searchsorted(y_axis, y, side="right") - 1)
        ix = min(max(ix, 0), len(x_axis) - 2)
        iy = min(max(iy, 0), len(y_axis) - 2)
        x0 = float(x_axis[ix])
        x1 = float(x_axis[ix + 1])
        y0 = float(y_axis[iy])
        y1 = float(y_axis[iy + 1])
        tx = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 <= y0 else (y - y0) / (y1 - y0)
        z00 = float(z_grid[iy, ix])
        z10 = float(z_grid[iy, ix + 1])
        z01 = float(z_grid[iy + 1, ix])
        z11 = float(z_grid[iy + 1, ix + 1])
        z0 = z00 * (1.0 - tx) + z10 * tx
        z1 = z01 * (1.0 - tx) + z11 * tx
        return z0 * (1.0 - ty) + z1 * ty

    def efficiency(self, torque, rpm):
        """
        作用：从 CSV 电机图或内置解析模型中返回电机效率。
        输入：torque: 扭矩；rpm: 转速。
        输出：效率标量。
        """
        if self.motor_map_loaded and self.map_rpm_axis is not None and self.map_torque_axis is not None:
            tq_query = abs(float(torque))
            if self.map_torque_axis[0] < 0.0:
                tq_query = float(torque)
            eta_map = self._interp_bilinear(
                self.map_rpm_axis,
                self.map_torque_axis,
                self.map_eff_grid,
                abs(float(rpm)),
                tq_query,
            )
            return float(np.clip(eta_map, 0.60, 0.99))
        # 使用简化二维电机图，在中等转速和扭矩区域设置高效率岛。
        t = min(abs(torque) / max(self.max_torque, 1e-8), 1.5)
        s = min(abs(rpm) / max(self.max_rpm, 1e-8), 1.5)
        eta = 0.82
        eta += 0.13 * np.exp(-((t - 0.45) ** 2) / 0.05 - ((s - 0.42) ** 2) / 0.08)
        eta += 0.04 * np.exp(-((t - 0.75) ** 2) / 0.04 - ((s - 0.70) ** 2) / 0.05)
        eta -= 0.06 * (s**1.7)
        eta -= 0.04 * ((t - 0.55) ** 2)
        return float(np.clip(eta, 0.72, 0.97))

    def reset_vehicle(self):
        """
        作用：构造 episode 起点使用的标准车辆初始状态。
        输入：无。
        输出：包含速度、SOC、转速、扭矩、距离的状态字典。
        """
        speed = 10.0
        return {
            "speed": float(speed),
            "soc": float(self.max_soc),
            "rpm": float(speed * self.rpm_per_mps),
            "torque": 0.0,
            "distance": 0.0,
        }

    def road_feature(self, step_idx):
        """
        作用：返回指定时间步的坡度和弯道限速。
        输入：step_idx: 仿真步索引。
        输出：grade 与 curve_v_lim。
        """
        idx = min(max(int(step_idx), 0), self.horizon)
        grade = float(self.slope_profile[idx])
        curve_v_lim = float(self.curve_speed_limit[idx])
        return grade, curve_v_lim

    def step_vehicle(self, vehicle, torque_cmd, regen_gain=1.0, step_idx=0):
        """
        作用：推进一步车辆动力学并计算能耗与能量回收。
        输入：vehicle: 当前车辆状态；torque_cmd: 扭矩命令；regen_gain: 回收增益；step_idx: 步索引。
        输出：下一车辆状态、能耗、回收量和位移增量。
        """
        target_torque = float(np.clip(torque_cmd, -self.max_torque, self.max_torque))
        prev_torque = float(vehicle["torque"])
        max_delta_torque = self.max_torque_rate * self.dt
        rate_limited_torque = prev_torque + float(np.clip(target_torque - prev_torque, -max_delta_torque, max_delta_torque))
        alpha = self.dt / (self.torque_time_const + self.dt)
        torque = float(np.clip(prev_torque + alpha * (rate_limited_torque - prev_torque), -self.max_torque, self.max_torque))
        speed = float(vehicle["speed"])
        soc = float(vehicle["soc"])
        grade, curve_v_lim = self.road_feature(step_idx)
        overspeed_curve = max(0.0, speed - curve_v_lim)

        acc = (
            torque * self.torque_to_acc
            - self.drag_linear * speed
            - self.drag_quad * speed * speed
            - self.grade_acc_gain * grade
            - self.curve_drag_gain * overspeed_curve
        )
        speed = float(np.clip(speed + acc * self.dt, 0.0, self.max_speed))
        rpm = speed * self.rpm_per_mps
        dist_step = speed * self.dt

        eff = self.efficiency(torque, rpm)
        mech_power = torque * rpm / 9550.0
        if mech_power >= 0:
            batt_power = mech_power / max(eff, 1e-4)
        else:
            eff_regen = float(np.clip(self.regen_eff * regen_gain * (0.92 + 0.08 * eff), 0.20, 0.95))
            batt_power = mech_power * eff_regen

        soc_delta = batt_power * self.dt * self.base_soc_factor
        soc_prev = soc
        soc = float(np.clip(soc - soc_delta, 0.0, self.max_soc))
        actual_soc_delta = soc_prev - soc

        energy_consume = max(actual_soc_delta, 0.0)
        energy_recover = max(-actual_soc_delta, 0.0)

        next_vehicle = {
            "speed": speed,
            "soc": soc,
            "rpm": rpm,
            "torque": torque,
            "distance": vehicle["distance"] + dist_step,
        }
        return next_vehicle, energy_consume, energy_recover, dist_step

    def build_driver_torque_profile(self, style_name=None, rng=None):
        """
        作用：为给定驾驶风格生成平滑的开环扭矩轨迹。
        输入：style_name: 驾驶风格；rng: 随机源。
        输出：扭矩轨迹数组。
        """
        # 生成多风格、近似人工驾驶的开环扭矩曲线。
        style = style_name if style_name in self.STYLE_TO_IDX else self.driver_style
        t = np.arange(self.horizon, dtype=np.float32)
        knot_t = np.array([0, 12, 28, 52, 88, 120, 150, 176, 198, self.horizon - 1], dtype=np.float32)
        if style == "eco":
            knot_u = np.array([0.16, 0.72, 0.55, 0.30, -0.18, 0.01, 0.46, 0.22, 0.10, 0.06], dtype=np.float32)
        elif style == "sport":
            knot_u = np.array([0.34, 1.06, 0.88, 0.50, -0.36, 0.06, 0.78, 0.42, 0.20, 0.12], dtype=np.float32)
        else:
            knot_u = np.array([0.20, 0.88, 0.70, 0.36, -0.24, 0.02, 0.60, 0.28, 0.12, 0.08], dtype=np.float32)
        cmd = np.interp(t, knot_t, knot_u).astype(np.float32) * self.max_torque
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel = kernel / np.sum(kernel)
        cmd = np.convolve(cmd, kernel, mode="same").astype(np.float32)
        if rng is not None:
            amp_sigma = 0.02 if style == "sport" else 0.01
            amp_lo = 0.95 if style == "sport" else 0.97
            amp_hi = 1.06 if style == "sport" else 1.03
            amp = float(np.clip(rng.normal(1.0, amp_sigma), amp_lo, amp_hi))
            cmd = amp * cmd
        return np.clip(cmd, -self.max_torque, self.max_torque).astype(np.float32)

    def init_driver_pid_state(self, style_name=None):
        """
        作用：初始化闭环驾驶员控制器内部状态。
        输入：style_name: 驾驶风格。
        输出：PID 状态字典。
        """
        style = style_name if style_name in self.STYLE_TO_IDX else self.driver_style
        return {
            "style": style,
            "err_int": 0.0,
            "err_prev": 0.0,
            "cmd_prev": 0.0,
        }

    def driver_pid_command(self, pid_state, target_speed, current_speed, step_idx=0):
        """
        作用：根据目标速度和当前速度计算驾驶员扭矩命令。
        输入：pid_state: 驾驶员控制器状态；target_speed/current_speed: 目标与当前速度；step_idx: 步索引。
        输出：当前扭矩命令和更新后的 PID 状态。
        """
        style = str(pid_state.get("style", self.driver_style))
        gains = self.driver_pid_gains.get(style, self.driver_pid_gains["normal"])
        behavior = self.driver_pid_behavior.get(style, self.driver_pid_behavior["normal"])
        deadband = float(behavior["deadband"])
        int_clip = float(behavior["int_clip"])
        cmd_rate_limit = float(behavior["cmd_rate_limit"])
        speed_err = float(target_speed - current_speed)
        if abs(speed_err) < deadband:
            speed_err = 0.0
        err_int = float(
            np.clip(
                float(pid_state.get("err_int", 0.0)) + speed_err * self.dt,
                -int_clip,
                int_clip,
            )
        )
        err_prev = float(pid_state.get("err_prev", 0.0))
        err_der = (speed_err - err_prev) / max(self.dt, 1e-8)
        grade, _ = self.road_feature(step_idx)
        feedforward_acc = (
            self.drag_linear * target_speed
            + self.drag_quad * target_speed * target_speed
            + self.grade_acc_gain * grade
        )
        feedforward_torque = gains["ff"] * feedforward_acc / max(self.torque_to_acc, 1e-8)
        cmd_raw = (
            gains["kp"] * speed_err
            + gains["ki"] * err_int
            + gains["kd"] * err_der
            + feedforward_torque
            + gains["bias"]
        )
        cmd_prev = float(pid_state.get("cmd_prev", 0.0))
        max_delta = cmd_rate_limit * self.dt
        cmd_rate_limited = cmd_prev + float(np.clip(cmd_raw - cmd_prev, -max_delta, max_delta))
        cmd = float(np.clip(cmd_rate_limited, -self.max_torque, self.max_torque))
        next_state = {
            "style": style,
            "err_int": err_int,
            "err_prev": speed_err,
            "cmd_prev": cmd,
        }
        return cmd, next_state

    def build_driver_speed_profile(self, style_name=None, rng=None):
        """
        作用：先用开环轨迹预热，再构造带风格特性的速度规划。
        输入：style_name: 驾驶风格；rng: 随机源。
        输出：目标速度曲线数组。
        """
        style = style_name if style_name in self.STYLE_TO_IDX else self.driver_style
        seed_torque = self.build_driver_torque_profile(style_name=style, rng=rng)
        warmup_ref = self.simulate_open_loop(seed_torque)
        speed_target = np.array(warmup_ref["speed"], dtype=np.float32)
        style_speed_scale = {"eco": 0.97, "normal": 1.00, "sport": 1.08}
        speed_target *= float(style_speed_scale.get(style, 1.0))
        curve_cap_ratio = {"eco": 0.93, "normal": 0.96, "sport": 0.995}
        speed_target = np.minimum(speed_target, self.curve_speed_limit * float(curve_cap_ratio.get(style, 0.96)))
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel = kernel / np.sum(kernel)
        speed_target = np.convolve(speed_target, kernel, mode="same").astype(np.float32)
        speed_target[0] = float(warmup_ref["speed"][0])
        if rng is not None:
            jitter_std = 0.20 if style == "sport" else 0.12
            jitter = np.array(rng.normal(0.0, jitter_std, size=self.horizon + 1), dtype=np.float32)
            speed_target = speed_target + jitter
        speed_target = np.minimum(speed_target, self.curve_speed_limit * float(curve_cap_ratio.get(style, 0.96)))
        return np.clip(speed_target, 0.0, self.max_speed).astype(np.float32)

    def simulate_open_loop(self, torque_profile):
        """
        作用：在固定扭矩命令序列下滚动仿真车辆。
        输入：torque_profile: 扭矩序列。
        输出：包含轨迹、能耗和距离统计的字典。
        """
        vehicle = self.reset_vehicle()
        speed = [vehicle["speed"]]
        soc = [vehicle["soc"]]
        torque = [vehicle["torque"]]
        torque_cmd = [float(torque_profile[0])]
        distance = [vehicle["distance"]]
        step_energy = []
        step_recover = []
        step_distance = []
        slope = [float(self.slope_profile[0])]
        curve_vlim = [float(self.curve_speed_limit[0])]

        for t in range(self.horizon):
            cmd_t = float(torque_profile[t])
            vehicle, e_step, r_step, d_step = self.step_vehicle(vehicle, cmd_t, regen_gain=1.0, step_idx=t)
            speed.append(vehicle["speed"])
            soc.append(vehicle["soc"])
            torque.append(vehicle["torque"])
            torque_cmd.append(float(torque_profile[min(t + 1, self.horizon - 1)]))
            distance.append(vehicle["distance"])
            step_energy.append(e_step)
            step_recover.append(r_step)
            step_distance.append(d_step)
            slope.append(float(self.slope_profile[min(t + 1, self.horizon)]))
            curve_vlim.append(float(self.curve_speed_limit[min(t + 1, self.horizon)]))

        total_energy = float(np.sum(step_energy))
        total_recover = float(np.sum(step_recover))
        net_energy = float(total_energy - total_recover)
        total_distance = float(np.sum(step_distance))

        return {
            "speed": np.array(speed, dtype=np.float32),
            "soc": np.array(soc, dtype=np.float32),
            "torque": np.array(torque, dtype=np.float32),
            "torque_cmd": np.array(torque_cmd, dtype=np.float32),
            "distance": np.array(distance, dtype=np.float32),
            "step_energy": np.array(step_energy, dtype=np.float32),
            "step_recover": np.array(step_recover, dtype=np.float32),
            "step_distance": np.array(step_distance, dtype=np.float32),
            "slope": np.array(slope, dtype=np.float32),
            "curve_speed_limit": np.array(curve_vlim, dtype=np.float32),
            "total_energy": total_energy,
            "total_recover": total_recover,
            "net_energy": net_energy,
            "total_distance": total_distance,
            "energy_per_dist": total_energy / max(total_distance, 1e-8),
            "net_energy_per_dist": net_energy / max(total_distance, 1e-8),
        }

    def simulate_closed_loop_driver(self, speed_target_profile, style_name=None):
        """
        作用：让风格化 PID 驾驶员跟随速度规划并完成闭环仿真。
        输入：speed_target_profile: 目标速度曲线；style_name: 驾驶风格。
        输出：包含闭环轨迹与能耗统计的字典。
        """
        style = style_name if style_name in self.STYLE_TO_IDX else self.driver_style
        vehicle = self.reset_vehicle()
        pid_state = self.init_driver_pid_state(style_name=style)
        speed = [vehicle["speed"]]
        soc = [vehicle["soc"]]
        torque = [vehicle["torque"]]
        torque_cmd = []
        distance = [vehicle["distance"]]
        step_energy = []
        step_recover = []
        step_distance = []
        slope = [float(self.slope_profile[0])]
        curve_vlim = [float(self.curve_speed_limit[0])]
        speed_target = [float(speed_target_profile[0])]

        for t in range(self.horizon):
            target_speed_t = float(speed_target_profile[t])
            driver_cmd_t, pid_state = self.driver_pid_command(
                pid_state,
                target_speed_t,
                vehicle["speed"],
                step_idx=t,
            )
            torque_cmd.append(driver_cmd_t)
            regen_gain_on_brake = float(self.driver_regen_gain_on_brake.get(style, 0.92))
            regen_gain = 1.0 if driver_cmd_t >= 0.0 else regen_gain_on_brake
            vehicle, e_step, r_step, d_step = self.step_vehicle(
                vehicle, driver_cmd_t, regen_gain=regen_gain, step_idx=t
            )
            speed.append(vehicle["speed"])
            soc.append(vehicle["soc"])
            torque.append(vehicle["torque"])
            distance.append(vehicle["distance"])
            step_energy.append(e_step)
            step_recover.append(r_step)
            step_distance.append(d_step)
            slope.append(float(self.slope_profile[min(t + 1, self.horizon)]))
            curve_vlim.append(float(self.curve_speed_limit[min(t + 1, self.horizon)]))
            speed_target.append(float(speed_target_profile[min(t + 1, self.horizon)]))

        if len(torque_cmd) == 0:
            torque_cmd = [0.0]
        torque_cmd_full = np.concatenate(
            [
                np.array(torque_cmd, dtype=np.float32),
                np.array([torque_cmd[-1]], dtype=np.float32),
            ],
            axis=0,
        )

        total_energy = float(np.sum(step_energy))
        total_recover = float(np.sum(step_recover))
        net_energy = float(total_energy - total_recover)
        total_distance = float(np.sum(step_distance))

        return {
            "speed": np.array(speed, dtype=np.float32),
            "speed_target": np.array(speed_target, dtype=np.float32),
            "soc": np.array(soc, dtype=np.float32),
            "torque": np.array(torque, dtype=np.float32),
            "torque_cmd": torque_cmd_full.astype(np.float32),
            "distance": np.array(distance, dtype=np.float32),
            "step_energy": np.array(step_energy, dtype=np.float32),
            "step_recover": np.array(step_recover, dtype=np.float32),
            "step_distance": np.array(step_distance, dtype=np.float32),
            "slope": np.array(slope, dtype=np.float32),
            "curve_speed_limit": np.array(curve_vlim, dtype=np.float32),
            "total_energy": total_energy,
            "total_recover": total_recover,
            "net_energy": net_energy,
            "total_distance": total_distance,
            "energy_per_dist": total_energy / max(total_distance, 1e-8),
            "net_energy_per_dist": net_energy / max(total_distance, 1e-8),
        }

    def simulate_driver_reference(self, style_name=None, rng=None):
        """
        作用：生成 RL 环境使用的最终驾驶员参考轨迹。
        输入：style_name: 驾驶风格；rng: 随机源。
        输出：驾驶员参考字典。
        """
        style = style_name if style_name in self.STYLE_TO_IDX else self.driver_style
        speed_target = self.build_driver_speed_profile(style_name=style, rng=rng)
        return self.simulate_closed_loop_driver(speed_target, style_name=style)


class DriverReferenceEnergyEnv:
    """
    作用：定义一个要求智能体在贴近驾驶员参考的同时节能的环境。
    输入：无。
    输出：DriverReferenceEnergyEnv 类。
    """

    STYLE_CONTROL_BASE = {
        "driver_follow_coef": 0.16,
        "energy_lag_gate_gain": 0.62,
        "speed_penalty_deadband_mps": 0.05,
        "coast_torque_coef": 0.14,
        "catchup_pos_boost": 1.20,
        "catchup_residual_boost": 1.06,
        "micro_residual_lag_boost": 1.12,
        "dist_catchup_torque_gain": 0.012,
        "dist_catchup_torque_clip": 0.45,
        "dist_catchup_deadband_m": 0.16,
        "dist_catchup_progress_start": 0.20,
        "launch_catchup_torque_gain": 0.009,
        "launch_catchup_torque_clip": 0.26,
        "launch_progress_end": 0.10,
        "launch_speed_deadband_mps": 0.06,
        "launch_dist_deadband_m": 0.05,
        "launch_negative_torque_deadband_ratio": 0.035,
        "launch_negative_torque_penalty_coef": 0.10,
        "launch_negative_torque_constraint_scale": 0.70,
        "launch_negative_torque_streak_gain": 0.28,
        "launch_negative_torque_phase_scale": 1.35,
        "bias_torque_clip": 0.9,
        "bias_neutral_torque_gain": 0.0,
        "bias_neutral_torque_clip": 0.0,
        "neg_bias_penalty_coef": 0.02,
        "isochronous_penalty_coef": 0.03,
        "isochronous_cost_scale": 0.12,
        "isochronous_debt_speed_gain": 0.24,
        "isochronous_debt_dist_gain": 0.10,
        "isochronous_release_speed_gain": 0.62,
        "isochronous_release_dist_gain": 0.24,
        "isochronous_debt_clip": 4.0,
        "isochronous_speed_release_deadband_mps": 0.05,
        "isochronous_dist_release_deadband_m": 0.35,
        "final_isochronous_penalty_coef": 0.0,
        "slow_bias_drag_debit_coef": 0.40,
        "slow_bias_kinetic_debit_coef": 0.08,
        "slow_bias_debit_progress_start": 0.08,
        "slow_bias_debit_clip": 0.0014,
        "slow_bias_speed_deadband_mps": 0.15,
        "slow_bias_speed_deadband_ratio": 0.010,
        "slow_bias_speed_deadband_min_mps": 0.035,
        "slow_bias_early_tighten_progress": 0.16,
        "slow_bias_early_deadband_scale": 0.30,
        "slow_bias_early_gate_floor": 0.55,
        "slow_bias_discount_rate": 0.50,
        "coast_bonus_coef": 0.10,
        "efficiency_direct_reward_coef": 0.12,
        "energy_positive_reward_coef": 0.95,
        "energy_negative_penalty_coef": 1.30,
        "regen_recover_coef": 0.0,
        "regen_target_base": 0.86,
        "regen_target_gain": 0.12,
        "regen_target_tol": 0.52,
        "constraint_underspeed_cost_target": 0.12,
        "constraint_underspeed_rel_speed_tol": 0.0080,
        "constraint_underspeed_rel_dist_tol": 0.0080,
        "constraint_window_energy_cost_target": 0.08,
        "window_energy_horizon": 12,
        "window_negative_saving_tol": 0.0015,
        "window_negative_gate_progress": 0.08,
        "window_negative_streak_gain": 0.35,
        "segment_tracking_horizon": 12,
        "segment_tracking_gate_progress": 0.06,
        "segment_speed_abs_tol_ratio": 0.65,
        "segment_dist_abs_tol_ratio": 0.60,
        "segment_speed_bias_tol_ratio": 0.28,
        "segment_dist_bias_tol_ratio": 0.22,
        "segment_tracking_constraint_scale": 0.72,
        "segment_tracking_streak_gain": 0.28,
        "window_positive_reward_coef": 0.018,
        "window_negative_reward_penalty_coef": 0.06,
        "window_constraint_scale": 1.20,
        "window_positive_saving_cap": 0.015,
        "window_oversave_penalty_coef": 0.03,
        "window_oversave_gate_end": 0.55,
        "window_late_positive_bonus_coef": 0.02,
        "phase_accel_torque_threshold_ratio": 0.18,
        "residual_phase_launch_limit_ratio": 0.80,
        "residual_phase_accel_limit_ratio": 0.78,
        "residual_phase_cruise_limit_ratio": 0.90,
        "residual_phase_decel_limit_ratio": 0.72,
        "efficiency_cruise_reward_scale": 1.12,
        "efficiency_accel_reward_scale": 0.92,
        "efficiency_decel_reward_scale": 0.95,
        "efficiency_load_gate_floor_ratio": 0.12,
    }
    STYLE_CONTROL_PROFILES = {
        "sport": {
            "driver_follow_coef": 0.18,
            "energy_lag_gate_gain": 0.78,
            "speed_penalty_deadband_mps": 0.04,
            "coast_torque_coef": 0.13,
            "catchup_pos_boost": 1.22,
            "catchup_residual_boost": 1.08,
            "micro_residual_lag_boost": 1.16,
            "dist_catchup_torque_gain": 0.010,
            "dist_catchup_torque_clip": 0.55,
            "launch_catchup_torque_gain": 0.012,
            "launch_catchup_torque_clip": 0.34,
            "launch_progress_end": 0.11,
            "launch_speed_deadband_mps": 0.05,
            "launch_dist_deadband_m": 0.04,
            "launch_negative_torque_deadband_ratio": 0.030,
            "launch_negative_torque_penalty_coef": 0.11,
            "launch_negative_torque_constraint_scale": 0.75,
            "launch_negative_torque_streak_gain": 0.30,
            "launch_negative_torque_phase_scale": 1.40,
            "bias_torque_clip": 1.00,
            "bias_neutral_torque_gain": 0.022,
            "bias_neutral_torque_clip": 0.65,
            "neg_bias_penalty_coef": 0.03,
            "isochronous_penalty_coef": 0.04,
            "isochronous_cost_scale": 0.14,
            "final_isochronous_penalty_coef": 0.0,
            "slow_bias_drag_debit_coef": 0.38,
            "slow_bias_kinetic_debit_coef": 0.08,
            "slow_bias_debit_clip": 0.0013,
            "slow_bias_speed_deadband_mps": 0.12,
            "slow_bias_speed_deadband_ratio": 0.009,
            "slow_bias_speed_deadband_min_mps": 0.030,
            "slow_bias_early_tighten_progress": 0.18,
            "slow_bias_early_deadband_scale": 0.24,
            "slow_bias_early_gate_floor": 0.62,
            "slow_bias_discount_rate": 0.55,
            "coast_bonus_coef": 0.08,
            "efficiency_direct_reward_coef": 0.10,
            "energy_positive_reward_coef": 0.92,
            "energy_negative_penalty_coef": 1.38,
            "regen_recover_coef": 0.065,
            "regen_target_base": 0.92,
            "regen_target_gain": 0.14,
            "regen_target_tol": 0.42,
            "constraint_underspeed_cost_target": 0.13,
            "constraint_underspeed_rel_speed_tol": 0.0085,
            "constraint_underspeed_rel_dist_tol": 0.0085,
            "constraint_window_energy_cost_target": 0.10,
            "window_energy_horizon": 10,
            "window_negative_saving_tol": 0.0010,
            "window_negative_gate_progress": 0.08,
            "window_negative_streak_gain": 0.38,
            "segment_tracking_horizon": 10,
            "segment_tracking_gate_progress": 0.05,
            "segment_speed_abs_tol_ratio": 0.62,
            "segment_dist_abs_tol_ratio": 0.56,
            "segment_speed_bias_tol_ratio": 0.26,
            "segment_dist_bias_tol_ratio": 0.20,
            "segment_tracking_constraint_scale": 0.76,
            "segment_tracking_streak_gain": 0.30,
            "window_positive_reward_coef": 0.014,
            "window_negative_reward_penalty_coef": 0.06,
            "window_constraint_scale": 1.28,
            "window_positive_saving_cap": 0.012,
            "window_oversave_penalty_coef": 0.04,
            "window_oversave_gate_end": 0.58,
            "window_late_positive_bonus_coef": 0.025,
            "phase_accel_torque_threshold_ratio": 0.20,
            "residual_phase_launch_limit_ratio": 0.84,
            "residual_phase_accel_limit_ratio": 0.82,
            "residual_phase_cruise_limit_ratio": 0.88,
            "residual_phase_decel_limit_ratio": 0.74,
            "efficiency_cruise_reward_scale": 1.10,
            "efficiency_accel_reward_scale": 0.96,
            "efficiency_decel_reward_scale": 1.00,
            "efficiency_load_gate_floor_ratio": 0.14,
        },
        "normal": {
            "dist_catchup_torque_gain": 0.015,
            "dist_catchup_torque_clip": 0.62,
            "launch_catchup_torque_gain": 0.010,
            "launch_catchup_torque_clip": 0.30,
            "launch_progress_end": 0.10,
            "launch_speed_deadband_mps": 0.06,
            "launch_dist_deadband_m": 0.05,
            "launch_negative_torque_deadband_ratio": 0.032,
            "launch_negative_torque_penalty_coef": 0.12,
            "launch_negative_torque_constraint_scale": 0.82,
            "launch_negative_torque_streak_gain": 0.34,
            "launch_negative_torque_phase_scale": 1.45,
            "bias_neutral_torque_gain": 0.012,
            "bias_neutral_torque_clip": 0.35,
            "neg_bias_penalty_coef": 0.025,
            "isochronous_penalty_coef": 0.05,
            "isochronous_cost_scale": 0.16,
            "final_isochronous_penalty_coef": 0.0,
            "slow_bias_drag_debit_coef": 0.45,
            "slow_bias_kinetic_debit_coef": 0.10,
            "slow_bias_debit_progress_start": 0.05,
            "slow_bias_debit_clip": 0.0016,
            "slow_bias_speed_deadband_mps": 0.15,
            "slow_bias_speed_deadband_ratio": 0.010,
            "slow_bias_speed_deadband_min_mps": 0.035,
            "slow_bias_early_tighten_progress": 0.16,
            "slow_bias_early_deadband_scale": 0.26,
            "slow_bias_early_gate_floor": 0.60,
            "slow_bias_discount_rate": 0.50,
            "coast_bonus_coef": 0.15,
            "efficiency_direct_reward_coef": 0.14,
            "energy_positive_reward_coef": 0.95,
            "energy_negative_penalty_coef": 1.34,
            "regen_recover_coef": 0.035,
            "regen_target_base": 0.90,
            "regen_target_gain": 0.13,
            "regen_target_tol": 0.46,
            "constraint_underspeed_cost_target": 0.10,
            "constraint_underspeed_rel_speed_tol": 0.0075,
            "constraint_underspeed_rel_dist_tol": 0.0075,
            "constraint_window_energy_cost_target": 0.04,
            "window_energy_horizon": 12,
            "window_negative_saving_tol": 0.0005,
            "window_negative_gate_progress": 0.04,
            "window_negative_streak_gain": 0.46,
            "segment_tracking_horizon": 12,
            "segment_tracking_gate_progress": 0.04,
            "segment_speed_abs_tol_ratio": 0.60,
            "segment_dist_abs_tol_ratio": 0.54,
            "segment_speed_bias_tol_ratio": 0.24,
            "segment_dist_bias_tol_ratio": 0.18,
            "segment_tracking_constraint_scale": 0.88,
            "segment_tracking_streak_gain": 0.34,
            "window_positive_reward_coef": 0.014,
            "window_negative_reward_penalty_coef": 0.07,
            "window_constraint_scale": 1.45,
            "window_positive_saving_cap": 0.012,
            "window_oversave_penalty_coef": 0.04,
            "window_oversave_gate_end": 0.55,
            "window_late_positive_bonus_coef": 0.025,
            "phase_accel_torque_threshold_ratio": 0.17,
            "residual_phase_launch_limit_ratio": 0.82,
            "residual_phase_accel_limit_ratio": 0.80,
            "residual_phase_cruise_limit_ratio": 0.90,
            "residual_phase_decel_limit_ratio": 0.72,
            "efficiency_cruise_reward_scale": 1.14,
            "efficiency_accel_reward_scale": 0.94,
            "efficiency_decel_reward_scale": 0.96,
            "efficiency_load_gate_floor_ratio": 0.12,
        },
        "eco": {
            "driver_follow_coef": 0.15,
            "energy_lag_gate_gain": 0.56,
            "speed_penalty_deadband_mps": 0.06,
            "coast_torque_coef": 0.18,
            "catchup_pos_boost": 1.18,
            "catchup_residual_boost": 1.05,
            "micro_residual_lag_boost": 1.10,
            "dist_catchup_torque_gain": 0.013,
            "dist_catchup_torque_clip": 0.52,
            "launch_catchup_torque_gain": 0.008,
            "launch_catchup_torque_clip": 0.22,
            "launch_progress_end": 0.09,
            "launch_speed_deadband_mps": 0.06,
            "launch_dist_deadband_m": 0.05,
            "launch_negative_torque_deadband_ratio": 0.038,
            "launch_negative_torque_penalty_coef": 0.09,
            "launch_negative_torque_constraint_scale": 0.65,
            "launch_negative_torque_streak_gain": 0.26,
            "launch_negative_torque_phase_scale": 1.30,
            "bias_torque_clip": 0.95,
            "bias_neutral_torque_gain": 0.015,
            "bias_neutral_torque_clip": 0.40,
            "neg_bias_penalty_coef": 0.022,
            "isochronous_penalty_coef": 0.035,
            "isochronous_cost_scale": 0.13,
            "final_isochronous_penalty_coef": 0.0,
            "slow_bias_drag_debit_coef": 0.36,
            "slow_bias_kinetic_debit_coef": 0.09,
            "slow_bias_debit_progress_start": 0.06,
            "slow_bias_debit_clip": 0.0014,
            "slow_bias_speed_deadband_mps": 0.18,
            "slow_bias_speed_deadband_ratio": 0.012,
            "slow_bias_speed_deadband_min_mps": 0.040,
            "slow_bias_early_tighten_progress": 0.14,
            "slow_bias_early_deadband_scale": 0.34,
            "slow_bias_early_gate_floor": 0.50,
            "slow_bias_discount_rate": 0.45,
            "coast_bonus_coef": 0.18,
            "efficiency_direct_reward_coef": 0.16,
            "energy_positive_reward_coef": 0.98,
            "energy_negative_penalty_coef": 1.24,
            "regen_recover_coef": 0.050,
            "regen_target_base": 0.96,
            "regen_target_gain": 0.11,
            "regen_target_tol": 0.44,
            "constraint_underspeed_cost_target": 0.14,
            "constraint_underspeed_rel_speed_tol": 0.0090,
            "constraint_underspeed_rel_dist_tol": 0.0090,
            "constraint_window_energy_cost_target": 0.09,
            "window_energy_horizon": 14,
            "window_negative_saving_tol": 0.0018,
            "window_negative_gate_progress": 0.10,
            "window_negative_streak_gain": 0.30,
            "segment_tracking_horizon": 14,
            "segment_tracking_gate_progress": 0.08,
            "segment_speed_abs_tol_ratio": 0.68,
            "segment_dist_abs_tol_ratio": 0.62,
            "segment_speed_bias_tol_ratio": 0.30,
            "segment_dist_bias_tol_ratio": 0.24,
            "segment_tracking_constraint_scale": 0.68,
            "segment_tracking_streak_gain": 0.24,
            "window_positive_reward_coef": 0.018,
            "window_negative_reward_penalty_coef": 0.05,
            "window_constraint_scale": 1.18,
            "window_positive_saving_cap": 0.014,
            "window_oversave_penalty_coef": 0.025,
            "window_oversave_gate_end": 0.52,
            "window_late_positive_bonus_coef": 0.02,
            "phase_accel_torque_threshold_ratio": 0.16,
            "residual_phase_launch_limit_ratio": 0.80,
            "residual_phase_accel_limit_ratio": 0.78,
            "residual_phase_cruise_limit_ratio": 0.92,
            "residual_phase_decel_limit_ratio": 0.74,
            "efficiency_cruise_reward_scale": 1.16,
            "efficiency_accel_reward_scale": 0.90,
            "efficiency_decel_reward_scale": 0.98,
            "efficiency_load_gate_floor_ratio": 0.10,
        },
    }

    def __init__(self, scenario, reference, residual_limit=10.0, obs_stack=1):
        """
        作用：绑定一个场景与参考轨迹，并初始化奖励和控制相关超参数。
        输入：scenario/reference: 场景与驾驶员参考；residual_limit: residual 动作边界。
        输出：无。
        """
        self.scenario = scenario
        self.reference = reference
        self.horizon = scenario.horizon
        self.obs_stack = max(1, int(obs_stack))

        self.residual_limit = residual_limit
        self.max_torque = scenario.max_torque
        self.max_speed = scenario.max_speed
        self.max_soc = scenario.max_soc

        self.speed_soft_tol = 0.8
        self.dist_soft_tol = 3.0
        self.speed_fail_tol = 3.0
        self.dist_fail_tol = 10.0
        self.constraint_speed_tol = 0.6
        self.constraint_dist_tol = 3.6
        self.mode = "track"
        self.energy_weight = 0.0
        self.lambda_speed = 0.0
        self.lambda_dist = 0.0
        self.lambda_smooth = 0.0
        self.lambda_net_energy = 0.0
        self.lambda_projection = 0.0
        self.lambda_underspeed = 0.0
        self.lambda_window_energy = 0.0
        self.recover_margin_mid = 0.90
        self.recover_margin_high = 1.15
        self.recover_residual_scale_mid = 0.50
        self.recover_residual_scale_high = 0.22
        self.torque_eff_coef = 0.10
        self.regen_eff_coef = 0.04
        self.efficiency_direct_reward_coef = 0.12
        self.energy_positive_reward_coef = 0.95
        self.energy_negative_penalty_coef = 1.30
        self.regen_recover_coef = 0.0
        self.regen_target_base = 0.86
        self.regen_target_gain = 0.12
        self.regen_target_tol = 0.52
        self.coast_torque_coef = 0.14
        # 驾驶辅助式微调：以驾驶员命令为主，智能体只做小范围修正。
        self.micro_residual_base = 2.2
        self.micro_residual_ratio = 0.10
        self.micro_residual_max = 6.0
        self.micro_residual_lag_boost = 1.12
        self.residual_blend = 0.64
        self.driver_follow_coef = 0.16
        self.energy_lag_gate_gain = 0.62
        self.style_energy_scale = {"eco": 1.16, "normal": 1.00, "sport": 1.02}
        self.preview_steps = (1, 3, 6)
        self.preview_ff_torque_gain = 0.050
        self.preview_ff_speed_gain = 0.70
        self.preview_ff_clip = 1.0
        self.launch_catchup_torque_gain = 0.009
        self.launch_catchup_torque_clip = 0.26
        self.launch_progress_end = 0.10
        self.launch_speed_deadband_mps = 0.06
        self.launch_dist_deadband_m = 0.05
        self.dist_catchup_torque_gain = 0.012
        self.dist_catchup_torque_clip = 0.45
        self.dist_catchup_deadband_m = 0.16
        self.dist_catchup_progress_start = 0.20
        self.energy_pos_limit_ratio = 0.85
        self.energy_neg_limit_ratio = 1.05
        self.speed_penalty_deadband_mps = 0.05
        self.catchup_pos_boost = 1.20
        self.catchup_residual_boost = 1.06
        self.lag_penalty_speed_coef = 0.10
        self.lag_penalty_dist_coef = 0.06
        self.lag_hard_coef = 0.18
        # 抗慢开偏差控制：避免节能策略依赖持续低于参考速度。
        self.bias_neutral_torque_gain = 0.0
        self.bias_neutral_torque_clip = 0.0
        self.bias_rel_speed_deadband = 0.0014
        self.bias_rel_dist_deadband = 0.0012
        self.neg_bias_penalty_coef = 0.02
        self.neg_bias_speed_deadband_mps = 0.02
        self.neg_bias_dist_deadband_m = 0.22
        self.isochronous_penalty_coef = 0.03
        self.isochronous_cost_scale = 0.12
        self.isochronous_debt_speed_gain = 0.24
        self.isochronous_debt_dist_gain = 0.10
        self.isochronous_release_speed_gain = 0.62
        self.isochronous_release_dist_gain = 0.24
        self.isochronous_debt_clip = 4.0
        self.isochronous_speed_release_deadband_mps = 0.05
        self.isochronous_dist_release_deadband_m = 0.35
        self.slow_bias_drag_debit_coef = 0.40
        self.slow_bias_kinetic_debit_coef = 0.08
        self.slow_bias_debit_progress_start = 0.08
        self.slow_bias_debit_clip = 0.0014
        self.slow_bias_speed_deadband_mps = 0.15
        self.slow_bias_speed_deadband_ratio = 0.010
        self.slow_bias_speed_deadband_min_mps = 0.035
        self.slow_bias_early_tighten_progress = 0.16
        self.slow_bias_early_deadband_scale = 0.30
        self.slow_bias_early_gate_floor = 0.55
        self.slow_bias_discount_rate = 0.50
        self.coast_bonus_coef = 0.10
        self.constraint_speed_cost_target = 0.12
        self.constraint_dist_cost_target = 0.18
        self.constraint_smooth_cost_target = 0.10
        self.constraint_net_energy_cost_target = 0.02
        self.constraint_projection_cost_target = 0.10
        self.constraint_underspeed_cost_target = 0.12
        self.constraint_window_energy_cost_target = 0.08
        self.constraint_residual_delta_tol = 1.20
        self.constraint_residual_abs_tol = 2.80
        self.constraint_regen_delta_tol = 0.16
        self.constraint_coast_delta_tol = 0.20
        self.constraint_net_energy_margin = 0.03
        self.constraint_projection_l1_tol = 0.16
        self.constraint_projection_l2_tol = 0.30
        self.constraint_underspeed_rel_speed_tol = 0.0080
        self.constraint_underspeed_rel_dist_tol = 0.0080
        self.window_energy_horizon = 12
        self.window_negative_saving_tol = 0.0015
        self.window_negative_gate_progress = 0.12
        self.window_negative_streak_gain = 0.22
        self.segment_tracking_horizon = 12
        self.segment_tracking_gate_progress = 0.06
        self.segment_speed_abs_tol_ratio = 0.65
        self.segment_dist_abs_tol_ratio = 0.60
        self.segment_speed_bias_tol_ratio = 0.28
        self.segment_dist_bias_tol_ratio = 0.22
        self.segment_tracking_constraint_scale = 0.72
        self.segment_tracking_streak_gain = 0.28
        self.window_positive_reward_coef = 0.035
        self.window_negative_reward_penalty_coef = 0.12
        self.constraint_cost_clip = 8.0
        self.eff_map_gain_coef = 0.26
        self.eff_map_penalty_coef = 0.10
        self.bias_ema_alpha = 0.10
        self.bias_kp_speed = 0.14
        self.bias_kp_dist = 0.12
        self.bias_ki_dist = 0.006
        self.bias_int_clip = 20.0
        self.bias_torque_clip = 0.9
        self.final_lag_penalty_coef = 0.0
        self.final_isochronous_penalty_coef = 0.0
        self.energy_neg_coef_good = 1.05
        self.energy_neg_coef_bad = 0.45
        self.final_energy_pos_coef = 7.0
        self.final_energy_neg_coef = 6.0
        self.tail_drift_start = 0.90
        self.tail_dist_kp = 0.14
        self.tail_dist_ki = 0.005
        self.tail_dist_int_clip = 12.0
        self.tail_drift_torque_clip = 1.0
        self.preview_speed_noise_std = 0.10
        self.preview_torque_noise_std = 1.0
        self.preview_noise_speed_clip = 0.50
        self.preview_noise_torque_clip = 4.0
        self.eval_mode = False
        self.obs_noise_scale = 1.0
        self.driver_pid_state = None
        self.driver_cmd_now = None
        self.action_proxy_limit = 0.98
        self.action_squash_eps = 0.02
        self.dist_err_int = 0.0
        self.signed_speed_err_ema = 0.0
        self.signed_dist_err_ema = 0.0
        self.signed_dist_err_int = 0.0
        self._apply_style_control_profile()

        self.base_state_dim = 35
        self.state_dim = self.base_state_dim * self.obs_stack
        self.action_dim = 3
        self._obs_history = deque(maxlen=self.obs_stack)

    def _apply_style_control_profile(self):
        """
        作用：在共享默认参数上叠加当前风格的控制系数。
        输入：无。
        输出：无。
        """
        style = str(getattr(self.scenario, "driver_style", "normal"))
        for key, value in self.STYLE_CONTROL_BASE.items():
            setattr(self, key, value)
        for key, value in self.STYLE_CONTROL_PROFILES.get(style, {}).items():
            setattr(self, key, value)

    def set_stage(
        self,
        mode,
        energy_weight=None,
        lambda_speed=None,
        lambda_dist=None,
        lambda_smooth=None,
        lambda_net_energy=None,
        lambda_projection=None,
        lambda_underspeed=None,
        lambda_window_energy=None,
    ):
        """
        作用：在纯跟踪阶段和节能优化阶段之间切换。
        输入：mode: 阶段名；energy_weight 与各类 lambda: 可选阶段参数。
        输出：无。
        """
        if mode not in ("track", "energy"):
            raise ValueError("mode must be 'track' or 'energy'")
        self.mode = mode
        if energy_weight is not None:
            self.energy_weight = float(max(0.0, energy_weight))
        if any(
            x is not None
            for x in (
                lambda_speed,
                lambda_dist,
                lambda_smooth,
                lambda_net_energy,
                lambda_projection,
                lambda_underspeed,
                lambda_window_energy,
            )
        ):
            if lambda_speed is None:
                lambda_speed = 0.0
            if lambda_dist is None:
                lambda_dist = 0.0
            if lambda_smooth is None:
                lambda_smooth = 0.0
            if lambda_net_energy is None:
                lambda_net_energy = 0.0
            if lambda_projection is None:
                lambda_projection = 0.0
            if lambda_underspeed is None:
                lambda_underspeed = 0.0
            if lambda_window_energy is None:
                lambda_window_energy = 0.0
        self.set_constraint_multipliers(
            lambda_speed=lambda_speed,
            lambda_dist=lambda_dist,
            lambda_smooth=lambda_smooth,
            lambda_net_energy=lambda_net_energy,
            lambda_projection=lambda_projection,
            lambda_underspeed=lambda_underspeed,
            lambda_window_energy=lambda_window_energy,
        )

    def get_constraint_multipliers(self):
        """
        作用：返回当前多约束 Lagrange 乘子，供训练与 checkpoint 统一存取。
        输入：无。
        输出：包含四类约束乘子的字典。
        """
        return {
            "speed": float(self.lambda_speed),
            "distance": float(self.lambda_dist),
            "smoothness": float(self.lambda_smooth),
            "net_energy": float(self.lambda_net_energy),
            "projection": float(self.lambda_projection),
            "underspeed": float(self.lambda_underspeed),
            "window_energy": float(self.lambda_window_energy),
        }

    def set_constraint_multipliers(
        self,
        multipliers=None,
        lambda_speed=None,
        lambda_dist=None,
        lambda_smooth=None,
        lambda_net_energy=None,
        lambda_projection=None,
        lambda_underspeed=None,
        lambda_window_energy=None,
    ):
        """
        作用：统一更新多约束 Lagrange 乘子，避免多处手工散写。
        输入：multipliers: 可选字典；其余参数为单独覆盖值。
        输出：无。
        """
        if multipliers is not None:
            lambda_speed = multipliers.get("speed", lambda_speed)
            lambda_dist = multipliers.get("distance", lambda_dist)
            lambda_smooth = multipliers.get("smoothness", lambda_smooth)
            lambda_net_energy = multipliers.get("net_energy", lambda_net_energy)
            lambda_projection = multipliers.get("projection", lambda_projection)
            lambda_underspeed = multipliers.get("underspeed", lambda_underspeed)
            lambda_window_energy = multipliers.get("window_energy", lambda_window_energy)
        if lambda_speed is not None:
            self.lambda_speed = float(max(0.0, lambda_speed))
        if lambda_dist is not None:
            self.lambda_dist = float(max(0.0, lambda_dist))
        if lambda_smooth is not None:
            self.lambda_smooth = float(max(0.0, lambda_smooth))
        if lambda_net_energy is not None:
            self.lambda_net_energy = float(max(0.0, lambda_net_energy))
        if lambda_projection is not None:
            self.lambda_projection = float(max(0.0, lambda_projection))
        if lambda_underspeed is not None:
            self.lambda_underspeed = float(max(0.0, lambda_underspeed))
        if lambda_window_energy is not None:
            self.lambda_window_energy = float(max(0.0, lambda_window_energy))

    def set_reference(self, scenario, reference):
        """
        作用：在不重建环境的情况下切换新的场景和参考轨迹。
        输入：scenario/reference: 新的场景和参考。
        输出：无。
        """
        self.scenario = scenario
        self.reference = reference
        self.horizon = scenario.horizon
        self.max_torque = scenario.max_torque
        self.max_speed = scenario.max_speed
        self.max_soc = scenario.max_soc
        self.driver_pid_state = None
        self.driver_cmd_now = None
        self._obs_history.clear()
        self._apply_style_control_profile()

    def set_eval_mode(self, enabled):
        """
        作用：在确定性评估时关闭观测噪声。
        输入：enabled: 是否开启评估模式。
        输出：无。
        """
        self.eval_mode = bool(enabled)
        self.obs_noise_scale = 0.0 if self.eval_mode else 1.0

    def _compute_margin_ratio(self, ref_speed_now, ref_dist_now):
        """
        作用：计算当前 rollout 距离跟踪约束边界还有多近。
        输入：ref_speed_now/ref_dist_now: 当前参考速度与距离。
        输出：约束裕度比值。
        """
        if self.mode != "energy":
            return 0.0
        speed_err_now = abs(self.vehicle["speed"] - ref_speed_now)
        dist_err_now = abs(self.vehicle["distance"] - ref_dist_now)
        return max(
            speed_err_now / max(self.constraint_speed_tol, 1e-8),
            dist_err_now / max(self.constraint_dist_tol, 1e-8),
        )

    def _compute_action_limits(self, margin_ratio, ref_torque=0.0, progress_ratio=None):
        """
        作用：根据当前跟踪裕度动态收紧或放宽 residual 动作边界。
        输入：margin_ratio: 当前约束裕度比值。
        输出：dynamic_limit、pos_limit、neg_limit。
        """
        if progress_ratio is None:
            progress_ratio = self.step_count / max(self.horizon - 1, 1)

        if self.mode == "energy":
            if margin_ratio <= 0.40:
                dynamic_limit = self.residual_limit
            elif margin_ratio <= 0.90:
                dynamic_limit = 0.85 * self.residual_limit
            else:
                dynamic_limit = 0.60 * self.residual_limit
            launch_phase = progress_ratio <= self.launch_progress_end
            decel_phase = ref_torque < -1.5
            accel_phase = ref_torque > self.phase_accel_torque_threshold_ratio * self.max_torque
            if launch_phase:
                dynamic_limit *= self.residual_phase_launch_limit_ratio
            elif decel_phase:
                dynamic_limit *= self.residual_phase_decel_limit_ratio
            elif accel_phase:
                dynamic_limit *= self.residual_phase_accel_limit_ratio
            else:
                dynamic_limit *= self.residual_phase_cruise_limit_ratio
        else:
            dynamic_limit = self.residual_limit

        if self.mode == "energy" and margin_ratio <= 1.0:
            pos_limit = self.energy_pos_limit_ratio * dynamic_limit
            neg_limit = self.energy_neg_limit_ratio * dynamic_limit
        else:
            pos_limit = dynamic_limit
            neg_limit = dynamic_limit
        return dynamic_limit, pos_limit, neg_limit

    def _energy_focus_gain(self, tracking_margin):
        """
        作用：根据跟踪质量动态调整节能优化强度。
        输入：tracking_margin: 当前跟踪误差裕度。
        输出：节能权重增益。
        """
        if self.mode != "energy":
            return 0.0
        if tracking_margin <= 0.40:
            return 1.80
        if tracking_margin <= 0.70:
            return 1.35
        if tracking_margin <= 1.00:
            return 0.95
        return 0.25

    def _smooth_coeffs(self, tracking_margin):
        """
        作用：返回与当前跟踪质量相关的平滑惩罚系数。
        输入：tracking_margin: 当前跟踪误差裕度。
        输出：平滑、residual、regen、coast 的惩罚系数。
        """
        if self.mode == "energy":
            margin_scale = min(tracking_margin, 1.5) / 1.5
            # 越贴近参考，平滑约束越强；偏差越大，恢复自由度越高。
            smooth_coeff = 0.010 + 0.012 * (1.0 - margin_scale)
            residual_coeff = 0.003 + 0.004 * (1.0 - margin_scale)
            regen_smooth_coeff = 0.007 + 0.004 * (1.0 - margin_scale)
            coast_smooth_coeff = 0.004 + 0.003 * (1.0 - margin_scale)
        else:
            smooth_coeff = 0.016
            residual_coeff = 0.005
            regen_smooth_coeff = 0.008
            coast_smooth_coeff = 0.004
        return smooth_coeff, residual_coeff, regen_smooth_coeff, coast_smooth_coeff

    def _apply_preview_noise(self, value, std, clip_abs, lo=None, hi=None):
        """
        作用：在训练时给预瞄量加入有界噪声，并做上下界裁剪。
        输入：value: 原值；std: 噪声标准差；clip_abs: 噪声裁剪；lo/hi: 上下界。
        输出：加噪并裁剪后的标量。
        """
        val = float(value)
        if self.mode != "energy":
            std = 0.0
        if self.obs_noise_scale > 1e-8 and std > 0.0:
            n = float(np.random.normal(0.0, std * self.obs_noise_scale))
            n = float(np.clip(n, -clip_abs, clip_abs))
            val += n
        if lo is not None or hi is not None:
            lo_v = -1e18 if lo is None else float(lo)
            hi_v = 1e18 if hi is None else float(hi)
            val = float(np.clip(val, lo_v, hi_v))
        return val

    def _style_regen_floor(self, style_name, margin_ratio):
        """
        作用：给策略侧动作参数化提供当前风格下的 regen 基准下限。
        输入：style_name: 风格名；margin_ratio: 跟踪裕度比值。
        输出：regen 增益基准值。
        """
        if style_name == "sport":
            driver_regen_base = float(self.scenario.driver_regen_gain_on_brake.get("sport", 0.86))
            regen_floor = driver_regen_base + 0.03
            if margin_ratio <= 0.90:
                regen_floor += 0.01
            return float(np.clip(regen_floor, 0.30, 1.20))
        if style_name == "normal":
            driver_regen_base = float(self.scenario.driver_regen_gain_on_brake.get("normal", 0.92))
            regen_floor = driver_regen_base - 0.01
            if margin_ratio <= 0.85:
                regen_floor += 0.01
            return float(np.clip(regen_floor, 0.30, 1.16))
        if style_name == "eco":
            driver_regen_base = float(self.scenario.driver_regen_gain_on_brake.get("eco", 0.95))
            regen_floor = driver_regen_base - 0.01
            if margin_ratio <= 0.95:
                regen_floor += 0.005
            return float(np.clip(regen_floor, 0.30, 1.20))
        return 1.0

    def _style_coast_thresholds(self, style_name):
        """
        作用：返回当前风格下用于滑行参数化的阈值集合。
        输入：style_name: 风格名。
        输出：speed_th、dist_th、margin_th、rise_th、speed_buf。
        """
        coast_speed_th = 0.25
        coast_dist_th = 0.40
        coast_margin_th = 0.90
        rise_th = 3.8
        speed_buf = 0.12
        if self.mode == "energy" and style_name == "normal":
            coast_speed_th = 0.16
            coast_dist_th = 0.28
            coast_margin_th = 0.98
            rise_th = 3.8
            speed_buf = 0.12
        elif self.mode == "energy" and style_name == "sport":
            coast_speed_th = 0.20
            coast_dist_th = 0.30
            coast_margin_th = 0.94
            rise_th = 3.4
            speed_buf = 0.16
        elif self.mode == "energy" and style_name == "eco":
            coast_speed_th = 0.18
            coast_dist_th = 0.30
            coast_margin_th = 0.96
            rise_th = 3.8
            speed_buf = 0.12
        return coast_speed_th, coast_dist_th, coast_margin_th, rise_th, speed_buf

    def _semantic_action_hints(self, ref_speed, ref_dist, ref_torque, ref_torque_p1, style_name, step_idx=None):
        """
        作用：生成给策略侧动作参数化使用的状态相关动作提示。
        输入：当前参考速度/距离/扭矩与下一步参考扭矩，以及风格名。
        输出：margin_ratio、residual_scale、decel_hint、regen_center/range、coast_center/range。
        """
        speed_err_now = self.vehicle["speed"] - ref_speed
        dist_err_now = self.vehicle["distance"] - ref_dist
        margin_ratio = self._compute_margin_ratio(ref_speed, ref_dist)
        if step_idx is None:
            step_idx = self.step_count
        progress_ratio = step_idx / max(self.horizon - 1, 1)
        _, pos_limit, neg_limit = self._compute_action_limits(
            margin_ratio,
            ref_torque=ref_torque,
            progress_ratio=progress_ratio,
        )
        lagging_ctx = (speed_err_now < -0.10) or (dist_err_now < -0.60)
        ahead_ctx = (speed_err_now > 0.10) or (dist_err_now > 0.60)

        micro_limit = min(
            self.micro_residual_max,
            self.micro_residual_base + self.micro_residual_ratio * abs(ref_torque),
        )
        if lagging_ctx:
            micro_limit = min(self.micro_residual_max, micro_limit * self.micro_residual_lag_boost)

        micro_pos_limit = min(pos_limit, micro_limit)
        micro_neg_limit = min(neg_limit, micro_limit)
        if style_name == "normal":
            if ahead_ctx and margin_ratio <= 0.95:
                micro_neg_limit = min(micro_neg_limit * 1.45, self.micro_residual_max * 1.45)
                micro_pos_limit = max(0.72 * micro_limit, micro_limit * 0.88)
            elif lagging_ctx:
                micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.10)
        elif style_name == "sport":
            micro_neg_limit = min(micro_neg_limit, 1.00 * micro_limit)
            if lagging_ctx:
                micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.22)
                micro_neg_limit = min(micro_neg_limit, 0.88 * micro_limit)
            elif ahead_ctx and margin_ratio <= 0.95:
                micro_neg_limit = min(micro_neg_limit, 0.95 * micro_limit)
                micro_pos_limit = max(0.74 * micro_limit, micro_limit * 0.90)
        elif style_name == "eco":
            if ahead_ctx and margin_ratio <= 1.0:
                micro_neg_limit = min(micro_neg_limit * 1.20, self.micro_residual_max * 1.20)
                micro_pos_limit = max(0.76 * micro_limit, micro_limit * 0.92)
            elif lagging_ctx:
                micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.10)
                micro_neg_limit = min(micro_neg_limit, 0.90 * micro_limit)

        residual_scale = 0.5 * (micro_pos_limit + micro_neg_limit) / max(self.residual_limit, 1e-8)
        residual_scale = 0.35 + 0.65 * residual_scale
        residual_scale = float(np.clip(residual_scale, 0.35, 1.0))

        decel_context = ref_torque < -1.5 or self.vehicle["speed"] > ref_speed + 0.5
        if decel_context:
            regen_center = self._regen_gain_to_action(self._style_regen_floor(style_name, margin_ratio))
            if margin_ratio <= 0.75:
                regen_range = 0.38
            elif margin_ratio <= 1.0:
                regen_range = 0.30
            else:
                regen_range = 0.20
        else:
            regen_center = self._regen_gain_to_action(1.0)
            regen_range = 0.18
        regen_range = float(np.clip(regen_range, 0.12, 0.55))
        regen_room = max(0.16, self.action_proxy_limit - abs(regen_center))
        regen_range = min(regen_range, regen_room)

        speed_ahead = self.vehicle["speed"] - ref_speed
        dist_ahead = self.vehicle["distance"] - ref_dist
        coast_speed_th, coast_dist_th, coast_margin_th, rise_th, speed_buf = self._style_coast_thresholds(style_name)
        coast_gate = 0.0
        if self.mode == "energy" and (not decel_context) and ref_torque > 0.0:
            speed_soft = float(np.clip(speed_ahead / max(coast_speed_th + 0.10, 1e-8), 0.0, 1.0))
            dist_soft = float(np.clip(dist_ahead / max(coast_dist_th + 0.14, 1e-8), 0.0, 1.0))
            margin_soft = float(np.clip(1.0 - max(0.0, margin_ratio - coast_margin_th) / 0.22, 0.0, 1.0))
            coast_gate = min(speed_soft, dist_soft, margin_soft)
            predicted_torque_rise = ref_torque_p1 - ref_torque
            if predicted_torque_rise > rise_th and speed_ahead < (coast_speed_th + speed_buf):
                coast_gate *= 0.35
        if coast_gate > 0.0:
            pref_gain = 0.12 + 0.62 * coast_gate
            if style_name == "eco":
                pref_gain += 0.05 * coast_gate
            elif style_name == "sport":
                pref_gain -= 0.04 * coast_gate
            pref_gain = float(np.clip(pref_gain, 0.0, 1.0))
            coast_center = self._coast_gain_to_action(pref_gain)
            coast_range = 0.18 + 0.34 * coast_gate
        else:
            coast_center = -1.0
            coast_range = 0.18
        coast_room = max(0.16, self.action_proxy_limit - abs(coast_center))
        coast_range = float(min(np.clip(coast_range, 0.12, 0.55), coast_room))
        return (
            float(np.clip(margin_ratio / 1.5, 0.0, 1.5)),
            residual_scale,
            1.0 if decel_context else 0.0,
            float(regen_center),
            float(regen_range),
            float(coast_center),
            float(coast_range),
        )

    def _encode_state(self, step_idx):
        """
        作用：组装并归一化送给策略网络的观测向量。
        输入：step_idx: 当前时间步。
        输出：状态向量 ndarray。
        """
        ref_speed_plan_arr = self.reference["speed_target"] if "speed_target" in self.reference else self.reference["speed"]
        ref_speed = float(self.reference["speed"][step_idx])
        ref_dist = float(self.reference["distance"][step_idx])
        if self.driver_cmd_now is None:
            ref_torque = float(self.reference["torque_cmd"][min(step_idx, self.horizon - 1)])
        else:
            ref_torque = float(self.driver_cmd_now)
        p1 = min(step_idx + self.preview_steps[0], self.horizon)
        p2 = min(step_idx + self.preview_steps[1], self.horizon)
        p3 = min(step_idx + self.preview_steps[2], self.horizon)
        ref_speed_p1 = float(ref_speed_plan_arr[p1])
        ref_speed_p2 = float(ref_speed_plan_arr[p2])
        ref_speed_p3 = float(ref_speed_plan_arr[p3])
        ref_torque_p1 = float(self.reference["torque_cmd"][min(p1, self.horizon - 1)])
        ref_torque_p2 = float(self.reference["torque_cmd"][min(p2, self.horizon - 1)])
        ref_torque_p3 = float(self.reference["torque_cmd"][min(p3, self.horizon - 1)])
        ref_speed_p1_obs = self._apply_preview_noise(
            ref_speed_p1, self.preview_speed_noise_std, self.preview_noise_speed_clip, 0.0, self.max_speed
        )
        ref_speed_p2_obs = self._apply_preview_noise(
            ref_speed_p2, 1.2 * self.preview_speed_noise_std, 1.2 * self.preview_noise_speed_clip, 0.0, self.max_speed
        )
        ref_speed_p3_obs = self._apply_preview_noise(
            ref_speed_p3, 1.4 * self.preview_speed_noise_std, 1.4 * self.preview_noise_speed_clip, 0.0, self.max_speed
        )
        ref_torque_p1_obs = self._apply_preview_noise(
            ref_torque_p1, self.preview_torque_noise_std, self.preview_noise_torque_clip, -self.max_torque, self.max_torque
        )
        ref_torque_p2_obs = self._apply_preview_noise(
            ref_torque_p2, 1.2 * self.preview_torque_noise_std, 1.2 * self.preview_noise_torque_clip, -self.max_torque, self.max_torque
        )
        ref_torque_p3_obs = self._apply_preview_noise(
            ref_torque_p3, 1.4 * self.preview_torque_noise_std, 1.4 * self.preview_noise_torque_clip, -self.max_torque, self.max_torque
        )

        speed_err = self.vehicle["speed"] - ref_speed
        dist_err = self.vehicle["distance"] - ref_dist
        dist_scale = max(self.reference["total_distance"], 1.0)
        style_name, _, _, _ = self._style_flags()
        (
            margin_ratio_norm,
            residual_scale_hint,
            decel_hint,
            regen_center_hint,
            regen_range_hint,
            coast_center_hint,
            coast_range_hint,
        ) = self._semantic_action_hints(
            ref_speed=ref_speed,
            ref_dist=ref_dist,
            ref_torque=ref_torque,
            ref_torque_p1=ref_torque_p1,
            style_name=style_name,
            step_idx=step_idx,
        )
        torque_trend = ref_torque_p1 - ref_torque
        progress_feature = float(np.clip(0.35 * (step_idx / max(self.horizon, 1)), 0.0, 0.35))
        accel_phase = float(
            np.clip(
                max(ref_torque, 0.0) / 18.0
                + 0.45 * max(torque_trend, 0.0) / max(self.max_torque, 1e-8),
                0.0,
                1.0,
            )
        )
        decel_phase = float(
            np.clip(
                max(-ref_torque, 0.0) / 12.0
                + 0.55 * float(decel_hint)
                + 0.25 * max(self.vehicle["speed"] - ref_speed, 0.0),
                0.0,
                1.0,
            )
        )
        coastable_phase = float(
            np.clip(
                0.70 * max(0.0, 0.5 * (coast_center_hint + 1.0))
                + 0.30 * np.clip((coast_range_hint - 0.12) / 0.43, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        torque_trend_norm = float(np.clip(torque_trend / self.max_torque, -1.0, 1.0))
        return np.array(
            [
                self.vehicle["speed"] / self.max_speed,
                self.vehicle["soc"] / self.max_soc,
                self.vehicle["torque"] / self.max_torque,
                ref_speed / self.max_speed,
                ref_torque / self.max_torque,
                speed_err / self.max_speed,
                dist_err / dist_scale,
                progress_feature,
                accel_phase,
                decel_phase,
                coastable_phase,
                torque_trend_norm,
                self.prev_residual / self.residual_limit,
                self.scenario.slope_profile[min(step_idx, self.horizon)] / 0.12,
                self.reference["curve_speed_limit"][min(step_idx, self.horizon)] / self.max_speed,
                self.prev_regen_gain,
                self.prev_coast_gain,
                ref_speed_p1_obs / self.max_speed,
                ref_speed_p2_obs / self.max_speed,
                ref_speed_p3_obs / self.max_speed,
                ref_torque_p1_obs / self.max_torque,
                ref_torque_p2_obs / self.max_torque,
                ref_torque_p3_obs / self.max_torque,
                (ref_speed_p1_obs - ref_speed) / self.max_speed,
                (ref_torque_p1_obs - ref_torque) / self.max_torque,
                self.scenario.driver_style_onehot[0],
                self.scenario.driver_style_onehot[1],
                self.scenario.driver_style_onehot[2],
                margin_ratio_norm,
                residual_scale_hint,
                decel_hint,
                regen_center_hint,
                regen_range_hint,
                coast_center_hint,
                coast_range_hint,
            ],
            dtype=np.float32,
        )

    def _stack_observation(self, obs, reset_history=False):
        """
        作用：维护固定长度的观测历史，并输出拼接后的短时序状态。
        输入：obs: 当前基础观测；reset_history: 是否用当前观测重置整段历史。
        输出：拼接后的堆叠观测向量。
        """
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if reset_history or len(self._obs_history) == 0:
            self._obs_history.clear()
            for _ in range(self.obs_stack):
                self._obs_history.append(obs_arr.copy())
        else:
            self._obs_history.append(obs_arr.copy())
        return np.concatenate(list(self._obs_history), axis=0).astype(np.float32, copy=False)

    def _regen_gain_to_action(self, regen_gain):
        """
        作用：把实际 regen 增益反推回策略动作空间中的第二维代理动作。
        输入：regen_gain: 实际执行的 regen 增益。
        输出：[-1, 1] 区间内的代理动作。
        """
        return float(np.clip((float(regen_gain) - 0.75) / 0.45, -self.action_proxy_limit, self.action_proxy_limit))

    def _coast_gain_to_action(self, coast_gain):
        """
        作用：把实际滑行增益反推回策略动作空间中的第三维代理动作。
        输入：coast_gain: 实际执行的 coast 增益。
        输出：[-1, 1] 区间内的代理动作。
        """
        return float(np.clip(2.0 * float(coast_gain) - 1.0, -self.action_proxy_limit, self.action_proxy_limit))

    def _hint_proxy_bounds(self, center, span, low_anchor=None):
        """
        作用：根据动作语义提示生成代理动作空间中的有效上下界。
        输入：center/span: 提示中心与范围；low_anchor: 可选的下界锚点。
        输出：裁剪后的 low/high。
        """
        low = float(center) - float(span)
        if low_anchor is not None:
            low = max(low, float(low_anchor))
        high = float(center) + float(span)
        low = float(np.clip(low, -self.action_proxy_limit, self.action_proxy_limit))
        high = float(np.clip(high, -self.action_proxy_limit, self.action_proxy_limit))
        if high < low:
            high = low
        return low, high

    def _action_to_raw(self, action_vec):
        """
        作用：把环境动作空间中的执行动作反推回 tanh-squash 前空间。
        输入：action_vec: [residual, regen_proxy, coast_proxy] 动作向量。
        输出：对应的 squash 前动作向量。
        """
        act = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        scaled = np.array(
            [
                act[0] / max(self.residual_limit, 1e-8),
                act[1],
                act[2],
            ],
            dtype=np.float32,
        )
        scaled = np.clip(scaled, -1.0 + self.action_squash_eps, 1.0 - self.action_squash_eps)
        raw = 0.5 * (np.log1p(scaled) - np.log1p(-scaled))
        return raw.astype(np.float32, copy=False)

    def reset(self, seed=None):
        """
        作用：重置 episode 计数与缓存，并返回初始观测。
        输入：seed: 可选随机种子。
        输出：初始状态和空 info。
        """
        if seed is not None:
            np.random.seed(seed)
        self.vehicle = self.scenario.reset_vehicle()
        self.step_count = 0
        self.prev_residual = 0.0
        self.prev_regen_gain = 1.0
        self.prev_coast_gain = 0.0
        self.cum_speed_abs_err = 0.0
        self.cum_dist_abs_err = 0.0
        self.cum_energy = 0.0
        self.cum_energy_accounted = 0.0
        self.cum_recover = 0.0
        self.cum_distance = 0.0
        self.cum_steps = 0
        self.dist_err_int = 0.0
        self.signed_speed_err_ema = 0.0
        self.signed_dist_err_ema = 0.0
        self.signed_dist_err_int = 0.0
        self.isochronous_speed_debt = 0.0
        self.isochronous_dist_debt = 0.0
        self.prev_speed_deficit_mps = 0.0
        self.cum_slow_bias_energy_debit = 0.0
        self.window_agent_energy_hist = deque(maxlen=int(self.window_energy_horizon))
        self.window_ref_energy_hist = deque(maxlen=int(self.window_energy_horizon))
        self.window_step_distance_hist = deque(maxlen=int(self.window_energy_horizon))
        self.window_ref_distance_hist = deque(maxlen=int(self.window_energy_horizon))
        self.window_negative_streak = 0
        self.segment_speed_err_hist = deque(maxlen=int(self.segment_tracking_horizon))
        self.segment_dist_err_hist = deque(maxlen=int(self.segment_tracking_horizon))
        self.segment_tracking_streak = 0
        self.launch_negative_torque_streak = 0
        self.driver_pid_state = self.scenario.init_driver_pid_state(style_name=self.scenario.driver_style)
        self.driver_cmd_now = float(self.reference["torque_cmd"][0]) if len(self.reference["torque_cmd"]) > 0 else 0.0
        base_obs = self._encode_state(0)
        return self._stack_observation(base_obs, reset_history=True), {}

    def _style_flags(self):
        """
        作用：返回当前驾驶风格名称及其布尔快捷标记。
        输入：无。
        输出：style_name、is_normal_style、is_sport_style、is_eco_style。
        """
        style_name = str(getattr(self.scenario, "driver_style", "normal"))
        return (
            style_name,
            style_name == "normal",
            style_name == "sport",
            style_name == "eco",
        )

    def _update_bias_state(self, speed_err_now, dist_err_now):
        """
        作用：更新用于抗慢开偏差修正的 EMA 和积分量。
        输入：speed_err_now/dist_err_now: 当前速度和距离误差。
        输出：lag_err_now。
        """
        lag_err_now = min(dist_err_now, 0.0)
        if self.mode == "energy":
            ema_alpha = self.bias_ema_alpha
            lag_speed = min(speed_err_now, 0.0)
            lag_dist = min(dist_err_now, 0.0)
            self.signed_speed_err_ema = (1.0 - ema_alpha) * self.signed_speed_err_ema + ema_alpha * lag_speed
            self.signed_dist_err_ema = (1.0 - ema_alpha) * self.signed_dist_err_ema + ema_alpha * lag_dist
            self.signed_dist_err_int = float(
                np.clip(
                    self.signed_dist_err_int + lag_dist,
                    -self.bias_int_clip,
                    self.bias_int_clip,
                )
            )
            self.dist_err_int = float(
                np.clip(
                    self.dist_err_int + lag_err_now,
                    -self.tail_dist_int_clip,
                    self.tail_dist_int_clip,
                )
            )
        else:
            self.signed_speed_err_ema = 0.0
            self.signed_dist_err_ema = 0.0
            self.signed_dist_err_int = 0.0
            self.dist_err_int = 0.0
        return lag_err_now

    def _compute_slow_bias_energy_debit(
        self,
        ref_speed,
        agent_speed,
        ref_torque,
        ref_step_energy,
        dist_err,
        progress_ratio,
        step_idx,
        coast_gain=0.0,
        coast_allowed=False,
        decel_context=False,
    ):
        """
        作用：估算由于慢开带来的阻力减小和动能欠账，并把这部分能耗加回账本。
        输入：参考/实际速度、参考扭矩、距离误差、进度与步索引。
        输出：总回补能耗、阻力回补、动能回补。
        """
        if self.mode != "energy":
            self.prev_speed_deficit_mps = 0.0
            return 0.0, 0.0, 0.0
        speed_deficit = max(0.0, float(ref_speed - agent_speed))
        if speed_deficit <= 1e-8:
            self.prev_speed_deficit_mps = 0.0
            return 0.0, 0.0, 0.0
        if decel_context or float(ref_torque) <= 1.5:
            self.prev_speed_deficit_mps = speed_deficit
            return 0.0, 0.0, 0.0
        if coast_allowed and float(coast_gain) >= 0.12:
            self.prev_speed_deficit_mps = speed_deficit
            return 0.0, 0.0, 0.0
        dynamic_deadband = float(
            np.clip(
                ref_speed * self.slow_bias_speed_deadband_ratio,
                self.slow_bias_speed_deadband_min_mps,
                self.slow_bias_speed_deadband_mps,
            )
        )
        early_phase = float(
            np.clip(
                progress_ratio / max(self.slow_bias_early_tighten_progress, 1e-8),
                0.0,
                1.0,
            )
        )
        effective_deadband = dynamic_deadband * (
            self.slow_bias_early_deadband_scale
            + (1.0 - self.slow_bias_early_deadband_scale) * early_phase
        )
        taxed_speed_deficit = max(0.0, speed_deficit - effective_deadband)
        if taxed_speed_deficit <= 1e-8:
            self.prev_speed_deficit_mps = speed_deficit
            return 0.0, 0.0, 0.0
        agent_speed_taxed = max(0.0, float(ref_speed - taxed_speed_deficit))
        progress_gate = float(
            np.clip(
                (progress_ratio - self.slow_bias_debit_progress_start)
                / max(1.0 - self.slow_bias_debit_progress_start, 1e-8),
                0.0,
                1.0,
            )
        )
        early_gate = self.slow_bias_early_gate_floor * float(
            np.clip(
                1.0 - progress_ratio / max(self.slow_bias_early_tighten_progress, 1e-8),
                0.0,
                1.0,
            )
        )
        progress_gate = max(progress_gate, early_gate)
        if progress_gate <= 0.0:
            self.prev_speed_deficit_mps = speed_deficit
            return 0.0, 0.0, 0.0

        _, curve_v_lim = self.scenario.road_feature(step_idx)
        drag_ref = (
            self.scenario.drag_linear * ref_speed
            + self.scenario.drag_quad * ref_speed * ref_speed
            + self.scenario.curve_drag_gain * max(0.0, ref_speed - curve_v_lim)
        )
        drag_agent = (
            self.scenario.drag_linear * agent_speed_taxed
            + self.scenario.drag_quad * agent_speed_taxed * agent_speed_taxed
            + self.scenario.curve_drag_gain * max(0.0, agent_speed_taxed - curve_v_lim)
        )
        resistive_acc_gap = max(0.0, drag_ref - drag_agent)
        resistive_torque_gap = resistive_acc_gap / max(self.scenario.torque_to_acc, 1e-8)
        ref_rpm = max(ref_speed, agent_speed_taxed, 0.1) * self.scenario.rpm_per_mps
        eff_proxy = max(
            self.scenario.efficiency(max(abs(ref_torque), resistive_torque_gap), ref_rpm),
            1e-4,
        )
        drag_mech_power = resistive_torque_gap * ref_rpm / 9550.0
        drag_debit = (
            self.slow_bias_drag_debit_coef
            * max(0.0, drag_mech_power)
            / eff_proxy
            * self.scenario.dt
            * self.scenario.base_soc_factor
        )

        kinetic_gap = max(0.0, 0.5 * (ref_speed * ref_speed - agent_speed_taxed * agent_speed_taxed))
        avg_speed = max(0.5 * (ref_speed + agent_speed_taxed), 0.1)
        kinetic_rpm = avg_speed * self.scenario.rpm_per_mps
        kinetic_torque_proxy = kinetic_gap / max(self.scenario.torque_to_acc * self.scenario.dt, 1e-8)
        kinetic_eff = max(
            self.scenario.efficiency(max(abs(ref_torque), kinetic_torque_proxy), kinetic_rpm),
            1e-4,
        )
        kinetic_debit = (
            self.slow_bias_kinetic_debit_coef
            * kinetic_gap
            * self.scenario.base_soc_factor
            / kinetic_eff
        )

        dist_lag_scale = 1.0 + 0.10 * float(
            np.clip(max(0.0, -dist_err) / max(self.constraint_dist_tol, 1e-8), 0.0, 2.0)
        )
        propulsion_gate = float(np.clip((float(ref_torque) - 4.0) / 14.0, 0.0, 1.0))
        coast_suppression = 1.0 - float(np.clip(float(coast_gain) / 0.25, 0.0, 1.0))
        debit_scale = (
            progress_gate
            * dist_lag_scale
            * propulsion_gate
            * coast_suppression
            * float(np.clip(self.slow_bias_discount_rate, 0.0, 1.0))
        )
        total_debit = debit_scale * (drag_debit + kinetic_debit)
        step_cap = min(self.slow_bias_debit_clip, 0.35 * max(float(ref_step_energy), 0.0))
        total_debit = float(np.clip(total_debit, 0.0, step_cap))
        drag_debit = float(min(total_debit, debit_scale * drag_debit))
        kinetic_debit = float(max(0.0, total_debit - drag_debit))
        self.prev_speed_deficit_mps = speed_deficit
        return total_debit, drag_debit, kinetic_debit

    def _apply_residual_policy(
        self,
        raw_residual,
        ref_torque,
        pos_limit,
        margin_ratio,
        lagging_ctx,
        ahead_ctx,
        style_name,
    ):
        """
        作用：按风格相关的安全启发式规则修正原始 residual 扭矩动作。
        输入：raw_residual: 原始 residual；ref_torque: 参考扭矩；其余参数为上下文信息。
        输出：处理后的 residual 扭矩。
        """
        residual = float(raw_residual)
        if self.mode == "energy":
            if ahead_ctx:
                if margin_ratio > self.recover_margin_high:
                    residual *= self.recover_residual_scale_high
                elif margin_ratio > self.recover_margin_mid:
                    residual *= self.recover_residual_scale_mid
            elif lagging_ctx and residual > 0.0:
                residual = min(residual * self.catchup_residual_boost, pos_limit)

            # 让智能体始终扮演驾驶员命令附近的微调器。
            micro_limit = min(
                self.micro_residual_max,
                self.micro_residual_base + self.micro_residual_ratio * abs(ref_torque),
            )
            if lagging_ctx:
                micro_limit = min(self.micro_residual_max, micro_limit * self.micro_residual_lag_boost)

            micro_pos_limit = micro_limit
            micro_neg_limit = micro_limit
            if style_name == "normal":
                if ahead_ctx and margin_ratio <= 0.95:
                    micro_neg_limit = min(self.micro_residual_max * 1.45, micro_limit * 1.45)
                    micro_pos_limit = max(0.72 * micro_limit, micro_limit * 0.88)
                elif lagging_ctx:
                    micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.10)
            elif style_name == "sport":
                micro_neg_limit = 1.00 * micro_limit
                if lagging_ctx:
                    micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.22)
                    micro_neg_limit = 0.88 * micro_limit
                elif ahead_ctx and margin_ratio <= 0.95:
                    micro_neg_limit = 0.95 * micro_limit
                    micro_pos_limit = max(0.74 * micro_limit, micro_limit * 0.90)
            elif style_name == "eco":
                if ahead_ctx and margin_ratio <= 1.0:
                    micro_neg_limit = min(self.micro_residual_max * 1.20, micro_limit * 1.20)
                    micro_pos_limit = max(0.76 * micro_limit, micro_limit * 0.92)
                elif lagging_ctx:
                    micro_pos_limit = min(self.micro_residual_max, micro_limit * 1.10)
                    micro_neg_limit = 0.90 * micro_limit

            residual = float(np.clip(residual, -micro_neg_limit, micro_pos_limit))
            blend = self.residual_blend
            if style_name == "sport":
                blend = max(0.42, self.residual_blend - 0.08)
            residual = blend * residual + (1.0 - blend) * self.prev_residual
            residual = float(np.clip(residual, -micro_neg_limit, micro_pos_limit))
            return residual

        track_micro_limit = min(4.0, 1.8 + 0.06 * abs(ref_torque))
        residual = float(np.clip(residual, -track_micro_limit, track_micro_limit))
        residual = 0.65 * residual + 0.35 * self.prev_residual
        return float(np.clip(residual, -track_micro_limit, track_micro_limit))

    def _apply_regen_policy(
        self,
        action,
        ref_torque,
        ref_speed_now,
        margin_ratio,
        style_name,
        regen_center_hint,
        regen_range_hint,
        coast_center_hint,
        coast_range_hint,
        decel_hint=None,
    ):
        """
        作用：按策略侧动作语义提示把代理动作映射为 regen/coast 增益。
        输入：action: 策略动作；其余参数为当前上下文与动作提示。
        输出：regen_gain、coast_gain 和 decel_context。
        """
        if decel_hint is None:
            decel_context = ref_torque < -1.5 or self.vehicle["speed"] > ref_speed_now + 0.5
        else:
            decel_context = bool(float(decel_hint) >= 0.5)
        regen_low_anchor = float(regen_center_hint) if decel_context else None
        regen_low, regen_high = self._hint_proxy_bounds(
            regen_center_hint,
            regen_range_hint,
            low_anchor=regen_low_anchor,
        )
        regen_raw = float(np.clip(action[1], regen_low, regen_high))
        if (float(coast_center_hint) <= (-self.action_proxy_limit + 1e-4)) and (float(coast_range_hint) <= 0.18 + 1e-6):
            coast_raw = -self.action_proxy_limit
        else:
            coast_low, coast_high = self._hint_proxy_bounds(coast_center_hint, coast_range_hint)
            coast_raw = float(np.clip(action[2] if len(action) > 2 else -1.0, coast_low, coast_high))
        coast_gain = 0.5 * (coast_raw + 1.0)
        regen_gain = 0.30 + 0.90 * (regen_raw + 1.0) * 0.5
        return regen_gain, coast_gain, decel_context

    def _apply_coast_policy(
        self,
        torque_cmd_unclipped,
        coast_gain,
        ref_torque,
        decel_context,
    ):
        """
        作用：按策略已参数化的 coast 增益缩放扭矩命令。
        输入：torque_cmd_unclipped: 未裁剪扭矩；coast_gain: 滑行增益；其余参数为上下文信息。
        输出：更新后的扭矩命令、coast_gain 与 coast_allowed。
        """
        coast_allowed = (
            self.mode == "energy"
            and (not decel_context)
            and ref_torque > 0.0
            and coast_gain > 1e-3
        )
        if not coast_allowed:
            return torque_cmd_unclipped, 0.0, False
        coast_scale = 1.0 - self.coast_torque_coef * coast_gain
        coast_floor = 0.80
        torque_cmd_unclipped *= max(coast_scale, coast_floor)
        return torque_cmd_unclipped, coast_gain, True

    def step(self, action):
        """
        作用：推进一步 RL 环境，计算奖励并返回下一观测。
        输入：action: 智能体动作。
        输出：next_state、reward、terminated、truncated、info。
        """
        style_name, is_normal_style, is_sport_style, is_eco_style = self._style_flags()
        ref_speed_now = float(self.reference["speed"][self.step_count])
        ref_dist_now = float(self.reference["distance"][self.step_count])
        ref_speed_plan_now = (
            float(self.reference["speed_target"][self.step_count])
            if "speed_target" in self.reference
            else ref_speed_now
        )
        speed_err_now = self.vehicle["speed"] - ref_speed_now
        dist_err_now = self.vehicle["distance"] - ref_dist_now
        total_dist_scale = max(float(self.reference["total_distance"]), 1e-8)
        speed_rel_bias_now = speed_err_now / max(self.max_speed, 1e-8)
        dist_rel_bias_now = dist_err_now / total_dist_scale
        proposed_action = np.asarray(action, dtype=np.float32).reshape(-1)
        proposed_bound = np.array(
            [self.residual_limit, self.action_proxy_limit, self.action_proxy_limit],
            dtype=np.float32,
        )
        proposed_action_violation = np.maximum(np.abs(proposed_action) - proposed_bound, 0.0)
        proposed_action_violation_linf = float(np.max(proposed_action_violation))
        margin_ratio = self._compute_margin_ratio(ref_speed_now, ref_dist_now)
        progress_ratio = self.step_count / max(self.horizon - 1, 1)
        lagging_ctx = (speed_err_now < -0.10) or (dist_err_now < -0.60)
        ahead_ctx = (speed_err_now > 0.10) or (dist_err_now > 0.60)
        lag_err_now = self._update_bias_state(speed_err_now, dist_err_now)

        if self.driver_pid_state is None:
            self.driver_pid_state = self.scenario.init_driver_pid_state(style_name=self.scenario.driver_style)
        ref_torque, self.driver_pid_state = self.scenario.driver_pid_command(
            self.driver_pid_state,
            ref_speed_plan_now,
            self.vehicle["speed"],
            step_idx=self.step_count,
        )
        self.driver_cmd_now = float(ref_torque)
        dynamic_limit, pos_limit, neg_limit = self._compute_action_limits(
            margin_ratio,
            ref_torque=ref_torque,
            progress_ratio=progress_ratio,
        )
        if self.mode == "energy" and lagging_ctx:
            pos_limit = min(self.residual_limit * 1.15, pos_limit * self.catchup_pos_boost)
        p1_idx = min(self.step_count + self.preview_steps[0], self.horizon)
        ref_torque_p1 = float(self.reference["torque_cmd"][min(p1_idx, self.horizon - 1)])
        ref_speed_p1 = (
            float(self.reference["speed_target"][p1_idx])
            if "speed_target" in self.reference
            else float(self.reference["speed"][p1_idx])
        )
        (
            _margin_ratio_hint,
            _residual_scale_hint,
            decel_hint,
            regen_center_hint,
            regen_range_hint,
            coast_center_hint,
            coast_range_hint,
        ) = self._semantic_action_hints(
            ref_speed_now,
            ref_dist_now,
            ref_torque,
            ref_torque_p1,
            style_name,
            step_idx=self.step_count,
        )
        residual = self._apply_residual_policy(
            raw_residual=np.clip(action[0], -neg_limit, pos_limit),
            ref_torque=ref_torque,
            pos_limit=pos_limit,
            margin_ratio=margin_ratio,
            lagging_ctx=lagging_ctx,
            ahead_ctx=ahead_ctx,
            style_name=style_name,
        )
        regen_gain, coast_gain, decel_context = self._apply_regen_policy(
            action=action,
            ref_torque=ref_torque,
            ref_speed_now=ref_speed_now,
            margin_ratio=margin_ratio,
            style_name=style_name,
            regen_center_hint=regen_center_hint,
            regen_range_hint=regen_range_hint,
            coast_center_hint=coast_center_hint,
            coast_range_hint=coast_range_hint,
            decel_hint=decel_hint,
        )
        drift_corr_torque = 0.0
        if self.mode == "energy" and progress_ratio >= self.tail_drift_start:
            tail_gain = (progress_ratio - self.tail_drift_start) / max(1.0 - self.tail_drift_start, 1e-8)
            if lag_err_now < -3.2:
                base_corr = -(self.tail_dist_kp * lag_err_now + self.tail_dist_ki * self.dist_err_int)
                drift_corr_torque = float(
                    np.clip(base_corr * tail_gain, -self.tail_drift_torque_clip, self.tail_drift_torque_clip)
                )
                if decel_context:
                    drift_corr_torque *= 0.30
        bias_corr_torque = 0.0
        if self.mode == "energy" and (speed_err_now < -0.08 or dist_err_now < -0.60):
            progress_gain = 0.50 + 0.70 * progress_ratio
            bias_raw = -(
                self.bias_kp_speed * self.signed_speed_err_ema
                + self.bias_kp_dist * self.signed_dist_err_ema
                + self.bias_ki_dist * self.signed_dist_err_int
            ) * progress_gain
            lag_boost = min(max(0.0, -dist_err_now) / max(self.constraint_dist_tol, 1e-8), 1.5)
            bias_clip = self.bias_torque_clip * (1.0 + 0.9 * lag_boost)
            bias_corr_torque = float(np.clip(bias_raw, -bias_clip, bias_clip))
            if decel_context and bias_corr_torque > 0.0:
                bias_corr_torque *= 0.35
        bias_neutral_torque = 0.0
        if self.mode == "energy" and (not decel_context) and (progress_ratio >= 0.35) and (is_normal_style or is_sport_style or is_eco_style):
            slow_speed_rel = max(0.0, -speed_rel_bias_now - self.bias_rel_speed_deadband)
            slow_dist_rel = max(0.0, -dist_rel_bias_now - self.bias_rel_dist_deadband)
            slow_bias_score = slow_speed_rel + 0.75 * slow_dist_rel
            if slow_bias_score > 0.0:
                neutral_gain = 0.55 + 0.55 * progress_ratio
                neutral_torque_gain = self.bias_neutral_torque_gain
                neutral_torque_clip = self.bias_neutral_torque_clip
                if is_sport_style:
                    neutral_gain *= 0.70
                    neutral_torque_gain *= 0.75
                    neutral_torque_clip = min(neutral_torque_clip, 0.75)
                elif is_eco_style:
                    neutral_gain *= 0.95
                    neutral_torque_gain *= 0.85
                    neutral_torque_clip = min(neutral_torque_clip, 0.28)
                neutral_raw = neutral_torque_gain * self.max_torque * slow_bias_score * neutral_gain
                if ahead_ctx:
                    neutral_raw *= 0.25
                bias_neutral_torque = float(np.clip(neutral_raw, 0.0, neutral_torque_clip))
        launch_catchup_torque = 0.0
        if self.mode == "energy" and (not decel_context) and progress_ratio <= self.launch_progress_end:
            launch_speed_lag = max(0.0, -(speed_err_now + self.launch_speed_deadband_mps))
            launch_dist_lag = max(0.0, -(dist_err_now + self.launch_dist_deadband_m))
            if launch_speed_lag > 0.0 or launch_dist_lag > 0.0:
                phase_gain = float(np.clip(1.0 - progress_ratio / max(self.launch_progress_end, 1e-8), 0.20, 1.0))
                speed_norm = launch_speed_lag / max(self.constraint_speed_tol, 1e-8)
                dist_norm = launch_dist_lag / max(self.constraint_dist_tol, 1e-8)
                launch_need = float(np.clip(0.85 * speed_norm + 0.55 * dist_norm, 0.0, 1.6))
                launch_raw = self.launch_catchup_torque_gain * self.max_torque * launch_need * phase_gain
                launch_catchup_torque = float(np.clip(launch_raw, 0.0, self.launch_catchup_torque_clip))
        dist_catchup_torque = 0.0
        if self.mode == "energy" and (not decel_context) and progress_ratio >= self.dist_catchup_progress_start:
            dist_lag_m = max(0.0, -(dist_err_now + self.dist_catchup_deadband_m))
            if dist_lag_m > 0.0:
                dist_norm = float(np.clip(dist_lag_m / max(self.constraint_dist_tol, 1e-8), 0.0, 1.6))
                speed_catchup_gate = float(np.clip(1.0 - max(0.0, speed_err_now) / 0.35, 0.25, 1.0))
                margin_gate = float(np.clip(1.15 - margin_ratio, 0.30, 1.15))
                dist_raw = self.dist_catchup_torque_gain * self.max_torque * dist_norm * speed_catchup_gate * margin_gate
                dist_catchup_torque = float(np.clip(dist_raw, 0.0, self.dist_catchup_torque_clip))
        preview_corr_torque = 0.0
        preview_trigger = (
            (dist_err_now < -1.20)
            or (progress_ratio > 0.72 and dist_err_now < -0.65)
            or (speed_err_now < -0.45)
        )
        if self.mode == "energy" and preview_trigger and (not decel_context):
            torque_rise = max(0.0, ref_torque_p1 - ref_torque)
            speed_rise = max(0.0, ref_speed_p1 - ref_speed_now)
            lag_scale_ff = float(
                np.clip(
                    max(
                        max(0.0, -speed_err_now) / max(self.constraint_speed_tol, 1e-8),
                        max(0.0, -dist_err_now) / max(self.constraint_dist_tol, 1e-8),
                    ),
                    0.0,
                    1.25,
                )
            )
            ff_raw = (
                self.preview_ff_torque_gain * torque_rise
                + self.preview_ff_speed_gain * (speed_rise / max(self.max_speed, 1e-8)) * self.max_torque
            )
            preview_corr_torque = float(np.clip(ff_raw * lag_scale_ff, 0.0, self.preview_ff_clip))
        torque_cmd_unclipped = (
            ref_torque
            + residual
            + drift_corr_torque
            + bias_corr_torque
            + preview_corr_torque
            + bias_neutral_torque
            + launch_catchup_torque
            + dist_catchup_torque
        )
        torque_cmd_unclipped, coast_gain, coast_allowed = self._apply_coast_policy(
            torque_cmd_unclipped=torque_cmd_unclipped,
            coast_gain=coast_gain,
            ref_torque=ref_torque,
            decel_context=decel_context,
        )
        torque_cmd = float(np.clip(torque_cmd_unclipped, -self.max_torque, self.max_torque))

        prev_residual = self.prev_residual
        prev_regen_gain = self.prev_regen_gain
        prev_coast_gain = self.prev_coast_gain
        self.vehicle, e_step, r_step, d_step = self.scenario.step_vehicle(
            self.vehicle, torque_cmd, regen_gain=regen_gain, step_idx=self.step_count
        )
        self.step_count += 1
        idx = min(self.step_count, self.horizon)

        ref_speed = float(self.reference["speed"][idx])
        ref_dist = float(self.reference["distance"][idx])
        ref_step_energy = float(self.reference["step_energy"][idx - 1])
        ref_step_recover = float(self.reference["step_recover"][idx - 1])
        ref_step_distance = float(self.reference["step_distance"][idx - 1])

        speed_err = self.vehicle["speed"] - ref_speed
        dist_err = self.vehicle["distance"] - ref_dist
        self.cum_speed_abs_err += abs(speed_err)
        self.cum_dist_abs_err += abs(dist_err)
        self.cum_energy += e_step
        self.cum_recover += r_step
        self.cum_distance += d_step
        self.cum_steps += 1

        if self.mode == "track":
            speed_soft = self.speed_soft_tol
            dist_soft = self.dist_soft_tol
            speed_w, dist_w = 4.0, 0.9
            alive_reward = 2.2
            bonus_speed, bonus_dist, bonus_val = 0.6, 1.8, 0.6
        else:
            speed_soft = 1.6
            dist_soft = 5.5
            speed_w, dist_w = 2.8, 0.4
            alive_reward = 1.6
            bonus_speed, bonus_dist, bonus_val = 0.9, 2.6, 0.35

        speed_pen_core = max(0.0, abs(speed_err) - self.speed_penalty_deadband_mps)
        speed_pen = (speed_pen_core / max(speed_soft, 1e-8)) ** 2
        dist_pen = (abs(dist_err) / dist_soft) ** 2
        track_penalty = speed_w * speed_pen + dist_w * dist_pen
        early_track_scale = 1.0

        tracking_margin = max(
            abs(speed_err) / max(self.constraint_speed_tol, 1e-8),
            abs(dist_err) / max(self.constraint_dist_tol, 1e-8),
        )
        energy_focus_gain = self._energy_focus_gain(tracking_margin)
        phase_energy_scale = 1.0
        effective_energy_weight = self.energy_weight * energy_focus_gain * phase_energy_scale
        lag_scale = 1.0
        ahead_scale = 1.0

        slow_bias_energy_debit = 0.0
        slow_bias_drag_debit = 0.0
        slow_bias_kinetic_debit = 0.0
        if self.mode == "energy":
            slow_bias_energy_debit, slow_bias_drag_debit, slow_bias_kinetic_debit = self._compute_slow_bias_energy_debit(
                ref_speed=ref_speed,
                agent_speed=self.vehicle["speed"],
                ref_torque=ref_torque,
                ref_step_energy=ref_step_energy,
                dist_err=dist_err,
                progress_ratio=progress_ratio,
                step_idx=idx,
                coast_gain=coast_gain,
                coast_allowed=coast_allowed,
                decel_context=decel_context,
            )
        accounted_e_step = e_step + slow_bias_energy_debit
        self.cum_energy_accounted += accounted_e_step
        self.cum_slow_bias_energy_debit += slow_bias_energy_debit
        self.window_agent_energy_hist.append(float(accounted_e_step))
        self.window_ref_energy_hist.append(float(ref_step_energy))
        self.window_step_distance_hist.append(float(d_step))
        self.window_ref_distance_hist.append(float(ref_step_distance))

        # 计算每一步相对固定驾驶员基线的效率收益。
        agent_epd_step = accounted_e_step / max(d_step, 1e-8)
        ref_epd_step = ref_step_energy / max(ref_step_distance, 1e-8)
        epd_improve_ratio = (ref_epd_step - agent_epd_step) / max(ref_epd_step, 1e-8)
        epd_improve_ratio = float(np.clip(epd_improve_ratio, -3.0, 3.0))
        step_energy_improve = (ref_step_energy - accounted_e_step) / max(ref_step_energy, 1e-8)
        step_energy_improve = float(np.clip(step_energy_improve, -2.0, 2.0))
        mixed_energy_improve = 0.75 * epd_improve_ratio + 0.25 * step_energy_improve
        rolling_window_saving = 0.0
        window_energy_constraint_cost = 0.0
        window_energy_reward = 0.0
        if len(self.window_agent_energy_hist) >= max(4, int(self.window_energy_horizon) // 2):
            rolling_ref_energy = float(np.sum(self.window_ref_energy_hist))
            rolling_agent_energy = float(np.sum(self.window_agent_energy_hist))
            rolling_ref_dist = float(np.sum(self.window_ref_distance_hist))
            rolling_agent_dist = float(np.sum(self.window_step_distance_hist))
            rolling_ref_epd = rolling_ref_energy / max(rolling_ref_dist, 1e-8)
            rolling_agent_epd = rolling_agent_energy / max(rolling_agent_dist, 1e-8)
            rolling_window_saving = float(
                np.clip((rolling_ref_epd - rolling_agent_epd) / max(rolling_ref_epd, 1e-8), -4.0, 4.0)
            )
            if self.mode == "energy":
                neg_window_excess = max(0.0, -rolling_window_saving - self.window_negative_saving_tol)
                if neg_window_excess > 1e-10:
                    self.window_negative_streak = min(self.window_negative_streak + 1, 6)
                else:
                    self.window_negative_streak = max(self.window_negative_streak - 1, 0)
                window_stage_gate = float(
                    np.clip(
                        (progress_ratio - self.window_negative_gate_progress)
                        / max(1.0 - self.window_negative_gate_progress, 1e-8),
                        0.0,
                        1.0,
                    )
                )
                streak_scale = 1.0 + self.window_negative_streak_gain * max(0, self.window_negative_streak - 1)
                window_energy_constraint_cost = (
                    self.window_constraint_scale
                    *
                    window_stage_gate
                    * streak_scale
                    * neg_window_excess
                    / max(self.window_negative_saving_tol, 1e-8)
                )
                window_energy_constraint_cost = float(
                    np.clip(window_energy_constraint_cost, 0.0, self.constraint_cost_clip)
                )
                pos_window_gain = max(0.0, rolling_window_saving)
                early_oversave_gate = float(
                    np.clip(
                        (self.window_oversave_gate_end - progress_ratio)
                        / max(self.window_oversave_gate_end, 1e-8),
                        0.0,
                        1.0,
                    )
                )
                late_window_gate = float(
                    np.clip(
                        (progress_ratio - 0.45) / 0.55,
                        0.0,
                        1.0,
                    )
                )
                oversave_excess = max(0.0, pos_window_gain - self.window_positive_saving_cap)
                window_energy_reward = window_stage_gate * (
                    self.window_positive_reward_coef * pos_window_gain
                    - self.window_negative_reward_penalty_coef * streak_scale * neg_window_excess
                )
                window_energy_reward += (
                    self.window_late_positive_bonus_coef * late_window_gate * pos_window_gain
                    - self.window_oversave_penalty_coef * early_oversave_gate * oversave_excess
                )
        lag_ratio = max(
            max(0.0, -speed_err) / max(self.constraint_speed_tol, 1e-8),
            max(0.0, -dist_err) / max(self.constraint_dist_tol, 1e-8),
        )
        if self.mode == "energy":
            energy_lag_gate = max(0.0, 1.0 - self.energy_lag_gate_gain * min(lag_ratio, 1.4))
        else:
            energy_lag_gate = 1.0
        positive_energy_gain = max(0.0, mixed_energy_improve)
        negative_energy_gain = max(0.0, -mixed_energy_improve)
        energy_term = (
            effective_energy_weight
            * energy_lag_gate
            * (
                self.energy_positive_reward_coef * positive_energy_gain
                - self.energy_negative_penalty_coef * negative_energy_gain
            )
        )
        if self.mode == "energy":
            neg_coef = self.energy_neg_coef_good if tracking_margin <= 1.0 else self.energy_neg_coef_bad
            energy_term -= (
                (neg_coef - 1.0)
                * self.energy_negative_penalty_coef
                * effective_energy_weight
                * negative_energy_gain
            )

        smooth_coeff, residual_coeff, regen_smooth_coeff, coast_smooth_coeff = self._smooth_coeffs(tracking_margin)
        smooth_penalty = smooth_coeff * abs(residual - prev_residual)
        residual_penalty = residual_coeff * abs(residual)
        regen_smooth_penalty = regen_smooth_coeff * abs(regen_gain - prev_regen_gain)
        coast_smooth_penalty = coast_smooth_coeff * abs(coast_gain - prev_coast_gain)

        track_gate = 0.0
        if self.mode == "energy":
            track_gate = max(0.0, 1.0 - min(tracking_margin, 1.5) / 1.5)
            torque_eff_gain = (abs(ref_torque) - abs(self.vehicle["torque"])) / max(self.max_torque, 1e-8)
            torque_eff_gain = float(np.clip(torque_eff_gain, -0.4, 0.4))
            ref_positive_torque_ratio = max(ref_torque, 0.0) / max(self.max_torque, 1e-8)
            launch_phase_gate = float(
                np.clip(
                    (self.launch_progress_end - progress_ratio) / max(self.launch_progress_end, 1e-8),
                    0.0,
                    1.0,
                )
            )
            accel_phase_gate = float(
                np.clip(
                    (ref_positive_torque_ratio - self.phase_accel_torque_threshold_ratio)
                    / max(1.0 - self.phase_accel_torque_threshold_ratio, 1e-8),
                    0.0,
                    1.0,
                )
            ) * (1.0 - launch_phase_gate)
            decel_phase_gate = 1.0 if decel_context else 0.0
            cruise_phase_gate = (1.0 - decel_phase_gate) * (1.0 - launch_phase_gate) * float(
                np.clip(
                    (self.phase_accel_torque_threshold_ratio - ref_positive_torque_ratio)
                    / max(self.phase_accel_torque_threshold_ratio, 1e-8),
                    0.0,
                    1.0,
                )
            )
            efficiency_phase_scale = (
                self.efficiency_cruise_reward_scale * cruise_phase_gate
                + self.efficiency_accel_reward_scale * accel_phase_gate
                + self.efficiency_decel_reward_scale * decel_phase_gate
            )
            efficiency_term = (
                self.torque_eff_coef
                * effective_energy_weight
                * track_gate
                * efficiency_phase_scale
                * torque_eff_gain
            )
            ref_torque_idx = float(self.reference["torque_cmd"][min(idx, self.horizon - 1)])
            ref_rpm_idx = ref_speed * self.scenario.rpm_per_mps
            agent_eta = float(self.scenario.efficiency(self.vehicle["torque"], self.vehicle["rpm"]))
            ref_eta = float(self.scenario.efficiency(ref_torque_idx, ref_rpm_idx))
            eta_delta = float(np.clip(agent_eta - ref_eta, -0.16, 0.16))
            efficiency_direct_reward = (
                self.efficiency_direct_reward_coef
                * track_gate
                * efficiency_phase_scale
                * eta_delta
            )
            eff_load_gate = float(
                np.clip(
                    (
                        max(abs(ref_torque_idx), abs(self.vehicle["torque"]))
                        - self.efficiency_load_gate_floor_ratio * self.max_torque
                    )
                    / max((0.35 - self.efficiency_load_gate_floor_ratio) * self.max_torque, 1e-8),
                    0.0,
                    1.0,
                )
            )
            eff_gain_coef = self.eff_map_gain_coef
            eff_penalty_coef = self.eff_map_penalty_coef
            if is_normal_style:
                # 普通风格特化：更偏向放大正向效率提升。
                eff_gain_coef *= 1.18
                eff_penalty_coef *= 0.88
            elif is_sport_style:
                eff_gain_coef *= 1.05
                eff_penalty_coef *= 1.15
            efficiency_term += (
                eff_gain_coef
                * effective_energy_weight
                * track_gate
                * efficiency_phase_scale
                * eff_load_gate
                * max(0.0, eta_delta)
            )
            efficiency_term -= (
                eff_penalty_coef
                * effective_energy_weight
                * track_gate
                * efficiency_phase_scale
                * eff_load_gate
                * max(0.0, -eta_delta)
            )
            regen_target = 1.0
            regen_recover_gain = 0.0
            if decel_context:
                decel_severity = float(np.clip(max(0.0, -ref_torque) / max(self.max_torque, 1e-8), 0.0, 1.0))
                regen_target = self.regen_target_base + self.regen_target_gain * decel_severity
                regen_match = 1.0 - min(abs(regen_gain - regen_target) / self.regen_target_tol, 1.0)
                efficiency_term += self.regen_eff_coef * effective_energy_weight * track_gate * regen_match
                regen_recover_gain = (r_step - ref_step_recover) / max(ref_step_recover, 1e-8)
                regen_recover_gain = float(np.clip(regen_recover_gain, -1.2, 1.2))
                if tracking_margin <= 1.0:
                    regen_recover_coef = self.regen_recover_coef
                    if is_normal_style:
                        regen_recover_coef = max(regen_recover_coef, 0.05)
                    efficiency_term += regen_recover_coef * effective_energy_weight * track_gate * regen_recover_gain
        else:
            efficiency_term = 0.0
            efficiency_direct_reward = 0.0
            agent_eta = 0.0
            ref_eta = 0.0
            eta_delta = 0.0
            eff_load_gate = 0.0
            regen_target = 1.0
            regen_recover_gain = 0.0
        follow_penalty = 0.0
        launch_negative_torque_penalty = 0.0
        if self.mode == "energy":
            follow_coef = self.driver_follow_coef
            if is_normal_style:
                if (tracking_margin <= 0.85) and (ref_torque > 5.0) and (not decel_context):
                    follow_coef *= 0.72
                elif decel_context:
                    follow_coef *= 0.85
            elif is_sport_style:
                if (tracking_margin <= 0.90) and (ref_torque > 8.0) and (not decel_context):
                    follow_coef *= 1.18
                elif decel_context:
                    follow_coef *= 1.05
            follow_penalty = follow_coef * abs(torque_cmd - ref_torque) / max(self.max_torque, 1e-8)
            launch_phase_gate = float(
                np.clip(
                    (self.launch_progress_end - progress_ratio) / max(self.launch_progress_end, 1e-8),
                    0.0,
                    1.0,
                )
            )
            launch_drive_gate = float(
                (not decel_context)
                and (not coast_allowed)
                and (ref_torque > 0.10 * self.max_torque)
            )
            launch_negative_gap = max(
                0.0,
                ref_torque - torque_cmd - self.launch_negative_torque_deadband_ratio * self.max_torque,
            ) / max(self.max_torque, 1e-8)
            if launch_phase_gate > 0.0 and launch_drive_gate > 0.5 and launch_negative_gap > 1e-10:
                self.launch_negative_torque_streak = min(self.launch_negative_torque_streak + 1, 8)
            else:
                self.launch_negative_torque_streak = max(self.launch_negative_torque_streak - 1, 0)
            launch_negative_streak_scale = (
                1.0 + self.launch_negative_torque_streak_gain * max(0, self.launch_negative_torque_streak - 1)
            )
            launch_negative_torque_penalty = (
                self.launch_negative_torque_penalty_coef
                * self.launch_negative_torque_phase_scale
                * launch_phase_gate
                * launch_drive_gate
                * launch_negative_streak_scale
                * launch_negative_gap
            )
        else:
            self.launch_negative_torque_streak = 0
        coast_bonus = 0.0
        if self.mode == "energy" and coast_allowed:
            torque_load = min(max(ref_torque, 0.0) / max(self.max_torque, 1e-8), 0.55)
            coast_bonus = self.coast_bonus_coef * effective_energy_weight * track_gate * coast_gain * torque_load

        speed_violation = max(0.0, abs(speed_err) - self.constraint_speed_tol)
        dist_violation = max(0.0, abs(dist_err) - self.constraint_dist_tol)
        self.segment_speed_err_hist.append(float(speed_err))
        self.segment_dist_err_hist.append(float(dist_err))
        neg_speed_bias = 0.0
        neg_dist_bias = 0.0
        lag_norm_speed = 0.0
        lag_norm_dist = 0.0
        underspeed_constraint_cost = 0.0
        isochronous_penalty = 0.0
        isochronous_constraint_cost = 0.0
        window_energy_constraint_cost = float(np.clip(window_energy_constraint_cost, 0.0, self.constraint_cost_clip))
        segment_speed_constraint_cost = 0.0
        segment_dist_constraint_cost = 0.0
        rel_speed_bias = 0.0
        rel_dist_bias = 0.0
        segment_speed_abs_mean = 0.0
        segment_dist_abs_mean = 0.0
        if len(self.segment_speed_err_hist) >= max(4, int(self.segment_tracking_horizon) // 2):
            segment_speed_arr = np.asarray(self.segment_speed_err_hist, dtype=np.float32)
            segment_dist_arr = np.asarray(self.segment_dist_err_hist, dtype=np.float32)
            segment_speed_abs_mean = float(np.mean(np.abs(segment_speed_arr)))
            segment_dist_abs_mean = float(np.mean(np.abs(segment_dist_arr)))
            segment_speed_bias_mean = float(-np.mean(segment_speed_arr))
            segment_dist_bias_mean = float(-np.mean(segment_dist_arr))
            segment_speed_abs_excess = max(
                0.0,
                segment_speed_abs_mean - self.segment_speed_abs_tol_ratio * self.constraint_speed_tol,
            )
            segment_dist_abs_excess = max(
                0.0,
                segment_dist_abs_mean - self.segment_dist_abs_tol_ratio * self.constraint_dist_tol,
            )
            segment_speed_bias_excess = max(
                0.0,
                segment_speed_bias_mean - self.segment_speed_bias_tol_ratio * self.constraint_speed_tol,
            )
            segment_dist_bias_excess = max(
                0.0,
                segment_dist_bias_mean - self.segment_dist_bias_tol_ratio * self.constraint_dist_tol,
            )
            segment_tracking_excess = max(
                segment_speed_abs_excess / max(self.constraint_speed_tol, 1e-8),
                segment_dist_abs_excess / max(self.constraint_dist_tol, 1e-8),
                0.75 * segment_speed_bias_excess / max(self.constraint_speed_tol, 1e-8),
                0.75 * segment_dist_bias_excess / max(self.constraint_dist_tol, 1e-8),
            )
            if self.mode == "energy" and segment_tracking_excess > 1e-10:
                self.segment_tracking_streak = min(self.segment_tracking_streak + 1, 8)
            else:
                self.segment_tracking_streak = max(self.segment_tracking_streak - 1, 0)
            segment_tracking_gate = float(
                np.clip(
                    (progress_ratio - self.segment_tracking_gate_progress)
                    / max(1.0 - self.segment_tracking_gate_progress, 1e-8),
                    0.0,
                    1.0,
                )
            )
            segment_streak_scale = (
                1.0 + self.segment_tracking_streak_gain * max(0, self.segment_tracking_streak - 1)
            )
            segment_speed_constraint_cost = (
                self.segment_tracking_constraint_scale
                * segment_tracking_gate
                * segment_streak_scale
                * (
                    segment_speed_abs_excess / max(self.constraint_speed_tol, 1e-8)
                    + 0.55 * segment_speed_bias_excess / max(self.constraint_speed_tol, 1e-8)
                )
            )
            segment_dist_constraint_cost = (
                self.segment_tracking_constraint_scale
                * segment_tracking_gate
                * segment_streak_scale
                * (
                    segment_dist_abs_excess / max(self.constraint_dist_tol, 1e-8)
                    + 0.45 * segment_dist_bias_excess / max(self.constraint_dist_tol, 1e-8)
                )
            )
        else:
            self.segment_tracking_streak = max(self.segment_tracking_streak - 1, 0)
        if self.mode == "energy":
            lag_norm_speed = max(0.0, -speed_err) / max(self.constraint_speed_tol, 1e-8)
            lag_norm_dist = max(0.0, -dist_err) / max(self.constraint_dist_tol, 1e-8)
            lag_penalty = (
                self.lag_penalty_speed_coef * lag_norm_speed
                + self.lag_penalty_dist_coef * lag_norm_dist
            )
            lag_penalty += self.lag_hard_coef * (
                max(0.0, lag_norm_speed - 0.80) ** 2
                + 0.60 * max(0.0, lag_norm_dist - 0.80) ** 2
            )
            neg_speed_bias = max(0.0, -(speed_err + self.neg_bias_speed_deadband_mps))
            neg_dist_bias = max(0.0, -(dist_err + self.neg_bias_dist_deadband_m))
            neg_bias_penalty = self.neg_bias_penalty_coef * (
                (neg_speed_bias / max(self.constraint_speed_tol, 1e-8))
                + 0.45 * (neg_dist_bias / max(self.constraint_dist_tol, 1e-8))
            )
            # 在轨迹后段施加更强的抗慢开压力。
            bias_progress_scale = float(np.clip((progress_ratio - 0.55) / 0.45, 0.0, 1.0))
            neg_bias_penalty *= bias_progress_scale
            rel_speed_bias = speed_err / max(ref_speed, 1e-8)
            rel_dist_bias = dist_err / max(ref_dist, 1.0)
            speed_debt_in = self.isochronous_debt_speed_gain * neg_speed_bias
            dist_debt_in = self.isochronous_debt_dist_gain * neg_dist_bias
            speed_debt_release = self.isochronous_release_speed_gain * max(
                0.0,
                speed_err - self.isochronous_speed_release_deadband_mps,
            )
            dist_debt_release = self.isochronous_release_dist_gain * max(
                0.0,
                dist_err - self.isochronous_dist_release_deadband_m,
            )
            self.isochronous_speed_debt = float(
                np.clip(
                    self.isochronous_speed_debt + speed_debt_in - speed_debt_release,
                    0.0,
                    self.isochronous_debt_clip,
                )
            )
            self.isochronous_dist_debt = float(
                np.clip(
                    self.isochronous_dist_debt + dist_debt_in - dist_debt_release,
                    0.0,
                    self.isochronous_debt_clip,
                )
            )
            underspeed_rel_speed = max(
                0.0,
                -rel_speed_bias - self.constraint_underspeed_rel_speed_tol,
            )
            underspeed_rel_dist = max(
                0.0,
                -rel_dist_bias - self.constraint_underspeed_rel_dist_tol,
            )
            underspeed_stage_gate = float(np.clip((progress_ratio - 0.10) / 0.90, 0.0, 1.0))
            debt_stage_gate = float(np.clip((progress_ratio - 0.12) / 0.88, 0.0, 1.0))
            late_stage_gate = float(np.clip((progress_ratio - 0.45) / 0.55, 0.0, 1.0))
            isochronous_speed_norm = self.isochronous_speed_debt / max(self.constraint_speed_tol, 1e-8)
            isochronous_dist_norm = self.isochronous_dist_debt / max(self.constraint_dist_tol, 1e-8)
            isochronous_penalty = self.isochronous_penalty_coef * debt_stage_gate * (
                0.85 * isochronous_speed_norm
                + 1.20 * isochronous_dist_norm
                + late_stage_gate
                * (
                    underspeed_rel_speed / max(self.constraint_underspeed_rel_speed_tol, 1e-8)
                    + 1.10 * underspeed_rel_dist / max(self.constraint_underspeed_rel_dist_tol, 1e-8)
                )
            )
            underspeed_constraint_cost = underspeed_stage_gate * (
                underspeed_rel_speed / max(self.constraint_underspeed_rel_speed_tol, 1e-8)
                + 0.65 * underspeed_rel_dist / max(self.constraint_underspeed_rel_dist_tol, 1e-8)
            )
            isochronous_constraint_cost = self.isochronous_cost_scale * debt_stage_gate * (
                0.75 * isochronous_speed_norm
                + 1.25 * isochronous_dist_norm
                + 0.60
                * late_stage_gate
                * (
                    underspeed_rel_speed / max(self.constraint_underspeed_rel_speed_tol, 1e-8)
                    + 1.15 * underspeed_rel_dist / max(self.constraint_underspeed_rel_dist_tol, 1e-8)
                )
            )
            underspeed_constraint_cost += 0.20 * isochronous_constraint_cost
            underspeed_constraint_cost += (
                self.launch_negative_torque_constraint_scale * launch_negative_torque_penalty
            )
        else:
            lag_penalty = 0.0
            neg_bias_penalty = 0.0

        speed_constraint_cost = (
            speed_violation / max(self.constraint_speed_tol, 1e-8)
            + 0.50 * neg_speed_bias / max(self.constraint_speed_tol, 1e-8)
            + 0.35 * max(0.0, lag_norm_speed - 0.80) ** 2
            + segment_speed_constraint_cost
        )
        dist_constraint_cost = (
            dist_violation / max(self.constraint_dist_tol, 1e-8)
            + 0.35 * neg_dist_bias / max(self.constraint_dist_tol, 1e-8)
            + 0.25 * max(0.0, lag_norm_dist - 0.80) ** 2
            + segment_dist_constraint_cost
        )
        smooth_constraint_cost = (
            max(0.0, abs(residual - prev_residual) / max(self.constraint_residual_delta_tol, 1e-8) - 1.0)
            + 0.35 * max(0.0, abs(residual) / max(self.constraint_residual_abs_tol, 1e-8) - 1.0)
            + 0.40 * max(0.0, abs(regen_gain - prev_regen_gain) / max(self.constraint_regen_delta_tol, 1e-8) - 1.0)
            + 0.20 * max(0.0, abs(coast_gain - prev_coast_gain) / max(self.constraint_coast_delta_tol, 1e-8) - 1.0)
        )
        agent_net_epd_step = (accounted_e_step - r_step) / max(d_step, 1e-8)
        ref_net_epd_step = (ref_step_energy - ref_step_recover) / max(ref_step_distance, 1e-8)
        net_energy_gap = (agent_net_epd_step - ref_net_epd_step) / max(abs(ref_net_epd_step), 1e-8)
        net_energy_constraint_cost = max(0.0, net_energy_gap - self.constraint_net_energy_margin)
        speed_constraint_cost = float(np.clip(speed_constraint_cost, 0.0, self.constraint_cost_clip))
        dist_constraint_cost = float(np.clip(dist_constraint_cost, 0.0, self.constraint_cost_clip))
        smooth_constraint_cost = float(np.clip(smooth_constraint_cost, 0.0, self.constraint_cost_clip))
        net_energy_constraint_cost = float(np.clip(net_energy_constraint_cost, 0.0, self.constraint_cost_clip))
        underspeed_constraint_cost = float(np.clip(underspeed_constraint_cost, 0.0, self.constraint_cost_clip))

        if self.mode == "track":
            base_reward = (
                alive_reward
                - track_penalty
                - smooth_penalty
                - residual_penalty
                - regen_smooth_penalty
                - coast_smooth_penalty
            )
            if abs(speed_err) < bonus_speed and abs(dist_err) < bonus_dist:
                base_reward += bonus_val
        else:
            base_reward = float(
                energy_term
                + efficiency_term
                + efficiency_direct_reward
                + window_energy_reward
                + coast_bonus
                - follow_penalty
                - launch_negative_torque_penalty
                - lag_penalty
            )

        reward = float(base_reward)

        terminated = False
        if abs(speed_err) > self.speed_fail_tol or abs(dist_err) > self.dist_fail_tol:
            terminated = True
            speed_constraint_cost = float(
                min(
                    self.constraint_cost_clip,
                    speed_constraint_cost + max(0.0, abs(speed_err) - self.speed_fail_tol) / max(self.constraint_speed_tol, 1e-8) + 2.0,
                )
            )
            dist_constraint_cost = float(
                min(
                    self.constraint_cost_clip,
                    dist_constraint_cost + max(0.0, abs(dist_err) - self.dist_fail_tol) / max(self.constraint_dist_tol, 1e-8) + 2.0,
                )
            )
            if self.mode == "track":
                reward -= 120.0

        if self.vehicle["soc"] <= self.scenario.min_soc * 0.95:
            terminated = True
            net_energy_constraint_cost = float(min(self.constraint_cost_clip, net_energy_constraint_cost + 2.0))
            if self.mode == "track":
                reward -= 80.0

        truncated = self.step_count >= self.horizon
        if truncated and (not terminated) and self.mode == "energy" and self.cum_steps > 0:
            epd_agent_gross = self.cum_energy_accounted / max(self.cum_distance, 1e-8)
            epd_ref_gross = float(self.reference["energy_per_dist"])
            final_epd_improve = (epd_ref_gross - epd_agent_gross) / max(epd_ref_gross, 1e-8)
            final_epd_improve = float(np.clip(final_epd_improve, -0.6, 0.6))
            epd_agent_net = (self.cum_energy_accounted - self.cum_recover) / max(self.cum_distance, 1e-8)
            epd_ref_net = float(self.reference["net_energy"]) / max(float(self.reference["total_distance"]), 1e-8)
            final_net_epd_improve = (epd_ref_net - epd_agent_net) / max(epd_ref_net, 1e-8)
            final_net_epd_improve = float(np.clip(final_net_epd_improve, -0.6, 0.6))
            final_energy_improve = 0.70 * final_epd_improve + 0.30 * final_net_epd_improve
            mean_speed_err = self.cum_speed_abs_err / max(self.cum_steps, 1)
            mean_dist_err = self.cum_dist_abs_err / max(self.cum_steps, 1)
            final_lag_penalty = self.final_lag_penalty_coef * (
                max(0.0, -speed_err) / max(self.constraint_speed_tol, 1e-8)
                + max(0.0, -dist_err) / max(self.constraint_dist_tol, 1e-8)
            )
            final_isochronous_penalty = self.final_isochronous_penalty_coef * (
                self.isochronous_speed_debt / max(self.constraint_speed_tol, 1e-8)
                + 1.15 * self.isochronous_dist_debt / max(self.constraint_dist_tol, 1e-8)
                + max(0.0, -speed_err) / max(self.constraint_speed_tol, 1e-8)
                + 1.15 * max(0.0, -dist_err) / max(self.constraint_dist_tol, 1e-8)
            )
            reward -= final_lag_penalty
            if mean_speed_err <= self.constraint_speed_tol and mean_dist_err <= self.constraint_dist_tol:
                reward += self.energy_weight * self.final_energy_pos_coef * final_energy_improve
            elif final_energy_improve < 0.0:
                reward += self.energy_weight * self.final_energy_neg_coef * final_energy_improve
            base_reward = float(reward)

        self.prev_residual = residual
        self.prev_regen_gain = regen_gain
        self.prev_coast_gain = coast_gain
        executed_action = np.array(
            [
                float(residual),
                self._regen_gain_to_action(regen_gain),
                self._coast_gain_to_action(coast_gain),
            ],
            dtype=np.float32,
        )
        executed_raw_action = self._action_to_raw(executed_action)
        action_projection_delta = executed_action - proposed_action
        projection_weights = np.array(
            [
                1.0,
                1.0 if decel_context else 0.10,
                1.0 if (self.mode == "energy" and (not decel_context) and ref_torque > 0.0) else 0.12,
            ],
            dtype=np.float32,
        )
        projection_weight_sum = float(max(np.sum(projection_weights), 1e-8))
        weighted_projection_delta = action_projection_delta * projection_weights
        action_projection_l1 = float(np.sum(np.abs(weighted_projection_delta)) / projection_weight_sum)
        action_projection_l2 = float(np.linalg.norm(weighted_projection_delta) / np.sqrt(projection_weight_sum))
        projection_constraint_cost = (
            max(0.0, action_projection_l1 / max(self.constraint_projection_l1_tol, 1e-8) - 1.0)
            + 0.50 * max(0.0, action_projection_l2 / max(self.constraint_projection_l2_tol, 1e-8) - 1.0)
        )
        projection_constraint_cost = float(np.clip(projection_constraint_cost, 0.0, self.constraint_cost_clip))

        info = {
            "step_energy": e_step,
            "step_energy_accounted": accounted_e_step,
            "step_recover": r_step,
            "step_distance": d_step,
            "speed_error": speed_err,
            "distance_error": dist_err,
            "ref_speed": ref_speed,
            "ref_distance": ref_dist,
            "ref_torque": ref_torque,
            "torque_cmd": torque_cmd,
            "residual": residual,
            "regen_gain": regen_gain,
            "coast_gain": coast_gain,
            "dynamic_limit": dynamic_limit,
            "residual_pos_limit": pos_limit,
            "residual_neg_limit": neg_limit,
            "ref_step_energy": ref_step_energy,
            "ref_step_recover": ref_step_recover,
            "ref_step_distance": ref_step_distance,
            "epd_improve_ratio": epd_improve_ratio,
            "step_energy_improve": step_energy_improve,
            "mixed_energy_improve": mixed_energy_improve,
            "rolling_window_saving": rolling_window_saving,
            "window_negative_streak": int(self.window_negative_streak),
            "window_energy_reward": float(window_energy_reward),
            "slow_bias_energy_debit": slow_bias_energy_debit,
            "slow_bias_drag_debit": slow_bias_drag_debit,
            "slow_bias_kinetic_debit": slow_bias_kinetic_debit,
            "cum_slow_bias_energy_debit": self.cum_slow_bias_energy_debit,
            "effective_energy_weight": effective_energy_weight,
            "phase_energy_scale": float(phase_energy_scale),
            "early_track_scale": float(early_track_scale),
            "energy_lag_gate": energy_lag_gate,
            "speed_penalty_deadband_mps": self.speed_penalty_deadband_mps,
            "lag_scale": lag_scale,
            "ahead_scale": ahead_scale,
            "efficiency_term": efficiency_term,
            "efficiency_direct_reward": efficiency_direct_reward,
            "isochronous_penalty": isochronous_penalty,
            "isochronous_constraint_cost": isochronous_constraint_cost,
            "isochronous_speed_debt": self.isochronous_speed_debt,
            "isochronous_dist_debt": self.isochronous_dist_debt,
            "regen_target": regen_target,
            "regen_recover_gain": regen_recover_gain,
            "drift_corr_torque": drift_corr_torque,
            "bias_corr_torque": bias_corr_torque,
            "bias_neutral_torque": bias_neutral_torque,
            "launch_catchup_torque": launch_catchup_torque,
            "dist_catchup_torque": dist_catchup_torque,
            "preview_corr_torque": preview_corr_torque,
            "dist_err_int": self.dist_err_int,
            "lag_penalty": lag_penalty,
            "neg_bias_penalty": neg_bias_penalty,
            "follow_penalty": follow_penalty,
            "launch_negative_torque_penalty": float(launch_negative_torque_penalty),
            "launch_negative_torque_streak": int(self.launch_negative_torque_streak),
            "segment_speed_constraint_cost": float(segment_speed_constraint_cost),
            "segment_dist_constraint_cost": float(segment_dist_constraint_cost),
            "segment_speed_abs_mean": float(segment_speed_abs_mean),
            "segment_dist_abs_mean": float(segment_dist_abs_mean),
            "segment_tracking_streak": int(self.segment_tracking_streak),
            "coast_bonus": coast_bonus,
            "coast_smooth_penalty": coast_smooth_penalty,
            "agent_eta": agent_eta,
            "ref_eta": ref_eta,
            "eta_delta": eta_delta,
            "eff_load_gate": eff_load_gate,
            "speed_violation": speed_violation,
            "dist_violation": dist_violation,
            "proposed_action": proposed_action.astype(np.float32, copy=False),
            "proposed_action_bound_violation": proposed_action_violation.astype(np.float32, copy=False),
            "proposed_action_bound_violation_linf": proposed_action_violation_linf,
            "executed_action": executed_action,
            "executed_raw_action": executed_raw_action,
            "action_projection_delta": action_projection_delta.astype(np.float32, copy=False),
            "action_projection_weighted_delta": weighted_projection_delta.astype(np.float32, copy=False),
            "action_projection_weights": projection_weights,
            "action_projection_l1": action_projection_l1,
            "action_projection_l2": action_projection_l2,
            "base_reward": float(base_reward),
            "objective_reward": float(base_reward),
            "constraint_speed_cost": speed_constraint_cost,
            "constraint_distance_cost": dist_constraint_cost,
            "constraint_smoothness_cost": smooth_constraint_cost,
            "constraint_net_energy_cost": net_energy_constraint_cost,
            "constraint_projection_cost": projection_constraint_cost,
            "constraint_underspeed_cost": underspeed_constraint_cost,
            "constraint_window_energy_cost": window_energy_constraint_cost,
            "constraint_lambdas": self.get_constraint_multipliers(),
            "agent_net_epd_step": float(agent_net_epd_step),
            "ref_net_epd_step": float(ref_net_epd_step),
            "net_energy_gap": float(net_energy_gap),
            "speed_rel_bias_step": float(rel_speed_bias * 100.0),
            "dist_rel_bias_step": float(rel_dist_bias * 100.0),
            "final_lag_penalty": float(final_lag_penalty) if truncated and (not terminated) and self.mode == "energy" and self.cum_steps > 0 else 0.0,
            "final_isochronous_penalty": float(final_isochronous_penalty) if truncated and (not terminated) and self.mode == "energy" and self.cum_steps > 0 else 0.0,
        }
        next_obs = self._stack_observation(self._encode_state(idx), reset_history=False)
        info["base_state_dim"] = int(self.base_state_dim)
        info["obs_stack"] = int(self.obs_stack)
        return next_obs, reward, terminated, truncated, info


