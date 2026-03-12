import copy
import numpy as np

from motor_agent import make_pbar


def evaluate_rollout(
    rollout,
    eval_ref,
    scenario=None,
    speed_mae_limit=0.9,
    dist_mae_limit=4.8,
    speed_rel_limit=2.0,
    dist_rel_limit=2.0,
    max_negative_speed_bias_pct=0.8,
    max_negative_dist_bias_pct=0.8,
    smooth_delta_ratio_limit=1.12,
    smooth_jerk_ratio_limit=1.18,
):
    """
    作用：在跟踪和平滑约束下评估单条 rollout 相对驾驶员参考的表现。
    输入：rollout: 智能体轨迹；eval_ref: 驾驶员参考；其余参数为评估阈值。
    输出：包含节能、跟踪、平滑等指标的字典。
    """
    rollout_total_energy = float(rollout.get("total_energy_accounted", rollout["total_energy"]))
    rollout_energy_per_dist = float(rollout.get("energy_per_dist_accounted", rollout["energy_per_dist"]))
    rollout_net_energy_per_dist = float(rollout.get("net_energy_per_dist_accounted", rollout["net_energy_per_dist"]))
    rollout_total_energy_raw = float(rollout.get("total_energy_raw", rollout_total_energy))
    rollout_energy_per_dist_raw = float(rollout.get("energy_per_dist_raw", rollout_energy_per_dist))
    rollout_net_energy_per_dist_raw = float(rollout.get("net_energy_per_dist_raw", rollout_net_energy_per_dist))
    saving_total_pct = (eval_ref["total_energy"] - rollout_total_energy) / max(eval_ref["total_energy"], 1e-12) * 100.0
    saving_total_raw_pct = (eval_ref["total_energy"] - rollout_total_energy_raw) / max(eval_ref["total_energy"], 1e-12) * 100.0
    saving_epd_pct = (eval_ref["energy_per_dist"] - rollout_energy_per_dist) / max(eval_ref["energy_per_dist"], 1e-12) * 100.0
    saving_epd_raw_pct = (eval_ref["energy_per_dist"] - rollout_energy_per_dist_raw) / max(eval_ref["energy_per_dist"], 1e-12) * 100.0
    driver_net_epd = float(eval_ref["net_energy"]) / max(float(eval_ref["total_distance"]), 1e-8)
    saving_net_epd_pct = (driver_net_epd - rollout_net_energy_per_dist) / max(driver_net_epd, 1e-12) * 100.0
    saving_net_epd_raw_pct = (driver_net_epd - rollout_net_energy_per_dist_raw) / max(driver_net_epd, 1e-12) * 100.0
    low_speed_benefit_total_pct = saving_total_raw_pct - saving_total_pct
    low_speed_benefit_epd_pct = saving_epd_raw_pct - saving_epd_pct
    low_speed_benefit_net_epd_pct = saving_net_epd_raw_pct - saving_net_epd_pct
    recover_delta_pct = (
        (float(rollout["total_recover"]) - float(eval_ref["total_recover"]))
        / max(float(eval_ref["total_recover"]), 1e-12)
        * 100.0
    )
    recover_ratio_pct = float(rollout["total_recover"]) / max(float(eval_ref["total_recover"]), 1e-12) * 100.0

    eval_avg_speed = float(np.mean(eval_ref["speed"][1:]))
    eval_total_distance = float(eval_ref["distance"][-1])
    rollout_total_distance = float(rollout["total_distance"])
    launch_horizon = int(rollout.get("launch_metric_horizon", 20))
    signed_speed_diff = float(rollout["avg_speed"] - eval_avg_speed)
    signed_dist_diff = float(rollout["final_distance"] - eval_total_distance)
    avg_speed_diff = abs(signed_speed_diff)
    final_dist_diff = abs(signed_dist_diff)
    speed_rel_diff_pct = avg_speed_diff / max(eval_avg_speed, 1e-8) * 100.0
    dist_rel_diff_pct = final_dist_diff / max(eval_total_distance, 1e-8) * 100.0
    speed_rel_bias_pct = signed_speed_diff / max(eval_avg_speed, 1e-8) * 100.0
    dist_rel_bias_pct = signed_dist_diff / max(eval_total_distance, 1e-8) * 100.0
    launch_speed_bias = float(rollout.get("launch_speed_bias_mean", 0.0))
    launch_dist_bias = float(rollout.get("launch_dist_bias_mean", 0.0))
    launch_ref_speed_mean = float(rollout.get("launch_ref_speed_mean", 0.0))
    launch_ref_distance_mean = float(rollout.get("launch_ref_distance_mean", 0.0))
    launch_speed_rel_bias_pct = launch_speed_bias / max(launch_ref_speed_mean, 1e-8) * 100.0
    launch_dist_rel_bias_pct = launch_dist_bias / max(eval_total_distance, 1e-8) * 100.0
    distance_deficit_m = max(0.0, eval_total_distance - rollout_total_distance)
    distance_comp_energy = distance_deficit_m * float(eval_ref["energy_per_dist"])
    distance_comp_net_energy = distance_deficit_m * driver_net_epd
    ref_final_speed = float(eval_ref["speed"][-1])
    agent_final_speed = float(rollout["speed"][-1])
    speed_deficit_terminal = max(0.0, ref_final_speed - agent_final_speed)
    kinetic_gap = max(0.0, 0.5 * (ref_final_speed * ref_final_speed - agent_final_speed * agent_final_speed))
    eta_motor = 0.85
    if scenario is not None:
        ref_terminal_torque = 0.0
        if "torque" in eval_ref and len(eval_ref["torque"]) > 0:
            ref_terminal_torque = float(eval_ref["torque"][-1])
        elif "torque_cmd" in eval_ref and len(eval_ref["torque_cmd"]) > 0:
            ref_terminal_torque = float(eval_ref["torque_cmd"][-1])
        avg_speed = max(0.5 * (ref_final_speed + agent_final_speed), 0.1)
        kinetic_rpm = avg_speed * float(getattr(scenario, "rpm_per_mps", 430.0))
        kinetic_torque_proxy = kinetic_gap / max(
            float(getattr(scenario, "torque_to_acc", 0.042)) * max(float(getattr(scenario, "dt", 0.1)), 1e-8),
            1e-8,
        )
        eta_motor = max(
            float(scenario.efficiency(max(abs(ref_terminal_torque), kinetic_torque_proxy), kinetic_rpm)),
            1e-4,
        )
        kinetic_comp_energy = kinetic_gap * float(getattr(scenario, "base_soc_factor", 0.0009)) / eta_motor
    else:
        kinetic_comp_energy = kinetic_gap * 0.0009 / eta_motor
    rollout_total_energy_iso = rollout_total_energy + distance_comp_energy + kinetic_comp_energy
    rollout_net_energy_iso = float(rollout.get("net_energy_accounted", rollout["net_energy"])) + distance_comp_net_energy + kinetic_comp_energy
    rollout_iso_distance = max(eval_total_distance, rollout_total_distance)
    rollout_iso_epd = rollout_total_energy_iso / max(rollout_iso_distance, 1e-8)
    rollout_net_iso_epd = rollout_net_energy_iso / max(rollout_iso_distance, 1e-8)
    saving_total_isochronous_pct = (eval_ref["total_energy"] - rollout_total_energy_iso) / max(eval_ref["total_energy"], 1e-12) * 100.0
    saving_isochronous_pct = (eval_ref["energy_per_dist"] - rollout_iso_epd) / max(eval_ref["energy_per_dist"], 1e-12) * 100.0
    saving_net_isochronous_pct = (driver_net_epd - rollout_net_iso_epd) / max(driver_net_epd, 1e-12) * 100.0

    step_energy_accounted = np.array(rollout.get("step_energy_accounted", rollout.get("step_energy", [])), dtype=np.float32)
    ref_step_energy = np.array(eval_ref.get("step_energy", []), dtype=np.float32)
    step_distance = np.array(rollout.get("step_distance", []), dtype=np.float32)
    ref_step_distance = np.array(eval_ref.get("step_distance", []), dtype=np.float32)

    def segment_saving_pct(agent_energy, ref_energy, agent_dist, ref_dist):
        if agent_energy.size == 0 or ref_energy.size == 0 or agent_dist.size == 0 or ref_dist.size == 0:
            return 0.0
        agent_epd = float(np.sum(agent_energy) / max(float(np.sum(agent_dist)), 1e-8))
        ref_epd = float(np.sum(ref_energy) / max(float(np.sum(ref_dist)), 1e-8))
        return float(np.clip((ref_epd - agent_epd) / max(ref_epd, 1e-12) * 100.0, -20.0, 20.0))

    def window_energy_saving_pct(agent_energy, ref_energy):
        if agent_energy.size == 0 or ref_energy.size == 0:
            return 0.0
        agent_total = float(np.sum(agent_energy))
        ref_total = float(np.sum(ref_energy))
        return float(np.clip((ref_total - agent_total) / max(ref_total, 1e-8) * 100.0, -20.0, 20.0))

    n_steps = int(min(step_energy_accounted.size, ref_step_energy.size, step_distance.size, ref_step_distance.size))
    window_horizon = 12
    worst_window_saving_pct = 0.0
    negative_window_ratio = 0.0
    if n_steps >= window_horizon:
        window_savings = []
        for start in range(0, n_steps - window_horizon + 1):
            end = start + window_horizon
            window_savings.append(
                window_energy_saving_pct(step_energy_accounted[start:end], ref_step_energy[start:end])
            )
        if window_savings:
            window_savings = np.array(window_savings, dtype=np.float32)
            worst_window_saving_pct = float(np.min(window_savings))
            negative_window_ratio = float(np.mean(window_savings < 0.0))

    split_idx = max(1, n_steps // 2)
    front_half_saving_pct = segment_saving_pct(
        step_energy_accounted[:split_idx],
        ref_step_energy[:split_idx],
        step_distance[:split_idx],
        ref_step_distance[:split_idx],
    )
    back_half_saving_pct = segment_saving_pct(
        step_energy_accounted[split_idx:n_steps],
        ref_step_energy[split_idx:n_steps],
        step_distance[split_idx:n_steps],
        ref_step_distance[split_idx:n_steps],
    )
    front_back_saving_gap_pct = float(abs(front_half_saving_pct - back_half_saving_pct))
    speed_series = np.array(rollout.get("speed", []), dtype=np.float32)
    ref_speed_series = np.array(rollout.get("ref_speed", []), dtype=np.float32)
    dist_series = np.array(rollout.get("distance", []), dtype=np.float32)
    ref_dist_series = np.array(rollout.get("ref_distance", []), dtype=np.float32)
    seg_steps = int(max(6, rollout.get("segment_metric_horizon", 12)))
    worst_segment_speed_mae = 0.0
    worst_segment_dist_mae = 0.0
    if (
        speed_series.size > 1
        and ref_speed_series.size > 1
        and dist_series.size > 1
        and ref_dist_series.size > 1
    ):
        usable_steps = int(min(speed_series.size, ref_speed_series.size, dist_series.size, ref_dist_series.size) - 1)
        if usable_steps >= seg_steps:
            speed_err_series = np.abs(speed_series[1 : usable_steps + 1] - ref_speed_series[1 : usable_steps + 1])
            dist_err_series = np.abs(dist_series[1 : usable_steps + 1] - ref_dist_series[1 : usable_steps + 1])
            speed_mae_windows = []
            dist_mae_windows = []
            for start in range(0, usable_steps - seg_steps + 1):
                end = start + seg_steps
                speed_mae_windows.append(float(np.mean(speed_err_series[start:end])))
                dist_mae_windows.append(float(np.mean(dist_err_series[start:end])))
            if speed_mae_windows:
                worst_segment_speed_mae = float(max(speed_mae_windows))
                worst_segment_dist_mae = float(max(dist_mae_windows))

    driver_torque_src = eval_ref["torque"] if "torque" in eval_ref else eval_ref["torque_cmd"]
    driver_torque_arr = np.array(driver_torque_src, dtype=np.float32)
    driver_torque_delta_mean = (
        float(np.mean(np.abs(np.diff(driver_torque_arr)))) if driver_torque_arr.size > 1 else 0.0
    )
    driver_torque_delta_arr = (
        np.abs(np.diff(driver_torque_arr)) if driver_torque_arr.size > 1 else np.array([0.0], dtype=np.float32)
    )
    driver_torque_jerk_mean = (
        float(np.mean(np.abs(np.diff(driver_torque_delta_arr)))) if driver_torque_delta_arr.size > 1 else 0.0
    )
    agent_torque_delta_mean = float(rollout.get("torque_delta_mean", 0.0))
    agent_torque_jerk_mean = float(rollout.get("torque_jerk_mean", 0.0))
    smooth_delta_ratio = agent_torque_delta_mean / max(driver_torque_delta_mean, 1e-6)
    smooth_jerk_ratio = agent_torque_jerk_mean / max(driver_torque_jerk_mean, 1e-6)
    smooth_ok = bool(
        (smooth_delta_ratio <= smooth_delta_ratio_limit)
        and (smooth_jerk_ratio <= smooth_jerk_ratio_limit)
    )
    tracking_ok = (
        rollout["speed_mae"] <= speed_mae_limit
        and rollout["distance_mae"] <= dist_mae_limit
        and speed_rel_diff_pct <= speed_rel_limit
        and dist_rel_diff_pct <= dist_rel_limit
        and speed_rel_bias_pct >= -max_negative_speed_bias_pct
        and dist_rel_bias_pct >= -max_negative_dist_bias_pct
    )

    return {
        "saving_total_pct": float(saving_total_pct),
        "saving_total_raw_pct": float(saving_total_raw_pct),
        "saving_total_isochronous_pct": float(saving_total_isochronous_pct),
        "saving_epd_pct": float(saving_epd_pct),
        "saving_epd_raw_pct": float(saving_epd_raw_pct),
        "saving_isochronous_pct": float(saving_isochronous_pct),
        "saving_net_epd_pct": float(saving_net_epd_pct),
        "saving_net_epd_raw_pct": float(saving_net_epd_raw_pct),
        "saving_net_isochronous_pct": float(saving_net_isochronous_pct),
        "low_speed_benefit_total_pct": float(low_speed_benefit_total_pct),
        "low_speed_benefit_epd_pct": float(low_speed_benefit_epd_pct),
        "low_speed_benefit_net_epd_pct": float(low_speed_benefit_net_epd_pct),
        "distance_deficit_m": float(distance_deficit_m),
        "speed_deficit_terminal_mps": float(speed_deficit_terminal),
        "distance_comp_energy": float(distance_comp_energy),
        "distance_comp_net_energy": float(distance_comp_net_energy),
        "kinetic_comp_energy": float(kinetic_comp_energy),
        "terminal_kinetic_gap": float(kinetic_gap),
        "isochronous_eta_motor": float(eta_motor),
        "recover_delta_pct": float(recover_delta_pct),
        "recover_ratio_pct": float(recover_ratio_pct),
        "speed_rel_diff_pct": float(speed_rel_diff_pct),
        "dist_rel_diff_pct": float(dist_rel_diff_pct),
        "speed_rel_bias_pct": float(speed_rel_bias_pct),
        "dist_rel_bias_pct": float(dist_rel_bias_pct),
        "launch_metric_horizon": int(launch_horizon),
        "launch_speed_bias_mean": float(launch_speed_bias),
        "launch_dist_bias_mean": float(launch_dist_bias),
        "launch_ref_speed_mean": float(launch_ref_speed_mean),
        "launch_ref_distance_mean": float(launch_ref_distance_mean),
        "launch_speed_rel_bias_pct": float(launch_speed_rel_bias_pct),
        "launch_dist_rel_bias_pct": float(launch_dist_rel_bias_pct),
        "front_half_saving_pct": float(front_half_saving_pct),
        "back_half_saving_pct": float(back_half_saving_pct),
        "front_back_saving_gap_pct": float(front_back_saving_gap_pct),
        "segment_metric_horizon": int(seg_steps),
        "worst_segment_speed_mae": float(worst_segment_speed_mae),
        "worst_segment_dist_mae": float(worst_segment_dist_mae),
        "worst_window_saving_pct": float(worst_window_saving_pct),
        "negative_window_ratio": float(negative_window_ratio),
        "speed_bias": float(signed_speed_diff),
        "dist_bias": float(signed_dist_diff),
        "driver_torque_delta_mean": float(driver_torque_delta_mean),
        "driver_torque_jerk_mean": float(driver_torque_jerk_mean),
        "smooth_delta_ratio": float(smooth_delta_ratio),
        "smooth_jerk_ratio": float(smooth_jerk_ratio),
        "proposed_action_bound_violation_linf_mean": float(rollout.get("proposed_action_bound_violation_linf_mean", 0.0)),
        "proposed_action_bound_violation_linf_max": float(rollout.get("proposed_action_bound_violation_linf_max", 0.0)),
        "smooth_ok": bool(smooth_ok),
        "tracking_ok": bool(tracking_ok),
    }


def evaluate_agent_on_pairs(
    env,
    agent,
    eval_pairs,
    speed_mae_limit=0.9,
    dist_mae_limit=4.8,
    speed_rel_limit=2.0,
    dist_rel_limit=2.0,
    max_negative_speed_bias_pct=0.8,
    max_negative_dist_bias_pct=0.8,
    smooth_delta_ratio_limit=1.18,
    smooth_jerk_ratio_limit=1.25,
):
    """
    作用：在每个评估场景上运行确定性策略并汇总结果。
    输入：env: 环境；agent: 智能体；eval_pairs: 场景与参考列表；其余参数为评估阈值。
    输出：每个评估样本的 rollout 与指标列表。
    """
    prev_scenario = env.scenario
    prev_reference = env.reference
    prev_eval_mode = bool(getattr(env, "eval_mode", False))
    if hasattr(env, "set_eval_mode"):
        env.set_eval_mode(True)
    entries = []
    for eval_scenario, eval_ref in eval_pairs:
        env.set_reference(eval_scenario, eval_ref)
        rollout = rollout_agent_episode(env, agent)
        metrics = evaluate_rollout(
            rollout,
            eval_ref,
            scenario=eval_scenario,
            speed_mae_limit=speed_mae_limit,
            dist_mae_limit=dist_mae_limit,
            speed_rel_limit=speed_rel_limit,
            dist_rel_limit=dist_rel_limit,
            max_negative_speed_bias_pct=max_negative_speed_bias_pct,
            max_negative_dist_bias_pct=max_negative_dist_bias_pct,
            smooth_delta_ratio_limit=smooth_delta_ratio_limit,
            smooth_jerk_ratio_limit=smooth_jerk_ratio_limit,
        )
        entries.append(
            {
                "scenario": eval_scenario,
                "reference": eval_ref,
                "rollout": rollout,
                "metrics": metrics,
            }
        )
    env.set_reference(prev_scenario, prev_reference)
    if hasattr(env, "set_eval_mode"):
        env.set_eval_mode(prev_eval_mode)
    return entries


def train_stage(
    env,
    agent,
    num_episodes,
    stage_name,
    scenario_refs=None,
    eval_pairs=None,
    use_lagrange=False,
    energy_weight_start=0.0,
    energy_weight_end=0.0,
    lagrange_lr_speed=0.4,
    lagrange_lr_dist=0.08,
    lagrange_lr_smooth=0.05,
    lagrange_lr_net=0.08,
    lagrange_lr_projection=0.035,
    lagrange_lr_underspeed=0.08,
    lagrange_lr_window_energy=0.14,
    lagrange_update_controller="integral",
    lagrange_pid_kp_scale=0.35,
    lagrange_pid_kd_scale=0.10,
    lagrange_pid_integral_decay=0.90,
    lagrange_pid_integral_clip=6.0,
    lagrange_pid_delta_clip=1.5,
    target_speed_violation=0.15,
    target_dist_violation=1.00,
    target_smooth_violation=None,
    target_net_violation=None,
    target_projection_violation=None,
    target_underspeed_violation=None,
    target_window_energy_violation=None,
    early_stop_patience=None,
    eval_interval=20,
    speed_mae_limit=0.9,
    dist_mae_limit=4.8,
    speed_rel_limit=2.0,
    dist_rel_limit=2.0,
    max_negative_speed_bias_pct=0.8,
    max_negative_dist_bias_pct=0.8,
    smooth_delta_ratio_limit=1.18,
    smooth_jerk_ratio_limit=1.25,
):
    """
    作用：训练一个阶段的策略，并维护满足约束的最优 checkpoint。
    输入：env/agent: 环境与智能体；num_episodes/stage_name 等: 阶段配置。
    输出：return 曲线、saving 曲线、lambda_speed 历史和 lambda_dist 历史。
    """
    return_list = []
    energy_per_dist_list = []
    lambda_speed_hist = []
    lambda_dist_hist = []
    stop_early = False
    if target_smooth_violation is None:
        target_smooth_violation = float(getattr(env, "constraint_smooth_cost_target", 0.10))
    if target_net_violation is None:
        target_net_violation = float(getattr(env, "constraint_net_energy_cost_target", 0.02))
    if target_projection_violation is None:
        target_projection_violation = float(getattr(env, "constraint_projection_cost_target", 0.10))
    if target_underspeed_violation is None:
        target_underspeed_violation = float(getattr(env, "constraint_underspeed_cost_target", 0.06))
    if target_window_energy_violation is None:
        target_window_energy_violation = float(getattr(env, "constraint_window_energy_cost_target", 0.08))

    if eval_pairs is None or len(eval_pairs) == 0:
        eval_pairs = [(env.scenario, env.reference)]

    best_feasible = None
    best_feasible_epd = 1e18
    best_feasible_worst_epd = 1e18
    best_feasible_worst_style_epd = 1e18
    best_feasible_saving = None
    best_feasible_saving_epd = 1e18
    best_feasible_saving_worst_epd = 1e18
    best_feasible_saving_worst_style_epd = 1e18
    best_feasible_saving_net_epd = 1e18
    best_feasible_saving_worst_net_epd = 1e18
    best_feasible_saving_worst_style_net_epd = 1e18
    best_feasible_saving_total = 1e18
    best_feasible_saving_worst_total = 1e18
    best_feasible_saving_worst_style_total = 1e18
    best_feasible_saving_worst_recover_delta = -1e18
    best_feasible_saving_worst_style_recover_delta = -1e18
    no_improve_count = 0
    eval_styles = sorted(
        set(str(getattr(s, "driver_style", "normal")) for s, _ in eval_pairs)
    )
    single_eval_sport = (len(eval_styles) == 1) and (eval_styles[0] == "sport")
    net_saving_floor_pct = 0.0 if single_eval_sport else -0.35
    recover_drop_tol_pct = 6.0 if single_eval_sport else 8.0
    baseline_eval_total = float(np.mean([float(ref["total_energy"]) for _, ref in eval_pairs]))
    baseline_eval_epd = float(np.mean([ref["energy_per_dist"] for _, ref in eval_pairs]))
    baseline_eval_net_epd = float(
        np.mean([float(ref["net_energy"]) / max(float(ref["total_distance"]), 1e-8) for _, ref in eval_pairs])
    )
    scenario_order = None
    scenario_ptr = 0
    if scenario_refs:
        scenario_order = np.random.permutation(len(scenario_refs)).tolist()
    lagrange_update_controller = str(lagrange_update_controller).strip().lower()
    if lagrange_update_controller not in {"integral", "pid"}:
        lagrange_update_controller = "integral"
    lagrange_pid_state = {
        "speed": {"integral": 0.0, "prev_error": 0.0},
        "distance": {"integral": 0.0, "prev_error": 0.0},
        "smoothness": {"integral": 0.0, "prev_error": 0.0},
        "net_energy": {"integral": 0.0, "prev_error": 0.0},
        "projection": {"integral": 0.0, "prev_error": 0.0},
        "underspeed": {"integral": 0.0, "prev_error": 0.0},
        "window_energy": {"integral": 0.0, "prev_error": 0.0},
    }

    def integral_update_lambda(current_lambda, error, lr_i):
        """
        作用：使用经典积分形式更新 CMDP 的 Lagrange 乘子。
        输入：current_lambda: 当前乘子；error: 本轮约束误差；lr_i: 更新增益。
        输出：更新后的乘子。
        """
        return float(np.clip(current_lambda + lr_i * error, 0.0, 12.0))

    def pid_update_lambda(current_lambda, error, lr_i, key):
        """
        作用：用 PID 形式更新 CMDP 的 Lagrange 乘子，降低纯积分更新的滞后与振荡。
        输入：current_lambda: 当前乘子；error: 本轮约束误差；lr_i: 积分主增益；key: 约束名。
        输出：更新后的乘子。
        """
        state = lagrange_pid_state[key]
        state["integral"] = float(
            np.clip(
                lagrange_pid_integral_decay * state["integral"] + error,
                -lagrange_pid_integral_clip,
                lagrange_pid_integral_clip,
            )
        )
        d_error = float(error - state["prev_error"])
        state["prev_error"] = float(error)
        delta = (
            (lagrange_pid_kp_scale * lr_i) * error
            + lr_i * state["integral"]
            + (lagrange_pid_kd_scale * lr_i) * d_error
        )
        delta = float(np.clip(delta, -lagrange_pid_delta_clip, lagrange_pid_delta_clip))
        return float(np.clip(current_lambda + delta, 0.0, 12.0))

    def snapshot_train_state():
        """
        作用：统一保存策略、critic 与多约束乘子，供回滚或 early-stop 恢复。
        输入：无。
        输出：可恢复的训练状态字典。
        """
        return {
            "actor": copy.deepcopy(agent.actor.state_dict()),
            "critic": copy.deepcopy(agent.critic.state_dict()),
            "agent_aux": copy.deepcopy(agent.get_auxiliary_state_dict()),
            "constraint_multipliers": env.get_constraint_multipliers(),
            "energy_weight": float(env.energy_weight),
            "explore_decay": float(agent.explore_decay),
            "entropy_coef": float(agent.entropy_coef),
            "lagrange_pid_state": copy.deepcopy(lagrange_pid_state),
        }

    def restore_train_state(state):
        """
        作用：恢复由 snapshot_train_state 保存的完整训练状态。
        输入：state: 训练状态字典。
        输出：无。
        """
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        agent.load_auxiliary_state_dict(state.get("agent_aux"))
        env.set_constraint_multipliers(state.get("constraint_multipliers", {}))
        env.energy_weight = float(state["energy_weight"])
        agent.explore_decay = float(state["explore_decay"])
        if "entropy_coef" in state:
            agent.entropy_coef = float(state["entropy_coef"])
        if "lagrange_pid_state" in state:
            for key, pid_state in state["lagrange_pid_state"].items():
                if key in lagrange_pid_state:
                    lagrange_pid_state[key] = {
                        "integral": float(pid_state.get("integral", 0.0)),
                        "prev_error": float(pid_state.get("prev_error", 0.0)),
                    }

    def eval_on_pairs():
        """
        作用：把多评估场景结果聚合成 checkpoint 选择使用的统计量。
        输入：无，闭包内直接使用 env、agent 和 eval_pairs。
        输出：多项均值、最坏值和最坏风格统计量元组。
        """
        eval_entries = evaluate_agent_on_pairs(
            env,
            agent,
            eval_pairs,
            speed_mae_limit=speed_mae_limit,
            dist_mae_limit=dist_mae_limit,
            speed_rel_limit=speed_rel_limit,
            dist_rel_limit=dist_rel_limit,
            max_negative_speed_bias_pct=max_negative_speed_bias_pct,
            max_negative_dist_bias_pct=max_negative_dist_bias_pct,
            smooth_delta_ratio_limit=smooth_delta_ratio_limit,
            smooth_jerk_ratio_limit=smooth_jerk_ratio_limit,
        )
        feasible_all = all(x["metrics"]["tracking_ok"] for x in eval_entries)
        total_energy_list = [float(x["rollout"]["total_energy"]) for x in eval_entries if "rollout" in x]
        epd_list = [float(x["rollout"]["energy_per_dist"]) for x in eval_entries if "rollout" in x]
        net_epd_list = [float(x["rollout"]["net_energy_per_dist"]) for x in eval_entries if "rollout" in x]
        recover_delta_list = [float(x["metrics"]["recover_delta_pct"]) for x in eval_entries]
        style_epd = {}
        style_net_epd = {}
        style_total = {}
        style_recover_delta = {}
        for x in eval_entries:
            style_name = str(getattr(x.get("scenario"), "driver_style", "normal"))
            style_total.setdefault(style_name, []).append(float(x["rollout"]["total_energy"]))
            style_epd.setdefault(style_name, []).append(float(x["rollout"]["energy_per_dist"]))
            style_net_epd.setdefault(style_name, []).append(float(x["rollout"]["net_energy_per_dist"]))
            style_recover_delta.setdefault(style_name, []).append(float(x["metrics"]["recover_delta_pct"]))
        style_mean_total = [float(np.mean(v)) for _, v in style_total.items()]
        style_mean_epd = [float(np.mean(v)) for _, v in style_epd.items()]
        style_mean_net_epd = [float(np.mean(v)) for _, v in style_net_epd.items()]
        style_mean_recover_delta = [float(np.mean(v)) for _, v in style_recover_delta.items()]
        worst_style_total = float(np.max(style_mean_total)) if len(style_mean_total) > 0 else float(np.max(total_energy_list))
        worst_style_epd = float(np.max(style_mean_epd)) if len(style_mean_epd) > 0 else float(np.max(epd_list))
        worst_style_net_epd = float(np.max(style_mean_net_epd)) if len(style_mean_net_epd) > 0 else float(np.max(net_epd_list))
        worst_style_recover_delta = (
            float(np.min(style_mean_recover_delta)) if len(style_mean_recover_delta) > 0 else float(np.min(recover_delta_list))
        )
        return (
            feasible_all,
            float(np.mean(total_energy_list)),
            float(np.max(total_energy_list)),
            worst_style_total,
            float(np.mean(epd_list)),
            float(np.max(epd_list)),
            worst_style_epd,
            float(np.mean(net_epd_list)),
            float(np.max(net_epd_list)),
            worst_style_net_epd,
            float(np.min(recover_delta_list)),
            worst_style_recover_delta,
        )

    rounds = 10
    episodes_per_round = num_episodes // rounds
    remainder = num_episodes % rounds
    global_ep = 0
    for i in range(rounds):
        this_round = episodes_per_round + (1 if i < remainder else 0)
        if this_round == 0:
            continue
        with make_pbar(total=this_round, desc=f"{stage_name} {i+1}/{rounds}") as pbar:
            for _ in range(this_round):
                progress = global_ep / max(num_episodes - 1, 1)
                if scenario_refs:
                    if scenario_ptr >= len(scenario_order):
                        scenario_order = np.random.permutation(len(scenario_refs)).tolist()
                        scenario_ptr = 0
                    scenario_idx = int(scenario_order[scenario_ptr])
                    scenario_ptr += 1
                    env.set_reference(scenario_refs[scenario_idx][0], scenario_refs[scenario_idx][1])
                if env.mode == "energy":
                    base_weight = float(energy_weight_start + progress * (energy_weight_end - energy_weight_start))
                    style_name = str(getattr(env.scenario, "driver_style", "normal"))
                    style_scale = float(env.style_energy_scale.get(style_name, 1.0))
                    env.energy_weight = float(max(0.0, base_weight * style_scale))

                episode_return = 0.0
                total_energy = 0.0
                total_distance = 0.0
                speed_constraint_cost_sum = 0.0
                dist_constraint_cost_sum = 0.0
                smooth_constraint_cost_sum = 0.0
                net_constraint_cost_sum = 0.0
                projection_constraint_cost_sum = 0.0
                underspeed_constraint_cost_sum = 0.0
                window_energy_constraint_cost_sum = 0.0
                step_count = 0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "raw_actions": [],
                    "next_states": [],
                    "rewards": [],
                    "terminateds": [],
                    "constraint_costs": {
                        "speed": [],
                        "distance": [],
                        "smoothness": [],
                        "net_energy": [],
                        "projection": [],
                        "underspeed": [],
                        "window_energy": [],
                    },
                    "constraint_lambdas": env.get_constraint_multipliers(),
                }

                state, _ = env.reset()
                done = False

                while not done:
                    action, raw_action = agent.take_action(state, return_raw=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    total_energy += float(info.get("step_energy_accounted", info["step_energy"]))
                    total_distance += info["step_distance"]
                    speed_constraint_cost_sum += float(info.get("constraint_speed_cost", 0.0))
                    dist_constraint_cost_sum += float(info.get("constraint_distance_cost", 0.0))
                    smooth_constraint_cost_sum += float(info.get("constraint_smoothness_cost", 0.0))
                    net_constraint_cost_sum += float(info.get("constraint_net_energy_cost", 0.0))
                    projection_constraint_cost_sum += float(info.get("constraint_projection_cost", 0.0))
                    underspeed_constraint_cost_sum += float(info.get("constraint_underspeed_cost", 0.0))
                    window_energy_constraint_cost_sum += float(info.get("constraint_window_energy_cost", 0.0))
                    step_count += 1

                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["raw_actions"].append(raw_action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(float(info.get("base_reward", info.get("objective_reward", reward))))
                    transition_dict["terminateds"].append(bool(terminated))
                    transition_dict["constraint_costs"]["speed"].append(float(info.get("constraint_speed_cost", 0.0)))
                    transition_dict["constraint_costs"]["distance"].append(float(info.get("constraint_distance_cost", 0.0)))
                    transition_dict["constraint_costs"]["smoothness"].append(float(info.get("constraint_smoothness_cost", 0.0)))
                    transition_dict["constraint_costs"]["net_energy"].append(float(info.get("constraint_net_energy_cost", 0.0)))
                    transition_dict["constraint_costs"]["projection"].append(float(info.get("constraint_projection_cost", 0.0)))
                    transition_dict["constraint_costs"]["underspeed"].append(float(info.get("constraint_underspeed_cost", 0.0)))
                    transition_dict["constraint_costs"]["window_energy"].append(float(info.get("constraint_window_energy_cost", 0.0)))

                    state = next_state
                    episode_return += reward

                return_list.append(float(episode_return))
                episode_epd = float(total_energy / max(total_distance, 1e-8))
                ref_epd_episode = float(env.reference["energy_per_dist"])
                episode_saving_pct = (ref_epd_episode - episode_epd) / max(ref_epd_episode, 1e-8) * 100.0
                energy_per_dist_list.append(float(episode_saving_pct))
                agent.update(transition_dict)

                avg_speed_violation = speed_constraint_cost_sum / max(step_count, 1)
                avg_dist_violation = dist_constraint_cost_sum / max(step_count, 1)
                avg_smooth_violation = smooth_constraint_cost_sum / max(step_count, 1)
                avg_net_violation = net_constraint_cost_sum / max(step_count, 1)
                avg_projection_violation = projection_constraint_cost_sum / max(step_count, 1)
                avg_underspeed_violation = underspeed_constraint_cost_sum / max(step_count, 1)
                avg_window_energy_violation = window_energy_constraint_cost_sum / max(step_count, 1)
                if use_lagrange and env.mode == "energy":
                    speed_error = float(avg_speed_violation - target_speed_violation)
                    dist_error = float(avg_dist_violation - target_dist_violation)
                    smooth_error = float(avg_smooth_violation - target_smooth_violation)
                    net_error = float(avg_net_violation - target_net_violation)
                    projection_error = float(avg_projection_violation - target_projection_violation)
                    underspeed_error = float(avg_underspeed_violation - target_underspeed_violation)
                    window_energy_error = float(avg_window_energy_violation - target_window_energy_violation)
                    env.set_constraint_multipliers(
                        lambda_speed=(
                            pid_update_lambda(env.lambda_speed, speed_error, lagrange_lr_speed, "speed")
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(env.lambda_speed, speed_error, lagrange_lr_speed)
                        ),
                        lambda_dist=(
                            pid_update_lambda(env.lambda_dist, dist_error, lagrange_lr_dist, "distance")
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(env.lambda_dist, dist_error, lagrange_lr_dist)
                        ),
                        lambda_smooth=(
                            pid_update_lambda(env.lambda_smooth, smooth_error, lagrange_lr_smooth, "smoothness")
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(env.lambda_smooth, smooth_error, lagrange_lr_smooth)
                        ),
                        lambda_net_energy=(
                            pid_update_lambda(env.lambda_net_energy, net_error, lagrange_lr_net, "net_energy")
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(env.lambda_net_energy, net_error, lagrange_lr_net)
                        ),
                        lambda_projection=(
                            pid_update_lambda(
                                env.lambda_projection,
                                projection_error,
                                lagrange_lr_projection,
                                "projection",
                            )
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(
                                env.lambda_projection,
                                projection_error,
                                lagrange_lr_projection,
                            )
                        ),
                        lambda_underspeed=(
                            pid_update_lambda(
                                env.lambda_underspeed,
                                underspeed_error,
                                lagrange_lr_underspeed,
                                "underspeed",
                            )
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(
                                env.lambda_underspeed,
                                underspeed_error,
                                lagrange_lr_underspeed,
                            )
                        ),
                        lambda_window_energy=(
                            pid_update_lambda(
                                env.lambda_window_energy,
                                window_energy_error,
                                lagrange_lr_window_energy,
                                "window_energy",
                            )
                            if lagrange_update_controller == "pid"
                            else integral_update_lambda(
                                env.lambda_window_energy,
                                window_energy_error,
                                lagrange_lr_window_energy,
                            )
                        ),
                    )

                lambda_speed_hist.append(env.lambda_speed)
                lambda_dist_hist.append(env.lambda_dist)

                if (global_ep + 1) % 10 == 0:
                    postfix = {
                        "episode": f"{global_ep + 1:.0f}",
                        "avg_return": f"{np.mean(return_list[-10:]):.2f}",
                        "avg_save_pct": f"{np.mean(energy_per_dist_list[-10:]):.2f}%",
                        "explore": f"{agent.explore_decay:.3f}",
                        "ent": f"{agent.entropy_coef:.5f}",
                    }
                    if env.mode == "energy":
                        postfix["e_w"] = f"{env.energy_weight:.2f}"
                        postfix["lam_s"] = f"{env.lambda_speed:.2f}"
                        postfix["lam_d"] = f"{env.lambda_dist:.2f}"
                        postfix["lam_sm"] = f"{env.lambda_smooth:.2f}"
                        postfix["lam_n"] = f"{env.lambda_net_energy:.2f}"
                        postfix["lam_p"] = f"{env.lambda_projection:.2f}"
                        postfix["lam_u"] = f"{env.lambda_underspeed:.2f}"
                        postfix["lam_w"] = f"{env.lambda_window_energy:.2f}"
                    pbar.set_postfix(postfix)

                do_eval = (
                    env.mode == "energy"
                    and early_stop_patience is not None
                    and eval_interval > 0
                    and ((global_ep + 1) % eval_interval == 0)
                )
                if do_eval:
                    (
                        feasible,
                        mean_total,
                        worst_total,
                        worst_style_total,
                        mean_epd,
                        worst_epd,
                        worst_style_epd,
                        mean_net_epd,
                        worst_net_epd,
                        worst_style_net_epd,
                        worst_recover_delta,
                        worst_style_recover_delta,
                    ) = eval_on_pairs()
                    improved = False
                    if feasible:
                        feasible_better = (
                            (worst_style_epd < best_feasible_worst_style_epd - 1e-12)
                            or (
                                abs(worst_style_epd - best_feasible_worst_style_epd) <= 1e-12
                                and (
                                    (worst_epd < best_feasible_worst_epd - 1e-12)
                                    or (
                                        abs(worst_epd - best_feasible_worst_epd) <= 1e-12
                                        and mean_epd < best_feasible_epd
                                    )
                                )
                            )
                        )
                        if feasible_better:
                            best_feasible_epd = mean_epd
                            best_feasible_worst_epd = worst_epd
                            best_feasible_worst_style_epd = worst_style_epd
                            best_feasible = snapshot_train_state()
                            improved = True
                    mean_total_saving = (baseline_eval_total - mean_total) / max(baseline_eval_total, 1e-12) * 100.0
                    worst_total_saving = (baseline_eval_total - worst_total) / max(baseline_eval_total, 1e-12) * 100.0
                    worst_style_total_saving = (baseline_eval_total - worst_style_total) / max(baseline_eval_total, 1e-12) * 100.0
                    mean_epd_saving = (baseline_eval_epd - mean_epd) / max(baseline_eval_epd, 1e-12) * 100.0
                    mean_net_saving = (baseline_eval_net_epd - mean_net_epd) / max(baseline_eval_net_epd, 1e-12) * 100.0
                    worst_net_saving = (baseline_eval_net_epd - worst_net_epd) / max(baseline_eval_net_epd, 1e-12) * 100.0
                    worst_style_net_saving = (
                        (baseline_eval_net_epd - worst_style_net_epd) / max(baseline_eval_net_epd, 1e-12) * 100.0
                    )
                    net_saving_ok = (
                        worst_net_saving >= net_saving_floor_pct and worst_style_net_saving >= net_saving_floor_pct
                    )
                    recover_ok = (
                        worst_recover_delta >= -recover_drop_tol_pct
                        and worst_style_recover_delta >= -recover_drop_tol_pct
                    )
                    total_saving_ok = (worst_total_saving > 0.0) and (worst_style_total_saving > 0.0)
                    if feasible and total_saving_ok and net_saving_ok and recover_ok:
                        saving_quality = (
                            0.50 * worst_style_total_saving
                            + 0.25 * mean_total_saving
                            + 0.15 * mean_epd_saving
                            + 0.10 * mean_net_saving
                        )
                        best_quality = (
                            0.50
                            * (
                                (baseline_eval_total - best_feasible_saving_worst_style_total)
                                / max(baseline_eval_total, 1e-12)
                                * 100.0
                            )
                            + 0.25
                            * (
                                (baseline_eval_total - best_feasible_saving_total)
                                / max(baseline_eval_total, 1e-12)
                                * 100.0
                            )
                            + 0.15
                            * (
                                (baseline_eval_epd - best_feasible_saving_epd)
                                / max(baseline_eval_epd, 1e-12)
                                * 100.0
                            )
                            + 0.10
                            * (
                                (baseline_eval_net_epd - best_feasible_saving_net_epd)
                                / max(baseline_eval_net_epd, 1e-12)
                                * 100.0
                            )
                        )
                        saving_better = (
                            (saving_quality > best_quality + 1e-12)
                            or (
                                abs(saving_quality - best_quality) <= 1e-12
                                and (
                                    (worst_style_total < best_feasible_saving_worst_style_total - 1e-12)
                                    or (
                                        abs(worst_style_total - best_feasible_saving_worst_style_total) <= 1e-12
                                        and (
                                            (worst_style_net_epd < best_feasible_saving_worst_style_net_epd - 1e-12)
                                            or (
                                                abs(worst_style_net_epd - best_feasible_saving_worst_style_net_epd) <= 1e-12
                                                and (
                                                    (worst_recover_delta > best_feasible_saving_worst_recover_delta + 1e-12)
                                                    or (
                                                        abs(worst_recover_delta - best_feasible_saving_worst_recover_delta) <= 1e-12
                                                        and mean_total < best_feasible_saving_total
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        if saving_better:
                            best_feasible_saving_total = mean_total
                            best_feasible_saving_worst_total = worst_total
                            best_feasible_saving_worst_style_total = worst_style_total
                            best_feasible_saving_epd = mean_epd
                            best_feasible_saving_worst_epd = worst_epd
                            best_feasible_saving_worst_style_epd = worst_style_epd
                            best_feasible_saving_net_epd = mean_net_epd
                            best_feasible_saving_worst_net_epd = worst_net_epd
                            best_feasible_saving_worst_style_net_epd = worst_style_net_epd
                            best_feasible_saving_worst_recover_delta = worst_recover_delta
                            best_feasible_saving_worst_style_recover_delta = worst_style_recover_delta
                            best_feasible_saving = snapshot_train_state()
                            improved = True

                    ckpt_available = (best_feasible is not None) or (best_feasible_saving is not None)
                    if improved:
                        no_improve_count = 0
                    elif ckpt_available:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0

                    if ckpt_available and no_improve_count >= early_stop_patience:
                        restore_ckpt = best_feasible_saving if best_feasible_saving is not None else best_feasible
                        if restore_ckpt is not None:
                            restore_train_state(restore_ckpt)
                            stop_early = True
                pbar.update(1)
                global_ep += 1
                if stop_early:
                    break
            if stop_early:
                break

    if env.mode == "energy" and (best_feasible is not None or best_feasible_saving is not None):
        (
            current_feasible,
            current_mean_total,
            current_worst_total,
            current_worst_style_total,
            current_mean_epd,
            current_worst_epd,
            current_worst_style_epd,
            current_mean_net_epd,
            current_worst_net_epd,
            current_worst_style_net_epd,
            current_worst_recover_delta,
            current_worst_style_recover_delta,
        ) = eval_on_pairs()
        current_worst_total_saving = (baseline_eval_total - current_worst_total) / max(baseline_eval_total, 1e-12) * 100.0
        current_worst_style_total_saving = (
            (baseline_eval_total - current_worst_style_total) / max(baseline_eval_total, 1e-12) * 100.0
        )
        current_worst_net_saving = (baseline_eval_net_epd - current_worst_net_epd) / max(baseline_eval_net_epd, 1e-12) * 100.0
        current_worst_style_net_saving = (
            (baseline_eval_net_epd - current_worst_style_net_epd) / max(baseline_eval_net_epd, 1e-12) * 100.0
        )
        current_saving = (
            current_feasible
            and (current_worst_total_saving > 0.0)
            and (current_worst_style_total_saving > 0.0)
            and (current_worst_net_saving >= net_saving_floor_pct)
            and (current_worst_style_net_saving >= net_saving_floor_pct)
            and (current_worst_recover_delta >= -recover_drop_tol_pct)
            and (current_worst_style_recover_delta >= -recover_drop_tol_pct)
        )
        target_ckpt = best_feasible_saving if best_feasible_saving is not None else best_feasible
        target_total = best_feasible_saving_total if best_feasible_saving is not None else current_mean_total
        target_worst_total = best_feasible_saving_worst_total if best_feasible_saving is not None else current_worst_total
        target_worst_style_total = (
            best_feasible_saving_worst_style_total if best_feasible_saving is not None else current_worst_style_total
        )
        target_epd = best_feasible_saving_epd if best_feasible_saving is not None else best_feasible_epd
        target_worst_epd = best_feasible_saving_worst_epd if best_feasible_saving is not None else best_feasible_worst_epd
        target_worst_style_epd = (
            best_feasible_saving_worst_style_epd if best_feasible_saving is not None else best_feasible_worst_style_epd
        )
        target_net_epd = best_feasible_saving_net_epd if best_feasible_saving is not None else current_mean_net_epd
        target_worst_net_epd = (
            best_feasible_saving_worst_net_epd if best_feasible_saving is not None else current_worst_net_epd
        )
        target_worst_style_net_epd = (
            best_feasible_saving_worst_style_net_epd
            if best_feasible_saving is not None
            else current_worst_style_net_epd
        )
        target_worst_recover_delta = (
            best_feasible_saving_worst_recover_delta if best_feasible_saving is not None else current_worst_recover_delta
        )
        target_worst_style_recover_delta = (
            best_feasible_saving_worst_style_recover_delta
            if best_feasible_saving is not None
            else current_worst_style_recover_delta
        )
        degrade = (
            (current_worst_style_total > target_worst_style_total + 1e-12)
            or (
                abs(current_worst_style_total - target_worst_style_total) <= 1e-12
                and (
                    (current_worst_style_net_epd > target_worst_style_net_epd + 1e-12)
                    or (
                        abs(current_worst_style_net_epd - target_worst_style_net_epd) <= 1e-12
                        and (
                            (current_worst_style_recover_delta < target_worst_style_recover_delta - 1e-12)
                            or (
                                abs(current_worst_style_recover_delta - target_worst_style_recover_delta) <= 1e-12
                                and (
                                    (current_worst_total > target_worst_total + 1e-12)
                                    or (
                                        abs(current_worst_total - target_worst_total) <= 1e-12
                                        and (
                                            (current_worst_net_epd > target_worst_net_epd + 1e-12)
                                            or (
                                                abs(current_worst_net_epd - target_worst_net_epd) <= 1e-12
                                                and (
                                                    (current_mean_total > target_total + 1e-12)
                                                    or (
                                                        abs(current_mean_total - target_total) <= 1e-12
                                                        and (
                                                            (current_mean_net_epd > target_net_epd + 1e-12)
                                                            or (
                                                                abs(current_mean_net_epd - target_net_epd) <= 1e-12
                                                                and (
                                                                    (current_mean_epd > target_epd + 1e-12)
                                                                    or (
                                                                        abs(current_mean_epd - target_epd) <= 1e-12
                                                                        and current_worst_epd > target_worst_epd
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        if target_ckpt is not None and ((not current_saving and best_feasible_saving is not None) or (not current_feasible) or degrade):
            restore_train_state(target_ckpt)

    return return_list, energy_per_dist_list, lambda_speed_hist, lambda_dist_hist


def rollout_agent_episode(env, agent):
    """
    作用：用确定性策略运行一个 episode，并记录轨迹级诊断信息。
    输入：env: 环境；agent: 智能体。
    输出：包含轨迹、动作、能耗和平滑指标的字典。
    """
    state, _ = env.reset()
    speed = [env.vehicle["speed"]]
    ref_speed = [env.reference["speed"][0]]
    distance = [env.vehicle["distance"]]
    ref_distance = [env.reference["distance"][0]]
    torque = [env.vehicle["torque"]]
    ref_torque = [env.reference["torque_cmd"][0]]
    soc = [env.vehicle["soc"]]
    actions = []
    executed_actions = []
    regen = []
    coast = []
    residual = []
    torque_cmd = []
    action_projection_delta = []
    proposed_action_bound_violation_linf = []

    total_energy_raw = 0.0
    total_energy_accounted = 0.0
    total_recover = 0.0
    total_distance = 0.0
    total_slow_bias_energy_debit = 0.0
    step_energy = []
    step_energy_raw = []
    step_recover = []
    step_distance = []
    speed_err_abs = []
    dist_err_abs = []
    segment_speed_constraint_cost = []
    segment_dist_constraint_cost = []

    done = False
    while not done:
        action = agent.take_action(state, deterministic=True)
        next_state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        actions.append(np.array(action, dtype=np.float32))
        executed_actions.append(np.array(info.get("executed_action", action), dtype=np.float32))
        action_projection_delta.append(np.array(info.get("action_projection_delta", np.zeros_like(action)), dtype=np.float32))
        proposed_action_bound_violation_linf.append(float(info.get("proposed_action_bound_violation_linf", 0.0)))
        regen.append(float(info["regen_gain"]))
        coast.append(float(info.get("coast_gain", 0.0)))
        residual.append(float(info.get("residual", 0.0)))
        torque_cmd.append(float(info.get("torque_cmd", env.vehicle["torque"])))
        speed.append(env.vehicle["speed"])
        ref_speed.append(info["ref_speed"])
        distance.append(env.vehicle["distance"])
        ref_distance.append(info["ref_distance"])
        torque.append(env.vehicle["torque"])
        ref_torque.append(info["ref_torque"])
        soc.append(env.vehicle["soc"])

        raw_step_energy = float(info["step_energy"])
        accounted_step_energy = float(info.get("step_energy_accounted", raw_step_energy))
        total_energy_raw += raw_step_energy
        total_energy_accounted += accounted_step_energy
        total_recover += info["step_recover"]
        total_distance += info["step_distance"]
        total_slow_bias_energy_debit += float(info.get("slow_bias_energy_debit", 0.0))
        step_energy.append(accounted_step_energy)
        step_energy_raw.append(raw_step_energy)
        step_recover.append(float(info["step_recover"]))
        step_distance.append(float(info["step_distance"]))
        speed_err_abs.append(abs(info["speed_error"]))
        dist_err_abs.append(abs(info["distance_error"]))
        segment_speed_constraint_cost.append(float(info.get("segment_speed_constraint_cost", 0.0)))
        segment_dist_constraint_cost.append(float(info.get("segment_dist_constraint_cost", 0.0)))
        state = next_state

    net_energy = total_energy_accounted - total_recover
    net_energy_raw = total_energy_raw - total_recover
    torque_arr = np.array(torque, dtype=np.float32)
    torque_delta_mean = float(np.mean(np.abs(np.diff(torque_arr)))) if torque_arr.size > 1 else 0.0
    torque_delta_arr = np.abs(np.diff(torque_arr)) if torque_arr.size > 1 else np.array([0.0], dtype=np.float32)
    torque_jerk_mean = float(np.mean(np.abs(np.diff(torque_delta_arr)))) if torque_delta_arr.size > 1 else 0.0
    launch_metric_horizon = min(20, max(len(speed) - 1, 0))
    if launch_metric_horizon > 0:
        launch_slice = slice(1, launch_metric_horizon + 1)
        launch_speed_arr = np.array(speed[launch_slice], dtype=np.float32)
        launch_ref_speed_arr = np.array(ref_speed[launch_slice], dtype=np.float32)
        launch_distance_arr = np.array(distance[launch_slice], dtype=np.float32)
        launch_ref_distance_arr = np.array(ref_distance[launch_slice], dtype=np.float32)
        launch_speed_bias_mean = float(np.mean(launch_speed_arr - launch_ref_speed_arr))
        launch_dist_bias_mean = float(np.mean(launch_distance_arr - launch_ref_distance_arr))
        launch_ref_speed_mean = float(np.mean(launch_ref_speed_arr))
        launch_ref_distance_mean = float(np.mean(launch_ref_distance_arr))
    else:
        launch_speed_bias_mean = 0.0
        launch_dist_bias_mean = 0.0
        launch_ref_speed_mean = 0.0
        launch_ref_distance_mean = 0.0
    return {
        "speed": np.array(speed, dtype=np.float32),
        "ref_speed": np.array(ref_speed, dtype=np.float32),
        "distance": np.array(distance, dtype=np.float32),
        "ref_distance": np.array(ref_distance, dtype=np.float32),
        "torque": np.array(torque, dtype=np.float32),
        "ref_torque": np.array(ref_torque, dtype=np.float32),
        "soc": np.array(soc, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "executed_actions": np.array(executed_actions, dtype=np.float32),
        "action_projection_delta": np.array(action_projection_delta, dtype=np.float32),
        "regen_gain": np.array(regen, dtype=np.float32),
        "coast_gain": np.array(coast, dtype=np.float32),
        "residual": np.array(residual, dtype=np.float32),
        "torque_cmd": np.array(torque_cmd, dtype=np.float32),
        "total_energy": float(total_energy_accounted),
        "total_energy_accounted": float(total_energy_accounted),
        "total_energy_raw": float(total_energy_raw),
        "total_recover": float(total_recover),
        "net_energy": float(net_energy),
        "net_energy_accounted": float(net_energy),
        "net_energy_raw": float(net_energy_raw),
        "total_slow_bias_energy_debit": float(total_slow_bias_energy_debit),
        "total_distance": float(total_distance),
        "step_energy": np.array(step_energy, dtype=np.float32),
        "step_energy_accounted": np.array(step_energy, dtype=np.float32),
        "step_energy_raw": np.array(step_energy_raw, dtype=np.float32),
        "step_recover": np.array(step_recover, dtype=np.float32),
        "step_distance": np.array(step_distance, dtype=np.float32),
        "energy_per_dist": float(total_energy_accounted / max(total_distance, 1e-8)),
        "energy_per_dist_accounted": float(total_energy_accounted / max(total_distance, 1e-8)),
        "energy_per_dist_raw": float(total_energy_raw / max(total_distance, 1e-8)),
        "slow_bias_energy_debit_per_dist": float(total_slow_bias_energy_debit / max(total_distance, 1e-8)),
        "net_energy_per_dist": float(net_energy / max(total_distance, 1e-8)),
        "net_energy_per_dist_accounted": float(net_energy / max(total_distance, 1e-8)),
        "net_energy_per_dist_raw": float(net_energy_raw / max(total_distance, 1e-8)),
        "speed_mae": float(np.mean(speed_err_abs)),
        "distance_mae": float(np.mean(dist_err_abs)),
        "avg_speed": float(np.mean(speed[1:])),
        "final_distance": float(distance[-1]),
        "launch_metric_horizon": int(launch_metric_horizon),
        "segment_metric_horizon": int(getattr(env, "segment_tracking_horizon", 12)),
        "launch_speed_bias_mean": float(launch_speed_bias_mean),
        "launch_dist_bias_mean": float(launch_dist_bias_mean),
        "launch_ref_speed_mean": float(launch_ref_speed_mean),
        "launch_ref_distance_mean": float(launch_ref_distance_mean),
        "segment_speed_constraint_cost_mean": float(
            np.mean(segment_speed_constraint_cost) if len(segment_speed_constraint_cost) > 0 else 0.0
        ),
        "segment_dist_constraint_cost_mean": float(
            np.mean(segment_dist_constraint_cost) if len(segment_dist_constraint_cost) > 0 else 0.0
        ),
        "torque_delta_mean": torque_delta_mean,
        "torque_jerk_mean": torque_jerk_mean,
        "action_projection_l1_mean": float(
            np.mean(np.abs(np.array(action_projection_delta, dtype=np.float32))) if len(action_projection_delta) > 0 else 0.0
        ),
        "action_projection_l2_mean": float(
            np.mean(np.linalg.norm(np.array(action_projection_delta, dtype=np.float32), axis=1)) if len(action_projection_delta) > 0 else 0.0
        ),
        "proposed_action_bound_violation_linf_mean": float(
            np.mean(proposed_action_bound_violation_linf) if len(proposed_action_bound_violation_linf) > 0 else 0.0
        ),
        "proposed_action_bound_violation_linf_max": float(
            np.max(proposed_action_bound_violation_linf) if len(proposed_action_bound_violation_linf) > 0 else 0.0
        ),
    }


