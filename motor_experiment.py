import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import copy
from pathlib import Path

import numpy as np
import torch

from motor_env import MPS_TO_KMH, build_random_road_profiles, StraightRoadScenario, DriverReferenceEnergyEnv
from motor_agent import PPOContinuous
from motor_training import train_stage, evaluate_agent_on_pairs
from motor_plotting import (
    plot_training_curves,
    plot_tracking,
    plot_energy_saving,
    plot_saving_trace,
    plot_window_saving_trace,
    export_saving_trace_csv,
    plot_motor_efficiency_map,
    export_motor_map_csv_template,
)


DEFAULT_SINGLE_STYLE_CFG = {
    "energy_stage": {
        "e_start": 0.90,
        "e_end": 2.35,
        "lr_s": 0.40,
        "lr_d": 0.04,
        "tgt_s": 0.10,
        "tgt_d": 0.60,
        "patience": 6,
        "eval_interval": 16,
    },
    "net_saving_floor_pct": -0.35,
    "recover_drop_tol_pct": 8.0,
    "bias_guard_speed_floor": -0.35,
    "bias_guard_dist_floor": -0.35,
    "bias_excess_floor": 0.24,
    "tracking_bias_speed_limit_pct": 0.8,
    "tracking_bias_dist_limit_pct": 0.8,
}

SINGLE_STYLE_CFG = {
    "sport": {
        "energy_stage": {
            "e_start": 0.90,
            "e_end": 2.25,
            "lr_s": 0.42,
            "lr_d": 0.05,
            "tgt_s": 0.08,
            "tgt_d": 0.56,
            "patience": 6,
            "eval_interval": 14,
        },
        "net_saving_floor_pct": 0.0,
        "recover_drop_tol_pct": 6.0,
        "drop_seed_zero": True,
        "polish": {
            "episodes_fast": 14,
            "episodes_full": 36,
            "max_explore": 0.18,
            "stage_weight": 1.35,
            "e_start": 1.30,
            "e_end": 1.95,
            "lr_s": 0.50,
            "lr_d": 0.07,
            "tgt_s": 0.07,
            "tgt_d": 0.56,
            "patience": 4,
            "eval_interval": 8,
            "stage_name": "SportPolish",
        },
    },
    "eco": {
        "energy_stage": {
            "e_start": 0.95,
            "e_end": 2.10,
            "lr_s": 0.36,
            "lr_d": 0.040,
            "tgt_s": 0.09,
            "tgt_d": 0.62,
            "patience": 6,
            "eval_interval": 16,
        },
        "net_saving_floor_pct": -0.10,
        "recover_drop_tol_pct": 8.0,
        "bias_guard_speed_floor": -0.65,
        "bias_guard_dist_floor": -0.65,
        "bias_excess_floor": 0.45,
        "polish": {
            "episodes_fast": 16,
            "episodes_full": 40,
            "max_explore": 0.16,
            "stage_weight": 1.40,
            "e_start": 1.30,
            "e_end": 1.95,
            "lr_s": 0.32,
            "lr_d": 0.030,
            "tgt_s": 0.09,
            "tgt_d": 0.62,
            "patience": 4,
            "eval_interval": 8,
            "stage_name": "EcoPolish",
        },
    },
    "normal": {
        "bias_guard_speed_floor": -0.45,
        "bias_guard_dist_floor": -0.45,
        "bias_excess_floor": 0.35,
        "tracking_bias_speed_limit_pct": 0.50,
        "tracking_bias_dist_limit_pct": 0.50,
        "ppo": {
            # Normal style is sensitive to persistent exploration drift.
            # Use a narrower adaptive entropy band to improve bias guard stability.
            "init_explore_decay": 0.82,
            "max_explore": 0.95,
            "entropy_coef": 0.00025,
            "entropy_coef_max": 0.0012,
            "entropy_adapt_rate": 0.05,
            "entropy_kl_low_ratio": 0.80,
            "entropy_kl_high_ratio": 1.20,
            "explore_expand": 1.03,
            "explore_shrink": 0.96,
        },
        "energy_stage": {
            "e_start": 0.80,
            "e_end": 1.75,
            "lr_s": 0.60,
            "lr_d": 0.09,
            "tgt_s": 0.05,
            "tgt_d": 0.42,
            "patience": 6,
            "eval_interval": 14,
        },
        "bias_repair": {
            "episodes_fast": 14,
            "episodes_full": 32,
            "max_explore": 0.12,
            "stage_weight": 1.10,
            "e_start": 0.95,
            "e_end": 1.20,
            "lr_s": 0.65,
            "lr_d": 0.10,
            "tgt_s": 0.05,
            "tgt_d": 0.40,
            "patience": 3,
            "eval_interval": 6,
            "stage_name": "NormalBiasRepair",
        },
        "polish": {
            "episodes_fast": 14,
            "episodes_full": 36,
            "max_explore": 0.14,
            "stage_weight": 1.30,
            "e_start": 1.15,
            "e_end": 1.65,
            "lr_s": 0.48,
            "lr_d": 0.060,
            "tgt_s": 0.06,
            "tgt_d": 0.46,
            "patience": 4,
            "eval_interval": 8,
            "stage_name": "NormalPolish",
        },
    },
}

DEFAULT_PPO_CFG = {
    "init_explore_decay": 0.90,
    "min_explore": 0.08,
    "max_explore": 1.20,
    "entropy_coef": 0.0003,
    "entropy_coef_min": 0.00005,
    "entropy_coef_max": 0.0025,
    "entropy_adapt_rate": 0.08,
    "entropy_kl_low_ratio": 0.65,
    "entropy_kl_high_ratio": 1.35,
    "explore_expand": 1.06,
    "explore_shrink": 0.94,
    "std_offset": 0.02,
    "min_std": 0.015,
    "max_std": 0.35,
    "init_std_bias": -2.2,
    "use_return_rms": 1.0,
    "return_rms_eps": 1e-4,
    "return_rms_clip": 5.0,
}

DEFAULT_MEMORY_CFG = {
    "obs_stack": 1,
}



def run_experiment():
    """
    作用：运行完整的电机 PPO 训练、评估与结果导出流程。
    输入：无，运行参数主要通过环境变量控制。
    输出：无，会在工作目录中生成模型、日志和图片文件。
    """
    fast_run = os.getenv("FAST_RUN", "0") == "1"
    long_run = os.getenv("LONG_RUN", "0") == "1"

    actor_lr = 6e-5
    critic_lr = 3e-4
    track_episodes = 90 if fast_run else 170
    energy_episodes = 140 if fast_run else 220
    if long_run:
        track_episodes = int(os.getenv("TRACK_EPISODES", "280"))
        energy_episodes = int(os.getenv("ENERGY_EPISODES", "420"))
    run_mode_name = "fast" if fast_run else ("long" if long_run else "full")
    num_episodes = track_episodes + energy_episodes
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    residual_bound = 14.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    horizon = 220
    dt = 0.1
    max_speed_mps = 80.0 / MPS_TO_KMH
    num_train_roads = 3 if fast_run else 5
    num_eval_roads = 2 if fast_run else 3
    if long_run:
        num_train_roads = int(os.getenv("NUM_TRAIN_ROADS", "7"))
        num_eval_roads = int(os.getenv("NUM_EVAL_ROADS", "4"))
    # ????????????????????????????????????
    multi_style = False
    target_driver_style = os.getenv("TARGET_DRIVER_STYLE", "eco").strip().lower()
    if target_driver_style not in StraightRoadScenario.STYLE_TO_IDX:
        target_driver_style = "eco"
    driver_styles = [target_driver_style]
    run_tag = os.getenv("RUN_TAG", "").strip()
    output_dir = Path("artifacts") / target_driver_style / (run_tag if run_tag else run_mode_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    motor_map_csv = os.getenv("MOTOR_MAP_CSV", "").strip()
    if motor_map_csv == "":
        for candidate in (
            "motor_efficiency_map_real.csv",
            "motor_efficiency_map.csv",
            "motor_map_real.csv",
            "motor_map.csv",
            "sys_eff_pivot.csv",
            os.path.join("data", "motor_efficiency_map_real.csv"),
            os.path.join("data", "motor_efficiency_map.csv"),
            os.path.join("data", "motor_map_real.csv"),
            os.path.join("data", "motor_map.csv"),
            os.path.join("data", "sys_eff_pivot.csv"),
        ):
            if Path(candidate).exists():
                motor_map_csv = str(Path(candidate))
                break
    road_rng = np.random.default_rng(2026)

    train_pairs = []
    eval_pairs = []
    for i in range(num_train_roads):
        slope, curve_vlim = build_random_road_profiles(horizon, max_speed_mps, road_rng)
        for style in driver_styles:
            scenario_i = StraightRoadScenario(
                horizon=horizon,
                dt=dt,
                slope_profile=slope,
                curve_speed_limit=curve_vlim,
                scenario_name=f"train_{i}_{style}",
                driver_style=style,
                motor_map_csv=motor_map_csv if motor_map_csv else None,
            )
            ref_i = scenario_i.simulate_driver_reference(style_name=style, rng=road_rng)
            train_pairs.append((scenario_i, ref_i))
    for i in range(num_eval_roads):
        slope, curve_vlim = build_random_road_profiles(horizon, max_speed_mps, road_rng)
        for style in driver_styles:
            scenario_i = StraightRoadScenario(
                horizon=horizon,
                dt=dt,
                slope_profile=slope,
                curve_speed_limit=curve_vlim,
                scenario_name=f"eval_unseen_{i}_{style}",
                driver_style=style,
                motor_map_csv=motor_map_csv if motor_map_csv else None,
            )
            ref_i = scenario_i.simulate_driver_reference(style_name=style, rng=road_rng)
            eval_pairs.append((scenario_i, ref_i))

    canonical_eval_ref = eval_pairs[0][1]
    baseline_eval_epd = float(np.mean([x[1]["energy_per_dist"] for x in eval_pairs]))
    baseline_eval_total_energy = float(np.mean([float(x[1]["total_energy"]) for x in eval_pairs]))
    plot_motor_efficiency_map(train_pairs[0][0], output_path=output_dir / "motor_efficiency_map_reference.png")
    if os.getenv("EXPORT_MOTOR_MAP_TEMPLATE", "0") == "1":
        export_motor_map_csv_template(train_pairs[0][0], output_dir / "motor_efficiency_map_template.csv")
    ppo_cfg = copy.deepcopy(DEFAULT_PPO_CFG)
    ppo_env_map = {
        "init_explore_decay": "PPO_INIT_EXPLORE_DECAY",
        "min_explore": "PPO_MIN_EXPLORE",
        "max_explore": "PPO_MAX_EXPLORE",
        "entropy_coef": "PPO_ENTROPY_COEF",
        "entropy_coef_min": "PPO_ENTROPY_COEF_MIN",
        "entropy_coef_max": "PPO_ENTROPY_COEF_MAX",
        "entropy_adapt_rate": "PPO_ENTROPY_ADAPT_RATE",
        "entropy_kl_low_ratio": "PPO_ENTROPY_KL_LOW_RATIO",
        "entropy_kl_high_ratio": "PPO_ENTROPY_KL_HIGH_RATIO",
        "explore_expand": "PPO_EXPLORE_EXPAND",
        "explore_shrink": "PPO_EXPLORE_SHRINK",
        "std_offset": "PPO_STD_OFFSET",
        "min_std": "PPO_MIN_STD",
        "max_std": "PPO_MAX_STD",
        "init_std_bias": "PPO_INIT_STD_BIAS",
        "use_return_rms": "PPO_USE_RETURN_RMS",
        "return_rms_eps": "PPO_RETURN_RMS_EPS",
        "return_rms_clip": "PPO_RETURN_RMS_CLIP",
    }
    ppo_env_overrides = {}
    for cfg_key, env_name in ppo_env_map.items():
        env_val = os.getenv(env_name, "").strip()
        if env_val != "":
            ppo_env_overrides[cfg_key] = float(env_val)
    memory_cfg = copy.deepcopy(DEFAULT_MEMORY_CFG)
    obs_stack_env = os.getenv("OBS_STACK", "").strip()
    if obs_stack_env != "":
        memory_cfg["obs_stack"] = max(1, int(obs_stack_env))
    single_style_focus = (not multi_style) and (len(driver_styles) == 1)
    single_style_name = driver_styles[0] if single_style_focus else None
    single_style_cfg = copy.deepcopy(DEFAULT_SINGLE_STYLE_CFG)
    if single_style_name in SINGLE_STYLE_CFG:
        single_style_cfg.update(copy.deepcopy(SINGLE_STYLE_CFG[single_style_name]))
    style_ppo_cfg = single_style_cfg.get("ppo")
    if isinstance(style_ppo_cfg, dict):
        ppo_cfg.update({k: float(v) for k, v in style_ppo_cfg.items()})
    if ppo_env_overrides:
        ppo_cfg.update(ppo_env_overrides)
    lagrange_update_controller = os.getenv("LAGRANGE_UPDATE_CONTROLLER", "integral").strip().lower()
    if lagrange_update_controller not in {"integral", "pid"}:
        lagrange_update_controller = "integral"
    lagrange_pid_cfg = {
        "kp_scale": float(os.getenv("LAGRANGE_PID_KP_SCALE", "0.35")),
        "kd_scale": float(os.getenv("LAGRANGE_PID_KD_SCALE", "0.10")),
        "integral_decay": float(os.getenv("LAGRANGE_PID_INTEGRAL_DECAY", "0.90")),
        "integral_clip": float(os.getenv("LAGRANGE_PID_INTEGRAL_CLIP", "6.0")),
        "delta_clip": float(os.getenv("LAGRANGE_PID_DELTA_CLIP", "1.5")),
    }

    print("===== Driver Baseline on Unseen Roads (No RL) =====")
    mode_name = "FAST_RUN" if fast_run else ("LONG_RUN" if long_run else "FULL_RUN")
    print(f"Mode: {mode_name}")
    print(f"Driver style mode: {'MULTI_STYLE' if multi_style else 'SINGLE_STYLE'} ({','.join(driver_styles)})")
    print(f"Eval roads x styles: {num_eval_roads} x {len(driver_styles)} = {len(eval_pairs)}")
    print(f"Driver mean total energy (SOC drop): {baseline_eval_total_energy:.6f}")
    print(f"Driver mean E/Dist: {baseline_eval_epd:.8f}")
    print(f"Motor efficiency source: {train_pairs[0][0].motor_map_source}")
    print(f"Motor map csv: {motor_map_csv if motor_map_csv else 'None'}")
    print(f"Observation memory cfg: obs_stack={memory_cfg['obs_stack']}")
    print(
        "PPO policy cfg: "
        f"explore_init={ppo_cfg['init_explore_decay']:.3f}, explore_range=[{ppo_cfg['min_explore']:.3f}, {ppo_cfg['max_explore']:.3f}], "
        f"entropy_coef={ppo_cfg['entropy_coef']:.5f}, entropy_range=[{ppo_cfg['entropy_coef_min']:.5f}, {ppo_cfg['entropy_coef_max']:.5f}], "
        f"adapt_rate={ppo_cfg['entropy_adapt_rate']:.3f}, kl_band=[{ppo_cfg['entropy_kl_low_ratio']:.2f}, {ppo_cfg['entropy_kl_high_ratio']:.2f}], "
        f"std=[{ppo_cfg['min_std']:.3f}, {ppo_cfg['max_std']:.3f}], std_offset={ppo_cfg['std_offset']:.3f}, "
        f"init_std_bias={ppo_cfg['init_std_bias']:.3f}, "
        f"return_rms={'on' if float(ppo_cfg['use_return_rms']) >= 0.5 else 'off'} "
        f"(eps={ppo_cfg['return_rms_eps']:.0e}, clip={ppo_cfg['return_rms_clip']:.1f})"
    )
    print(
        "Lagrange controller cfg: "
        f"mode={lagrange_update_controller}, "
        f"pid=(kp={lagrange_pid_cfg['kp_scale']:.2f}, kd={lagrange_pid_cfg['kd_scale']:.2f}, "
        f"i_decay={lagrange_pid_cfg['integral_decay']:.2f}, i_clip={lagrange_pid_cfg['integral_clip']:.1f}, "
        f"delta_clip={lagrange_pid_cfg['delta_clip']:.1f})"
    )
    canonical_avg_speed_mps = float(canonical_eval_ref["speed"][1:].mean())
    print(f"Canonical eval avg speed: {canonical_avg_speed_mps:.3f} m/s ({canonical_avg_speed_mps * MPS_TO_KMH:.2f} km/h)")
    energy_stage_cfg = single_style_cfg["energy_stage"]
    net_saving_floor_pct = float(single_style_cfg["net_saving_floor_pct"])
    recover_drop_tol_pct = float(single_style_cfg["recover_drop_tol_pct"])
    tracking_bias_speed_limit_pct = float(
        single_style_cfg.get("tracking_bias_speed_limit_pct", DEFAULT_SINGLE_STYLE_CFG["tracking_bias_speed_limit_pct"])
    )
    tracking_bias_dist_limit_pct = float(
        single_style_cfg.get("tracking_bias_dist_limit_pct", DEFAULT_SINGLE_STYLE_CFG["tracking_bias_dist_limit_pct"])
    )
    train_limits_strict = {
        "speed_mae_limit": 0.9,
        "dist_mae_limit": 4.8,
        "speed_rel_limit": 1.98,
        "dist_rel_limit": 1.98,
        "max_negative_speed_bias_pct": tracking_bias_speed_limit_pct,
        "max_negative_dist_bias_pct": tracking_bias_dist_limit_pct,
        "smooth_delta_ratio_limit": 1.18,
        "smooth_jerk_ratio_limit": 1.25,
    }
    eval_limits_strict = {
        "speed_mae_limit": 0.9,
        "dist_mae_limit": 4.8,
        "speed_rel_limit": 2.0,
        "dist_rel_limit": 2.0,
        "max_negative_speed_bias_pct": tracking_bias_speed_limit_pct,
        "max_negative_dist_bias_pct": tracking_bias_dist_limit_pct,
        "smooth_delta_ratio_limit": 1.18,
        "smooth_jerk_ratio_limit": 1.25,
    }
    if single_style_focus and (single_style_name == "normal"):
        experiment_constraint_names = ("speed", "distance", "smoothness", "net_energy", "projection", "underspeed")
    else:
        experiment_constraint_names = ("speed", "distance", "smoothness", "net_energy", "projection")

    seed_list_env = os.getenv("SEED_LIST", "").strip()
    if seed_list_env:
        seeds = [int(x.strip()) for x in seed_list_env.split(",") if x.strip() != ""]
        if len(seeds) == 0:
            raise ValueError("SEED_LIST is set but no valid seed is provided.")
    elif fast_run:
        seeds = [0, 1, 2]
    else:
        default_seed_count = 7 if long_run else 5
        seed_count = int(os.getenv("SEED_COUNT", str(default_seed_count)))
        seed_count = max(1, seed_count)
        seeds = list(range(seed_count))
    if (not seed_list_env) and bool(single_style_cfg.get("drop_seed_zero", False)) and (len(seeds) > 1):
        # 在激进单风格训练中，随机种子 0 往往会收敛到接近纯跟踪的局部最优。
        seeds = [s for s in seeds if s != 0]
        if len(seeds) == 0:
            seeds = [0]
    all_return_curves = []
    all_e_per_dist_curves = []
    eval_stats_list = []
    lagrange_s_last = []
    lagrange_d_last = []
    lagrange_sm_last = []
    lagrange_n_last = []
    lagrange_p_last = []
    lagrange_u_last = []
    lagrange_w_last = []

    best_rollout = None
    model_output_path = output_dir / "best_motor_ppo.pt"
    best_seed = None
    best_score = -1e18
    best_has_feasible = False
    best_feasible_metric = -1e18
    best_feasible_raw_robust = -1e18
    best_feasible_mean_saving = -1e18
    best_fallback_score = -1e18
    best_torque_delta = 1e18
    best_torque_jerk = 1e18
    best_rollout_ref = None
    best_rollout_metrics = None

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = DriverReferenceEnergyEnv(
            train_pairs[0][0],
            train_pairs[0][1],
            residual_limit=residual_bound,
            obs_stack=memory_cfg["obs_stack"],
        )
        agent = PPOContinuous(
            env.state_dim,
            hidden_dim,
            env.action_dim,
            actor_lr,
            critic_lr,
            lmbda,
            epochs,
            eps,
            gamma,
            device,
            action_bound=np.array([residual_bound, 1.0, 1.0], dtype=np.float32),
            base_state_dim=env.base_state_dim,
            minibatch_size=64,
            target_kl=0.01,
            constraint_names=experiment_constraint_names,
            **ppo_cfg,
        )

        def seed_eval_track_save():
            """
            作用：判断当前策略是否满足评估阶段的跟踪与节能门槛。
            输入：无，闭包内直接使用 env、agent 和 eval_pairs。
            输出：track_ok 与 save_ok 两个布尔值。
            """
            eval_entries = evaluate_agent_on_pairs(
                env,
                agent,
                eval_pairs,
                **eval_limits_strict,
            )
            all_track_ok = all(x["metrics"]["tracking_ok"] for x in eval_entries)
            all_save_ok = all(
                (float(x["metrics"]["saving_total_pct"]) > 0.0)
                and (float(x["metrics"]["saving_epd_pct"]) > 0.0)
                and (float(x["metrics"]["saving_net_epd_pct"]) >= net_saving_floor_pct)
                and (float(x["metrics"]["recover_delta_pct"]) >= -recover_drop_tol_pct)
                and bool(x["metrics"]["smooth_ok"])
                for x in eval_entries
            )
            return all_track_ok, all_save_ok

        def seed_eval_bias_guard():
            """
            作用：计算当前策略是否满足 slow-bias guard。
            输入：无，闭包内直接使用 env、agent 和 eval_pairs。
            输出：bias_ok、speed_rel_bias、dist_rel_bias。
            """
            eval_entries = evaluate_agent_on_pairs(
                env,
                agent,
                eval_pairs,
                **eval_limits_strict,
            )
            speed_rel_bias_each = [float(x["metrics"]["speed_rel_bias_pct"]) for x in eval_entries]
            dist_rel_bias_each = [float(x["metrics"]["dist_rel_bias_pct"]) for x in eval_entries]
            speed_rel_bias = float(np.mean(speed_rel_bias_each))
            dist_rel_bias = float(np.mean(dist_rel_bias_each))
            bias_guard_speed_floor = float(
                single_style_cfg.get("bias_guard_speed_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_speed_floor"])
            )
            bias_guard_dist_floor = float(
                single_style_cfg.get("bias_guard_dist_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_dist_floor"])
            )
            bias_ok = bool((speed_rel_bias >= bias_guard_speed_floor) and (dist_rel_bias >= bias_guard_dist_floor))
            return bias_ok, speed_rel_bias, dist_rel_bias

        def seed_eval_summary():
            """
            作用：构建单个 seed 的汇总指标，用于报告和模型选择。
            输入：无，闭包内直接使用 env、agent 和 eval_pairs。
            输出：包含节能、平滑、偏差和鲁棒性指标的字典。
            """
            eval_entries = evaluate_agent_on_pairs(
                env,
                agent,
                eval_pairs,
                **eval_limits_strict,
            )
            track_flags = [bool(x["metrics"]["tracking_ok"]) for x in eval_entries]
            saving_total_each = [float(x["metrics"].get("saving_total_isochronous_pct", x["metrics"]["saving_total_pct"])) for x in eval_entries]
            saving_epd_each = [float(x["metrics"].get("saving_isochronous_pct", x["metrics"]["saving_epd_pct"])) for x in eval_entries]
            saving_net_epd_each = [float(x["metrics"].get("saving_net_isochronous_pct", x["metrics"]["saving_net_epd_pct"])) for x in eval_entries]
            recover_delta_each = [float(x["metrics"]["recover_delta_pct"]) for x in eval_entries]
            smooth_ok_each = [bool(x["metrics"]["smooth_ok"]) for x in eval_entries]
            smooth_delta_ratio_each = [float(x["metrics"]["smooth_delta_ratio"]) for x in eval_entries]
            smooth_jerk_ratio_each = [float(x["metrics"]["smooth_jerk_ratio"]) for x in eval_entries]
            style_saving = {}
            style_saving_total = {}
            style_saving_net = {}
            style_recover_delta = {}
            style_smooth_ok = {}
            for x in eval_entries:
                style_name = str(getattr(x.get("scenario"), "driver_style", "normal"))
                style_saving_total.setdefault(style_name, []).append(float(x["metrics"].get("saving_total_isochronous_pct", x["metrics"]["saving_total_pct"])))
                style_saving.setdefault(style_name, []).append(float(x["metrics"].get("saving_isochronous_pct", x["metrics"]["saving_epd_pct"])))
                style_saving_net.setdefault(style_name, []).append(float(x["metrics"].get("saving_net_isochronous_pct", x["metrics"]["saving_net_epd_pct"])))
                style_recover_delta.setdefault(style_name, []).append(float(x["metrics"]["recover_delta_pct"]))
                style_smooth_ok.setdefault(style_name, []).append(bool(x["metrics"]["smooth_ok"]))
            style_mean_saving_total = [float(np.mean(v)) for v in style_saving_total.values()]
            style_mean_saving = [float(np.mean(v)) for v in style_saving.values()]
            style_mean_saving_net = [float(np.mean(v)) for v in style_saving_net.values()]
            style_mean_recover_delta = [float(np.mean(v)) for v in style_recover_delta.values()]
            mean_saving_total = float(np.mean(saving_total_each))
            mean_saving = float(np.mean(saving_epd_each))
            mean_saving_net = float(np.mean(saving_net_epd_each))
            worst_saving_total = float(np.min(saving_total_each))
            worst_saving = float(np.min(saving_epd_each))
            worst_saving_net = float(np.min(saving_net_epd_each))
            worst_recover_delta = float(np.min(recover_delta_each))
            worst_style_saving_total = (
                float(np.min(style_mean_saving_total)) if len(style_mean_saving_total) > 0 else worst_saving_total
            )
            worst_style_saving = float(np.min(style_mean_saving)) if len(style_mean_saving) > 0 else worst_saving
            worst_style_saving_net = (
                float(np.min(style_mean_saving_net)) if len(style_mean_saving_net) > 0 else worst_saving_net
            )
            worst_style_recover_delta = (
                float(np.min(style_mean_recover_delta)) if len(style_mean_recover_delta) > 0 else worst_recover_delta
            )
            robust_case_saving = float(0.70 * mean_saving + 0.30 * worst_saving)
            robust_style_saving = float(0.45 * mean_saving + 0.35 * worst_style_saving + 0.20 * worst_saving)
            robust_case_saving_net = float(0.70 * mean_saving_net + 0.30 * worst_saving_net)
            robust_style_saving_net = float(
                0.45 * mean_saving_net + 0.35 * worst_style_saving_net + 0.20 * worst_saving_net
            )
            robust_case_saving_total = float(0.70 * mean_saving_total + 0.30 * worst_saving_total)
            robust_style_saving_total = float(
                0.45 * mean_saving_total + 0.35 * worst_style_saving_total + 0.20 * worst_saving_total
            )
            robust_combined_saving = float(
                0.50 * robust_style_saving_total + 0.30 * robust_style_saving + 0.20 * robust_style_saving_net
            )
            return {
                "tracking_ok": bool(np.all(track_flags)),
                "saving_ok": bool(
                    np.all(np.array(saving_total_each) > 0.0)
                    and np.all(np.array(saving_epd_each) > 0.0)
                    and np.all(np.array(saving_net_epd_each) >= net_saving_floor_pct)
                    and np.all(np.array(recover_delta_each) >= -recover_drop_tol_pct)
                    and np.all(np.array(smooth_ok_each))
                    and all(bool(np.all(v)) for v in style_smooth_ok.values())
                ),
                "smooth_ok": bool(np.all(np.array(smooth_ok_each))),
                "smooth_delta_ratio_mean": float(np.mean(smooth_delta_ratio_each)),
                "smooth_jerk_ratio_mean": float(np.mean(smooth_jerk_ratio_each)),
                "smooth_delta_ratio_worst": float(np.max(smooth_delta_ratio_each)),
                "smooth_jerk_ratio_worst": float(np.max(smooth_jerk_ratio_each)),
                "mean_saving_total": mean_saving_total,
                "mean_saving": mean_saving,
                "mean_saving_net": mean_saving_net,
                "worst_saving_total": worst_saving_total,
                "worst_saving": worst_saving,
                "worst_saving_net": worst_saving_net,
                "worst_recover_delta": worst_recover_delta,
                "worst_style_saving_total": worst_style_saving_total,
                "worst_style_saving": worst_style_saving,
                "worst_style_saving_net": worst_style_saving_net,
                "worst_style_recover_delta": worst_style_recover_delta,
                "robust_case_saving_total": robust_case_saving_total,
                "robust_style_saving_total": robust_style_saving_total,
                "robust_case_saving": robust_case_saving,
                "robust_style_saving": robust_style_saving,
                "robust_case_saving_net": robust_case_saving_net,
                "robust_style_saving_net": robust_style_saving_net,
                "robust_combined_saving": robust_combined_saving,
            }

        def run_energy_stage(
            episodes,
            stage_name,
            e_start,
            e_end,
            lr_s,
            lr_d,
            tgt_s,
            tgt_d,
            early_stop_patience,
            eval_interval,
        ):
            """
            作用：按给定配置运行当前 seed 的一个节能训练阶段。
            输入：episodes、stage_name 及各类能量阶段超参数。
            输出：train_stage 返回的四个历史序列。
            """
            return train_stage(
                env,
                agent,
                episodes,
                stage_name=stage_name,
                scenario_refs=train_pairs,
                eval_pairs=eval_pairs,
                use_lagrange=True,
                energy_weight_start=e_start,
                energy_weight_end=e_end,
                lagrange_lr_speed=lr_s,
                lagrange_lr_dist=lr_d,
                lagrange_update_controller=lagrange_update_controller,
                lagrange_pid_kp_scale=lagrange_pid_cfg["kp_scale"],
                lagrange_pid_kd_scale=lagrange_pid_cfg["kd_scale"],
                lagrange_pid_integral_decay=lagrange_pid_cfg["integral_decay"],
                lagrange_pid_integral_clip=lagrange_pid_cfg["integral_clip"],
                lagrange_pid_delta_clip=lagrange_pid_cfg["delta_clip"],
                target_speed_violation=tgt_s,
                target_dist_violation=tgt_d,
                early_stop_patience=early_stop_patience,
                eval_interval=eval_interval,
                **train_limits_strict,
            )

        def merge_stage_outputs(
            ret_buf,
            e_buf,
            lam_s_buf,
            lam_d_buf,
            ret_new,
            e_new,
            lam_s_new,
            lam_d_new,
        ):
            """
            作用：把某个阶段的历史结果追加到当前 seed 的总缓冲区。
            输入：ret/e/lam_s/lam_d 的旧缓冲区与新结果。
            输出：无。
            """
            ret_buf.extend(ret_new)
            e_buf.extend(e_new)
            if lam_s_new:
                lam_s_buf.extend(lam_s_new)
            if lam_d_new:
                lam_d_buf.extend(lam_d_new)

        def snapshot_policy_state():
            """
            作用：保存当前策略及相关标量状态，供后续回滚。
            输入：无。
            输出：可用于恢复的状态字典。
            """
            return {
                "actor": copy.deepcopy(agent.actor.state_dict()),
                "critic": copy.deepcopy(agent.critic.state_dict()),
                "agent_aux": copy.deepcopy(agent.get_auxiliary_state_dict()),
                "constraint_multipliers": env.get_constraint_multipliers(),
                "energy_weight": float(env.energy_weight),
                "explore_decay": float(agent.explore_decay),
                "entropy_coef": float(agent.entropy_coef),
            }

        def restore_policy_state(state):
            """
            作用：恢复先前保存的策略与 checkpoint 状态。
            输入：state: 由 snapshot_policy_state 生成的状态字典。
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

        def current_lagrange_values():
            multipliers = env.get_constraint_multipliers()
            return (
                float(multipliers.get("speed", 0.0)),
                float(multipliers.get("distance", 0.0)),
                float(multipliers.get("smoothness", 0.0)),
                float(multipliers.get("net_energy", 0.0)),
                float(multipliers.get("projection", 0.0)),
                float(multipliers.get("underspeed", 0.0)),
                float(multipliers.get("window_energy", 0.0)),
            )

        def append_lagrange_snapshot():
            lam_s, lam_d, lam_sm, lam_n, lam_p, lam_u, lam_w = current_lagrange_values()
            lagrange_s_last.append(lam_s)
            lagrange_d_last.append(lam_d)
            lagrange_sm_last.append(lam_sm)
            lagrange_n_last.append(lam_n)
            lagrange_p_last.append(lam_p)
            lagrange_u_last.append(lam_u)
            lagrange_w_last.append(lam_w)

        def overwrite_lagrange_snapshot():
            lam_s, lam_d, lam_sm, lam_n, lam_p, lam_u, lam_w = current_lagrange_values()
            lagrange_s_last[-1] = lam_s
            lagrange_d_last[-1] = lam_d
            lagrange_sm_last[-1] = lam_sm
            lagrange_n_last[-1] = lam_n
            lagrange_p_last[-1] = lam_p
            lagrange_u_last[-1] = lam_u
            lagrange_w_last[-1] = lam_w

        env.set_stage("track", energy_weight=0.0, lambda_speed=0.0, lambda_dist=0.0)
        ret_track, e_track, _, _ = train_stage(
            env,
            agent,
            track_episodes,
            stage_name=f"Seed{seed}-Track",
            scenario_refs=train_pairs,
            eval_pairs=eval_pairs,
            use_lagrange=False,
            energy_weight_start=0.0,
            energy_weight_end=0.0,
        )
        track_stage_ckpt = {
            "actor": copy.deepcopy(agent.actor.state_dict()),
            "critic": copy.deepcopy(agent.critic.state_dict()),
            "agent_aux": copy.deepcopy(agent.get_auxiliary_state_dict()),
            "constraint_multipliers": env.get_constraint_multipliers(),
            "explore_decay": float(agent.explore_decay),
            "entropy_coef": float(agent.entropy_coef),
        }

        env.set_stage("energy", energy_weight=0.6, lambda_speed=0.0, lambda_dist=0.0)
        ret_energy, e_energy, lam_s_hist, lam_d_hist = run_energy_stage(
            episodes=energy_episodes,
            stage_name=f"Seed{seed}-Energy",
            e_start=energy_stage_cfg["e_start"],
            e_end=energy_stage_cfg["e_end"],
            lr_s=energy_stage_cfg["lr_s"],
            lr_d=energy_stage_cfg["lr_d"],
            tgt_s=energy_stage_cfg["tgt_s"],
            tgt_d=energy_stage_cfg["tgt_d"],
            early_stop_patience=energy_stage_cfg["patience"],
            eval_interval=energy_stage_cfg["eval_interval"],
        )
        ret_energy_all = list(ret_energy)
        e_energy_all = list(e_energy)
        if single_style_focus:
            polish_cfg = single_style_cfg.get("polish")
            if polish_cfg is not None:
                if fast_run:
                    polish_episodes = int(polish_cfg["episodes_fast"])
                else:
                    polish_episodes = int(polish_cfg["episodes_full"])
                agent.explore_decay = max(agent.min_explore, min(agent.explore_decay, float(polish_cfg["max_explore"])))
                env.set_stage(
                    "energy",
                    energy_weight=float(polish_cfg["stage_weight"]),
                    lambda_speed=0.0,
                    lambda_dist=0.0,
                )
                ret_polish, e_polish, lam_s_hist_polish, lam_d_hist_polish = run_energy_stage(
                    episodes=polish_episodes,
                    stage_name=f"Seed{seed}-{polish_cfg['stage_name']}",
                    e_start=float(polish_cfg["e_start"]),
                    e_end=float(polish_cfg["e_end"]),
                    lr_s=float(polish_cfg["lr_s"]),
                    lr_d=float(polish_cfg["lr_d"]),
                    tgt_s=float(polish_cfg["tgt_s"]),
                    tgt_d=float(polish_cfg["tgt_d"]),
                    early_stop_patience=int(polish_cfg["patience"]),
                    eval_interval=int(polish_cfg["eval_interval"]),
                )
                merge_stage_outputs(
                    ret_energy_all,
                    e_energy_all,
                    lam_s_hist,
                    lam_d_hist,
                    ret_polish,
                    e_polish,
                    lam_s_hist_polish,
                    lam_d_hist_polish,
                )

        env.set_stage("energy", energy_weight=energy_stage_cfg["e_end"])
        seed_track_ok, seed_save_ok = seed_eval_track_save()
        used_bias_repair = False
        if (
            single_style_focus
            and (single_style_name == "normal")
            and seed_track_ok
            and seed_save_ok
            and isinstance(single_style_cfg.get("bias_repair"), dict)
        ):
            bias_ok, speed_rel_bias_before, dist_rel_bias_before = seed_eval_bias_guard()
            if not bias_ok:
                bias_cfg = single_style_cfg["bias_repair"]
                bias_ckpt = snapshot_policy_state()
                bias_guard_speed_floor = float(
                    single_style_cfg.get("bias_guard_speed_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_speed_floor"])
                )
                bias_guard_dist_floor = float(
                    single_style_cfg.get("bias_guard_dist_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_dist_floor"])
                )
                bias_margin_before = min(
                    speed_rel_bias_before - bias_guard_speed_floor,
                    dist_rel_bias_before - bias_guard_dist_floor,
                )
                if fast_run:
                    bias_episodes = int(bias_cfg["episodes_fast"])
                else:
                    bias_episodes = int(bias_cfg["episodes_full"])
                agent.explore_decay = max(agent.min_explore, min(agent.explore_decay, float(bias_cfg["max_explore"])))
                env.set_stage(
                    "energy",
                    energy_weight=float(bias_cfg["stage_weight"]),
                    lambda_speed=0.0,
                    lambda_dist=0.0,
                )
                ret_bias, e_bias, lam_s_hist_bias, lam_d_hist_bias = run_energy_stage(
                    episodes=bias_episodes,
                    stage_name=f"Seed{seed}-{bias_cfg['stage_name']}",
                    e_start=float(bias_cfg["e_start"]),
                    e_end=float(bias_cfg["e_end"]),
                    lr_s=float(bias_cfg["lr_s"]),
                    lr_d=float(bias_cfg["lr_d"]),
                    tgt_s=float(bias_cfg["tgt_s"]),
                    tgt_d=float(bias_cfg["tgt_d"]),
                    early_stop_patience=int(bias_cfg["patience"]),
                    eval_interval=int(bias_cfg["eval_interval"]),
                )
                env.set_stage("energy", energy_weight=energy_stage_cfg["e_end"])
                seed_track_ok, seed_save_ok = seed_eval_track_save()
                bias_ok_after, speed_rel_bias_after, dist_rel_bias_after = seed_eval_bias_guard()
                bias_margin_after = min(
                    speed_rel_bias_after - bias_guard_speed_floor,
                    dist_rel_bias_after - bias_guard_dist_floor,
                )
                bias_margin_improved = bias_margin_after > (bias_margin_before + 0.015)
                if seed_track_ok and seed_save_ok and (bias_ok_after or bias_margin_improved):
                    merge_stage_outputs(
                        ret_energy_all,
                        e_energy_all,
                        lam_s_hist,
                        lam_d_hist,
                        ret_bias,
                        e_bias,
                        lam_s_hist_bias,
                        lam_d_hist_bias,
                    )
                    used_bias_repair = True
                else:
                    restore_policy_state(bias_ckpt)
                    env.set_stage("energy", energy_weight=energy_stage_cfg["e_end"])
                    seed_track_ok, seed_save_ok = seed_eval_track_save()
        used_track_repair = False
        used_safe_energy = False
        if not (seed_track_ok and seed_save_ok):
            rescue_plan = (
                [
                    {"episodes": 28, "e_start": 1.90, "e_end": 2.20, "lr_s": 0.45, "lr_d": 0.06, "tgt_s": 0.10, "tgt_d": 0.85},
                    {"episodes": 22, "e_start": 1.45, "e_end": 1.80, "lr_s": 0.55, "lr_d": 0.08, "tgt_s": 0.08, "tgt_d": 0.72},
                    {"episodes": 18, "e_start": 1.05, "e_end": 1.25, "lr_s": 0.70, "lr_d": 0.14, "tgt_s": 0.06, "tgt_d": 0.55},
                ]
                if fast_run
                else [
                    {"episodes": 65, "e_start": 1.95, "e_end": 2.25, "lr_s": 0.45, "lr_d": 0.06, "tgt_s": 0.10, "tgt_d": 0.85},
                    {"episodes": 50, "e_start": 1.50, "e_end": 1.85, "lr_s": 0.55, "lr_d": 0.08, "tgt_s": 0.08, "tgt_d": 0.72},
                    {"episodes": 40, "e_start": 1.05, "e_end": 1.30, "lr_s": 0.70, "lr_d": 0.14, "tgt_s": 0.06, "tgt_d": 0.55},
                ]
            )
            for rescue_idx, cfg in enumerate(rescue_plan, start=1):
                if seed_track_ok and seed_save_ok:
                    break
                env.set_stage("energy", energy_weight=cfg["e_start"])
                ret_rescue, e_rescue, lam_s_hist_rescue, lam_d_hist_rescue = run_energy_stage(
                    episodes=cfg["episodes"],
                    stage_name=f"Seed{seed}-Rescue{rescue_idx}",
                    e_start=cfg["e_start"],
                    e_end=cfg["e_end"],
                    lr_s=cfg["lr_s"],
                    lr_d=cfg["lr_d"],
                    tgt_s=cfg["tgt_s"],
                    tgt_d=cfg["tgt_d"],
                    early_stop_patience=5,
                    eval_interval=10,
                )
                merge_stage_outputs(
                    ret_energy_all,
                    e_energy_all,
                    lam_s_hist,
                    lam_d_hist,
                    ret_rescue,
                    e_rescue,
                    lam_s_hist_rescue,
                    lam_d_hist_rescue,
                )
                seed_track_ok, seed_save_ok = seed_eval_track_save()
            if not seed_track_ok:
                used_track_repair = True
                repair_track_episodes = 22 if fast_run else 45
                env.set_stage("track", energy_weight=0.0, lambda_speed=0.0, lambda_dist=0.0)
                ret_repair_track, e_repair_track, _, _ = train_stage(
                    env,
                    agent,
                    repair_track_episodes,
                    stage_name=f"Seed{seed}-TrackRepair",
                    scenario_refs=train_pairs,
                    eval_pairs=eval_pairs,
                    use_lagrange=False,
                    energy_weight_start=0.0,
                    energy_weight_end=0.0,
                )
                ret_energy_all.extend(ret_repair_track)
                e_energy_all.extend(e_repair_track)

                repair_energy_episodes = 18 if fast_run else 35
                env.set_stage("energy", energy_weight=1.20, lambda_speed=0.0, lambda_dist=0.0)
                ret_repair_energy, e_repair_energy, lam_s_hist_repair, lam_d_hist_repair = run_energy_stage(
                    episodes=repair_energy_episodes,
                    stage_name=f"Seed{seed}-EnergyRepair",
                    e_start=1.20,
                    e_end=1.80,
                    lr_s=0.65,
                    lr_d=0.12,
                    tgt_s=0.05,
                    tgt_d=0.40,
                    early_stop_patience=4,
                    eval_interval=8,
                )
                merge_stage_outputs(
                    ret_energy_all,
                    e_energy_all,
                    lam_s_hist,
                    lam_d_hist,
                    ret_repair_energy,
                    e_repair_energy,
                    lam_s_hist_repair,
                    lam_d_hist_repair,
                )

        seed_summary = seed_eval_summary()
        is_sport_single_style = bool(single_style_focus and (single_style_name == "sport"))
        is_eco_single_style = bool(single_style_focus and (single_style_name == "eco"))
        boost_threshold = 0.40 if fast_run else 0.45
        if is_sport_single_style:
            boost_threshold = 0.52 if fast_run else 0.58
        elif is_eco_single_style:
            boost_threshold = 0.45 if fast_run else 0.52
        if seed_summary["tracking_ok"] and seed_summary["saving_ok"] and (seed_summary["mean_saving"] < boost_threshold):
            boost_ckpt = snapshot_policy_state()
            boost_plan = [
                {
                    "name": "EfficiencyBoost",
                    "episodes": 8 if fast_run else 22,
                    "e_start": 1.20,
                    "e_end": 1.55,
                    "lr_s": 0.48,
                    "lr_d": 0.07,
                    "tgt_s": 0.08,
                    "tgt_d": 0.65,
                    "patience": 3,
                    "eval_interval": 6,
                }
            ]
            low_saving_trigger = 0.30 if fast_run else 0.32
            if is_sport_single_style:
                boost_plan = [
                    {
                        "name": "EfficiencyBoost",
                        "episodes": 10 if fast_run else 28,
                        "e_start": 1.28,
                        "e_end": 1.85,
                        "lr_s": 0.46,
                        "lr_d": 0.07,
                        "tgt_s": 0.08,
                        "tgt_d": 0.60,
                        "patience": 3,
                        "eval_interval": 6,
                    }
                ]
                low_saving_trigger = 0.42 if fast_run else 0.48
            elif is_eco_single_style:
                boost_plan = [
                    {
                        "name": "EfficiencyBoost",
                        "episodes": 12 if fast_run else 34,
                        "e_start": 1.25,
                        "e_end": 1.95,
                        "lr_s": 0.34,
                        "lr_d": 0.035,
                        "tgt_s": 0.09,
                        "tgt_d": 0.64,
                        "patience": 4,
                        "eval_interval": 8,
                    }
                ]
                low_saving_trigger = 0.38 if fast_run else 0.44
            if seed_summary["mean_saving"] < low_saving_trigger:
                boost_plan.append(
                    {
                        "name": "EfficiencyBoostStrong",
                        "episodes": 12 if fast_run else 32,
                        "e_start": 1.25,
                        "e_end": 1.85,
                        "lr_s": 0.44,
                        "lr_d": 0.06,
                        "tgt_s": 0.07,
                        "tgt_d": 0.55,
                        "patience": 4,
                        "eval_interval": 8,
                    }
                )
                if is_sport_single_style:
                    boost_plan[-1] = {
                        "name": "EfficiencyBoostStrong",
                        "episodes": 16 if fast_run else 44,
                        "e_start": 1.45,
                        "e_end": 2.20,
                        "lr_s": 0.42,
                        "lr_d": 0.06,
                        "tgt_s": 0.08,
                        "tgt_d": 0.62,
                        "patience": 4,
                        "eval_interval": 8,
                    }
                elif is_eco_single_style:
                    boost_plan[-1] = {
                        "name": "EfficiencyBoostStrong",
                        "episodes": 20 if fast_run else 54,
                        "e_start": 1.45,
                        "e_end": 2.15,
                        "lr_s": 0.30,
                        "lr_d": 0.030,
                        "tgt_s": 0.09,
                        "tgt_d": 0.66,
                        "patience": 4,
                        "eval_interval": 8,
                    }

            best_boost = None
            for cfg in boost_plan:
                restore_policy_state(boost_ckpt)
                env.set_stage("energy", energy_weight=cfg["e_start"], lambda_speed=0.0, lambda_dist=0.0)
                ret_boost, e_boost, lam_s_hist_boost, lam_d_hist_boost = run_energy_stage(
                    episodes=cfg["episodes"],
                    stage_name=f"Seed{seed}-{cfg['name']}",
                    e_start=cfg["e_start"],
                    e_end=cfg["e_end"],
                    lr_s=cfg["lr_s"],
                    lr_d=cfg["lr_d"],
                    tgt_s=cfg["tgt_s"],
                    tgt_d=cfg["tgt_d"],
                    early_stop_patience=cfg["patience"],
                    eval_interval=cfg["eval_interval"],
                )
                post_boost_summary = seed_eval_summary()
                robust_delta = post_boost_summary["robust_combined_saving"] - seed_summary["robust_combined_saving"]
                total_worst_delta = post_boost_summary["worst_saving_total"] - seed_summary["worst_saving_total"]
                worst_delta = post_boost_summary["worst_saving"] - seed_summary["worst_saving"]
                worst_style_delta = post_boost_summary["worst_style_saving"] - seed_summary["worst_style_saving"]
                net_worst_delta = post_boost_summary["worst_saving_net"] - seed_summary["worst_saving_net"]
                net_worst_style_delta = (
                    post_boost_summary["worst_style_saving_net"] - seed_summary["worst_style_saving_net"]
                )
                recover_worst_delta = post_boost_summary["worst_recover_delta"] - seed_summary["worst_recover_delta"]
                gross_delta = post_boost_summary["mean_saving"] - seed_summary["mean_saving"]
                if is_sport_single_style:
                    boost_accept = (
                        post_boost_summary["tracking_ok"]
                        and post_boost_summary["saving_ok"]
                        and (
                            (
                                gross_delta > 0.03
                                and total_worst_delta >= -0.02
                                and worst_delta >= -0.02
                                and worst_style_delta >= -0.02
                                and net_worst_delta >= -0.03
                                and net_worst_style_delta >= -0.03
                                and recover_worst_delta >= -0.60
                            )
                            or (
                                robust_delta > 0.03
                                and gross_delta >= -0.01
                                and net_worst_delta >= -0.02
                                and net_worst_style_delta >= -0.02
                                and recover_worst_delta >= -0.40
                            )
                        )
                    )
                elif is_eco_single_style:
                    boost_accept = (
                        post_boost_summary["tracking_ok"]
                        and post_boost_summary["saving_ok"]
                        and (
                            (
                                gross_delta > 0.03
                                and total_worst_delta >= -0.02
                                and worst_delta >= -0.02
                                and worst_style_delta >= -0.02
                                and net_worst_delta >= -0.04
                                and net_worst_style_delta >= -0.04
                                and recover_worst_delta >= -0.80
                            )
                            or (
                                robust_delta > 0.03
                                and gross_delta >= -0.01
                                and net_worst_delta >= -0.03
                                and net_worst_style_delta >= -0.03
                                and recover_worst_delta >= -0.60
                            )
                        )
                    )
                else:
                    boost_accept = (
                        post_boost_summary["tracking_ok"]
                        and post_boost_summary["saving_ok"]
                        and (
                            (robust_delta > 0.04)
                            or (
                                robust_delta > 0.0
                                and total_worst_delta >= -0.02
                                and worst_delta >= -0.02
                                and worst_style_delta >= -0.02
                                and net_worst_delta >= -0.02
                                and net_worst_style_delta >= -0.02
                                and recover_worst_delta >= -0.30
                            )
                        )
                    )
                if boost_accept:
                    candidate = {
                        "cfg_name": cfg["name"],
                        "summary": post_boost_summary,
                        "state": snapshot_policy_state(),
                        "ret": ret_boost,
                        "e": e_boost,
                        "lam_s": lam_s_hist_boost,
                        "lam_d": lam_d_hist_boost,
                    }
                    if is_sport_single_style:
                        if (
                            best_boost is None
                            or (candidate["summary"]["mean_saving"] > best_boost["summary"]["mean_saving"] + 1e-12)
                            or (
                                abs(candidate["summary"]["mean_saving"] - best_boost["summary"]["mean_saving"]) <= 1e-12
                                and candidate["summary"]["robust_combined_saving"]
                                > best_boost["summary"]["robust_combined_saving"]
                            )
                        ):
                            best_boost = candidate
                    elif is_eco_single_style:
                        if (
                            best_boost is None
                            or (
                                candidate["summary"]["robust_combined_saving"]
                                > best_boost["summary"]["robust_combined_saving"] + 1e-12
                            )
                            or (
                                abs(
                                    candidate["summary"]["robust_combined_saving"]
                                    - best_boost["summary"]["robust_combined_saving"]
                                )
                                <= 1e-12
                                and candidate["summary"]["mean_saving"] > best_boost["summary"]["mean_saving"]
                            )
                        ):
                            best_boost = candidate
                    else:
                        if (
                            best_boost is None
                            or (
                                candidate["summary"]["robust_combined_saving"]
                                > best_boost["summary"]["robust_combined_saving"] + 1e-12
                            )
                            or (
                                abs(
                                    candidate["summary"]["robust_combined_saving"]
                                    - best_boost["summary"]["robust_combined_saving"]
                                )
                                <= 1e-12
                                and candidate["summary"]["mean_saving"] > best_boost["summary"]["mean_saving"]
                            )
                        ):
                            best_boost = candidate

            if best_boost is not None:
                restore_policy_state(best_boost["state"])
                merge_stage_outputs(
                    ret_energy_all,
                    e_energy_all,
                    lam_s_hist,
                    lam_d_hist,
                    best_boost["ret"],
                    best_boost["e"],
                    best_boost["lam_s"],
                    best_boost["lam_d"],
                )
                print(
                    f"[Seed {seed}] {best_boost['cfg_name']} accepted: "
                    f"robustCombo {seed_summary['robust_combined_saving']:.2f}% -> {best_boost['summary']['robust_combined_saving']:.2f}% "
                    f"(meanGross {seed_summary['mean_saving']:.2f}% -> {best_boost['summary']['mean_saving']:.2f}%, "
                    f"meanNet {seed_summary['mean_saving_net']:.2f}% -> {best_boost['summary']['mean_saving_net']:.2f}%)"
                )
            else:
                restore_policy_state(boost_ckpt)
                print(
                    f"[Seed {seed}] EfficiencyBoost rolled back: "
                    f"robustCombo {seed_summary['robust_combined_saving']:.2f}% -> no acceptable candidate"
                )

        return_list = ret_track + ret_energy_all
        e_per_dist_list = e_track + e_energy_all
        append_lagrange_snapshot()

        env.set_stage("energy", energy_weight=energy_stage_cfg["e_end"])
        final_track_ok, _ = seed_eval_track_save()
        if not final_track_ok:
            used_safe_energy = True
            agent.actor.load_state_dict(track_stage_ckpt["actor"])
            agent.critic.load_state_dict(track_stage_ckpt["critic"])
            agent.load_auxiliary_state_dict(track_stage_ckpt.get("agent_aux"))
            env.set_constraint_multipliers(track_stage_ckpt.get("constraint_multipliers", {}))
            agent.explore_decay = track_stage_ckpt["explore_decay"]
            agent.entropy_coef = float(track_stage_ckpt.get("entropy_coef", agent.entropy_coef))
            safe_energy_episodes = 24 if fast_run else 55
            env.set_stage("energy", energy_weight=0.80, lambda_speed=0.0, lambda_dist=0.0)
            ret_safe, e_safe, lam_s_hist_safe, lam_d_hist_safe = run_energy_stage(
                episodes=safe_energy_episodes,
                stage_name=f"Seed{seed}-SafeEnergy",
                e_start=0.80,
                e_end=1.15,
                lr_s=0.80,
                lr_d=0.16,
                tgt_s=0.05,
                tgt_d=0.50,
                early_stop_patience=4,
                eval_interval=8,
            )
            return_list += ret_safe
            e_per_dist_list += e_safe
            if lam_s_hist_safe:
                lam_s_hist = lam_s_hist_safe
            if lam_d_hist_safe:
                lam_d_hist = lam_d_hist_safe
            overwrite_lagrange_snapshot()
            env.set_stage("energy", energy_weight=1.15)
            safe_ckpt = {
                "actor": copy.deepcopy(agent.actor.state_dict()),
                "critic": copy.deepcopy(agent.critic.state_dict()),
                "agent_aux": copy.deepcopy(agent.get_auxiliary_state_dict()),
                "constraint_multipliers": env.get_constraint_multipliers(),
                "energy_weight": float(env.energy_weight),
                "explore_decay": float(agent.explore_decay),
                "entropy_coef": float(agent.entropy_coef),
            }
            safe_track_ok, safe_save_ok = seed_eval_track_save()
            if safe_track_ok:
                if safe_save_ok:
                    rebound_episodes = 10 if fast_run else 24
                    ret_rebound, e_rebound, lam_s_hist_rebound, lam_d_hist_rebound = run_energy_stage(
                        episodes=rebound_episodes,
                        stage_name=f"Seed{seed}-ReboundEnergy",
                        e_start=1.25,
                        e_end=1.72,
                        lr_s=0.55,
                        lr_d=0.10,
                        tgt_s=0.07,
                        tgt_d=0.65,
                        early_stop_patience=3,
                        eval_interval=6,
                    )
                    rebound_track_ok, rebound_save_ok = seed_eval_track_save()
                    if rebound_track_ok and rebound_save_ok:
                        return_list += ret_rebound
                        e_per_dist_list += e_rebound
                        if lam_s_hist_rebound:
                            lam_s_hist = lam_s_hist_rebound
                        if lam_d_hist_rebound:
                            lam_d_hist = lam_d_hist_rebound
                    else:
                        agent.actor.load_state_dict(safe_ckpt["actor"])
                        agent.critic.load_state_dict(safe_ckpt["critic"])
                        agent.load_auxiliary_state_dict(safe_ckpt.get("agent_aux"))
                        env.set_constraint_multipliers(safe_ckpt.get("constraint_multipliers", {}))
                        env.energy_weight = safe_ckpt["energy_weight"]
                        agent.explore_decay = safe_ckpt["explore_decay"]
                        agent.entropy_coef = float(safe_ckpt.get("entropy_coef", agent.entropy_coef))
                else:
                    # 安全阶段恢复了跟踪但没有恢复节能，此时保留安全检查点，并依赖后续最终节能保护。
                    agent.actor.load_state_dict(safe_ckpt["actor"])
                    agent.critic.load_state_dict(safe_ckpt["critic"])
                    agent.load_auxiliary_state_dict(safe_ckpt.get("agent_aux"))
                    env.set_constraint_multipliers(safe_ckpt.get("constraint_multipliers", {}))
                    env.energy_weight = safe_ckpt["energy_weight"]
                    agent.explore_decay = safe_ckpt["explore_decay"]
                    agent.entropy_coef = float(safe_ckpt.get("entropy_coef", agent.entropy_coef))
            else:
                # 最后一道保护：如果安全阶段仍然违反跟踪约束，则保留更保守的辅助策略。
                agent.actor.load_state_dict(track_stage_ckpt["actor"])
                agent.critic.load_state_dict(track_stage_ckpt["critic"])
                agent.load_auxiliary_state_dict(track_stage_ckpt.get("agent_aux"))
                env.set_constraint_multipliers(track_stage_ckpt.get("constraint_multipliers", {}))
                agent.explore_decay = track_stage_ckpt["explore_decay"]
                agent.entropy_coef = float(track_stage_ckpt.get("entropy_coef", agent.entropy_coef))
                ultra_episodes = 10 if fast_run else 20
                env.set_stage("energy", energy_weight=0.70, lambda_speed=0.0, lambda_dist=0.0)
                ret_ultra, e_ultra, lam_s_hist_ultra, lam_d_hist_ultra = run_energy_stage(
                    episodes=ultra_episodes,
                    stage_name=f"Seed{seed}-UltraSafeEnergy",
                    e_start=0.70,
                    e_end=0.95,
                    lr_s=0.85,
                    lr_d=0.18,
                    tgt_s=0.04,
                    tgt_d=0.45,
                    early_stop_patience=3,
                    eval_interval=6,
                )
                return_list += ret_ultra
                e_per_dist_list += e_ultra
                if lam_s_hist_ultra:
                    lam_s_hist = lam_s_hist_ultra
                if lam_d_hist_ultra:
                    lam_d_hist = lam_d_hist_ultra

        # 在多风格评估集上执行每个随机种子的最终跟踪保护。
        seed_track_ok, _ = seed_eval_track_save()
        if not seed_track_ok:
            agent.actor.load_state_dict(track_stage_ckpt["actor"])
            agent.critic.load_state_dict(track_stage_ckpt["critic"])
            agent.load_auxiliary_state_dict(track_stage_ckpt.get("agent_aux"))
            env.set_constraint_multipliers(track_stage_ckpt.get("constraint_multipliers", {}))
            agent.explore_decay = track_stage_ckpt["explore_decay"]
            agent.entropy_coef = float(track_stage_ckpt.get("entropy_coef", agent.entropy_coef))
            guard_episodes = 8 if fast_run else 16
            env.set_stage("energy", energy_weight=0.65, lambda_speed=0.0, lambda_dist=0.0)
            ret_guard, e_guard, lam_s_hist_guard, lam_d_hist_guard = run_energy_stage(
                episodes=guard_episodes,
                stage_name=f"Seed{seed}-FinalGuard",
                e_start=0.65,
                e_end=0.90,
                lr_s=0.90,
                lr_d=0.20,
                tgt_s=0.04,
                tgt_d=0.40,
                early_stop_patience=3,
                eval_interval=4,
            )
            return_list += ret_guard
            e_per_dist_list += e_guard
            if lam_s_hist_guard:
                lam_s_hist = lam_s_hist_guard
            if lam_d_hist_guard:
                lam_d_hist = lam_d_hist_guard
            overwrite_lagrange_snapshot()

        # 对每个随机种子执行最终的总能耗、净能耗和回收量保护。
        seed_track_ok, seed_save_ok = seed_eval_track_save()
        if seed_track_ok and (not seed_save_ok):
            used_safe_energy = True
            save_guard_base_state = snapshot_policy_state()
            save_guard_base_summary = seed_eval_summary()
            base_regen_recover_coef = float(env.regen_recover_coef)
            base_coast_torque_coef = float(env.coast_torque_coef)
            base_driver_follow_coef = float(env.driver_follow_coef)
            base_regen_target_base = float(env.regen_target_base)
            base_regen_target_gain = float(env.regen_target_gain)
            base_regen_target_tol = float(env.regen_target_tol)
            base_neg_bias_penalty_coef = float(env.neg_bias_penalty_coef)
            base_bias_neutral_torque_gain = float(env.bias_neutral_torque_gain)
            base_bias_neutral_torque_clip = float(env.bias_neutral_torque_clip)
            is_sport_style = str(getattr(env.scenario, "driver_style", "normal")) == "sport"
            is_normal_style = str(getattr(env.scenario, "driver_style", "normal")) == "normal"
            is_eco_style = str(getattr(env.scenario, "driver_style", "normal")) == "eco"
            save_guard_plan = (
                [
                    {
                        "name": "SaveGuardA",
                        "episodes": 16,
                        "stage_weight": 1.05,
                        "e_start": 1.00,
                        "e_end": 1.45,
                        "lr_s": 0.74,
                        "lr_d": 0.13,
                        "tgt_s": 0.06,
                        "tgt_d": 0.50,
                    },
                    {
                        "name": "SaveGuardB",
                        "episodes": 24,
                        "stage_weight": 1.20,
                        "e_start": 1.20,
                        "e_end": 1.85,
                        "lr_s": 0.64,
                        "lr_d": 0.10,
                        "tgt_s": 0.07,
                        "tgt_d": 0.58,
                    },
                ]
                if fast_run
                else [
                    {
                        "name": "SaveGuardA",
                        "episodes": 34,
                        "stage_weight": 1.05,
                        "e_start": 1.00,
                        "e_end": 1.45,
                        "lr_s": 0.74,
                        "lr_d": 0.13,
                        "tgt_s": 0.06,
                        "tgt_d": 0.50,
                    },
                    {
                        "name": "SaveGuardB",
                        "episodes": 52,
                        "stage_weight": 1.22,
                        "e_start": 1.18,
                        "e_end": 1.95,
                        "lr_s": 0.62,
                        "lr_d": 0.10,
                        "tgt_s": 0.07,
                        "tgt_d": 0.60,
                    },
                ]
            )
            best_save_guard = None
            for cfg in save_guard_plan:
                restore_policy_state(save_guard_base_state)
                if is_sport_style:
                    env.regen_recover_coef = max(base_regen_recover_coef, 0.10)
                    env.coast_torque_coef = min(base_coast_torque_coef, 0.082)
                    env.driver_follow_coef = max(base_driver_follow_coef, 0.24)
                    env.regen_target_base = max(base_regen_target_base, 0.94)
                    env.regen_target_gain = max(base_regen_target_gain, 0.16)
                    env.regen_target_tol = min(base_regen_target_tol, 0.38)
                elif is_normal_style:
                    env.regen_recover_coef = max(base_regen_recover_coef, 0.055)
                    env.coast_torque_coef = min(base_coast_torque_coef, 0.11)
                    env.driver_follow_coef = max(base_driver_follow_coef, 0.20)
                    env.regen_target_base = max(base_regen_target_base, 0.91)
                    env.regen_target_gain = max(base_regen_target_gain, 0.13)
                    env.regen_target_tol = min(base_regen_target_tol, 0.44)
                    env.neg_bias_penalty_coef = max(base_neg_bias_penalty_coef, 0.05)
                    env.bias_neutral_torque_gain = max(base_bias_neutral_torque_gain, 0.016)
                    env.bias_neutral_torque_clip = max(base_bias_neutral_torque_clip, 0.42)
                elif is_eco_style:
                    env.regen_recover_coef = max(base_regen_recover_coef, 0.065)
                    env.coast_torque_coef = max(base_coast_torque_coef, 0.22)
                    env.driver_follow_coef = min(base_driver_follow_coef, 0.14)
                    env.regen_target_base = max(base_regen_target_base, 0.96)
                    env.regen_target_gain = max(base_regen_target_gain, 0.12)
                    env.regen_target_tol = min(base_regen_target_tol, 0.42)
                    env.neg_bias_penalty_coef = max(base_neg_bias_penalty_coef, 0.02)
                    env.bias_neutral_torque_gain = max(base_bias_neutral_torque_gain, 0.010)
                    env.bias_neutral_torque_clip = max(base_bias_neutral_torque_clip, 0.25)
                env.set_stage("energy", energy_weight=cfg["stage_weight"], lambda_speed=0.0, lambda_dist=0.0)
                ret_save_guard, e_save_guard, lam_s_save_guard, lam_d_save_guard = run_energy_stage(
                    episodes=cfg["episodes"],
                    stage_name=f"Seed{seed}-{cfg['name']}",
                    e_start=cfg["e_start"],
                    e_end=cfg["e_end"],
                    lr_s=cfg["lr_s"],
                    lr_d=cfg["lr_d"],
                    tgt_s=cfg["tgt_s"],
                    tgt_d=cfg["tgt_d"],
                    early_stop_patience=4,
                    eval_interval=8,
                )
                post_save_guard_summary = seed_eval_summary()
                sg_track_ok, sg_save_ok = seed_eval_track_save()
                improved_total = (
                    post_save_guard_summary["mean_saving_total"]
                    > save_guard_base_summary["mean_saving_total"] + 0.10
                )
                improved_gross = (
                    post_save_guard_summary["mean_saving"]
                    > save_guard_base_summary["mean_saving"] + 0.08
                )
                track_acceptable = (
                    sg_track_ok
                    and (
                        post_save_guard_summary["mean_saving_net"]
                        >= max((net_saving_floor_pct - 0.03), (save_guard_base_summary["mean_saving_net"] - 0.04))
                    )
                    and (
                        post_save_guard_summary["worst_recover_delta"]
                        >= max((-recover_drop_tol_pct - 0.8), (save_guard_base_summary["worst_recover_delta"] - 1.0))
                    )
                )
                if track_acceptable:
                    priority = 0
                    if sg_save_ok:
                        priority = 2
                    elif improved_total and improved_gross:
                        priority = 1
                    candidate = {
                        "priority": priority,
                        "summary": post_save_guard_summary,
                        "state": snapshot_policy_state(),
                        "ret": ret_save_guard,
                        "e": e_save_guard,
                        "lam_s": lam_s_save_guard,
                        "lam_d": lam_d_save_guard,
                    }
                    if (
                        best_save_guard is None
                        or (candidate["priority"] > best_save_guard["priority"])
                        or (
                            candidate["priority"] == best_save_guard["priority"]
                            and candidate["summary"]["mean_saving_total"]
                            > best_save_guard["summary"]["mean_saving_total"] + 1e-12
                        )
                        or (
                            candidate["priority"] == best_save_guard["priority"]
                            and abs(
                                candidate["summary"]["mean_saving_total"]
                                - best_save_guard["summary"]["mean_saving_total"]
                            )
                            <= 1e-12
                            and candidate["summary"]["mean_saving"]
                            > best_save_guard["summary"]["mean_saving"]
                        )
                        or (
                            candidate["priority"] == best_save_guard["priority"]
                            and abs(
                                candidate["summary"]["mean_saving_total"]
                                - best_save_guard["summary"]["mean_saving_total"]
                            )
                            <= 1e-12
                            and abs(
                                candidate["summary"]["mean_saving"]
                                - best_save_guard["summary"]["mean_saving"]
                            )
                            <= 1e-12
                            and candidate["summary"]["robust_combined_saving"]
                            > best_save_guard["summary"]["robust_combined_saving"]
                        )
                    ):
                        best_save_guard = candidate

            if (
                best_save_guard is not None
                and (
                    (best_save_guard["priority"] >= 1)
                    or (
                        best_save_guard["summary"]["mean_saving_total"]
                        > save_guard_base_summary["mean_saving_total"] + 0.01
                    )
                )
                and (
                    best_save_guard["summary"]["mean_saving_net"]
                    >= (save_guard_base_summary["mean_saving_net"] - 0.04)
                )
                and (
                    best_save_guard["summary"]["worst_recover_delta"]
                    >= (save_guard_base_summary["worst_recover_delta"] - 1.0)
                )
            ):
                restore_policy_state(best_save_guard["state"])
                return_list += best_save_guard["ret"]
                e_per_dist_list += best_save_guard["e"]
                if best_save_guard["lam_s"]:
                    lam_s_hist = best_save_guard["lam_s"]
                if best_save_guard["lam_d"]:
                    lam_d_hist = best_save_guard["lam_d"]
                overwrite_lagrange_snapshot()
            else:
                restore_policy_state(save_guard_base_state)

            env.regen_recover_coef = base_regen_recover_coef
            env.coast_torque_coef = base_coast_torque_coef
            env.driver_follow_coef = base_driver_follow_coef
            env.regen_target_base = base_regen_target_base
            env.regen_target_gain = base_regen_target_gain
            env.regen_target_tol = base_regen_target_tol
            env.neg_bias_penalty_coef = base_neg_bias_penalty_coef
            env.bias_neutral_torque_gain = base_bias_neutral_torque_gain
            env.bias_neutral_torque_clip = base_bias_neutral_torque_clip

        # 对接近零的负节能结果做补救：在不丢失跟踪的前提下把总节能和毛节能推回正值。
        seed_track_ok, seed_save_ok = seed_eval_track_save()
        if seed_track_ok and (not seed_save_ok):
            nudge_base_summary = seed_eval_summary()
            nudge_base_state = snapshot_policy_state()
            if (
                nudge_base_summary["mean_saving_total"] > -0.20
                and nudge_base_summary["mean_saving"] > -0.20
            ):
                nudge_episodes = 10 if fast_run else 26
                env.set_stage("energy", energy_weight=1.30, lambda_speed=0.0, lambda_dist=0.0)
                ret_nudge, e_nudge, lam_s_nudge, lam_d_nudge = run_energy_stage(
                    episodes=nudge_episodes,
                    stage_name=f"Seed{seed}-SavingNudge",
                    e_start=1.25,
                    e_end=1.95,
                    lr_s=0.52,
                    lr_d=0.08,
                    tgt_s=0.07,
                    tgt_d=0.58,
                    early_stop_patience=3,
                    eval_interval=6,
                )
                nudge_track_ok, nudge_save_ok = seed_eval_track_save()
                nudge_summary = seed_eval_summary()
                nudge_improved = (
                    nudge_summary["mean_saving_total"] > nudge_base_summary["mean_saving_total"] + 0.03
                ) or (
                    nudge_summary["mean_saving"] > nudge_base_summary["mean_saving"] + 0.03
                )
                if nudge_track_ok and (nudge_save_ok or nudge_improved):
                    return_list += ret_nudge
                    e_per_dist_list += e_nudge
                    if lam_s_nudge:
                        lam_s_hist = lam_s_nudge
                    if lam_d_nudge:
                        lam_d_hist = lam_d_nudge
                    overwrite_lagrange_snapshot()
                else:
                    restore_policy_state(nudge_base_state)

        saving_epd_each = []
        saving_epd_raw_each = []
        saving_iso_epd_each = []
        saving_net_epd_each = []
        saving_net_epd_raw_each = []
        saving_net_iso_epd_each = []
        recover_delta_each = []
        saving_total_each = []
        saving_total_raw_each = []
        saving_total_iso_each = []
        low_speed_benefit_epd_each = []
        low_speed_benefit_net_epd_each = []
        low_speed_benefit_total_each = []
        distance_deficit_each = []
        kinetic_comp_energy_each = []
        distance_comp_energy_each = []
        speed_mae_each = []
        dist_mae_each = []
        speed_rel_each = []
        dist_rel_each = []
        speed_bias_each = []
        dist_bias_each = []
        speed_rel_bias_each = []
        dist_rel_bias_each = []
        launch_speed_bias_each = []
        launch_dist_bias_each = []
        launch_speed_rel_bias_each = []
        launch_dist_rel_bias_each = []
        front_half_saving_each = []
        back_half_saving_each = []
        front_back_gap_each = []
        worst_window_saving_each = []
        negative_window_ratio_each = []
        torque_delta_each = []
        torque_jerk_each = []
        smooth_ok_each = []
        smooth_delta_ratio_each = []
        smooth_jerk_ratio_each = []
        bound_violation_mean_each = []
        bound_violation_max_each = []
        tracking_ok_each = []
        style_saving_epd = {}
        style_saving_iso_epd = {}
        style_saving_net_epd = {}
        style_saving_net_iso_epd = {}
        style_recover_delta = {}
        style_saving_total = {}
        style_saving_total_iso = {}
        style_tracking_ok = {}
        style_smooth_ok = {}
        canonical_rollout = None
        canonical_ref = None

        eval_entries = evaluate_agent_on_pairs(
            env,
            agent,
            eval_pairs,
            **eval_limits_strict,
        )
        for eval_idx, entry in enumerate(eval_entries):
            rollout = entry["rollout"]
            eval_ref = entry["reference"]
            metrics = entry["metrics"]
            if eval_idx == 0:
                canonical_rollout = rollout
                canonical_ref = eval_ref
            style_name = str(getattr(entry.get("scenario"), "driver_style", "normal"))

            saving_epd_each.append(metrics["saving_epd_pct"])
            saving_epd_raw_each.append(metrics.get("saving_epd_raw_pct", metrics["saving_epd_pct"]))
            saving_iso_epd_each.append(metrics.get("saving_isochronous_pct", metrics["saving_epd_pct"]))
            saving_net_epd_each.append(metrics["saving_net_epd_pct"])
            saving_net_epd_raw_each.append(metrics.get("saving_net_epd_raw_pct", metrics["saving_net_epd_pct"]))
            saving_net_iso_epd_each.append(metrics.get("saving_net_isochronous_pct", metrics["saving_net_epd_pct"]))
            recover_delta_each.append(metrics["recover_delta_pct"])
            saving_total_each.append(metrics["saving_total_pct"])
            saving_total_raw_each.append(metrics.get("saving_total_raw_pct", metrics["saving_total_pct"]))
            saving_total_iso_each.append(metrics.get("saving_total_isochronous_pct", metrics["saving_total_pct"]))
            low_speed_benefit_epd_each.append(metrics.get("low_speed_benefit_epd_pct", 0.0))
            low_speed_benefit_net_epd_each.append(metrics.get("low_speed_benefit_net_epd_pct", 0.0))
            low_speed_benefit_total_each.append(metrics.get("low_speed_benefit_total_pct", 0.0))
            distance_deficit_each.append(metrics.get("distance_deficit_m", 0.0))
            kinetic_comp_energy_each.append(metrics.get("kinetic_comp_energy", 0.0))
            distance_comp_energy_each.append(metrics.get("distance_comp_energy", 0.0))
            speed_mae_each.append(float(rollout["speed_mae"]))
            dist_mae_each.append(float(rollout["distance_mae"]))
            speed_rel_each.append(metrics["speed_rel_diff_pct"])
            dist_rel_each.append(metrics["dist_rel_diff_pct"])
            speed_bias_each.append(metrics["speed_bias"])
            dist_bias_each.append(metrics["dist_bias"])
            speed_rel_bias_each.append(metrics["speed_rel_bias_pct"])
            dist_rel_bias_each.append(metrics["dist_rel_bias_pct"])
            launch_speed_bias_each.append(metrics.get("launch_speed_bias_mean", 0.0))
            launch_dist_bias_each.append(metrics.get("launch_dist_bias_mean", 0.0))
            launch_speed_rel_bias_each.append(metrics.get("launch_speed_rel_bias_pct", 0.0))
            launch_dist_rel_bias_each.append(metrics.get("launch_dist_rel_bias_pct", 0.0))
            front_half_saving_each.append(metrics.get("front_half_saving_pct", 0.0))
            back_half_saving_each.append(metrics.get("back_half_saving_pct", 0.0))
            front_back_gap_each.append(metrics.get("front_back_saving_gap_pct", 0.0))
            worst_window_saving_each.append(metrics.get("worst_window_saving_pct", 0.0))
            negative_window_ratio_each.append(metrics.get("negative_window_ratio", 0.0))
            torque_delta_each.append(float(rollout["torque_delta_mean"]))
            torque_jerk_each.append(float(rollout["torque_jerk_mean"]))
            smooth_ok_each.append(bool(metrics["smooth_ok"]))
            smooth_delta_ratio_each.append(float(metrics["smooth_delta_ratio"]))
            smooth_jerk_ratio_each.append(float(metrics["smooth_jerk_ratio"]))
            bound_violation_mean_each.append(float(metrics.get("proposed_action_bound_violation_linf_mean", 0.0)))
            bound_violation_max_each.append(float(metrics.get("proposed_action_bound_violation_linf_max", 0.0)))
            tracking_ok_each.append(metrics["tracking_ok"])
            style_saving_epd.setdefault(style_name, []).append(float(metrics["saving_epd_pct"]))
            style_saving_iso_epd.setdefault(style_name, []).append(float(metrics.get("saving_isochronous_pct", metrics["saving_epd_pct"])))
            style_saving_net_epd.setdefault(style_name, []).append(float(metrics["saving_net_epd_pct"]))
            style_saving_net_iso_epd.setdefault(style_name, []).append(float(metrics.get("saving_net_isochronous_pct", metrics["saving_net_epd_pct"])))
            style_recover_delta.setdefault(style_name, []).append(float(metrics["recover_delta_pct"]))
            style_saving_total.setdefault(style_name, []).append(float(metrics["saving_total_pct"]))
            style_saving_total_iso.setdefault(style_name, []).append(float(metrics.get("saving_total_isochronous_pct", metrics["saving_total_pct"])))
            style_tracking_ok.setdefault(style_name, []).append(bool(metrics["tracking_ok"]))
            style_smooth_ok.setdefault(style_name, []).append(bool(metrics["smooth_ok"]))

        seed_saving_epd = float(np.mean(saving_epd_each))
        seed_saving_epd_raw = float(np.mean(saving_epd_raw_each))
        seed_saving_iso_epd = float(np.mean(saving_iso_epd_each))
        seed_saving_net_epd = float(np.mean(saving_net_epd_each))
        seed_saving_net_epd_raw = float(np.mean(saving_net_epd_raw_each))
        seed_saving_net_iso_epd = float(np.mean(saving_net_iso_epd_each))
        seed_recover_delta = float(np.mean(recover_delta_each))
        seed_saving_total = float(np.mean(saving_total_each))
        seed_saving_total_raw = float(np.mean(saving_total_raw_each))
        seed_saving_total_iso = float(np.mean(saving_total_iso_each))
        seed_low_speed_benefit_epd = float(np.mean(low_speed_benefit_epd_each))
        seed_low_speed_benefit_net_epd = float(np.mean(low_speed_benefit_net_epd_each))
        seed_low_speed_benefit_total = float(np.mean(low_speed_benefit_total_each))
        seed_distance_deficit = float(np.mean(distance_deficit_each))
        seed_distance_comp_energy = float(np.mean(distance_comp_energy_each))
        seed_kinetic_comp_energy = float(np.mean(kinetic_comp_energy_each))
        seed_speed_mae = float(np.mean(speed_mae_each))
        seed_dist_mae = float(np.mean(dist_mae_each))
        seed_speed_rel = float(np.mean(speed_rel_each))
        seed_dist_rel = float(np.mean(dist_rel_each))
        seed_speed_bias = float(np.mean(speed_bias_each))
        seed_dist_bias = float(np.mean(dist_bias_each))
        seed_speed_rel_bias = float(np.mean(speed_rel_bias_each))
        seed_dist_rel_bias = float(np.mean(dist_rel_bias_each))
        seed_launch_speed_bias = float(np.mean(launch_speed_bias_each))
        seed_launch_dist_bias = float(np.mean(launch_dist_bias_each))
        seed_launch_speed_rel_bias = float(np.mean(launch_speed_rel_bias_each))
        seed_launch_dist_rel_bias = float(np.mean(launch_dist_rel_bias_each))
        seed_front_half_saving = float(np.mean(front_half_saving_each))
        seed_back_half_saving = float(np.mean(back_half_saving_each))
        seed_front_back_saving_gap = float(np.mean(front_back_gap_each))
        seed_worst_window_saving = float(np.min(worst_window_saving_each)) if len(worst_window_saving_each) > 0 else 0.0
        seed_negative_window_ratio = float(np.mean(negative_window_ratio_each))
        seed_torque_delta = float(np.mean(torque_delta_each))
        seed_torque_jerk = float(np.mean(torque_jerk_each))
        seed_smooth_delta_ratio = float(np.mean(smooth_delta_ratio_each))
        seed_smooth_jerk_ratio = float(np.mean(smooth_jerk_ratio_each))
        seed_smooth_delta_ratio_worst = float(np.max(smooth_delta_ratio_each))
        seed_smooth_jerk_ratio_worst = float(np.max(smooth_jerk_ratio_each))
        seed_bound_violation_mean = float(np.mean(bound_violation_mean_each))
        seed_bound_violation_max = float(np.max(bound_violation_max_each))
        seed_smooth_ok = bool(np.all(np.array(smooth_ok_each)))
        style_mean_saving_epd = {k: float(np.mean(v)) for k, v in style_saving_epd.items()}
        style_mean_saving_iso_epd = {k: float(np.mean(v)) for k, v in style_saving_iso_epd.items()}
        style_mean_saving_net_epd = {k: float(np.mean(v)) for k, v in style_saving_net_epd.items()}
        style_mean_saving_net_iso_epd = {k: float(np.mean(v)) for k, v in style_saving_net_iso_epd.items()}
        style_mean_recover_delta = {k: float(np.mean(v)) for k, v in style_recover_delta.items()}
        style_mean_saving_total = {k: float(np.mean(v)) for k, v in style_saving_total.items()}
        style_mean_saving_total_iso = {k: float(np.mean(v)) for k, v in style_saving_total_iso.items()}
        style_all_tracking_ok = {k: bool(np.all(v)) for k, v in style_tracking_ok.items()}
        style_all_smooth_ok = {k: bool(np.all(v)) for k, v in style_smooth_ok.items()}
        seed_worst_style = min(style_mean_saving_iso_epd, key=style_mean_saving_iso_epd.get) if len(style_mean_saving_iso_epd) > 0 else "unknown"
        seed_worst_style_saving_epd = float(style_mean_saving_epd[seed_worst_style]) if len(style_mean_saving_epd) > 0 else float(np.min(saving_epd_each))
        seed_worst_style_saving_iso_epd = float(style_mean_saving_iso_epd[seed_worst_style]) if len(style_mean_saving_iso_epd) > 0 else float(np.min(saving_iso_epd_each))
        seed_worst_style_saving_total = (
            float(style_mean_saving_total[seed_worst_style])
            if len(style_mean_saving_total) > 0
            else float(np.min(saving_total_each))
        )
        seed_worst_style_saving_total_iso = (
            float(style_mean_saving_total_iso[seed_worst_style])
            if len(style_mean_saving_total_iso) > 0
            else float(np.min(saving_total_iso_each))
        )
        seed_worst_style_saving_net_epd = (
            float(style_mean_saving_net_epd[seed_worst_style])
            if len(style_mean_saving_net_epd) > 0
            else float(np.min(saving_net_epd_each))
        )
        seed_worst_style_saving_net_iso_epd = (
            float(style_mean_saving_net_iso_epd[seed_worst_style])
            if len(style_mean_saving_net_iso_epd) > 0
            else float(np.min(saving_net_iso_epd_each))
        )
        seed_worst_style_recover_delta = (
            float(style_mean_recover_delta[seed_worst_style])
            if len(style_mean_recover_delta) > 0
            else float(np.min(recover_delta_each))
        )
        seed_tracking_ok = bool(np.all(tracking_ok_each)) and bool(np.all(list(style_all_tracking_ok.values())))
        seed_tracking_saving_ok = bool(
            seed_tracking_ok
            and seed_smooth_ok
            and np.all(np.array(saving_total_iso_each) > 0.0)
            and np.all(np.array(saving_iso_epd_each) > 0.0)
            and np.all(np.array(saving_net_iso_epd_each) >= net_saving_floor_pct)
            and np.all(np.array(recover_delta_each) >= -recover_drop_tol_pct)
            and np.all(np.array(smooth_ok_each))
            and all(v > 0.0 for v in style_mean_saving_total_iso.values())
            and all(v > 0.0 for v in style_mean_saving_iso_epd.values())
            and all(v >= net_saving_floor_pct for v in style_mean_saving_net_iso_epd.values())
            and all(v >= -recover_drop_tol_pct for v in style_mean_recover_delta.values())
            and all(v for v in style_all_smooth_ok.values())
        )
        style_selection_cfg = SINGLE_STYLE_CFG.get(seed_worst_style, {})
        bias_guard_speed_floor = float(style_selection_cfg.get("bias_guard_speed_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_speed_floor"]))
        bias_guard_dist_floor = float(style_selection_cfg.get("bias_guard_dist_floor", DEFAULT_SINGLE_STYLE_CFG["bias_guard_dist_floor"]))
        seed_bias_guard_ok = bool(
            (seed_speed_rel_bias >= bias_guard_speed_floor)
            and (seed_dist_rel_bias >= bias_guard_dist_floor)
        )
        seed_worst_saving_total = float(np.min(saving_total_each))
        seed_worst_saving_epd = float(np.min(saving_epd_each))
        seed_worst_saving_net_epd = float(np.min(saving_net_epd_each))
        seed_worst_saving_total_iso = float(np.min(saving_total_iso_each))
        seed_worst_saving_iso_epd = float(np.min(saving_iso_epd_each))
        seed_worst_saving_net_iso_epd = float(np.min(saving_net_iso_epd_each))
        seed_worst_recover_delta = float(np.min(recover_delta_each))
        seed_robust_saving_case_total = 0.70 * seed_saving_total_iso + 0.30 * seed_worst_saving_total_iso
        seed_robust_saving_style_total = (
            0.45 * seed_saving_total_iso + 0.35 * seed_worst_style_saving_total_iso + 0.20 * seed_worst_saving_total_iso
        )
        seed_robust_saving_case = 0.70 * seed_saving_iso_epd + 0.30 * seed_worst_saving_iso_epd
        seed_robust_saving_style = 0.45 * seed_saving_iso_epd + 0.35 * seed_worst_style_saving_iso_epd + 0.20 * seed_worst_saving_iso_epd
        seed_robust_saving_case_net = 0.70 * seed_saving_net_iso_epd + 0.30 * seed_worst_saving_net_iso_epd
        seed_robust_saving_style_net = (
            0.45 * seed_saving_net_iso_epd + 0.35 * seed_worst_style_saving_net_iso_epd + 0.20 * seed_worst_saving_net_iso_epd
        )
        seed_robust_saving_joint = (
            0.50 * seed_robust_saving_style_total
            + 0.30 * seed_robust_saving_style
            + 0.20 * seed_robust_saving_style_net
        )
        bias_excess_floor = float(style_selection_cfg.get("bias_excess_floor", DEFAULT_SINGLE_STYLE_CFG["bias_excess_floor"]))
        speed_bias_excess = max(0.0, -seed_speed_rel_bias - bias_excess_floor)
        dist_bias_excess = max(0.0, -seed_dist_rel_bias - bias_excess_floor)
        recover_shortfall = max(0.0, -seed_worst_recover_delta - recover_drop_tol_pct)
        seed_slow_bias_penalty = 0.65 * speed_bias_excess + 0.35 * dist_bias_excess
        seed_window_payback_penalty = (
            0.12 * max(0.0, -seed_worst_window_saving)
            + 0.35 * seed_negative_window_ratio
            + 0.06 * seed_front_back_saving_gap
        )
        seed_bias_adjusted_metric = (
            seed_robust_saving_joint
            - 0.45 * seed_slow_bias_penalty
            - 0.08 * recover_shortfall
            - seed_window_payback_penalty
        )

        eval_stats = {
            "seed": seed,
            "rollout": canonical_rollout,
            "driver_ref": canonical_ref,
            "saving_total_pct": seed_saving_total,
            "saving_total_raw_pct": seed_saving_total_raw,
            "saving_total_isochronous_pct": seed_saving_total_iso,
            "saving_total_worst_pct": seed_worst_saving_total,
            "saving_epd_pct": seed_saving_epd,
            "saving_epd_raw_pct": seed_saving_epd_raw,
            "saving_isochronous_pct": seed_saving_iso_epd,
            "saving_net_epd_pct": seed_saving_net_epd,
            "saving_net_epd_raw_pct": seed_saving_net_epd_raw,
            "saving_net_isochronous_pct": seed_saving_net_iso_epd,
            "low_speed_benefit_total_pct": seed_low_speed_benefit_total,
            "low_speed_benefit_epd_pct": seed_low_speed_benefit_epd,
            "low_speed_benefit_net_epd_pct": seed_low_speed_benefit_net_epd,
            "distance_deficit_m": seed_distance_deficit,
            "distance_comp_energy": seed_distance_comp_energy,
            "kinetic_comp_energy": seed_kinetic_comp_energy,
            "recover_delta_pct": seed_recover_delta,
            "recover_delta_worst_pct": seed_worst_recover_delta,
            "saving_epd_worst_pct": seed_worst_saving_epd,
            "saving_net_epd_worst_pct": seed_worst_saving_net_epd,
            "style_mean_saving_total": style_mean_saving_total,
            "style_mean_saving_epd": style_mean_saving_epd,
            "style_mean_saving_total_isochronous": style_mean_saving_total_iso,
            "style_mean_saving_isochronous": style_mean_saving_iso_epd,
            "style_mean_saving_net_epd": style_mean_saving_net_epd,
            "style_mean_saving_net_isochronous": style_mean_saving_net_iso_epd,
            "style_mean_recover_delta": style_mean_recover_delta,
            "worst_style": seed_worst_style,
            "worst_style_saving_total_pct": seed_worst_style_saving_total,
            "worst_style_saving_epd_pct": seed_worst_style_saving_epd,
            "worst_style_saving_total_isochronous_pct": seed_worst_style_saving_total_iso,
            "worst_style_saving_isochronous_pct": seed_worst_style_saving_iso_epd,
            "worst_style_saving_net_epd_pct": seed_worst_style_saving_net_epd,
            "worst_style_saving_net_isochronous_pct": seed_worst_style_saving_net_iso_epd,
            "worst_style_recover_delta_pct": seed_worst_style_recover_delta,
            "robust_saving_case_total_pct": float(seed_robust_saving_case_total),
            "robust_saving_total_pct": float(seed_robust_saving_style_total),
            "robust_saving_case_pct": float(seed_robust_saving_case),
            "robust_saving_pct": float(seed_robust_saving_style),
            "robust_saving_case_net_pct": float(seed_robust_saving_case_net),
            "robust_saving_net_pct": float(seed_robust_saving_style_net),
            "robust_saving_joint_pct": float(seed_robust_saving_joint),
            "slow_bias_penalty_pct": float(seed_slow_bias_penalty),
            "bias_adjusted_metric_pct": float(seed_bias_adjusted_metric),
            "speed_mae_mean": seed_speed_mae,
            "dist_mae_mean": seed_dist_mae,
            "speed_rel_diff_pct": seed_speed_rel,
            "dist_rel_diff_pct": seed_dist_rel,
            "speed_bias_mean": seed_speed_bias,
            "dist_bias_mean": seed_dist_bias,
            "speed_rel_bias_pct": seed_speed_rel_bias,
            "dist_rel_bias_pct": seed_dist_rel_bias,
            "launch_speed_bias_mean": seed_launch_speed_bias,
            "launch_dist_bias_mean": seed_launch_dist_bias,
            "launch_speed_rel_bias_pct": seed_launch_speed_rel_bias,
            "launch_dist_rel_bias_pct": seed_launch_dist_rel_bias,
            "front_half_saving_pct": seed_front_half_saving,
            "back_half_saving_pct": seed_back_half_saving,
            "front_back_saving_gap_pct": seed_front_back_saving_gap,
            "worst_window_saving_pct": seed_worst_window_saving,
            "negative_window_ratio": seed_negative_window_ratio,
            "window_payback_penalty_pct": float(seed_window_payback_penalty),
            "torque_delta_mean": seed_torque_delta,
            "torque_jerk_mean": seed_torque_jerk,
            "smooth_delta_ratio_mean": seed_smooth_delta_ratio,
            "smooth_jerk_ratio_mean": seed_smooth_jerk_ratio,
            "smooth_delta_ratio_worst": seed_smooth_delta_ratio_worst,
            "smooth_jerk_ratio_worst": seed_smooth_jerk_ratio_worst,
            "proposed_action_bound_violation_linf_mean": seed_bound_violation_mean,
            "proposed_action_bound_violation_linf_max": seed_bound_violation_max,
            "smooth_ok": seed_smooth_ok,
            "tracking_ok": seed_tracking_ok,
            "tracking_saving_ok": seed_tracking_saving_ok,
            "bias_guard_ok": seed_bias_guard_ok,
        }
        eval_stats_list.append(eval_stats)

        # 让训练曲线只反映标准两阶段训练主线，
        # 避免修复阶段或回退阶段污染学习趋势展示。
        all_return_curves.append(ret_track + list(ret_energy))
        all_e_per_dist_curves.append(e_track + list(e_energy))

        print(
            f"[Seed {seed}] "
            f"MeanSaving(Total)={seed_saving_total_iso:.2f}% acc {seed_saving_total:.2f}% raw {seed_saving_total_raw:.2f}% (worst iso {seed_worst_saving_total_iso:.2f}%), "
            f"MeanSaving(E/Dist)={seed_saving_iso_epd:.2f}% acc {seed_saving_epd:.2f}% raw {seed_saving_epd_raw:.2f}% (worst iso {seed_worst_saving_iso_epd:.2f}%), "
            f"MeanSaving(Net E/Dist)={seed_saving_net_iso_epd:.2f}% acc {seed_saving_net_epd:.2f}% raw {seed_saving_net_epd_raw:.2f}% (worst iso {seed_worst_saving_net_iso_epd:.2f}%), "
            f"LowSpeedBenefit(total/gross/net)={seed_low_speed_benefit_total:.2f}%/{seed_low_speed_benefit_epd:.2f}%/{seed_low_speed_benefit_net_epd:.2f}%, "
            f"IsoDebt(dist/Edist/Ekin)={seed_distance_deficit:.3f}m/{seed_distance_comp_energy:.6f}/{seed_kinetic_comp_energy:.6f}, "
            f"RecoverΔ={seed_recover_delta:.2f}% (worst {seed_worst_recover_delta:.2f}%), "
            f"StyleWorst={seed_worst_style}:total iso {seed_worst_style_saving_total_iso:.2f}% / gross iso {seed_worst_style_saving_iso_epd:.2f}% / net iso {seed_worst_style_saving_net_iso_epd:.2f}% / rec {seed_worst_style_recover_delta:.2f}%, "
            f"RobustJointSaving={seed_robust_saving_joint:.2f}%, "
            f"BiasAdjMetric={seed_bias_adjusted_metric:.2f}%, "
            f"SpeedMAE={seed_speed_mae:.3f}, "
            f"DistMAE={seed_dist_mae:.3f}, "
            f"TorqueΔmean={seed_torque_delta:.3f}, "
            f"TorqueJerkMean={seed_torque_jerk:.3f}, "
            f"SmoothRatioΔ(mean/worst)={seed_smooth_delta_ratio:.3f}/{seed_smooth_delta_ratio_worst:.3f}, "
            f"SmoothRatioJ(mean/worst)={seed_smooth_jerk_ratio:.3f}/{seed_smooth_jerk_ratio_worst:.3f}, "
            f"SpeedRel={seed_speed_rel:.2f}%, DistRel={seed_dist_rel:.2f}%, "
            f"SpeedBias={seed_speed_bias:.3f}m/s({seed_speed_rel_bias:.2f}%), "
            f"LaunchBias={seed_launch_speed_bias:.3f}m/s({seed_launch_speed_rel_bias:.2f}%), "
            f"DistBias={seed_dist_bias:.3f}m({seed_dist_rel_bias:.2f}%), "
            f"Save(front/back/gap)={seed_front_half_saving:.2f}%/{seed_back_half_saving:.2f}%/{seed_front_back_saving_gap:.2f}%, "
            f"WorstWin={seed_worst_window_saving:.2f}%, NegWin={seed_negative_window_ratio * 100.0:.1f}%, "
            f"SmoothAll={seed_smooth_ok}, "
            f"TrackingAll={seed_tracking_ok}, "
            f"Track+SaveAll={seed_tracking_saving_ok}, "
            f"BiasGuard={seed_bias_guard_ok}, "
            f"BiasRepair={used_bias_repair}, TrackRepair={used_track_repair}, SafeEnergy={used_safe_energy}, "
            f"LamS={env.lambda_speed:.2f}, LamD={env.lambda_dist:.2f}"
        )

        score = (
            2.0 * seed_saving_total_iso
            + 1.2 * seed_saving_iso_epd
            + 0.8 * seed_saving_net_iso_epd
            - 1.2 * seed_speed_mae
            - 0.15 * seed_dist_mae
            - 0.08 * seed_speed_rel
        )
        if seed_saving_total_iso < 0:
            score -= 7.0
        if seed_saving_iso_epd < 0:
            score -= 5.0
        if seed_saving_net_iso_epd < net_saving_floor_pct:
            score -= 7.0
        if seed_worst_recover_delta < -recover_drop_tol_pct:
            score -= 6.0
        if not seed_tracking_ok:
            score -= 220.0
        fallback_score = score - 0.50 * seed_slow_bias_penalty - 0.08 * recover_shortfall

        if seed_tracking_saving_ok and seed_bias_guard_ok:
            # 硬约束选模：先优先保证低慢开偏差下的鲁棒节能，再比较平滑性。
            robust_gap = seed_bias_adjusted_metric - best_feasible_metric
            choose = False
            if (not best_has_feasible) or (best_rollout is None):
                choose = True
            elif robust_gap > 0.06:
                choose = True
            elif abs(robust_gap) <= 0.06:
                raw_gap = seed_robust_saving_joint - best_feasible_raw_robust
                if raw_gap > 0.05:
                    choose = True
                elif abs(raw_gap) <= 0.05:
                    mean_gap = seed_saving_iso_epd - best_feasible_mean_saving
                    if mean_gap > 0.05:
                        choose = True
                    elif abs(mean_gap) <= 0.05:
                        smoother = (
                            (seed_torque_delta < best_torque_delta - 1e-4)
                            or (abs(seed_torque_delta - best_torque_delta) <= 1e-4 and seed_torque_jerk < best_torque_jerk)
                        )
                        choose = smoother
            if choose:
                best_has_feasible = True
                best_feasible_metric = seed_bias_adjusted_metric
                best_feasible_raw_robust = seed_robust_saving_joint
                best_feasible_mean_saving = seed_saving_iso_epd
                best_score = seed_bias_adjusted_metric
                best_torque_delta = seed_torque_delta
                best_torque_jerk = seed_torque_jerk
                best_seed = seed
                best_rollout = canonical_rollout
                best_rollout_ref = canonical_ref
                best_rollout_metrics = eval_stats
                torch.save(
                    {
                        "seed": seed,
                        "score": best_score,
                        "actor_state_dict": agent.actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict(),
                        "agent_aux_state_dict": agent.get_auxiliary_state_dict(),
                        "constraint_multipliers": env.get_constraint_multipliers(),
                    },
                    model_output_path,
                )
        elif (not best_has_feasible) and seed_tracking_ok and fallback_score > best_fallback_score:
            # 回退策略：如果不存在硬约束可行策略，则保留跟踪成功且偏差惩罚后评分最高的模型。
            best_fallback_score = fallback_score
            best_score = fallback_score
            best_torque_delta = seed_torque_delta
            best_torque_jerk = seed_torque_jerk
            best_seed = seed
            best_rollout = canonical_rollout
            best_rollout_ref = canonical_ref
            best_rollout_metrics = eval_stats
            torch.save(
                {
                    "seed": seed,
                    "score": fallback_score,
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic_state_dict": agent.critic.state_dict(),
                    "agent_aux_state_dict": agent.get_auxiliary_state_dict(),
                    "constraint_multipliers": env.get_constraint_multipliers(),
                },
                model_output_path,
            )

    min_curve_len = min(len(x) for x in all_return_curves)
    return_arr = np.array([x[:min_curve_len] for x in all_return_curves if isinstance(x, list)], dtype=np.float32)
    e_per_dist_arr = np.array([x[:min_curve_len] for x in all_e_per_dist_curves if isinstance(x, list)], dtype=np.float32)
    mean_return_curve = return_arr.mean(axis=0)
    mean_e_per_dist_curve = e_per_dist_arr.mean(axis=0)

    plot_training_curves(
        mean_return_curve,
        mean_e_per_dist_curve,
        stage_split=track_episodes,
        output_path=output_dir / "motor_ppo_result.png",
    )

    if best_rollout is not None and best_rollout_ref is not None:
        plot_tracking(
            best_rollout,
            best_rollout_ref,
            metrics=best_rollout_metrics,
            output_path=output_dir / "motor_ppo_trajectory.png",
        )
        plot_energy_saving(
            best_rollout_ref,
            best_rollout,
            best_seed,
            metrics=best_rollout_metrics,
            output_path=output_dir / "energy_saving_effect.png",
        )
        plot_saving_trace(
            best_rollout_ref,
            best_rollout,
            metrics=best_rollout_metrics,
            output_path=output_dir / "saving_trace.png",
        )
        plot_window_saving_trace(
            best_rollout_ref,
            best_rollout,
            metrics=best_rollout_metrics,
            output_path=output_dir / "saving_window_trace.png",
        )
        export_saving_trace_csv(
            best_rollout_ref,
            best_rollout,
            metrics=best_rollout_metrics,
            output_path=output_dir / "saving_trace.csv",
        )

    saving_epd_list = np.array([x["saving_epd_pct"] for x in eval_stats_list], dtype=np.float32)
    saving_epd_raw_list = np.array([x.get("saving_epd_raw_pct", x["saving_epd_pct"]) for x in eval_stats_list], dtype=np.float32)
    saving_iso_epd_list = np.array([x.get("saving_isochronous_pct", x["saving_epd_pct"]) for x in eval_stats_list], dtype=np.float32)
    saving_net_epd_list = np.array([x["saving_net_epd_pct"] for x in eval_stats_list], dtype=np.float32)
    saving_net_epd_raw_list = np.array([x.get("saving_net_epd_raw_pct", x["saving_net_epd_pct"]) for x in eval_stats_list], dtype=np.float32)
    saving_net_iso_epd_list = np.array([x.get("saving_net_isochronous_pct", x["saving_net_epd_pct"]) for x in eval_stats_list], dtype=np.float32)
    recover_delta_list = np.array([x["recover_delta_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_saving_list = np.array([x["robust_saving_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_saving_net_list = np.array([x["robust_saving_net_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_saving_total_list = np.array([x["robust_saving_total_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_saving_joint_list = np.array([x["robust_saving_joint_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_case_saving_list = np.array([x["robust_saving_case_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_case_saving_net_list = np.array([x["robust_saving_case_net_pct"] for x in eval_stats_list], dtype=np.float32)
    robust_case_saving_total_list = np.array([x["robust_saving_case_total_pct"] for x in eval_stats_list], dtype=np.float32)
    worst_style_saving_list = np.array([x["worst_style_saving_epd_pct"] for x in eval_stats_list], dtype=np.float32)
    saving_total_list = np.array([x["saving_total_pct"] for x in eval_stats_list], dtype=np.float32)
    saving_total_raw_list = np.array([x.get("saving_total_raw_pct", x["saving_total_pct"]) for x in eval_stats_list], dtype=np.float32)
    saving_total_iso_list = np.array([x.get("saving_total_isochronous_pct", x["saving_total_pct"]) for x in eval_stats_list], dtype=np.float32)
    low_speed_benefit_epd_list = np.array([x.get("low_speed_benefit_epd_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    low_speed_benefit_net_epd_list = np.array([x.get("low_speed_benefit_net_epd_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    low_speed_benefit_total_list = np.array([x.get("low_speed_benefit_total_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    distance_deficit_list = np.array([x.get("distance_deficit_m", 0.0) for x in eval_stats_list], dtype=np.float32)
    distance_comp_energy_list = np.array([x.get("distance_comp_energy", 0.0) for x in eval_stats_list], dtype=np.float32)
    kinetic_comp_energy_list = np.array([x.get("kinetic_comp_energy", 0.0) for x in eval_stats_list], dtype=np.float32)
    speed_mae_list = np.array([x["speed_mae_mean"] for x in eval_stats_list], dtype=np.float32)
    dist_mae_list = np.array([x["dist_mae_mean"] for x in eval_stats_list], dtype=np.float32)
    bound_violation_mean_list = np.array(
        [x.get("proposed_action_bound_violation_linf_mean", 0.0) for x in eval_stats_list],
        dtype=np.float32,
    )
    bound_violation_max_list = np.array(
        [x.get("proposed_action_bound_violation_linf_max", 0.0) for x in eval_stats_list],
        dtype=np.float32,
    )
    speed_rel_list = np.array([x["speed_rel_diff_pct"] for x in eval_stats_list], dtype=np.float32)
    dist_rel_list = np.array([x["dist_rel_diff_pct"] for x in eval_stats_list], dtype=np.float32)
    speed_bias_list = np.array([x["speed_bias_mean"] for x in eval_stats_list], dtype=np.float32)
    dist_bias_list = np.array([x["dist_bias_mean"] for x in eval_stats_list], dtype=np.float32)
    speed_rel_bias_list = np.array([x["speed_rel_bias_pct"] for x in eval_stats_list], dtype=np.float32)
    dist_rel_bias_list = np.array([x["dist_rel_bias_pct"] for x in eval_stats_list], dtype=np.float32)
    launch_speed_bias_list = np.array([x.get("launch_speed_bias_mean", 0.0) for x in eval_stats_list], dtype=np.float32)
    launch_dist_bias_list = np.array([x.get("launch_dist_bias_mean", 0.0) for x in eval_stats_list], dtype=np.float32)
    launch_speed_rel_bias_list = np.array([x.get("launch_speed_rel_bias_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    launch_dist_rel_bias_list = np.array([x.get("launch_dist_rel_bias_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    front_half_saving_list = np.array([x.get("front_half_saving_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    back_half_saving_list = np.array([x.get("back_half_saving_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    front_back_gap_list = np.array([x.get("front_back_saving_gap_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    worst_window_saving_list = np.array([x.get("worst_window_saving_pct", 0.0) for x in eval_stats_list], dtype=np.float32)
    negative_window_ratio_list = np.array([x.get("negative_window_ratio", 0.0) for x in eval_stats_list], dtype=np.float32)
    torque_delta_list = np.array([x["torque_delta_mean"] for x in eval_stats_list], dtype=np.float32)
    torque_jerk_list = np.array([x["torque_jerk_mean"] for x in eval_stats_list], dtype=np.float32)
    smooth_delta_ratio_mean_list = np.array([x["smooth_delta_ratio_mean"] for x in eval_stats_list], dtype=np.float32)
    smooth_jerk_ratio_mean_list = np.array([x["smooth_jerk_ratio_mean"] for x in eval_stats_list], dtype=np.float32)
    smooth_delta_ratio_worst_list = np.array([x["smooth_delta_ratio_worst"] for x in eval_stats_list], dtype=np.float32)
    smooth_jerk_ratio_worst_list = np.array([x["smooth_jerk_ratio_worst"] for x in eval_stats_list], dtype=np.float32)
    track_ok_list = np.array([1.0 if x["tracking_ok"] else 0.0 for x in eval_stats_list], dtype=np.float32)
    smooth_ok_list = np.array([1.0 if x["smooth_ok"] else 0.0 for x in eval_stats_list], dtype=np.float32)
    feasible_saving_list = np.array([1.0 if x["tracking_saving_ok"] else 0.0 for x in eval_stats_list], dtype=np.float32)
    bias_guard_list = np.array([1.0 if x["bias_guard_ok"] else 0.0 for x in eval_stats_list], dtype=np.float32)

    print("\n===== Multi-seed Summary =====")
    print(f"Seeds: {seeds}")
    print(
        f"Train roads x styles: {num_train_roads} x {len(driver_styles)} = {len(train_pairs)}, "
        f"Unseen eval roads x styles: {num_eval_roads} x {len(driver_styles)} = {len(eval_pairs)}"
    )
    print(f"Two-stage episodes: track={track_episodes}, energy={energy_episodes}")
    print(f"Driver E/Dist (unseen mean): {baseline_eval_epd:.8f}")
    print(f"Agent saving(E/Dist, isochronous) mean (unseen): {np.mean([x.get('saving_isochronous_pct', x['saving_epd_pct']) for x in eval_stats_list]):.2f}%")
    print(f"Agent saving(Net E/Dist, isochronous) mean (unseen): {np.mean([x.get('saving_net_isochronous_pct', x['saving_net_epd_pct']) for x in eval_stats_list]):.2f}%")
    print(f"Saving(E/Dist) mean +- std: {saving_epd_list.mean():.2f}% +- {saving_epd_list.std():.2f}%")
    print(f"Saving(E/Dist raw) mean +- std: {saving_epd_raw_list.mean():.2f}% +- {saving_epd_raw_list.std():.2f}%")
    print(f"Saving(E/Dist isochronous) mean +- std: {saving_iso_epd_list.mean():.2f}% +- {saving_iso_epd_list.std():.2f}%")
    print(f"Saving(Net E/Dist) mean +- std: {saving_net_epd_list.mean():.2f}% +- {saving_net_epd_list.std():.2f}%")
    print(f"Saving(Net E/Dist raw) mean +- std: {saving_net_epd_raw_list.mean():.2f}% +- {saving_net_epd_raw_list.std():.2f}%")
    print(f"Saving(Net E/Dist isochronous) mean +- std: {saving_net_iso_epd_list.mean():.2f}% +- {saving_net_iso_epd_list.std():.2f}%")
    print(f"Recover delta mean +- std: {recover_delta_list.mean():.2f}% +- {recover_delta_list.std():.2f}%")
    print(f"Worst-style saving mean +- std: {worst_style_saving_list.mean():.2f}% +- {worst_style_saving_list.std():.2f}%")
    print(f"RobustStyleSaving(Total) mean +- std: {robust_saving_total_list.mean():.2f}% +- {robust_saving_total_list.std():.2f}%")
    print(f"RobustStyleSaving mean +- std: {robust_saving_list.mean():.2f}% +- {robust_saving_list.std():.2f}%")
    print(f"RobustStyleSaving(Net) mean +- std: {robust_saving_net_list.mean():.2f}% +- {robust_saving_net_list.std():.2f}%")
    print(f"RobustJointSaving mean +- std: {robust_saving_joint_list.mean():.2f}% +- {robust_saving_joint_list.std():.2f}%")
    print(f"RobustCaseSaving(Total) mean +- std: {robust_case_saving_total_list.mean():.2f}% +- {robust_case_saving_total_list.std():.2f}%")
    print(f"RobustCaseSaving mean +- std: {robust_case_saving_list.mean():.2f}% +- {robust_case_saving_list.std():.2f}%")
    print(f"RobustCaseSaving(Net) mean +- std: {robust_case_saving_net_list.mean():.2f}% +- {robust_case_saving_net_list.std():.2f}%")
    print(f"Saving(Total) mean +- std: {saving_total_list.mean():.2f}% +- {saving_total_list.std():.2f}%")
    print(f"Saving(Total raw) mean +- std: {saving_total_raw_list.mean():.2f}% +- {saving_total_raw_list.std():.2f}%")
    print(f"Saving(Total isochronous) mean +- std: {saving_total_iso_list.mean():.2f}% +- {saving_total_iso_list.std():.2f}%")
    print(
        f"Low-speed benefit(total/gross/net) mean +- std: "
        f"{low_speed_benefit_total_list.mean():.2f}% +- {low_speed_benefit_total_list.std():.2f}% / "
        f"{low_speed_benefit_epd_list.mean():.2f}% +- {low_speed_benefit_epd_list.std():.2f}% / "
        f"{low_speed_benefit_net_epd_list.mean():.2f}% +- {low_speed_benefit_net_epd_list.std():.2f}%"
    )
    print(
        f"Isochronous debt(dist/Edist/Ekin) mean +- std: "
        f"{distance_deficit_list.mean():.3f}m +- {distance_deficit_list.std():.3f} / "
        f"{distance_comp_energy_list.mean():.6f} +- {distance_comp_energy_list.std():.6f} / "
        f"{kinetic_comp_energy_list.mean():.6f} +- {kinetic_comp_energy_list.std():.6f}"
    )
    print(f"Speed MAE mean +- std: {speed_mae_list.mean():.3f} +- {speed_mae_list.std():.3f}")
    print(f"Distance MAE mean +- std: {dist_mae_list.mean():.3f} +- {dist_mae_list.std():.3f}")
    print(f"Speed rel diff mean +- std: {speed_rel_list.mean():.2f}% +- {speed_rel_list.std():.2f}%")
    print(f"Distance rel diff mean +- std: {dist_rel_list.mean():.2f}% +- {dist_rel_list.std():.2f}%")
    print(f"Speed bias mean +- std: {speed_bias_list.mean():.3f}m/s +- {speed_bias_list.std():.3f}m/s ({speed_rel_bias_list.mean():.2f}% +- {speed_rel_bias_list.std():.2f}%)")
    print(f"Launch speed bias mean +- std: {launch_speed_bias_list.mean():.3f}m/s +- {launch_speed_bias_list.std():.3f}m/s ({launch_speed_rel_bias_list.mean():.2f}% +- {launch_speed_rel_bias_list.std():.2f}%)")
    print(f"Distance bias mean +- std: {dist_bias_list.mean():.3f}m +- {dist_bias_list.std():.3f}m ({dist_rel_bias_list.mean():.2f}% +- {dist_rel_bias_list.std():.2f}%)")
    print(f"Launch distance bias mean +- std: {launch_dist_bias_list.mean():.3f}m +- {launch_dist_bias_list.std():.3f}m ({launch_dist_rel_bias_list.mean():.2f}% +- {launch_dist_rel_bias_list.std():.2f}%)")
    print(
        f"Saving front/back/gap mean +- std: "
        f"{front_half_saving_list.mean():.2f}% / {back_half_saving_list.mean():.2f}% / "
        f"{front_back_gap_list.mean():.2f}% +- {front_back_gap_list.std():.2f}%"
    )
    print(
        f"Worst window saving mean +- std: {worst_window_saving_list.mean():.2f}% +- {worst_window_saving_list.std():.2f}%, "
        f"Negative window ratio mean +- std: {negative_window_ratio_list.mean() * 100.0:.1f}% +- {negative_window_ratio_list.std() * 100.0:.1f}%"
    )
    print(f"TorqueΔmean mean +- std: {torque_delta_list.mean():.3f} +- {torque_delta_list.std():.3f}")
    print(f"TorqueJerkMean mean +- std: {torque_jerk_list.mean():.3f} +- {torque_jerk_list.std():.3f}")
    print(
        f"Smooth ratio Δ mean +- std: {smooth_delta_ratio_mean_list.mean():.3f} +- {smooth_delta_ratio_mean_list.std():.3f} "
        f"(worst mean {smooth_delta_ratio_worst_list.mean():.3f})"
    )
    print(
        f"Smooth ratio jerk mean +- std: {smooth_jerk_ratio_mean_list.mean():.3f} +- {smooth_jerk_ratio_mean_list.std():.3f} "
        f"(worst mean {smooth_jerk_ratio_worst_list.mean():.3f})"
    )
    print(f"Tracking success ratio: {track_ok_list.mean() * 100.0:.1f}%")
    print(f"Smoothness success ratio: {smooth_ok_list.mean() * 100.0:.1f}%")
    print(f"Tracking+Saving success ratio: {feasible_saving_list.mean() * 100.0:.1f}%")
    print(f"Bias guard success ratio: {bias_guard_list.mean() * 100.0:.1f}%")
    print(f"Final lambda speed mean: {np.mean(lagrange_s_last):.2f}")
    print(f"Final lambda dist mean: {np.mean(lagrange_d_last):.2f}")
    print(f"Final lambda smooth mean: {np.mean(lagrange_sm_last):.2f}")
    print(f"Final lambda net-energy mean: {np.mean(lagrange_n_last):.2f}")
    print(f"Final lambda projection mean: {np.mean(lagrange_p_last):.2f}")
    print(f"Final lambda underspeed mean: {np.mean(lagrange_u_last):.2f}")
    print(f"Final lambda window-energy mean: {np.mean(lagrange_w_last):.2f}")
    best_metric_type = "bias_adjusted_robust_joint_saving_metric" if best_has_feasible else "tracking_constrained_fallback_score"
    print(f"Best seed by hard-constraint objective: {best_seed}, metric={best_score:.3f}, type={best_metric_type}")

    meta = {
        "seeds": seeds,
        "track_episodes": int(track_episodes),
        "energy_episodes": int(energy_episodes),
        "num_train_roads": int(num_train_roads),
        "num_eval_roads": int(num_eval_roads),
        "driver_styles": driver_styles,
        "num_train_scenarios": int(len(train_pairs)),
        "num_eval_scenarios": int(len(eval_pairs)),
        "driver_total_energy_eval_mean": float(baseline_eval_total_energy),
        "driver_energy_per_dist_eval_mean": float(baseline_eval_epd),
        "motor_map_csv": motor_map_csv if motor_map_csv else None,
        "motor_map_source": train_pairs[0][0].motor_map_source if len(train_pairs) > 0 else "unknown",
        "policy_distribution": "tanh_squashed_gaussian",
        "ppo_cfg": {k: float(v) for k, v in ppo_cfg.items()},
        "constraint_names": list(experiment_constraint_names),
        "lagrange_update_controller": lagrange_update_controller,
        "lagrange_pid_cfg": {k: float(v) for k, v in lagrange_pid_cfg.items()},
        "memory_cfg": {"obs_stack": int(memory_cfg["obs_stack"])},
        "best_seed": int(best_seed) if best_seed is not None else None,
        "best_score": float(best_score),
        "best_selection_metric": "style_aware_bias_adjusted_robust_joint_saving_metric_if_track_and_save_feasible_else_tracking_constrained_score",
        "best_has_feasible_tracking_saving_policy": bool(best_has_feasible),
        "saving_epd_mean_pct": float(saving_epd_list.mean()),
        "saving_epd_std_pct": float(saving_epd_list.std()),
        "saving_epd_raw_mean_pct": float(saving_epd_raw_list.mean()),
        "saving_epd_raw_std_pct": float(saving_epd_raw_list.std()),
        "saving_isochronous_mean_pct": float(saving_iso_epd_list.mean()),
        "saving_isochronous_std_pct": float(saving_iso_epd_list.std()),
        "saving_net_epd_mean_pct": float(saving_net_epd_list.mean()),
        "saving_net_epd_std_pct": float(saving_net_epd_list.std()),
        "saving_net_epd_raw_mean_pct": float(saving_net_epd_raw_list.mean()),
        "saving_net_epd_raw_std_pct": float(saving_net_epd_raw_list.std()),
        "saving_net_isochronous_mean_pct": float(saving_net_iso_epd_list.mean()),
        "saving_net_isochronous_std_pct": float(saving_net_iso_epd_list.std()),
        "recover_delta_mean_pct": float(recover_delta_list.mean()),
        "recover_delta_std_pct": float(recover_delta_list.std()),
        "robust_saving_mean_pct": float(robust_saving_list.mean()),
        "robust_saving_std_pct": float(robust_saving_list.std()),
        "robust_saving_total_mean_pct": float(robust_saving_total_list.mean()),
        "robust_saving_total_std_pct": float(robust_saving_total_list.std()),
        "robust_saving_net_mean_pct": float(robust_saving_net_list.mean()),
        "robust_saving_net_std_pct": float(robust_saving_net_list.std()),
        "robust_saving_joint_mean_pct": float(robust_saving_joint_list.mean()),
        "robust_saving_joint_std_pct": float(robust_saving_joint_list.std()),
        "robust_case_saving_mean_pct": float(robust_case_saving_list.mean()),
        "robust_case_saving_std_pct": float(robust_case_saving_list.std()),
        "robust_case_saving_total_mean_pct": float(robust_case_saving_total_list.mean()),
        "robust_case_saving_total_std_pct": float(robust_case_saving_total_list.std()),
        "robust_case_saving_net_mean_pct": float(robust_case_saving_net_list.mean()),
        "robust_case_saving_net_std_pct": float(robust_case_saving_net_list.std()),
        "worst_style_saving_mean_pct": float(worst_style_saving_list.mean()),
        "worst_style_saving_std_pct": float(worst_style_saving_list.std()),
        "saving_total_mean_pct": float(saving_total_list.mean()),
        "saving_total_std_pct": float(saving_total_list.std()),
        "saving_total_raw_mean_pct": float(saving_total_raw_list.mean()),
        "saving_total_raw_std_pct": float(saving_total_raw_list.std()),
        "saving_total_isochronous_mean_pct": float(saving_total_iso_list.mean()),
        "saving_total_isochronous_std_pct": float(saving_total_iso_list.std()),
        "low_speed_benefit_epd_mean_pct": float(low_speed_benefit_epd_list.mean()),
        "low_speed_benefit_epd_std_pct": float(low_speed_benefit_epd_list.std()),
        "low_speed_benefit_net_epd_mean_pct": float(low_speed_benefit_net_epd_list.mean()),
        "low_speed_benefit_net_epd_std_pct": float(low_speed_benefit_net_epd_list.std()),
        "low_speed_benefit_total_mean_pct": float(low_speed_benefit_total_list.mean()),
        "low_speed_benefit_total_std_pct": float(low_speed_benefit_total_list.std()),
        "distance_deficit_mean_m": float(distance_deficit_list.mean()),
        "distance_deficit_std_m": float(distance_deficit_list.std()),
        "distance_comp_energy_mean": float(distance_comp_energy_list.mean()),
        "distance_comp_energy_std": float(distance_comp_energy_list.std()),
        "kinetic_comp_energy_mean": float(kinetic_comp_energy_list.mean()),
        "kinetic_comp_energy_std": float(kinetic_comp_energy_list.std()),
        "speed_mae_mean": float(speed_mae_list.mean()),
        "dist_mae_mean": float(dist_mae_list.mean()),
        "proposed_action_bound_violation_linf_mean": float(bound_violation_mean_list.mean()),
        "proposed_action_bound_violation_linf_max": float(bound_violation_max_list.max()),
        "speed_rel_diff_mean_pct": float(speed_rel_list.mean()),
        "dist_rel_diff_mean_pct": float(dist_rel_list.mean()),
        "speed_bias_mean": float(speed_bias_list.mean()),
        "speed_bias_std": float(speed_bias_list.std()),
        "dist_bias_mean": float(dist_bias_list.mean()),
        "dist_bias_std": float(dist_bias_list.std()),
        "speed_rel_bias_mean_pct": float(speed_rel_bias_list.mean()),
        "speed_rel_bias_std_pct": float(speed_rel_bias_list.std()),
        "dist_rel_bias_mean_pct": float(dist_rel_bias_list.mean()),
        "dist_rel_bias_std_pct": float(dist_rel_bias_list.std()),
        "launch_speed_bias_mean": float(launch_speed_bias_list.mean()),
        "launch_speed_bias_std": float(launch_speed_bias_list.std()),
        "launch_dist_bias_mean": float(launch_dist_bias_list.mean()),
        "launch_dist_bias_std": float(launch_dist_bias_list.std()),
        "launch_speed_rel_bias_mean_pct": float(launch_speed_rel_bias_list.mean()),
        "launch_speed_rel_bias_std_pct": float(launch_speed_rel_bias_list.std()),
        "launch_dist_rel_bias_mean_pct": float(launch_dist_rel_bias_list.mean()),
        "launch_dist_rel_bias_std_pct": float(launch_dist_rel_bias_list.std()),
        "front_half_saving_mean_pct": float(front_half_saving_list.mean()),
        "front_half_saving_std_pct": float(front_half_saving_list.std()),
        "back_half_saving_mean_pct": float(back_half_saving_list.mean()),
        "back_half_saving_std_pct": float(back_half_saving_list.std()),
        "front_back_saving_gap_mean_pct": float(front_back_gap_list.mean()),
        "front_back_saving_gap_std_pct": float(front_back_gap_list.std()),
        "worst_window_saving_mean_pct": float(worst_window_saving_list.mean()),
        "worst_window_saving_std_pct": float(worst_window_saving_list.std()),
        "negative_window_ratio_mean": float(negative_window_ratio_list.mean()),
        "negative_window_ratio_std": float(negative_window_ratio_list.std()),
        "torque_delta_mean": float(torque_delta_list.mean()),
        "torque_delta_std": float(torque_delta_list.std()),
        "torque_jerk_mean": float(torque_jerk_list.mean()),
        "torque_jerk_std": float(torque_jerk_list.std()),
        "smooth_delta_ratio_mean": float(smooth_delta_ratio_mean_list.mean()),
        "smooth_delta_ratio_std": float(smooth_delta_ratio_mean_list.std()),
        "smooth_delta_ratio_worst_mean": float(smooth_delta_ratio_worst_list.mean()),
        "smooth_delta_ratio_worst_std": float(smooth_delta_ratio_worst_list.std()),
        "smooth_jerk_ratio_mean": float(smooth_jerk_ratio_mean_list.mean()),
        "smooth_jerk_ratio_std": float(smooth_jerk_ratio_mean_list.std()),
        "smooth_jerk_ratio_worst_mean": float(smooth_jerk_ratio_worst_list.mean()),
        "smooth_jerk_ratio_worst_std": float(smooth_jerk_ratio_worst_list.std()),
        "tracking_success_ratio": float(track_ok_list.mean()),
        "smooth_success_ratio": float(smooth_ok_list.mean()),
        "tracking_saving_success_ratio": float(feasible_saving_list.mean()),
        "bias_guard_success_ratio": float(bias_guard_list.mean()),
        "lagrange_speed_mean": float(np.mean(lagrange_s_last)),
        "lagrange_dist_mean": float(np.mean(lagrange_d_last)),
        "lagrange_smooth_mean": float(np.mean(lagrange_sm_last)),
        "lagrange_net_energy_mean": float(np.mean(lagrange_n_last)),
        "lagrange_projection_mean": float(np.mean(lagrange_p_last)),
        "lagrange_underspeed_mean": float(np.mean(lagrange_u_last)),
        "lagrange_window_energy_mean": float(np.mean(lagrange_w_last)),
    }
    meta["target_driver_style"] = target_driver_style
    meta["output_dir"] = str(output_dir)
    with open(output_dir / "best_motor_ppo_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_experiment()
