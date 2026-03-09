from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from motor_env import MPS_TO_KMH


def _apply_plot_theme():
    plt.rcParams.update(
        {
            "axes.facecolor": "#f7f8fb",
            "figure.facecolor": "white",
            "axes.edgecolor": "#d6d9e0",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.28,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d6d9e0",
            "legend.framealpha": 0.92,
        }
    )


def _kpi_box(ax, text, loc=(0.02, 0.98), color="#111827"):
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=color,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d6d9e0", "alpha": 0.95},
    )


def build_saving_trace(driver_ref, best_rollout, metrics=None):
    """
    作用：构建按 step 变化的节能轨迹，便于导出 CSV 或绘图。
    输入：driver_ref: 驾驶员参考；best_rollout: 智能体 rollout；metrics: 可选评估指标。
    输出：包含每步与累计节能数组的字典。
    """
    step_distance = np.asarray(best_rollout["step_distance"], dtype=np.float32)
    ref_step_distance = np.asarray(driver_ref["step_distance"], dtype=np.float32)
    ref_step_energy = np.asarray(driver_ref["step_energy"], dtype=np.float32)
    agent_step_energy_accounted = np.asarray(best_rollout["step_energy_accounted"], dtype=np.float32)
    agent_step_energy_raw = np.asarray(best_rollout.get("step_energy_raw", agent_step_energy_accounted), dtype=np.float32)
    ref_step_recover = np.asarray(driver_ref["step_recover"], dtype=np.float32)
    agent_step_recover = np.asarray(best_rollout["step_recover"], dtype=np.float32)

    n = int(min(len(step_distance), len(ref_step_distance), len(ref_step_energy), len(agent_step_energy_accounted)))
    step_distance = step_distance[:n]
    ref_step_distance = ref_step_distance[:n]
    ref_step_energy = ref_step_energy[:n]
    agent_step_energy_accounted = agent_step_energy_accounted[:n]
    agent_step_energy_raw = agent_step_energy_raw[:n]
    ref_step_recover = ref_step_recover[:n]
    agent_step_recover = agent_step_recover[:n]

    ref_step_epd = ref_step_energy / np.maximum(ref_step_distance, 1e-8)
    agent_step_epd_accounted = agent_step_energy_accounted / np.maximum(step_distance, 1e-8)
    agent_step_epd_raw = agent_step_energy_raw / np.maximum(step_distance, 1e-8)
    step_saving_accounted_pct = (ref_step_epd - agent_step_epd_accounted) / np.maximum(ref_step_epd, 1e-8) * 100.0
    step_saving_raw_pct = (ref_step_epd - agent_step_epd_raw) / np.maximum(ref_step_epd, 1e-8) * 100.0

    ref_step_net_epd = np.maximum(ref_step_energy - ref_step_recover, 0.0) / np.maximum(ref_step_distance, 1e-8)
    agent_step_net_epd_accounted = np.maximum(agent_step_energy_accounted - agent_step_recover, 0.0) / np.maximum(step_distance, 1e-8)
    agent_step_net_epd_raw = np.maximum(agent_step_energy_raw - agent_step_recover, 0.0) / np.maximum(step_distance, 1e-8)
    step_net_saving_accounted_pct = (ref_step_net_epd - agent_step_net_epd_accounted) / np.maximum(ref_step_net_epd, 1e-8) * 100.0
    step_net_saving_raw_pct = (ref_step_net_epd - agent_step_net_epd_raw) / np.maximum(ref_step_net_epd, 1e-8) * 100.0

    cum_dist = np.cumsum(step_distance)
    ref_cum_energy = np.cumsum(ref_step_energy)
    agent_cum_energy_accounted = np.cumsum(agent_step_energy_accounted)
    agent_cum_energy_raw = np.cumsum(agent_step_energy_raw)
    ref_cum_net_energy = np.cumsum(np.maximum(ref_step_energy - ref_step_recover, 0.0))
    agent_cum_net_energy_accounted = np.cumsum(np.maximum(agent_step_energy_accounted - agent_step_recover, 0.0))
    agent_cum_net_energy_raw = np.cumsum(np.maximum(agent_step_energy_raw - agent_step_recover, 0.0))

    ref_running_epd = ref_cum_energy / np.maximum(cum_dist, 1e-8)
    agent_running_epd_accounted = agent_cum_energy_accounted / np.maximum(cum_dist, 1e-8)
    agent_running_epd_raw = agent_cum_energy_raw / np.maximum(cum_dist, 1e-8)
    ref_running_net_epd = ref_cum_net_energy / np.maximum(cum_dist, 1e-8)
    agent_running_net_epd_accounted = agent_cum_net_energy_accounted / np.maximum(cum_dist, 1e-8)
    agent_running_net_epd_raw = agent_cum_net_energy_raw / np.maximum(cum_dist, 1e-8)

    running_saving_accounted_pct = (ref_running_epd - agent_running_epd_accounted) / np.maximum(ref_running_epd, 1e-8) * 100.0
    running_saving_raw_pct = (ref_running_epd - agent_running_epd_raw) / np.maximum(ref_running_epd, 1e-8) * 100.0
    running_net_saving_accounted_pct = (ref_running_net_epd - agent_running_net_epd_accounted) / np.maximum(ref_running_net_epd, 1e-8) * 100.0
    running_net_saving_raw_pct = (ref_running_net_epd - agent_running_net_epd_raw) / np.maximum(ref_running_net_epd, 1e-8) * 100.0

    if metrics is None:
        iso_extra_energy = 0.0
        iso_extra_net_energy = 0.0
    else:
        iso_extra_energy = float(metrics.get("distance_comp_energy", 0.0)) + float(metrics.get("kinetic_comp_energy", 0.0))
        iso_extra_net_energy = float(metrics.get("distance_comp_net_energy", 0.0)) + float(metrics.get("kinetic_comp_energy", 0.0))
    progress = (np.arange(n, dtype=np.float32) + 1.0) / max(float(n), 1.0)
    agent_running_epd_iso = (agent_cum_energy_accounted + iso_extra_energy * progress) / np.maximum(cum_dist, 1e-8)
    agent_running_net_epd_iso = (agent_cum_net_energy_accounted + iso_extra_net_energy * progress) / np.maximum(cum_dist, 1e-8)
    running_saving_iso_pct = (ref_running_epd - agent_running_epd_iso) / np.maximum(ref_running_epd, 1e-8) * 100.0
    running_net_saving_iso_pct = (ref_running_net_epd - agent_running_net_epd_iso) / np.maximum(ref_running_net_epd, 1e-8) * 100.0

    def rolling_sum(arr, window=10):
        sums = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            start = max(0, i - window + 1)
            sums[i] = float(np.sum(arr[start : i + 1]))
        return sums

    def rolling_energy_dist_ratio(step_energy, step_dist, window=10):
        energy_sum = rolling_sum(step_energy, window=window)
        dist_sum = rolling_sum(step_dist, window=window)
        return energy_sum / np.maximum(dist_sum, 1e-8)

    rolling_ref_epd = rolling_energy_dist_ratio(ref_step_energy, ref_step_distance, window=10)
    rolling_agent_epd_accounted = rolling_energy_dist_ratio(agent_step_energy_accounted, step_distance, window=10)
    rolling_agent_epd_raw = rolling_energy_dist_ratio(agent_step_energy_raw, step_distance, window=10)
    rolling_ref_energy = rolling_sum(ref_step_energy, window=10)
    rolling_agent_energy_accounted = rolling_sum(agent_step_energy_accounted, window=10)
    rolling_agent_energy_raw = rolling_sum(agent_step_energy_raw, window=10)
    rolling_ref_net_epd = rolling_energy_dist_ratio(np.maximum(ref_step_energy - ref_step_recover, 0.0), ref_step_distance, window=10)
    rolling_agent_net_epd_accounted = rolling_energy_dist_ratio(
        np.maximum(agent_step_energy_accounted - agent_step_recover, 0.0),
        step_distance,
        window=10,
    )
    rolling_agent_net_epd_raw = rolling_energy_dist_ratio(
        np.maximum(agent_step_energy_raw - agent_step_recover, 0.0),
        step_distance,
        window=10,
    )
    rolling_ref_net_energy = rolling_sum(np.maximum(ref_step_energy - ref_step_recover, 0.0), window=10)
    rolling_agent_net_energy_accounted = rolling_sum(np.maximum(agent_step_energy_accounted - agent_step_recover, 0.0), window=10)
    rolling_agent_net_energy_raw = rolling_sum(np.maximum(agent_step_energy_raw - agent_step_recover, 0.0), window=10)
    rolling_saving_accounted_pct = (rolling_ref_epd - rolling_agent_epd_accounted) / np.maximum(rolling_ref_epd, 1e-8) * 100.0
    rolling_saving_raw_pct = (rolling_ref_epd - rolling_agent_epd_raw) / np.maximum(rolling_ref_epd, 1e-8) * 100.0
    rolling_net_saving_accounted_pct = (rolling_ref_net_epd - rolling_agent_net_epd_accounted) / np.maximum(rolling_ref_net_epd, 1e-8) * 100.0
    rolling_net_saving_raw_pct = (rolling_ref_net_epd - rolling_agent_net_epd_raw) / np.maximum(rolling_ref_net_epd, 1e-8) * 100.0
    rolling_raw_energy_saved = rolling_ref_energy - rolling_agent_energy_raw
    rolling_accounted_energy_saved = rolling_ref_energy - rolling_agent_energy_accounted
    rolling_pure_energy_saved = rolling_ref_energy - rolling_agent_energy_accounted
    rolling_pure_net_energy_saved = rolling_ref_net_energy - rolling_agent_net_energy_accounted
    if n > 0 and (iso_extra_energy != 0.0 or iso_extra_net_energy != 0.0):
        iso_step_energy = np.full(n, iso_extra_energy / max(float(n), 1.0), dtype=np.float32)
        iso_step_net_energy = np.full(n, iso_extra_net_energy / max(float(n), 1.0), dtype=np.float32)
        rolling_pure_energy_saved = rolling_ref_energy - (rolling_agent_energy_accounted + rolling_sum(iso_step_energy, window=10))
        rolling_pure_net_energy_saved = rolling_ref_net_energy - (
            rolling_agent_net_energy_accounted + rolling_sum(iso_step_net_energy, window=10)
        )

    return {
        "step": np.arange(1, n + 1, dtype=np.int32),
        "step_distance": step_distance,
        "ref_step_epd": ref_step_epd,
        "agent_step_epd_accounted": agent_step_epd_accounted,
        "agent_step_epd_raw": agent_step_epd_raw,
        "step_saving_accounted_pct": step_saving_accounted_pct,
        "step_saving_raw_pct": step_saving_raw_pct,
        "step_net_saving_accounted_pct": step_net_saving_accounted_pct,
        "step_net_saving_raw_pct": step_net_saving_raw_pct,
        "running_saving_accounted_pct": running_saving_accounted_pct,
        "running_saving_raw_pct": running_saving_raw_pct,
        "running_saving_iso_pct": running_saving_iso_pct,
        "running_net_saving_accounted_pct": running_net_saving_accounted_pct,
        "running_net_saving_raw_pct": running_net_saving_raw_pct,
        "running_net_saving_iso_pct": running_net_saving_iso_pct,
        "rolling_saving_accounted_pct": rolling_saving_accounted_pct,
        "rolling_saving_raw_pct": rolling_saving_raw_pct,
        "rolling_net_saving_accounted_pct": rolling_net_saving_accounted_pct,
        "rolling_net_saving_raw_pct": rolling_net_saving_raw_pct,
        "rolling_raw_energy_saved": rolling_raw_energy_saved,
        "rolling_accounted_energy_saved": rolling_accounted_energy_saved,
        "rolling_pure_energy_saved": rolling_pure_energy_saved,
        "rolling_pure_net_energy_saved": rolling_pure_net_energy_saved,
    }


def export_saving_trace_csv(driver_ref, best_rollout, metrics=None, output_path="saving_trace.csv"):
    """
    作用：把逐步节能轨迹导出为 CSV。
    输入：driver_ref: 驾驶员参考；best_rollout: 智能体 rollout；metrics: 可选评估指标。
    输出：无。
    """
    trace = build_saving_trace(driver_ref, best_rollout, metrics=metrics)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(trace.keys())
    stacked = np.column_stack([np.asarray(trace[k]) for k in trace.keys()])
    np.savetxt(output_path, stacked, delimiter=",", header=header, comments="", fmt="%.8f")


def plot_saving_trace(driver_ref, best_rollout, metrics=None, output_path="saving_trace.png"):
    """
    作用：绘制节能率随 step 变化的曲线，突出 raw/accounted/isochronous 三种口径。
    输入：driver_ref: 驾驶员参考；best_rollout: 智能体 rollout；metrics: 可选评估指标。
    输出：无，保存图片到文件。
    """
    trace = build_saving_trace(driver_ref, best_rollout, metrics=metrics)
    t = trace["step"]
    _apply_plot_theme()
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 8.6), sharex=True)

    ax = axes[0]
    ax.plot(t, trace["running_saving_raw_pct"], color="#9ca3af", linewidth=1.8, label="Running Raw Saving")
    ax.plot(t, trace["running_saving_accounted_pct"], color="#ea580c", linewidth=2.1, label="Running Accounted Saving")
    ax.plot(t, trace["running_saving_iso_pct"], color="#155eef", linewidth=2.4, label="Running Pure Ctrl Saving")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Gross Saving (%)")
    ax.set_title("Saving Over Steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _kpi_box(
        ax,
        (
            f"Final Pure: {float(trace['running_saving_iso_pct'][-1]):.2f}%\n"
            f"Final Accounted: {float(trace['running_saving_accounted_pct'][-1]):.2f}%\n"
            f"Final Raw: {float(trace['running_saving_raw_pct'][-1]):.2f}%"
        ),
    )

    ax = axes[1]
    ax.plot(t, trace["step_saving_raw_pct"], color="#cbd5e1", linewidth=1.2, alpha=0.9, label="Step Raw Saving")
    ax.plot(t, trace["step_saving_accounted_pct"], color="#dc2626", linewidth=1.5, alpha=0.95, label="Step Accounted Saving")
    ax.plot(t, trace["running_net_saving_iso_pct"], color="#0891b2", linewidth=2.0, label="Running Pure Net Saving")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Saving (%)")
    ax.set_title("Step Saving + Running Net Saving")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_window_saving_trace(driver_ref, best_rollout, metrics=None, output_path="saving_window_trace.png"):
    """
    作用：绘制 10 步滑动窗口节能率，突出后半段局部节能表现。
    输入：driver_ref: 驾驶员参考；best_rollout: 智能体 rollout；metrics: 可选评估指标。
    输出：无，保存图片到文件。
    """
    trace = build_saving_trace(driver_ref, best_rollout, metrics=metrics)
    t = trace["step"]
    _apply_plot_theme()
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 8.2), sharex=True)

    ax = axes[0]
    ax.plot(t, trace["rolling_raw_energy_saved"], color="#9ca3af", linewidth=1.6, label="10-step Raw Energy Saved")
    ax.plot(t, trace["rolling_accounted_energy_saved"], color="#ea580c", linewidth=1.9, label="10-step Accounted Energy Saved")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Energy Saved")
    ax.set_title("10-Step Rolling Gross Energy Saved")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _kpi_box(
        ax,
        (
            f"Window Accounted(end): {float(trace['rolling_accounted_energy_saved'][-1]):.6f}\n"
            f"Window Raw(end): {float(trace['rolling_raw_energy_saved'][-1]):.6f}"
        ),
    )

    ax = axes[1]
    ax.plot(t, trace["rolling_pure_energy_saved"], color="#155eef", linewidth=2.1, label="10-step Pure Energy Saved")
    ax.plot(t, trace["rolling_pure_net_energy_saved"], color="#0891b2", linewidth=2.0, label="10-step Pure Net Energy Saved")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy Saved")
    ax.set_title("10-Step Rolling Pure Energy Saved")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(mean_return_curve, mean_saving_curve, stage_split=None, output_path="motor_ppo_result.png"):
    """
    作用：绘制跨 seed 平均后的训练回报和节能趋势图。
    输入：mean_return_curve: 平均回报曲线；mean_saving_curve: 平均节能曲线；stage_split: 阶段切换位置。
    输出：无，保存图片到文件。
    """
    def moving_average(arr, window):
        """
        作用：计算不居中的移动平均并返回对齐后的横轴。
        输入：arr: 原序列；window: 窗口大小。
        输出：平滑后的序列和对应横轴索引。
        """
        window = int(max(3, min(window, len(arr))))
        if window >= len(arr):
            return np.array(arr, dtype=np.float32), np.arange(len(arr))
        smooth = np.convolve(arr, np.ones(window) / window, mode="valid")
        x_idx = np.arange(window - 1, len(arr))
        return smooth, x_idx

    def ema(arr, alpha=0.08):
        """
        作用：用简单指数滑动平均平滑序列。
        输入：arr: 原序列；alpha: 平滑系数。
        输出：平滑后的数组。
        """
        out = np.array(arr, dtype=np.float32).copy()
        if out.size == 0:
            return out
        for i in range(1, out.size):
            out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]
        return out

    window_size = 30
    return_smooth, x_return = moving_average(mean_return_curve, window_size)
    saving_smooth, x_saving = moving_average(mean_saving_curve, window_size)
    saving_ema = ema(mean_saving_curve, alpha=0.08)
    x_full = np.arange(len(mean_return_curve))

    _apply_plot_theme()
    plt.figure(figsize=(13.5, 6.0))
    plt.subplot(1, 2, 1)
    plt.plot(x_full, mean_return_curve, color="#93c5fd", linewidth=1.0, alpha=0.35, label="Raw Return")
    plt.plot(x_return, return_smooth, color="#155eef", linewidth=2.4, label="Moving Avg Return")
    if stage_split is not None and stage_split >= (window_size - 1):
        plt.axvspan(stage_split, len(x_full) - 1, color="#fde68a", alpha=0.12)
        plt.axvline(stage_split, color="#b45309", linestyle="--", linewidth=1.5, label="Energy Stage")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Mean Returns")
    plt.title("Two-Stage PPO (Return)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    _kpi_box(
        plt.gca(),
        f"Final return(avg): {float(return_smooth[-1]):.2f}\nEpisodes: {len(mean_return_curve)}",
    )

    plt.subplot(1, 2, 2)
    plt.plot(x_full, mean_saving_curve, color="#f59e0b", linewidth=1.0, alpha=0.20, label="Raw Saving")
    plt.plot(x_full, saving_ema, color="#c2410c", linewidth=1.8, alpha=0.9, label="EMA Saving")
    plt.plot(x_saving, saving_smooth, color="#ea580c", linewidth=2.5, label="Moving Avg Saving")
    plt.fill_between(x_saving, 0.0, saving_smooth, where=saving_smooth >= 0.0, color="#fdba74", alpha=0.28)
    plt.fill_between(x_saving, 0.0, saving_smooth, where=saving_smooth < 0.0, color="#fca5a5", alpha=0.20)
    if stage_split is not None and stage_split >= (window_size - 1):
        plt.axvspan(stage_split, len(x_full) - 1, color="#fde68a", alpha=0.12)
        plt.axvline(stage_split, color="#b45309", linestyle="--", linewidth=1.5, label="Energy Stage")
    plt.axhline(y=0.0, color="#16a34a", linestyle="--", linewidth=1.5, label="No Saving (0%)")
    plt.xlabel("Episodes")
    plt.ylabel("Saving(E/Dist) [% vs episode driver]")
    plt.title("Two-Stage PPO (Saving % Trend)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _kpi_box(
        plt.gca(),
        f"Final saving(avg): {float(saving_smooth[-1]):.2f}%\nPeak EMA: {float(np.max(saving_ema)):.2f}%",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tracking(best_rollout, driver_ref, metrics=None, output_path="motor_ppo_trajectory.png"):
    """
    作用：可视化最优智能体在跟踪驾驶员同时节能的整体表现。
    输入：best_rollout: 智能体轨迹；driver_ref: 驾驶员参考轨迹。
    输出：无，保存图片到文件。
    """
    def running_epd(step_energy, step_distance):
        """
        作用：按累计距离计算运行中的单位距离能耗。
        输入：step_energy: 每步能耗；step_distance: 每步距离。
        输出：累计能耗强度数组。
        """
        cum_energy = np.cumsum(step_energy)
        cum_dist = np.cumsum(step_distance)
        return cum_energy / np.maximum(cum_dist, 1e-8)

    t = np.arange(len(best_rollout["speed"]))
    u_t = np.arange(len(best_rollout["residual"]))
    speed_ref_kmh = best_rollout["ref_speed"] * MPS_TO_KMH
    speed_agent_kmh = best_rollout["speed"] * MPS_TO_KMH
    speed_err = best_rollout["speed"] - best_rollout["ref_speed"]
    speed_err_kmh = speed_err * MPS_TO_KMH
    dist_err = best_rollout["distance"] - best_rollout["ref_distance"]
    if metrics is None:
        saving_epd = (driver_ref["energy_per_dist"] - best_rollout["energy_per_dist"]) / max(driver_ref["energy_per_dist"], 1e-12) * 100.0
        saving_total = (driver_ref["total_energy"] - best_rollout["total_energy"]) / max(driver_ref["total_energy"], 1e-12) * 100.0
        saving_net_epd = (
            (driver_ref["net_energy"] / max(driver_ref["total_distance"], 1e-8) - best_rollout["net_energy_per_dist"])
            / max(driver_ref["net_energy"] / max(driver_ref["total_distance"], 1e-8), 1e-12)
            * 100.0
        )
        saving_epd_raw = saving_epd
        saving_total_raw = saving_total
        saving_net_epd_raw = saving_net_epd
    else:
        saving_epd = float(metrics.get("saving_isochronous_pct", metrics.get("saving_epd_pct", 0.0)))
        saving_total = float(metrics.get("saving_total_isochronous_pct", metrics.get("saving_total_pct", 0.0)))
        saving_net_epd = float(metrics.get("saving_net_isochronous_pct", metrics.get("saving_net_epd_pct", 0.0)))
        saving_epd_raw = float(metrics.get("saving_epd_raw_pct", metrics.get("saving_epd_pct", saving_epd)))
        saving_total_raw = float(metrics.get("saving_total_raw_pct", metrics.get("saving_total_pct", saving_total)))
        saving_net_epd_raw = float(metrics.get("saving_net_epd_raw_pct", metrics.get("saving_net_epd_pct", saving_net_epd)))

    driver_epd_running = running_epd(driver_ref["step_energy"], driver_ref["step_distance"])
    driver_net_epd_running = running_epd(
        np.maximum(driver_ref["step_energy"] - driver_ref["step_recover"], 0.0),
        driver_ref["step_distance"],
    )
    agent_epd_running = running_epd(best_rollout["step_energy"], best_rollout["step_distance"])
    agent_net_epd_running = running_epd(
        np.maximum(best_rollout["step_energy"] - best_rollout["step_recover"], 0.0),
        best_rollout["step_distance"],
    )

    _apply_plot_theme()
    fig, axes = plt.subplots(4, 2, figsize=(15.8, 12.6))
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(t, speed_ref_kmh, label="Driver Ref", color="#6b7280", linewidth=2.0)
    ax.plot(t, speed_agent_kmh, label="Agent", color="#155eef", linewidth=2.4)
    ax.fill_between(t, speed_ref_kmh, speed_agent_kmh, color="#93c5fd", alpha=0.18)
    if "curve_speed_limit" in driver_ref:
        ax.plot(t, driver_ref["curve_speed_limit"][: len(t)] * MPS_TO_KMH, label="Curve Limit", color="#9467bd", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Speed Tracking (MAE={best_rollout['speed_mae'] * MPS_TO_KMH:.2f} km/h)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _kpi_box(
        ax,
        f"Pure Ctrl Saving: {saving_epd:.2f}%\nRaw Saving: {saving_epd_raw:.2f}%\nPure Net: {saving_net_epd:.2f}%",
    )

    ax = axes[1]
    ax.plot(t, best_rollout["ref_distance"], label="Driver Ref", color="#6b7280", linewidth=2.0)
    ax.plot(t, best_rollout["distance"], label="Agent", color="#ea580c", linewidth=2.4)
    ax.fill_between(t, best_rollout["ref_distance"], best_rollout["distance"], color="#fdba74", alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title(f"Distance Tracking (MAE={best_rollout['distance_mae']:.3f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    speed_tol_kmh = 0.9 * MPS_TO_KMH
    ax.axhspan(-speed_tol_kmh, speed_tol_kmh, color="#22c55e", alpha=0.14, label="Tolerance Band")
    ax.plot(t, speed_err_kmh, color="#155eef", linewidth=1.8, label="Speed Error")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (km/h)")
    ax.set_title("Speed Error")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.axhspan(-4.8, 4.8, color="#22c55e", alpha=0.14, label="Tolerance Band")
    ax.plot(t, dist_err, color="#ea580c", linewidth=1.8, label="Distance Error")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Error")
    ax.set_title("Distance Error")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(t, best_rollout["ref_torque"], color="#6b7280", linewidth=1.7, label="Driver Torque Cmd")
    ax.plot(u_t, best_rollout["torque_cmd"], color="#dc2626", linewidth=1.8, label="Agent Torque Cmd")
    ax.plot(t, best_rollout["torque"], color="#a16207", linewidth=1.5, label="Motor Actual Torque")
    ax.set_xlabel("Step")
    ax.set_ylabel("Torque")
    ax.set_title("Torque Tracking / Actuator Effect")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(u_t, best_rollout["residual"], color="#dc2626", linewidth=1.8, label="Residual Torque")
    ax.set_xlabel("Step")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(u_t, best_rollout["regen_gain"], color="#16a34a", linewidth=1.4, label="Regen Gain")
    ax2.plot(u_t, best_rollout["coast_gain"], color="#0891b2", linewidth=1.4, label="Coast Gain")
    ax2.set_ylabel("Gain")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.set_title("Control Actions")

    ax = axes[6]
    ax.plot(t, driver_ref["soc"][: len(t)], label="Driver Ref SOC", color="#6b7280", linewidth=2.0)
    ax.plot(t, best_rollout["soc"], label="Agent SOC", color="#16a34a", linewidth=2.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("SOC")
    ax.set_title("SOC Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[7]
    e_t = np.arange(1, len(agent_epd_running) + 1)
    ax.plot(e_t, driver_epd_running, color="#6b7280", linewidth=2.0, label="Driver E/Dist")
    ax.plot(e_t, agent_epd_running, color="#155eef", linewidth=2.2, label="Agent E/Dist")
    ax.plot(e_t, driver_net_epd_running, color="#8c564b", linewidth=1.3, linestyle="--", label="Driver Net E/Dist")
    ax.plot(e_t, agent_net_epd_running, color="#17becf", linewidth=1.3, linestyle="--", label="Agent Net E/Dist")
    ax.set_xlabel("Step")
    ax.set_ylabel("Running Energy/Distance")
    ax.set_title("Cumulative Energy Intensity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        (
            f"Tracking + Energy | Pure Ctrl Saving(E/Dist)={saving_epd:.2f}% | "
            f"Pure Ctrl Saving(Total)={saving_total:.2f}% | Pure Ctrl Net E/Dist={saving_net_epd:.2f}%"
        ),
        fontsize=13,
        y=1.01,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_energy_saving(driver_ref, best_rollout, best_seed, metrics=None, output_path="energy_saving_effect.png"):
    """
    作用：并排绘制驾驶员与智能体的总能耗、毛能耗强度和净能耗强度。
    输入：driver_ref: 驾驶员参考；best_rollout: 智能体轨迹；best_seed: 最优种子。
    输出：无，保存图片到文件。
    """
    def add_bar_labels(ax, bars, fmt):
        """
        作用：给柱状图中的每个柱体标注数值。
        输入：ax: 坐标轴；bars: 柱体集合；fmt: 数值格式。
        输出：无。
        """
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, format(val, fmt), ha="center", va="bottom", fontsize=9)

    labels = ["Driver Baseline", f"Best Agent (seed={best_seed})"]
    if metrics is None:
        total_energy_agent = best_rollout["total_energy"]
        total_energy_agent_raw = total_energy_agent
        e_per_dist_agent = best_rollout["energy_per_dist"]
        e_per_dist_agent_raw = e_per_dist_agent
        driver_net_epd = driver_ref["net_energy"] / max(driver_ref["total_distance"], 1e-8)
        net_e_per_dist_agent = best_rollout["net_energy_per_dist"]
        net_e_per_dist_agent_raw = net_e_per_dist_agent
        saving_total = (driver_ref["total_energy"] - total_energy_agent) / max(driver_ref["total_energy"], 1e-12) * 100.0
        saving_epd = (driver_ref["energy_per_dist"] - e_per_dist_agent) / max(driver_ref["energy_per_dist"], 1e-12) * 100.0
        saving_net_epd = (driver_net_epd - net_e_per_dist_agent) / max(driver_net_epd, 1e-12) * 100.0
        saving_total_raw = saving_total
        saving_epd_raw = saving_epd
        saving_net_epd_raw = saving_net_epd
        low_speed_total = 0.0
        low_speed_epd = 0.0
        low_speed_net = 0.0
    else:
        driver_net_epd = driver_ref["net_energy"] / max(driver_ref["total_distance"], 1e-8)
        saving_total = float(metrics.get("saving_total_isochronous_pct", metrics.get("saving_total_pct", 0.0)))
        saving_epd = float(metrics.get("saving_isochronous_pct", metrics.get("saving_epd_pct", 0.0)))
        saving_net_epd = float(metrics.get("saving_net_isochronous_pct", metrics.get("saving_net_epd_pct", 0.0)))
        saving_total_raw = float(metrics.get("saving_total_raw_pct", metrics.get("saving_total_pct", saving_total)))
        saving_epd_raw = float(metrics.get("saving_epd_raw_pct", metrics.get("saving_epd_pct", saving_epd)))
        saving_net_epd_raw = float(metrics.get("saving_net_epd_raw_pct", metrics.get("saving_net_epd_pct", saving_net_epd)))
        low_speed_total = float(metrics.get("low_speed_benefit_total_pct", 0.0))
        low_speed_epd = float(metrics.get("low_speed_benefit_epd_pct", 0.0))
        low_speed_net = float(metrics.get("low_speed_benefit_net_epd_pct", 0.0))
        total_energy_agent = driver_ref["total_energy"] * (1.0 - saving_total / 100.0)
        total_energy_agent_raw = driver_ref["total_energy"] * (1.0 - saving_total_raw / 100.0)
        e_per_dist_agent = driver_ref["energy_per_dist"] * (1.0 - saving_epd / 100.0)
        e_per_dist_agent_raw = driver_ref["energy_per_dist"] * (1.0 - saving_epd_raw / 100.0)
        net_e_per_dist_agent = driver_net_epd * (1.0 - saving_net_epd / 100.0)
        net_e_per_dist_agent_raw = driver_net_epd * (1.0 - saving_net_epd_raw / 100.0)
    total_energy = [driver_ref["total_energy"], total_energy_agent]
    e_per_dist = [driver_ref["energy_per_dist"], e_per_dist_agent]
    driver_net_epd = driver_ref["net_energy"] / max(driver_ref["total_distance"], 1e-8)
    net_e_per_dist = [driver_net_epd, net_e_per_dist_agent]

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))

    bars1 = axes[0].bar(labels, total_energy, color=["#9ca3af", "#155eef"], width=0.58)
    axes[0].set_ylabel("Total Energy (SOC drop)")
    axes[0].set_title("Total Energy")
    add_bar_labels(axes[0], bars1, ".6f")
    axes[0].axhline(total_energy[0], color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.7)
    _kpi_box(axes[0], f"Pure: {saving_total:.2f}%\nRaw: {saving_total_raw:.2f}%\nLow-speed: {low_speed_total:.2f}%", loc=(0.04, 0.94))

    bars2 = axes[1].bar(labels, e_per_dist, color=["#9ca3af", "#ea580c"], width=0.58)
    axes[1].set_ylabel("Energy Per Distance")
    axes[1].set_title("Gross E/Dist")
    add_bar_labels(axes[1], bars2, ".8f")
    axes[1].axhline(e_per_dist[0], color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.7)
    _kpi_box(axes[1], f"Pure: {saving_epd:.2f}%\nRaw: {saving_epd_raw:.2f}%\nLow-speed: {low_speed_epd:.2f}%", loc=(0.04, 0.94))

    bars3 = axes[2].bar(labels, net_e_per_dist, color=["#9ca3af", "#0891b2"], width=0.58)
    axes[2].set_ylabel("Net Energy Per Distance")
    axes[2].set_title("Net E/Dist")
    add_bar_labels(axes[2], bars3, ".8f")
    axes[2].axhline(net_e_per_dist[0], color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.7)
    _kpi_box(axes[2], f"Pure: {saving_net_epd:.2f}%\nRaw: {saving_net_epd_raw:.2f}%\nLow-speed: {low_speed_net:.2f}%", loc=(0.04, 0.94))

    fig.suptitle(
        f"Pure Control Saving: total={saving_total:.2f}% | gross E/Dist={saving_epd:.2f}% | net E/Dist={saving_net_epd:.2f}%",
        fontsize=12,
        y=1.02,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_motor_efficiency_map(scenario, output_path="motor_efficiency_map_reference.png"):
    """
    作用：绘制场景模型或 CSV 电机图对应的效率地图。
    输入：scenario: 场景对象。
    输出：无，保存图片到文件。
    """
    if scenario.motor_map_loaded and scenario.map_rpm_axis is not None and scenario.map_torque_axis is not None:
        rpm_grid = scenario.map_rpm_axis
        torque_grid = scenario.map_torque_axis
        eff_mesh = scenario.map_eff_grid
        rpm_mesh, tq_mesh = np.meshgrid(rpm_grid, torque_grid)
        title = "Motor Torque-Speed-Efficiency Map (CSV)"
    else:
        rpm_grid = np.linspace(0.0, scenario.max_rpm, 180, dtype=np.float32)
        torque_grid = np.linspace(-scenario.max_torque, scenario.max_torque, 220, dtype=np.float32)
        rpm_mesh, tq_mesh = np.meshgrid(rpm_grid, torque_grid)
        eff_mesh = np.vectorize(scenario.efficiency)(tq_mesh, rpm_mesh)
        title = "Motor Torque-Speed-Efficiency Map (Model)"

    fig, ax = plt.subplots(figsize=(9, 5))
    levels = np.linspace(0.72, 0.97, 18)
    cs = ax.contourf(rpm_mesh, tq_mesh, eff_mesh, levels=levels, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Motor Efficiency")
    ax.contour(rpm_mesh, tq_mesh, eff_mesh, levels=[0.80, 0.85, 0.90, 0.93, 0.95], colors="white", linewidths=0.8)
    ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    ax.set_xlabel("Motor Speed (rpm)")
    ax.set_ylabel("Motor Torque (Nm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_motor_map_csv_template(scenario, csv_path):
    """
    作用：把内置解析电机图导出成可外部编辑的 CSV 模板。
    输入：scenario: 场景对象；csv_path: 输出路径。
    输出：无。
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rpm_grid = np.linspace(0.0, scenario.max_rpm, 60, dtype=np.float32)
    torque_grid = np.linspace(0.0, scenario.max_torque, 50, dtype=np.float32)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rpm,torque,efficiency\n")
        for tq in torque_grid:
            for rpm in rpm_grid:
                eta = scenario.efficiency(float(tq), float(rpm))
                f.write(f"{float(rpm):.6f},{float(tq):.6f},{float(eta):.6f}\n")


