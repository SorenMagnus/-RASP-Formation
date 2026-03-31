"""Paper-oriented table and figure export helpers."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apflf.analysis.stats import write_csv_rows
from apflf.decision.mode_base import parse_mode_label
from apflf.env.geometry import box_clearance
from apflf.env.road import Road
from apflf.safety.cbf import boundary_barrier
from apflf.sim.replay import ReplayBundle, load_replay_bundle

TABLE_METRICS = (
    "leader_goal_error",
    "time_to_goal",
    "min_ttc",
    "min_obstacle_clearance",
    "collision_count",
    "boundary_violation_count",
    "fallback_ratio",
    "terminal_formation_error",
    "longitudinal_jerk_rms",
    "steer_rate_rms",
    "mean_step_runtime_ms",
    "qp_solve_time_mean_ms",
)
FIGURE_METRICS = (
    "leader_goal_error",
    "min_obstacle_clearance",
    "fallback_ratio",
    "terminal_formation_error",
)
METRIC_LABELS = {
    "leader_goal_error": "Goal Error (m)",
    "time_to_goal": "Time to Goal (s)",
    "min_ttc": "Min TTC (s)",
    "min_obstacle_clearance": "Min Clearance (m)",
    "collision_count": "Collisions",
    "boundary_violation_count": "Boundary Violations",
    "fallback_ratio": "Fallback Ratio",
    "terminal_formation_error": "Formation Error (m)",
    "longitudinal_jerk_rms": "Longitudinal Jerk RMS",
    "steer_rate_rms": "Steer Rate RMS",
    "mean_step_runtime_ms": "Mean Step Runtime (ms)",
    "qp_solve_time_mean_ms": "Mean QP Solve Time (ms)",
}
MODE_COMPONENT_VALUES = {
    "topology": ("line", "diamond"),
    "behavior": (
        "follow",
        "yield_left",
        "yield_right",
        "escape_left",
        "escape_right",
        "recover_left",
        "recover_right",
    ),
    "gain": ("nominal", "cautious", "assertive"),
}


def _format_value(mean: object, ci_low: object, ci_high: object) -> str:
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (mean, ci_low, ci_high)
    ):
        return "--"
    mean = float(mean)
    ci_low = float(ci_low)
    ci_high = float(ci_high)
    radius = max(abs(ci_high - mean), abs(mean - ci_low))
    return f"{mean:.3f} +/- {radius:.3f}"


def _escape_latex(value: object) -> str:
    text = str(value)
    return text.replace("_", "\\_").replace("%", "\\%")


def _write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot export an empty LaTeX table.")
    headers = list(rows[0].keys())
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\hline",
        " & ".join(_escape_latex(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(row[header]) for header in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _main_table_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        table_row: dict[str, object] = {
            "scenario": row["scenario"],
            "method": row["method"],
            "n": row["n"],
        }
        for metric in TABLE_METRICS:
            table_row[metric] = _format_value(
                row.get(f"{metric}_mean"),
                row.get(f"{metric}_ci_low"),
                row.get(f"{metric}_ci_high"),
            )
        rows.append(table_row)
    return rows


def _comparison_table_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in comparison_rows:
        rows.append(
            {
                "scenario": row["scenario"],
                "method": row["method"],
                "metric": row["metric"],
                "reference_mean": f"{float(row['reference_mean']):.3f}",
                "method_mean": f"{float(row['method_mean']):.3f}",
                "mean_delta": f"{float(row['mean_delta']):.3f}",
                "delta_ci": _format_value(
                    row["mean_delta"],
                    row["delta_ci_low"],
                    row["delta_ci_high"],
                ),
                "comparison": str(row.get("comparison_kind", "--")),
                "p_value": f"{float(row['p_value']):.4f}"
                if math.isfinite(float(row["p_value"]))
                else "--",
                "wilcoxon_p": f"{float(row['wilcoxon_p_value']):.4f}"
                if math.isfinite(float(row["wilcoxon_p_value"]))
                else "--",
                "cohen_d": f"{float(row['cohen_d']):.3f}"
                if math.isfinite(float(row["cohen_d"]))
                else "--",
            }
        )
    return rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dir(run_dir_value: object) -> Path | None:
    if not isinstance(run_dir_value, str) or not run_dir_value.strip():
        return None
    candidate = Path(run_dir_value)
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _trajectory_artifact_path(run_dir: Path, seed: int) -> Path:
    return run_dir / "traj" / f"seed_{seed:04d}.npz"


def _load_bundle_from_row(
    row: dict[str, object],
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> ReplayBundle | None:
    run_dir = _resolve_run_dir(row.get("output_dir"))
    seed_value = row.get("seed")
    if run_dir is None or not isinstance(seed_value, (int, float)):
        return None
    seed = int(seed_value)
    artifact_path = _trajectory_artifact_path(run_dir, seed)
    if not artifact_path.exists():
        return None
    cache_key = (run_dir, seed)
    if cache_key not in replay_cache:
        replay_cache[cache_key] = load_replay_bundle(run_dir, seed)
    return replay_cache[cache_key]


def _save_placeholder_figure(output_path: Path, *, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.axis("off")
    axis.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.40,
        message,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _state_position_history(bundle: ReplayBundle) -> np.ndarray:
    state_steps = [bundle.initial_states, *[snapshot.states for snapshot in bundle.snapshots]]
    return np.asarray(
        [[[state.x, state.y] for state in states] for states in state_steps],
        dtype=float,
    )


def _obstacle_position_history(bundle: ReplayBundle) -> np.ndarray:
    obstacle_steps = [bundle.initial_obstacles, *[snapshot.obstacles for snapshot in bundle.snapshots]]
    if not obstacle_steps or not obstacle_steps[0]:
        return np.zeros((len(obstacle_steps), 0, 2), dtype=float)
    return np.asarray(
        [[[obstacle.x, obstacle.y] for obstacle in obstacles] for obstacles in obstacle_steps],
        dtype=float,
    )


def _leader_risk_series(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray(
        [snapshot.nominal_diagnostics.leader_risk_score for snapshot in bundle.snapshots],
        dtype=float,
    )


def _leader_target_speed_series(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray(
        [snapshot.nominal_diagnostics.leader_target_speed for snapshot in bundle.snapshots],
        dtype=float,
    )


def _times_from_bundle(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray([snapshot.time for snapshot in bundle.snapshots], dtype=float)


def _min_clearance_series(bundle: ReplayBundle) -> np.ndarray:
    clearances: list[float] = []
    for snapshot in bundle.snapshots:
        min_clearance = float("inf")
        for state_index, state in enumerate(snapshot.states):
            for other_index, other_state in enumerate(snapshot.states):
                if other_index <= state_index:
                    continue
                min_clearance = min(
                    min_clearance,
                    box_clearance(
                        state,
                        bundle.config.controller.vehicle_length,
                        bundle.config.controller.vehicle_width,
                        other_state,
                        bundle.config.controller.vehicle_length,
                        bundle.config.controller.vehicle_width,
                    ),
                )
            for obstacle in snapshot.obstacles:
                min_clearance = min(
                    min_clearance,
                    box_clearance(
                        state,
                        bundle.config.controller.vehicle_length,
                        bundle.config.controller.vehicle_width,
                        obstacle,
                        obstacle.length,
                        obstacle.width,
                    ),
                )
        clearances.append(float("nan") if math.isinf(min_clearance) else float(min_clearance))
    return np.asarray(clearances, dtype=float)


def _min_boundary_margin_series(bundle: ReplayBundle) -> np.ndarray:
    road = Road(bundle.config.scenario.road)
    margins: list[float] = []
    for snapshot in bundle.snapshots:
        margin = min(
            boundary_barrier(
                state=state,
                road=road,
                vehicle_length=bundle.config.controller.vehicle_length,
                vehicle_width=bundle.config.controller.vehicle_width,
            )
            for state in snapshot.states
        )
        margins.append(float(margin))
    return np.asarray(margins, dtype=float)


def _mean_safety_correction_series(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray(
        [
            float(np.mean(np.asarray(snapshot.safety_corrections, dtype=float)))
            if snapshot.safety_corrections
            else 0.0
            for snapshot in bundle.snapshots
        ],
        dtype=float,
    )


def _max_safety_slack_series(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray(
        [
            float(np.max(np.asarray(snapshot.safety_slacks, dtype=float)))
            if snapshot.safety_slacks
            else 0.0
            for snapshot in bundle.snapshots
        ],
        dtype=float,
    )


def _safety_fallback_count_series(bundle: ReplayBundle) -> np.ndarray:
    return np.asarray(
        [
            int(np.count_nonzero(np.asarray(snapshot.safety_fallbacks, dtype=bool)))
            for snapshot in bundle.snapshots
        ],
        dtype=int,
    )


def _mode_component_series(bundle: ReplayBundle, component: str) -> np.ndarray:
    values = MODE_COMPONENT_VALUES[component]
    encoded: list[int] = []
    for snapshot in bundle.snapshots:
        parsed = parse_mode_label(snapshot.mode)
        encoded.append(values.index(getattr(parsed, component)))
    return np.asarray(encoded, dtype=int)


def _mode_change_count(bundle: ReplayBundle) -> int:
    if not bundle.snapshots:
        return 0
    changes = 0
    previous = bundle.snapshots[0].mode
    for snapshot in bundle.snapshots[1:]:
        if snapshot.mode != previous:
            changes += 1
            previous = snapshot.mode
    return changes


def _plot_metric_overview(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        raise ValueError("Metric overview requires at least one summary row.")

    scenarios = sorted({str(row["scenario"]) for row in summary_rows})
    methods = sorted({str(row["method"]) for row in summary_rows})
    index_lookup = {(str(row["scenario"]), str(row["method"])): row for row in summary_rows}
    x_positions = np.arange(len(scenarios), dtype=float)
    bar_width = 0.8 / max(len(methods), 1)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, FIGURE_METRICS, strict=True):
        for method_index, method in enumerate(methods):
            means = []
            lower_errors = []
            upper_errors = []
            for scenario in scenarios:
                row = index_lookup[(scenario, method)]
                mean = float(row[f"{metric}_mean"])
                ci_low = float(row[f"{metric}_ci_low"])
                ci_high = float(row[f"{metric}_ci_high"])
                means.append(mean)
                lower_errors.append(max(mean - ci_low, 0.0))
                upper_errors.append(max(ci_high - mean, 0.0))
            centers = x_positions + (method_index - 0.5 * (len(methods) - 1)) * bar_width
            axis.bar(centers, means, width=bar_width, label=method, alpha=0.88)
            axis.errorbar(
                centers,
                means,
                yerr=np.asarray([lower_errors, upper_errors], dtype=float),
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3.0,
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(scenarios, rotation=15)
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=min(len(methods), 4))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_safety_efficiency(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        raise ValueError("Safety-efficiency figure requires at least one summary row.")

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    methods = sorted({str(row["method"]) for row in summary_rows})
    color_map = plt.get_cmap("tab10")
    for method_index, method in enumerate(methods):
        method_rows = [row for row in summary_rows if str(row["method"]) == method]
        x_values = [float(row["leader_goal_error_mean"]) for row in method_rows]
        y_values = [float(row["fallback_ratio_mean"]) for row in method_rows]
        axis.scatter(
            x_values,
            y_values,
            s=70,
            label=method,
            color=color_map(method_index % color_map.N),
            alpha=0.9,
        )
        for row, x_value, y_value in zip(method_rows, x_values, y_values, strict=True):
            axis.annotate(str(row["scenario"]), (x_value, y_value), fontsize=8, alpha=0.8)
    axis.set_xlabel(METRIC_LABELS["leader_goal_error"])
    axis.set_ylabel(METRIC_LABELS["fallback_ratio"])
    axis.set_title("Safety-Efficiency Trade-off")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _select_representative_rows(
    raw_rows: list[dict[str, object]],
    *,
    preferred_method: str = "no_rl",
    limit: int = 4,
) -> list[dict[str, object]]:
    rows_by_scenario: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        rows_by_scenario.setdefault(str(row.get("scenario", "")), []).append(row)

    selected: list[dict[str, object]] = []
    for scenario in sorted(rows_by_scenario):
        scenario_rows = sorted(
            rows_by_scenario[scenario],
            key=lambda row: (
                str(row.get("method", "")) != preferred_method,
                str(row.get("method", "")),
                int(row.get("seed", 0)),
            ),
        )
        if scenario_rows:
            selected.append(scenario_rows[0])
        if len(selected) >= limit:
            break
    return selected


def _plot_trajectory_overview(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    representatives = _select_representative_rows(raw_rows, preferred_method="no_rl", limit=4)
    bundles: list[tuple[dict[str, object], ReplayBundle]] = []
    for row in representatives:
        bundle = _load_bundle_from_row(row, replay_cache)
        if bundle is not None:
            bundles.append((row, bundle))
    if not bundles:
        _save_placeholder_figure(
            output_path,
            title="Trajectory Overview",
            message="No replayable trajectory artifacts were available for the selected runs.",
        )
        return

    columns = min(2, len(bundles))
    rows = int(math.ceil(len(bundles) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 4.5 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis, (row, bundle) in zip(axes_array.flat, bundles, strict=False):
        state_positions = _state_position_history(bundle)
        obstacle_positions = _obstacle_position_history(bundle)
        road = bundle.config.scenario.road
        axis.axhline(road.lane_center_y + road.half_width, color="black", linewidth=1.0, linestyle="--")
        axis.axhline(road.lane_center_y - road.half_width, color="black", linewidth=1.0, linestyle="--")
        for vehicle_index in range(state_positions.shape[1]):
            axis.plot(
                state_positions[:, vehicle_index, 0],
                state_positions[:, vehicle_index, 1],
                linewidth=2.2 if vehicle_index == 0 else 1.6,
                label=f"veh{vehicle_index}" if vehicle_index < 3 else None,
            )
        for obstacle_index in range(obstacle_positions.shape[1]):
            axis.plot(
                obstacle_positions[:, obstacle_index, 0],
                obstacle_positions[:, obstacle_index, 1],
                color="grey",
                alpha=0.7,
                linewidth=1.1,
            )
        axis.set_title(
            f"{row['scenario']} | {row['method']} | seed={int(row['seed'])}",
            fontsize=10,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.25)
    for axis in axes_array.flat[len(bundles) :]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _representative_row_for_diagnostics(raw_rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not raw_rows:
        return None
    priorities = ("rl_param_only", "no_rl", "adaptive_apf")
    sorted_rows = sorted(
        raw_rows,
        key=lambda row: (
            next(
                (index for index, method in enumerate(priorities) if str(row.get("method", "")) == method),
                len(priorities),
            ),
            str(row.get("scenario", "")),
            int(row.get("seed", 0)),
        ),
    )
    return sorted_rows[0]


def _plot_risk_clearance_timeseries(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    row = _representative_row_for_diagnostics(raw_rows)
    bundle = None if row is None else _load_bundle_from_row(row, replay_cache)
    if row is None or bundle is None:
        _save_placeholder_figure(
            output_path,
            title="Risk / Clearance Time Series",
            message="No replayable artifact was available for the representative diagnostic run.",
        )
        return

    times = _times_from_bundle(bundle)
    risk = _leader_risk_series(bundle)
    clearance = _min_clearance_series(bundle)
    boundary_margin = _min_boundary_margin_series(bundle)
    target_speed = _leader_target_speed_series(bundle)

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    axes[0].plot(times, risk, color="tab:red", label="leader risk")
    axes[0].set_ylabel("Risk")
    axes[0].set_title(f"{row['scenario']} | {row['method']} | seed={int(row['seed'])}")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, clearance, color="tab:blue", label="min clearance")
    axes[1].plot(times, boundary_margin, color="tab:green", label="boundary margin")
    axes[1].set_ylabel("Safety Margin (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(times, target_speed, color="tab:purple", label="leader target speed")
    axes[2].set_ylabel("Target Speed (m/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_qp_correction_timeline(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    row = _representative_row_for_diagnostics(raw_rows)
    bundle = None if row is None else _load_bundle_from_row(row, replay_cache)
    if row is None or bundle is None:
        _save_placeholder_figure(
            output_path,
            title="QP Correction Timeline",
            message="No replayable artifact was available for the representative diagnostic run.",
        )
        return

    times = _times_from_bundle(bundle)
    mean_correction = _mean_safety_correction_series(bundle)
    max_slack = _max_safety_slack_series(bundle)
    fallback_count = _safety_fallback_count_series(bundle)

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    axes[0].plot(times, mean_correction, color="tab:orange")
    axes[0].set_ylabel("Mean QP Correction")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, max_slack, color="tab:olive")
    axes[1].set_ylabel("Max Slack")
    axes[1].grid(alpha=0.25)
    axes[2].step(times, fallback_count, where="post", color="tab:red")
    axes[2].set_ylabel("Fallback Count")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_mode_timeline(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    candidate_rows = sorted(
        raw_rows,
        key=lambda row: (
            str(row.get("method", "")) != "no_rl",
            str(row.get("scenario", "")),
            int(row.get("seed", 0)),
        ),
    )
    bundles: list[tuple[dict[str, object], ReplayBundle]] = []
    for row in candidate_rows:
        bundle = _load_bundle_from_row(row, replay_cache)
        if bundle is not None:
            bundles.append((row, bundle))
    if not bundles:
        _save_placeholder_figure(
            output_path,
            title="Mode Timeline",
            message="No replayable artifact was available for mode-transition visualization.",
        )
        return
    row, bundle = max(bundles, key=lambda item: _mode_change_count(item[1]))

    times = _times_from_bundle(bundle)
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    for axis, component in zip(axes, ("topology", "behavior", "gain"), strict=True):
        values = MODE_COMPONENT_VALUES[component]
        encoded = _mode_component_series(bundle, component)
        axis.step(times, encoded, where="post", color="tab:blue")
        axis.set_yticks(range(len(values)))
        axis.set_yticklabels(values)
        axis.set_ylabel(component.title())
        axis.grid(alpha=0.25)
    axes[0].set_title(f"{row['scenario']} | {row['method']} | seed={int(row['seed'])}")
    axes[-1].set_xlabel("Time (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_runtime_histogram(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    step_runtimes: list[float] = []
    safety_runtimes: list[float] = []
    for row in raw_rows:
        bundle = _load_bundle_from_row(row, replay_cache)
        if bundle is None:
            continue
        step_runtimes.extend(snapshot.step_runtime * 1000.0 for snapshot in bundle.snapshots)
        safety_runtimes.extend(snapshot.safety_runtime * 1000.0 for snapshot in bundle.snapshots)
    if not step_runtimes:
        _save_placeholder_figure(
            output_path,
            title="Runtime Histogram",
            message="No replayable artifact was available for runtime aggregation.",
        )
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(step_runtimes, bins=30, color="tab:blue", alpha=0.85)
    axes[0].set_title("Step Runtime")
    axes[0].set_xlabel("Milliseconds")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)
    axes[1].hist(safety_runtimes, bins=30, color="tab:orange", alpha=0.85)
    axes[1].set_title("Safety Runtime")
    axes[1].set_xlabel("Milliseconds")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_failure_case_panel(
    raw_rows: list[dict[str, object]],
    output_path: Path,
    replay_cache: dict[tuple[Path, int], ReplayBundle | None],
) -> None:
    if not raw_rows:
        _save_placeholder_figure(
            output_path,
            title="Failure Case Panel",
            message="No run rows were available for failure-case selection.",
        )
        return
    row = max(
        raw_rows,
        key=lambda current: (
            float(current.get("collision_count", 0)),
            float(current.get("boundary_violation_count", 0)),
            float(current.get("leader_goal_error", 0.0)),
        ),
    )
    bundle = _load_bundle_from_row(row, replay_cache)
    if bundle is None:
        _save_placeholder_figure(
            output_path,
            title="Failure Case Panel",
            message="The selected failure case did not have a replayable trajectory artifact.",
        )
        return

    state_positions = _state_position_history(bundle)
    obstacle_positions = _obstacle_position_history(bundle)
    times = _times_from_bundle(bundle)
    clearance = _min_clearance_series(bundle)
    mean_correction = _mean_safety_correction_series(bundle)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    road = bundle.config.scenario.road
    axes[0].axhline(road.lane_center_y + road.half_width, color="black", linewidth=1.0, linestyle="--")
    axes[0].axhline(road.lane_center_y - road.half_width, color="black", linewidth=1.0, linestyle="--")
    for vehicle_index in range(state_positions.shape[1]):
        axes[0].plot(
            state_positions[:, vehicle_index, 0],
            state_positions[:, vehicle_index, 1],
            linewidth=2.2 if vehicle_index == 0 else 1.6,
        )
    for obstacle_index in range(obstacle_positions.shape[1]):
        axes[0].plot(
            obstacle_positions[:, obstacle_index, 0],
            obstacle_positions[:, obstacle_index, 1],
            color="grey",
            alpha=0.7,
            linewidth=1.1,
        )
    axes[0].set_title(
        f"Failure Candidate: {row['scenario']} | {row['method']} | seed={int(row['seed'])}",
        fontsize=10,
    )
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, clearance, color="tab:red", label="min clearance")
    axes[1].plot(times, mean_correction, color="tab:blue", label="mean qp correction")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Safety Margin vs QP Intervention")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def export_paper_artifacts(
    *,
    raw_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Export CSV/LaTeX tables and paper-ready PDF figures."""

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    main_table_rows = _main_table_rows(summary_rows)
    comparison_table_rows = _comparison_table_rows(comparison_rows)

    write_csv_rows(raw_rows, output_dir / "all_runs.csv")
    write_csv_rows(summary_rows, tables_dir / "main_results.csv")
    write_csv_rows(main_table_rows, tables_dir / "main_results_pretty.csv")
    write_csv_rows(comparison_rows, tables_dir / "pairwise_vs_reference.csv")
    if comparison_table_rows:
        write_csv_rows(comparison_table_rows, tables_dir / "pairwise_vs_reference_pretty.csv")
    _write_latex_table(main_table_rows, tables_dir / "main_results.tex")
    if comparison_table_rows:
        _write_latex_table(comparison_table_rows, tables_dir / "pairwise_vs_reference.tex")

    replay_cache: dict[tuple[Path, int], ReplayBundle | None] = {}
    _plot_metric_overview(summary_rows, figures_dir / "metric_overview.pdf")
    _plot_safety_efficiency(summary_rows, figures_dir / "safety_efficiency_tradeoff.pdf")
    _plot_trajectory_overview(raw_rows, figures_dir / "trajectory_overview.pdf", replay_cache)
    _plot_risk_clearance_timeseries(raw_rows, figures_dir / "risk_clearance_timeseries.pdf", replay_cache)
    _plot_qp_correction_timeline(raw_rows, figures_dir / "qp_correction_timeline.pdf", replay_cache)
    _plot_mode_timeline(raw_rows, figures_dir / "mode_timeline.pdf", replay_cache)
    _plot_runtime_histogram(raw_rows, figures_dir / "runtime_histogram.pdf", replay_cache)
    _plot_failure_case_panel(raw_rows, figures_dir / "failure_case_panel.pdf", replay_cache)
