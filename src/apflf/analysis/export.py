"""Paper-oriented table and figure export helpers."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apflf.analysis.stats import write_csv_rows

TABLE_METRICS = (
    "leader_goal_error",
    "time_to_goal",
    "min_ttc",
    "min_obstacle_clearance",
    "collision_count",
    "boundary_violation_count",
    "fallback_ratio",
    "terminal_formation_error",
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
}


def _format_value(mean: object, ci_low: object, ci_high: object) -> str:
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (mean, ci_low, ci_high)):
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
                row[f"{metric}_mean"],
                row[f"{metric}_ci_low"],
                row[f"{metric}_ci_high"],
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
                "p_value": f"{float(row['p_value']):.4f}" if math.isfinite(float(row["p_value"])) else "--",
                "wilcoxon_p": f"{float(row['wilcoxon_p_value']):.4f}"
                if math.isfinite(float(row["wilcoxon_p_value"]))
                else "--",
                "cohen_d": f"{float(row['cohen_d']):.3f}" if math.isfinite(float(row["cohen_d"])) else "--",
            }
        )
    return rows


def _plot_metric_overview(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        raise ValueError("Metric overview requires at least one summary row.")

    scenarios = sorted({str(row["scenario"]) for row in summary_rows})
    methods = sorted({str(row["method"]) for row in summary_rows})
    index_lookup = {
        (str(row["scenario"]), str(row["method"])): row for row in summary_rows
    }
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
            axis.bar(
                centers,
                means,
                width=bar_width,
                label=method,
                alpha=0.88,
            )
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

    _plot_metric_overview(summary_rows, figures_dir / "metric_overview.pdf")
    _plot_safety_efficiency(summary_rows, figures_dir / "safety_efficiency_tradeoff.pdf")
