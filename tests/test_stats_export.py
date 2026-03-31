"""Tests for paper statistics and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apflf.analysis.export import export_paper_artifacts
from apflf.analysis.stats import (
    aggregate_metric,
    aggregate_metric_with_ci,
    pairwise_compare_to_reference,
    summarize_experiments,
)


def _raw_rows() -> list[dict[str, object]]:
    return [
        {
            "scenario": "s1_local_minima",
            "method": "no_rl",
            "seed": 0,
            "leader_goal_error": 10.0,
            "time_to_goal": 12.0,
            "mean_speed": 5.0,
            "leader_path_length_ratio": 1.05,
            "min_ttc": 1.8,
            "min_obstacle_clearance": 0.8,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.2,
            "terminal_formation_error": 1.0,
            "longitudinal_jerk_rms": 0.4,
            "steer_rate_rms": 0.3,
            "mean_step_runtime_ms": 12.0,
            "qp_solve_time_mean_ms": 1.2,
        },
        {
            "scenario": "s1_local_minima",
            "method": "no_rl",
            "seed": 1,
            "leader_goal_error": 8.0,
            "time_to_goal": 11.5,
            "mean_speed": 5.5,
            "leader_path_length_ratio": 1.03,
            "min_ttc": 2.1,
            "min_obstacle_clearance": 0.9,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.1,
            "terminal_formation_error": 0.8,
            "longitudinal_jerk_rms": 0.3,
            "steer_rate_rms": 0.2,
            "mean_step_runtime_ms": 11.0,
            "qp_solve_time_mean_ms": 1.0,
        },
        {
            "scenario": "s1_local_minima",
            "method": "apf",
            "seed": 0,
            "leader_goal_error": 14.0,
            "time_to_goal": 14.0,
            "mean_speed": 4.0,
            "leader_path_length_ratio": 1.12,
            "min_ttc": 0.9,
            "min_obstacle_clearance": 0.4,
            "collision_count": 1,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.3,
            "terminal_formation_error": 1.6,
            "longitudinal_jerk_rms": 0.9,
            "steer_rate_rms": 0.7,
            "mean_step_runtime_ms": 10.0,
            "qp_solve_time_mean_ms": 0.8,
        },
        {
            "scenario": "s1_local_minima",
            "method": "apf",
            "seed": 1,
            "leader_goal_error": 15.0,
            "time_to_goal": 14.5,
            "mean_speed": 3.8,
            "leader_path_length_ratio": 1.15,
            "min_ttc": 0.8,
            "min_obstacle_clearance": 0.3,
            "collision_count": 1,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.35,
            "terminal_formation_error": 1.8,
            "longitudinal_jerk_rms": 1.0,
            "steer_rate_rms": 0.8,
            "mean_step_runtime_ms": 10.5,
            "qp_solve_time_mean_ms": 0.9,
        },
    ]


def test_aggregate_metric_returns_mean_and_ci() -> None:
    aggregate = aggregate_metric(values=np.asarray([1.0, 2.0, 3.0]))
    assert aggregate.count == 3
    assert aggregate.mean == 2.0
    assert aggregate.ci_low <= aggregate.mean <= aggregate.ci_high


def test_aggregate_metric_constant_values_collapse_ci() -> None:
    aggregate = aggregate_metric(values=np.asarray([2.5, 2.5, 2.5]))
    assert aggregate.count == 3
    assert aggregate.mean == 2.5
    assert aggregate.std == 0.0
    assert aggregate.ci_low == 2.5
    assert aggregate.ci_high == 2.5


def test_aggregate_metric_with_t_interval_is_supported() -> None:
    aggregate = aggregate_metric_with_ci(
        np.asarray([1.0, 2.0, 3.0]),
        ci_method="t",
    )
    assert aggregate.count == 3
    assert aggregate.ci_low <= aggregate.mean <= aggregate.ci_high


def test_summarize_and_pairwise_comparison_generate_rows() -> None:
    summary_rows = summarize_experiments(_raw_rows())
    comparison_rows = pairwise_compare_to_reference(
        _raw_rows(),
        reference_method="no_rl",
    )

    assert len(summary_rows) == 2
    assert any(row["method"] == "apf" for row in summary_rows)
    assert any(row["metric"] == "leader_goal_error" for row in comparison_rows)


def test_pairwise_comparison_handles_identical_constant_samples() -> None:
    rows = [
        {
            "scenario": "s_constant",
            "method": "no_rl",
            "leader_goal_error": 5.0,
            "mean_speed": 3.0,
            "min_obstacle_clearance": 0.8,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.2,
            "terminal_formation_error": 0.5,
            "longitudinal_jerk_rms": 0.1,
            "steer_rate_rms": 0.1,
            "mean_step_runtime_ms": 10.0,
            "qp_solve_time_mean_ms": 1.0,
        },
        {
            "scenario": "s_constant",
            "method": "no_rl",
            "leader_goal_error": 5.0,
            "mean_speed": 3.0,
            "min_obstacle_clearance": 0.8,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.2,
            "terminal_formation_error": 0.5,
            "longitudinal_jerk_rms": 0.1,
            "steer_rate_rms": 0.1,
            "mean_step_runtime_ms": 10.0,
            "qp_solve_time_mean_ms": 1.0,
        },
        {
            "scenario": "s_constant",
            "method": "apf",
            "leader_goal_error": 5.0,
            "mean_speed": 3.0,
            "min_obstacle_clearance": 0.8,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.2,
            "terminal_formation_error": 0.5,
            "longitudinal_jerk_rms": 0.1,
            "steer_rate_rms": 0.1,
            "mean_step_runtime_ms": 10.0,
            "qp_solve_time_mean_ms": 1.0,
        },
        {
            "scenario": "s_constant",
            "method": "apf",
            "leader_goal_error": 5.0,
            "mean_speed": 3.0,
            "min_obstacle_clearance": 0.8,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "fallback_ratio": 0.2,
            "terminal_formation_error": 0.5,
            "longitudinal_jerk_rms": 0.1,
            "steer_rate_rms": 0.1,
            "mean_step_runtime_ms": 10.0,
            "qp_solve_time_mean_ms": 1.0,
        },
    ]
    comparison_rows = pairwise_compare_to_reference(
        rows,
        reference_method="no_rl",
    )

    goal_error_rows = [row for row in comparison_rows if row["metric"] == "leader_goal_error"]
    assert goal_error_rows
    assert goal_error_rows[0]["comparison_kind"] == "independent"
    assert goal_error_rows[0]["paired_n"] == 0
    assert goal_error_rows[0]["t_statistic"] == 0.0
    assert goal_error_rows[0]["p_value"] == 1.0
    assert np.isnan(float(goal_error_rows[0]["wilcoxon_p_value"]))


def test_pairwise_comparison_reports_paired_seed_statistics() -> None:
    comparison_rows = pairwise_compare_to_reference(
        _raw_rows(),
        reference_method="no_rl",
    )

    goal_error_rows = [row for row in comparison_rows if row["metric"] == "leader_goal_error"]
    assert goal_error_rows
    row = goal_error_rows[0]
    assert row["comparison_kind"] == "paired"
    assert row["paired_n"] == 2
    assert row["mean_delta"] > 0.0
    assert row["delta_ci_low"] <= row["mean_delta"] <= row["delta_ci_high"]


def test_export_paper_artifacts_writes_tables_and_figures(tmp_path: Path) -> None:
    raw_rows = _raw_rows()
    summary_rows = summarize_experiments(raw_rows)
    comparison_rows = pairwise_compare_to_reference(
        raw_rows,
        reference_method="no_rl",
    )

    export_paper_artifacts(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        output_dir=tmp_path,
    )

    assert (tmp_path / "all_runs.csv").exists()
    assert (tmp_path / "tables" / "main_results.csv").exists()
    assert (tmp_path / "tables" / "main_results.tex").exists()
    assert (tmp_path / "figures" / "metric_overview.pdf").exists()
    assert (tmp_path / "figures" / "safety_efficiency_tradeoff.pdf").exists()
    assert (tmp_path / "figures" / "trajectory_overview.pdf").exists()
    assert (tmp_path / "figures" / "risk_clearance_timeseries.pdf").exists()
    assert (tmp_path / "figures" / "qp_correction_timeline.pdf").exists()
    assert (tmp_path / "figures" / "mode_timeline.pdf").exists()
    assert (tmp_path / "figures" / "runtime_histogram.pdf").exists()
    assert (tmp_path / "figures" / "failure_case_panel.pdf").exists()
    pretty_table = (tmp_path / "tables" / "main_results_pretty.csv").read_text(encoding="utf-8")
    assert "+/-" in pretty_table
