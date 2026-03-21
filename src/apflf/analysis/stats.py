"""Statistical aggregation helpers for reproducible paper experiments."""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats as scipy_stats


DEFAULT_METRICS = (
    "leader_goal_error",
    "time_to_goal",
    "mean_speed",
    "leader_path_length_ratio",
    "min_ttc",
    "min_obstacle_clearance",
    "collision_count",
    "boundary_violation_count",
    "fallback_ratio",
    "terminal_formation_error",
)


@dataclass(frozen=True)
class MetricAggregate:
    """Summary statistics for one metric over repeated seeds."""

    count: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def _coerce_csv_value(value: str) -> object:
    """Parse CSV values back into stable Python scalar types."""

    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        parsed_int = int(value)
    except ValueError:
        pass
    else:
        if "." not in value and "e" not in lowered:
            return parsed_int
    try:
        return float(value)
    except ValueError:
        return value


def read_summary_csv(path: Path) -> list[dict[str, object]]:
    """Read an experiment summary CSV into typed row dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce_csv_value(value) for key, value in row.items()} for row in reader]


def write_csv_rows(rows: Iterable[dict[str, object]], path: Path) -> None:
    """Write rows to CSV while preserving column order."""

    rows = list(rows)
    if not rows:
        raise ValueError("At least one row is required to write a CSV file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_values(rows: list[dict[str, object]], metric: str) -> np.ndarray:
    values = [float(row[metric]) for row in rows if metric in row and isinstance(row[metric], (int, float, bool))]
    if not values:
        return np.zeros(0, dtype=float)
    return np.asarray(values, dtype=float)


def aggregate_metric(values: np.ndarray, confidence: float = 0.95) -> MetricAggregate:
    """Aggregate repeated-seed metric values into mean/std/confidence interval."""

    if values.size == 0:
        return MetricAggregate(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if values.size == 1:
        return MetricAggregate(1, mean, 0.0, mean, mean)

    std = float(np.std(values, ddof=1))
    if std <= 1e-12:
        return MetricAggregate(int(values.size), mean, 0.0, mean, mean)
    sem = std / math.sqrt(values.size)
    interval = scipy_stats.t.interval(
        confidence,
        df=values.size - 1,
        loc=mean,
        scale=sem,
    )
    ci_low, ci_high = (float(interval[0]), float(interval[1]))
    return MetricAggregate(values.size, mean, std, ci_low, ci_high)


def summarize_experiments(
    rows: list[dict[str, object]],
    *,
    group_keys: tuple[str, ...] = ("scenario", "method"),
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> list[dict[str, object]]:
    """Summarize repeated-seed experiments into one wide row per group."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary_row = {key: value for key, value in zip(group_keys, group, strict=True)}
        summary_row["n"] = len(group_rows)
        for metric in metrics:
            aggregate = aggregate_metric(_metric_values(group_rows, metric))
            summary_row[f"{metric}_mean"] = aggregate.mean
            summary_row[f"{metric}_std"] = aggregate.std
            summary_row[f"{metric}_ci_low"] = aggregate.ci_low
            summary_row[f"{metric}_ci_high"] = aggregate.ci_high
        summary_rows.append(summary_row)
    return summary_rows


def _cohen_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute the pooled-variance Cohen's d effect size."""

    if sample_a.size < 2 or sample_b.size < 2:
        return float("nan")
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled_denominator = (sample_a.size + sample_b.size - 2)
    if pooled_denominator <= 0:
        return float("nan")
    pooled_std = math.sqrt(
        ((sample_a.size - 1) * var_a + (sample_b.size - 1) * var_b) / pooled_denominator
    )
    if pooled_std <= 1e-12:
        return 0.0
    return float((np.mean(sample_a) - np.mean(sample_b)) / pooled_std)


def _paired_effect_size(differences: np.ndarray) -> float:
    """Compute standardized paired effect size from seed-aligned differences."""

    if differences.size < 2:
        return float("nan")
    std = float(np.std(differences, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(differences) / std)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic bootstrap confidence interval for the sample mean."""

    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or float(np.std(values)) <= 1e-12:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(num_resamples, values.size))
    sample_means = np.mean(values[sample_indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return (
        float(np.quantile(sample_means, alpha)),
        float(np.quantile(sample_means, 1.0 - alpha)),
    )


def _paired_metric_values(
    reference_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    metric: str,
    *,
    pair_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference_by_key = {
        row[pair_key]: float(row[metric])
        for row in reference_rows
        if pair_key in row and metric in row and isinstance(row[metric], (int, float, bool))
    }
    method_by_key = {
        row[pair_key]: float(row[metric])
        for row in method_rows
        if pair_key in row and metric in row and isinstance(row[metric], (int, float, bool))
    }
    shared_keys = sorted(set(reference_by_key).intersection(method_by_key))
    if not shared_keys:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return (
        np.asarray([method_by_key[key] for key in shared_keys], dtype=float),
        np.asarray([reference_by_key[key] for key in shared_keys], dtype=float),
    )


def pairwise_compare_to_reference(
    rows: list[dict[str, object]],
    *,
    reference_method: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    scenario_key: str = "scenario",
    method_key: str = "method",
    pair_key: str = "seed",
) -> list[dict[str, object]]:
    """Compare every method against a reference method inside each scenario."""

    rows_by_scenario: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_scenario.setdefault(row[scenario_key], []).append(row)

    comparison_rows: list[dict[str, object]] = []
    for scenario, scenario_rows in sorted(rows_by_scenario.items(), key=lambda item: item[0]):
        reference_rows = [row for row in scenario_rows if row[method_key] == reference_method]
        if not reference_rows:
            continue
        methods = sorted({row[method_key] for row in scenario_rows if row[method_key] != reference_method})
        for method in methods:
            method_rows = [row for row in scenario_rows if row[method_key] == method]
            for metric in metrics:
                reference_values = _metric_values(reference_rows, metric)
                method_values = _metric_values(method_rows, metric)
                if reference_values.size == 0 or method_values.size == 0:
                    continue
                paired_method_values, paired_reference_values = _paired_metric_values(
                    reference_rows,
                    method_rows,
                    metric,
                    pair_key=pair_key,
                )
                comparison_kind = "independent"
                paired_n = 0
                t_stat = float("nan")
                p_value = float("nan")
                wilcoxon_stat = float("nan")
                wilcoxon_p = float("nan")
                delta_ci_low = float("nan")
                delta_ci_high = float("nan")
                effect_size = _cohen_d(method_values, reference_values)
                mean_delta = float(np.mean(method_values) - np.mean(reference_values))

                if paired_method_values.size > 0:
                    paired_n = int(paired_method_values.size)
                    differences = paired_method_values - paired_reference_values
                    mean_delta = float(np.mean(differences))
                    delta_ci_low, delta_ci_high = _bootstrap_mean_ci(differences)
                    effect_size = _paired_effect_size(differences)
                    comparison_kind = "paired"
                    if paired_method_values.size == 1:
                        t_stat = 0.0 if abs(mean_delta) <= 1e-12 else math.copysign(float("inf"), mean_delta)
                        p_value = 1.0 if abs(mean_delta) <= 1e-12 else 0.0
                        wilcoxon_stat = t_stat
                        wilcoxon_p = p_value
                    else:
                        if np.all(np.abs(differences) <= 1e-12):
                            t_stat = 0.0
                            p_value = 1.0
                            wilcoxon_stat = 0.0
                            wilcoxon_p = 1.0
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                test = scipy_stats.ttest_rel(
                                    paired_method_values,
                                    paired_reference_values,
                                )
                            t_stat = float(test.statistic)
                            p_value = float(test.pvalue)
                            try:
                                wilcoxon = scipy_stats.wilcoxon(differences)
                            except ValueError:
                                wilcoxon_stat = float("nan")
                                wilcoxon_p = float("nan")
                            else:
                                wilcoxon_stat = float(wilcoxon.statistic)
                                wilcoxon_p = float(wilcoxon.pvalue)
                elif reference_values.size >= 2 and method_values.size >= 2:
                    reference_std = float(np.std(reference_values, ddof=1))
                    method_std = float(np.std(method_values, ddof=1))
                    delta_ci_low, delta_ci_high = _bootstrap_mean_ci(
                        method_values - np.mean(reference_values),
                    )
                    if reference_std <= 1e-12 and method_std <= 1e-12:
                        if abs(mean_delta) <= 1e-12:
                            t_stat = 0.0
                            p_value = 1.0
                        else:
                            t_stat = math.copysign(float("inf"), mean_delta)
                            p_value = 0.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            test = scipy_stats.ttest_ind(
                                method_values,
                                reference_values,
                                equal_var=False,
                            )
                        t_stat = float(test.statistic)
                        p_value = float(test.pvalue)
                comparison_rows.append(
                    {
                        scenario_key: scenario,
                        method_key: method,
                        "reference_method": reference_method,
                        "metric": metric,
                        "comparison_kind": comparison_kind,
                        "paired_n": paired_n,
                        "reference_mean": float(np.mean(reference_values)),
                        "method_mean": float(np.mean(method_values)),
                        "mean_delta": mean_delta,
                        "delta_ci_low": delta_ci_low,
                        "delta_ci_high": delta_ci_high,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "wilcoxon_statistic": wilcoxon_stat,
                        "wilcoxon_p_value": wilcoxon_p,
                        "cohen_d": effect_size,
                    }
                )
    return comparison_rows
