"""Statistical aggregation helpers for reproducible paper experiments."""

from __future__ import annotations

import csv
import math
import warnings
from collections import Counter
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

    return aggregate_metric_with_ci(
        values,
        confidence=confidence,
        ci_method="bootstrap",
    )


def aggregate_metric_with_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    ci_method: str = "bootstrap",
    num_resamples: int = 2000,
    seed: int = 0,
) -> MetricAggregate:
    """Aggregate repeated-seed metric values with a deterministic CI estimator."""

    if values.size == 0:
        return MetricAggregate(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if values.size == 1:
        return MetricAggregate(1, mean, 0.0, mean, mean)

    std = float(np.std(values, ddof=1))
    if std <= 1e-12:
        return MetricAggregate(int(values.size), mean, 0.0, mean, mean)

    if ci_method == "t":
        sem = std / math.sqrt(values.size)
        interval = scipy_stats.t.interval(
            confidence,
            df=values.size - 1,
            loc=mean,
            scale=sem,
        )
        ci_low, ci_high = (float(interval[0]), float(interval[1]))
        return MetricAggregate(values.size, mean, std, ci_low, ci_high)

    if ci_method == "bootstrap":
        ci_low, ci_high = _bootstrap_mean_ci(
            values,
            confidence=confidence,
            num_resamples=num_resamples,
            seed=seed,
        )
        return MetricAggregate(values.size, mean, std, ci_low, ci_high)

    raise ValueError(f"Unsupported CI method: {ci_method}")


def summarize_experiments(
    rows: list[dict[str, object]],
    *,
    group_keys: tuple[str, ...] = ("scenario", "method"),
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    ci_method: str = "bootstrap",
    num_resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Summarize repeated-seed experiments into one wide row per group."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for group_index, (group, group_rows) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        summary_row = {key: value for key, value in zip(group_keys, group, strict=True)}
        summary_row["n"] = len(group_rows)
        for metric_index, metric in enumerate(metrics):
            aggregate = aggregate_metric_with_ci(
                _metric_values(group_rows, metric),
                ci_method=ci_method,
                num_resamples=num_resamples,
                seed=seed + group_index * 1000 + metric_index,
            )
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


def _canonical_cell_key(
    row: dict[str, object],
    *,
    scenario_key: str,
    method_key: str,
) -> tuple[object, object]:
    return (row[scenario_key], row[method_key])


def _seed_scalar(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _join_values(values: Iterable[object], *, separator: str = ",") -> str:
    rendered = [str(value) for value in values if str(value)]
    return separator.join(rendered)


def summarize_canonical_progress(
    matrix_index_rows: list[dict[str, object]],
    acceptance: dict[str, object],
) -> dict[str, object]:
    """Summarize bundle progress from per-cell validation rows."""

    expected_rows = [row for row in matrix_index_rows if bool(row.get("expected_cell", True))]
    status_counts = Counter(str(row.get("status", "invalid")) for row in expected_rows)
    expected_seed_total = sum(int(row.get("expected_seed_count", 0)) for row in expected_rows)
    observed_seed_total = sum(
        min(
            int(row.get("observed_seed_count", 0)),
            int(row.get("expected_seed_count", 0)),
        )
        for row in expected_rows
    )
    if expected_seed_total <= 0:
        bundle_progress = 1.0
    else:
        bundle_progress = float(observed_seed_total) / float(expected_seed_total)
    primary_observed_row_count = int(acceptance.get("primary_observed_row_count", 0))
    return {
        "primary_method": acceptance.get("primary_method", ""),
        "num_expected_cells": len(expected_rows),
        "num_complete_cells": status_counts["complete"],
        "num_partial_cells": status_counts["partial"],
        "num_missing_cells": status_counts["missing"],
        "num_invalid_cells": status_counts["invalid"],
        "remaining_cell_count": len(expected_rows) - status_counts["complete"],
        "expected_seed_total": expected_seed_total,
        "observed_seed_total": observed_seed_total,
        "bundle_progress": bundle_progress,
        "bundle_complete": bool(acceptance.get("bundle_complete", False)),
        "config_hash_consistent": bool(acceptance.get("config_hash_consistent", False)),
        "paired_alignment_complete": bool(acceptance.get("paired_alignment_complete", False)),
        "primary_safety_valid": (
            acceptance.get("primary_safety_valid")
            if primary_observed_row_count > 0
            else None
        ),
        "primary_observed_row_count": primary_observed_row_count,
        "missing_cells": acceptance.get("missing_cells", []),
        "invalid_cells": acceptance.get("invalid_cells", []),
        "unexpected_cells": acceptance.get("unexpected_cells", []),
    }


def validate_canonical_bundle(
    rows: list[dict[str, object]],
    *,
    expected_cells: list[dict[str, object]],
    expected_seeds: list[int],
    primary_method: str,
    scenario_key: str = "scenario",
    method_key: str = "method",
    variant_type_key: str = "variant_type",
    variant_name_key: str = "variant_name",
    seed_key: str = "seed",
    config_hash_key: str = "config_hash",
    run_id_key: str = "run_id",
    config_path_key: str = "config_path",
    output_dir_key: str = "output_dir",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate canonical matrix completeness and return CSV/JSON-friendly reports."""

    expected_seed_values = sorted({_seed_scalar(seed) for seed in expected_seeds if _seed_scalar(seed) is not None})
    expected_seed_set = set(expected_seed_values)
    expected_seed_count = len(expected_seed_values)
    expected_by_key = {
        _canonical_cell_key(cell, scenario_key=scenario_key, method_key=method_key): cell
        for cell in expected_cells
    }

    rows_by_cell: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        rows_by_cell.setdefault(
            _canonical_cell_key(row, scenario_key=scenario_key, method_key=method_key),
            [],
        ).append(row)

    reference_seed_map: dict[object, set[int]] = {}
    for scenario in {cell[scenario_key] for cell in expected_cells}:
        reference_rows = rows_by_cell.get((scenario, primary_method), [])
        reference_seed_map[scenario] = {
            seed
            for seed in (_seed_scalar(row.get(seed_key)) for row in reference_rows)
            if seed is not None
        }

    def _cell_report(
        *,
        cell: dict[str, object],
        cell_rows: list[dict[str, object]],
        expected_cell: bool,
    ) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
        scenario = cell[scenario_key]
        method = cell[method_key]
        observed_seed_list = [
            seed
            for seed in (_seed_scalar(row.get(seed_key)) for row in cell_rows)
            if seed is not None
        ]
        observed_seed_counter = Counter(observed_seed_list)
        observed_seed_values = sorted(observed_seed_counter)
        observed_seed_set = set(observed_seed_values)
        missing_seed_values = sorted(expected_seed_set - observed_seed_set) if expected_cell else []
        unexpected_seed_values = sorted(observed_seed_set - expected_seed_set)
        duplicate_seed_values = sorted(
            seed for seed, count in observed_seed_counter.items() if count > 1
        )
        config_hash_values = sorted(
            {
                str(row[config_hash_key]).strip()
                for row in cell_rows
                if config_hash_key in row and str(row[config_hash_key]).strip()
            }
        )
        config_hash_consistent = len(config_hash_values) == 1
        paired_seed_values = sorted(
            observed_seed_set.intersection(reference_seed_map.get(scenario, set()))
        )
        paired_seed_complete = method == primary_method or (
            len(paired_seed_values) == expected_seed_count
        )
        cell_complete = (
            expected_cell
            and not missing_seed_values
            and not unexpected_seed_values
            and not duplicate_seed_values
            and bool(cell_rows)
        )

        validation_errors: list[str] = []
        if not expected_cell:
            validation_errors.append("unexpected_cell")
        if not cell_rows:
            validation_errors.append("missing_cell")
        if missing_seed_values:
            validation_errors.append("missing_seeds")
        if unexpected_seed_values:
            validation_errors.append("unexpected_seeds")
        if duplicate_seed_values:
            validation_errors.append("duplicate_seeds")
        if cell_rows and not config_hash_consistent:
            validation_errors.append("config_hash_mismatch")
        if cell_rows and method != primary_method and not paired_seed_complete:
            validation_errors.append("paired_alignment_incomplete")

        invalid_status_errors = {
            "unexpected_cell",
            "unexpected_seeds",
            "duplicate_seeds",
            "config_hash_mismatch",
        }
        observed_seed_count = len(observed_seed_values)
        if any(error in invalid_status_errors for error in validation_errors):
            status = "invalid"
        elif observed_seed_count == 0:
            status = "missing"
        elif cell_complete:
            status = "complete"
        else:
            status = "partial"
        cell_valid = expected_cell and not validation_errors
        paired_seed_coverage = (
            float(len(paired_seed_values)) / float(expected_seed_count)
            if expected_seed_count
            else 1.0
        )
        progress_ratio = (
            min(observed_seed_count, expected_seed_count) / float(expected_seed_count)
            if expected_seed_count
            else 1.0
        )
        row = {
            "scenario": scenario,
            "method": method,
            "variant_type": cell.get(variant_type_key, "unknown"),
            "variant_name": cell.get(variant_name_key, method),
            "expected_cell": expected_cell,
            "run_id": _join_values(
                sorted({row.get(run_id_key, "") for row in cell_rows}),
                separator=";",
            ),
            "config_path": _join_values(
                sorted({row.get(config_path_key, "") for row in cell_rows}),
                separator=";",
            ),
            "output_dir": _join_values(
                sorted({row.get(output_dir_key, "") for row in cell_rows}),
                separator=";",
            ),
            "row_count": len(cell_rows),
            "expected_seed_count": expected_seed_count,
            "observed_seed_count": observed_seed_count,
            "actual_seed_count": observed_seed_count,
            "paired_seed_count": len(paired_seed_values),
            "paired_seed_coverage": paired_seed_coverage,
            "paired_seed_complete": paired_seed_complete,
            "missing_seed_count": len(missing_seed_values),
            "missing_seeds": _join_values(missing_seed_values),
            "unexpected_seed_count": len(unexpected_seed_values),
            "unexpected_seeds": _join_values(unexpected_seed_values),
            "duplicate_seed_count": len(duplicate_seed_values),
            "duplicate_seeds": _join_values(duplicate_seed_values),
            "config_hash_consistent": config_hash_consistent,
            "config_hash_count": len(config_hash_values),
            "config_hash_values": _join_values(config_hash_values, separator=";"),
            "progress_ratio": progress_ratio,
            "status": status,
            "cell_complete": cell_complete,
            "complete": cell_complete,
            "cell_valid": cell_valid,
            "validation_errors": _join_values(validation_errors, separator=";"),
        }

        missing_payload = None
        invalid_payload = None
        if expected_cell and (not cell_complete or "missing_seeds" in validation_errors or "missing_cell" in validation_errors):
            missing_payload = {
                "scenario": scenario,
                "method": method,
                "variant_type": cell.get(variant_type_key, "unknown"),
                "variant_name": cell.get(variant_name_key, method),
                "missing_seeds": missing_seed_values,
                "missing_seed_count": len(missing_seed_values),
            }
        if validation_errors:
            invalid_payload = {
                "scenario": scenario,
                "method": method,
                "variant_type": cell.get(variant_type_key, "unknown"),
                "variant_name": cell.get(variant_name_key, method),
                "errors": validation_errors,
            }
        return row, missing_payload, invalid_payload

    matrix_index_rows: list[dict[str, object]] = []
    missing_cells: list[dict[str, object]] = []
    invalid_cells: list[dict[str, object]] = []
    unexpected_cells: list[dict[str, object]] = []

    for key in sorted(expected_by_key):
        row, missing_payload, invalid_payload = _cell_report(
            cell=expected_by_key[key],
            cell_rows=rows_by_cell.get(key, []),
            expected_cell=True,
        )
        matrix_index_rows.append(row)
        if missing_payload is not None:
            missing_cells.append(missing_payload)
        if invalid_payload is not None:
            invalid_cells.append(invalid_payload)

    for key in sorted(set(rows_by_cell).difference(expected_by_key)):
        cell_rows = rows_by_cell[key]
        first_row = cell_rows[0]
        unexpected_cell = {
            scenario_key: first_row[scenario_key],
            method_key: first_row[method_key],
            variant_type_key: first_row.get(variant_type_key, "unknown"),
            variant_name_key: first_row.get(variant_name_key, first_row[method_key]),
        }
        row, _, invalid_payload = _cell_report(
            cell=unexpected_cell,
            cell_rows=cell_rows,
            expected_cell=False,
        )
        matrix_index_rows.append(row)
        unexpected_cells.append(
            {
                "scenario": unexpected_cell[scenario_key],
                "method": unexpected_cell[method_key],
                "variant_type": unexpected_cell[variant_type_key],
                "variant_name": unexpected_cell[variant_name_key],
            }
        )
        if invalid_payload is not None:
            invalid_cells.append(invalid_payload)

    primary_rows = [row for row in rows if row.get(method_key) == primary_method]
    primary_observed_row_count = len(primary_rows)
    primary_safety_valid = bool(primary_rows) and all(
        float(row.get("collision_count", float("nan"))) <= 0.0
        and float(row.get("boundary_violation_count", float("nan"))) <= 0.0
        for row in primary_rows
        if isinstance(row.get("collision_count"), (int, float, bool))
        and isinstance(row.get("boundary_violation_count"), (int, float, bool))
    )
    config_hash_consistent = all(bool(row["config_hash_consistent"]) for row in matrix_index_rows)
    paired_alignment_complete = all(bool(row["paired_seed_complete"]) for row in matrix_index_rows)
    bundle_complete = (
        not missing_cells
        and not invalid_cells
        and not unexpected_cells
        and config_hash_consistent
        and paired_alignment_complete
    )
    acceptance = {
        "primary_method": primary_method,
        "expected_seed_count": expected_seed_count,
        "expected_seeds": expected_seed_values,
        "expected_cell_count": len(expected_by_key),
        "num_expected_cells": len(expected_by_key),
        "observed_cell_count": len(rows_by_cell),
        "valid_cell_count": sum(1 for row in matrix_index_rows if bool(row["cell_valid"])),
        "complete_cell_count": sum(1 for row in matrix_index_rows if bool(row["cell_complete"])),
        "num_complete_cells": sum(1 for row in matrix_index_rows if bool(row["cell_complete"])),
        "config_hash_consistent": config_hash_consistent,
        "paired_alignment_complete": paired_alignment_complete,
        "bundle_complete": bundle_complete,
        "primary_safety_valid": primary_safety_valid,
        "primary_observed_row_count": primary_observed_row_count,
        "missing_cells": missing_cells,
        "invalid_cells": invalid_cells,
        "unexpected_cells": unexpected_cells,
    }
    return matrix_index_rows, acceptance
