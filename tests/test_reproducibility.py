"""Reproducibility tests."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from apflf.sim.runner import run_batch
from apflf.utils.config import load_config

NONDETERMINISTIC_SUMMARY_FIELDS = {
    "mean_step_runtime_ms",
    "max_step_runtime_ms",
    "mean_mode_runtime_ms",
    "max_mode_runtime_ms",
    "mean_controller_runtime_ms",
    "max_controller_runtime_ms",
    "mean_safety_runtime_ms",
    "max_safety_runtime_ms",
    "qp_solve_time_mean_ms",
    "qp_solve_time_max_ms",
}
NONDETERMINISTIC_TRAJECTORY_KEYS = {
    "step_runtimes",
    "mode_runtimes",
    "controller_runtimes",
    "safety_runtimes",
    "qp_solve_times",
}


def _read_summary(path: Path) -> list[dict[str, str]]:
    """Read summary.csv rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strip_nondeterministic_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in NONDETERMINISTIC_SUMMARY_FIELDS
        }
        for row in rows
    ]


def test_same_seed_produces_identical_summary_and_trajectory(tmp_path: Path) -> None:
    """The same seed should reproduce the same behavior and deterministic outputs."""

    default_config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(default_config_path)
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            output_root=str(tmp_path),
            save_traj=True,
        ),
    )

    output_a = run_batch(config=config, seeds=[7], exp_id="run_a")
    output_b = run_batch(config=config, seeds=[7], exp_id="run_b")

    summary_a = _read_summary(output_a / "summary.csv")
    summary_b = _read_summary(output_b / "summary.csv")
    assert _strip_nondeterministic_fields(summary_a) == _strip_nondeterministic_fields(summary_b)
    for field in NONDETERMINISTIC_SUMMARY_FIELDS:
        value_a = float(summary_a[0][field])
        value_b = float(summary_b[0][field])
        assert np.isfinite(value_a)
        assert np.isfinite(value_b)
        assert value_a >= 0.0
        assert value_b >= 0.0

    traj_a = np.load(output_a / "traj" / "seed_0007.npz")
    traj_b = np.load(output_b / "traj" / "seed_0007.npz")
    for key in traj_a.files:
        if key in NONDETERMINISTIC_TRAJECTORY_KEYS:
            assert traj_a[key].shape == traj_b[key].shape
            assert np.all(np.isfinite(traj_a[key]))
            assert np.all(np.isfinite(traj_b[key]))
            assert np.all(traj_a[key] >= 0.0)
            assert np.all(traj_b[key] >= 0.0)
            if key == "qp_solve_times":
                assert np.array_equal(traj_a[key] > 0.0, traj_b[key] > 0.0)
            continue
        assert np.array_equal(traj_a[key], traj_b[key])
