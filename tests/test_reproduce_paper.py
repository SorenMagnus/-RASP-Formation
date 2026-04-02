from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    *,
    seed: int,
    config_hash: str,
    leader_goal_error: float,
    collision_count: int = 0,
    boundary_violation_count: int = 0,
) -> dict[str, object]:
    return {
        "seed": seed,
        "config_hash": config_hash,
        "git_commit": "deadbeef",
        "num_steps": 200,
        "sim_time": 20.0,
        "leader_final_x": 25.0 - leader_goal_error,
        "leader_goal_error": leader_goal_error,
        "time_to_goal": 12.0 + 0.5 * seed,
        "time_to_team_goal": 12.0 + 0.5 * seed,
        "mean_speed": 4.0,
        "max_speed": 5.0,
        "leader_path_length": 30.0,
        "leader_path_length_ratio": 1.02,
        "leader_path_efficiency": 0.98,
        "min_ttc": 1.8,
        "min_boundary_margin": 0.7,
        "min_obstacle_clearance": 0.8,
        "collision_count": collision_count,
        "boundary_violation_count": boundary_violation_count,
        "terminal_formation_error": 0.4,
        "terminal_max_team_lag": 0.6,
        "formation_recovered": True,
        "time_to_recover_formation": 2.5,
        "longitudinal_jerk_rms": 0.2,
        "steer_rate_rms": 0.1,
        "accel_saturation_rate": 0.0,
        "steer_saturation_rate": 0.0,
        "mean_safety_correction": 0.0,
        "safety_interventions": 0,
        "slack_mean": 0.0,
        "slack_max": 0.0,
        "max_safety_slack": 0.0,
        "fallback_count": 0,
        "fallback_events": 0,
        "fallback_ratio": 0.0,
        "mean_step_runtime_ms": 10.0,
        "max_step_runtime_ms": 12.0,
        "mean_mode_runtime_ms": 1.0,
        "max_mode_runtime_ms": 1.5,
        "mean_controller_runtime_ms": 2.0,
        "max_controller_runtime_ms": 2.5,
        "mean_safety_runtime_ms": 3.0,
        "max_safety_runtime_ms": 3.5,
        "qp_solve_count": 0,
        "qp_engagement_rate": 0.0,
        "qp_solve_time_mean_ms": 0.0,
        "qp_solve_time_max_ms": 0.0,
        "qp_iteration_mean": 0.0,
        "qp_iteration_max": 0,
        "reached_goal": True,
        "team_goal_reached": True,
    }


def _expected_cell(
    *,
    exp_id: str,
    scenario: str,
    method: str,
    variant_type: str = "method",
    variant_name: str | None = None,
) -> dict[str, object]:
    resolved_variant_name = method if variant_name is None else variant_name
    if variant_type == "ablation":
        run_id = f"{scenario}__adaptive_apf__{resolved_variant_name}"
        config_name = f"{scenario}__ablation__{resolved_variant_name}.yaml"
    else:
        run_id = f"{scenario}__{method}"
        config_name = f"{scenario}__method__{method}.yaml"
    return {
        "scenario": scenario,
        "method": method,
        "variant_type": variant_type,
        "variant_name": resolved_variant_name,
        "run_id": run_id,
        "config_path": str(Path("outputs") / exp_id / "generated_configs" / config_name),
        "output_dir": str(Path("outputs") / exp_id / "runs" / run_id),
    }


def _write_manifest(
    path: Path,
    *,
    exp_id: str,
    expected_seeds: list[int],
    expected_cells: list[dict[str, object]],
    canonical_matrix: bool = False,
) -> None:
    expected_scenarios = sorted({str(cell["scenario"]) for cell in expected_cells})
    expected_methods = sorted(
        {
            str(cell["method"])
            for cell in expected_cells
            if str(cell["variant_type"]) == "method"
        }
    )
    expected_ablations = sorted(
        {
            str(cell["variant_name"])
            for cell in expected_cells
            if str(cell["variant_type"]) == "ablation"
        }
    )
    payload = {
        "exp_id": exp_id,
        "canonical_matrix": canonical_matrix,
        "primary_method": "no_rl",
        "expected_seed_count": len(expected_seeds),
        "expected_seeds": expected_seeds,
        "expected_scenarios": expected_scenarios,
        "expected_methods": expected_methods,
        "expected_ablations": expected_ablations,
        "expected_cells": expected_cells,
        "observed_row_count": 0,
        "observed_run_count": 0,
        "observed_git_commits": [],
        "matrix_index_path": "matrix_index.csv",
        "paper_acceptance_path": "paper_acceptance.json",
        "all_runs_path": "all_runs.csv",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_runtime_state_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "method",
        "variant_type",
        "variant_name",
        "run_id",
        "config_path",
        "output_dir",
        "expected_seed_count",
        "completed_seed_count",
        "completed_progress",
        "runtime_status",
        "started_at",
        "last_heartbeat",
        "finished_at",
        "heartbeat_age_seconds",
        "stalled",
        "runner_pid",
        "runner_started_at",
        "process_alive",
        "orphaned",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_reproduce_paper_writes_manifest_index_and_acceptance(tmp_path: Path) -> None:
    module = _load_module("reproduce_paper_test", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        def fake_run_single_configuration(*, run_id: str, seeds: list[int], **kwargs) -> Path:
            output_dir = paper_dir / "runs" / run_id
            method_name = run_id.split("__", maxsplit=1)[1]
            rows = [
                _summary_row(
                    seed=seed,
                    config_hash=f"cfg_{method_name}",
                    leader_goal_error=8.0 if method_name == "no_rl" else 9.0,
                )
                for seed in seeds
            ]
            _write_summary_csv(output_dir / "summary.csv", rows)
            return output_dir

        def fake_export(*, output_dir: Path, **kwargs) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

        module._run_single_configuration = fake_run_single_configuration
        module.export_paper_artifacts = fake_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--scenarios",
                "s1_local_minima",
                "--methods",
                "no_rl",
                "apf",
                "--ablations",
                "--seeds",
                "0",
                "1",
            ]
        )
        assert exit_code == 0

        manifest = json.loads((paper_dir / "manifest.json").read_text(encoding="utf-8"))
        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        matrix_index = list(
            csv.DictReader((paper_dir / "matrix_index.csv").open("r", encoding="utf-8"))
        )
        cell_progress = list(
            csv.DictReader((paper_dir / "cell_progress.csv").open("r", encoding="utf-8"))
        )

        assert manifest["primary_method"] == "no_rl"
        assert manifest["expected_seeds"] == [0, 1]
        assert acceptance["bundle_complete"] is True
        assert acceptance["primary_safety_valid"] is True
        assert run_progress["bundle_progress"] == 1.0
        assert run_progress["num_complete_cells"] == 2
        assert len(matrix_index) == 2
        assert len(cell_progress) == 2
        assert all(row["cell_valid"] == "True" for row in matrix_index)
        assert all(row["status"] == "complete" for row in cell_progress)
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_marks_incomplete_bundle(tmp_path: Path) -> None:
    module = _load_module("reproduce_paper_test_incomplete", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        def fake_run_single_configuration(*, run_id: str, seeds: list[int], **kwargs) -> Path:
            output_dir = paper_dir / "runs" / run_id
            method_name = run_id.split("__", maxsplit=1)[1]
            if method_name == "no_rl":
                rows = [_summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0)]
            else:
                rows = [
                    _summary_row(seed=0, config_hash="cfg_apf_a", leader_goal_error=10.0),
                    _summary_row(seed=1, config_hash="cfg_apf_b", leader_goal_error=11.0),
                ]
            _write_summary_csv(output_dir / "summary.csv", rows)
            return output_dir

        def fake_export(*, output_dir: Path, **kwargs) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

        module._run_single_configuration = fake_run_single_configuration
        module.export_paper_artifacts = fake_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--scenarios",
                "s1_local_minima",
                "--methods",
                "no_rl",
                "apf",
                "--ablations",
                "--seeds",
                "0",
                "1",
            ]
        )
        assert exit_code == 0

        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        matrix_index = list(
            csv.DictReader((paper_dir / "matrix_index.csv").open("r", encoding="utf-8"))
        )
        cell_progress = list(
            csv.DictReader((paper_dir / "cell_progress.csv").open("r", encoding="utf-8"))
        )

        assert acceptance["bundle_complete"] is False
        assert run_progress["bundle_progress"] == 0.75
        assert run_progress["num_partial_cells"] == 1
        assert run_progress["num_invalid_cells"] == 1
        assert {cell["method"] for cell in acceptance["missing_cells"]} == {"no_rl"}
        assert {cell["method"] for cell in acceptance["invalid_cells"]} == {"no_rl", "apf"}
        assert any(
            row["method"] == "apf" and row["config_hash_consistent"] == "False"
            for row in matrix_index
        )
        assert any(
            row["method"] == "no_rl" and row["status"] == "partial"
            for row in cell_progress
        )
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_skip_existing_restores_progress_from_disk() -> None:
    module = _load_module("reproduce_paper_test_resume", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id
    no_rl_dir = paper_dir / "runs" / "s1_local_minima__no_rl"

    try:
        _write_summary_csv(
            no_rl_dir / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0),
                _summary_row(seed=1, config_hash="cfg_no_rl", leader_goal_error=8.5),
            ],
        )

        call_counter = {"count": 0}

        def fake_run_batch(*, config, seeds, exp_id):
            call_counter["count"] += 1
            output_dir = repo_root / config.experiment.output_root / exp_id
            rows = [
                _summary_row(seed=seed, config_hash="cfg_apf", leader_goal_error=9.5)
                for seed in seeds
            ]
            _write_summary_csv(output_dir / "summary.csv", rows)
            return output_dir

        def fake_export(*, output_dir: Path, **kwargs) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

        module.run_batch = fake_run_batch
        module.export_paper_artifacts = fake_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--scenarios",
                "s1_local_minima",
                "--methods",
                "no_rl",
                "apf",
                "--ablations",
                "--seeds",
                "0",
                "1",
                "--skip-existing",
            ]
        )
        assert exit_code == 0

        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        cell_progress = list(
            csv.DictReader((paper_dir / "cell_progress.csv").open("r", encoding="utf-8"))
        )

        assert call_counter["count"] == 1
        assert run_progress["bundle_progress"] == 1.0
        assert run_progress["num_complete_cells"] == 2
        assert any(
            row["method"] == "no_rl" and row["status"] == "complete"
            for row in cell_progress
        )
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_status_only_refreshes_bundle_from_disk() -> None:
    module = _load_module("reproduce_paper_test_status_only", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__no_rl" / "summary.csv",
            [_summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0)],
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__apf" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_apf_a", leader_goal_error=10.0),
                _summary_row(seed=1, config_hash="cfg_apf_b", leader_goal_error=10.5),
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("status-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("status-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--status-only",
            ]
        )
        assert exit_code == 0

        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        cell_progress = list(
            csv.DictReader((paper_dir / "cell_progress.csv").open("r", encoding="utf-8"))
        )

        assert run_progress["bundle_progress"] == 0.75
        assert run_progress["remaining_cell_count"] == 2
        assert run_progress["num_expected_cells"] == 2
        assert acceptance["bundle_complete"] is False
        assert any(row["method"] == "no_rl" and row["status"] == "partial" for row in cell_progress)
        assert any(row["method"] == "apf" and row["status"] == "invalid" for row in cell_progress)
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_validate_only_returns_nonzero_for_incomplete_bundle() -> None:
    module = _load_module("reproduce_paper_test_validate_incomplete", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__no_rl" / "summary.csv",
            [_summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0)],
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__apf" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_apf_a", leader_goal_error=10.0),
                _summary_row(seed=1, config_hash="cfg_apf_b", leader_goal_error=10.5),
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("validate-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("validate-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--seeds",
                "0",
                "1",
                "2",
                "--validate-only",
            ]
        )
        assert exit_code == 1

        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        assert acceptance["bundle_complete"] is False
        assert acceptance["primary_safety_valid"] is True
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_validate_only_returns_zero_for_complete_bundle() -> None:
    module = _load_module("reproduce_paper_test_validate_complete", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__no_rl" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0),
                _summary_row(seed=1, config_hash="cfg_no_rl", leader_goal_error=8.5),
            ],
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__apf" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_apf", leader_goal_error=10.0),
                _summary_row(seed=1, config_hash="cfg_apf", leader_goal_error=10.5),
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("validate-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("validate-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--seeds",
                "0",
                "1",
                "2",
                "--validate-only",
            ]
        )
        assert exit_code == 0

        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        assert run_progress["bundle_progress"] == 1.0
        assert acceptance["bundle_complete"] is True
        assert acceptance["primary_safety_valid"] is True
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_audit_requires_manifest_or_canonical_matrix() -> None:
    module = _load_module("reproduce_paper_test_manifest_required", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        def fail_run(*args, **kwargs):
            raise AssertionError("audit bootstrap failure should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("audit bootstrap failure should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        status_exit = module.main(["--exp-id", exp_id, "--status-only"])
        validate_exit = module.main(["--exp-id", exp_id, "--validate-only"])

        assert status_exit == 2
        assert validate_exit == 2
        assert not (paper_dir / "manifest.json").exists()
        assert not (paper_dir / "run_progress.json").exists()
        assert not (paper_dir / "paper_acceptance.json").exists()
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_status_only_is_idempotent_with_manifest() -> None:
    module = _load_module("reproduce_paper_test_status_idempotent", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__no_rl" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_no_rl", leader_goal_error=8.0),
                _summary_row(seed=1, config_hash="cfg_no_rl", leader_goal_error=8.5),
            ],
        )
        _write_summary_csv(
            paper_dir / "runs" / "s1_local_minima__apf" / "summary.csv",
            [
                _summary_row(seed=0, config_hash="cfg_apf", leader_goal_error=10.0),
                _summary_row(seed=1, config_hash="cfg_apf", leader_goal_error=10.5),
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("status-only idempotency should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("status-only idempotency should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        first_exit = module.main(["--exp-id", exp_id, "--status-only"])
        first_snapshot = {
            "manifest": (paper_dir / "manifest.json").read_text(encoding="utf-8"),
            "run_progress": (paper_dir / "run_progress.json").read_text(encoding="utf-8"),
            "cell_progress": (paper_dir / "cell_progress.csv").read_text(encoding="utf-8"),
            "run_runtime_state": (paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"),
            "cell_runtime_state": (paper_dir / "cell_runtime_state.csv").read_text(encoding="utf-8"),
            "matrix_index": (paper_dir / "matrix_index.csv").read_text(encoding="utf-8"),
            "acceptance": (paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"),
        }

        second_exit = module.main(["--exp-id", exp_id, "--status-only"])
        second_snapshot = {
            "manifest": (paper_dir / "manifest.json").read_text(encoding="utf-8"),
            "run_progress": (paper_dir / "run_progress.json").read_text(encoding="utf-8"),
            "cell_progress": (paper_dir / "cell_progress.csv").read_text(encoding="utf-8"),
            "run_runtime_state": (paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"),
            "cell_runtime_state": (paper_dir / "cell_runtime_state.csv").read_text(encoding="utf-8"),
            "matrix_index": (paper_dir / "matrix_index.csv").read_text(encoding="utf-8"),
            "acceptance": (paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"),
        }

        assert first_exit == 0
        assert second_exit == 0
        assert first_snapshot == second_snapshot
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_writes_runtime_state_files_during_run() -> None:
    module = _load_module("reproduce_paper_test_runtime_files", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        observed: dict[str, object] = {}

        def fake_run_single_configuration(*, run_id: str, seeds: list[int], **kwargs) -> Path:
            run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
            cell_runtime = list(
                csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8"))
            )
            observed["run_runtime"] = run_runtime
            observed["cell_runtime"] = cell_runtime
            output_dir = paper_dir / "runs" / run_id
            rows = [
                _summary_row(seed=seed, config_hash="cfg_no_rl", leader_goal_error=8.0 + 0.5 * seed)
                for seed in seeds
            ]
            _write_summary_csv(output_dir / "summary.csv", rows)
            return output_dir

        def fake_export(*, output_dir: Path, **kwargs) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

        module._run_single_configuration = fake_run_single_configuration
        module.export_paper_artifacts = fake_export

        exit_code = module.main(
            [
                "--exp-id",
                exp_id,
                "--scenarios",
                "s1_local_minima",
                "--methods",
                "no_rl",
                "--ablations",
                "--seeds",
                "0",
                "1",
            ]
        )

        assert exit_code == 0
        run_runtime = observed["run_runtime"]
        cell_runtime = observed["cell_runtime"]
        assert run_runtime["running_cell_count"] == 1
        assert run_runtime["num_running_cells"] == 1
        assert run_runtime["num_complete_cells"] == 0
        assert sum(1 for row in cell_runtime if row["runtime_status"] == "running") == 1
        active_row = next(row for row in cell_runtime if row["runtime_status"] == "running")
        assert active_row["run_id"] == "s1_local_minima__no_rl"

        final_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        assert final_runtime["running_cell_count"] == 0
        assert final_runtime["num_complete_cells"] == 1
        assert final_runtime["bundle_completed_progress"] == 1.0
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_status_only_reports_stalled_running_cell() -> None:
    module = _load_module("reproduce_paper_test_runtime_stalled", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [
                {
                    **expected_cells[0],
                    "expected_seed_count": 2,
                    "completed_seed_count": 0,
                    "completed_progress": 0.0,
                    "runtime_status": "running",
                    "started_at": "2026-04-01T00:00:00Z",
                    "last_heartbeat": "2026-04-01T00:00:00Z",
                    "finished_at": "",
                    "heartbeat_age_seconds": 0,
                    "stalled": False,
                },
                {
                    **expected_cells[1],
                    "expected_seed_count": 2,
                    "completed_seed_count": 0,
                    "completed_progress": 0.0,
                    "runtime_status": "pending",
                    "started_at": "",
                    "last_heartbeat": "",
                    "finished_at": "",
                    "heartbeat_age_seconds": "",
                    "stalled": False,
                },
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("status-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("status-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = module.main(["--exp-id", exp_id, "--status-only"])

        assert exit_code == 0
        rendered = stdout.getvalue()
        assert "running_cell_count=1" in rendered
        assert "active_cell_run_id=s1_local_minima__no_rl" in rendered
        assert "stalled=True" in rendered

        run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        assert run_runtime["running_cell_count"] == 1
        assert run_runtime["num_pending_cells"] == 1
        assert run_runtime["num_running_cells"] == 1
        assert run_runtime["num_complete_cells"] == 0
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_status_only_enforces_single_running_cell() -> None:
    module = _load_module("reproduce_paper_test_runtime_single_running", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="apf"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [
                {
                    **expected_cells[0],
                    "expected_seed_count": 2,
                    "completed_seed_count": 0,
                    "completed_progress": 0.0,
                    "runtime_status": "running",
                    "started_at": "2026-04-02T00:00:00Z",
                    "last_heartbeat": "2026-04-02T00:00:00Z",
                    "finished_at": "",
                    "heartbeat_age_seconds": 0,
                    "stalled": False,
                },
                {
                    **expected_cells[1],
                    "expected_seed_count": 2,
                    "completed_seed_count": 0,
                    "completed_progress": 0.0,
                    "runtime_status": "running",
                    "started_at": "2026-04-02T00:01:00Z",
                    "last_heartbeat": "2026-04-02T00:01:00Z",
                    "finished_at": "",
                    "heartbeat_age_seconds": 0,
                    "stalled": False,
                },
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("status-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("status-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(["--exp-id", exp_id, "--status-only"])

        assert exit_code == 0
        run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        cell_runtime = list(
            csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8"))
        )
        assert run_runtime["running_cell_count"] == 1
        assert sum(1 for row in cell_runtime if row["runtime_status"] == "running") == 1
        assert any(
            row["run_id"] == "s1_local_minima__apf" and row["runtime_status"] == "running"
            for row in cell_runtime
        )
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_reproduce_paper_validate_only_keeps_acceptance_independent_of_runtime_state() -> None:
    module = _load_module("reproduce_paper_test_runtime_acceptance", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [
                {
                    **expected_cells[0],
                    "expected_seed_count": 2,
                    "completed_seed_count": 0,
                    "completed_progress": 0.0,
                    "runtime_status": "running",
                    "started_at": "2026-04-01T00:00:00Z",
                    "last_heartbeat": "2026-04-01T00:00:00Z",
                    "finished_at": "",
                    "heartbeat_age_seconds": 0,
                    "stalled": False,
                },
            ],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("validate-only should not launch experiment runs")

        def fail_export(*args, **kwargs):
            raise AssertionError("validate-only should not export paper artifacts")

        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(["--exp-id", exp_id, "--validate-only"])

        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        run_progress = json.loads((paper_dir / "run_progress.json").read_text(encoding="utf-8"))
        run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))

        assert exit_code == 1
        assert acceptance["bundle_complete"] is False
        assert run_progress["bundle_progress"] == 0.0
        assert run_runtime["running_cell_count"] == 1
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_runtime_process_fields_present_in_artifacts() -> None:
    """runtime process field generation: run_runtime_state.json and cell_runtime_state.csv
    must contain runner_pid, runner_started_at, process_alive, orphaned."""
    module = _load_module("reproduce_paper_test_process_fields", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        def fake_run_single_configuration(*, run_id: str, seeds: list[int], **kwargs) -> Path:
            output_dir = paper_dir / "runs" / run_id
            rows = [
                _summary_row(seed=seed, config_hash="cfg_no_rl", leader_goal_error=8.0)
                for seed in seeds
            ]
            _write_summary_csv(output_dir / "summary.csv", rows)
            return output_dir

        def fake_export(*, output_dir: Path, **kwargs) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

        module._run_single_configuration = fake_run_single_configuration
        module.export_paper_artifacts = fake_export

        exit_code = module.main(
            [
                "--exp-id", exp_id,
                "--scenarios", "s1_local_minima",
                "--methods", "no_rl",
                "--ablations",
                "--seeds", "0",
            ]
        )
        assert exit_code == 0

        run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        cell_runtime = list(
            csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8"))
        )

        assert len(cell_runtime) == 1
        row = cell_runtime[0]
        required_fields = ["runner_pid", "runner_started_at", "process_alive", "orphaned"]
        for field in required_fields:
            assert field in row, f"Missing field '{field}' in cell_runtime_state.csv"

        active_cell = run_runtime.get("active_cell")
        if active_cell is not None:
            for field in required_fields:
                assert field in active_cell, f"Missing field '{field}' in run_runtime_state.json active_cell"
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_liveness_own_process_alive() -> None:
    """liveness test: when PID exists and start time matches, process_alive = true."""
    module = _load_module("reproduce_paper_test_liveness_alive", "scripts/reproduce_paper.py")
    my_pid = os.getpid()
    my_start = module._get_process_start_time(my_pid)
    assert module._check_process_alive(my_pid, my_start) is True


def test_liveness_dead_process_not_alive() -> None:
    """liveness test: when PID does not exist, process_alive = false."""
    module = _load_module("reproduce_paper_test_liveness_dead", "scripts/reproduce_paper.py")
    assert module._check_process_alive(99999999, None) is False
    assert module._check_process_alive(None, None) is False
    assert module._check_process_alive(0, None) is False


def test_liveness_detached_child_process_alive() -> None:
    """liveness test: a detached child process should still be recognized as alive.

    This guards the Windows-specific case where ``os.kill(pid, 0)`` can fail for a
    live child created in a new process group.
    """
    module = _load_module("reproduce_paper_test_liveness_detached", "scripts/reproduce_paper.py")
    creationflags = 0
    creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        creationflags=creationflags,
    )
    try:
        time.sleep(0.3)
        child_start = module._get_process_start_time(child.pid)
        assert module._check_process_alive(child.pid, child_start) is True
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def test_orphan_detection_when_running_but_dead() -> None:
    """orphan detection: if runtime_status=running and process_alive=false then orphaned=true."""
    module = _load_module("reproduce_paper_test_orphan_detection", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [{
                **expected_cells[0],
                "expected_seed_count": 2,
                "completed_seed_count": 0,
                "completed_progress": 0.0,
                "runtime_status": "running",
                "started_at": "2026-04-01T00:00:00Z",
                "last_heartbeat": "2026-04-01T00:00:00Z",
                "finished_at": "",
                "heartbeat_age_seconds": 0,
                "stalled": False,
                "runner_pid": 99999999,
                "runner_started_at": "2026-04-01T00:00:00Z",
                "process_alive": True,
                "orphaned": False,
            }],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("status-only should not launch runs")
        def fail_export(*args, **kwargs):
            raise AssertionError("status-only should not export")
        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = module.main(["--exp-id", exp_id, "--status-only"])

        assert exit_code == 0
        rendered = stdout.getvalue()
        assert "orphaned=True" in rendered
        assert "process_alive=False" in rendered

        cell_runtime = list(
            csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8"))
        )
        running_row = next(r for r in cell_runtime if r["runtime_status"] == "running")
        assert running_row["orphaned"] == "True"
        assert running_row["process_alive"] == "False"
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_stall_and_orphan_are_independent_signals() -> None:
    """stall/orphan decoupling: stalled and orphaned must not be treated as the same signal.
    A cell can be stalled but not orphaned (if pid is alive but heartbeat expired),
    or orphaned but not stalled (if pid is dead but heartbeat is recent)."""
    module = _load_module("reproduce_paper_test_stall_orphan_decoupling", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        # Case: dead pid + stale heartbeat => stalled=True AND orphaned=True
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [{
                **expected_cells[0],
                "expected_seed_count": 2,
                "completed_seed_count": 0,
                "completed_progress": 0.0,
                "runtime_status": "running",
                "started_at": "2026-01-01T00:00:00Z",
                "last_heartbeat": "2026-01-01T00:00:00Z",
                "finished_at": "",
                "heartbeat_age_seconds": 0,
                "stalled": False,
                "runner_pid": 99999999,
                "runner_started_at": "2026-01-01T00:00:00Z",
                "process_alive": True,
                "orphaned": False,
            }],
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("should not run")
        def fail_export(*args, **kwargs):
            raise AssertionError("should not export")
        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            module.main(["--exp-id", exp_id, "--status-only"])

        rendered = stdout.getvalue()
        # Both signals should be true (dead pid + stale heartbeat)
        assert "stalled=True" in rendered
        assert "orphaned=True" in rendered
        # Now test case: alive pid + recent heartbeat => stalled=False, orphaned=False
        my_pid = os.getpid()
        my_start = module._get_process_start_time(my_pid)
        now = module._utc_now()
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [{
                **expected_cells[0],
                "expected_seed_count": 2,
                "completed_seed_count": 0,
                "completed_progress": 0.0,
                "runtime_status": "running",
                "started_at": module._format_timestamp(now),
                "last_heartbeat": module._format_timestamp(now),
                "finished_at": "",
                "heartbeat_age_seconds": 0,
                "stalled": False,
                "runner_pid": my_pid,
                "runner_started_at": module._format_timestamp(my_start) if my_start else module._format_timestamp(now),
                "process_alive": True,
                "orphaned": False,
            }],
        )

        stdout2 = io.StringIO()
        with contextlib.redirect_stdout(stdout2):
            module.main(["--exp-id", exp_id, "--status-only"])

        rendered2 = stdout2.getvalue()
        assert "stalled=False" in rendered2
        assert "orphaned=False" in rendered2
        assert "process_alive=True" in rendered2
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)


def test_acceptance_invariance_with_process_liveness_fields() -> None:
    """acceptance invariance: even if runtime says running, process_alive=true,
    orphaned=false, if summary.csv is incomplete, validate-only must still fail."""
    module = _load_module("reproduce_paper_test_acceptance_invariance_liveness", "scripts/reproduce_paper.py")
    repo_root = Path(__file__).resolve().parents[1]
    exp_id = f"test_paper_reproduce_{uuid.uuid4().hex}"
    paper_dir = repo_root / "outputs" / exp_id

    try:
        expected_cells = [
            _expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl"),
        ]
        _write_manifest(
            paper_dir / "manifest.json",
            exp_id=exp_id,
            expected_seeds=[0, 1],
            expected_cells=expected_cells,
        )
        my_pid = os.getpid()
        now = module._utc_now()
        _write_runtime_state_csv(
            paper_dir / "cell_runtime_state.csv",
            [{
                **expected_cells[0],
                "expected_seed_count": 2,
                "completed_seed_count": 0,
                "completed_progress": 0.0,
                "runtime_status": "running",
                "started_at": module._format_timestamp(now),
                "last_heartbeat": module._format_timestamp(now),
                "finished_at": "",
                "heartbeat_age_seconds": 0,
                "stalled": False,
                "runner_pid": my_pid,
                "runner_started_at": module._format_timestamp(now),
                "process_alive": True,
                "orphaned": False,
            }],
        )
        # No summary.csv at all

        def fail_run(*args, **kwargs):
            raise AssertionError("validate-only should not launch runs")
        def fail_export(*args, **kwargs):
            raise AssertionError("validate-only should not export")
        module._run_single_configuration = fail_run
        module.export_paper_artifacts = fail_export

        exit_code = module.main(["--exp-id", exp_id, "--validate-only"])
        assert exit_code == 1  # Must fail because no summary.csv

        acceptance = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        assert acceptance["bundle_complete"] is False

        run_runtime = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        assert run_runtime["running_cell_count"] == 1
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)
