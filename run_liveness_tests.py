"""Run the 6 new process-liveness tests using a single module load to avoid KeyboardInterrupt."""
import sys
import os
import importlib.util
import traceback
import json
import csv
import io
import contextlib
import shutil
import uuid

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, "src"))
os.chdir(repo_root)

# Load module ONCE
spec = importlib.util.spec_from_file_location(
    "rp", os.path.join(repo_root, "scripts", "reproduce_paper.py")
)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

from pathlib import Path

passed = 0
failed = 0

def mark(name, ok):
    global passed, failed
    if ok:
        print(f"PASS: {name}")
        passed += 1
    else:
        failed += 1

# ── Test 1: liveness_own_process_alive ──
name = "test_liveness_own_process_alive"
try:
    my_pid = os.getpid()
    my_start = rp._get_process_start_time(my_pid)
    assert rp._check_process_alive(my_pid, my_start) is True, "own process should be alive"
    mark(name, True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)

# ── Test 2: liveness_dead_process ──
name = "test_liveness_dead_process_not_alive"
try:
    assert rp._check_process_alive(99999999, None) is False
    assert rp._check_process_alive(None, None) is False
    assert rp._check_process_alive(0, None) is False
    mark(name, True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)


# ── Test helpers ──
def _summary_row(*, seed, config_hash, leader_goal_error, collision_count=0, boundary_violation_count=0):
    return {
        "seed": seed, "config_hash": config_hash, "git_commit": "deadbeef",
        "num_steps": 200, "sim_time": 20.0,
        "leader_final_x": 25.0 - leader_goal_error,
        "leader_goal_error": leader_goal_error,
        "time_to_goal": 12.0 + 0.5 * seed, "time_to_team_goal": 12.0 + 0.5 * seed,
        "mean_speed": 4.0, "max_speed": 5.0,
        "leader_path_length": 30.0, "leader_path_length_ratio": 1.02,
        "leader_path_efficiency": 0.98, "min_ttc": 1.8,
        "min_boundary_margin": 0.7, "min_obstacle_clearance": 0.8,
        "collision_count": collision_count, "boundary_violation_count": boundary_violation_count,
        "terminal_formation_error": 0.4, "terminal_max_team_lag": 0.6,
        "formation_recovered": True, "time_to_recover_formation": 2.5,
        "longitudinal_jerk_rms": 0.2, "steer_rate_rms": 0.1,
        "accel_saturation_rate": 0.0, "steer_saturation_rate": 0.0,
        "mean_safety_correction": 0.0, "safety_interventions": 0,
        "slack_mean": 0.0, "slack_max": 0.0, "max_safety_slack": 0.0,
        "fallback_count": 0, "fallback_events": 0, "fallback_ratio": 0.0,
        "mean_step_runtime_ms": 10.0, "max_step_runtime_ms": 12.0,
        "mean_mode_runtime_ms": 1.0, "max_mode_runtime_ms": 1.5,
        "mean_controller_runtime_ms": 2.0, "max_controller_runtime_ms": 2.5,
        "mean_safety_runtime_ms": 3.0, "max_safety_runtime_ms": 3.5,
        "qp_solve_count": 0, "qp_engagement_rate": 0.0,
        "qp_solve_time_mean_ms": 0.0, "qp_solve_time_max_ms": 0.0,
        "qp_iteration_mean": 0.0, "qp_iteration_max": 0,
        "reached_goal": True, "team_goal_reached": True,
    }

def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _expected_cell(*, exp_id, scenario, method, variant_type="method", variant_name=None):
    vn = method if variant_name is None else variant_name
    if variant_type == "ablation":
        run_id = f"{scenario}__adaptive_apf__{vn}"
        config_name = f"{scenario}__ablation__{vn}.yaml"
    else:
        run_id = f"{scenario}__{method}"
        config_name = f"{scenario}__method__{method}.yaml"
    return {
        "scenario": scenario, "method": method,
        "variant_type": variant_type, "variant_name": vn,
        "run_id": run_id,
        "config_path": str(Path("outputs") / exp_id / "generated_configs" / config_name),
        "output_dir": str(Path("outputs") / exp_id / "runs" / run_id),
    }

def _write_manifest(path, *, exp_id, expected_seeds, expected_cells, canonical_matrix=False):
    scenarios = sorted({str(c["scenario"]) for c in expected_cells})
    methods = sorted({str(c["method"]) for c in expected_cells if str(c["variant_type"]) == "method"})
    ablations = sorted({str(c["variant_name"]) for c in expected_cells if str(c["variant_type"]) == "ablation"})
    payload = {
        "exp_id": exp_id, "canonical_matrix": canonical_matrix, "primary_method": "no_rl",
        "expected_seed_count": len(expected_seeds), "expected_seeds": expected_seeds,
        "expected_scenarios": scenarios, "expected_methods": methods, "expected_ablations": ablations,
        "expected_cells": expected_cells, "observed_row_count": 0, "observed_run_count": 0,
        "observed_git_commits": [], "matrix_index_path": "matrix_index.csv",
        "paper_acceptance_path": "paper_acceptance.json", "all_runs_path": "all_runs.csv",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _write_runtime_csv(path, rows):
    fieldnames = ["scenario", "method", "variant_type", "variant_name", "run_id",
                  "config_path", "output_dir", "expected_seed_count", "completed_seed_count",
                  "completed_progress", "runtime_status", "started_at", "last_heartbeat",
                  "finished_at", "heartbeat_age_seconds", "stalled",
                  "runner_pid", "runner_started_at", "process_alive", "orphaned"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Test 3: runtime_process_fields_present_in_artifacts ──
name = "test_runtime_process_fields_present_in_artifacts"
try:
    exp_id = f"test_pp_{uuid.uuid4().hex[:8]}"
    paper_dir = Path(repo_root) / "outputs" / exp_id

    orig_run = rp._run_single_configuration
    orig_export = rp.export_paper_artifacts
    def fake_run(*, run_id, seeds, **kw):
        d = paper_dir / "runs" / run_id
        _write_csv(d / "summary.csv", [_summary_row(seed=s, config_hash="cfg", leader_goal_error=8.0) for s in seeds])
        return d
    def fake_export(*, output_dir, **kw):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "all_runs.csv").write_text("scenario,method\n", encoding="utf-8")

    rp._run_single_configuration = fake_run
    rp.export_paper_artifacts = fake_export
    try:
        ec = rp.main(["--exp-id", exp_id, "--scenarios", "s1_local_minima", "--methods", "no_rl", "--ablations", "--seeds", "0"])
        assert ec == 0
        cell_rt = list(csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8")))
        for f in ["runner_pid", "runner_started_at", "process_alive", "orphaned"]:
            assert f in cell_rt[0], f"Missing {f}"
        mark(name, True)
    finally:
        rp._run_single_configuration = orig_run
        rp.export_paper_artifacts = orig_export
        shutil.rmtree(paper_dir, ignore_errors=True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)


# ── Test 4: orphan_detection_when_running_but_dead ──
name = "test_orphan_detection_when_running_but_dead"
try:
    exp_id = f"test_pp_{uuid.uuid4().hex[:8]}"
    paper_dir = Path(repo_root) / "outputs" / exp_id
    cells = [_expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl")]
    _write_manifest(paper_dir / "manifest.json", exp_id=exp_id, expected_seeds=[0, 1], expected_cells=cells)
    _write_runtime_csv(paper_dir / "cell_runtime_state.csv", [{
        **cells[0], "expected_seed_count": 2, "completed_seed_count": 0, "completed_progress": 0.0,
        "runtime_status": "running", "started_at": "2026-04-01T00:00:00Z",
        "last_heartbeat": "2026-04-01T00:00:00Z", "finished_at": "", "heartbeat_age_seconds": 0,
        "stalled": False, "runner_pid": 99999999, "runner_started_at": "2026-04-01T00:00:00Z",
        "process_alive": True, "orphaned": False,
    }])

    orig_run = rp._run_single_configuration
    orig_export = rp.export_paper_artifacts
    rp._run_single_configuration = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no run"))
    rp.export_paper_artifacts = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no export"))
    try:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ec = rp.main(["--exp-id", exp_id, "--status-only"])
        assert ec == 0
        rendered = stdout.getvalue()
        assert "orphaned=True" in rendered, f"Expected orphaned=True in output:\n{rendered}"
        assert "process_alive=False" in rendered
        cell_rt = list(csv.DictReader((paper_dir / "cell_runtime_state.csv").open("r", encoding="utf-8")))
        rrow = next(r for r in cell_rt if r["runtime_status"] == "running")
        assert rrow["orphaned"] == "True"
        assert rrow["process_alive"] == "False"
        mark(name, True)
    finally:
        rp._run_single_configuration = orig_run
        rp.export_paper_artifacts = orig_export
        shutil.rmtree(paper_dir, ignore_errors=True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)


# ── Test 5: stall_and_orphan_are_independent_signals ──
name = "test_stall_and_orphan_are_independent_signals"
try:
    exp_id = f"test_pp_{uuid.uuid4().hex[:8]}"
    paper_dir = Path(repo_root) / "outputs" / exp_id
    cells = [_expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl")]
    _write_manifest(paper_dir / "manifest.json", exp_id=exp_id, expected_seeds=[0, 1], expected_cells=cells)

    # Case 1: dead pid + stale heartbeat => both stalled and orphaned
    _write_runtime_csv(paper_dir / "cell_runtime_state.csv", [{
        **cells[0], "expected_seed_count": 2, "completed_seed_count": 0, "completed_progress": 0.0,
        "runtime_status": "running", "started_at": "2026-01-01T00:00:00Z",
        "last_heartbeat": "2026-01-01T00:00:00Z", "finished_at": "", "heartbeat_age_seconds": 0,
        "stalled": False, "runner_pid": 99999999, "runner_started_at": "2026-01-01T00:00:00Z",
        "process_alive": True, "orphaned": False,
    }])

    orig_run = rp._run_single_configuration
    orig_export = rp.export_paper_artifacts
    rp._run_single_configuration = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no"))
    rp.export_paper_artifacts = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no"))
    try:
        stdout1 = io.StringIO()
        with contextlib.redirect_stdout(stdout1):
            rp.main(["--exp-id", exp_id, "--status-only"])
        r1 = stdout1.getvalue()
        assert "stalled=True" in r1, f"Expected stalled=True:\n{r1}"
        assert "orphaned=True" in r1, f"Expected orphaned=True:\n{r1}"

        # Case 2: alive pid + recent heartbeat => neither stalled nor orphaned
        now = rp._utc_now()
        my_pid = os.getpid()
        my_start = rp._get_process_start_time(my_pid)
        _write_runtime_csv(paper_dir / "cell_runtime_state.csv", [{
            **cells[0], "expected_seed_count": 2, "completed_seed_count": 0, "completed_progress": 0.0,
            "runtime_status": "running", "started_at": rp._format_timestamp(now),
            "last_heartbeat": rp._format_timestamp(now), "finished_at": "", "heartbeat_age_seconds": 0,
            "stalled": False, "runner_pid": my_pid,
            "runner_started_at": rp._format_timestamp(my_start) if my_start else rp._format_timestamp(now),
            "process_alive": True, "orphaned": False,
        }])

        stdout2 = io.StringIO()
        with contextlib.redirect_stdout(stdout2):
            rp.main(["--exp-id", exp_id, "--status-only"])
        r2 = stdout2.getvalue()
        assert "stalled=False" in r2, f"Expected stalled=False:\n{r2}"
        assert "orphaned=False" in r2, f"Expected orphaned=False:\n{r2}"
        assert "process_alive=True" in r2
        mark(name, True)
    finally:
        rp._run_single_configuration = orig_run
        rp.export_paper_artifacts = orig_export
        shutil.rmtree(paper_dir, ignore_errors=True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)


# ── Test 6: acceptance_invariance_with_process_liveness_fields ──
name = "test_acceptance_invariance_with_process_liveness_fields"
try:
    exp_id = f"test_pp_{uuid.uuid4().hex[:8]}"
    paper_dir = Path(repo_root) / "outputs" / exp_id
    cells = [_expected_cell(exp_id=exp_id, scenario="s1_local_minima", method="no_rl")]
    _write_manifest(paper_dir / "manifest.json", exp_id=exp_id, expected_seeds=[0, 1], expected_cells=cells)
    now = rp._utc_now()
    my_pid = os.getpid()
    _write_runtime_csv(paper_dir / "cell_runtime_state.csv", [{
        **cells[0], "expected_seed_count": 2, "completed_seed_count": 0, "completed_progress": 0.0,
        "runtime_status": "running", "started_at": rp._format_timestamp(now),
        "last_heartbeat": rp._format_timestamp(now), "finished_at": "", "heartbeat_age_seconds": 0,
        "stalled": False, "runner_pid": my_pid, "runner_started_at": rp._format_timestamp(now),
        "process_alive": True, "orphaned": False,
    }])
    # No summary.csv!

    orig_run = rp._run_single_configuration
    orig_export = rp.export_paper_artifacts
    rp._run_single_configuration = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no"))
    rp.export_paper_artifacts = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no"))
    try:
        ec = rp.main(["--exp-id", exp_id, "--validate-only"])
        assert ec == 1, f"Expected exit code 1, got {ec}"
        acc = json.loads((paper_dir / "paper_acceptance.json").read_text(encoding="utf-8"))
        assert acc["bundle_complete"] is False
        rt = json.loads((paper_dir / "run_runtime_state.json").read_text(encoding="utf-8"))
        assert rt["running_cell_count"] == 1
        mark(name, True)
    finally:
        rp._run_single_configuration = orig_run
        rp.export_paper_artifacts = orig_export
        shutil.rmtree(paper_dir, ignore_errors=True)
except Exception as e:
    print(f"FAIL: {name}: {e}")
    traceback.print_exc()
    mark(name, False)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed out of 6")
sys.exit(0 if failed == 0 else 1)
