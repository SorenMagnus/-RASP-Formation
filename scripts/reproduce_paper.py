"""Run a reproducible experiment matrix and export paper artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.analysis.export import export_paper_artifacts  # noqa: E402
from apflf.analysis.stats import (  # noqa: E402
    DEFAULT_METRICS,
    pairwise_compare_to_reference,
    read_summary_csv,
    summarize_canonical_progress,
    summarize_experiments,
    validate_canonical_bundle,
    write_csv_rows,
)
from apflf.sim.runner import run_batch  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402

PRIMARY_METHOD = "no_rl"
DEFAULT_SCENARIOS = [
    "s1_local_minima",
    "s2_dynamic_crossing",
    "s3_narrow_passage",
    "s4_overtake_interaction",
    "s5_dense_multi_agent",
]
DEFAULT_METHODS = [PRIMARY_METHOD, "apf", "apf_lf", "st_apf", "dwa", "orca"]
CANONICAL_SEEDS = list(range(30))
CANONICAL_SCENARIOS = list(DEFAULT_SCENARIOS)
CANONICAL_METHODS = list(DEFAULT_METHODS)
BASELINE_CONFIGS = {
    "apf": "configs/baselines/apf.yaml",
    "apf_lf": "configs/baselines/apf_lf.yaml",
    "st_apf": "configs/baselines/st_apf.yaml",
    "dwa": "configs/baselines/dwa.yaml",
    "orca": "configs/baselines/orca.yaml",
}
ABLATION_CONFIGS = {
    "no_cbf": "configs/ablations/no_cbf.yaml",
    "no_fsm": "configs/ablations/no_fsm.yaml",
    "no_risk_adaptation": "configs/ablations/no_risk_adaptation.yaml",
    "no_st_terms": "configs/ablations/no_st_terms.yaml",
    "no_escape": "configs/ablations/no_escape.yaml",
}
RL_METHOD_SPECS = {
    "no_rl": {"decision_kind": "fsm", "checkpoint_arg": None},
    "adaptive_apf": {"decision_kind": "fsm", "checkpoint_arg": None},
    "rl_param_only": {"decision_kind": "rl", "checkpoint_arg": "rl_param_checkpoint"},
    "rl_mode_only": {"decision_kind": "rl", "checkpoint_arg": "rl_mode_checkpoint"},
    "rl_full_supervisor": {"decision_kind": "rl", "checkpoint_arg": "rl_full_checkpoint"},
}
RUNTIME_HEARTBEAT_INTERVAL_SECONDS = 30
RUNTIME_STALL_TIMEOUT_SECONDS = 900
RUNTIME_PROCESS_START_TOLERANCE_SECONDS = 5
RUNTIME_STATUS_PENDING = "pending"
RUNTIME_STATUS_RUNNING = "running"
RUNTIME_STATUS_COMPLETE = "complete"
RUNTIME_STATUS_FAILED = "failed"
RUNTIME_STATUS_VALUES = {
    RUNTIME_STATUS_PENDING,
    RUNTIME_STATUS_RUNNING,
    RUNTIME_STATUS_COMPLETE,
    RUNTIME_STATUS_FAILED,
}
_RUNTIME_WRITE_LOCK = threading.Lock()


def _get_process_start_time(pid: int) -> datetime | None:
    """Return the UTC start time of the process with the given PID, or None if not found.

    Uses ctypes on Windows (avoids subprocess timeout issues) or ``ps`` on Unix.
    """
    if pid <= 0:
        return None
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                creation = ctypes.wintypes.FILETIME()
                exit_ft = ctypes.wintypes.FILETIME()
                kernel_ft = ctypes.wintypes.FILETIME()
                user_ft = ctypes.wintypes.FILETIME()
                ok = kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_ft),
                    ctypes.byref(kernel_ft),
                    ctypes.byref(user_ft),
                )
                if not ok:
                    return None
                # FILETIME is 100-ns intervals since 1601-01-01 UTC
                ft_value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                # Epoch delta: 1601-01-01 to 1970-01-01 in 100-ns ticks
                EPOCH_DELTA = 116444736000000000
                timestamp = (ft_value - EPOCH_DELTA) / 1e7
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            finally:
                kernel32.CloseHandle(handle)
        else:
            result = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.strip()
            if not output:
                return None
            return datetime.strptime(output, "%a %b %d %H:%M:%S %Y").replace(
                tzinfo=timezone.utc
            )
    except Exception:
        return None
    return None


def _check_process_alive(
    pid: int | None,
    runner_started_at: datetime | None,
) -> bool:
    """Return True if the PID refers to the same live process that was recorded.

    Uses a tolerance of RUNTIME_PROCESS_START_TOLERANCE_SECONDS to account for
    clock skew between the recording and the OS process table.
    """
    if pid is None or pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.wintypes.DWORD()
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                if not ok:
                    return False
                if exit_code.value != STILL_ACTIVE:
                    return False
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    if runner_started_at is None:
        return True
    actual_start = _get_process_start_time(pid)
    if actual_start is None:
        return True
    delta = abs((actual_start - runner_started_at).total_seconds())
    return delta <= RUNTIME_PROCESS_START_TOLERANCE_SECONDS


def _get_current_process_info() -> tuple[int, datetime]:
    """Return (pid, start_timestamp) for the current Python process."""
    pid = os.getpid()
    start_time = _get_process_start_time(pid)
    if start_time is None:
        start_time = _utc_now()
    return pid, start_time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper reproduction matrix.")
    parser.add_argument("--exp-id", default="paper_reproduce", help="Output directory name under outputs/.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Deterministic seed list.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names without the .yaml suffix.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(ABLATION_CONFIGS),
        help="Ablation names to evaluate on top of the primary method.",
    )
    parser.add_argument(
        "--canonical-matrix",
        action="store_true",
        help="Expand to the paper-scale white-box matrix: S1-S5, 30 seeds, baselines, and all ablations.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-run outputs when the summary.csv already exists.",
    )
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument(
        "--status-only",
        action="store_true",
        help="Refresh canonical bundle ledger files from disk without launching any runs.",
    )
    audit_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the canonical bundle from disk only and return a non-zero exit code if it is incomplete or invalid.",
    )
    parser.add_argument("--rl-param-checkpoint", default=os.getenv("APFLF_RL_PARAM_ONLY_CKPT"))
    parser.add_argument("--rl-mode-checkpoint", default=os.getenv("APFLF_RL_MODE_ONLY_CKPT"))
    parser.add_argument("--rl-full-checkpoint", default=os.getenv("APFLF_RL_FULL_SUPERVISOR_CKPT"))
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use deterministic RL policy evaluation for RL methods.",
    )
    return parser


def _scenario_config_path(repo_root: Path, scenario_name: str) -> Path:
    return repo_root / "configs" / "scenarios" / f"{scenario_name}.yaml"


def _generated_config_path(
    *,
    repo_root: Path,
    generated_dir: Path,
    scenario_name: str,
    variant_name: str,
    variant_type: str,
    paper_exp_id: str,
) -> Path:
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path("outputs") / paper_exp_id / "runs"
    extends_items = [repo_root / "configs" / "scenarios" / f"{scenario_name}.yaml"]
    if variant_type == "method" and variant_name in BASELINE_CONFIGS:
        extends_items.append(repo_root / BASELINE_CONFIGS[variant_name])
    if variant_type == "ablation":
        extends_items.append(repo_root / ABLATION_CONFIGS[variant_name])
    method_label = variant_name if variant_type == "method" else f"adaptive_apf__{variant_name}"
    config_payload = {
        "extends": [Path(os.path.relpath(path, start=generated_dir)).as_posix() for path in extends_items],
        "experiment": {
            "name": f"{scenario_name}_{method_label}",
            "output_root": output_root.as_posix(),
            "save_traj": True,
        },
    }
    config_path = generated_dir / f"{scenario_name}__{variant_type}__{variant_name}.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def _safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _apply_decision_override(
    *,
    config_path: Path,
    decision_kind: str | None,
    rl_checkpoint: str | None,
    deterministic_eval: bool,
):
    config = load_config(config_path)
    if decision_kind is None and rl_checkpoint is None and not deterministic_eval:
        return config
    rl_config = replace(
        config.decision.rl,
        checkpoint_path=config.decision.rl.checkpoint_path if rl_checkpoint is None else rl_checkpoint,
        deterministic_eval=bool(deterministic_eval or config.decision.rl.deterministic_eval),
    )
    return replace(
        config,
        decision=replace(
            config.decision,
            kind=config.decision.kind if decision_kind is None else decision_kind,
            rl=rl_config,
        ),
    )


def _run_single_configuration(
    *,
    config_path: Path,
    seeds: list[int],
    run_id: str,
    skip_existing: bool,
    decision_kind: str | None = None,
    rl_checkpoint: str | None = None,
    deterministic_eval: bool = False,
) -> Path:
    config = _apply_decision_override(
        config_path=config_path,
        decision_kind=decision_kind,
        rl_checkpoint=rl_checkpoint,
        deterministic_eval=deterministic_eval,
    )
    repo_root = Path(__file__).resolve().parents[1]
    expected_output_dir = repo_root / config.experiment.output_root / run_id
    summary_path = expected_output_dir / "summary.csv"
    if skip_existing and summary_path.exists():
        return expected_output_dir
    return run_batch(config=config, seeds=seeds, exp_id=run_id)


def _method_override(method_name: str, args: argparse.Namespace) -> tuple[str | None, str | None, bool]:
    if method_name in BASELINE_CONFIGS:
        return (None, None, False)
    if method_name not in RL_METHOD_SPECS:
        raise ValueError(f"Unsupported method in reproduce_paper: {method_name}")
    spec = RL_METHOD_SPECS[method_name]
    checkpoint_arg = spec["checkpoint_arg"]
    checkpoint_path = None if checkpoint_arg is None else getattr(args, checkpoint_arg)
    return (str(spec["decision_kind"]), checkpoint_path, checkpoint_arg is not None)


def _expected_cells(
    *,
    paper_exp_id: str,
    scenarios: list[str],
    methods: list[str],
    ablations: list[str],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for scenario_name in scenarios:
        for method_name in methods:
            run_id = f"{scenario_name}__{method_name}"
            cells.append(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    "variant_type": "method",
                    "variant_name": method_name,
                    "run_id": run_id,
                    "config_path": str(
                        Path("outputs")
                        / paper_exp_id
                        / "generated_configs"
                        / f"{scenario_name}__method__{method_name}.yaml"
                    ),
                    "output_dir": str(Path("outputs") / paper_exp_id / "runs" / run_id),
                }
            )
        for ablation_name in ablations:
            method_label = f"adaptive_apf__{ablation_name}"
            run_id = f"{scenario_name}__{method_label}"
            cells.append(
                {
                    "scenario": scenario_name,
                    "method": method_label,
                    "variant_type": "ablation",
                    "variant_name": ablation_name,
                    "run_id": run_id,
                    "config_path": str(
                        Path("outputs")
                        / paper_exp_id
                        / "generated_configs"
                        / f"{scenario_name}__ablation__{ablation_name}.yaml"
                    ),
                    "output_dir": str(Path("outputs") / paper_exp_id / "runs" / run_id),
                }
            )
    return cells


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_manifest(
    *,
    exp_id: str,
    canonical_matrix: bool,
    expected_seeds: list[int],
    scenarios: list[str],
    methods: list[str],
    ablations: list[str],
    expected_cells: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
    acceptance_path: Path,
    matrix_index_path: Path,
) -> dict[str, object]:
    return {
        "exp_id": exp_id,
        "canonical_matrix": canonical_matrix,
        "primary_method": PRIMARY_METHOD,
        "expected_seed_count": len(expected_seeds),
        "expected_seeds": expected_seeds,
        "expected_scenarios": scenarios,
        "expected_methods": methods,
        "expected_ablations": ablations,
        "expected_cells": expected_cells,
        "observed_row_count": len(raw_rows),
        "observed_run_count": len({str(row.get("run_id", "")) for row in raw_rows}),
        "observed_git_commits": sorted(
            {str(row["git_commit"]) for row in raw_rows if str(row.get("git_commit", "")).strip()}
        ),
        "matrix_index_path": matrix_index_path.name,
        "paper_acceptance_path": acceptance_path.name,
        "all_runs_path": "all_runs.csv",
    }


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a JSON object: {path}")
    return payload


def _manifest_list_of_str(payload: dict[str, object], key: str) -> list[str]:
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Manifest field '{key}' must be a list.")
    return [str(value) for value in values]


def _manifest_list_of_int(payload: dict[str, object], key: str) -> list[int]:
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Manifest field '{key}' must be a list.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Manifest field '{key}' contains a non-integer value: {value!r}")
        if isinstance(value, int):
            normalized.append(value)
            continue
        if isinstance(value, float) and float(value).is_integer():
            normalized.append(int(value))
            continue
        raise ValueError(f"Manifest field '{key}' contains a non-integer value: {value!r}")
    return normalized


def _manifest_expected_cells(payload: dict[str, object]) -> list[dict[str, object]]:
    values = payload.get("expected_cells")
    if not isinstance(values, list):
        raise ValueError("Manifest field 'expected_cells' must be a list.")
    required_keys = (
        "scenario",
        "method",
        "variant_type",
        "variant_name",
        "run_id",
        "config_path",
        "output_dir",
    )
    normalized: list[dict[str, object]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"Manifest expected_cells[{index}] must be an object.")
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            raise ValueError(
                f"Manifest expected_cells[{index}] is missing required keys: {', '.join(missing_keys)}"
            )
        normalized.append({key: value[key] for key in required_keys})
    return normalized


def _resolve_audit_spec(
    *,
    paper_dir: Path,
    exp_id: str,
    canonical_matrix: bool,
) -> tuple[bool, list[int], list[str], list[str], list[str], list[dict[str, object]]]:
    manifest_path = paper_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        return (
            bool(manifest.get("canonical_matrix", False)),
            _manifest_list_of_int(manifest, "expected_seeds"),
            _manifest_list_of_str(manifest, "expected_scenarios"),
            _manifest_list_of_str(manifest, "expected_methods"),
            _manifest_list_of_str(manifest, "expected_ablations"),
            _manifest_expected_cells(manifest),
        )

    if canonical_matrix:
        scenarios = list(CANONICAL_SCENARIOS)
        methods = list(CANONICAL_METHODS)
        ablations = list(ABLATION_CONFIGS)
        expected_seeds = list(CANONICAL_SEEDS)
        expected_cells = _expected_cells(
            paper_exp_id=exp_id,
            scenarios=scenarios,
            methods=methods,
            ablations=ablations,
        )
        return True, expected_seeds, scenarios, methods, ablations, expected_cells

    raise FileNotFoundError(
        "Manifest-driven audit requires an existing manifest.json or explicit --canonical-matrix."
    )


def _runtime_state_paths(paper_dir: Path) -> tuple[Path, Path]:
    return (paper_dir / "run_runtime_state.json", paper_dir / "cell_runtime_state.csv")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime | None:
    rendered = str(value).strip()
    if not rendered:
        return None
    try:
        return datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return None


def _runtime_cell_key(payload: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(payload.get("scenario", "")),
        str(payload.get("method", "")),
        str(payload.get("variant_type", "")),
        str(payload.get("variant_name", "")),
    )


def _runtime_sort_key(payload: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(payload.get("scenario", "")),
        str(payload.get("variant_type", "")),
        str(payload.get("variant_name", "")),
        str(payload.get("method", "")),
    )


def _summary_path_for_cell(repo_root: Path, cell: dict[str, object]) -> Path:
    return repo_root / str(cell["output_dir"]) / "summary.csv"


def _normalize_runtime_status(value: object) -> str:
    rendered = str(value).strip().lower()
    if rendered in RUNTIME_STATUS_VALUES:
        return rendered
    return RUNTIME_STATUS_PENDING


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    rendered = str(value).strip().lower()
    return rendered in {"1", "true", "yes", "y"}


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_runtime_cell_rows(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, object]] = {}
        for row in reader:
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            rows[run_id] = dict(row)
        return rows


def _runtime_overrides(
    *,
    runtime_status: str,
    now: datetime,
    reset_started_at: bool = False,
    finished_at: bool = False,
    runner_pid: int | None = None,
    runner_started_at: datetime | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "runtime_status": runtime_status,
        "last_heartbeat": _format_timestamp(now),
        "heartbeat_age_seconds": 0 if runtime_status == RUNTIME_STATUS_RUNNING else "",
        "stalled": False,
    }
    if reset_started_at:
        payload["started_at"] = _format_timestamp(now)
    if finished_at:
        payload["finished_at"] = _format_timestamp(now)
    elif runtime_status == RUNTIME_STATUS_RUNNING:
        payload["finished_at"] = ""
    if runner_pid is not None:
        payload["runner_pid"] = runner_pid
    if runner_started_at is not None:
        payload["runner_started_at"] = _format_timestamp(runner_started_at)
    return payload


def _build_runtime_cell_rows(
    *,
    expected_cells: list[dict[str, object]],
    matrix_index_rows: list[dict[str, object]],
    previous_rows: dict[str, dict[str, object]],
    runtime_overrides: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    matrix_index_by_key = {
        _runtime_cell_key(row): row
        for row in matrix_index_rows
        if bool(row.get("expected_cell", True))
    }
    overrides = runtime_overrides or {}
    rows: list[dict[str, object]] = []
    for expected_cell in sorted(expected_cells, key=_runtime_sort_key):
        key = _runtime_cell_key(expected_cell)
        matrix_row = matrix_index_by_key.get(key, {})
        run_id = str(expected_cell["run_id"])
        previous = previous_rows.get(run_id, {})
        completed_seed_count = _to_int(matrix_row.get("actual_seed_count", 0))
        completed_progress = _to_float(matrix_row.get("progress_ratio", 0.0))
        expected_seed_count = max(_to_int(matrix_row.get("expected_seed_count", 0)), 0)
        runtime_status = _normalize_runtime_status(previous.get("runtime_status", RUNTIME_STATUS_PENDING))
        if _to_bool(matrix_row.get("cell_complete", False)):
            runtime_status = RUNTIME_STATUS_COMPLETE
        elif runtime_status not in {RUNTIME_STATUS_RUNNING, RUNTIME_STATUS_FAILED}:
            runtime_status = RUNTIME_STATUS_PENDING

        prev_runner_pid = _to_int(previous.get("runner_pid", 0))
        prev_runner_started_at_str = str(previous.get("runner_started_at", ""))
        prev_runner_started_at = _parse_timestamp(prev_runner_started_at_str)

        row = {
            "scenario": str(expected_cell.get("scenario", "")),
            "method": str(expected_cell.get("method", "")),
            "variant_type": str(expected_cell.get("variant_type", "")),
            "variant_name": str(expected_cell.get("variant_name", "")),
            "run_id": run_id,
            "config_path": str(expected_cell.get("config_path", "")),
            "output_dir": str(expected_cell.get("output_dir", "")),
            "expected_seed_count": expected_seed_count,
            "completed_seed_count": completed_seed_count,
            "completed_progress": completed_progress,
            "runtime_status": runtime_status,
            "started_at": str(previous.get("started_at", "")),
            "last_heartbeat": str(previous.get("last_heartbeat", "")),
            "finished_at": str(previous.get("finished_at", "")),
            "heartbeat_age_seconds": previous.get("heartbeat_age_seconds", ""),
            "stalled": _to_bool(previous.get("stalled", False)),
            "runner_pid": prev_runner_pid if prev_runner_pid > 0 else "",
            "runner_started_at": prev_runner_started_at_str,
            "process_alive": False,
            "orphaned": False,
        }
        if row["runtime_status"] != RUNTIME_STATUS_RUNNING:
            row["heartbeat_age_seconds"] = ""
            row["stalled"] = False
            row["process_alive"] = False
            row["orphaned"] = False
        else:
            alive = _check_process_alive(
                prev_runner_pid if prev_runner_pid > 0 else None,
                prev_runner_started_at,
            )
            row["process_alive"] = alive
            row["orphaned"] = not alive
        override = overrides.get(run_id)
        if override:
            row.update(override)
            if "runner_pid" in override or "runner_started_at" in override:
                new_pid = _to_int(row.get("runner_pid", 0))
                new_started_at = _parse_timestamp(row.get("runner_started_at"))
                if row["runtime_status"] == RUNTIME_STATUS_RUNNING:
                    alive = _check_process_alive(
                        new_pid if new_pid > 0 else None,
                        new_started_at,
                    )
                    row["process_alive"] = alive
                    row["orphaned"] = not alive
        rows.append(row)

    running_rows = [row for row in rows if row["runtime_status"] == RUNTIME_STATUS_RUNNING]
    if len(running_rows) > 1:
        def _running_rank(payload: dict[str, object]) -> tuple[str, str, str]:
            return (
                str(payload.get("last_heartbeat", "")),
                str(payload.get("started_at", "")),
                str(payload.get("run_id", "")),
            )

        keep_run_id = max(running_rows, key=_running_rank)["run_id"]
        for row in rows:
            if row["runtime_status"] == RUNTIME_STATUS_RUNNING and row["run_id"] != keep_run_id:
                row["runtime_status"] = RUNTIME_STATUS_FAILED
                row["heartbeat_age_seconds"] = ""
                row["stalled"] = False
                row["process_alive"] = False
                row["orphaned"] = False
    return rows


def _summarize_runtime_rows(
    *,
    exp_id: str,
    runtime_rows: list[dict[str, object]],
) -> dict[str, object]:
    status_counts = Counter(str(row.get("runtime_status", RUNTIME_STATUS_PENDING)) for row in runtime_rows)
    expected_seed_total = sum(_to_int(row.get("expected_seed_count", 0)) for row in runtime_rows)
    completed_seed_total = sum(
        min(_to_int(row.get("completed_seed_count", 0)), _to_int(row.get("expected_seed_count", 0)))
        for row in runtime_rows
    )
    if expected_seed_total <= 0:
        bundle_completed_progress = 1.0
    else:
        bundle_completed_progress = float(completed_seed_total) / float(expected_seed_total)
    running_rows = [row for row in runtime_rows if row.get("runtime_status") == RUNTIME_STATUS_RUNNING]
    active_cell = dict(running_rows[0]) if running_rows else None
    return {
        "exp_id": exp_id,
        "cell_runtime_state_path": "cell_runtime_state.csv",
        "heartbeat_interval_seconds": RUNTIME_HEARTBEAT_INTERVAL_SECONDS,
        "stall_timeout_seconds": RUNTIME_STALL_TIMEOUT_SECONDS,
        "num_expected_cells": len(runtime_rows),
        "num_pending_cells": status_counts[RUNTIME_STATUS_PENDING],
        "num_running_cells": status_counts[RUNTIME_STATUS_RUNNING],
        "num_complete_cells": status_counts[RUNTIME_STATUS_COMPLETE],
        "num_failed_cells": status_counts[RUNTIME_STATUS_FAILED],
        "running_cell_count": status_counts[RUNTIME_STATUS_RUNNING],
        "expected_seed_total": expected_seed_total,
        "completed_seed_total": completed_seed_total,
        "bundle_completed_progress": bundle_completed_progress,
        "active_cell": active_cell,
    }


def _write_runtime_artifacts(
    *,
    paper_dir: Path,
    exp_id: str,
    expected_cells: list[dict[str, object]],
    matrix_index_rows: list[dict[str, object]],
    runtime_overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_runtime_path, cell_runtime_path = _runtime_state_paths(paper_dir)
    with _RUNTIME_WRITE_LOCK:
        previous_rows = _load_runtime_cell_rows(cell_runtime_path)
        runtime_rows = _build_runtime_cell_rows(
            expected_cells=expected_cells,
            matrix_index_rows=matrix_index_rows,
            previous_rows=previous_rows,
            runtime_overrides=runtime_overrides,
        )
        runtime_summary = _summarize_runtime_rows(
            exp_id=exp_id,
            runtime_rows=runtime_rows,
        )
        write_csv_rows(runtime_rows, cell_runtime_path)
        _write_json(runtime_summary, run_runtime_path)
    return runtime_rows, runtime_summary


def _live_runtime_details(runtime_summary: dict[str, object]) -> dict[str, object] | None:
    active_cell = runtime_summary.get("active_cell")
    if not isinstance(active_cell, dict):
        return None
    last_heartbeat = _parse_timestamp(active_cell.get("last_heartbeat"))
    heartbeat_age_seconds: int | None = None
    stalled = False
    if last_heartbeat is not None:
        heartbeat_age_seconds = max(0, int((_utc_now() - last_heartbeat).total_seconds()))
        stalled = heartbeat_age_seconds > RUNTIME_STALL_TIMEOUT_SECONDS
    runner_pid = _to_int(active_cell.get("runner_pid", 0))
    runner_started_at = _parse_timestamp(active_cell.get("runner_started_at"))
    process_alive = _check_process_alive(
        runner_pid if runner_pid > 0 else None,
        runner_started_at,
    )
    orphaned = not process_alive
    return {
        "scenario": str(active_cell.get("scenario", "")),
        "variant_type": str(active_cell.get("variant_type", "")),
        "variant_name": str(active_cell.get("variant_name", "")),
        "method": str(active_cell.get("method", "")),
        "run_id": str(active_cell.get("run_id", "")),
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "stalled": stalled,
        "runner_pid": runner_pid if runner_pid > 0 else None,
        "process_alive": process_alive,
        "orphaned": orphaned,
    }


def _print_runtime_audit_summary(*, runtime_summary: dict[str, object]) -> None:
    print(f"bundle_completed_progress={float(runtime_summary['bundle_completed_progress']):.6f}")
    print(f"running_cell_count={int(runtime_summary['running_cell_count'])}")
    print(f"num_pending_cells={int(runtime_summary['num_pending_cells'])}")
    print(f"num_running_cells={int(runtime_summary['num_running_cells'])}")
    print(f"num_complete_cells={int(runtime_summary['num_complete_cells'])}")
    print(f"num_failed_cells={int(runtime_summary['num_failed_cells'])}")
    active_details = _live_runtime_details(runtime_summary)
    if active_details is None:
        return
    print(f"active_cell_scenario={active_details['scenario']}")
    print(f"active_cell_variant_type={active_details['variant_type']}")
    print(f"active_cell_variant_name={active_details['variant_name']}")
    print(f"active_cell_method={active_details['method']}")
    print(f"active_cell_run_id={active_details['run_id']}")
    print(f"heartbeat_age_seconds={active_details['heartbeat_age_seconds']}")
    print(f"stalled={active_details['stalled']}")
    print(f"runner_pid={active_details['runner_pid']}")
    print(f"process_alive={active_details['process_alive']}")
    print(f"orphaned={active_details['orphaned']}")


def _runtime_heartbeat_loop(
    *,
    stop_event: threading.Event,
    paper_dir: Path,
    exp_id: str,
    expected_cells: list[dict[str, object]],
    matrix_index_rows: list[dict[str, object]],
    run_id: str,
    runner_pid: int,
    runner_started_at: datetime,
) -> None:
    while not stop_event.wait(RUNTIME_HEARTBEAT_INTERVAL_SECONDS):
        now = _utc_now()
        _write_runtime_artifacts(
            paper_dir=paper_dir,
            exp_id=exp_id,
            expected_cells=expected_cells,
            matrix_index_rows=matrix_index_rows,
            runtime_overrides={
                run_id: _runtime_overrides(
                    runtime_status=RUNTIME_STATUS_RUNNING,
                    now=now,
                    runner_pid=runner_pid,
                    runner_started_at=runner_started_at,
                )
            },
        )


def _infer_unexpected_cell_from_run_id(run_id: str) -> dict[str, object]:
    scenario_name, _, remainder = run_id.partition("__")
    if "__adaptive_apf__" in run_id:
        _, _, ablation_name = run_id.partition("__adaptive_apf__")
        return {
            "scenario": scenario_name,
            "method": f"adaptive_apf__{ablation_name}",
            "variant_type": "ablation",
            "variant_name": ablation_name,
            "run_id": run_id,
        }
    return {
        "scenario": scenario_name,
        "method": remainder,
        "variant_type": "method",
        "variant_name": remainder,
        "run_id": run_id,
    }


def _collect_disk_rows(
    *,
    repo_root: Path,
    paper_dir: Path,
    expected_cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    seen_run_ids: set[str] = set()
    for cell in expected_cells:
        run_id = str(cell["run_id"])
        seen_run_ids.add(run_id)
        output_dir = repo_root / str(cell["output_dir"])
        summary_path = output_dir / "summary.csv"
        if not summary_path.exists():
            continue
        for row in read_summary_csv(summary_path):
            raw_rows.append(
                {
                    "scenario": cell["scenario"],
                    "method": cell["method"],
                    "variant_type": cell["variant_type"],
                    "variant_name": cell["variant_name"],
                    "run_id": run_id,
                    "config_path": cell["config_path"],
                    "output_dir": _safe_relative(output_dir, repo_root),
                    **row,
                }
            )

    runs_dir = paper_dir / "runs"
    if runs_dir.exists():
        for summary_path in sorted(runs_dir.glob("*/summary.csv")):
            run_id = summary_path.parent.name
            if run_id in seen_run_ids:
                continue
            inferred = _infer_unexpected_cell_from_run_id(run_id)
            for row in read_summary_csv(summary_path):
                raw_rows.append(
                    {
                        "scenario": inferred["scenario"],
                        "method": inferred["method"],
                        "variant_type": inferred["variant_type"],
                        "variant_name": inferred["variant_name"],
                        "run_id": run_id,
                        "config_path": "",
                        "output_dir": _safe_relative(summary_path.parent, repo_root),
                        **row,
                    }
                )
    return raw_rows


def _cell_progress_rows(matrix_index_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in matrix_index_rows:
        rows.append(
            {
                "scenario": row["scenario"],
                "method": row["method"],
                "variant_type": row["variant_type"],
                "variant_name": row["variant_name"],
                "run_id": row["run_id"],
                "output_dir": row["output_dir"],
                "expected_seed_count": row["expected_seed_count"],
                "observed_seed_count": row["observed_seed_count"],
                "actual_seed_count": row["actual_seed_count"],
                "missing_seed_count": row["missing_seed_count"],
                "progress_ratio": row["progress_ratio"],
                "status": row["status"],
                "config_hash_consistent": row["config_hash_consistent"],
                "validation_errors": row["validation_errors"],
            }
        )
    return rows


def _write_sealed_artifacts(
    *,
    repo_root: Path,
    paper_dir: Path,
    exp_id: str,
    canonical_matrix: bool,
    expected_seeds: list[int],
    scenarios: list[str],
    methods: list[str],
    ablations: list[str],
    expected_cells: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    raw_rows = _collect_disk_rows(
        repo_root=repo_root,
        paper_dir=paper_dir,
        expected_cells=expected_cells,
    )
    matrix_index_rows, acceptance = validate_canonical_bundle(
        raw_rows,
        expected_cells=expected_cells,
        expected_seeds=expected_seeds,
        primary_method=PRIMARY_METHOD,
    )
    progress = summarize_canonical_progress(matrix_index_rows, acceptance)
    matrix_index_path = paper_dir / "matrix_index.csv"
    acceptance_path = paper_dir / "paper_acceptance.json"
    manifest_path = paper_dir / "manifest.json"
    cell_progress_path = paper_dir / "cell_progress.csv"
    run_progress_path = paper_dir / "run_progress.json"
    write_csv_rows(matrix_index_rows, matrix_index_path)
    write_csv_rows(_cell_progress_rows(matrix_index_rows), cell_progress_path)
    _write_json(acceptance, acceptance_path)
    _write_json(progress, run_progress_path)
    _write_json(
        _build_manifest(
            exp_id=exp_id,
            canonical_matrix=canonical_matrix,
            expected_seeds=expected_seeds,
            scenarios=scenarios,
            methods=methods,
            ablations=ablations,
            expected_cells=expected_cells,
            raw_rows=raw_rows,
            acceptance_path=acceptance_path,
            matrix_index_path=matrix_index_path,
        ),
        manifest_path,
    )
    return raw_rows, matrix_index_rows, acceptance, progress


def _sorted_cell_payloads(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        cells,
        key=lambda cell: (
            str(cell.get("scenario", "")),
            str(cell.get("variant_type", "")),
            str(cell.get("variant_name", "")),
            str(cell.get("method", "")),
        ),
    )


def _print_bundle_audit_summary(
    *,
    paper_dir: Path,
    progress: dict[str, object],
    acceptance: dict[str, object],
) -> None:
    print(f"paper_dir={paper_dir}")
    print(f"bundle_progress={float(progress['bundle_progress']):.6f}")
    print(f"num_expected_cells={int(progress['num_expected_cells'])}")
    print(f"num_complete_cells={int(progress['num_complete_cells'])}")
    print(f"remaining_cell_count={int(progress['remaining_cell_count'])}")
    print(f"bundle_complete={bool(acceptance['bundle_complete'])}")
    print(f"primary_safety_valid={acceptance['primary_safety_valid']}")

    for label, cells in (
        ("missing_cells", acceptance.get("missing_cells", [])),
        ("invalid_cells", acceptance.get("invalid_cells", [])),
        ("unexpected_cells", acceptance.get("unexpected_cells", [])),
    ):
        sorted_cells = _sorted_cell_payloads(list(cells))
        print(f"{label}={len(sorted_cells)}")
        for cell in sorted_cells:
            suffix_parts: list[str] = []
            if "missing_seed_count" in cell:
                suffix_parts.append(f"missing_seed_count={cell['missing_seed_count']}")
            if "errors" in cell:
                suffix_parts.append(f"errors={','.join(str(error) for error in cell['errors'])}")
            suffix = ""
            if suffix_parts:
                suffix = " | " + " | ".join(suffix_parts)
            print(
                "  - "
                f"{cell.get('scenario', '')} | {cell.get('variant_type', '')} | "
                f"{cell.get('variant_name', '')} | method={cell.get('method', '')}{suffix}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "outputs" / args.exp_id
    if args.status_only or args.validate_only:
        try:
            (
                resolved_canonical_matrix,
                resolved_seeds,
                resolved_scenarios,
                resolved_methods,
                resolved_ablations,
                resolved_expected_cells,
            ) = _resolve_audit_spec(
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                canonical_matrix=bool(args.canonical_matrix),
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        _, matrix_index_rows, acceptance, progress = _write_sealed_artifacts(
            repo_root=repo_root,
            paper_dir=paper_dir,
            exp_id=args.exp_id,
            canonical_matrix=resolved_canonical_matrix,
            expected_seeds=resolved_seeds,
            scenarios=resolved_scenarios,
            methods=resolved_methods,
            ablations=resolved_ablations,
            expected_cells=resolved_expected_cells,
        )
        _, runtime_summary = _write_runtime_artifacts(
            paper_dir=paper_dir,
            exp_id=args.exp_id,
            expected_cells=resolved_expected_cells,
            matrix_index_rows=matrix_index_rows,
        )
        _print_bundle_audit_summary(
            paper_dir=paper_dir,
            progress=progress,
            acceptance=acceptance,
        )
        _print_runtime_audit_summary(runtime_summary=runtime_summary)
        if args.validate_only:
            bundle_valid = bool(acceptance["bundle_complete"]) and bool(acceptance["primary_safety_valid"])
            return 0 if bundle_valid else 1
        return 0

    if args.canonical_matrix:
        args.seeds = list(CANONICAL_SEEDS)
        args.scenarios = list(CANONICAL_SCENARIOS)
        args.methods = list(CANONICAL_METHODS)
        args.ablations = list(ABLATION_CONFIGS)

    generated_dir = paper_dir / "generated_configs"
    expected_cells = _expected_cells(
        paper_exp_id=args.exp_id,
        scenarios=list(args.scenarios),
        methods=list(args.methods),
        ablations=list(args.ablations),
    )
    raw_rows, matrix_index_rows, _, _ = _write_sealed_artifacts(
        repo_root=repo_root,
        paper_dir=paper_dir,
        exp_id=args.exp_id,
        canonical_matrix=bool(args.canonical_matrix),
        expected_seeds=list(args.seeds),
        scenarios=list(args.scenarios),
        methods=list(args.methods),
        ablations=list(args.ablations),
        expected_cells=expected_cells,
    )
    _write_runtime_artifacts(
        paper_dir=paper_dir,
        exp_id=args.exp_id,
        expected_cells=expected_cells,
        matrix_index_rows=matrix_index_rows,
    )
    expected_cells_by_run_id = {
        str(cell["run_id"]): cell
        for cell in expected_cells
    }
    for scenario_name in args.scenarios:
        scenario_path = _scenario_config_path(repo_root, scenario_name)
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario config does not exist: {scenario_path}")
        for method_name in args.methods:
            if method_name not in BASELINE_CONFIGS and method_name not in RL_METHOD_SPECS:
                raise ValueError(f"Unsupported method in reproduce_paper: {method_name}")
            decision_kind, rl_checkpoint, checkpoint_required = _method_override(method_name, args)
            if checkpoint_required and not rl_checkpoint:
                continue
            config_path = _generated_config_path(
                repo_root=repo_root,
                generated_dir=generated_dir,
                scenario_name=scenario_name,
                variant_name=method_name,
                variant_type="method",
                paper_exp_id=args.exp_id,
            )
            run_id = f"{scenario_name}__{method_name}"
            expected_cell = expected_cells_by_run_id[run_id]
            if args.skip_existing and _summary_path_for_cell(repo_root, expected_cell).exists():
                raw_rows, matrix_index_rows, _, _ = _write_sealed_artifacts(
                    repo_root=repo_root,
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    canonical_matrix=bool(args.canonical_matrix),
                    expected_seeds=list(args.seeds),
                    scenarios=list(args.scenarios),
                    methods=list(args.methods),
                    ablations=list(args.ablations),
                    expected_cells=expected_cells,
                )
                _write_runtime_artifacts(
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    expected_cells=expected_cells,
                    matrix_index_rows=matrix_index_rows,
                )
                continue

            start_now = _utc_now()
            current_pid, current_pid_start = _get_current_process_info()
            _write_runtime_artifacts(
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                expected_cells=expected_cells,
                matrix_index_rows=matrix_index_rows,
                runtime_overrides={
                    run_id: _runtime_overrides(
                        runtime_status=RUNTIME_STATUS_RUNNING,
                        now=start_now,
                        reset_started_at=True,
                        runner_pid=current_pid,
                        runner_started_at=current_pid_start,
                    )
                },
            )
            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(
                target=_runtime_heartbeat_loop,
                kwargs={
                    "stop_event": stop_event,
                    "paper_dir": paper_dir,
                    "exp_id": args.exp_id,
                    "expected_cells": expected_cells,
                    "matrix_index_rows": matrix_index_rows,
                    "run_id": run_id,
                    "runner_pid": current_pid,
                    "runner_started_at": current_pid_start,
                },
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                _run_single_configuration(
                    config_path=config_path,
                    seeds=args.seeds,
                    run_id=run_id,
                    skip_existing=args.skip_existing,
                    decision_kind=decision_kind,
                    rl_checkpoint=rl_checkpoint,
                    deterministic_eval=args.deterministic_eval,
                )
            except Exception:
                stop_event.set()
                heartbeat_thread.join(timeout=5.0)
                failure_now = _utc_now()
                _write_runtime_artifacts(
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    expected_cells=expected_cells,
                    matrix_index_rows=matrix_index_rows,
                    runtime_overrides={
                        run_id: _runtime_overrides(
                            runtime_status=RUNTIME_STATUS_FAILED,
                            now=failure_now,
                            finished_at=True,
                        )
                    },
                )
                raise
            stop_event.set()
            heartbeat_thread.join(timeout=5.0)
            raw_rows, matrix_index_rows, _, _ = _write_sealed_artifacts(
                repo_root=repo_root,
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                canonical_matrix=bool(args.canonical_matrix),
                expected_seeds=list(args.seeds),
                scenarios=list(args.scenarios),
                methods=list(args.methods),
                ablations=list(args.ablations),
                expected_cells=expected_cells,
            )
            matrix_row = next(
                (
                    row
                    for row in matrix_index_rows
                    if _runtime_cell_key(row) == _runtime_cell_key(expected_cell)
                ),
                {},
            )
            completed_status = (
                RUNTIME_STATUS_COMPLETE
                if _to_bool(matrix_row.get("cell_complete", False))
                else RUNTIME_STATUS_PENDING
            )
            finish_now = _utc_now()
            _write_runtime_artifacts(
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                expected_cells=expected_cells,
                matrix_index_rows=matrix_index_rows,
                runtime_overrides={
                    run_id: _runtime_overrides(
                        runtime_status=completed_status,
                        now=finish_now,
                        finished_at=True,
                    )
                },
            )
        for ablation_name in args.ablations:
            if ablation_name not in ABLATION_CONFIGS:
                raise ValueError(f"Unsupported ablation in reproduce_paper: {ablation_name}")
            method_label = f"adaptive_apf__{ablation_name}"
            config_path = _generated_config_path(
                repo_root=repo_root,
                generated_dir=generated_dir,
                scenario_name=scenario_name,
                variant_name=ablation_name,
                variant_type="ablation",
                paper_exp_id=args.exp_id,
            )
            run_id = f"{scenario_name}__{method_label}"
            expected_cell = expected_cells_by_run_id[run_id]
            if args.skip_existing and _summary_path_for_cell(repo_root, expected_cell).exists():
                raw_rows, matrix_index_rows, _, _ = _write_sealed_artifacts(
                    repo_root=repo_root,
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    canonical_matrix=bool(args.canonical_matrix),
                    expected_seeds=list(args.seeds),
                    scenarios=list(args.scenarios),
                    methods=list(args.methods),
                    ablations=list(args.ablations),
                    expected_cells=expected_cells,
                )
                _write_runtime_artifacts(
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    expected_cells=expected_cells,
                    matrix_index_rows=matrix_index_rows,
                )
                continue

            start_now = _utc_now()
            current_pid, current_pid_start = _get_current_process_info()
            _write_runtime_artifacts(
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                expected_cells=expected_cells,
                matrix_index_rows=matrix_index_rows,
                runtime_overrides={
                    run_id: _runtime_overrides(
                        runtime_status=RUNTIME_STATUS_RUNNING,
                        now=start_now,
                        reset_started_at=True,
                        runner_pid=current_pid,
                        runner_started_at=current_pid_start,
                    )
                },
            )
            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(
                target=_runtime_heartbeat_loop,
                kwargs={
                    "stop_event": stop_event,
                    "paper_dir": paper_dir,
                    "exp_id": args.exp_id,
                    "expected_cells": expected_cells,
                    "matrix_index_rows": matrix_index_rows,
                    "run_id": run_id,
                    "runner_pid": current_pid,
                    "runner_started_at": current_pid_start,
                },
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                _run_single_configuration(
                    config_path=config_path,
                    seeds=args.seeds,
                    run_id=run_id,
                    skip_existing=args.skip_existing,
                )
            except Exception:
                stop_event.set()
                heartbeat_thread.join(timeout=5.0)
                failure_now = _utc_now()
                _write_runtime_artifacts(
                    paper_dir=paper_dir,
                    exp_id=args.exp_id,
                    expected_cells=expected_cells,
                    matrix_index_rows=matrix_index_rows,
                    runtime_overrides={
                        run_id: _runtime_overrides(
                            runtime_status=RUNTIME_STATUS_FAILED,
                            now=failure_now,
                            finished_at=True,
                        )
                    },
                )
                raise
            stop_event.set()
            heartbeat_thread.join(timeout=5.0)
            raw_rows, matrix_index_rows, _, _ = _write_sealed_artifacts(
                repo_root=repo_root,
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                canonical_matrix=bool(args.canonical_matrix),
                expected_seeds=list(args.seeds),
                scenarios=list(args.scenarios),
                methods=list(args.methods),
                ablations=list(args.ablations),
                expected_cells=expected_cells,
            )
            matrix_row = next(
                (
                    row
                    for row in matrix_index_rows
                    if _runtime_cell_key(row) == _runtime_cell_key(expected_cell)
                ),
                {},
            )
            completed_status = (
                RUNTIME_STATUS_COMPLETE
                if _to_bool(matrix_row.get("cell_complete", False))
                else RUNTIME_STATUS_PENDING
            )
            finish_now = _utc_now()
            _write_runtime_artifacts(
                paper_dir=paper_dir,
                exp_id=args.exp_id,
                expected_cells=expected_cells,
                matrix_index_rows=matrix_index_rows,
                runtime_overrides={
                    run_id: _runtime_overrides(
                        runtime_status=completed_status,
                        now=finish_now,
                        finished_at=True,
                    )
                },
            )

    summary_rows = summarize_experiments(
        raw_rows,
        group_keys=("scenario", "method"),
        metrics=DEFAULT_METRICS,
    )
    comparison_rows = pairwise_compare_to_reference(
        raw_rows,
        reference_method=PRIMARY_METHOD,
        metrics=DEFAULT_METRICS,
    )
    export_paper_artifacts(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        output_dir=paper_dir,
    )
    _write_sealed_artifacts(
        repo_root=repo_root,
        paper_dir=paper_dir,
        exp_id=args.exp_id,
        canonical_matrix=bool(args.canonical_matrix),
        expected_seeds=list(args.seeds),
        scenarios=list(args.scenarios),
        methods=list(args.methods),
        ablations=list(args.ablations),
        expected_cells=expected_cells,
    )
    print(paper_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
