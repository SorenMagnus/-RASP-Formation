"""Build paper-ready tables and figures from existing output directories.

This script is intentionally offline-only: it reads completed run directories
under ``outputs/`` and never touches the live training / benchmark code path.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    summarize_experiments,
    write_csv_rows,
)


KNOWN_BASELINE_METHODS = {"apf", "apf_lf", "st_apf", "dwa", "orca"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect existing output directories into one offline paper bundle."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Existing run output directory containing summary.csv. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for merged CSV tables and figures.",
    )
    parser.add_argument(
        "--reference-method",
        default="no_rl",
        help="Reference method used for pairwise comparison tables.",
    )
    return parser


def _safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_resolved_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _infer_scenario(run_dir: Path, config_payload: dict[str, object]) -> str:
    experiment = config_payload.get("experiment")
    if isinstance(experiment, dict):
        name = experiment.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return run_dir.name


def _infer_method(run_dir: Path, config_payload: dict[str, object]) -> str:
    controller = config_payload.get("controller")
    decision = config_payload.get("decision")
    controller_kind = ""
    decision_kind = ""
    checkpoint_path = ""
    if isinstance(controller, dict):
        controller_kind = str(controller.get("kind", "")).strip().lower()
    if isinstance(decision, dict):
        decision_kind = str(decision.get("kind", "")).strip().lower()
        rl_block = decision.get("rl")
        if isinstance(rl_block, dict):
            checkpoint_path = str(rl_block.get("checkpoint_path", "")).strip().lower()

    if decision_kind == "rl":
        if "param" in checkpoint_path:
            return "rl_param_only"
        if "mode" in checkpoint_path:
            return "rl_mode_only"
        if "full" in checkpoint_path or "supervisor" in checkpoint_path:
            return "rl_full_supervisor"
        return "rl"
    if controller_kind in KNOWN_BASELINE_METHODS:
        return controller_kind
    if controller_kind == "adaptive_apf" and decision_kind == "fsm":
        return "no_rl"
    if controller_kind:
        return controller_kind
    if decision_kind:
        return decision_kind
    return run_dir.name


def _inventory_row(*, run_dir: Path, repo_root: Path, config_payload: dict[str, object]) -> dict[str, object]:
    decision = config_payload.get("decision") if isinstance(config_payload, dict) else None
    controller = config_payload.get("controller") if isinstance(config_payload, dict) else None
    rl_block = decision.get("rl") if isinstance(decision, dict) else None
    return {
        "run_dir": _safe_relative(run_dir, repo_root),
        "scenario": _infer_scenario(run_dir, config_payload),
        "method": _infer_method(run_dir, config_payload),
        "decision_kind": decision.get("kind", "") if isinstance(decision, dict) else "",
        "controller_kind": controller.get("kind", "") if isinstance(controller, dict) else "",
        "rl_checkpoint_path": rl_block.get("checkpoint_path", "") if isinstance(rl_block, dict) else "",
        "config_resolved_path": _safe_relative(run_dir / "config_resolved.yaml", repo_root),
        "summary_path": _safe_relative(run_dir / "summary.csv", repo_root),
    }


def _raw_rows_for_run(*, run_dir: Path, repo_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary.csv under: {run_dir}")
    config_payload = _read_resolved_config(run_dir)
    scenario = _infer_scenario(run_dir, config_payload)
    method = _infer_method(run_dir, config_payload)
    raw_rows: list[dict[str, object]] = []
    for row in read_summary_csv(summary_path):
        raw_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "variant_type": "method",
                "variant_name": method,
                "run_id": run_dir.name,
                "output_dir": _safe_relative(run_dir, repo_root),
                **row,
            }
        )
    return raw_rows, _inventory_row(run_dir=run_dir, repo_root=repo_root, config_payload=config_payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dir]

    raw_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_raw_rows, inventory_row = _raw_rows_for_run(run_dir=run_dir, repo_root=repo_root)
        raw_rows.extend(run_raw_rows)
        inventory_rows.append(inventory_row)

    summary_rows = summarize_experiments(
        raw_rows,
        group_keys=("scenario", "method"),
        metrics=DEFAULT_METRICS,
    )
    comparison_rows = pairwise_compare_to_reference(
        raw_rows,
        reference_method=args.reference_method,
        metrics=DEFAULT_METRICS,
    )

    export_paper_artifacts(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        output_dir=output_dir,
    )
    write_csv_rows(inventory_rows, output_dir / "tables" / "run_inventory.csv")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
