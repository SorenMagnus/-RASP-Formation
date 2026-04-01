"""Run a reproducible experiment matrix and export paper artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
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

    if args.canonical_matrix:
        args.seeds = list(CANONICAL_SEEDS)
        args.scenarios = list(CANONICAL_SCENARIOS)
        args.methods = list(CANONICAL_METHODS)
        args.ablations = list(ABLATION_CONFIGS)

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "outputs" / args.exp_id
    generated_dir = paper_dir / "generated_configs"
    expected_cells = _expected_cells(
        paper_exp_id=args.exp_id,
        scenarios=list(args.scenarios),
        methods=list(args.methods),
        ablations=list(args.ablations),
    )
    raw_rows, _, _, _ = _write_sealed_artifacts(
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
    if args.status_only or args.validate_only:
        _, _, acceptance, progress = _write_sealed_artifacts(
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
        _print_bundle_audit_summary(
            paper_dir=paper_dir,
            progress=progress,
            acceptance=acceptance,
        )
        if args.validate_only:
            bundle_valid = bool(acceptance["bundle_complete"]) and bool(acceptance["primary_safety_valid"])
            return 0 if bundle_valid else 1
        return 0

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
            _run_single_configuration(
                config_path=config_path,
                seeds=args.seeds,
                run_id=run_id,
                skip_existing=args.skip_existing,
                decision_kind=decision_kind,
                rl_checkpoint=rl_checkpoint,
                deterministic_eval=args.deterministic_eval,
            )
            raw_rows, _, _, _ = _write_sealed_artifacts(
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
            _run_single_configuration(
                config_path=config_path,
                seeds=args.seeds,
                run_id=run_id,
                skip_existing=args.skip_existing,
            )
            raw_rows, _, _, _ = _write_sealed_artifacts(
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
