"""Run a reproducible experiment matrix and export paper artifacts."""

from __future__ import annotations

import argparse
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
    summarize_experiments,
)
from apflf.sim.runner import run_batch  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402

PRIMARY_METHOD = "no_rl"
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
        default=["s1_local_minima", "s2_dynamic_crossing", "s3_narrow_passage"],
        help="Scenario names without the .yaml suffix.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[PRIMARY_METHOD, "rl_param_only", "apf", "apf_lf", "st_apf", "dwa", "orca"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(ABLATION_CONFIGS),
        help="Ablation names to evaluate on top of the primary method.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-run outputs when the summary.csv already exists.",
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "outputs" / args.exp_id
    generated_dir = paper_dir / "generated_configs"
    raw_rows: list[dict[str, object]] = []

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
            output_dir = _run_single_configuration(
                config_path=config_path,
                seeds=args.seeds,
                run_id=run_id,
                skip_existing=args.skip_existing,
                decision_kind=decision_kind,
                rl_checkpoint=rl_checkpoint,
                deterministic_eval=args.deterministic_eval,
            )
            for row in read_summary_csv(output_dir / "summary.csv"):
                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "method": method_name,
                        "variant_type": "method",
                        "variant_name": method_name,
                        "run_id": run_id,
                        "config_path": str(config_path.relative_to(repo_root)),
                        "output_dir": str(output_dir.relative_to(repo_root)),
                        **row,
                    }
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
            output_dir = _run_single_configuration(
                config_path=config_path,
                seeds=args.seeds,
                run_id=run_id,
                skip_existing=args.skip_existing,
            )
            for row in read_summary_csv(output_dir / "summary.csv"):
                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "method": method_label,
                        "variant_type": "ablation",
                        "variant_name": ablation_name,
                        "run_id": run_id,
                        "config_path": str(config_path.relative_to(repo_root)),
                        "output_dir": str(output_dir.relative_to(repo_root)),
                        **row,
                    }
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
    print(paper_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
