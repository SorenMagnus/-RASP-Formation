"""Compare `no_rl` and `rl_param_only` on the stage-5 dense multi-agent scenario."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.analysis.stats import read_summary_csv  # noqa: E402
from apflf.sim.runner import run_batch  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark S5 with and without the RL supervisor.")
    parser.add_argument(
        "--config",
        default="configs/scenarios/s5_dense_multi_agent.yaml",
        help="Scenario config path.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rl-checkpoint", default=None, help="Checkpoint for `rl_param_only`.")
    parser.add_argument("--exp-id-prefix", default="benchmark_s5_rl")
    parser.add_argument("--deterministic-eval", action="store_true")
    return parser


def _print_rows(label: str, rows: list[dict[str, object]]) -> None:
    print(label)
    for row in rows:
        print(
            "seed={seed} leader_final_x={leader_final_x:.6f} reached_goal={reached_goal} "
            "fallback_events={fallback_events} safety_interventions={safety_interventions}".format(
                **row
            )
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / args.config)

    no_rl_dir = run_batch(
        config=replace(
            config,
            decision=replace(config.decision, kind="fsm"),
        ),
        seeds=args.seeds,
        exp_id=f"{args.exp_id_prefix}__no_rl",
    )
    _print_rows("no_rl", read_summary_csv(no_rl_dir / "summary.csv"))

    if args.rl_checkpoint:
        rl_dir = run_batch(
            config=replace(
                config,
                decision=replace(
                    config.decision,
                    kind="rl",
                    rl=replace(
                        config.decision.rl,
                        checkpoint_path=args.rl_checkpoint,
                        deterministic_eval=bool(args.deterministic_eval),
                    ),
                ),
            ),
            seeds=args.seeds,
            exp_id=f"{args.exp_id_prefix}__rl_param_only",
        )
        _print_rows("rl_param_only", read_summary_csv(rl_dir / "summary.csv"))
    else:
        print("rl_param_only skipped because --rl-checkpoint was not provided.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
