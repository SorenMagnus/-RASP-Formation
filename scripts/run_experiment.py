"""Batch experiment command-line entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Inject the local `src/` tree so the script can be run directly."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.sim.runner import run_batch  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Run APF-LF experiments from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Deterministic seed list.")
    parser.add_argument("--exp-id", default=None, help="Optional output directory name.")
    parser.add_argument("--decision", choices=["fsm", "rl"], default=None, help="Override the decision module kind.")
    parser.add_argument("--rl-checkpoint", default=None, help="Optional RL supervisor checkpoint path.")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use deterministic RL policy evaluation when `--decision rl` is active.",
    )
    return parser


def main() -> int:
    """Run the configured batch and print the output directory."""

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.decision is not None or args.rl_checkpoint is not None or args.deterministic_eval:
        rl_config = replace(
            config.decision.rl,
            checkpoint_path=config.decision.rl.checkpoint_path if args.rl_checkpoint is None else args.rl_checkpoint,
            deterministic_eval=bool(args.deterministic_eval or config.decision.rl.deterministic_eval),
        )
        config = replace(
            config,
            decision=replace(
                config.decision,
                kind=config.decision.kind if args.decision is None else args.decision,
                rl=rl_config,
            ),
        )
    output_dir = run_batch(config=config, seeds=args.seeds, exp_id=args.exp_id)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
