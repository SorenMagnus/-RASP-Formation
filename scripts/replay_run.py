"""Replay a saved rollout artifact and optionally verify its summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.sim.replay import compare_summary_dicts, read_summary_row, recompute_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a saved rollout artifact.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory containing config_resolved.yaml and traj/.")
    parser.add_argument("--seed", required=True, type=int, help="Seed id to replay.")
    parser.add_argument(
        "--verify-summary",
        action="store_true",
        help="Compare the replayed summary with the saved summary.csv row.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    replay_summary = recompute_summary(run_dir, args.seed)
    print(json.dumps(replay_summary, indent=2, ensure_ascii=False, sort_keys=True))

    if args.verify_summary:
        saved_summary = read_summary_row(run_dir, args.seed)
        compared_saved = {key: value for key, value in saved_summary.items() if key not in {"seed", "config_hash", "git_commit"}}
        mismatches = compare_summary_dicts(compared_saved, replay_summary)
        if mismatches:
            for key, values in sorted(mismatches.items()):
                print(f"mismatch {key}: saved={values[0]!r} replay={values[1]!r}", file=sys.stderr)
            return 1
        print("summary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
