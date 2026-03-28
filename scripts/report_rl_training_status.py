"""Read checkpoint/log artifacts and print a concise RL training status report."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROLLOUT_DONE_PATTERN = re.compile(
    r"^\[ppo\]\s+rollout_done\s+.*?timesteps_done=(?P<timesteps>\d+)/(?P<total>\d+)"
    r".*?elapsed_s=(?P<elapsed>[0-9.]+)\s+rollout_index=(?P<rollout_index>\d+)"
    r"\s+rollout_seed=(?P<rollout_seed>\d+)\s+batch_steps=(?P<batch_steps>\d+)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report checkpoint/log status for an RL training run.")
    parser.add_argument(
        "--train-dir",
        default="outputs/rl_train_s5_param_only",
        help="Training output directory containing checkpoints/ and logs/.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="Nominal total timesteps target used when the checkpoint does not store it.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=8,
        help="How many trailing stdout log lines to include in the report.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit a JSON report instead of human-readable text.",
    )
    return parser


def _read_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max(limit, 0) :]


def _latest_rollout(stdout_lines: list[str]) -> dict[str, object] | None:
    matches: list[dict[str, object]] = []
    for line in stdout_lines:
        match = ROLLOUT_DONE_PATTERN.match(line.strip())
        if not match:
            continue
        matches.append(
            {
                "timesteps_done": int(match.group("timesteps")),
                "total_timesteps": int(match.group("total")),
                "elapsed_s": float(match.group("elapsed")),
                "rollout_index": int(match.group("rollout_index")),
                "rollout_seed": int(match.group("rollout_seed")),
                "batch_steps": int(match.group("batch_steps")),
            }
        )
    if not matches:
        return None
    return matches[-1]


def _estimate_eta(rollout: dict[str, object] | None) -> tuple[float | None, str | None]:
    if rollout is None:
        return None, None
    timesteps_done = int(rollout["timesteps_done"])
    total_timesteps = int(rollout["total_timesteps"])
    elapsed_s = float(rollout["elapsed_s"])
    if timesteps_done <= 0 or elapsed_s <= 0.0 or total_timesteps <= timesteps_done:
        return 0.0 if total_timesteps <= timesteps_done else None, None
    steps_per_second = timesteps_done / elapsed_s
    if steps_per_second <= 1e-12:
        return None, None
    remaining_s = (total_timesteps - timesteps_done) / steps_per_second
    eta_iso = datetime.now(timezone.utc).astimezone() + timedelta(seconds=remaining_s)
    return remaining_s, eta_iso.isoformat(timespec="seconds")


def build_report(*, train_dir: Path, total_timesteps: int, tail: int) -> dict[str, object]:
    checkpoints_dir = train_dir / "checkpoints"
    logs_dir = train_dir / "logs"
    latest_path = checkpoints_dir / "latest.pt"
    main_path = checkpoints_dir / "main.pt"
    stdout_path = logs_dir / "main_stdout.log"
    stderr_path = logs_dir / "main_stderr.log"
    supervisor_path = logs_dir / "supervisor.log"

    checkpoint = _read_checkpoint(latest_path)
    stdout_tail = _tail_lines(stdout_path, tail)
    stderr_tail = _tail_lines(stderr_path, min(4, tail))
    supervisor_tail = _tail_lines(supervisor_path, min(6, tail))
    rollout = _latest_rollout(stdout_tail if stdout_tail else _tail_lines(stdout_path, 256))
    remaining_s, eta_iso = _estimate_eta(rollout)

    checkpoint_total = total_timesteps
    checkpoint_timesteps = None
    checkpoint_rollout_seed_next = None
    if checkpoint is not None:
        checkpoint_timesteps = int(checkpoint.get("timesteps_done", 0))
        checkpoint_rollout_seed_next = int(checkpoint.get("rollout_seed_next", 0))
        trainer_config = checkpoint.get("trainer_config")
        if isinstance(trainer_config, dict):
            maybe_total = trainer_config.get("total_timesteps")
            if isinstance(maybe_total, int) and maybe_total > 0:
                checkpoint_total = maybe_total

    progress_pct = None
    if checkpoint_timesteps is not None and checkpoint_total > 0:
        progress_pct = 100.0 * checkpoint_timesteps / checkpoint_total

    return {
        "train_dir": str(train_dir.resolve()),
        "latest_exists": latest_path.exists(),
        "main_exists": main_path.exists(),
        "latest_checkpoint": str(latest_path.resolve()),
        "main_checkpoint": str(main_path.resolve()),
        "timesteps_done": checkpoint_timesteps,
        "rollout_seed_next": checkpoint_rollout_seed_next,
        "total_timesteps": checkpoint_total,
        "progress_pct": progress_pct,
        "estimated_remaining_s": remaining_s,
        "estimated_finish_time": eta_iso,
        "safe_to_shutdown": latest_path.exists(),
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
        "supervisor_log": str(supervisor_path.resolve()),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "supervisor_tail": supervisor_tail,
        "latest_rollout": rollout,
    }


def _print_human(report: dict[str, object]) -> None:
    print(f"train_dir={report['train_dir']}")
    print(
        "checkpoint="
        f"{report['timesteps_done']}/{report['total_timesteps']} "
        f"progress={report['progress_pct']:.2f}%"
        if isinstance(report["progress_pct"], float)
        else "checkpoint=unavailable"
    )
    print(f"rollout_seed_next={report['rollout_seed_next']}")
    print(f"safe_to_shutdown={report['safe_to_shutdown']}")
    remaining_s = report["estimated_remaining_s"]
    if isinstance(remaining_s, (int, float)) and math.isfinite(float(remaining_s)):
        print(f"estimated_remaining_h={float(remaining_s) / 3600.0:.2f}")
    if report["estimated_finish_time"]:
        print(f"estimated_finish_time={report['estimated_finish_time']}")
    latest_rollout = report["latest_rollout"]
    if isinstance(latest_rollout, dict):
        print(
            "latest_rollout="
            f"index={latest_rollout['rollout_index']} "
            f"seed={latest_rollout['rollout_seed']} "
            f"timesteps_done={latest_rollout['timesteps_done']}"
        )
    print("stdout_tail:")
    for line in report["stdout_tail"]:
        print(f"  {line}")
    print("stderr_tail:")
    for line in report["stderr_tail"]:
        print(f"  {line}")
    print("supervisor_tail:")
    for line in report["supervisor_tail"]:
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = build_report(
        train_dir=Path(args.train_dir),
        total_timesteps=args.total_timesteps,
        tail=args.tail,
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
