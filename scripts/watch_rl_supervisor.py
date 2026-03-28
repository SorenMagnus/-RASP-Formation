"""Watchdog for long-running stage-1 RL training and benchmark execution.

This script avoids PowerShell's UTF-16 redirection so progress logs stay readable
in VS Code after resume/restart cycles on Windows.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch and resume the stage-1 RL supervisor training/benchmark loop."
    )
    parser.add_argument(
        "--config",
        default="configs/scenarios/s5_dense_multi_agent.yaml",
        help="Scenario config used for training and benchmark.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--steps-per-rollout", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="outputs/rl_train_s5_param_only/checkpoints/main.pt",
        help="Final training checkpoint path.",
    )
    parser.add_argument("--exp-id-prefix", default="s5_rl_stage1_cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--disable-stay-awake",
        action="store_true",
        help="Do not request a temporary stay-awake state from Windows.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log_paths(output_path: Path) -> dict[str, Path]:
    logs_dir = output_path.parent.parent / "logs"
    return {
        "main_stdout": logs_dir / "main_stdout.log",
        "main_stderr": logs_dir / "main_stderr.log",
        "supervisor": logs_dir / "supervisor.log",
        "benchmark_stdout": logs_dir / "benchmark_stdout.log",
        "benchmark_stderr": logs_dir / "benchmark_stderr.log",
    }


def _append_utf8_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)


def _read_processes(command_pattern: str) -> list[dict[str, object]]:
    ps_command = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -like '*{command_pattern}*' }} | "
        "Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps_command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    raw = completed.stdout.strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def _env_with_utf8() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _spawn_python(args: list[str], *, stdout_path: Path, stderr_path: Path) -> subprocess.Popen[bytes]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, *args],
            cwd=_repo_root(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=_env_with_utf8(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise


def _contains_nul(path: Path) -> bool:
    if not path.exists():
        return False
    return b"\x00" in path.read_bytes()


def _rotate_if_mixed(path: Path) -> Path | None:
    if not _contains_nul(path):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rotated = path.with_name(f"{path.stem}.mixed_powershell_{stamp}{path.suffix}")
    path.replace(rotated)
    return rotated


class StayAwake:
    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._kernel32 = getattr(ctypes, "windll", None)

    def enable(self) -> None:
        if self._kernel32 is None:
            return
        self._kernel32.kernel32.SetThreadExecutionState(
            self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED
        )

    def disable(self) -> None:
        if self._kernel32 is None:
            return
        self._kernel32.kernel32.SetThreadExecutionState(self._ES_CONTINUOUS)


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    output_path = (repo_root / args.output).resolve()
    latest_path = output_path.with_name("latest.pt")
    log_paths = _log_paths(output_path)
    benchmark_summary = repo_root / f"outputs/{args.exp_id_prefix}__rl_param_only/summary.csv"
    stay_awake = StayAwake()

    if not args.disable_stay_awake:
        stay_awake.enable()

    for log_key in ("main_stdout", "main_stderr"):
        rotated = _rotate_if_mixed(log_paths[log_key])
        if rotated is not None:
            _append_utf8_line(
                log_paths["supervisor"],
                f"rotated mixed-encoding log {log_paths[log_key].name} -> {rotated.name}",
            )

    _append_utf8_line(
        log_paths["supervisor"],
        "watchdog started; system sleep prevented while watchdog is alive",
    )

    last_status = ""
    last_latest_stamp = ""
    try:
        while True:
            train_processes = _read_processes("scripts/train_rl_supervisor.py")
            bench_processes = _read_processes("scripts/benchmark_s5_rl.py")
            latest_exists = latest_path.exists()
            main_exists = output_path.exists()

            if latest_exists:
                stamp = latest_path.stat().st_mtime_ns
                if str(stamp) != last_latest_stamp:
                    last_latest_stamp = str(stamp)
                    _append_utf8_line(
                        log_paths["supervisor"],
                        f"checkpoint available; safe_to_shutdown=True latest.pt.mtime_ns={stamp}",
                    )

            if main_exists:
                if not bench_processes and not benchmark_summary.exists():
                    proc = _spawn_python(
                        [
                            "scripts/benchmark_s5_rl.py",
                            "--config",
                            args.config,
                            "--seeds",
                            "0",
                            "1",
                            "2",
                            "--rl-checkpoint",
                            str(output_path),
                            "--exp-id-prefix",
                            args.exp_id_prefix,
                            "--deterministic-eval",
                        ],
                        stdout_path=log_paths["benchmark_stdout"],
                        stderr_path=log_paths["benchmark_stderr"],
                    )
                    _append_utf8_line(log_paths["supervisor"], f"benchmark pid={proc.pid}")
                elif benchmark_summary.exists():
                    _append_utf8_line(
                        log_paths["supervisor"],
                        "benchmark summary detected; watchdog exiting normally",
                    )
                    break
            elif not train_processes:
                train_args = [
                    "scripts/train_rl_supervisor.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(args.seed),
                    "--total-timesteps",
                    str(args.total_timesteps),
                    "--steps-per-rollout",
                    str(args.steps_per_rollout),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--device",
                    args.device,
                    "--output",
                    str(output_path),
                ]
                if latest_exists:
                    train_args.extend(["--resume-from", str(latest_path)])
                    _append_utf8_line(
                        log_paths["supervisor"],
                        f"starting training from checkpoint {latest_path}",
                    )
                else:
                    _append_utf8_line(log_paths["supervisor"], "starting training from scratch")
                proc = _spawn_python(
                    train_args,
                    stdout_path=log_paths["main_stdout"],
                    stderr_path=log_paths["main_stderr"],
                )
                _append_utf8_line(log_paths["supervisor"], f"training pid={proc.pid}")
                train_processes = [{"ProcessId": proc.pid}]

            status = (
                f"train={train_processes[0]['ProcessId'] if train_processes else 'none'} "
                f"main={main_exists} latest={latest_exists} "
                f"bench={bench_processes[0]['ProcessId'] if bench_processes else 'none'}"
            )
            if status != last_status:
                _append_utf8_line(log_paths["supervisor"], status)
                last_status = status

            time.sleep(max(args.poll_seconds, 5))
    finally:
        if not args.disable_stay_awake:
            stay_awake.disable()
        _append_utf8_line(
            log_paths["supervisor"],
            "watchdog exited and released stay-awake request",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
