"""Train a stage-1 `rl_param_only` supervisor with the custom Torch PPO backend."""

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

from apflf.rl.env import SupervisorTrainingEnv  # noqa: E402
from apflf.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the stage-1 RL supervisor.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--steps-per-rollout", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
    config = replace(
        config,
        decision=replace(config.decision, kind="fsm"),
    )
    env = SupervisorTrainingEnv(config=config)
    trainer = PPOTrainer(
        env=env,
        theta_config=config.decision.rl.theta,
        config=PPOConfig(
            total_timesteps=args.total_timesteps,
            steps_per_rollout=args.steps_per_rollout,
            learning_rate=args.learning_rate,
        ),
        device=args.device,
    )
    logs = trainer.train(seed=args.seed)
    trainer.save_checkpoint(Path(args.output))
    if logs:
        print(logs[-1])
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
