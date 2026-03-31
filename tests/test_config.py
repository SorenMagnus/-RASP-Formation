"""Config loader regression tests."""

from __future__ import annotations

import math
from pathlib import Path

from apflf.utils.config import load_config


def test_load_config_accepts_resolved_export_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_like.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: resolved_like",
                f"  output_root: {tmp_path.as_posix()}",
                "  save_traj: true",
                "simulation:",
                "  dt: 0.1",
                "  steps: 10",
                "  wheelbase: 2.8",
                "  target_speed: 8.0",
                "  bounds:",
                "    accel_min: -2.5",
                "    accel_max: 2.0",
                "    steer_min: -0.4",
                "    steer_max: 0.4",
                "    speed_min: 0.0",
                "    speed_max: 12.0",
                "controller:",
                "  kind: adaptive_apf",
                "  repulsive_gain_min: 11.0",
                "  repulsive_gain_max: 21.0",
                "  road_gain_min: 4.0",
                "  road_gain_max: 13.0",
                "safety:",
                "  enabled: true",
                "  solver: osqp",
                "  safe_distance: 0.75",
                "  barrier_decay: 2.0",
                "  slack_penalty: 900.0",
                "  max_slack: 1.5",
                "  road_boundary_margin: 0.2",
                "  fallback_brake: 1.8",
                "  fallback_steer_gain: 0.35",
                "decision:",
                "  kind: fsm",
                "  default_mode: topology=line|behavior=follow|gain=nominal",
                "scenario:",
                "  vehicle_count: 3",
                "  spacing: 8.0",
                "  spawn_jitter_std: 0.0",
                "  initial_speed: 2.0",
                "  goal_x: 40.0",
                "  goal_tolerance: 0.5",
                "  road:",
                "    length: 80.0",
                "    lane_center_y: 0.0",
                "    half_width: 3.5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert math.isclose(config.simulation.bounds.steer_min, -0.4)
    assert math.isclose(config.simulation.bounds.steer_max, 0.4)
    assert math.isclose(config.controller.repulsive_gain_min, 11.0)
    assert math.isclose(config.controller.repulsive_gain_max, 21.0)
    assert math.isclose(config.controller.road_gain_min, 4.0)
    assert math.isclose(config.controller.road_gain_max, 13.0)
    assert math.isclose(config.safety.safe_distance, 0.75)
    assert math.isclose(config.safety.barrier_decay, 2.0)
    assert math.isclose(config.safety.slack_penalty, 900.0)


def test_load_config_parses_rl_reward_weights(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "reward_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"extends: { (repo_root / 'configs' / 'default.yaml').as_posix() }",
                "decision:",
                "  rl:",
                "    reward:",
                "      progress_weight: 1.5",
                "      formation_weight: 0.25",
                "      intervention_weight: 0.3",
                "      qp_weight: 0.4",
                "      fallback_weight: 0.5",
                "      slack_weight: 0.6",
                "      theta_rate_weight: 0.07",
                "      goal_reward: 6.0",
                "      collision_penalty: 11.0",
                "      boundary_penalty: 9.0",
                "      correction_epsilon: 1e-5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert math.isclose(config.decision.rl.reward.progress_weight, 1.5)
    assert math.isclose(config.decision.rl.reward.formation_weight, 0.25)
    assert math.isclose(config.decision.rl.reward.intervention_weight, 0.3)
    assert math.isclose(config.decision.rl.reward.qp_weight, 0.4)
    assert math.isclose(config.decision.rl.reward.fallback_weight, 0.5)
    assert math.isclose(config.decision.rl.reward.slack_weight, 0.6)
    assert math.isclose(config.decision.rl.reward.theta_rate_weight, 0.07)
    assert math.isclose(config.decision.rl.reward.goal_reward, 6.0)
    assert math.isclose(config.decision.rl.reward.collision_penalty, 11.0)
    assert math.isclose(config.decision.rl.reward.boundary_penalty, 9.0)
    assert math.isclose(config.decision.rl.reward.correction_epsilon, 1e-5)
