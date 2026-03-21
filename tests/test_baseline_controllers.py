"""Tests for DWA and ORCA baseline controllers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from apflf.controllers.base import build_controller
from apflf.env.road import Road
from apflf.utils.config import load_config
from apflf.utils.types import Observation, ObstacleState, State


@pytest.mark.parametrize("kind", ["dwa", "orca"])
def test_controller_factory_supports_new_baselines(kind: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    controller = build_controller(
        config=replace(config.controller, kind=kind),
        bounds=config.simulation.bounds,
        road=Road(config.scenario.road),
        target_speed=config.simulation.target_speed,
        wheelbase=config.simulation.wheelbase,
        dt=config.simulation.dt,
    )

    assert controller is not None


@pytest.mark.parametrize("kind", ["dwa", "orca"])
def test_baseline_controller_actions_are_bounded_and_obstacle_responsive(kind: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    controller = build_controller(
        config=replace(config.controller, kind=kind),
        bounds=config.simulation.bounds,
        road=Road(config.scenario.road),
        target_speed=config.simulation.target_speed,
        wheelbase=config.simulation.wheelbase,
        dt=config.simulation.dt,
    )
    observation = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=0.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=-8.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=-16.0, y=0.0, yaw=0.0, speed=5.0),
        ),
        road=config.scenario.road,
        goal_x=80.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState(
                obstacle_id="lead_blocker",
                x=10.0,
                y=0.0,
                yaw=0.0,
                speed=0.0,
                length=4.5,
                width=1.9,
            ),
        ),
    )

    actions = controller.compute_actions(
        observation=observation,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert len(actions) == 3
    for action in actions:
        assert np.isfinite(action.accel)
        assert np.isfinite(action.steer)
        assert config.simulation.bounds.accel_min <= action.accel <= config.simulation.bounds.accel_max
        assert config.simulation.bounds.steer_min <= action.steer <= config.simulation.bounds.steer_max
    assert abs(actions[0].steer) > 1e-3 or actions[0].accel < 0.0


@pytest.mark.parametrize("baseline_name", ["dwa.yaml", "orca.yaml"])
def test_baseline_override_configs_load(tmp_path: Path, baseline_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    composed_path = tmp_path / f"test_{baseline_name}"
    composed_path.write_text(
        yaml.safe_dump(
            {
                "extends": [
                    str(repo_root / "configs" / "scenarios" / "s1_local_minima.yaml"),
                    str(repo_root / "configs" / "baselines" / baseline_name),
                ]
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_config(composed_path)

    assert config.controller.kind == baseline_name.removesuffix(".yaml")
