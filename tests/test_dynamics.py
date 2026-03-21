"""动力学与障碍物模型测试。"""

from __future__ import annotations

import math

import pytest

from apflf.env.dynamics import VehicleDynamics
from apflf.env.geometry import normalize_angle
from apflf.env.obstacles import build_obstacle_models, sample_obstacles
from apflf.utils.types import Action, InputBounds, ObstacleConfig, State


def test_vehicle_dynamics_clips_inputs_and_speed_floor() -> None:
    """动力学应裁剪输入并保持速度非负。"""

    dynamics = VehicleDynamics(
        wheelbase=2.5,
        bounds=InputBounds(
            accel_min=-1.0,
            accel_max=2.0,
            steer_min=-0.3,
            steer_max=0.3,
            speed_min=0.0,
            speed_max=10.0,
        ),
    )
    state = State(x=0.0, y=0.0, yaw=0.0, speed=1.0)
    action = Action(accel=-10.0, steer=1.0)

    next_state = dynamics.step(state=state, action=action, dt=1.0)

    assert next_state.x == pytest.approx(1.0)
    assert next_state.y == pytest.approx(0.0)
    assert next_state.speed == pytest.approx(0.0)
    assert next_state.yaw == pytest.approx(1.0 * math.tan(0.3) / 2.5)


def test_vehicle_dynamics_normalizes_heading() -> None:
    """积分后的航向角应始终归一化到 [-pi, pi]。"""

    dynamics = VehicleDynamics(
        wheelbase=2.0,
        bounds=InputBounds(
            accel_min=-2.0,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=20.0,
        ),
    )
    state = State(x=0.0, y=0.0, yaw=math.pi - 0.05, speed=4.0)
    action = Action(accel=0.0, steer=0.5)

    next_state = dynamics.step(state=state, action=action, dt=1.0)

    expected_yaw = normalize_angle((math.pi - 0.05) + 4.0 * math.tan(0.5) / 2.0)
    assert -math.pi <= next_state.yaw <= math.pi
    assert next_state.yaw == pytest.approx(expected_yaw)


def test_obstacle_models_support_static_and_constant_velocity() -> None:
    """障碍物模型应能稳定输出静态和匀速轨迹。"""

    obstacle_models = build_obstacle_models(
        (
            ObstacleConfig(
                obstacle_id="static_0",
                motion_model="static",
                x=10.0,
                y=1.0,
                yaw=0.2,
                speed=3.0,
                length=4.0,
                width=2.0,
            ),
            ObstacleConfig(
                obstacle_id="dynamic_0",
                motion_model="constant_velocity",
                x=0.0,
                y=-2.0,
                yaw=math.pi / 2.0,
                speed=2.0,
                length=4.5,
                width=1.8,
            ),
        )
    )

    states = sample_obstacles(obstacle_models, time=1.5)

    assert states[0].x == pytest.approx(10.0)
    assert states[0].y == pytest.approx(1.0)
    assert states[0].speed == pytest.approx(0.0)
    assert states[1].x == pytest.approx(0.0, abs=1e-12)
    assert states[1].y == pytest.approx(1.0)
    assert states[1].speed == pytest.approx(2.0)
