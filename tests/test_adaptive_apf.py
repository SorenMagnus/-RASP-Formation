"""风险自适应 APF 控制器测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apflf.controllers.adaptive_apf import AdaptiveAPFController
from apflf.controllers.base import build_controller
from apflf.decision.mode_base import StaticModeDecision
from apflf.env.dynamics import VehicleDynamics
from apflf.env.road import Road
from apflf.env.scenarios import ScenarioFactory
from apflf.safety.safety_filter import PassThroughSafetyFilter
from apflf.sim.world import World
from apflf.utils.config import load_config
from apflf.utils.types import ControllerConfig, InputBounds, Observation, ObstacleState, RoadGeometry, State


def _make_controller_config(kind: str = "adaptive_apf") -> ControllerConfig:
    """构造一个用于单测的控制器配置。"""

    return ControllerConfig(
        kind=kind,
        vehicle_length=4.8,
        vehicle_width=1.9,
        speed_gain=0.8,
        gap_gain=0.35,
        lateral_gain=0.22,
        heading_gain=0.65,
        attraction_gain=1.15,
        repulsive_gain=14.0,
        road_gain=8.0,
        formation_gain=1.2,
        consensus_gain=0.25,
        obstacle_influence_distance=15.0,
        vehicle_influence_distance=10.0,
        road_influence_margin=1.2,
        st_velocity_gain=0.7,
        ttc_gain=0.8,
        ttc_threshold=4.0,
        risk_distance_scale=12.0,
        risk_speed_scale=4.0,
        risk_ttc_threshold=5.0,
        risk_sigmoid_slope=4.0,
        risk_reference=0.45,
        adaptive_alpha=1.2,
        repulsive_gain_min=8.0,
        repulsive_gain_max=32.0,
        road_gain_min=3.0,
        road_gain_max=15.0,
        stagnation_speed_threshold=0.4,
        stagnation_progress_threshold=0.03,
        stagnation_force_threshold=0.5,
        stagnation_steps=3,
        stagnation_cooldown_steps=2,
    )


def _make_controller() -> AdaptiveAPFController:
    """构造一个用于单测的自适应控制器实例。"""

    return AdaptiveAPFController(
        config=_make_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)),
        target_speed=8.0,
    )


@pytest.mark.parametrize(
    "kind",
    ["formation_cruise", "apf", "st_apf", "apf_lf", "adaptive_apf", "dwa", "orca"],
)
def test_controller_factory_supports_all_nominal_kinds(kind: str) -> None:
    """控制器工厂应支持当前实现的所有名义控制器类型。"""

    controller = build_controller(
        config=_make_controller_config(kind),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)),
        target_speed=8.0,
    )

    assert controller is not None


def test_schedule_gains_are_bounded_and_monotonic() -> None:
    """风险调度应随风险升高而单调增加，且全程有界。"""

    controller = _make_controller()

    low_rep, low_road = controller.schedule_gains(0.05)
    high_rep, high_road = controller.schedule_gains(0.95)

    assert controller.config.repulsive_gain_min <= low_rep <= controller.config.repulsive_gain_max
    assert controller.config.repulsive_gain_min <= high_rep <= controller.config.repulsive_gain_max
    assert controller.config.road_gain_min <= low_road <= controller.config.road_gain_max
    assert controller.config.road_gain_min <= high_road <= controller.config.road_gain_max
    assert high_rep > low_rep
    assert high_road > low_road


def test_risk_score_increases_when_situation_worsens() -> None:
    """间距更小、闭合速度更大、TTC 更低时，风险分数应更高。"""

    controller = _make_controller()

    safe_score = controller.compute_risk_score(
        clearance=20.0,
        closing_speed=0.1,
        ttc=float("inf"),
        boundary_margin=2.5,
    )
    dangerous_score = controller.compute_risk_score(
        clearance=1.5,
        closing_speed=3.0,
        ttc=0.8,
        boundary_margin=0.2,
    )

    assert 0.0 <= safe_score <= 1.0
    assert 0.0 <= dangerous_score <= 1.0
    assert dangerous_score > safe_score


def test_stagnation_detection_has_basic_anti_chattering() -> None:
    """持续低进度状态应触发停滞，随后进入冷却期。"""

    controller = _make_controller()

    assert controller.update_stagnation(progress_delta=0.0, speed=0.0, force_norm=0.0) is False
    assert controller.update_stagnation(progress_delta=0.0, speed=0.0, force_norm=0.0) is False
    assert controller.update_stagnation(progress_delta=0.0, speed=0.0, force_norm=0.0) is True
    assert controller.update_stagnation(progress_delta=0.0, speed=0.0, force_norm=0.0) is False


def test_adaptive_apf_tapers_nonrelevant_obstacle_lateral_push_after_local_relock() -> None:
    """The leader should keep steering toward the relocked side in the stage58 staggered-blocker geometry."""

    controller = _make_controller()
    observation = Observation(
        step_index=58,
        time=5.8,
        states=(
            State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
            State(x=17.40, y=-1.05, yaw=0.08, speed=1.55),
            State(x=9.50, y=-0.84, yaw=0.01, speed=1.40),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"

    raw_obstacle_force = controller._sum_obstacle_forces(
        observation=observation,
        index=0,
        repulsive_gain=17.5,
    )
    shaped_obstacle_force = controller._adaptive_obstacle_force(
        observation=observation,
        index=0,
        mode=mode,
        repulsive_gain=17.5,
    )
    actions = controller.compute_actions(observation, mode=mode)

    assert shaped_obstacle_force[0] == pytest.approx(raw_obstacle_force[0])
    assert shaped_obstacle_force[1] > raw_obstacle_force[1]
    assert actions[0].steer > 0.05


def test_scenario_config_extends_and_adaptive_controller_runs_one_step() -> None:
    """典型障碍场景配置应能通过 extends 加载并完成一步仿真。"""

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "scenarios"
        / "s1_local_minima.yaml"
    )
    config = load_config(config_path)
    assert config.controller.kind == "adaptive_apf"
    assert len(config.scenario.obstacles) >= 1

    scenario = ScenarioFactory(config).build(seed=0)
    road = Road(scenario.road)
    controller = build_controller(
        config=config.controller,
        bounds=config.simulation.bounds,
        road=road,
        target_speed=config.simulation.target_speed,
    )
    world = World(
        scenario=scenario,
        dynamics=VehicleDynamics(
            wheelbase=config.simulation.wheelbase,
            bounds=config.simulation.bounds,
        ),
        controller=controller,
        mode_decision=StaticModeDecision(default_mode=config.decision.default_mode),
        safety_filter=PassThroughSafetyFilter(),
        dt=config.simulation.dt,
    )

    snapshot = world.step()

    assert len(snapshot.states) == config.scenario.vehicle_count
    assert len(snapshot.obstacles) == len(config.scenario.obstacles)
    assert snapshot.time == pytest.approx(config.simulation.dt)


def test_scenario_and_baseline_overrides_can_be_composed(tmp_path: Path) -> None:
    """场景配置与基线覆盖层应能组合出完整可运行配置。"""

    repo_root = Path(__file__).resolve().parents[1]
    composed_path = tmp_path / "s1_apf.yaml"
    composed_path.write_text(
        yaml.safe_dump(
            {
                "extends": [
                    str(repo_root / "configs" / "scenarios" / "s1_local_minima.yaml"),
                    str(repo_root / "configs" / "baselines" / "apf.yaml"),
                ],
                "experiment": {"name": "test_s1_apf"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = load_config(composed_path)

    assert config.controller.kind == "apf"
    assert len(config.scenario.obstacles) == 3


@pytest.mark.parametrize(
    "scenario_name",
    [
        "s1_local_minima.yaml",
        "s2_dynamic_crossing.yaml",
        "s3_narrow_passage.yaml",
        "s4_overtake_interaction.yaml",
        "s5_dense_multi_agent.yaml",
    ],
)
def test_all_scenario_configs_load_and_step(scenario_name: str) -> None:
    """所有典型障碍场景都应能完成最小一步仿真。"""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "scenarios" / scenario_name)
    scenario = ScenarioFactory(config).build(seed=0)
    road = Road(scenario.road)
    controller = build_controller(
        config=config.controller,
        bounds=config.simulation.bounds,
        road=road,
        target_speed=config.simulation.target_speed,
    )
    world = World(
        scenario=scenario,
        dynamics=VehicleDynamics(
            wheelbase=config.simulation.wheelbase,
            bounds=config.simulation.bounds,
        ),
        controller=controller,
        mode_decision=StaticModeDecision(default_mode=config.decision.default_mode),
        safety_filter=PassThroughSafetyFilter(),
        dt=config.simulation.dt,
    )

    snapshot = world.step()

    assert len(snapshot.obstacles) == len(config.scenario.obstacles)
    assert len(snapshot.states) == config.scenario.vehicle_count
