"""风险自适应 APF 控制器测试。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
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


def _make_stage5_observation(*, step_index: int, time: float, leader_state: State) -> Observation:
    """Construct a compact stage5-style staggered-blocker observation for leader regression tests."""

    controller = _make_controller()
    return Observation(
        step_index=step_index,
        time=time,
        states=(
            leader_state,
            State(x=leader_state.x - 7.9, y=-1.05, yaw=0.08, speed=max(leader_state.speed, 1.0)),
            State(x=leader_state.x - 15.8, y=-0.84, yaw=0.01, speed=max(leader_state.speed - 0.1, 0.8)),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )


def _pre_release_staggered_cap(
    controller: AdaptiveAPFController,
    *,
    observation: Observation,
    state: State,
    mode: str,
    base_target_speed: float,
) -> float:
    """Reconstruct the pre-release staggered cap for release-regression tests."""

    front_obstacles = controller._leader_front_obstacles(observation, state)
    side_sign = controller._leader_behavior_side_sign(
        observation,
        state,
        mode,
        front_obstacles=front_obstacles,
    )
    assert side_sign is not None

    nominal_relevant = controller._leader_relevant_obstacles(
        observation,
        front_obstacles=front_obstacles,
        side_sign=side_sign,
    )
    alternate_relevant = controller._leader_relevant_obstacles(
        observation,
        front_obstacles=front_obstacles,
        side_sign=-side_sign,
    )
    assert nominal_relevant
    assert alternate_relevant

    target_y = controller._leader_behavior_target_y(observation, state, mode)
    assert target_y is not None

    state_front_x = state.x + 0.5 * controller.config.vehicle_length
    nominal_rear_gap = min(
        obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in nominal_relevant
    )
    alternate_rear_gap = min(
        obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in alternate_relevant
    )
    lateral_error = abs(float(target_y - state.y))
    lateral_dead_zone = 0.25 * controller.config.vehicle_width
    lateral_window = max(
        1.5 * controller.config.vehicle_width,
        0.5 * observation.road.half_width,
    )
    engage_distance = max(
        0.25 * controller.config.obstacle_influence_distance,
        1.0 * controller.config.vehicle_length,
    )
    alternate_lookahead = max(
        0.5 * controller.config.obstacle_influence_distance,
        1.5 * controller.config.vehicle_length,
    )
    a_nominal_gap = float(
        max(min((engage_distance - nominal_rear_gap) / max(engage_distance, 1e-6), 1.0), 0.0)
    )
    a_alternate_gap = float(
        max(
            min(
                (alternate_lookahead - alternate_rear_gap) / max(alternate_lookahead, 1e-6),
                1.0,
            ),
            0.0,
        )
    )
    a_lateral = float(
        max(
            min(
                (lateral_error - lateral_dead_zone) / max(lateral_window - lateral_dead_zone, 1e-6),
                1.0,
            ),
            0.0,
        )
    )
    a_staggered = a_nominal_gap * a_alternate_gap * a_lateral

    min_gap = min(max(nominal_rear_gap, 0.0), max(alternate_rear_gap, 0.0))
    max_brake = max(abs(controller.bounds.accel_min), 1e-6)
    staggered_gap_speed = math.sqrt(max(0.0, 2.0 * max_brake * min_gap))
    blend = float(max(min(a_staggered * 8.0, 0.80), 0.0))
    crawl_floor = float(
        max(min(0.30 + 0.25 * max(state.speed - 0.35, 0.0), 0.70), 0.30)
    )
    return max(
        crawl_floor,
        (1.0 - blend) * base_target_speed + blend * staggered_gap_speed,
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


def test_adaptive_apf_records_leader_nominal_diagnostics() -> None:
    """Leader nominal diagnostics should expose risk, cap chain, and force decomposition."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=60,
        time=6.0,
        leader_state=State(x=26.45, y=-0.28, yaw=0.02, speed=1.05),
    )

    controller.compute_actions(
        observation=observation,
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )
    diagnostics = controller.consume_step_diagnostics()

    assert diagnostics.leader_risk_score > 0.0
    assert diagnostics.leader_base_reference_speed >= diagnostics.leader_hazard_speed_cap
    assert diagnostics.leader_hazard_speed_cap >= diagnostics.leader_release_speed_cap
    assert diagnostics.leader_staggered_speed_cap <= diagnostics.leader_hazard_speed_cap
    assert diagnostics.leader_release_speed_cap >= diagnostics.leader_staggered_speed_cap
    assert diagnostics.leader_target_speed <= diagnostics.leader_release_speed_cap
    assert diagnostics.leader_staggered_activation >= 0.0
    assert diagnostics.leader_edge_hold_activation >= 0.0

    force_sum = (
        np.asarray(diagnostics.leader_force.attractive, dtype=float)
        + np.asarray(diagnostics.leader_force.formation, dtype=float)
        + np.asarray(diagnostics.leader_force.consensus, dtype=float)
        + np.asarray(diagnostics.leader_force.road, dtype=float)
        + np.asarray(diagnostics.leader_force.obstacle, dtype=float)
        + np.asarray(diagnostics.leader_force.peer, dtype=float)
        + np.asarray(diagnostics.leader_force.behavior, dtype=float)
        + np.asarray(diagnostics.leader_force.guidance, dtype=float)
        + np.asarray(diagnostics.leader_force.escape, dtype=float)
    )
    assert np.allclose(force_sum, np.asarray(diagnostics.leader_force.total, dtype=float))


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

    assert abs(float(shaped_obstacle_force[0])) < abs(float(raw_obstacle_force[0]))
    assert shaped_obstacle_force[0] > raw_obstacle_force[0]
    assert shaped_obstacle_force[1] > raw_obstacle_force[1]
    assert actions[0].steer > 0.05


def test_adaptive_apf_keeps_full_leader_reference_speed_before_local_flip() -> None:
    """Early hazard motion should not be throttled when the leader is still aligned with the nominal side."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=48,
        time=4.8,
        leader_state=State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
    )

    reference_speed = controller._reference_speed(
        observation,
        index=0,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert reference_speed == pytest.approx(controller.target_speed)


def test_adaptive_apf_preflip_target_y_starts_shifting_before_hard_local_flip() -> None:
    """Just before the hard local flip, the leader target_y should already start drifting toward the alternate corridor."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=55,
        time=5.5,
        leader_state=State(x=24.813, y=-0.623, yaw=-0.22, speed=1.70),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    front_obstacles = controller._leader_front_obstacles(observation, observation.states[0])
    nominal_side_sign = controller._mode_behavior_side_sign(mode)
    assert nominal_side_sign == -1.0
    assert (
        controller._leader_behavior_side_sign(
            observation,
            observation.states[0],
            mode,
            front_obstacles=front_obstacles,
        )
        == nominal_side_sign
    )

    nominal_target_y = controller._leader_side_target_y(
        observation=observation,
        state=observation.states[0],
        mode=mode,
        front_obstacles=front_obstacles,
        side_sign=nominal_side_sign,
    )
    alternate_target_y = controller._leader_side_target_y(
        observation=observation,
        state=observation.states[0],
        mode=mode,
        front_obstacles=front_obstacles,
        side_sign=-nominal_side_sign,
        apply_flip_overshoot=False,
    )
    target_y = controller._leader_behavior_target_y(
        observation,
        observation.states[0],
        mode,
    )

    assert nominal_target_y is not None
    assert alternate_target_y is not None
    assert target_y is not None
    assert target_y > nominal_target_y + 0.20
    assert target_y < alternate_target_y


def test_adaptive_apf_preflip_dual_blocker_window_reduces_leader_reference_speed() -> None:
    """The dual-blocker preflip window should start throttling the leader before the hard local flip."""

    controller = _make_controller()
    dual_observation = _make_stage5_observation(
        step_index=55,
        time=5.5,
        leader_state=State(x=24.813, y=-0.623, yaw=-0.22, speed=1.70),
    )
    single_observation = Observation(
        step_index=55,
        time=5.5,
        states=dual_observation.states,
        road=dual_observation.road,
        goal_x=dual_observation.goal_x,
        desired_offsets=dual_observation.desired_offsets,
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("far_right", 70.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    dual_front_obstacles = controller._leader_front_obstacles(dual_observation, dual_observation.states[0])
    dual_target_y = controller._leader_behavior_target_y(
        dual_observation,
        dual_observation.states[0],
        mode,
    )
    dual_activation = controller._leader_staggered_hazard_activation(
        observation=dual_observation,
        state=dual_observation.states[0],
        mode=mode,
        front_obstacles=dual_front_obstacles,
        target_y=dual_target_y,
    )
    dual_reference_speed = controller._reference_speed(
        dual_observation,
        index=0,
        mode=mode,
    )
    single_reference_speed = controller._reference_speed(
        single_observation,
        index=0,
        mode=mode,
    )

    assert dual_activation > 0.05
    assert dual_reference_speed < 2.0
    assert dual_reference_speed < single_reference_speed
    assert single_reference_speed == pytest.approx(controller.target_speed)


def test_adaptive_apf_staggered_activation_stays_zero_before_competition_window_opens() -> None:
    """Early hazard motion should not trigger the staggered dual-blocker speed governor."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=48,
        time=4.8,
        leader_state=State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    front_obstacles = controller._leader_front_obstacles(observation, observation.states[0])
    target_y = controller._leader_behavior_target_y(
        observation,
        observation.states[0],
        mode,
    )
    activation = controller._leader_staggered_hazard_activation(
        observation=observation,
        state=observation.states[0],
        mode=mode,
        front_obstacles=front_obstacles,
        target_y=target_y,
    )
    reference_speed = controller._reference_speed(
        observation,
        index=0,
        mode=mode,
    )

    assert activation == pytest.approx(0.0)
    assert reference_speed == pytest.approx(controller.target_speed)


def test_adaptive_apf_leader_reference_speed_throttles_during_staggered_hazard_reorientation() -> None:
    """Once the staggered blocker forces a large lateral reorientation, the leader should stop commanding cruise speed."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=56,
        time=5.6,
        leader_state=State(x=25.02, y=-0.77, yaw=-0.27, speed=1.58),
    )

    reference_speed = controller._reference_speed(
        observation,
        index=0,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )
    actions = controller.compute_actions(
        observation,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert reference_speed < 3.0
    assert actions[0].accel < 0.5


def test_adaptive_apf_caps_low_speed_hazard_braking_before_self_stop() -> None:
    """Near stop, leader hazard control should keep a tiny crawl without switching to positive nominal accel."""

    controller = _make_controller()
    observation = Observation(
        step_index=75,
        time=7.5,
        states=(
            State(x=26.630610145654906, y=-0.7868884432306016, yaw=0.06153123751741285, speed=0.04279428232288743),
            State(x=18.730610145654907, y=-1.05, yaw=0.08, speed=1.0),
            State(x=10.830610145654905, y=-0.84, yaw=0.01, speed=0.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    actions = controller.compute_actions(
        observation,
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )

    assert -0.031 <= actions[0].accel <= 1e-9
    assert actions[0].steer > 0.05


def test_adaptive_apf_overtake_guidance_overcomes_road_push_during_incomplete_lane_shift() -> None:
    """In the S4-style overtake slice, hazard guidance should keep the leader turning toward the active bypass lane."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "scenarios" / "s4_overtake_interaction.yaml")
    controller = AdaptiveAPFController(
        config=config.controller,
        bounds=config.simulation.bounds,
        road=Road(config.scenario.road),
        target_speed=config.simulation.target_speed,
    )
    observation = Observation(
        step_index=54,
        time=5.4,
        states=(
            State(x=25.36, y=-1.67, yaw=0.0, speed=2.38),
            State(x=17.36, y=-1.60, yaw=0.0, speed=2.20),
            State(x=9.36, y=-1.55, yaw=0.0, speed=2.10),
        ),
        road=controller.road.geometry,
        goal_x=config.scenario.goal_x,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("slow_lead", 32.8, 0.0, 0.0, 2.0, 4.8, 2.0),
            ObstacleState("side_block", 58.7, 1.8, 0.0, 4.2, 4.6, 2.0),
        ),
    )

    action = controller.compute_actions(
        observation,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )[0]

    assert action.steer < -0.05


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


def test_nonrelevant_shaping_preserves_relevant_obstacle_force() -> None:
    """Relevant obstacles must never be shaped — shaping factor r must equal 0."""

    controller = _make_controller()
    # Obstacle on the left (y=2.0) is relevant when side_sign > 0 (yield_left)
    # because it sits at center_y ± relevant_threshold.
    observation = Observation(
        step_index=60,
        time=6.0,
        states=(
            State(x=25.0, y=-0.5, yaw=0.0, speed=1.5),
            State(x=17.0, y=-1.0, yaw=0.0, speed=1.4),
            State(x=9.0, y=-0.8, yaw=0.0, speed=1.3),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            # This obstacle is near center_y (y=-0.3), relevant for yield_left (side_sign=+1)
            ObstacleState("center_blocker", 30.0, -0.3, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"

    raw_force = controller._sum_obstacle_forces(
        observation=observation, index=0, repulsive_gain=17.5,
    )
    shaped_force = controller._adaptive_obstacle_force(
        observation=observation, index=0, mode=mode, repulsive_gain=17.5,
    )

    # Relevant obstacle → no shaping at all
    assert shaped_force[0] == pytest.approx(raw_force[0], abs=1e-9)
    assert shaped_force[1] == pytest.approx(raw_force[1], abs=1e-9)


def test_nonrelevant_shaping_skips_aligned_lateral_push() -> None:
    """When a nonrelevant obstacle's lateral push already favours the bypass side, shaping must not activate."""

    controller = _make_controller()
    # Obstacle far above (y=3.0) pushes the leader downward (negative y force).
    # With yield_right (side_sign=-1), downward push is ALIGNED → no shaping needed.
    observation = Observation(
        step_index=62,
        time=6.2,
        states=(
            State(x=25.0, y=0.0, yaw=0.0, speed=1.5),
            State(x=17.0, y=-1.0, yaw=0.0, speed=1.4),
            State(x=9.0, y=-0.8, yaw=0.0, speed=1.3),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper_blocker", 30.0, 3.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    raw_force = controller._sum_obstacle_forces(
        observation=observation, index=0, repulsive_gain=17.5,
    )
    shaped_force = controller._adaptive_obstacle_force(
        observation=observation, index=0, mode=mode, repulsive_gain=17.5,
    )

    # Aligned push → identity
    assert shaped_force[0] == pytest.approx(raw_force[0], abs=1e-9)
    assert shaped_force[1] == pytest.approx(raw_force[1], abs=1e-9)


def test_nonrelevant_shaping_bounded_total_reduction() -> None:
    """Total reduction must stay ≤ 0.90, and force_x must be exactly preserved."""

    controller = _make_controller()
    state = State(x=25.0, y=-0.7, yaw=-0.2, speed=1.4)
    # Very far lateral obstacle that is nonrelevant: y=3.0, leader at y=-0.7
    obstacle = ObstacleState("far_upper", 30.0, 3.0, 0.0, 0.0, 4.8, 2.0)

    raw_force = controller._obstacle_force(state, obstacle, repulsive_gain=17.5)

    # side_sign = +1 (yield_left), force_y from upper obstacle is negative (pushes down) → adverse
    shaped_force = controller._shape_leader_nonrelevant_obstacle_force(
        state=state,
        obstacle=obstacle,
        side_sign=1.0,
        force=raw_force,
    )

    # force_x exactly preserved
    assert shaped_force[0] == pytest.approx(float(raw_force[0]), abs=1e-9)
    # force_y magnitude reduced but not zeroed (reduction ≤ 0.90 means ≥ 10% remains)
    assert abs(float(shaped_force[1])) >= 0.10 * abs(float(raw_force[1])) - 1e-9
    # force_y magnitude strictly less than raw (shaping did activate)
    if abs(float(raw_force[1])) > 1e-9:
        assert abs(float(shaped_force[1])) < abs(float(raw_force[1]))


def test_adaptive_apf_staggered_dual_blocker_further_throttles_leader_speed() -> None:
    """When both nominal-side and alternate-side blockers compete in the staggered corridor
    and lateral reorientation is incomplete, the staggered governor must further reduce speed
    below the single-blocker hazard limit."""

    controller = _make_controller()
    # Leader deep inside the staggered zone with significant lateral offset
    observation = _make_stage5_observation(
        step_index=58,
        time=5.8,
        leader_state=State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"

    # Get single-stage hazard limit (without staggered governor)
    hazard_only_speed = controller._leader_hazard_speed_limit(
        observation=observation,
        state=observation.states[0],
        mode=mode,
        base_target_speed=controller.target_speed,
    )
    # Get full reference speed (hazard limit + staggered governor)
    full_reference_speed = controller._reference_speed(observation, index=0, mode=mode)

    # Staggered governor must further reduce below the hazard-only limit
    assert full_reference_speed < hazard_only_speed, (
        f"staggered governor should reduce speed: {full_reference_speed} < {hazard_only_speed}"
    )
    # A still-rolling leader should stay above the hard crawl floor.
    assert full_reference_speed > 0.30


def test_adaptive_apf_staggered_longitudinal_relief_bounded() -> None:
    """The staggered longitudinal relief must stay finite and bounded."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=58,
        time=5.8,
        leader_state=State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"
    front_obstacles = controller._leader_front_obstacles(observation, observation.states[0])
    side_sign = controller._leader_behavior_side_sign(
        observation,
        observation.states[0],
        mode,
        front_obstacles=front_obstacles,
    )
    assert side_sign is not None

    relief = controller._leader_staggered_longitudinal_relief(
        observation=observation,
        state=observation.states[0],
        mode=mode,
        front_obstacles=front_obstacles,
        side_sign=side_sign,
    )

    assert 0.0 < relief <= 0.98


def test_adaptive_apf_staggered_longitudinal_relief_inactive_single_blocker() -> None:
    """The staggered longitudinal relief must not activate without dual blockers."""

    controller = _make_controller()
    single_observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.46, y=-1.05, yaw=0.08, speed=2.0),
            State(x=7.56, y=-0.84, yaw=0.01, speed=1.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("single_blocker", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"
    front_obstacles = controller._leader_front_obstacles(single_observation, single_observation.states[0])
    side_sign = controller._leader_behavior_side_sign(
        single_observation,
        single_observation.states[0],
        mode,
        front_obstacles=front_obstacles,
    )
    assert side_sign is not None

    relief = controller._leader_staggered_longitudinal_relief(
        observation=single_observation,
        state=single_observation.states[0],
        mode=mode,
        front_obstacles=front_obstacles,
        side_sign=side_sign,
    )

    assert relief == pytest.approx(0.0)


def test_adaptive_apf_staggered_lateral_boost_bounded() -> None:
    """The staggered lateral boost must remain finite and bounded."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=58,
        time=5.8,
        leader_state=State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"

    boost = controller._leader_staggered_lateral_boost(
        observation=observation,
        state=observation.states[0],
        mode=mode,
    )

    assert 1.0 <= boost <= 3.0


def test_adaptive_apf_staggered_lateral_boost_inactive_single_blocker() -> None:
    """The staggered lateral boost must stay inactive without dual-blocker geometry."""

    controller = _make_controller()
    single_observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.46, y=-1.05, yaw=0.08, speed=2.0),
            State(x=7.56, y=-0.84, yaw=0.01, speed=1.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("single_blocker", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    boost = controller._leader_staggered_lateral_boost(
        observation=single_observation,
        state=single_observation.states[0],
        mode=mode,
    )

    assert boost == pytest.approx(1.0)


def test_adaptive_apf_staggered_steer_bias_bounded() -> None:
    """The staggered steer bias must remain finite, signed, and bounded."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=58,
        time=5.8,
        leader_state=State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"
    target_y = float(controller._leader_goal_target(observation, observation.states[0], mode)[1])
    boost = controller._leader_staggered_lateral_boost(
        observation=observation,
        state=observation.states[0],
        mode=mode,
    )

    steer_bias = controller._leader_staggered_steer_bias(
        state=observation.states[0],
        target_y=target_y,
        boost=boost,
    )

    assert 0.0 < steer_bias <= 0.16


def test_adaptive_apf_staggered_steer_bias_inactive_single_blocker() -> None:
    """The staggered steer bias must stay inactive without dual-blocker geometry."""

    controller = _make_controller()
    single_observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.46, y=-1.05, yaw=0.08, speed=2.0),
            State(x=7.56, y=-0.84, yaw=0.01, speed=1.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("single_blocker", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"
    target_y = float(controller._leader_goal_target(single_observation, single_observation.states[0], mode)[1])
    boost = controller._leader_staggered_lateral_boost(
        observation=single_observation,
        state=single_observation.states[0],
        mode=mode,
    )

    steer_bias = controller._leader_staggered_steer_bias(
        state=single_observation.states[0],
        target_y=target_y,
        boost=boost,
    )

    assert steer_bias == pytest.approx(0.0)


def test_adaptive_apf_staggered_governor_inactive_without_dual_blockers() -> None:
    """When only one side has obstacles (no staggered geometry), the staggered governor
    must not reduce the cruise speed — protecting non-staggered scenarios."""

    controller = _make_controller()
    # Leader approaching a SINGLE blocker (no alternate-side obstacle)
    single_blocker_observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.46, y=-1.05, yaw=0.08, speed=2.0),
            State(x=7.56, y=-0.84, yaw=0.01, speed=1.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            # Only one blocker — no staggered geometry
            ObstacleState("single_blocker", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"

    # Get hazard-limited speed
    hazard_only = controller._leader_hazard_speed_limit(
        observation=single_blocker_observation,
        state=single_blocker_observation.states[0],
        mode=mode,
        base_target_speed=controller.target_speed,
    )
    # Get full reference speed
    full_speed = controller._reference_speed(
        single_blocker_observation, index=0, mode=mode,
    )

    # Staggered governor must NOT reduce speed when only one blocker exists
    assert full_speed == pytest.approx(hazard_only, abs=1e-9), (
        f"staggered governor should be inactive: {full_speed} vs {hazard_only}"
    )


def test_adaptive_apf_relocked_edge_release_lifts_staggered_cap_but_stays_bounded() -> None:
    """Relocked edge-hold should lift the old staggered cap without restoring cruise speed."""

    controller = _make_controller()
    observation = _make_stage5_observation(
        step_index=58,
        time=5.8,
        leader_state=State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
    )
    mode = "topology=diamond|behavior=yield_left|gain=cautious"
    state = observation.states[0]
    front_obstacles = controller._leader_front_obstacles(observation, state)
    side_sign = controller._leader_behavior_side_sign(
        observation,
        state,
        mode,
        front_obstacles=front_obstacles,
    )
    assert side_sign is not None
    relevant_obstacles = controller._leader_relevant_obstacles(
        observation,
        front_obstacles=front_obstacles,
        side_sign=side_sign,
    )
    base_target_speed = controller._leader_hazard_speed_limit(
        observation=observation,
        state=state,
        mode=mode,
        base_target_speed=controller.target_speed,
    )
    staggered_cap = _pre_release_staggered_cap(
        controller,
        observation=observation,
        state=state,
        mode=mode,
        base_target_speed=base_target_speed,
    )

    released_cap = controller._leader_relocked_edge_speed_release(
        observation=observation,
        state=state,
        mode=mode,
        front_obstacles=front_obstacles,
        relevant_obstacles=relevant_obstacles,
        side_sign=side_sign,
        base_target_speed=base_target_speed,
        staggered_cap=staggered_cap,
    )

    assert released_cap > staggered_cap
    assert released_cap <= base_target_speed
    assert released_cap <= 1.20


def test_adaptive_apf_relocked_edge_release_degrades_exactly_when_inactive() -> None:
    """Without edge-hold activation, the release helper must return the old cap exactly."""

    controller = _make_controller()
    observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.46, y=-1.05, yaw=0.08, speed=2.0),
            State(x=7.56, y=-0.84, yaw=0.01, speed=1.8),
        ),
        road=controller.road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("single_blocker", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
        ),
    )
    mode = "topology=diamond|behavior=yield_right|gain=cautious"
    state = observation.states[0]
    front_obstacles = controller._leader_front_obstacles(observation, state)
    side_sign = controller._leader_behavior_side_sign(
        observation,
        state,
        mode,
        front_obstacles=front_obstacles,
    )
    assert side_sign is not None
    relevant_obstacles = controller._leader_relevant_obstacles(
        observation,
        front_obstacles=front_obstacles,
        side_sign=side_sign,
    )
    staggered_cap = 0.92

    released_cap = controller._leader_relocked_edge_speed_release(
        observation=observation,
        state=state,
        mode=mode,
        front_obstacles=front_obstacles,
        relevant_obstacles=relevant_obstacles,
        side_sign=side_sign,
        base_target_speed=1.60,
        staggered_cap=staggered_cap,
    )

    assert released_cap == staggered_cap
