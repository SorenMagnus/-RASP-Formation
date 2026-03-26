"""Tests for FSM mode decision and mode-aware nominal control."""

from __future__ import annotations

import pytest

from apflf.controllers.apf_lf import APFLFController
from apflf.decision.fsm_mode import FSMModeDecision
from apflf.decision.mode_base import DEFAULT_MODE_LABEL, parse_mode_label
from apflf.env.road import Road
from apflf.utils.types import (
    ControllerConfig,
    DecisionConfig,
    InputBounds,
    Observation,
    ObstacleState,
    RoadGeometry,
    State,
)


def _decision_config(*, hysteresis_steps: int = 2, stagnation_steps: int = 4, recover_exit_steps: int = 4) -> DecisionConfig:
    return DecisionConfig(
        kind="fsm",
        default_mode=DEFAULT_MODE_LABEL,
        hysteresis_steps=hysteresis_steps,
        risk_threshold_enter=0.55,
        risk_threshold_exit=0.30,
        clearance_threshold=8.0,
        ttc_threshold=4.0,
        boundary_margin_threshold=0.75,
        lookahead_distance=18.0,
        narrow_passage_margin=0.5,
        stagnation_speed_threshold=0.45,
        stagnation_progress_threshold=0.1,
        stagnation_steps=stagnation_steps,
        recover_exit_steps=recover_exit_steps,
    )


def _controller_config() -> ControllerConfig:
    return ControllerConfig(
        kind="apf_lf",
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
        stagnation_steps=6,
        stagnation_cooldown_steps=5,
    )


def _observation(
    *,
    leader_x: float,
    leader_speed: float,
    obstacles: tuple[ObstacleState, ...],
) -> Observation:
    road = RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)
    states = (
        State(x=leader_x, y=0.0, yaw=0.0, speed=leader_speed),
        State(x=leader_x - 8.0, y=0.0, yaw=0.0, speed=leader_speed),
        State(x=leader_x - 16.0, y=0.0, yaw=0.0, speed=leader_speed),
    )
    return Observation(
        step_index=0,
        time=0.0,
        states=states,
        road=road,
        goal_x=80.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=obstacles,
    )


def test_fsm_mode_switches_to_cautious_yield_with_hysteresis() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=2),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    observation = _observation(
        leader_x=0.0,
        leader_speed=5.0,
        obstacles=(
            ObstacleState("upper", 7.0, 1.6, 0.0, 0.0, 4.5, 2.0),
            ObstacleState("lower", 7.5, -1.6, 0.0, 0.0, 4.5, 2.0),
        ),
    )

    first_mode = decision.select_mode(observation)
    second_mode = decision.select_mode(observation)

    assert parse_mode_label(first_mode).behavior == "follow"
    parsed = parse_mode_label(second_mode)
    assert parsed.topology == "diamond"
    assert parsed.behavior.startswith("yield_")
    assert parsed.gain == "cautious"


def test_fsm_lookahead_uses_obstacle_rear_edge_for_long_blockers() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    observation = _observation(
        leader_x=21.0,
        leader_speed=6.0,
        obstacles=(
            ObstacleState("long_blocker", 44.0, -0.9, 0.0, 0.0, 6.0, 2.5),
        ),
    )

    front = decision._front_obstacles(observation, state=observation.states[0])

    assert tuple(obstacle.obstacle_id for obstacle in front) == ("long_blocker",)


def test_fsm_mode_switches_to_escape_after_stagnation() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1, stagnation_steps=3),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    observation = _observation(
        leader_x=2.0,
        leader_speed=0.1,
        obstacles=(
            ObstacleState("blocker", 7.0, 1.4, 0.0, 0.0, 4.5, 2.0),
        ),
    )

    mode = DEFAULT_MODE_LABEL
    for _ in range(5):
        mode = decision.select_mode(observation)

    parsed = parse_mode_label(mode)
    assert parsed.behavior.startswith("escape_")
    assert parsed.gain == "assertive"


def test_fsm_mode_locks_side_until_the_whole_team_clears_obstacles() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    road = RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)
    observation = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=22.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=8.0, y=0.0, yaw=0.0, speed=4.0),
            State(x=-4.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=road,
        goal_x=90.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("front", 41.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )

    first_mode = decision.select_mode(observation)
    second_mode = decision.select_mode(observation)
    assert parse_mode_label(second_mode).behavior == "yield_right"

    leader_cleared_observation = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=48.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=26.0, y=-0.4, yaw=0.0, speed=3.0),
            State(x=14.0, y=-0.8, yaw=0.0, speed=2.5),
        ),
        road=road,
        goal_x=90.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=observation.obstacles,
    )

    locked_mode = decision.select_mode(leader_cleared_observation)
    parsed_locked = parse_mode_label(locked_mode)
    assert parsed_locked.topology == "diamond"
    assert parsed_locked.behavior == "yield_right"


def test_fsm_preferred_side_prioritizes_the_nearest_asymmetric_blocker() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    observation = _observation(
        leader_x=12.0,
        leader_speed=5.0,
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    hazard = decision._assess_hazard(observation)

    assert hazard.preferred_side == "right"


def test_fsm_mode_can_relock_to_the_other_side_once_the_anchor_is_nearly_cleared() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    initial_observation = _observation(
        leader_x=12.0,
        leader_speed=5.0,
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    decision.select_mode(initial_observation)
    yield_right_mode = decision.select_mode(initial_observation)
    assert parse_mode_label(yield_right_mode).behavior == "yield_right"

    relock_observation = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=25.02, y=-0.77, yaw=-0.27, speed=1.58),
            State(x=17.10, y=-1.10, yaw=0.08, speed=1.65),
            State(x=9.30, y=-0.82, yaw=0.01, speed=1.45),
        ),
        road=initial_observation.road,
        goal_x=80.0,
        desired_offsets=initial_observation.desired_offsets,
        obstacles=initial_observation.obstacles,
    )

    decision.select_mode(relock_observation)
    relocked_mode = decision.select_mode(relock_observation)

    assert parse_mode_label(relocked_mode).behavior == "yield_left"


def test_fsm_hazard_side_relock_bypasses_generic_hysteresis() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=3),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    initial_observation = _observation(
        leader_x=12.0,
        leader_speed=5.0,
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    mode = DEFAULT_MODE_LABEL
    for _ in range(4):
        mode = decision.select_mode(initial_observation)
    assert parse_mode_label(mode).behavior == "yield_right"

    relock_observation = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=25.02, y=-0.77, yaw=-0.27, speed=1.58),
            State(x=17.10, y=-1.10, yaw=0.08, speed=1.65),
            State(x=9.30, y=-0.82, yaw=0.01, speed=1.45),
        ),
        road=initial_observation.road,
        goal_x=80.0,
        desired_offsets=initial_observation.desired_offsets,
        obstacles=initial_observation.obstacles,
    )

    relocked_mode = decision.select_mode(relock_observation)

    assert parse_mode_label(relocked_mode).behavior == "yield_left"


def test_fsm_mode_transitions_into_recover_after_hazard_clears() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    hazard_observation = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=22.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=10.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=-2.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5),
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
        ),
    )

    decision.select_mode(hazard_observation)
    yield_mode = decision.select_mode(hazard_observation)
    assert parse_mode_label(yield_mode).behavior == "yield_right"

    recovery_observation = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=46.0, y=-0.4, yaw=0.0, speed=4.0),
            State(x=26.0, y=-0.6, yaw=0.0, speed=3.2),
            State(x=8.0, y=-0.7, yaw=0.0, speed=2.8),
        ),
        road=hazard_observation.road,
        goal_x=95.0,
        desired_offsets=hazard_observation.desired_offsets,
        obstacles=(
            ObstacleState("upper_far", 13.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )

    decision.select_mode(recovery_observation)
    recover_mode = decision.select_mode(recovery_observation)
    parsed_recover = parse_mode_label(recover_mode)
    assert parsed_recover.topology == "line"
    assert parsed_recover.behavior == "recover_right"
    assert parsed_recover.gain == "nominal"


def test_fsm_keeps_recover_active_until_team_recenters() -> None:
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )
    hazard_observation = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=22.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=10.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=-2.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5),
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
        ),
    )

    decision.select_mode(hazard_observation)
    yield_mode = decision.select_mode(hazard_observation)
    assert parse_mode_label(yield_mode).behavior == "yield_right"

    still_offset_observation = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=46.0, y=-1.4, yaw=0.0, speed=4.2),
            State(x=38.0, y=-1.6, yaw=0.0, speed=4.0),
            State(x=30.0, y=-1.5, yaw=0.0, speed=3.8),
        ),
        road=hazard_observation.road,
        goal_x=95.0,
        desired_offsets=hazard_observation.desired_offsets,
        obstacles=(),
    )

    decision.select_mode(still_offset_observation)
    recover_mode = decision.select_mode(still_offset_observation)

    assert parse_mode_label(recover_mode).behavior == "recover_right"


def test_fsm_recover_exit_requires_sustained_satisfaction() -> None:
    """recover→follow 退出需要恢复条件连续满足 N_exit 步。

    如果恢复条件仅满足 1 步就被违背，则 recover 不应退出。
    """
    recover_exit_steps = 3
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1, recover_exit_steps=recover_exit_steps),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )

    road = RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)
    # 1. 先把 FSM 推入 yield 模式
    hazard_obs = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=22.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=10.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=-2.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
        ),
    )
    decision.select_mode(hazard_obs)
    yield_mode = decision.select_mode(hazard_obs)
    assert parse_mode_label(yield_mode).behavior == "yield_right"

    # 2. 推入 recover 模式：queued offset 大，无前方障碍
    still_offset_obs = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=46.0, y=-1.4, yaw=0.0, speed=4.2),
            State(x=38.0, y=-1.6, yaw=0.0, speed=4.0),
            State(x=30.0, y=-1.5, yaw=0.0, speed=3.8),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=hazard_obs.desired_offsets,
        obstacles=(),
    )
    decision.select_mode(still_offset_obs)
    recover_mode = decision.select_mode(still_offset_obs)
    assert parse_mode_label(recover_mode).behavior == "recover_right"

    # 3. 恢复完成条件满足的观测：编队靠近中心线、队友不落后
    recovered_obs = Observation(
        step_index=2,
        time=0.2,
        states=(
            State(x=60.0, y=-0.05, yaw=0.0, speed=5.0),
            State(x=52.0, y=-0.05, yaw=0.0, speed=5.0),
            State(x=44.0, y=-0.05, yaw=0.0, speed=5.0),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=hazard_obs.desired_offsets,
        obstacles=(),
    )

    # 喂 1 步恢复完成的观测
    mode_1 = decision.select_mode(recovered_obs)
    # 应该仍在 recover（连续满足不够）
    assert parse_mode_label(mode_1).behavior.startswith("recover_"), \
        f"Expected recover after 1 satisfied step, got {mode_1}"

    # 打断：再喂一步仍需恢复的观测
    still_offset_obs2 = Observation(
        step_index=3,
        time=0.3,
        states=(
            State(x=64.0, y=-1.4, yaw=0.0, speed=4.2),
            State(x=56.0, y=-1.6, yaw=0.0, speed=4.0),
            State(x=48.0, y=-1.5, yaw=0.0, speed=3.8),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=hazard_obs.desired_offsets,
        obstacles=(),
    )
    mode_interrupted = decision.select_mode(still_offset_obs2)
    assert parse_mode_label(mode_interrupted).behavior.startswith("recover_"), \
        "Expected recover to stay after interruption"

    # 重新连续恢复 N_exit 步
    for step_i in range(recover_exit_steps):
        mode_n = decision.select_mode(recovered_obs)
        if step_i < recover_exit_steps - 1:
            # 还在 recover（连续满足不够）
            assert parse_mode_label(mode_n).behavior.startswith("recover_"), \
                f"Expected recover at step {step_i+1}/{recover_exit_steps}, got {mode_n}"

    # 最后一步应该允许退出 recover（可能需要再经过 hysteresis）
    # 由于 hysteresis_steps=1 且 risk 已低于 exit，应该会回到 follow
    final_mode = decision.select_mode(recovered_obs)
    parsed_final = parse_mode_label(final_mode)
    assert parsed_final.behavior == "follow", \
        f"Expected follow after {recover_exit_steps} sustained steps, got {final_mode}"


def test_fsm_recover_exits_after_consecutive_satisfaction() -> None:
    """recover 退出迟滞：恢复条件连续满足 recover_exit_steps 步后立即退出。"""
    recover_exit_steps = 2
    decision = FSMModeDecision(
        config=_decision_config(hysteresis_steps=1, recover_exit_steps=recover_exit_steps),
        vehicle_length=4.8,
        vehicle_width=1.9,
        safe_distance=0.5,
    )

    road = RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5)
    # 先推入 yield
    hazard_obs = Observation(
        step_index=0,
        time=0.0,
        states=(
            State(x=22.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=10.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=-2.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
        ),
    )
    decision.select_mode(hazard_obs)
    decision.select_mode(hazard_obs)

    # 推入 recover
    offset_obs = Observation(
        step_index=1,
        time=0.1,
        states=(
            State(x=46.0, y=-1.4, yaw=0.0, speed=4.2),
            State(x=38.0, y=-1.6, yaw=0.0, speed=4.0),
            State(x=30.0, y=-1.5, yaw=0.0, speed=3.8),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=hazard_obs.desired_offsets,
        obstacles=(),
    )
    decision.select_mode(offset_obs)
    recover_mode = decision.select_mode(offset_obs)
    assert parse_mode_label(recover_mode).behavior.startswith("recover_")

    # 恢复完成的观测
    ok_obs = Observation(
        step_index=2,
        time=0.2,
        states=(
            State(x=60.0, y=-0.05, yaw=0.0, speed=5.0),
            State(x=52.0, y=-0.05, yaw=0.0, speed=5.0),
            State(x=44.0, y=-0.05, yaw=0.0, speed=5.0),
        ),
        road=road,
        goal_x=95.0,
        desired_offsets=hazard_obs.desired_offsets,
        obstacles=(),
    )

    # 第 1 步满足：仍在 recover
    m1 = decision.select_mode(ok_obs)
    assert parse_mode_label(m1).behavior.startswith("recover_")

    # 第 2 步满足：现在允许退出
    m2 = decision.select_mode(ok_obs)
    # 此时 candidate 应已不是 recover，加上 hysteresis_steps=1，可能还需 1 步
    m3 = decision.select_mode(ok_obs)
    assert parse_mode_label(m3).behavior == "follow", \
        f"Expected follow after {recover_exit_steps} sustained recovery steps, got {m3}"



def test_apf_lf_controller_consumes_mode_topology_and_behavior() -> None:
    road = Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = _observation(
        leader_x=18.0,
        leader_speed=5.0,
        obstacles=(
            ObstacleState("blocker", 24.0, -1.8, 0.0, 0.0, 4.6, 2.0),
        ),
    )

    nominal_actions = controller.compute_actions(
        observation,
        mode="topology=line|behavior=follow|gain=nominal",
    )
    yield_actions = controller.compute_actions(
        observation,
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )

    assert yield_actions[0].steer > nominal_actions[0].steer
    assert yield_actions[0].accel <= nominal_actions[0].accel
    assert yield_actions[1].steer > 0.0
    assert yield_actions[2].steer > 0.0


def test_apf_lf_controller_slows_leader_during_recovery() -> None:
    road = Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=12,
        time=1.2,
        states=(
            State(x=48.0, y=-0.4, yaw=0.0, speed=5.5),
            State(x=28.0, y=-0.7, yaw=0.0, speed=4.2),
            State(x=10.0, y=-0.9, yaw=0.0, speed=3.4),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    follow_actions = controller.compute_actions(
        observation,
        mode="topology=line|behavior=follow|gain=nominal",
    )
    recover_actions = controller.compute_actions(
        observation,
        mode="topology=line|behavior=recover_right|gain=nominal",
    )

    assert recover_actions[0].accel < follow_actions[0].accel


def test_apf_lf_recover_mode_does_not_push_follower_toward_wrong_boundary() -> None:
    road = Road(RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=153,
        time=15.3,
        states=(
            State(x=108.5, y=-0.2, yaw=0.29, speed=3.4),
            State(x=102.1, y=-2.37, yaw=-0.01, speed=5.7),
            State(x=88.9, y=-0.81, yaw=-0.51, speed=9.0),
        ),
        road=road.geometry,
        goal_x=105.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    recover_actions = controller.compute_actions(
        observation,
        mode="topology=line|behavior=recover_left|gain=nominal",
    )

    assert recover_actions[1].steer > 0.0


def test_apf_lf_follow_mode_creeps_stopped_follower_for_terminal_regrouping() -> None:
    road = Road(RoadGeometry(length=140.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=170,
        time=17.0,
        states=(
            State(x=96.0, y=-1.8, yaw=0.0, speed=0.0),
            State(x=88.2, y=1.0, yaw=0.0, speed=0.0),
            State(x=80.0, y=-1.2, yaw=0.0, speed=0.0),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    actions = controller.compute_actions(
        observation,
        mode="topology=line|behavior=follow|gain=nominal",
    )

    assert actions[1].accel > 0.0
    assert actions[1].steer < 0.0


def test_apf_lf_leader_goal_target_stays_ahead_after_goal_is_passed() -> None:
    road = Road(RoadGeometry(length=140.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    leader = State(x=101.0, y=-0.8, yaw=0.0, speed=0.0)
    observation = Observation(
        step_index=140,
        time=14.0,
        states=(
            leader,
            State(x=96.5, y=-0.6, yaw=0.0, speed=0.0),
            State(x=88.5, y=-0.4, yaw=0.0, speed=0.0),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    target = controller._leader_goal_target(observation, leader)

    assert target[0] > leader.x
    assert target[1] == observation.road.lane_center_y


def test_apf_lf_leader_goal_target_builds_a_true_bypass_offset() -> None:
    road = Road(RoadGeometry(length=160.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=12,
        time=1.2,
        states=(
            State(x=18.0, y=0.0, yaw=0.0, speed=5.0),
            State(x=10.0, y=0.0, yaw=0.0, speed=4.5),
            State(x=2.0, y=0.0, yaw=0.0, speed=4.0),
        ),
        road=road.geometry,
        goal_x=110.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("slow_lead", 22.0, 0.0, 0.0, 2.0, 4.8, 2.0),
            ObstacleState("side_block", 36.0, 1.8, 0.0, 4.2, 4.6, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert target[0] < observation.goal_x
    assert target[0] >= observation.states[0].x + controller.config.vehicle_length
    assert target[1] < -2.0


def test_apf_lf_leader_goal_target_temporarily_uses_near_preview_while_overtake_shift_is_incomplete() -> None:
    road = Road(RoadGeometry(length=160.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=54,
        time=5.4,
        states=(
            State(x=25.36, y=-1.67, yaw=0.0, speed=2.38),
            State(x=17.36, y=-1.60, yaw=0.0, speed=2.20),
            State(x=9.36, y=-1.55, yaw=0.0, speed=2.10),
        ),
        road=road.geometry,
        goal_x=110.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("slow_lead", 32.8, 0.0, 0.0, 2.0, 4.8, 2.0),
            ObstacleState("side_block", 58.7, 1.8, 0.0, 4.2, 4.6, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert target[0] >= observation.states[0].x + controller.config.vehicle_length
    assert target[0] < 36.0
    assert target[1] < -2.0


def test_apf_lf_leader_goal_target_can_flip_locally_when_staggered_blocker_changes_side() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=56,
        time=5.6,
        states=(
            State(x=25.02, y=-0.77, yaw=-0.27, speed=1.58),
            State(x=17.10, y=-1.10, yaw=0.08, speed=1.65),
            State(x=9.30, y=-0.82, yaw=0.01, speed=1.45),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert target[0] < observation.goal_x
    assert target[1] > 0.8


def test_apf_lf_leader_goal_target_keeps_forward_preview_before_local_flip() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=48,
        time=4.8,
        states=(
            State(x=23.36, y=-0.58, yaw=0.01, speed=2.45),
            State(x=15.30, y=-0.75, yaw=0.02, speed=2.30),
            State(x=7.30, y=-0.55, yaw=0.01, speed=2.10),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert target[0] > observation.states[0].x + 10.0
    assert target[0] < observation.goal_x
    assert target[1] < 0.0


def test_apf_lf_leader_goal_target_respects_relocked_left_side() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=58,
        time=5.8,
        states=(
            State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
            State(x=17.40, y=-1.05, yaw=0.08, speed=1.55),
            State(x=9.50, y=-0.84, yaw=0.01, speed=1.40),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )

    assert target[0] >= observation.states[0].x + controller.config.vehicle_length
    assert target[0] < observation.states[0].x + 10.0
    assert target[1] > 0.0


def test_apf_lf_leader_goal_target_keeps_relocked_preview_local_until_new_side_commitment_builds() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=58,
        time=5.8,
        states=(
            State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
            State(x=17.40, y=-1.05, yaw=0.08, speed=1.55),
            State(x=9.50, y=-0.84, yaw=0.01, speed=1.40),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )

    assert target[0] < 33.5
    assert target[0] >= 31.8
    assert target[1] == pytest.approx(0.45)


def test_apf_lf_relocked_preview_hold_stays_inactive_without_opposite_blocker() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=58,
        time=5.8,
        states=(
            State(x=25.30, y=-0.73, yaw=-0.21, speed=1.45),
            State(x=17.40, y=-1.05, yaw=0.08, speed=1.55),
            State(x=9.50, y=-0.84, yaw=0.01, speed=1.40),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("far_left", 80.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    target = controller._leader_goal_target(
        observation,
        observation.states[0],
        mode="topology=diamond|behavior=yield_left|gain=cautious",
    )

    assert target[0] > 33.0


def test_apf_lf_leader_nominal_steer_turns_positive_early_after_local_side_flip() -> None:
    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.5,
    )
    observation = Observation(
        step_index=56,
        time=5.6,
        states=(
            State(x=25.02, y=-0.77, yaw=-0.27, speed=1.58),
            State(x=17.10, y=-1.10, yaw=0.08, speed=1.65),
            State(x=9.30, y=-0.82, yaw=0.01, speed=1.45),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
        ),
    )

    actions = controller.compute_actions(
        observation,
        mode="topology=diamond|behavior=yield_right|gain=cautious",
    )

    assert actions[0].steer > 0.05


def test_apf_lf_recovery_speed_limit_keeps_leader_moving_when_team_is_ahead() -> None:
    road = Road(RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=195,
        time=19.5,
        states=(
            State(x=111.0, y=1.8, yaw=0.15, speed=0.2),
            State(x=120.7, y=0.2, yaw=3.25, speed=3.3),
            State(x=113.8, y=-1.8, yaw=4.1, speed=4.1),
        ),
        road=road.geometry,
        goal_x=105.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    braking_speed = controller._braking_speed(observation.states[0].x, observation.goal_x)
    limited_speed = controller._leader_recovery_speed_limit(
        observation=observation,
        target_speed=braking_speed,
        mode="topology=line|behavior=recover_left|gain=nominal",
    )

    assert braking_speed == 0.0
    assert limited_speed > 1.0
    assert limited_speed <= controller.target_speed


def test_apf_lf_recovery_speed_limit_creeps_when_team_still_lags_after_goal() -> None:
    road = Road(RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5))
    controller = APFLFController(
        config=_controller_config(),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        target_speed=8.0,
    )
    observation = Observation(
        step_index=195,
        time=19.5,
        states=(
            State(x=111.0, y=0.2, yaw=0.0, speed=0.2),
            State(x=100.0, y=0.0, yaw=0.0, speed=0.1),
            State(x=91.5, y=0.1, yaw=0.0, speed=0.1),
        ),
        road=road.geometry,
        goal_x=105.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(),
    )

    braking_speed = controller._braking_speed(observation.states[0].x, observation.goal_x)
    limited_speed = controller._leader_recovery_speed_limit(
        observation=observation,
        target_speed=braking_speed,
        mode="topology=line|behavior=recover_left|gain=nominal",
    )

    assert braking_speed == 0.0
    assert limited_speed > 0.0
    assert limited_speed < 2.5
