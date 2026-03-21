"""Tests for recovery-oriented summary metrics."""

from __future__ import annotations

import math

from apflf.analysis.metrics import compute_run_summary
from apflf.env.road import Road
from apflf.utils.types import Action, InputBounds, ObstacleState, RoadGeometry, Snapshot, State


def _noop_actions(count: int = 3) -> tuple[Action, ...]:
    return tuple(Action(accel=0.0, steer=0.0) for _ in range(count))


def _initial_states() -> tuple[State, ...]:
    return (
        State(x=0.0, y=0.0, yaw=0.0, speed=5.0),
        State(x=-8.0, y=0.0, yaw=0.0, speed=5.0),
        State(x=-16.0, y=0.0, yaw=0.0, speed=5.0),
    )


def _road() -> Road:
    return Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))


def _bounds() -> InputBounds:
    return InputBounds(
        accel_min=-3.0,
        accel_max=2.0,
        steer_min=-0.5,
        steer_max=0.5,
        speed_min=0.0,
        speed_max=20.0,
    )


def test_run_summary_marks_team_goal_unrecovered_when_only_leader_arrives() -> None:
    summary = compute_run_summary(
        road=_road(),
        goal_x=30.0,
        goal_tolerance=0.5,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        initial_states=_initial_states(),
        initial_obstacles=(),
        snapshots=(
            Snapshot(
                step_index=1,
                time=1.0,
                mode="topology=diamond|behavior=yield_right|gain=cautious",
                states=(
                    State(x=30.0, y=0.0, yaw=0.0, speed=0.0),
                    State(x=9.0, y=-0.6, yaw=0.0, speed=0.0),
                    State(x=-5.0, y=-0.7, yaw=0.0, speed=0.0),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
        ),
        vehicle_length=4.8,
        vehicle_width=1.9,
        bounds=_bounds(),
        dt=1.0,
    )

    assert summary["reached_goal"] is True
    assert summary["formation_recovered"] is False
    assert summary["team_goal_reached"] is False
    assert math.isnan(float(summary["time_to_recover_formation"]))


def test_run_summary_reports_recovery_time_once_follow_mode_and_line_shape_return() -> None:
    summary = compute_run_summary(
        road=_road(),
        goal_x=30.0,
        goal_tolerance=0.5,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        initial_states=_initial_states(),
        initial_obstacles=(),
        snapshots=(
            Snapshot(
                step_index=1,
                time=1.0,
                mode="topology=diamond|behavior=yield_right|gain=cautious",
                states=(
                    State(x=16.0, y=-0.3, yaw=0.0, speed=4.0),
                    State(x=4.0, y=-0.7, yaw=0.0, speed=3.2),
                    State(x=-7.0, y=-0.8, yaw=0.0, speed=2.8),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
            Snapshot(
                step_index=2,
                time=2.0,
                mode="topology=line|behavior=recover_right|gain=nominal",
                states=(
                    State(x=24.0, y=-0.2, yaw=0.0, speed=3.5),
                    State(x=15.0, y=-0.4, yaw=0.0, speed=3.4),
                    State(x=6.0, y=-0.4, yaw=0.0, speed=3.0),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
            Snapshot(
                step_index=3,
                time=3.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(
                    State(x=30.0, y=0.0, yaw=0.0, speed=1.5),
                    State(x=22.0, y=0.0, yaw=0.0, speed=1.5),
                    State(x=14.0, y=0.0, yaw=0.0, speed=1.5),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
        ),
        vehicle_length=4.8,
        vehicle_width=1.9,
        bounds=_bounds(),
        dt=1.0,
    )

    assert summary["reached_goal"] is True
    assert summary["formation_recovered"] is True
    assert summary["team_goal_reached"] is True
    assert float(summary["time_to_goal"]) == 3.0
    assert float(summary["time_to_team_goal"]) == 3.0
    assert float(summary["time_to_recover_formation"]) == 3.0
    assert float(summary["terminal_max_team_lag"]) == 0.0


def test_run_summary_keeps_goal_time_when_recovery_happens_earlier() -> None:
    summary = compute_run_summary(
        road=_road(),
        goal_x=30.0,
        goal_tolerance=0.5,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        initial_states=_initial_states(),
        initial_obstacles=(),
        snapshots=(
            Snapshot(
                step_index=1,
                time=1.0,
                mode="topology=diamond|behavior=yield_right|gain=cautious",
                states=(
                    State(x=10.0, y=-0.5, yaw=0.0, speed=3.0),
                    State(x=1.0, y=-0.6, yaw=0.0, speed=2.8),
                    State(x=-8.0, y=-0.6, yaw=0.0, speed=2.6),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
            Snapshot(
                step_index=2,
                time=2.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(
                    State(x=20.0, y=0.0, yaw=0.0, speed=3.0),
                    State(x=12.0, y=0.0, yaw=0.0, speed=3.0),
                    State(x=4.0, y=0.0, yaw=0.0, speed=3.0),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
            Snapshot(
                step_index=3,
                time=3.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(
                    State(x=30.0, y=0.0, yaw=0.0, speed=2.0),
                    State(x=22.0, y=0.0, yaw=0.0, speed=2.0),
                    State(x=14.0, y=0.0, yaw=0.0, speed=2.0),
                ),
                nominal_actions=_noop_actions(),
                safe_actions=_noop_actions(),
            ),
        ),
        vehicle_length=4.8,
        vehicle_width=1.9,
        bounds=_bounds(),
        dt=1.0,
    )

    assert float(summary["time_to_recover_formation"]) == 2.0
    assert float(summary["time_to_goal"]) == 3.0
    assert float(summary["time_to_team_goal"]) == 3.0


def test_run_summary_reports_paper_metrics_for_path_smoothness_and_runtime() -> None:
    obstacle = ObstacleState(
        obstacle_id="obs",
        x=12.0,
        y=0.0,
        yaw=0.0,
        speed=0.0,
        length=4.0,
        width=2.0,
    )
    summary = compute_run_summary(
        road=_road(),
        goal_x=6.0,
        goal_tolerance=0.1,
        desired_offsets=((0.0, 0.0),),
        initial_states=(State(x=0.0, y=0.0, yaw=0.0, speed=2.0),),
        initial_obstacles=(obstacle,),
        snapshots=(
            Snapshot(
                step_index=1,
                time=1.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(State(x=2.0, y=0.0, yaw=0.0, speed=2.0),),
                nominal_actions=(Action(accel=2.0, steer=0.5),),
                safe_actions=(Action(accel=2.0, steer=0.5),),
                obstacles=(obstacle,),
                qp_solve_times=(0.004,),
                qp_iterations=(12,),
                step_runtime=0.010,
                mode_runtime=0.001,
                controller_runtime=0.002,
                safety_runtime=0.003,
            ),
            Snapshot(
                step_index=2,
                time=2.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(State(x=4.0, y=0.0, yaw=0.0, speed=2.0),),
                nominal_actions=(Action(accel=-3.0, steer=-0.5),),
                safe_actions=(Action(accel=-3.0, steer=-0.5),),
                obstacles=(obstacle,),
                qp_solve_times=(0.0,),
                qp_iterations=(0,),
                step_runtime=0.020,
                mode_runtime=0.002,
                controller_runtime=0.003,
                safety_runtime=0.004,
            ),
            Snapshot(
                step_index=3,
                time=3.0,
                mode="topology=line|behavior=follow|gain=nominal",
                states=(State(x=6.0, y=0.0, yaw=0.0, speed=2.0),),
                nominal_actions=(Action(accel=0.0, steer=0.0),),
                safe_actions=(Action(accel=0.0, steer=0.0),),
                obstacles=(obstacle,),
                qp_solve_times=(0.006,),
                qp_iterations=(18,),
                step_runtime=0.030,
                mode_runtime=0.003,
                controller_runtime=0.004,
                safety_runtime=0.005,
            ),
        ),
        vehicle_length=4.0,
        vehicle_width=2.0,
        bounds=_bounds(),
        dt=1.0,
    )

    assert float(summary["time_to_goal"]) == 3.0
    assert float(summary["time_to_team_goal"]) == 3.0
    assert float(summary["leader_path_length"]) == 6.0
    assert float(summary["leader_path_length_ratio"]) == 1.0
    assert float(summary["leader_path_efficiency"]) == 1.0
    assert math.isclose(float(summary["min_ttc"]), 1.0, rel_tol=1e-6)
    assert math.isclose(float(summary["longitudinal_jerk_rms"]), math.sqrt(17.0), rel_tol=1e-6)
    assert math.isclose(float(summary["steer_rate_rms"]), math.sqrt(0.625), rel_tol=1e-6)
    assert math.isclose(float(summary["accel_saturation_rate"]), 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(summary["steer_saturation_rate"]), 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(summary["mean_step_runtime_ms"]), 20.0, rel_tol=1e-6)
    assert math.isclose(float(summary["max_step_runtime_ms"]), 30.0, rel_tol=1e-6)
    assert math.isclose(float(summary["mean_safety_runtime_ms"]), 4.0, rel_tol=1e-6)
    assert int(summary["qp_solve_count"]) == 2
    assert math.isclose(float(summary["qp_engagement_rate"]), 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(summary["qp_solve_time_mean_ms"]), 5.0, rel_tol=1e-6)
    assert math.isclose(float(summary["qp_solve_time_max_ms"]), 6.0, rel_tol=1e-6)
    assert math.isclose(float(summary["qp_iteration_mean"]), 15.0, rel_tol=1e-6)
    assert int(summary["qp_iteration_max"]) == 18
