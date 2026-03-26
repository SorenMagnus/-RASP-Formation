"""Experiment metric computation."""

from __future__ import annotations

import math

import numpy as np

from apflf.decision.mode_base import parse_mode_label
from apflf.env.geometry import box_clearance, rotation_matrix
from apflf.env.road import Road
from apflf.safety.cbf import boundary_barrier
from apflf.utils.types import InputBounds, ObstacleState, Snapshot, State


def build_state_history(
    initial_states: tuple[State, ...],
    snapshots: tuple[Snapshot, ...],
) -> np.ndarray:
    """Stack state histories into a tensor."""

    history = [[state.to_array() for state in initial_states]]
    for snapshot in snapshots:
        history.append([state.to_array() for state in snapshot.states])
    return np.asarray(history, dtype=float)


def build_obstacle_history(
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
) -> np.ndarray:
    """Stack obstacle histories into a tensor."""

    if not initial_obstacles:
        return np.zeros((len(snapshots) + 1, 0, 6), dtype=float)
    history = [[obstacle.to_numeric_array() for obstacle in initial_obstacles]]
    for snapshot in snapshots:
        history.append([obstacle.to_numeric_array() for obstacle in snapshot.obstacles])
    return np.asarray(history, dtype=float)


def build_action_history(snapshots: tuple[Snapshot, ...], field_name: str) -> np.ndarray:
    """Extract nominal or safe action histories."""

    actions = []
    for snapshot in snapshots:
        field = getattr(snapshot, field_name)
        actions.append([action.to_array() for action in field])
    if not actions:
        return np.zeros((0, 0, 2), dtype=float)
    return np.asarray(actions, dtype=float)


def build_scalar_history(
    snapshots: tuple[Snapshot, ...],
    field_name: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Extract per-step scalar diagnostics from snapshots."""

    values = [list(getattr(snapshot, field_name)) for snapshot in snapshots]
    if not values:
        return np.zeros((0, 0), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def build_step_scalar_series(
    snapshots: tuple[Snapshot, ...],
    field_name: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Extract scalar step-level diagnostics from snapshots."""

    if not snapshots:
        return np.zeros(0, dtype=dtype)
    return np.asarray([getattr(snapshot, field_name) for snapshot in snapshots], dtype=dtype)


def build_nominal_scalar_series(
    snapshots: tuple[Snapshot, ...],
    field_name: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Extract leader nominal-diagnostic scalar histories from snapshots."""

    if not snapshots:
        return np.zeros(0, dtype=dtype)
    return np.asarray(
        [getattr(snapshot.nominal_diagnostics, field_name) for snapshot in snapshots],
        dtype=dtype,
    )


def build_nominal_force_history(
    snapshots: tuple[Snapshot, ...],
    field_name: str,
) -> np.ndarray:
    """Extract leader nominal-force vector histories from snapshots."""

    if not snapshots:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(
        [getattr(snapshot.nominal_diagnostics.leader_force, field_name) for snapshot in snapshots],
        dtype=float,
    )


def _state_and_obstacle_steps(
    initial_states: tuple[State, ...],
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
) -> tuple[list[tuple[State, ...]], list[tuple[ObstacleState, ...]]]:
    state_steps = [initial_states, *[snapshot.states for snapshot in snapshots]]
    obstacle_steps = [initial_obstacles, *[snapshot.obstacles for snapshot in snapshots]]
    return state_steps, obstacle_steps


def _compute_safety_statistics(
    *,
    road: Road,
    initial_states: tuple[State, ...],
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
    vehicle_length: float,
    vehicle_width: float,
    safe_distance: float,
) -> tuple[float, float, int, int]:
    """Return boundary/obstacle minima and violation counts with unified sign."""

    state_steps, obstacle_steps = _state_and_obstacle_steps(
        initial_states=initial_states,
        initial_obstacles=initial_obstacles,
        snapshots=snapshots,
    )
    min_boundary_margin = float("inf")
    min_obstacle_clearance = float("inf")
    collision_count = 0
    boundary_violation_count = 0

    for states, obstacles in zip(state_steps, obstacle_steps, strict=True):
        for state_index, state in enumerate(states):
            boundary_value = boundary_barrier(
                state=state,
                road=road,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
            )
            min_boundary_margin = min(min_boundary_margin, boundary_value)
            if boundary_value < 0.0:
                boundary_violation_count += 1

            for other_index, other_state in enumerate(states):
                if other_index <= state_index:
                    continue
                clearance = box_clearance(
                    state,
                    vehicle_length,
                    vehicle_width,
                    other_state,
                    vehicle_length,
                    vehicle_width,
                )
                min_obstacle_clearance = min(min_obstacle_clearance, clearance)
                if clearance <= 0.0:
                    collision_count += 1

            for obstacle in obstacles:
                clearance = box_clearance(
                    state,
                    vehicle_length,
                    vehicle_width,
                    obstacle,
                    obstacle.length,
                    obstacle.width,
                )
                min_obstacle_clearance = min(min_obstacle_clearance, clearance)
                if clearance <= 0.0:
                    collision_count += 1

    if math.isinf(min_boundary_margin):
        min_boundary_margin = float("nan")
    if math.isinf(min_obstacle_clearance):
        min_obstacle_clearance = float("nan")
    return (
        min_boundary_margin,
        min_obstacle_clearance,
        collision_count,
        boundary_violation_count,
    )


def _velocity_vector(entity: State | ObstacleState) -> np.ndarray:
    return np.asarray(
        [
            entity.speed * math.cos(entity.yaw),
            entity.speed * math.sin(entity.yaw),
        ],
        dtype=float,
    )


def _pairwise_ttc(
    *,
    ego: State,
    other: State | ObstacleState,
    ego_length: float,
    ego_width: float,
    other_length: float,
    other_width: float,
) -> float:
    clearance = box_clearance(
        ego,
        ego_length,
        ego_width,
        other,
        other_length,
        other_width,
    )
    if clearance <= 0.0:
        return 0.0
    relative_position = np.asarray([other.x - ego.x, other.y - ego.y], dtype=float)
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-9:
        return 0.0
    relative_velocity = _velocity_vector(other) - _velocity_vector(ego)
    closing_speed = float(-np.dot(relative_position / distance, relative_velocity))
    if closing_speed <= 1e-9:
        return float("inf")
    return float(clearance / closing_speed)


def _compute_min_ttc(
    *,
    initial_states: tuple[State, ...],
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
    vehicle_length: float,
    vehicle_width: float,
) -> float:
    state_steps, obstacle_steps = _state_and_obstacle_steps(
        initial_states=initial_states,
        initial_obstacles=initial_obstacles,
        snapshots=snapshots,
    )
    min_ttc = float("inf")
    for states, obstacles in zip(state_steps, obstacle_steps, strict=True):
        for state_index, state in enumerate(states):
            for other_index, other_state in enumerate(states):
                if other_index <= state_index:
                    continue
                min_ttc = min(
                    min_ttc,
                    _pairwise_ttc(
                        ego=state,
                        other=other_state,
                        ego_length=vehicle_length,
                        ego_width=vehicle_width,
                        other_length=vehicle_length,
                        other_width=vehicle_width,
                    ),
                )
            for obstacle in obstacles:
                min_ttc = min(
                    min_ttc,
                    _pairwise_ttc(
                        ego=state,
                        other=obstacle,
                        ego_length=vehicle_length,
                        ego_width=vehicle_width,
                        other_length=obstacle.length,
                        other_width=obstacle.width,
                    ),
                )
    if math.isinf(min_ttc):
        return float("nan")
    return min_ttc


def _formation_spacing(desired_offsets: tuple[tuple[float, float], ...], vehicle_length: float) -> float:
    if len(desired_offsets) <= 1:
        return max(1.5 * vehicle_length, 1e-6)
    non_zero_offsets = [abs(offset[0]) for offset in desired_offsets[1:] if abs(offset[0]) > 1e-6]
    if not non_zero_offsets:
        return max(1.5 * vehicle_length, 1e-6)
    return max(min(non_zero_offsets), 1e-6)


def _formation_tracking_history(
    *,
    state_history: np.ndarray,
    desired_offsets: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_errors: list[float] = []
    max_errors: list[float] = []
    max_lags: list[float] = []

    for states in state_history:
        leader_position = states[0, :2]
        leader_rotation = rotation_matrix(float(states[0, 2]))
        errors: list[float] = []
        lags: list[float] = []
        for index in range(1, len(desired_offsets)):
            desired_position = leader_position + leader_rotation @ np.asarray(desired_offsets[index], dtype=float)
            error_vector = desired_position - states[index, :2]
            errors.append(float(np.linalg.norm(error_vector)))
            lags.append(max(float(error_vector[0]), 0.0))
        mean_errors.append(float(np.mean(errors)) if errors else 0.0)
        max_errors.append(float(max(errors, default=0.0)))
        max_lags.append(float(max(lags, default=0.0)))

    return (
        np.asarray(mean_errors, dtype=float),
        np.asarray(max_errors, dtype=float),
        np.asarray(max_lags, dtype=float),
    )


def _path_lengths(state_history: np.ndarray) -> np.ndarray:
    if state_history.shape[0] <= 1:
        return np.zeros(state_history.shape[1], dtype=float)
    position_deltas = np.diff(state_history[:, :, :2], axis=0)
    return np.sum(np.linalg.norm(position_deltas, axis=2), axis=0)


def _action_smoothness(action_history: np.ndarray, dt: float) -> tuple[float, float]:
    if action_history.shape[0] <= 1:
        return 0.0, 0.0
    accel_jerk = np.diff(action_history[:, :, 0], axis=0) / max(dt, 1e-9)
    steer_rate = np.diff(action_history[:, :, 1], axis=0) / max(dt, 1e-9)
    return (
        float(np.sqrt(np.mean(np.square(accel_jerk)))) if accel_jerk.size else 0.0,
        float(np.sqrt(np.mean(np.square(steer_rate)))) if steer_rate.size else 0.0,
    )


def _saturation_rate(values: np.ndarray, *, lower: float, upper: float, atol: float = 1e-6) -> float:
    if values.size == 0:
        return 0.0
    saturated = np.isclose(values, lower, atol=atol, rtol=0.0) | np.isclose(
        values,
        upper,
        atol=atol,
        rtol=0.0,
    )
    return float(np.mean(saturated))


def _runtime_summary(series: np.ndarray) -> tuple[float, float]:
    if series.size == 0:
        return 0.0, 0.0
    return float(np.mean(series) * 1000.0), float(np.max(series) * 1000.0)


def compute_run_summary(
    *,
    road: Road,
    goal_x: float,
    goal_tolerance: float,
    desired_offsets: tuple[tuple[float, float], ...],
    initial_states: tuple[State, ...],
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
    vehicle_length: float,
    vehicle_width: float,
    bounds: InputBounds,
    dt: float,
    safe_distance: float = 0.0,
) -> dict[str, float | int | bool]:
    """Compute aggregate metrics for a single rollout."""

    state_history = build_state_history(initial_states=initial_states, snapshots=snapshots)
    final_states = state_history[-1]
    initial_leader_position = state_history[0, 0, :2]
    final_leader_position = final_states[0, :2]
    leader_final_x = float(final_states[0, 0])
    leader_goal_error = float(max(goal_x - leader_final_x, 0.0))
    speed_values = state_history[:, :, 3]
    path_lengths = _path_lengths(state_history)
    leader_path_length = float(path_lengths[0]) if path_lengths.size else 0.0
    leader_displacement = float(np.linalg.norm(final_leader_position - initial_leader_position))
    leader_path_length_ratio = float(
        leader_path_length / max(leader_displacement, 1e-6)
    ) if leader_path_length > 0.0 else 1.0
    leader_path_efficiency = float(
        leader_displacement / max(leader_path_length, 1e-6)
    ) if leader_displacement > 0.0 else 1.0

    mean_formation_errors, max_formation_errors, max_team_lags = _formation_tracking_history(
        state_history=state_history,
        desired_offsets=desired_offsets,
    )
    terminal_formation_error = float(mean_formation_errors[-1]) if mean_formation_errors.size else 0.0
    terminal_max_team_lag = float(max_team_lags[-1]) if max_team_lags.size else 0.0
    sim_time = round(float(snapshots[-1].time), 10) if snapshots else 0.0

    (
        min_boundary_margin,
        min_obstacle_clearance,
        collision_count,
        boundary_violation_count,
    ) = _compute_safety_statistics(
        road=road,
        initial_states=initial_states,
        initial_obstacles=initial_obstacles,
        snapshots=snapshots,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        safe_distance=safe_distance,
    )
    min_ttc = _compute_min_ttc(
        initial_states=initial_states,
        initial_obstacles=initial_obstacles,
        snapshots=snapshots,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
    )

    safe_action_history = build_action_history(snapshots=snapshots, field_name="safe_actions")
    safety_corrections = build_scalar_history(snapshots, "safety_corrections")
    safety_slacks = build_scalar_history(snapshots, "safety_slacks")
    safety_fallbacks = build_scalar_history(snapshots, "safety_fallbacks", dtype=np.bool_)
    qp_solve_times = build_scalar_history(snapshots, "qp_solve_times")
    qp_iterations = build_scalar_history(snapshots, "qp_iterations", dtype=np.int64)
    step_runtimes = build_step_scalar_series(snapshots, "step_runtime")
    mode_runtimes = build_step_scalar_series(snapshots, "mode_runtime")
    controller_runtimes = build_step_scalar_series(snapshots, "controller_runtime")
    safety_runtimes = build_step_scalar_series(snapshots, "safety_runtime")
    fallback_count = int(np.count_nonzero(safety_fallbacks)) if safety_fallbacks.size else 0
    if safety_fallbacks.size:
        total_action_count = int(safety_fallbacks.size)
    elif safe_action_history.ndim == 3:
        total_action_count = int(safe_action_history.shape[0] * safe_action_history.shape[1])
    else:
        total_action_count = 0
    time_to_goal = 0.0 if leader_goal_error <= goal_tolerance and not snapshots else float("nan")
    time_to_team_goal = float("nan")
    spacing = _formation_spacing(desired_offsets, vehicle_length)
    recovery_error_threshold = 0.70 * spacing
    recovery_lag_threshold = 0.55 * spacing
    formation_recovered = bool(
        (max_formation_errors[-1] <= recovery_error_threshold if max_formation_errors.size else True)
        and (terminal_max_team_lag <= recovery_lag_threshold)
    )
    time_to_recover_formation = float("nan")
    for snapshot in snapshots:
        if snapshot.states[0].x >= goal_x - goal_tolerance:
            time_to_goal = float(snapshot.time)
            break
    seen_non_follow_mode = False
    for snapshot, step_max_error, step_max_lag in zip(
        snapshots,
        max_formation_errors[1:],
        max_team_lags[1:],
        strict=True,
    ):
        behavior = parse_mode_label(snapshot.mode).behavior
        if behavior != "follow":
            seen_non_follow_mode = True
            continue
        if (
            seen_non_follow_mode
            and step_max_error <= recovery_error_threshold
            and step_max_lag <= recovery_lag_threshold
        ):
            time_to_recover_formation = float(snapshot.time)
            break
    for snapshot, step_max_error, step_max_lag in zip(
        snapshots,
        max_formation_errors[1:],
        max_team_lags[1:],
        strict=True,
    ):
        if (
            snapshot.states[0].x >= goal_x - goal_tolerance
            and step_max_error <= recovery_error_threshold
            and step_max_lag <= recovery_lag_threshold
        ):
            time_to_team_goal = float(snapshot.time)
            break

    longitudinal_jerk_rms, steer_rate_rms = _action_smoothness(safe_action_history, dt)
    accel_saturation_rate = _saturation_rate(
        safe_action_history[:, :, 0] if safe_action_history.size else np.zeros(0, dtype=float),
        lower=bounds.accel_min,
        upper=bounds.accel_max,
    )
    steer_saturation_rate = _saturation_rate(
        safe_action_history[:, :, 1] if safe_action_history.size else np.zeros(0, dtype=float),
        lower=bounds.steer_min,
        upper=bounds.steer_max,
    )
    mean_step_runtime_ms, max_step_runtime_ms = _runtime_summary(step_runtimes)
    mean_mode_runtime_ms, max_mode_runtime_ms = _runtime_summary(mode_runtimes)
    mean_controller_runtime_ms, max_controller_runtime_ms = _runtime_summary(controller_runtimes)
    mean_safety_runtime_ms, max_safety_runtime_ms = _runtime_summary(safety_runtimes)
    qp_time_values = qp_solve_times[qp_solve_times > 0.0]
    qp_iteration_values = qp_iterations[qp_iterations > 0]

    return {
        "num_steps": int(len(snapshots)),
        "sim_time": sim_time,
        "leader_final_x": leader_final_x,
        "leader_goal_error": leader_goal_error,
        "time_to_goal": time_to_goal,
        "time_to_team_goal": time_to_team_goal,
        "mean_speed": float(np.mean(speed_values)),
        "max_speed": float(np.max(speed_values)),
        "leader_path_length": leader_path_length,
        "leader_path_length_ratio": leader_path_length_ratio,
        "leader_path_efficiency": leader_path_efficiency,
        "min_ttc": min_ttc,
        "min_boundary_margin": min_boundary_margin,
        "min_obstacle_clearance": min_obstacle_clearance,
        "collision_count": collision_count,
        "boundary_violation_count": boundary_violation_count,
        "terminal_formation_error": terminal_formation_error,
        "terminal_max_team_lag": terminal_max_team_lag,
        "formation_recovered": formation_recovered,
        "time_to_recover_formation": time_to_recover_formation,
        "longitudinal_jerk_rms": longitudinal_jerk_rms,
        "steer_rate_rms": steer_rate_rms,
        "accel_saturation_rate": accel_saturation_rate,
        "steer_saturation_rate": steer_saturation_rate,
        "mean_safety_correction": float(np.mean(safety_corrections))
        if safety_corrections.size
        else 0.0,
        "safety_interventions": int(np.count_nonzero(safety_corrections > 1e-6))
        if safety_corrections.size
        else 0,
        "slack_mean": float(np.mean(safety_slacks)) if safety_slacks.size else 0.0,
        "slack_max": float(np.max(safety_slacks)) if safety_slacks.size else 0.0,
        "max_safety_slack": float(np.max(safety_slacks)) if safety_slacks.size else 0.0,
        "fallback_count": fallback_count,
        "fallback_events": fallback_count,
        "fallback_ratio": float(fallback_count / total_action_count) if total_action_count else 0.0,
        "mean_step_runtime_ms": mean_step_runtime_ms,
        "max_step_runtime_ms": max_step_runtime_ms,
        "mean_mode_runtime_ms": mean_mode_runtime_ms,
        "max_mode_runtime_ms": max_mode_runtime_ms,
        "mean_controller_runtime_ms": mean_controller_runtime_ms,
        "max_controller_runtime_ms": max_controller_runtime_ms,
        "mean_safety_runtime_ms": mean_safety_runtime_ms,
        "max_safety_runtime_ms": max_safety_runtime_ms,
        "qp_solve_count": int(qp_time_values.size),
        "qp_engagement_rate": float(qp_time_values.size / total_action_count) if total_action_count else 0.0,
        "qp_solve_time_mean_ms": float(np.mean(qp_time_values) * 1000.0) if qp_time_values.size else 0.0,
        "qp_solve_time_max_ms": float(np.max(qp_time_values) * 1000.0) if qp_time_values.size else 0.0,
        "qp_iteration_mean": float(np.mean(qp_iteration_values)) if qp_iteration_values.size else 0.0,
        "qp_iteration_max": int(np.max(qp_iteration_values)) if qp_iteration_values.size else 0,
        "reached_goal": leader_goal_error <= goal_tolerance,
        "team_goal_reached": bool(leader_goal_error <= goal_tolerance and formation_recovered),
    }
