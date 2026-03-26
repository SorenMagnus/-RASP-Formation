"""Replay helpers for persisted rollout artifacts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apflf.analysis.metrics import compute_run_summary
from apflf.env.road import Road
from apflf.utils.config import load_config
from apflf.utils.types import (
    Action,
    NominalDiagnostics,
    NominalForceBreakdown,
    ObstacleState,
    Snapshot,
    State,
)


@dataclass(frozen=True)
class ReplayBundle:
    """Typed view of a saved rollout artifact."""

    config: object
    initial_states: tuple[State, ...]
    initial_obstacles: tuple[ObstacleState, ...]
    snapshots: tuple[Snapshot, ...]
    desired_offsets: tuple[tuple[float, float], ...]


def _state_from_array(values: np.ndarray) -> State:
    return State(
        x=float(values[0]),
        y=float(values[1]),
        yaw=float(values[2]),
        speed=float(values[3]),
    )


def _action_from_array(values: np.ndarray) -> Action:
    return Action(accel=float(values[0]), steer=float(values[1]))


def _obstacles_from_array(values: np.ndarray, obstacle_ids: np.ndarray) -> tuple[ObstacleState, ...]:
    obstacles: list[ObstacleState] = []
    for obstacle_id, obstacle_values in zip(obstacle_ids, values, strict=True):
        obstacles.append(
            ObstacleState(
                obstacle_id=str(obstacle_id),
                x=float(obstacle_values[0]),
                y=float(obstacle_values[1]),
                yaw=float(obstacle_values[2]),
                speed=float(obstacle_values[3]),
                length=float(obstacle_values[4]),
                width=float(obstacle_values[5]),
            )
        )
    return tuple(obstacles)


def _optional_array(data: np.lib.npyio.NpzFile, key: str, *, default: np.ndarray) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key])
    return default


def load_replay_bundle(run_dir: Path, seed: int) -> ReplayBundle:
    """Load a saved rollout artifact and reconstruct typed snapshots."""

    resolved_dir = Path(run_dir).resolve()
    config = load_config(resolved_dir / "config_resolved.yaml")
    artifact_path = resolved_dir / "traj" / f"seed_{seed:04d}.npz"
    with np.load(artifact_path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=float)
        obstacles = np.asarray(data["obstacles"], dtype=float)
        obstacle_ids = np.asarray(data["obstacle_ids"])
        nominal_actions = np.asarray(data["nominal_actions"], dtype=float)
        safe_actions = np.asarray(data["safe_actions"], dtype=float)
        safety_corrections = np.asarray(data["safety_corrections"], dtype=float)
        safety_slacks = np.asarray(data["safety_slacks"], dtype=float)
        safety_fallbacks = np.asarray(data["safety_fallbacks"], dtype=bool)
        qp_solve_times = _optional_array(
            data,
            "qp_solve_times",
            default=np.zeros_like(safety_corrections, dtype=float),
        )
        qp_iterations = _optional_array(
            data,
            "qp_iterations",
            default=np.zeros_like(safety_corrections, dtype=int),
        )
        leader_risk_scores = _optional_array(
            data,
            "leader_risk_scores",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_base_reference_speeds = _optional_array(
            data,
            "leader_base_reference_speeds",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_hazard_speed_caps = _optional_array(
            data,
            "leader_hazard_speed_caps",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_staggered_speed_caps = _optional_array(
            data,
            "leader_staggered_speed_caps",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_release_speed_caps = _optional_array(
            data,
            "leader_release_speed_caps",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_target_speeds = _optional_array(
            data,
            "leader_target_speeds",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_staggered_activations = _optional_array(
            data,
            "leader_staggered_activations",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_edge_hold_activations = _optional_array(
            data,
            "leader_edge_hold_activations",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        leader_force_attractive = _optional_array(
            data,
            "leader_force_attractive",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_formation = _optional_array(
            data,
            "leader_force_formation",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_consensus = _optional_array(
            data,
            "leader_force_consensus",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_road = _optional_array(
            data,
            "leader_force_road",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_obstacle = _optional_array(
            data,
            "leader_force_obstacle",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_peer = _optional_array(
            data,
            "leader_force_peer",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_behavior = _optional_array(
            data,
            "leader_force_behavior",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_guidance = _optional_array(
            data,
            "leader_force_guidance",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_escape = _optional_array(
            data,
            "leader_force_escape",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        leader_force_total = _optional_array(
            data,
            "leader_force_total",
            default=np.zeros((safe_actions.shape[0], 2), dtype=float),
        )
        step_runtimes = _optional_array(
            data,
            "step_runtimes",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        mode_runtimes = _optional_array(
            data,
            "mode_runtimes",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        controller_runtimes = _optional_array(
            data,
            "controller_runtimes",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        safety_runtimes = _optional_array(
            data,
            "safety_runtimes",
            default=np.zeros(safe_actions.shape[0], dtype=float),
        )
        desired_offsets = np.asarray(data["desired_offsets"], dtype=float)
        times = np.asarray(data["times"], dtype=float)
        modes = np.asarray(data["modes"])

    initial_states = tuple(_state_from_array(row) for row in states[0])
    initial_obstacles = _obstacles_from_array(obstacles[0], obstacle_ids)
    snapshots: list[Snapshot] = []
    for step_index in range(1, states.shape[0]):
        snapshots.append(
            Snapshot(
                step_index=step_index,
                time=float(times[step_index]),
                mode=str(modes[step_index - 1]),
                states=tuple(_state_from_array(row) for row in states[step_index]),
                nominal_actions=tuple(_action_from_array(row) for row in nominal_actions[step_index - 1]),
                safe_actions=tuple(_action_from_array(row) for row in safe_actions[step_index - 1]),
                obstacles=_obstacles_from_array(obstacles[step_index], obstacle_ids),
                safety_corrections=tuple(float(value) for value in safety_corrections[step_index - 1]),
                safety_slacks=tuple(float(value) for value in safety_slacks[step_index - 1]),
                safety_fallbacks=tuple(bool(value) for value in safety_fallbacks[step_index - 1]),
                qp_solve_times=tuple(float(value) for value in qp_solve_times[step_index - 1]),
                qp_iterations=tuple(int(value) for value in qp_iterations[step_index - 1]),
                step_runtime=float(step_runtimes[step_index - 1]),
                mode_runtime=float(mode_runtimes[step_index - 1]),
                controller_runtime=float(controller_runtimes[step_index - 1]),
                safety_runtime=float(safety_runtimes[step_index - 1]),
                nominal_diagnostics=NominalDiagnostics(
                    leader_risk_score=float(leader_risk_scores[step_index - 1]),
                    leader_base_reference_speed=float(leader_base_reference_speeds[step_index - 1]),
                    leader_hazard_speed_cap=float(leader_hazard_speed_caps[step_index - 1]),
                    leader_staggered_speed_cap=float(leader_staggered_speed_caps[step_index - 1]),
                    leader_release_speed_cap=float(leader_release_speed_caps[step_index - 1]),
                    leader_target_speed=float(leader_target_speeds[step_index - 1]),
                    leader_staggered_activation=float(leader_staggered_activations[step_index - 1]),
                    leader_edge_hold_activation=float(leader_edge_hold_activations[step_index - 1]),
                    leader_force=NominalForceBreakdown(
                        attractive=tuple(float(value) for value in leader_force_attractive[step_index - 1]),
                        formation=tuple(float(value) for value in leader_force_formation[step_index - 1]),
                        consensus=tuple(float(value) for value in leader_force_consensus[step_index - 1]),
                        road=tuple(float(value) for value in leader_force_road[step_index - 1]),
                        obstacle=tuple(float(value) for value in leader_force_obstacle[step_index - 1]),
                        peer=tuple(float(value) for value in leader_force_peer[step_index - 1]),
                        behavior=tuple(float(value) for value in leader_force_behavior[step_index - 1]),
                        guidance=tuple(float(value) for value in leader_force_guidance[step_index - 1]),
                        escape=tuple(float(value) for value in leader_force_escape[step_index - 1]),
                        total=tuple(float(value) for value in leader_force_total[step_index - 1]),
                    ),
                ),
            )
        )
    return ReplayBundle(
        config=config,
        initial_states=initial_states,
        initial_obstacles=initial_obstacles,
        snapshots=tuple(snapshots),
        desired_offsets=tuple((float(row[0]), float(row[1])) for row in desired_offsets),
    )


def recompute_summary(run_dir: Path, seed: int) -> dict[str, float | int | bool]:
    """Recompute summary metrics from a saved rollout artifact."""

    bundle = load_replay_bundle(run_dir, seed)
    return compute_run_summary(
        road=Road(bundle.config.scenario.road),
        goal_x=bundle.config.scenario.goal_x,
        goal_tolerance=bundle.config.scenario.goal_tolerance,
        desired_offsets=bundle.desired_offsets,
        initial_states=bundle.initial_states,
        initial_obstacles=bundle.initial_obstacles,
        snapshots=bundle.snapshots,
        vehicle_length=bundle.config.controller.vehicle_length,
        vehicle_width=bundle.config.controller.vehicle_width,
        bounds=bundle.config.simulation.bounds,
        dt=bundle.config.simulation.dt,
        safe_distance=bundle.config.safety.safe_distance,
    )


def read_summary_row(run_dir: Path, seed: int) -> dict[str, object]:
    """Read a single summary row from summary.csv by seed."""

    summary_path = Path(run_dir).resolve() / "summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) == seed:
                parsed: dict[str, object] = {}
                for key, value in row.items():
                    lowered = value.strip().lower()
                    if lowered in {"true", "false"}:
                        parsed[key] = lowered == "true"
                        continue
                    try:
                        parsed[key] = int(value)
                        continue
                    except ValueError:
                        pass
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                return parsed
    raise KeyError(f"Seed {seed} not found in {summary_path}.")


def compare_summary_dicts(expected: dict[str, object], actual: dict[str, object], *, atol: float = 1e-9) -> dict[str, tuple[object, object]]:
    """Return mismatched keys between two summary dictionaries."""

    mismatches: dict[str, tuple[object, object]] = {}
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches[key] = (expected_value, None)
            continue
        actual_value = actual[key]
        if isinstance(expected_value, bool) or isinstance(actual_value, bool):
            if bool(expected_value) != bool(actual_value):
                mismatches[key] = (expected_value, actual_value)
            continue
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            expected_float = float(expected_value)
            actual_float = float(actual_value)
            if math.isnan(expected_float) and math.isnan(actual_float):
                continue
            if abs(expected_float - actual_float) > atol:
                mismatches[key] = (expected_value, actual_value)
            continue
        if expected_value != actual_value:
            mismatches[key] = (expected_value, actual_value)
    return mismatches
