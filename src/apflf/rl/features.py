"""Feature extraction helpers for the white-box RL supervisor."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from apflf.decision.mode_base import (
    SUPPORTED_MODE_BEHAVIORS,
    SUPPORTED_MODE_GAINS,
    SUPPORTED_MODE_TOPOLOGIES,
    parse_mode_label,
)
from apflf.env.geometry import box_clearance, rotation_matrix
from apflf.utils.types import Observation, ObstacleState, State


@dataclass(frozen=True)
class PreviousSafetyFeedback:
    """Compact previous-step safety feedback exposed to the RL supervisor."""

    mean_correction: float = 0.0
    max_correction: float = 0.0
    max_slack: float = 0.0
    any_fallback: float = 0.0


class SupervisorObservationBuilder:
    """Construct fixed-width white-box observations for the RL supervisor."""

    def __init__(
        self,
        *,
        vehicle_length: float,
        vehicle_width: float,
        history_length: int,
        interaction_limit: int,
    ) -> None:
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.history_length = max(int(history_length), 1)
        self.interaction_limit = max(int(interaction_limit), 1)
        self._goal_errors: deque[float] = deque(maxlen=self.history_length)
        self._progress_deltas: deque[float] = deque(maxlen=self.history_length)
        self._previous_feedback = PreviousSafetyFeedback()

    @property
    def feature_dim(self) -> int:
        return 5 + 2 * self.history_length + 3 + 2 + 8 * self.interaction_limit + 4 + 12 + 8

    def reset(self) -> None:
        self._goal_errors.clear()
        self._progress_deltas.clear()
        self._previous_feedback = PreviousSafetyFeedback()

    def observe_feedback(
        self,
        *,
        safety_corrections: tuple[float, ...],
        safety_slacks: tuple[float, ...],
        safety_fallbacks: tuple[bool, ...],
    ) -> None:
        corrections = np.asarray(safety_corrections, dtype=float)
        slacks = np.asarray(safety_slacks, dtype=float)
        fallbacks = np.asarray(safety_fallbacks, dtype=float)
        self._previous_feedback = PreviousSafetyFeedback(
            mean_correction=float(np.mean(corrections)) if corrections.size else 0.0,
            max_correction=float(np.max(corrections)) if corrections.size else 0.0,
            max_slack=float(np.max(slacks)) if slacks.size else 0.0,
            any_fallback=float(np.max(fallbacks)) if fallbacks.size else 0.0,
        )

    def build(
        self,
        *,
        observation: Observation,
        mode: str,
        previous_theta: tuple[float, float, float, float],
        previous_theta_delta: tuple[float, float, float, float],
    ) -> np.ndarray:
        leader = observation.states[0]
        goal_error = float(max(observation.goal_x - leader.x, 0.0))
        progress_delta = (
            self._goal_errors[-1] - goal_error if self._goal_errors else 0.0
        )
        self._goal_errors.append(goal_error)
        self._progress_deltas.append(progress_delta)

        leader_features = np.asarray(
            [
                goal_error,
                leader.y - observation.road.lane_center_y,
                math.cos(leader.yaw),
                math.sin(leader.yaw),
                leader.speed,
            ],
            dtype=float,
        )
        history_features = np.concatenate(
            [
                self._pad_history(self._goal_errors),
                self._pad_history(self._progress_deltas),
            ]
        )
        formation_features = np.asarray(
            self._formation_aggregates(observation),
            dtype=float,
        )
        boundary_features = np.asarray(
            self._boundary_clearances(observation, leader),
            dtype=float,
        )
        interaction_features = self._interaction_features(observation, leader)
        safety_feedback = np.asarray(
            [
                self._previous_feedback.mean_correction,
                self._previous_feedback.max_correction,
                self._previous_feedback.max_slack,
                self._previous_feedback.any_fallback,
            ],
            dtype=float,
        )
        mode_one_hot = self._mode_one_hot(mode)
        theta_features = np.asarray(
            [*previous_theta, *previous_theta_delta],
            dtype=float,
        )
        return np.concatenate(
            [
                leader_features,
                history_features,
                formation_features,
                boundary_features,
                interaction_features,
                safety_feedback,
                mode_one_hot,
                theta_features,
            ]
        )

    def _pad_history(self, values: deque[float]) -> np.ndarray:
        padded = np.zeros(self.history_length, dtype=float)
        if not values:
            return padded
        history = np.asarray(tuple(values), dtype=float)
        padded[-history.size :] = history
        return padded

    def _formation_aggregates(self, observation: Observation) -> tuple[float, float, float]:
        leader = observation.states[0]
        leader_position = np.asarray([leader.x, leader.y], dtype=float)
        leader_rotation = rotation_matrix(leader.yaw)
        errors: list[float] = []
        lags: list[float] = []
        for index, state in enumerate(observation.states[1:], start=1):
            desired_position = leader_position + leader_rotation @ np.asarray(
                observation.desired_offsets[index],
                dtype=float,
            )
            error_vector = desired_position - np.asarray([state.x, state.y], dtype=float)
            errors.append(float(np.linalg.norm(error_vector)))
            lags.append(max(float(error_vector[0]), 0.0))
        if not errors:
            return (0.0, 0.0, 0.0)
        return (
            float(np.mean(errors)),
            float(np.max(errors)),
            float(np.max(lags)),
        )

    def _boundary_clearances(self, observation: Observation, leader: State) -> tuple[float, float]:
        upper_clearance = (
            observation.road.half_width
            - (leader.y - observation.road.lane_center_y)
            - 0.5 * self.vehicle_width
        )
        lower_clearance = (
            observation.road.half_width
            + (leader.y - observation.road.lane_center_y)
            - 0.5 * self.vehicle_width
        )
        return (float(upper_clearance), float(lower_clearance))

    def _interaction_features(self, observation: Observation, leader: State) -> np.ndarray:
        interaction_rows: list[np.ndarray] = []
        for peer in observation.states[1:]:
            interaction_rows.append(
                self._entity_features(
                    leader=leader,
                    other=peer,
                    other_length=self.vehicle_length,
                    other_width=self.vehicle_width,
                    is_vehicle=1.0,
                    is_obstacle=0.0,
                )
            )
        for obstacle in observation.obstacles:
            interaction_rows.append(
                self._entity_features(
                    leader=leader,
                    other=obstacle,
                    other_length=obstacle.length,
                    other_width=obstacle.width,
                    is_vehicle=0.0,
                    is_obstacle=1.0,
                )
            )
        interaction_rows.sort(key=lambda row: (row[4], abs(row[0])))
        padded = np.zeros((self.interaction_limit, 8), dtype=float)
        for index, row in enumerate(interaction_rows[: self.interaction_limit]):
            padded[index] = row
        return padded.reshape(-1)

    def _entity_features(
        self,
        *,
        leader: State,
        other: State | ObstacleState,
        other_length: float,
        other_width: float,
        is_vehicle: float,
        is_obstacle: float,
    ) -> np.ndarray:
        leader_velocity = np.asarray(
            [leader.speed * math.cos(leader.yaw), leader.speed * math.sin(leader.yaw)],
            dtype=float,
        )
        other_velocity = np.asarray(
            [other.speed * math.cos(other.yaw), other.speed * math.sin(other.yaw)],
            dtype=float,
        )
        relative_position = np.asarray([other.x - leader.x, other.y - leader.y], dtype=float)
        distance = float(np.linalg.norm(relative_position))
        clearance = box_clearance(
            leader,
            self.vehicle_length,
            self.vehicle_width,
            other,
            other_length,
            other_width,
        )
        if distance <= 1e-9:
            ttc = 0.0
        else:
            line_of_sight = relative_position / distance
            closing_speed = max(float(np.dot(leader_velocity - other_velocity, line_of_sight)), 0.0)
            ttc = float("inf") if closing_speed <= 1e-9 else distance / closing_speed
        finite_ttc = 50.0 if not math.isfinite(ttc) else ttc
        return np.asarray(
            [
                float(relative_position[0]),
                float(relative_position[1]),
                float(other_velocity[0] - leader_velocity[0]),
                float(other_velocity[1] - leader_velocity[1]),
                float(clearance),
                float(finite_ttc),
                float(is_vehicle),
                float(is_obstacle),
            ],
            dtype=float,
        )

    def _mode_one_hot(self, mode: str) -> np.ndarray:
        parsed = parse_mode_label(mode)
        topology_values = sorted(SUPPORTED_MODE_TOPOLOGIES)
        behavior_values = sorted(SUPPORTED_MODE_BEHAVIORS)
        gain_values = sorted(SUPPORTED_MODE_GAINS)
        vector = np.zeros(len(topology_values) + len(behavior_values) + len(gain_values), dtype=float)
        vector[topology_values.index(parsed.topology)] = 1.0
        vector[len(topology_values) + behavior_values.index(parsed.behavior)] = 1.0
        vector[len(topology_values) + len(behavior_values) + gain_values.index(parsed.gain)] = 1.0
        return vector
