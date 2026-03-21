"""Deterministic FSM mode decision with hysteresis."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from apflf.decision.mode_base import (
    ModeDecisionModule,
    compose_mode_label,
    parse_mode_label,
)
from apflf.env.geometry import box_clearance, rotation_matrix
from apflf.utils.types import DecisionConfig, Observation, ObstacleState, State


@dataclass(frozen=True)
class HazardAssessment:
    """Compact hazard summary used by the FSM transition logic."""

    risk_score: float
    narrow_passage: bool
    preferred_side: str
    has_front_obstacle: bool
    needs_recovery: bool
    team_formation_error: float
    max_teammate_lag: float
    max_centerline_offset: float


class FSMModeDecision(ModeDecisionModule):
    """Primary deterministic mode selector used by the project."""

    def __init__(
        self,
        *,
        config: DecisionConfig,
        vehicle_length: float,
        vehicle_width: float,
        safe_distance: float,
    ) -> None:
        self.config = config
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.safe_distance = safe_distance
        self._default_mode = parse_mode_label(config.default_mode).to_label()
        self._current_mode = self._default_mode
        self._pending_mode = self._default_mode
        self._pending_count = 0
        self._leader_x_history: deque[float] = deque(maxlen=config.stagnation_steps + 1)
        self._locked_side: str | None = None
        self._hazard_memory_active = False

    def select_mode(self, observation: Observation) -> str:
        """Select a discrete topology/behavior/gain mode."""

        leader = observation.states[0]
        self._leader_x_history.append(leader.x)
        hazard = self._assess_hazard(observation)
        candidate_mode = self._candidate_mode(observation=observation, hazard=hazard)
        return self._apply_hysteresis(candidate_mode)

    def _candidate_mode(self, *, observation: Observation, hazard: HazardAssessment) -> str:
        current_mode = parse_mode_label(self._current_mode)
        current_side = self._behavior_side(current_mode.behavior)
        preferred_side = self._locked_side or current_side or hazard.preferred_side
        active_front_obstacle = hazard.has_front_obstacle
        if self._hazard_memory_active and self._locked_side is not None:
            active_front_obstacle = self._has_active_front_obstacle(
                observation,
                side=preferred_side,
            )

        if self._is_stagnated(observation) and active_front_obstacle:
            topology = "diamond" if len(observation.states) > 2 else "line"
            self._locked_side = preferred_side
            self._hazard_memory_active = True
            return compose_mode_label(
                topology=topology,
                behavior=f"escape_{preferred_side}",
                gain="assertive",
            )

        if active_front_obstacle and (
            hazard.narrow_passage
            or hazard.risk_score >= self.config.risk_threshold_enter
            or current_mode.behavior.startswith(("yield_", "escape_"))
            or self._hazard_memory_active
        ):
            topology = "diamond" if len(observation.states) > 2 else "line"
            self._locked_side = preferred_side
            self._hazard_memory_active = True
            return compose_mode_label(
                topology=topology,
                behavior=f"yield_{preferred_side}",
                gain="cautious",
            )

        if self._hazard_memory_active and hazard.needs_recovery:
            self._locked_side = preferred_side
            return compose_mode_label(
                topology="line",
                behavior=f"recover_{preferred_side}",
                gain="nominal",
            )

        if hazard.risk_score <= self.config.risk_threshold_exit and not hazard.narrow_passage:
            self._locked_side = None
            self._hazard_memory_active = False
            return self._default_mode

        return current_mode.to_label()

    def _apply_hysteresis(self, candidate_mode: str) -> str:
        if candidate_mode == self._current_mode:
            self._pending_mode = candidate_mode
            self._pending_count = 0
            return self._current_mode

        if candidate_mode != self._pending_mode:
            self._pending_mode = candidate_mode
            self._pending_count = 1
            return self._current_mode

        self._pending_count += 1
        if self._pending_count >= self.config.hysteresis_steps:
            self._current_mode = candidate_mode
            self._pending_mode = candidate_mode
            self._pending_count = 0
        return self._current_mode

    def _assess_hazard(self, observation: Observation) -> HazardAssessment:
        leader = observation.states[0]
        formation_front_obstacles: list[ObstacleState] = []
        min_clearance = float("inf")
        min_ttc = float("inf")
        boundary_margin = float("inf")

        for state in observation.states:
            front_obstacles = self._front_obstacles(observation, state=state)
            formation_front_obstacles.extend(front_obstacles)
            for obstacle in front_obstacles:
                min_clearance = min(
                    min_clearance,
                    box_clearance(
                        state,
                        self.vehicle_length,
                        self.vehicle_width,
                        obstacle,
                        obstacle.length,
                        obstacle.width,
                    ),
                )
                min_ttc = min(min_ttc, self._time_to_collision(state, obstacle))

            boundary_margin = min(
                boundary_margin,
                observation.road.half_width - (
                    abs(state.y - observation.road.lane_center_y) + 0.5 * self.vehicle_width
                ),
            )

        if not formation_front_obstacles:
            min_clearance = float("inf")
            min_ttc = float("inf")
            boundary_margin = observation.road.half_width - (
                abs(leader.y - observation.road.lane_center_y) + 0.5 * self.vehicle_width
            )

        unique_front_obstacles = tuple(
            {obstacle.obstacle_id: obstacle for obstacle in formation_front_obstacles}.values()
        )
        risk_score = self._risk_score(
            min_clearance=min_clearance,
            min_ttc=min_ttc,
            boundary_margin=boundary_margin,
        )
        preferred_side = self._preferred_side(observation, unique_front_obstacles)
        if self._locked_side is not None and unique_front_obstacles:
            preferred_side = self._locked_side
        team_formation_error, max_teammate_lag = self._team_recovery_metrics(observation)
        spacing = self._formation_spacing(observation)
        max_centerline_offset = max(
            abs(state.y - observation.road.lane_center_y) for state in observation.states
        )
        centerline_recovery_threshold = max(
            0.30 * observation.road.half_width,
            0.75 * self.vehicle_width,
        )
        needs_recovery = (
            max_teammate_lag > 0.55 * spacing
            or team_formation_error > 0.70 * spacing
            or max_centerline_offset > centerline_recovery_threshold
        )
        return HazardAssessment(
            risk_score=risk_score,
            narrow_passage=self._is_narrow_passage(observation, unique_front_obstacles),
            preferred_side=preferred_side,
            has_front_obstacle=bool(unique_front_obstacles),
            needs_recovery=needs_recovery,
            team_formation_error=team_formation_error,
            max_teammate_lag=max_teammate_lag,
            max_centerline_offset=max_centerline_offset,
        )

    def _front_obstacles(
        self,
        observation: Observation,
        *,
        state: State,
        side: str | None = None,
    ) -> tuple[ObstacleState, ...]:
        front: list[ObstacleState] = []
        state_rear_x = state.x - 0.5 * self.vehicle_length
        state_front_x = state.x + 0.5 * self.vehicle_length
        for obstacle in observation.obstacles:
            if obstacle.x + 0.5 * obstacle.length < state_rear_x:
                continue
            obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
            longitudinal_gap = obstacle_rear_x - state_front_x
            if longitudinal_gap > self.config.lookahead_distance:
                continue
            if side is not None and not self._obstacle_relevant_to_side(
                observation,
                obstacle=obstacle,
                side=side,
            ):
                continue
            front.append(obstacle)
        return tuple(front)

    def _has_active_front_obstacle(self, observation: Observation, *, side: str) -> bool:
        for state in observation.states:
            if self._front_obstacles(observation, state=state, side=side):
                return True
        return False

    def _obstacle_relevant_to_side(
        self,
        observation: Observation,
        *,
        obstacle: ObstacleState,
        side: str,
    ) -> bool:
        center_y = observation.road.lane_center_y
        lower_bound = center_y - observation.road.half_width
        upper_bound = center_y + observation.road.half_width
        center_buffer = self.safe_distance + 0.25 * self.vehicle_width
        obstacle_lower = obstacle.y - 0.5 * obstacle.width - self.safe_distance
        obstacle_upper = obstacle.y + 0.5 * obstacle.width + self.safe_distance

        if side == "right":
            corridor_lower = lower_bound
            corridor_upper = center_y + center_buffer
        else:
            corridor_lower = center_y - center_buffer
            corridor_upper = upper_bound
        return obstacle_lower <= corridor_upper and obstacle_upper >= corridor_lower

    def _time_to_collision(self, leader: State, obstacle: ObstacleState) -> float:
        relative_position = np.asarray([obstacle.x - leader.x, obstacle.y - leader.y], dtype=float)
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-9:
            return 0.0

        line_of_sight = relative_position / distance
        leader_velocity = np.asarray(
            [leader.speed * math.cos(leader.yaw), leader.speed * math.sin(leader.yaw)],
            dtype=float,
        )
        obstacle_velocity = np.asarray(
            [obstacle.speed * math.cos(obstacle.yaw), obstacle.speed * math.sin(obstacle.yaw)],
            dtype=float,
        )
        closing_speed = max(float(np.dot(leader_velocity - obstacle_velocity, line_of_sight)), 0.0)
        if closing_speed <= 1e-9:
            return float("inf")
        return distance / closing_speed

    def _risk_score(
        self,
        *,
        min_clearance: float,
        min_ttc: float,
        boundary_margin: float,
    ) -> float:
        clearance_term = np.clip(
            (self.config.clearance_threshold - min_clearance) / self.config.clearance_threshold,
            0.0,
            1.0,
        )
        ttc_term = 0.0
        if math.isfinite(min_ttc):
            ttc_term = float(
                np.clip(
                    (self.config.ttc_threshold - min_ttc) / self.config.ttc_threshold,
                    0.0,
                    1.0,
                )
            )
        boundary_term = float(
            np.clip(
                (self.config.boundary_margin_threshold - boundary_margin)
                / self.config.boundary_margin_threshold,
                0.0,
                1.0,
            )
        )
        return float(max(clearance_term, ttc_term, boundary_term))

    def _is_narrow_passage(
        self,
        observation: Observation,
        front_obstacles: tuple[ObstacleState, ...],
    ) -> bool:
        if not front_obstacles:
            return False

        lower_bound = observation.road.lane_center_y - observation.road.half_width
        upper_bound = observation.road.lane_center_y + observation.road.half_width
        occupied_intervals = sorted(
            (
                max(lower_bound, obstacle.y - 0.5 * obstacle.width),
                min(upper_bound, obstacle.y + 0.5 * obstacle.width),
            )
            for obstacle in front_obstacles
        )
        merged_intervals: list[tuple[float, float]] = []
        for lower, upper in occupied_intervals:
            if not merged_intervals or lower > merged_intervals[-1][1]:
                merged_intervals.append((lower, upper))
                continue
            merged_lower, merged_upper = merged_intervals[-1]
            merged_intervals[-1] = (merged_lower, max(merged_upper, upper))

        free_gaps: list[float] = []
        cursor = lower_bound
        for lower, upper in merged_intervals:
            free_gaps.append(max(lower - cursor, 0.0))
            cursor = max(cursor, upper)
        free_gaps.append(max(upper_bound - cursor, 0.0))

        passage_width = max(free_gaps, default=upper_bound - lower_bound)
        required_width = self.vehicle_width + 2.0 * self.safe_distance + self.config.narrow_passage_margin
        return passage_width <= required_width

    def _preferred_side(
        self,
        observation: Observation,
        front_obstacles: tuple[ObstacleState, ...],
    ) -> str:
        if not front_obstacles:
            return "left"

        upper_bound = observation.road.lane_center_y + observation.road.half_width
        lower_bound = observation.road.lane_center_y - observation.road.half_width
        upper_outer_edge = max(
            [obstacle.y + 0.5 * obstacle.width for obstacle in front_obstacles if obstacle.y >= 0.0],
            default=observation.road.lane_center_y,
        )
        lower_outer_edge = min(
            [obstacle.y - 0.5 * obstacle.width for obstacle in front_obstacles if obstacle.y < 0.0],
            default=observation.road.lane_center_y,
        )
        left_channel = upper_bound - upper_outer_edge
        right_channel = lower_outer_edge - lower_bound
        if left_channel > right_channel + 1e-6:
            return "left"
        if right_channel > left_channel + 1e-6:
            return "right"

        leader = observation.states[0]
        nearest = min(
            front_obstacles,
            key=lambda obstacle: (obstacle.x - leader.x) ** 2 + (obstacle.y - leader.y) ** 2,
        )
        return "right" if nearest.y >= leader.y else "left"

    def _is_stagnated(self, observation: Observation) -> bool:
        if len(self._leader_x_history) < self._leader_x_history.maxlen:
            return False
        progress = self._leader_x_history[-1] - self._leader_x_history[0]
        leader_speed = observation.states[0].speed
        return (
            progress <= self.config.stagnation_progress_threshold
            and leader_speed <= self.config.stagnation_speed_threshold
        )

    def _formation_spacing(self, observation: Observation) -> float:
        if len(observation.desired_offsets) <= 1:
            return max(1.5 * self.vehicle_length, 1e-6)
        non_zero_offsets = [abs(offset[0]) for offset in observation.desired_offsets[1:] if abs(offset[0]) > 1e-6]
        if not non_zero_offsets:
            return max(1.5 * self.vehicle_length, 1e-6)
        return max(min(non_zero_offsets), 1e-6)

    def _team_recovery_metrics(self, observation: Observation) -> tuple[float, float]:
        leader = observation.states[0]
        leader_position = np.asarray([leader.x, leader.y], dtype=float)
        leader_rotation = rotation_matrix(leader.yaw)
        max_lag = 0.0
        max_error = 0.0
        for index, state in enumerate(observation.states[1:], start=1):
            desired_position = leader_position + leader_rotation @ np.asarray(
                observation.desired_offsets[index],
                dtype=float,
            )
            error_vector = desired_position - np.asarray([state.x, state.y], dtype=float)
            max_lag = max(max_lag, max(float(error_vector[0]), 0.0))
            max_error = max(max_error, float(np.linalg.norm(error_vector)))
        return max_error, max_lag

    def _behavior_side(self, behavior: str) -> str | None:
        if behavior.endswith("_left"):
            return "left"
        if behavior.endswith("_right"):
            return "right"
        return None
