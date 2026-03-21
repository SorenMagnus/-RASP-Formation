"""Dynamic-window baseline controller."""

from __future__ import annotations

import math

import numpy as np

from apflf.controllers.apf_lf import APFLFController
from apflf.decision.mode_base import parse_mode_label
from apflf.env.dynamics import VehicleDynamics
from apflf.env.geometry import box_clearance, normalize_angle
from apflf.utils.types import Action, Observation, ObstacleState, State


class DWAController(APFLFController):
    """Sample-and-rollout dynamic-window baseline."""

    def __init__(self, *args, wheelbase: float = 2.8, dt: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dt = max(float(dt), 1e-3)
        self.dynamics = VehicleDynamics(wheelbase=float(wheelbase), bounds=self.bounds)
        self.rollout_steps = max(3, int(math.ceil(0.6 / self.dt)))

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        actions: list[Action] = []
        for index, state in enumerate(observation.states):
            target = self._behavior_target(observation=observation, index=index, mode=mode)
            target_speed = self._mode_adjusted_target_speed(
                self._reference_speed(observation, index, mode),
                mode,
            )
            target_speed = self._hazard_speed_limit(
                observation=observation,
                index=index,
                mode=mode,
                target_speed=target_speed,
            )
            actions.append(
                self._select_action(
                    observation=observation,
                    index=index,
                    state=state,
                    target=target,
                    target_speed=target_speed,
                    mode=mode,
                )
            )
        return tuple(actions)

    def _behavior_target(self, *, observation: Observation, index: int, mode: str) -> np.ndarray:
        if index == 0:
            target = np.asarray([observation.goal_x, observation.road.lane_center_y], dtype=float)
            parsed_mode = parse_mode_label(mode)
            if parsed_mode.behavior != "follow":
                side_sign = 1.0 if parsed_mode.behavior.endswith("_left") else -1.0
                lateral_bias = side_sign * 0.45 * max(self.config.vehicle_width, 0.5)
                if parsed_mode.behavior.startswith("recover_"):
                    lateral_bias *= 0.45
                target[1] = self.road.clamp_lateral_position(
                    target[1] + lateral_bias,
                    half_extent_y=0.5 * self.config.vehicle_width,
                )
            return target
        return self._desired_global_position(observation, index, mode)

    def _hazard_speed_limit(
        self,
        *,
        observation: Observation,
        index: int,
        mode: str,
        target_speed: float,
    ) -> float:
        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow":
            return float(target_speed)
        hazard_distance = self._front_hazard_distance(observation=observation, index=index)
        capped_speed = min(target_speed, 0.45 * self.target_speed + 0.35 * max(hazard_distance, 0.0))
        if parsed_mode.behavior.startswith("recover_"):
            capped_speed = min(target_speed, 0.65 * self.target_speed)
        return float(np.clip(capped_speed, self.bounds.speed_min, self.bounds.speed_max))

    def _select_action(
        self,
        *,
        observation: Observation,
        index: int,
        state: State,
        target: np.ndarray,
        target_speed: float,
        mode: str,
    ) -> Action:
        best_action: Action | None = None
        best_score = -float("inf")

        for candidate in self._sample_actions(state=state, target=target, target_speed=target_speed):
            score = self._candidate_score(
                observation=observation,
                index=index,
                state=state,
                action=candidate,
                target=target,
                target_speed=target_speed,
                mode=mode,
            )
            if score > best_score:
                best_score = score
                best_action = candidate

        if best_action is not None:
            return best_action

        desired_heading = math.atan2(float(target[1] - state.y), max(float(target[0] - state.x), 1e-6))
        steer = float(
            np.clip(
                self.config.heading_gain * normalize_angle(desired_heading - state.yaw),
                self.bounds.steer_min,
                self.bounds.steer_max,
            )
        )
        accel = float(np.clip(self.bounds.accel_min, self.bounds.accel_min, self.bounds.accel_max))
        return Action(accel=accel, steer=steer)

    def _sample_actions(self, *, state: State, target: np.ndarray, target_speed: float) -> tuple[Action, ...]:
        desired_heading = math.atan2(float(target[1] - state.y), max(float(target[0] - state.x), 1e-6))
        desired_steer = self.config.heading_gain * normalize_angle(desired_heading - state.yaw)
        desired_accel = self.config.speed_gain * (target_speed - state.speed)
        accel_candidates = self._unique_clipped_values(
            values=(
                0.0,
                self.bounds.accel_min,
                desired_accel - 1.0,
                desired_accel,
                desired_accel + 1.0,
            ),
            lower=self.bounds.accel_min,
            upper=self.bounds.accel_max,
        )
        side_hint = 1.0 if target[1] >= state.y else -1.0
        steer_candidates = self._unique_clipped_values(
            values=(
                0.0,
                side_hint * 0.18,
                desired_steer - 0.18,
                desired_steer,
                desired_steer + 0.18,
            ),
            lower=self.bounds.steer_min,
            upper=self.bounds.steer_max,
        )
        return tuple(Action(accel=accel, steer=steer) for accel in accel_candidates for steer in steer_candidates)

    def _unique_clipped_values(
        self,
        *,
        values: tuple[float, ...],
        lower: float,
        upper: float,
    ) -> tuple[float, ...]:
        clipped_values: list[float] = []
        for value in values:
            clipped = float(np.clip(value, lower, upper))
            if any(abs(clipped - existing) <= 1e-6 for existing in clipped_values):
                continue
            clipped_values.append(clipped)
        return tuple(clipped_values)

    def _candidate_score(
        self,
        *,
        observation: Observation,
        index: int,
        state: State,
        action: Action,
        target: np.ndarray,
        target_speed: float,
        mode: str,
    ) -> float:
        trajectory = self._rollout(state=state, action=action)
        min_clearance, min_boundary_margin = self._trajectory_safety(
            observation=observation,
            index=index,
            trajectory=trajectory,
        )
        parsed_mode = parse_mode_label(mode)
        clearance_floor = 0.35 if parsed_mode.behavior != "follow" else 0.10
        boundary_floor = 0.18 if parsed_mode.behavior != "follow" else 0.05
        if min_clearance <= clearance_floor or min_boundary_margin < boundary_floor:
            return -float("inf")

        terminal_state = trajectory[-1]
        start_position = np.asarray([state.x, state.y], dtype=float)
        terminal_position = np.asarray([terminal_state.x, terminal_state.y], dtype=float)
        target_vector = target - start_position
        target_distance = float(np.linalg.norm(target_vector))
        if target_distance <= 1e-9:
            target_direction = np.asarray([1.0, 0.0], dtype=float)
        else:
            target_direction = target_vector / target_distance
        progress = float(np.dot(terminal_position - start_position, target_direction))
        goal_distance = float(np.linalg.norm(target - terminal_position))
        clearance_term = float(
            np.clip(min_clearance / max(self.config.obstacle_influence_distance, 1e-6), 0.0, 1.0)
        )
        boundary_term = float(
            np.clip(min_boundary_margin / max(self.config.road_influence_margin, 1e-6), 0.0, 1.0)
        )
        speed_term = abs(target_speed - terminal_state.speed) / max(self.bounds.speed_max, 1e-6)
        effort_term = (
            abs(action.accel) / max(abs(self.bounds.accel_min), abs(self.bounds.accel_max), 1e-6)
            + abs(action.steer) / max(abs(self.bounds.steer_min), abs(self.bounds.steer_max), 1e-6)
        )
        mode_bonus = 0.0
        if parsed_mode.behavior.startswith("recover_") and index == 0:
            max_lag, _, _ = self._team_alignment_metrics(observation)
            mode_bonus = -0.25 * float(np.clip(max_lag / max(self._formation_spacing(observation), 1e-6), 0.0, 2.0))
        if parsed_mode.behavior != "follow":
            side_sign = 1.0 if parsed_mode.behavior.endswith("_left") else -1.0
            mode_bonus += 0.50 * side_sign * action.steer / max(abs(self.bounds.steer_max), 1e-6)
            hazard_distance = self._front_hazard_distance(observation=observation, index=index)
            if hazard_distance < self.config.obstacle_influence_distance:
                mode_bonus -= 0.35 * max(action.accel, 0.0) / max(self.bounds.accel_max, 1e-6)
        normalized_progress = progress / max(target_distance, 1.0)
        normalized_goal_error = goal_distance / max(target_distance, 1.0)
        return float(
            2.2 * normalized_progress
            - 1.4 * normalized_goal_error
            + 0.9 * clearance_term
            + 0.5 * boundary_term
            - 0.35 * speed_term
            - 0.08 * effort_term
            + mode_bonus
        )

    def _rollout(self, *, state: State, action: Action) -> tuple[State, ...]:
        trajectory: list[State] = []
        rolled_state = state
        for _ in range(self.rollout_steps):
            rolled_state = self.dynamics.step(state=rolled_state, action=action, dt=self.dt)
            trajectory.append(rolled_state)
        return tuple(trajectory)

    def _trajectory_safety(
        self,
        *,
        observation: Observation,
        index: int,
        trajectory: tuple[State, ...],
    ) -> tuple[float, float]:
        min_clearance = float("inf")
        min_boundary_margin = float("inf")

        for step_offset, rolled_state in enumerate(trajectory, start=1):
            preview_time = step_offset * self.dt
            min_boundary_margin = min(
                min_boundary_margin,
                self.road.boundary_margin(
                    rolled_state.y,
                    half_extent_y=0.5 * self.config.vehicle_width,
                ),
            )
            for peer_index, peer_state in enumerate(observation.states):
                if peer_index == index:
                    continue
                predicted_peer = self._predict_entity_state(peer_state, preview_time)
                min_clearance = min(
                    min_clearance,
                    box_clearance(
                        rolled_state,
                        self.config.vehicle_length,
                        self.config.vehicle_width,
                        predicted_peer,
                        self.config.vehicle_length,
                        self.config.vehicle_width,
                    ),
                )
            for obstacle in observation.obstacles:
                predicted_obstacle = self._predict_entity_state(obstacle, preview_time)
                min_clearance = min(
                    min_clearance,
                    box_clearance(
                        rolled_state,
                        self.config.vehicle_length,
                        self.config.vehicle_width,
                        predicted_obstacle,
                        obstacle.length,
                        obstacle.width,
                    ),
                )

        if math.isinf(min_clearance):
            min_clearance = self.config.obstacle_influence_distance
        if math.isinf(min_boundary_margin):
            min_boundary_margin = self.road.geometry.half_width
        return float(min_clearance), float(min_boundary_margin)

    def _front_hazard_distance(self, *, observation: Observation, index: int) -> float:
        state = observation.states[index]
        ego_front_x = state.x + 0.5 * self.config.vehicle_length
        candidates: list[float] = []
        for peer_index, peer_state in enumerate(observation.states):
            if peer_index == index:
                continue
            peer_rear_x = peer_state.x - 0.5 * self.config.vehicle_length
            if peer_rear_x < ego_front_x:
                continue
            lateral_gap = abs(peer_state.y - state.y)
            if lateral_gap > 0.8 * self.config.vehicle_width:
                continue
            candidates.append(peer_rear_x - ego_front_x)
        for obstacle in observation.obstacles:
            obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
            if obstacle_rear_x < ego_front_x:
                continue
            lateral_gap = abs(obstacle.y - state.y)
            if lateral_gap > 0.5 * (self.config.vehicle_width + obstacle.width):
                continue
            candidates.append(obstacle_rear_x - ego_front_x)
        if not candidates:
            return self.config.obstacle_influence_distance
        return float(min(candidates))


    def _predict_entity_state(self, entity: State | ObstacleState, preview_time: float) -> State:
        return State(
            x=entity.x + preview_time * entity.speed * math.cos(entity.yaw),
            y=entity.y + preview_time * entity.speed * math.sin(entity.yaw),
            yaw=entity.yaw,
            speed=entity.speed,
        )
