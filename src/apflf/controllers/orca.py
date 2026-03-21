"""ORCA-style reciprocal-avoidance baseline controller."""

from __future__ import annotations

import math

import numpy as np

from apflf.controllers.apf_lf import APFLFController
from apflf.decision.mode_base import parse_mode_label
from apflf.env.geometry import normalize_angle
from apflf.utils.types import Action, Observation, ObstacleState, State


class ORCAController(APFLFController):
    """Velocity-obstacle-style baseline with reciprocal avoidance heuristics."""

    def __init__(self, *args, dt: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dt = max(float(dt), 1e-3)
        self.time_horizon = max(1.0, min(4.0, self.config.ttc_threshold))

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        actions: list[Action] = []
        for index, state in enumerate(observation.states):
            target = self._behavior_target(observation=observation, index=index, mode=mode)
            target_speed = self._mode_adjusted_target_speed(
                self._reference_speed(observation, index, mode),
                mode,
            )
            preferred_velocity = self._preferred_velocity(state=state, target=target, speed=target_speed)
            safe_velocity = self._apply_avoidance(
                observation=observation,
                index=index,
                state=state,
                preferred_velocity=preferred_velocity,
                mode=mode,
            )
            actions.append(self._velocity_to_action(state=state, velocity=safe_velocity))
        return tuple(actions)

    def _behavior_target(self, *, observation: Observation, index: int, mode: str) -> np.ndarray:
        if index == 0:
            target = np.asarray([observation.goal_x, observation.road.lane_center_y], dtype=float)
        else:
            target = self._desired_global_position(observation, index, mode)

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior != "follow":
            side_sign = 1.0 if parsed_mode.behavior.endswith("_left") else -1.0
            lateral_bias = side_sign * 0.40 * max(self.config.vehicle_width, 0.5)
            if parsed_mode.behavior.startswith("recover_"):
                lateral_bias *= 0.35
            target[1] = self.road.clamp_lateral_position(
                target[1] + lateral_bias,
                half_extent_y=0.5 * self.config.vehicle_width,
            )
        return target

    def _preferred_velocity(self, *, state: State, target: np.ndarray, speed: float) -> np.ndarray:
        delta = target - np.asarray([state.x, state.y], dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return np.zeros(2, dtype=float)
        direction = delta / distance
        return direction * float(np.clip(speed, self.bounds.speed_min, self.bounds.speed_max))

    def _apply_avoidance(
        self,
        *,
        observation: Observation,
        index: int,
        state: State,
        preferred_velocity: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        adjusted_velocity = preferred_velocity.astype(float, copy=True)
        for peer_index, peer_state in enumerate(observation.states):
            if peer_index == index:
                continue
            adjusted_velocity = self._avoid_entity(
                state=state,
                adjusted_velocity=adjusted_velocity,
                other=peer_state,
                other_length=self.config.vehicle_length,
                other_width=self.config.vehicle_width,
                responsibility=0.5,
                mode=mode,
            )
        for obstacle in observation.obstacles:
            adjusted_velocity = self._avoid_entity(
                state=state,
                adjusted_velocity=adjusted_velocity,
                other=obstacle,
                other_length=obstacle.length,
                other_width=obstacle.width,
                responsibility=1.0,
                mode=mode,
            )

        road_correction = np.clip(
            0.08 * self._road_force(state, road_gain=0.45 * self.config.road_gain),
            -1.5,
            1.5,
        )
        adjusted_velocity += road_correction
        speed = float(np.linalg.norm(adjusted_velocity))
        if speed > self.bounds.speed_max:
            adjusted_velocity *= self.bounds.speed_max / speed
        return adjusted_velocity

    def _avoid_entity(
        self,
        *,
        state: State,
        adjusted_velocity: np.ndarray,
        other: State | ObstacleState,
        other_length: float,
        other_width: float,
        responsibility: float,
        mode: str,
    ) -> np.ndarray:
        relative_position = np.asarray([other.x - state.x, other.y - state.y], dtype=float)
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-9:
            return adjusted_velocity

        normal = relative_position / distance
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        combined_radius = self._effective_radius(self.config.vehicle_length, self.config.vehicle_width) + self._effective_radius(
            other_length,
            other_width,
        )
        combined_radius += 0.15
        gap = distance - combined_radius
        other_velocity = np.asarray(
            [other.speed * math.cos(other.yaw), other.speed * math.sin(other.yaw)],
            dtype=float,
        )
        relative_velocity = adjusted_velocity - responsibility * other_velocity
        closing_speed = float(-np.dot(relative_velocity, normal))
        side_sign = self._side_sign(mode=mode, state=state, other=other)
        tangent_direction = side_sign * tangent

        if gap <= 0.0:
            correction = responsibility * (abs(gap) + 0.4) * (-normal + 0.35 * tangent_direction)
            return adjusted_velocity + correction

        if closing_speed <= 1e-6:
            return adjusted_velocity

        ttc = gap / closing_speed
        if ttc >= self.time_horizon:
            return adjusted_velocity

        urgency = (self.time_horizon - ttc) / self.time_horizon
        correction_scale = responsibility * urgency * max(np.linalg.norm(adjusted_velocity), state.speed, 1.0)
        correction = correction_scale * (-0.55 * normal + 0.85 * tangent_direction)
        return adjusted_velocity + correction

    def _effective_radius(self, length: float, width: float) -> float:
        return 0.35 * math.hypot(length, width)

    def _side_sign(self, *, mode: str, state: State, other: State | ObstacleState) -> float:
        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior.endswith("_left"):
            return 1.0
        if parsed_mode.behavior.endswith("_right"):
            return -1.0
        return -1.0 if other.y >= state.y else 1.0

    def _velocity_to_action(self, *, state: State, velocity: np.ndarray) -> Action:
        heading_direction = np.asarray([math.cos(state.yaw), math.sin(state.yaw)], dtype=float)
        lateral_direction = np.asarray([-math.sin(state.yaw), math.cos(state.yaw)], dtype=float)
        forward_speed = max(float(np.dot(velocity, heading_direction)), 0.0)
        lateral_speed = float(np.dot(velocity, lateral_direction))
        desired_heading = state.yaw + math.atan2(lateral_speed, max(forward_speed, 0.5))
        desired_speed = float(
            np.clip(
                math.hypot(forward_speed, lateral_speed),
                self.bounds.speed_min,
                self.bounds.speed_max,
            )
        )
        heading_error = normalize_angle(desired_heading - state.yaw)
        steer = float(np.clip(self.config.heading_gain * heading_error, self.bounds.steer_min, self.bounds.steer_max))
        accel = float(np.clip(self.config.speed_gain * (desired_speed - state.speed), self.bounds.accel_min, self.bounds.accel_max))
        return Action(accel=accel, steer=steer)
