"""APF 与 Leader-Follower 融合控制器。"""

from __future__ import annotations

import numpy as np

from apflf.controllers.apf import APFController
from apflf.controllers.lf import LeaderFollowerMixin
from apflf.decision.mode_base import parse_mode_label
from apflf.utils.types import Action, Observation, State


class APFLFController(LeaderFollowerMixin, APFController):
    """APF-LF 融合控制器。"""

    def _terminal_recovery_creep_speed(
        self,
        *,
        observation: Observation,
        index: int,
        desired_position: np.ndarray,
    ) -> float:
        """Return a tiny speed floor that lets stopped followers finish lateral regrouping."""

        if index == 0:
            return 0.0

        leader = observation.states[0]
        state = observation.states[index]
        lateral_error = abs(float(desired_position[1] - state.y))
        longitudinal_error = float(desired_position[0] - state.x)
        near_goal_window = max(2.0 * self.config.vehicle_length, 6.0)
        if leader.x < observation.goal_x - near_goal_window:
            return 0.0
        if leader.speed > 0.35 or state.speed > 0.35:
            return 0.0
        if lateral_error <= 0.60 * self.config.vehicle_width:
            return 0.0
        if longitudinal_error < -0.35 * self.config.vehicle_length:
            return 0.0
        for obstacle in observation.obstacles:
            if obstacle.x + 0.5 * obstacle.length < state.x - self.config.vehicle_length:
                continue
            if abs(obstacle.x - state.x) <= 2.0 * self.config.vehicle_length and abs(obstacle.y - state.y) <= (
                observation.road.half_width + 0.5 * obstacle.width
            ):
                return 0.0
        return min(self.bounds.speed_max, 0.35 + 0.18 * lateral_error)

    def _leader_goal_target(self, observation: Observation, state: State) -> np.ndarray:
        """Keep the leader's terminal attraction forward-progressive after it passes goal_x."""

        target_x = observation.goal_x
        if state.x >= observation.goal_x:
            target_x = state.x + max(0.5 * self.config.vehicle_length, 0.25 * max(state.speed, 1.0))
        return np.asarray([target_x, observation.road.lane_center_y], dtype=float)

    def _reference_speed(self, observation: Observation, index: int, mode: str) -> float:
        """构造 leader/follower 的纵向参考速度。"""

        state = observation.states[index]
        parsed_mode = parse_mode_label(mode)
        if index == 0:
            return self._leader_recovery_speed_limit(
                observation=observation,
                target_speed=self._braking_speed(state.x, observation.goal_x),
                mode=mode,
            )

        leader = observation.states[0]
        desired_position = self._desired_global_position(observation, index, mode)
        longitudinal_error = float(desired_position[0] - state.x)
        reference_speed = leader.speed
        if parsed_mode.behavior.startswith("recover_") and index > 1:
            reference_speed = observation.states[index - 1].speed
        raw_speed = reference_speed + self.config.gap_gain * longitudinal_error
        target_speed = float(np.clip(raw_speed, self.bounds.speed_min, self.bounds.speed_max))
        target_speed = max(
            target_speed,
            self._terminal_recovery_creep_speed(
                observation=observation,
                index=index,
                desired_position=desired_position,
            ),
        )
        return target_speed

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """输出 APF-LF 融合名义控制。"""

        actions: list[Action] = []
        repulsive_scale, road_scale, _ = self._mode_gain_scales(mode)
        for index, state in enumerate(observation.states):
            if index == 0:
                target = self._leader_goal_target(observation, state)
            else:
                target = self._desired_global_position(observation, index, mode)
            attractive_force = self._attractive_force(state, target)
            formation_force = self._formation_force(observation, index, mode)
            consensus_force = self._consensus_force(observation, index, mode)
            road_gain = self.config.road_gain * road_scale
            road_force = self._road_force(state, road_gain=road_gain)
            obstacle_force = self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=self.config.repulsive_gain * repulsive_scale,
            )
            peer_force = self._sum_peer_forces(
                observation=observation,
                index=index,
                repulsive_gain=self.config.repulsive_gain * repulsive_scale,
            )
            behavior_force = self._mode_behavior_force(mode=mode, road_gain=road_gain, index=index)
            total_force = (
                attractive_force
                + formation_force
                + consensus_force
                + road_force
                + obstacle_force
                + peer_force
                + behavior_force
            )
            actions.append(
                self._force_to_action(
                    state=state,
                    force=total_force,
                    target_speed=self._mode_adjusted_target_speed(
                        self._reference_speed(observation, index, mode),
                        mode,
                    ),
                )
            )
        return tuple(actions)
