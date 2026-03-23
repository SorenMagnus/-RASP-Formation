"""APF 与 Leader-Follower 融合控制器。"""

from __future__ import annotations

import numpy as np

from apflf.controllers.apf import APFController
from apflf.controllers.lf import LeaderFollowerMixin
from apflf.decision.mode_base import parse_mode_label
from apflf.utils.types import Action, Observation, ObstacleState, State


class APFLFController(LeaderFollowerMixin, APFController):
    """APF-LF 融合控制器。"""

    def _mode_behavior_side_sign(self, mode: str | None) -> float | None:
        """解析 mode 中的行为侧向符号。"""

        parsed_mode = parse_mode_label(mode or "")
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return None
        return 1.0 if parsed_mode.behavior.endswith("_left") else -1.0

    def _leader_passing_side_margins(
        self,
        observation: Observation,
        *,
        obstacle: ObstacleState,
    ) -> tuple[float, float]:
        center_y = observation.road.lane_center_y
        upper_limit = center_y + observation.road.half_width - 0.5 * self.config.vehicle_width
        lower_limit = center_y - observation.road.half_width + 0.5 * self.config.vehicle_width
        bypass_margin = max(0.45, 0.5 * self.config.road_influence_margin)
        inflated_half_width = 0.5 * obstacle.width + 0.5 * self.config.vehicle_width + bypass_margin
        left_target_y = obstacle.y + inflated_half_width
        right_target_y = obstacle.y - inflated_half_width
        return (upper_limit - left_target_y, right_target_y - lower_limit)

    def _leader_nonrelevant_clearance_activation(
        self,
        *,
        state: State,
        obstacle: ObstacleState,
    ) -> float:
        """Return a smooth activation once a non-relevant blocker is locally cleared longitudinally."""

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
        rear_gap = obstacle_rear_x - state_front_x
        activation_start_gap = 2.5
        activation_full_overlap = 1.5
        if rear_gap >= activation_start_gap:
            return 0.0
        raw_activation = float(
            np.clip(
                (activation_start_gap - rear_gap)
                / max(activation_start_gap + activation_full_overlap, 1e-6),
                0.0,
                1.0,
            )
        )
        return float(raw_activation * raw_activation * (3.0 - 2.0 * raw_activation))

    def _leader_front_obstacles(
        self,
        observation: Observation,
        state: State,
    ) -> tuple[ObstacleState, ...]:
        front: list[ObstacleState] = []
        state_front_x = state.x + 0.5 * self.config.vehicle_length
        lookahead_distance = max(
            1.5 * self.config.obstacle_influence_distance,
            5.0 * self.config.vehicle_length,
        )
        for obstacle in observation.obstacles:
            obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
            longitudinal_gap = obstacle_rear_x - state_front_x
            if longitudinal_gap < -0.5 * self.config.vehicle_length:
                continue
            if longitudinal_gap > lookahead_distance:
                continue
            front.append(obstacle)
        front.sort(
            key=lambda obstacle: (
                max(obstacle.x - 0.5 * obstacle.length - state_front_x, 0.0),
                abs(obstacle.y - state.y),
            )
        )
        return tuple(front)

    def _leader_behavior_side_sign(
        self,
        observation: Observation,
        state: State,
        mode: str | None,
        *,
        front_obstacles: tuple[ObstacleState, ...],
    ) -> float | None:
        side_sign = self._mode_behavior_side_sign(mode)
        if side_sign is None:
            return None
        if not front_obstacles:
            return side_sign

        nominal_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        if not nominal_relevant:
            return side_sign

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        side_switch_obstacles = tuple(
            obstacle
            for obstacle in nominal_relevant
            if (obstacle.x - 0.5 * obstacle.length - state_front_x) >= (-0.1 * self.config.vehicle_length)
        )
        anchor_obstacle = side_switch_obstacles[0] if side_switch_obstacles else nominal_relevant[0]
        left_margin, right_margin = self._leader_passing_side_margins(
            observation,
            obstacle=anchor_obstacle,
        )
        preferred_margin = left_margin if side_sign > 0.0 else right_margin
        alternate_margin = right_margin if side_sign > 0.0 else left_margin
        if preferred_margin < 0.0 and alternate_margin > preferred_margin + 0.15:
            return -side_sign
        center_y = observation.road.lane_center_y
        nominal_offset = side_sign * (state.y - center_y)
        commitment_threshold = max(0.34 * self.config.vehicle_width, 0.18 * observation.road.half_width)
        if nominal_offset < commitment_threshold:
            return side_sign

        alternate_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=-side_sign,
        )
        if not alternate_relevant:
            return side_sign

        nominal_anchor = min(
            nominal_relevant,
            key=lambda obstacle: max(obstacle.x - 0.5 * obstacle.length - state_front_x, 0.0),
        )
        alternate_anchor = min(
            alternate_relevant,
            key=lambda obstacle: max(obstacle.x - 0.5 * obstacle.length - state_front_x, 0.0),
        )
        nominal_rear_gap = nominal_anchor.x - 0.5 * nominal_anchor.length - state_front_x
        alternate_rear_gap = alternate_anchor.x - 0.5 * alternate_anchor.length - state_front_x
        flip_gap_threshold = max(0.12 * self.config.vehicle_length, 0.55)
        alternate_lookahead = max(0.5 * self.config.obstacle_influence_distance, 1.5 * self.config.vehicle_length)
        if nominal_rear_gap <= flip_gap_threshold and 0.0 <= alternate_rear_gap <= alternate_lookahead:
            return -side_sign
        return side_sign

    def _leader_relevant_obstacles(
        self,
        observation: Observation,
        *,
        front_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
    ) -> tuple[ObstacleState, ...]:
        """返回当前绕行侧真正决定局部 waypoint 的障碍物子集。"""

        center_y = observation.road.lane_center_y
        relevant_threshold = 0.5 * self.config.vehicle_width
        return tuple(
            obstacle
            for obstacle in front_obstacles
            if (
                obstacle.y <= center_y + relevant_threshold
                if side_sign > 0.0
                else obstacle.y >= center_y - relevant_threshold
            )
        )

    def _leader_side_channel_center_y(
        self,
        observation: Observation,
        *,
        relevant_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
    ) -> float:
        """Return the centerline of the active passing-side corridor for the ego-vehicle center."""

        center_y = observation.road.lane_center_y
        upper_limit = center_y + observation.road.half_width - 0.5 * self.config.vehicle_width
        lower_limit = center_y - observation.road.half_width + 0.5 * self.config.vehicle_width
        bypass_margin = max(0.45, 0.5 * self.config.road_influence_margin)

        if side_sign > 0.0:
            corridor_lower = center_y
            for obstacle in relevant_obstacles:
                inflated_half_width = (
                    0.5 * obstacle.width + 0.5 * self.config.vehicle_width + bypass_margin
                )
                corridor_lower = max(corridor_lower, obstacle.y + inflated_half_width)
            corridor_lower = min(corridor_lower, upper_limit)
            return float(0.5 * (corridor_lower + upper_limit))

        corridor_upper = center_y
        for obstacle in relevant_obstacles:
            inflated_half_width = (
                0.5 * obstacle.width + 0.5 * self.config.vehicle_width + bypass_margin
            )
            corridor_upper = min(corridor_upper, obstacle.y - inflated_half_width)
        corridor_upper = max(corridor_upper, lower_limit)
        return float(0.5 * (lower_limit + corridor_upper))

    def _leader_channel_centerline_blend(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str | None,
        front_obstacles: tuple[ObstacleState, ...],
        relevant_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
    ) -> float:
        """Blend edge-tracking toward the corridor centerline after the opposite blocker is locally cleared."""

        relevant_ids = {obstacle.obstacle_id for obstacle in relevant_obstacles}
        blend = 0.0
        for obstacle in front_obstacles:
            if obstacle.obstacle_id in relevant_ids:
                continue
            blend = max(
                blend,
                self._leader_nonrelevant_clearance_activation(
                    state=state,
                    obstacle=obstacle,
                ),
            )

        nominal_side_sign = self._mode_behavior_side_sign(mode)
        if nominal_side_sign is not None and nominal_side_sign != side_sign and relevant_obstacles:
            blend = max(blend, 0.35)
        return float(np.clip(blend, 0.0, 1.0))

    def _leader_flip_overshoot(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str | None,
        side_sign: float,
        relevant_obstacles: tuple[ObstacleState, ...],
    ) -> float:
        """在局部翻边时追加一个有界 overshoot，避免 leader 太晚回收。"""

        nominal_side_sign = self._mode_behavior_side_sign(mode)
        if nominal_side_sign is None or nominal_side_sign == side_sign or not relevant_obstacles:
            return 0.0

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        nearest_gap = min(
            obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in relevant_obstacles
        )
        flip_distance = max(1.25 * self.config.vehicle_length, 0.55 * self.config.obstacle_influence_distance)
        activation = float(np.clip((flip_distance - nearest_gap) / max(flip_distance, 1e-6), 0.0, 1.0))
        if activation <= 0.0:
            return 0.0

        max_overshoot = max(0.55 * self.config.vehicle_width, 0.35 * self.config.road_influence_margin)
        return float(side_sign * activation * max_overshoot)

    def _leader_hazard_target_x(
        self,
        observation: Observation,
        state: State,
        *,
        mode: str | None,
        side_sign: float,
        relevant_obstacles: tuple[ObstacleState, ...],
        target_y: float | None = None,
    ) -> float:
        """为 hazard 绕行阶段选择近端 waypoint，减少远端 goal_x 对横向转向的稀释。"""

        far_preview_x = float(
            min(
                observation.goal_x,
                state.x + max(self.config.obstacle_influence_distance, 2.5 * self.config.vehicle_length),
            )
        )
        nominal_side_sign = self._mode_behavior_side_sign(mode)
        if nominal_side_sign is None or nominal_side_sign == side_sign:
            if not relevant_obstacles:
                return far_preview_x

            if target_y is None:
                target_y = self._leader_behavior_target_y(observation, state, mode)
            if target_y is None:
                return far_preview_x

            lateral_error = abs(float(target_y - state.y))
            lateral_trigger = max(0.35 * self.config.vehicle_width, 0.18 * observation.road.half_width)
            if lateral_error <= lateral_trigger:
                return far_preview_x

            nearest_obstacle = min(
                relevant_obstacles,
                key=lambda obstacle: obstacle.x - 0.5 * obstacle.length,
            )
            state_front_x = state.x + 0.5 * self.config.vehicle_length
            rear_gap = nearest_obstacle.x - 0.5 * nearest_obstacle.length - state_front_x
            preview_engage_distance = max(0.55 * self.config.obstacle_influence_distance, 2.0 * self.config.vehicle_length)
            activation = float(
                np.clip(
                    (preview_engage_distance - rear_gap) / max(preview_engage_distance, 1e-6),
                    0.0,
                    1.0,
                )
            )
            if activation <= 0.0:
                return far_preview_x

            near_preview_x = float(
                min(
                    observation.goal_x,
                    max(state.x + 1.0 * self.config.vehicle_length, nearest_obstacle.x),
                )
            )
            return float((1.0 - activation) * far_preview_x + activation * near_preview_x)

        if not relevant_obstacles:
            return observation.goal_x

        nearest_obstacle = min(
            relevant_obstacles,
            key=lambda obstacle: obstacle.x - 0.5 * obstacle.length,
        )
        local_waypoint_x = max(
            state.x + 1.0 * self.config.vehicle_length,
            nearest_obstacle.x,
        )
        return float(min(observation.goal_x, local_waypoint_x))

    def _leader_behavior_target_y(
        self,
        observation: Observation,
        state: State,
        mode: str | None,
    ) -> float | None:
        front_obstacles = self._leader_front_obstacles(observation, state)
        side_sign = self._leader_behavior_side_sign(
            observation,
            state,
            mode,
            front_obstacles=front_obstacles,
        )
        if side_sign is None:
            return None

        center_y = observation.road.lane_center_y
        relevant_obstacles = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        if not relevant_obstacles:
            return None

        bypass_margin = max(0.45, 0.5 * self.config.road_influence_margin)
        edge_target_y = center_y
        for obstacle in relevant_obstacles:
            inflated_half_width = 0.5 * obstacle.width + 0.5 * self.config.vehicle_width + bypass_margin
            candidate_y = obstacle.y + side_sign * inflated_half_width
            if side_sign > 0.0:
                edge_target_y = max(edge_target_y, candidate_y)
            else:
                edge_target_y = min(edge_target_y, candidate_y)

        edge_target_y += self._leader_flip_overshoot(
            observation=observation,
            state=state,
            mode=mode,
            side_sign=side_sign,
            relevant_obstacles=relevant_obstacles,
        )
        channel_center_y = self._leader_side_channel_center_y(
            observation,
            relevant_obstacles=relevant_obstacles,
            side_sign=side_sign,
        )
        centerline_blend = self._leader_channel_centerline_blend(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            relevant_obstacles=relevant_obstacles,
            side_sign=side_sign,
        )
        target_y = (1.0 - centerline_blend) * edge_target_y + centerline_blend * channel_center_y
        max_center_offset = max(observation.road.half_width - 0.55 * self.config.vehicle_width, 0.0)
        return float(np.clip(target_y, center_y - max_center_offset, center_y + max_center_offset))

    def _leader_bypass_force(
        self,
        observation: Observation,
        state: State,
        mode: str | None,
        *,
        target_y: float | None = None,
        road_gain: float | None = None,
    ) -> np.ndarray:
        parsed_mode = parse_mode_label(mode or "")
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return np.zeros(2, dtype=float)

        if target_y is None:
            target_y = self._leader_behavior_target_y(observation, state, mode)
        if target_y is None:
            return np.zeros(2, dtype=float)

        lateral_error = float(target_y - state.y)
        if abs(lateral_error) <= 1e-6:
            return np.zeros(2, dtype=float)

        guidance_gain = max(0.45 * self.config.road_gain, 1.5 * self.config.attraction_gain)
        max_guidance = max(0.75 * self.config.road_gain, 2.5)
        force_y = guidance_gain * lateral_error
        if road_gain is not None:
            road_force_y = float(self._road_force(state, road_gain=road_gain)[1])
            opposing_road = max(0.0, -np.sign(lateral_error) * road_force_y)
            road_compensation = min(0.45 * opposing_road, 0.25 * road_gain)
            force_y += float(np.sign(lateral_error) * road_compensation)
            max_guidance += road_compensation

        force_y = float(np.clip(force_y, -max_guidance, max_guidance))
        return np.asarray([0.0, force_y], dtype=float)

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

    def _leader_goal_target(
        self,
        observation: Observation,
        state: State,
        mode: str | None = None,
    ) -> np.ndarray:
        """Keep the leader's terminal attraction forward-progressive after it passes goal_x."""

        target_x = observation.goal_x
        if state.x >= observation.goal_x:
            target_x = state.x + max(0.5 * self.config.vehicle_length, 0.25 * max(state.speed, 1.0))
        target_y = self._leader_behavior_target_y(observation, state, mode)
        parsed_mode = parse_mode_label(mode or "")
        if parsed_mode.behavior != "follow" and not parsed_mode.behavior.startswith("recover_"):
            front_obstacles = self._leader_front_obstacles(observation, state)
            side_sign = self._leader_behavior_side_sign(
                observation,
                state,
                mode,
                front_obstacles=front_obstacles,
            )
            if side_sign is not None:
                relevant_obstacles = self._leader_relevant_obstacles(
                    observation,
                    front_obstacles=front_obstacles,
                    side_sign=side_sign,
                )
                target_x = self._leader_hazard_target_x(
                    observation,
                    state,
                    mode=mode,
                    side_sign=side_sign,
                    relevant_obstacles=relevant_obstacles,
                    target_y=target_y,
                )
        if target_y is None:
            target_y = observation.road.lane_center_y
        return np.asarray([target_x, target_y], dtype=float)

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
            road_gain = self.config.road_gain * road_scale
            if index == 0:
                target = self._leader_goal_target(observation, state, mode)
                leader_guidance_force = self._leader_bypass_force(
                    observation,
                    state,
                    mode,
                    target_y=float(target[1]),
                    road_gain=road_gain,
                )
            else:
                target = self._desired_global_position(observation, index, mode)
                leader_guidance_force = np.zeros(2, dtype=float)
            attractive_force = self._attractive_force(state, target)
            formation_force = self._formation_force(observation, index, mode)
            consensus_force = self._consensus_force(observation, index, mode)
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
                + leader_guidance_force
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
