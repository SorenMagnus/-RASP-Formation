"""风险自适应 APF-LF 控制器。"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from apflf.controllers.apf_lf import APFLFController
from apflf.controllers.apf_st import spatio_temporal_factor
from apflf.decision.mode_base import parse_mode_label
from apflf.utils.types import Action, Observation, ObstacleState, State


class AdaptiveAPFController(APFLFController):
    """风险自适应 ST-APF-LF 主控制器。

    理论映射:
        - 风险指标: 由最小间距、闭合速度、TTC 与道路边界共同构造。
        - 有界调度: 对排斥增益与道路边界增益进行平滑有界投影。
        - 停滞检测: 以进度、速度与力范数联合判定，并通过冷却计数防抖。
    """

    def __init__(self, *args, **kwargs) -> None:
        """初始化自适应控制器内部状态。"""

        super().__init__(*args, **kwargs)
        self._leader_goal_errors: deque[float] = deque(maxlen=self.config.stagnation_steps + 1)
        self._stagnation_counter = 0
        self._stagnation_cooldown = 0
        self._escape_steps_remaining = 0
        self._escape_direction = 1.0

    def compute_risk_score(
        self,
        *,
        clearance: float,
        closing_speed: float,
        ttc: float,
        boundary_margin: float,
    ) -> float:
        """计算 0 到 1 之间的风险分数。"""

        distance_term = math.exp(-max(clearance, 0.0) / self.config.risk_distance_scale)
        speed_term = np.clip(closing_speed / self.config.risk_speed_scale, 0.0, 1.0)
        if math.isfinite(ttc):
            ttc_term = np.clip(
                (self.config.risk_ttc_threshold - ttc) / self.config.risk_ttc_threshold,
                0.0,
                1.0,
            )
        else:
            ttc_term = 0.0
        boundary_term = np.clip(
            (self.config.road_influence_margin - boundary_margin)
            / max(self.config.road_influence_margin, 1e-6),
            0.0,
            1.0,
        )
        score = 0.45 * distance_term + 0.2 * float(speed_term) + 0.2 * float(ttc_term) + 0.15 * float(boundary_term)
        return float(np.clip(score, 0.0, 1.0))

    def schedule_gains(self, risk_score: float) -> tuple[float, float]:
        """根据风险分数调度有界势场参数。"""

        activation = self._smooth_activation(risk_score)
        repulsive_gain = np.clip(
            self.config.repulsive_gain * (1.0 + self.config.adaptive_alpha * activation),
            self.config.repulsive_gain_min,
            self.config.repulsive_gain_max,
        )
        road_gain = np.clip(
            self.config.road_gain * (1.0 + 0.6 * self.config.adaptive_alpha * activation),
            self.config.road_gain_min,
            self.config.road_gain_max,
        )
        return float(repulsive_gain), float(road_gain)

    def update_stagnation(self, *, progress_delta: float, speed: float, force_norm: float) -> bool:
        """更新停滞检测状态并返回是否触发逃逸。"""

        if self._stagnation_cooldown > 0:
            self._stagnation_cooldown -= 1

        slow_progress = progress_delta <= self.config.stagnation_progress_threshold
        slow_speed = speed <= self.config.stagnation_speed_threshold
        weak_force = force_norm <= self.config.stagnation_force_threshold
        if slow_progress and slow_speed and weak_force:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0

        if (
            self._stagnation_cooldown == 0
            and self._stagnation_counter >= self.config.stagnation_steps
        ):
            self._stagnation_counter = 0
            self._stagnation_cooldown = self.config.stagnation_cooldown_steps
            self._escape_steps_remaining = self.config.stagnation_cooldown_steps
            return True
        return False

    def _smooth_activation(self, risk_score: float) -> float:
        """构造低风险回归 baseline 的平滑有界激活函数。"""

        clamped_risk = float(np.clip(risk_score, 0.0, 1.0))
        slope = self.config.risk_sigmoid_slope
        raw = 1.0 / (1.0 + math.exp(-slope * (clamped_risk - self.config.risk_reference)))
        baseline = 1.0 / (1.0 + math.exp(slope * self.config.risk_reference))
        normalized = (raw - baseline) / max(1.0 - baseline, 1e-6)
        return float(np.clip(normalized, 0.0, 1.0))

    def _dynamic_obstacle_factor(self, state: State, obstacle: ObstacleState) -> float:
        """复用 ST-APF 时空增强项。"""

        return spatio_temporal_factor(
            state=state,
            obstacle=obstacle,
            st_velocity_gain=self.config.st_velocity_gain,
            ttc_gain=self.config.ttc_gain,
            ttc_threshold=self.config.ttc_threshold,
        )

    def _risk_features(
        self,
        observation: Observation,
        state: State,
    ) -> tuple[float, float, float, float]:
        """提取最小间距、最大闭合速度、最小 TTC 与边界余量。"""

        min_clearance = float("inf")
        max_closing_speed = 0.0
        min_ttc = float("inf")
        for obstacle in observation.obstacles:
            relative_position = np.asarray([obstacle.x - state.x, obstacle.y - state.y], dtype=float)
            distance = float(np.linalg.norm(relative_position))
            clearance = max(distance - 0.5 * (obstacle.width + self.config.vehicle_width), 0.0)
            min_clearance = min(min_clearance, clearance)
            if distance <= 1e-9:
                max_closing_speed = max(max_closing_speed, state.speed + obstacle.speed)
                min_ttc = 0.0
                continue
            line_of_sight = relative_position / distance
            self_velocity = self._vehicle_velocity(state)
            obstacle_velocity = self._obstacle_velocity(obstacle)
            closing_speed = max(float(np.dot(self_velocity - obstacle_velocity, line_of_sight)), 0.0)
            max_closing_speed = max(max_closing_speed, closing_speed)
            if closing_speed > 1e-9:
                min_ttc = min(min_ttc, distance / closing_speed)

        if not observation.obstacles:
            min_clearance = 10.0 * self.config.risk_distance_scale

        boundary_margin = self.road.boundary_margin(
            state.y,
            half_extent_y=0.5 * self.config.vehicle_width,
        )
        return min_clearance, max_closing_speed, min_ttc, boundary_margin

    def _escape_force(self, state: State, observation: Observation) -> np.ndarray:
        """在停滞时生成轻量横向逃逸力。"""

        if self._escape_steps_remaining <= 0:
            return np.zeros(2, dtype=float)
        self._escape_steps_remaining -= 1
        centerline_error = state.y - observation.road.lane_center_y
        sign = self._escape_direction
        if abs(centerline_error) > 1e-6:
            sign = -math.copysign(1.0, centerline_error)
        return np.asarray([0.0, sign * 0.6 * self.config.road_gain], dtype=float)

    def _leader_nonrelevant_lateral_reduction(
        self,
        *,
        state: State,
        obstacle: ObstacleState,
    ) -> float:
        """Smoothly taper the adverse lateral push from a nearly cleared non-relevant blocker."""

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
        rear_gap = obstacle_rear_x - state_front_x
        reduction_start_gap = 1.2
        full_reduction_overlap = 1.8
        if rear_gap >= reduction_start_gap:
            return 0.0
        activation = float(
            np.clip(
                (reduction_start_gap - rear_gap) / max(reduction_start_gap + full_reduction_overlap, 1e-6),
                0.0,
                1.0,
            )
        )
        smooth_activation = activation * activation * (3.0 - 2.0 * activation)
        return 0.65 * smooth_activation

    def _shape_leader_nonrelevant_obstacle_force(
        self,
        *,
        state: State,
        obstacle: ObstacleState,
        side_sign: float,
        force: np.ndarray,
    ) -> np.ndarray:
        """Keep longitudinal repulsion intact while tapering only the adverse lateral component."""

        if side_sign * float(force[1]) >= 0.0:
            return force
        reduction = self._leader_nonrelevant_lateral_reduction(state=state, obstacle=obstacle)
        if reduction <= 0.0:
            return force
        return np.asarray([float(force[0]), (1.0 - reduction) * float(force[1])], dtype=float)

    def _adaptive_obstacle_force(
        self,
        *,
        observation: Observation,
        index: int,
        mode: str,
        repulsive_gain: float,
    ) -> np.ndarray:
        """Apply leader-only hazard shaping to obstacle repulsion and keep all other cases unchanged."""

        if index != 0:
            return self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
            )

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
            )

        state = observation.states[index]
        front_obstacles = self._leader_front_obstacles(observation, state)
        if not front_obstacles:
            return self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
            )

        side_sign = self._leader_behavior_side_sign(
            observation,
            state,
            mode,
            front_obstacles=front_obstacles,
        )
        if side_sign is None:
            return self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
            )

        relevant_obstacles = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        if not relevant_obstacles:
            return self._sum_obstacle_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
            )

        front_ids = {obstacle.obstacle_id for obstacle in front_obstacles}
        relevant_ids = {obstacle.obstacle_id for obstacle in relevant_obstacles}
        total = np.zeros(2, dtype=float)
        for obstacle in observation.obstacles:
            force = self._obstacle_force(state, obstacle, repulsive_gain=repulsive_gain)
            if obstacle.obstacle_id in front_ids and obstacle.obstacle_id not in relevant_ids:
                force = self._shape_leader_nonrelevant_obstacle_force(
                    state=state,
                    obstacle=obstacle,
                    side_sign=side_sign,
                    force=force,
                )
            total += force
        return total

    def _leader_hazard_speed_limit(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        base_target_speed: float,
    ) -> float:
        """Throttle leader forward speed in hazard traversal when lateral reorientation is still incomplete."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return base_target_speed

        front_obstacles = self._leader_front_obstacles(observation, state)
        if not front_obstacles:
            return base_target_speed

        side_sign = self._leader_behavior_side_sign(
            observation,
            state,
            mode,
            front_obstacles=front_obstacles,
        )
        if side_sign is None:
            return base_target_speed

        relevant_obstacles = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        if not relevant_obstacles:
            return base_target_speed

        target_y = self._leader_behavior_target_y(observation, state, mode)
        if target_y is None:
            return base_target_speed

        nominal_side_sign = self._mode_behavior_side_sign(mode)
        lateral_error = abs(float(target_y - state.y))
        local_flip_active = nominal_side_sign is not None and nominal_side_sign != side_sign
        if not local_flip_active and lateral_error <= 0.50 * self.config.vehicle_width:
            return base_target_speed

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        nearest_gap = min(
            obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in relevant_obstacles
        )
        engage_distance = max(0.25 * self.config.obstacle_influence_distance, 1.0 * self.config.vehicle_length)
        if nearest_gap >= engage_distance:
            return base_target_speed

        max_brake = max(abs(self.bounds.accel_min), 1e-6)
        gap_speed = math.sqrt(max(0.0, 2.0 * max_brake * max(nearest_gap, 0.0)))
        lateral_window = max(1.5 * self.config.vehicle_width, 0.5 * observation.road.half_width)
        alignment_scale = 1.0 / (1.0 + lateral_error / max(lateral_window, 1e-6))
        hazard_speed_cap = max(0.60, gap_speed * alignment_scale)
        if local_flip_active:
            hazard_speed_cap = min(hazard_speed_cap, state.speed + 0.35)
        return float(min(base_target_speed, hazard_speed_cap))

    def _reference_speed(self, observation: Observation, index: int, mode: str) -> float:
        """Construct reference speed with an extra leader hazard-speed governor for adaptive control."""

        base_target_speed = super()._reference_speed(observation, index, mode)
        if index != 0:
            return base_target_speed
        return self._leader_hazard_speed_limit(
            observation=observation,
            state=observation.states[index],
            mode=mode,
            base_target_speed=base_target_speed,
        )

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """输出风险自适应 APF-LF 名义控制。"""

        actions: list[Action] = []
        mode_repulsive_scale, mode_road_scale, _ = self._mode_gain_scales(mode)
        leader_goal_error = max(observation.goal_x - observation.states[0].x, 0.0)
        if self._leader_goal_errors:
            progress_delta = self._leader_goal_errors[-1] - leader_goal_error
        else:
            progress_delta = 0.0
        self._leader_goal_errors.append(leader_goal_error)

        leader_lateral_escape_sign = 1.0
        if observation.obstacles:
            nearest = min(
                observation.obstacles,
                key=lambda obstacle: (obstacle.x - observation.states[0].x) ** 2
                + (obstacle.y - observation.states[0].y) ** 2,
            )
            leader_lateral_escape_sign = -1.0 if nearest.y >= observation.states[0].y else 1.0
        self._escape_direction = leader_lateral_escape_sign

        leader_force_norm = 0.0
        cached_forces: list[np.ndarray] = []
        cached_speeds: list[float] = []
        for index, state in enumerate(observation.states):
            clearance, closing_speed, ttc, boundary_margin = self._risk_features(observation, state)
            risk_score = self.compute_risk_score(
                clearance=clearance,
                closing_speed=closing_speed,
                ttc=ttc,
                boundary_margin=boundary_margin,
            )
            repulsive_gain, road_gain = self.schedule_gains(risk_score)
            repulsive_gain *= mode_repulsive_scale
            road_gain *= mode_road_scale
            if index == 0:
                target = self._leader_goal_target(observation, state, mode)
                leader_guidance_force = self._leader_bypass_force(
                    observation,
                    state,
                    mode,
                    target_y=float(target[1]),
                )
            else:
                target = self._desired_global_position(observation, index, mode)
                leader_guidance_force = np.zeros(2, dtype=float)
            attractive_force = self._attractive_force(state, target)
            formation_force = self._formation_force(observation, index, mode)
            consensus_force = self._consensus_force(observation, index, mode)
            road_force = self._road_force(state, road_gain=road_gain)
            obstacle_force = self._adaptive_obstacle_force(
                observation=observation,
                index=index,
                mode=mode,
                repulsive_gain=repulsive_gain,
            )
            peer_force = self._sum_peer_forces(
                observation=observation,
                index=index,
                repulsive_gain=repulsive_gain,
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
            cached_forces.append(total_force)
            cached_speeds.append(
                self._mode_adjusted_target_speed(
                    self._reference_speed(observation, index, mode),
                    mode,
                )
            )
            if index == 0:
                leader_force_norm = float(np.linalg.norm(total_force))

        if self.update_stagnation(
            progress_delta=progress_delta,
            speed=observation.states[0].speed,
            force_norm=leader_force_norm,
        ):
            cached_forces[0] = cached_forces[0] + self._escape_force(observation.states[0], observation)
        elif self._escape_steps_remaining > 0:
            cached_forces[0] = cached_forces[0] + self._escape_force(observation.states[0], observation)

        for state, force, target_speed in zip(observation.states, cached_forces, cached_speeds, strict=True):
            actions.append(
                self._force_to_action(
                    state=state,
                    force=force,
                    target_speed=target_speed,
                )
            )
        return tuple(actions)
