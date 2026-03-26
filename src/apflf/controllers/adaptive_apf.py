"""风险自适应 APF-LF 控制器。"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from apflf.controllers.apf_lf import APFLFController
from apflf.controllers.apf_st import spatio_temporal_factor
from apflf.decision.mode_base import parse_mode_label
from apflf.utils.types import (
    Action,
    NominalDiagnostics,
    NominalForceBreakdown,
    Observation,
    ObstacleState,
    State,
)


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

    def _smoothstep01(self, value: float) -> float:
        """Return a C1-continuous activation in [0, 1]."""

        clamped = float(np.clip(value, 0.0, 1.0))
        return float(clamped * clamped * (3.0 - 2.0 * clamped))

    def _force_tuple(self, force: np.ndarray) -> tuple[float, float]:
        """Convert a force vector into a stable tuple for logging/replay."""

        return (float(force[0]), float(force[1]))

    def _theta_governor_release_bias(self) -> float:
        """Return the active leader governor-release floor bias, if any."""

        return self._active_theta_gain_scales()[3]

    def _rising_activation(
        self,
        value: float,
        *,
        start: float,
        full: float,
    ) -> float:
        """Return a smooth activation that rises from 0 to 1 over [start, full]."""

        if value <= start:
            return 0.0
        if value >= full:
            return 1.0
        raw = (value - start) / max(full - start, 1e-6)
        return self._smoothstep01(raw)

    def _falling_activation(
        self,
        value: float,
        *,
        full: float,
        zero: float,
    ) -> float:
        """Return a smooth activation that falls from 1 to 0 over [full, zero]."""

        if value <= full:
            return 1.0
        if value >= zero:
            return 0.0
        raw = (zero - value) / max(zero - full, 1e-6)
        return self._smoothstep01(raw)

    def _leader_nearest_rear_gap(
        self,
        *,
        state: State,
        obstacles: tuple[ObstacleState, ...],
    ) -> float:
        """Return the smallest rear-gap from the leader front bumper to the obstacle set."""

        if not obstacles:
            return float("inf")
        state_front_x = state.x + 0.5 * self.config.vehicle_length
        return min(
            obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in obstacles
        )

    def _leader_staggered_hazard_activation(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        front_obstacles: tuple[ObstacleState, ...],
        target_y: float | None,
    ) -> float:
        """Detect the dual-blocker hazard window that needs extra leader-only longitudinal restraint."""

        nominal_side_sign = self._mode_behavior_side_sign(mode)
        if nominal_side_sign is None or target_y is None:
            return 0.0

        nominal_obstacles = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=nominal_side_sign,
        )
        alternate_obstacles = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=-nominal_side_sign,
        )
        if not nominal_obstacles or not alternate_obstacles:
            return 0.0

        nominal_gap = self._leader_nearest_rear_gap(
            state=state,
            obstacles=nominal_obstacles,
        )
        alternate_gap = self._leader_nearest_rear_gap(
            state=state,
            obstacles=alternate_obstacles,
        )
        lateral_error = abs(float(target_y - state.y))

        a_nominal_gap = self._falling_activation(
            nominal_gap,
            full=max(0.12 * self.config.obstacle_influence_distance, 0.15 * self.config.vehicle_length),
            zero=max(0.45 * self.config.obstacle_influence_distance, 0.45 * self.config.vehicle_length),
        )
        a_alternate_gap = self._falling_activation(
            alternate_gap,
            full=max(0.08 * self.config.obstacle_influence_distance, 0.30 * self.config.vehicle_length),
            zero=max(0.28 * self.config.obstacle_influence_distance, 0.85 * self.config.vehicle_length),
        )
        a_lateral = self._rising_activation(
            lateral_error,
            start=0.30 * self.config.vehicle_width,
            full=max(1.10 * self.config.vehicle_width, 0.45 * observation.road.half_width),
        )
        return float(np.clip(a_nominal_gap * a_alternate_gap * a_lateral, 0.0, 1.0))

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

        return 0.85 * self._leader_nonrelevant_clearance_activation(
            state=state,
            obstacle=obstacle,
        )

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
        # 当 obstacle 横向中心远离当前绕行路径时，进一步削减反向横向分量
        lateral_distance = abs(obstacle.y - state.y)
        lateral_threshold = 0.5 * (obstacle.width + self.config.vehicle_width)
        if lateral_distance > lateral_threshold:
            excess = (lateral_distance - lateral_threshold) / max(lateral_threshold, 1e-6)
            lateral_boost = float(np.clip(0.15 * excess, 0.0, 0.10))
            reduction = min(reduction + lateral_boost, 0.90)
        return np.asarray([float(force[0]), (1.0 - reduction) * float(force[1])], dtype=float)

    def _shape_leader_staggered_obstacle_force(
        self,
        *,
        force: np.ndarray,
        reduction: float,
    ) -> np.ndarray:
        """Attenuate only the adverse longitudinal repulsion in staggered dual-blocker corridors."""

        if reduction <= 0.0 or float(force[0]) >= 0.0:
            return force
        clamped_reduction = float(np.clip(reduction, 0.0, 0.98))
        return np.asarray(
            [(1.0 - clamped_reduction) * float(force[0]), float(force[1])],
            dtype=float,
        )

    def _leader_staggered_longitudinal_relief(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        front_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
    ) -> float:
        """Return a bounded x-repulsion attenuation factor in staggered dual-blocker geometry."""

        nominal_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        alternate_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=-side_sign,
        )
        if not nominal_relevant or not alternate_relevant:
            return 0.0
        if {
            obstacle.obstacle_id for obstacle in nominal_relevant
        } == {
            obstacle.obstacle_id for obstacle in alternate_relevant
        }:
            return 0.0

        target_y = self._leader_behavior_target_y(observation, state, mode)
        if target_y is None:
            return 0.0
        lateral_error = abs(float(target_y - state.y))
        lateral_dead_zone = 0.25 * self.config.vehicle_width
        if lateral_error <= lateral_dead_zone:
            return 0.0

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        nominal_rear_gap = min(
            obs.x - 0.5 * obs.length - state_front_x for obs in nominal_relevant
        )
        alternate_rear_gap = min(
            obs.x - 0.5 * obs.length - state_front_x for obs in alternate_relevant
        )
        engage_distance = max(
            0.25 * self.config.obstacle_influence_distance,
            1.0 * self.config.vehicle_length,
        )
        alternate_lookahead = max(
            0.5 * self.config.obstacle_influence_distance,
            1.5 * self.config.vehicle_length,
        )
        if alternate_rear_gap < -0.5 * self.config.vehicle_length:
            return 0.0
        if alternate_rear_gap > alternate_lookahead:
            return 0.0

        a_nominal_gap = float(
            np.clip(
                (engage_distance - nominal_rear_gap) / max(engage_distance, 1e-6),
                0.0,
                1.0,
            )
        )
        a_alternate_gap = float(
            np.clip(
                (alternate_lookahead - alternate_rear_gap) / max(alternate_lookahead, 1e-6),
                0.0,
                1.0,
            )
        )
        lateral_window = max(
            1.5 * self.config.vehicle_width,
            0.5 * observation.road.half_width,
        )
        a_lateral = float(
            np.clip(
                (lateral_error - lateral_dead_zone)
                / max(lateral_window - lateral_dead_zone, 1e-6),
                0.0,
                1.0,
            )
        )
        reduction = 0.40 * a_nominal_gap + 0.35 * a_alternate_gap + 0.35 * a_lateral
        return float(np.clip(reduction, 0.0, 0.98))

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
        staggered_relief = self._leader_staggered_longitudinal_relief(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
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
            if obstacle.obstacle_id in front_ids and staggered_relief > 0.0:
                force = self._shape_leader_staggered_obstacle_force(
                    force=force,
                    reduction=staggered_relief,
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
        staggered_activation = self._leader_staggered_hazard_activation(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            target_y=target_y,
        )
        if (
            not local_flip_active
            and lateral_error <= 0.50 * self.config.vehicle_width
            and staggered_activation <= 0.0
        ):
            return base_target_speed

        nearest_gap = self._leader_nearest_rear_gap(
            state=state,
            obstacles=relevant_obstacles,
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
        if staggered_activation > 0.0:
            # Keep the gap-speed cap as the backbone and only add a bounded,
            # monotone compression inside the staggered dual-blocker window.
            hazard_speed_cap = max(0.60, hazard_speed_cap * (1.0 - 0.25 * staggered_activation))
        return float(min(base_target_speed, hazard_speed_cap))

    def _leader_relocked_edge_speed_release(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        front_obstacles: tuple[ObstacleState, ...],
        relevant_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
        base_target_speed: float,
        staggered_cap: float,
    ) -> float:
        """Release the staggered speed cap once relocked edge-hold geometry is established.

        Keeps the logic leader-only, smooth, and bounded:

        * exact degradation to ``staggered_cap`` when edge-hold activation is zero
        * monotone release floor in a small crawl-speed interval
        * hard upper bound at ``base_target_speed``
        """

        hold_activation = self._leader_relocked_edge_hold_activation(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            relevant_obstacles=relevant_obstacles,
            side_sign=side_sign,
        )
        if hold_activation <= 1e-6:
            return float(staggered_cap)

        release_bias = self._theta_governor_release_bias()
        release_floor = float(
            np.clip(
                0.85 + 0.35 * hold_activation + release_bias,
                0.85,
                1.20 + release_bias,
            )
        )
        released_cap = min(base_target_speed, max(staggered_cap, release_floor))
        return float(released_cap)

    def _leader_relocked_edge_speed_release_trace(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        front_obstacles: tuple[ObstacleState, ...],
        relevant_obstacles: tuple[ObstacleState, ...],
        side_sign: float,
        base_target_speed: float,
        staggered_cap: float,
    ) -> tuple[float, float]:
        """Return released cap and edge-hold activation for leader-only diagnostics."""

        hold_activation = self._leader_relocked_edge_hold_activation(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            relevant_obstacles=relevant_obstacles,
            side_sign=side_sign,
        )
        if hold_activation <= 1e-6:
            return (float(staggered_cap), 0.0)

        release_bias = self._theta_governor_release_bias()
        release_floor = float(
            np.clip(
                0.85 + 0.35 * hold_activation + release_bias,
                0.85,
                1.20 + release_bias,
            )
        )
        released_cap = min(base_target_speed, max(staggered_cap, release_floor))
        return (float(released_cap), float(hold_activation))

    def _leader_staggered_speed_trace(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        base_target_speed: float,
    ) -> tuple[float, float, float, float]:
        """Return released cap, pre-release cap, staggered activation, and edge-hold activation."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        front_obstacles = self._leader_front_obstacles(observation, state)
        if not front_obstacles:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        side_sign = self._leader_behavior_side_sign(
            observation,
            state,
            mode,
            front_obstacles=front_obstacles,
        )
        if side_sign is None:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        nominal_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        if not nominal_relevant:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        alternate_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=-side_sign,
        )
        if not alternate_relevant:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        target_y = self._leader_behavior_target_y(observation, state, mode)
        if target_y is None or state.speed <= 0.35:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        lateral_error = abs(float(target_y - state.y))
        lateral_dead_zone = 0.25 * self.config.vehicle_width
        if lateral_error <= lateral_dead_zone:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        nominal_rear_gap = min(
            obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in nominal_relevant
        )
        engage_distance = max(
            0.25 * self.config.obstacle_influence_distance,
            1.0 * self.config.vehicle_length,
        )
        a_nominal_gap = float(
            np.clip(
                (engage_distance - nominal_rear_gap) / max(engage_distance, 1e-6),
                0.0,
                1.0,
            )
        )

        alternate_rear_gap = min(
            obstacle.x - 0.5 * obstacle.length - state_front_x for obstacle in alternate_relevant
        )
        alternate_lookahead = max(
            0.5 * self.config.obstacle_influence_distance,
            1.5 * self.config.vehicle_length,
        )
        if alternate_rear_gap < -0.5 * self.config.vehicle_length:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)
        if alternate_rear_gap > alternate_lookahead:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)
        a_alternate_gap = float(
            np.clip(
                (alternate_lookahead - alternate_rear_gap) / max(alternate_lookahead, 1e-6),
                0.0,
                1.0,
            )
        )

        lateral_window = max(
            1.5 * self.config.vehicle_width,
            0.5 * observation.road.half_width,
        )
        a_lateral = float(
            np.clip(
                (lateral_error - lateral_dead_zone) / max(lateral_window - lateral_dead_zone, 1e-6),
                0.0,
                1.0,
            )
        )
        a_staggered = float(a_nominal_gap * a_alternate_gap * a_lateral)
        if a_staggered <= 1e-6:
            return (float(base_target_speed), float(base_target_speed), 0.0, 0.0)

        min_gap = min(max(nominal_rear_gap, 0.0), max(alternate_rear_gap, 0.0))
        max_brake = max(abs(self.bounds.accel_min), 1e-6)
        staggered_gap_speed = math.sqrt(max(0.0, 2.0 * max_brake * min_gap))
        blend = float(np.clip(a_staggered * 8.0, 0.0, 0.80))
        release_bias = self._theta_governor_release_bias()
        crawl_floor = float(
            np.clip(
                0.30 + 0.25 * max(state.speed - 0.35, 0.0) + release_bias,
                0.30,
                0.70 + release_bias,
            )
        )
        staggered_cap = max(
            crawl_floor,
            (1.0 - blend) * base_target_speed + blend * staggered_gap_speed,
        )
        released_cap, hold_activation = self._leader_relocked_edge_speed_release_trace(
            observation=observation,
            state=state,
            mode=mode,
            front_obstacles=front_obstacles,
            relevant_obstacles=nominal_relevant,
            side_sign=side_sign,
            base_target_speed=base_target_speed,
            staggered_cap=staggered_cap,
        )
        return (
            float(released_cap),
            float(staggered_cap),
            float(a_staggered),
            float(hold_activation),
        )

    def _leader_reference_speed_trace(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the leader reference-speed chain for diagnostics and logging."""

        base_reference_speed = super()._reference_speed(observation, 0, mode)
        hazard_speed_cap = self._leader_hazard_speed_limit(
            observation=observation,
            state=state,
            mode=mode,
            base_target_speed=base_reference_speed,
        )
        release_speed_cap, staggered_speed_cap, staggered_activation, edge_hold_activation = (
            self._leader_staggered_speed_trace(
                observation=observation,
                state=state,
                mode=mode,
                base_target_speed=hazard_speed_cap,
            )
        )
        return (
            float(base_reference_speed),
            float(hazard_speed_cap),
            float(staggered_speed_cap),
            float(release_speed_cap),
            float(staggered_activation),
            float(edge_hold_activation),
        )

    def _leader_staggered_hazard_speed_cap(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        base_target_speed: float,
    ) -> float:
        """Further throttle leader speed when staggered dual-blocker geometry is detected.

        Supplements ``_leader_hazard_speed_limit`` by detecting configurations
        where both nominal-side and alternate-side blockers simultaneously
        constrain the corridor.  Constructs three smooth activations:

        * ``a_nominal_gap``  – proximity of the nearest nominal-side blocker
        * ``a_alternate_gap`` – alternate-side blocker entering lookahead
        * ``a_lateral``       – lateral reorientation incompleteness

        Composite ``a_staggered = a_nominal_gap * a_alternate_gap * a_lateral``
        drives a gap-speed blend that is monotonically non-increasing and
        preserves a crawl-speed floor.  When ``a_staggered == 0`` the output
        degrades exactly to ``base_target_speed`` (minimal-intervention).

        Mathematical constraints (AI_MEMORY §5.3):
        * leader-only, hazard-only, smooth-bounded
        * does not modify ``safety_filter.py``
        * does not affect follower reference speed
        """

        released_cap, _, _, _ = self._leader_staggered_speed_trace(
            observation=observation,
            state=state,
            mode=mode,
            base_target_speed=base_target_speed,
        )
        return float(released_cap)

    def _leader_staggered_lateral_boost(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
    ) -> float:
        """Return a bounded leader guidance multiplier in staggered dual-blocker corridors."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return 1.0

        front_obstacles = self._leader_front_obstacles(observation, state)
        if not front_obstacles:
            return 1.0

        side_sign = self._leader_behavior_side_sign(
            observation,
            state,
            mode,
            front_obstacles=front_obstacles,
        )
        if side_sign is None:
            return 1.0

        nominal_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=side_sign,
        )
        alternate_relevant = self._leader_relevant_obstacles(
            observation,
            front_obstacles=front_obstacles,
            side_sign=-side_sign,
        )
        if not nominal_relevant or not alternate_relevant:
            return 1.0

        target_y = self._leader_behavior_target_y(observation, state, mode)
        if target_y is None:
            return 1.0
        lateral_error = abs(float(target_y - state.y))
        lateral_dead_zone = 0.25 * self.config.vehicle_width
        if lateral_error <= lateral_dead_zone:
            return 1.0

        state_front_x = state.x + 0.5 * self.config.vehicle_length
        nominal_rear_gap = min(
            obs.x - 0.5 * obs.length - state_front_x for obs in nominal_relevant
        )
        engage_distance = max(
            0.25 * self.config.obstacle_influence_distance,
            1.0 * self.config.vehicle_length,
        )
        a_gap = float(
            np.clip(
                (engage_distance - nominal_rear_gap) / max(engage_distance, 1e-6),
                0.0,
                1.0,
            )
        )

        a_lateral = float(
            np.clip(
                (lateral_error - lateral_dead_zone) / max(self.config.vehicle_width, 1e-6),
                0.0,
                1.0,
            )
        )

        boost_activation = a_gap * a_lateral
        if boost_activation <= 1e-6:
            return 1.0

        boost_gain = 2.2
        return float(1.0 + boost_gain * boost_activation)

    def _leader_staggered_steer_bias(
        self,
        *,
        state: State,
        target_y: float,
        boost: float,
    ) -> float:
        """Return a bounded leader steer bias once staggered dual-blocker guidance is active."""

        if boost <= 1.0:
            return 0.0
        lateral_error = float(target_y - state.y)
        lateral_dead_zone = 0.25 * self.config.vehicle_width
        if abs(lateral_error) <= lateral_dead_zone:
            return 0.0

        lateral_activation = float(
            np.clip(
                (abs(lateral_error) - lateral_dead_zone) / max(1.25 * self.config.vehicle_width, 1e-6),
                0.0,
                1.0,
            )
        )
        boost_activation = float(np.clip((boost - 1.0) / 1.2, 0.0, 1.0))
        max_bias = 0.16
        return float(np.sign(lateral_error) * max_bias * lateral_activation * boost_activation)

    def _reference_speed(self, observation: Observation, index: int, mode: str) -> float:
        """Construct reference speed with leader hazard-speed governor and staggered corridor governor."""

        if index != 0:
            return super()._reference_speed(observation, index, mode)
        _, _, _, release_speed_cap, _, _ = self._leader_reference_speed_trace(
            observation=observation,
            state=observation.states[index],
            mode=mode,
        )
        return release_speed_cap

    def _leader_low_speed_braking_cap(
        self,
        *,
        observation: Observation,
        state: State,
        mode: str,
        target_y: float | None,
        total_force: np.ndarray,
        scaled_target_speed: float,
    ) -> float:
        """Cap nominal braking in the near-stop hazard regime so saturated steering can keep making progress."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow" or parsed_mode.behavior.startswith("recover_"):
            return scaled_target_speed
        if state.speed > 0.35 or target_y is None:
            return scaled_target_speed

        lateral_error = abs(float(target_y - state.y))
        if lateral_error <= 0.90 * self.config.vehicle_width:
            return scaled_target_speed

        clipped_force_x = float(np.clip(total_force[0], -8.0, 8.0))
        speed_gain = max(self.config.speed_gain, 1e-6)
        desired_min_accel = -0.03
        desired_max_accel = 0.0
        speed_floor = state.speed + (desired_min_accel - 0.08 * clipped_force_x) / speed_gain
        speed_ceiling = state.speed + (desired_max_accel - 0.08 * clipped_force_x) / speed_gain
        if speed_floor > speed_ceiling:
            speed_floor, speed_ceiling = speed_ceiling, speed_floor
        return float(np.clip(scaled_target_speed, speed_floor, speed_ceiling))

    def compute_actions(
        self,
        observation: Observation,
        mode: str,
        theta: tuple[float, float, float, float] | None = None,
    ) -> tuple[Action, ...]:
        """输出风险自适应 APF-LF 名义控制。"""

        actions: list[Action] = []
        leader_diagnostics = NominalDiagnostics()
        mode_repulsive_scale, mode_road_scale, _ = self._mode_gain_scales(mode)
        theta_repulsive_scale, theta_road_scale, theta_formation_scale, _ = self._theta_gain_scales(theta)
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
        cached_steer_biases: list[float] = []
        leader_force_breakdown = NominalForceBreakdown()
        with self._theta_context(theta):
            for index, state in enumerate(observation.states):
                clearance, closing_speed, ttc, boundary_margin = self._risk_features(observation, state)
                risk_score = self.compute_risk_score(
                    clearance=clearance,
                    closing_speed=closing_speed,
                    ttc=ttc,
                    boundary_margin=boundary_margin,
                )
                repulsive_gain, road_gain = self.schedule_gains(risk_score)
                repulsive_gain *= mode_repulsive_scale * theta_repulsive_scale
                road_gain *= mode_road_scale * theta_road_scale
                if index == 0:
                    target = self._leader_goal_target(observation, state, mode)
                    leader_guidance_force = self._leader_bypass_force(
                        observation,
                        state,
                        mode,
                        target_y=float(target[1]),
                        road_gain=road_gain,
                    )
                    staggered_boost = self._leader_staggered_lateral_boost(
                        observation=observation,
                        state=state,
                        mode=mode,
                    )
                    leader_steer_bias = self._leader_staggered_steer_bias(
                        state=state,
                        target_y=float(target[1]),
                        boost=staggered_boost,
                    )
                    if staggered_boost > 1.0:
                        leader_guidance_force = leader_guidance_force * staggered_boost
                else:
                    target = self._desired_global_position(observation, index, mode)
                    leader_guidance_force = np.zeros(2, dtype=float)
                    leader_steer_bias = 0.0
                attractive_force = self._attractive_force(state, target)
                formation_force = self._formation_force(observation, index, mode) * theta_formation_scale
                consensus_force = self._consensus_force(observation, index, mode) * theta_formation_scale
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
                if index == 0:
                    (
                        leader_base_reference_speed,
                        leader_hazard_speed_cap,
                        leader_staggered_speed_cap,
                        leader_release_speed_cap,
                        leader_staggered_activation,
                        leader_edge_hold_activation,
                    ) = self._leader_reference_speed_trace(
                        observation=observation,
                        state=state,
                        mode=mode,
                    )
                    reference_speed = leader_release_speed_cap
                else:
                    reference_speed = super()._reference_speed(observation, index, mode)
                target_speed = self._mode_adjusted_target_speed(reference_speed, mode)
                if index == 0:
                    target_speed = self._leader_low_speed_braking_cap(
                        observation=observation,
                        state=state,
                        mode=mode,
                        target_y=float(target[1]),
                        total_force=total_force,
                        scaled_target_speed=target_speed,
                    )
                    leader_diagnostics = NominalDiagnostics(
                        leader_risk_score=float(risk_score),
                        leader_base_reference_speed=float(leader_base_reference_speed),
                        leader_hazard_speed_cap=float(leader_hazard_speed_cap),
                        leader_staggered_speed_cap=float(leader_staggered_speed_cap),
                        leader_release_speed_cap=float(leader_release_speed_cap),
                        leader_target_speed=float(target_speed),
                        leader_staggered_activation=float(leader_staggered_activation),
                        leader_edge_hold_activation=float(leader_edge_hold_activation),
                    )
                    leader_force_breakdown = NominalForceBreakdown(
                        attractive=self._force_tuple(attractive_force),
                        formation=self._force_tuple(formation_force),
                        consensus=self._force_tuple(consensus_force),
                        road=self._force_tuple(road_force),
                        obstacle=self._force_tuple(obstacle_force),
                        peer=self._force_tuple(peer_force),
                        behavior=self._force_tuple(behavior_force),
                        guidance=self._force_tuple(leader_guidance_force),
                        total=self._force_tuple(total_force),
                    )
                cached_forces.append(total_force)
                cached_speeds.append(target_speed)
                cached_steer_biases.append(leader_steer_bias if index == 0 else 0.0)
                if index == 0:
                    leader_force_norm = float(np.linalg.norm(total_force))

        leader_escape_force = np.zeros(2, dtype=float)
        if self.update_stagnation(
            progress_delta=progress_delta,
            speed=observation.states[0].speed,
            force_norm=leader_force_norm,
        ):
            leader_escape_force = self._escape_force(observation.states[0], observation)
            cached_forces[0] = cached_forces[0] + leader_escape_force
        elif self._escape_steps_remaining > 0:
            leader_escape_force = self._escape_force(observation.states[0], observation)
            cached_forces[0] = cached_forces[0] + leader_escape_force

        if cached_forces:
            self._record_step_diagnostics(
                NominalDiagnostics(
                    leader_risk_score=leader_diagnostics.leader_risk_score,
                    leader_base_reference_speed=leader_diagnostics.leader_base_reference_speed,
                    leader_hazard_speed_cap=leader_diagnostics.leader_hazard_speed_cap,
                    leader_staggered_speed_cap=leader_diagnostics.leader_staggered_speed_cap,
                    leader_release_speed_cap=leader_diagnostics.leader_release_speed_cap,
                    leader_target_speed=leader_diagnostics.leader_target_speed,
                    leader_staggered_activation=leader_diagnostics.leader_staggered_activation,
                    leader_edge_hold_activation=leader_diagnostics.leader_edge_hold_activation,
                    leader_force=NominalForceBreakdown(
                        attractive=leader_force_breakdown.attractive,
                        formation=leader_force_breakdown.formation,
                        consensus=leader_force_breakdown.consensus,
                        road=leader_force_breakdown.road,
                        obstacle=leader_force_breakdown.obstacle,
                        peer=leader_force_breakdown.peer,
                        behavior=leader_force_breakdown.behavior,
                        guidance=leader_force_breakdown.guidance,
                        escape=self._force_tuple(leader_escape_force),
                        total=self._force_tuple(cached_forces[0]),
                    ),
                )
            )

        for state, force, target_speed, steer_bias in zip(
            observation.states,
            cached_forces,
            cached_speeds,
            cached_steer_biases,
            strict=True,
        ):
            action = self._force_to_action(
                state=state,
                force=force,
                target_speed=target_speed,
            )
            if abs(steer_bias) > 1e-9:
                action = Action(
                    accel=action.accel,
                    steer=float(np.clip(action.steer + steer_bias, self.bounds.steer_min, self.bounds.steer_max)),
                )
            actions.append(action)
        return tuple(actions)
