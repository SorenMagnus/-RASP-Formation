"""名义控制器接口、公共工具与工厂。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from apflf.decision.mode_base import parse_mode_label
from apflf.env.geometry import box_clearance, normalize_angle, rotation_matrix
from apflf.env.road import Road
from apflf.utils.types import (
    Action,
    ControllerConfig,
    InputBounds,
    NominalDiagnostics,
    Observation,
    ObstacleState,
    State,
)


class Controller(ABC):
    """名义控制器抽象接口。"""

    @abstractmethod
    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """根据观测与离散模式输出名义控制。"""

    def consume_step_diagnostics(self) -> NominalDiagnostics:
        """Return the latest controller diagnostics, if any."""

        return NominalDiagnostics()


class BaseNominalController(Controller):
    """控制器公共基类。

    说明:
        该基类封装 APF 系列控制器共享的几何力场、速度参考与力到控制输入映射逻辑，
        以保证不同基线与主方法之间仅在理论项上差异化。
    """

    def __init__(
        self,
        config: ControllerConfig,
        bounds: InputBounds,
        road: Road,
        target_speed: float,
    ) -> None:
        """初始化控制器公共依赖。"""

        self.config = config
        self.bounds = bounds
        self.road = road
        self.target_speed = target_speed
        self._step_diagnostics = NominalDiagnostics()

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """默认不实现，留给子类覆写。"""

        raise NotImplementedError

    def consume_step_diagnostics(self) -> NominalDiagnostics:
        """Return the latest diagnostics and reset the cache for the next step."""

        diagnostics = self._step_diagnostics
        self._step_diagnostics = NominalDiagnostics()
        return diagnostics

    def _record_step_diagnostics(self, diagnostics: NominalDiagnostics) -> None:
        """Cache step diagnostics for the world loop to persist."""

        self._step_diagnostics = diagnostics

    def _static_goal_target(self, observation: Observation, index: int, mode: str) -> np.ndarray:
        """返回静态终点编队参考点。"""

        offset_x, offset_y = self._relative_offset_for_mode(
            observation=observation,
            index=index,
            mode=mode,
        )
        return np.asarray(
            [observation.goal_x + offset_x, observation.road.lane_center_y + offset_y],
            dtype=float,
        )

    def _formation_spacing(self, observation: Observation) -> float:
        """Infer the nominal longitudinal spacing used by the formation."""

        if len(observation.desired_offsets) <= 1:
            return max(1.5 * self.config.vehicle_length, 1e-6)
        non_zero_offsets = [abs(offset[0]) for offset in observation.desired_offsets[1:] if abs(offset[0]) > 1e-6]
        if not non_zero_offsets:
            return max(1.5 * self.config.vehicle_length, 1e-6)
        return max(min(non_zero_offsets), 1e-6)

    def _line_target_position(self, observation: Observation, index: int) -> np.ndarray:
        """Return the nominal line-formation target in world coordinates."""

        leader = observation.states[0]
        return np.asarray([leader.x, leader.y], dtype=float) + rotation_matrix(leader.yaw) @ np.asarray(
            observation.desired_offsets[index],
            dtype=float,
        )

    def _team_alignment_profile(self, observation: Observation) -> tuple[float, float, float, float]:
        """Return lag/lead and formation-error statistics in the leader body frame."""

        if len(observation.states) <= 1:
            return (0.0, 0.0, 0.0, 0.0)

        leader = observation.states[0]
        world_to_body = rotation_matrix(-leader.yaw)
        lags: list[float] = []
        leads: list[float] = []
        errors: list[float] = []
        for index, state in enumerate(observation.states[1:], start=1):
            desired_position = self._line_target_position(observation, index)
            error_world = desired_position - np.asarray([state.x, state.y], dtype=float)
            error_body = world_to_body @ error_world
            longitudinal_error = float(error_body[0])
            errors.append(float(np.linalg.norm(error_world)))
            lags.append(max(longitudinal_error, 0.0))
            leads.append(max(-longitudinal_error, 0.0))
        return (
            float(max(lags, default=0.0)),
            float(max(leads, default=0.0)),
            float(max(errors, default=0.0)),
            float(np.mean(errors)) if errors else 0.0,
        )

    def _team_alignment_metrics(self, observation: Observation) -> tuple[float, float, float]:
        """Return max longitudinal lag, max formation error, and mean formation error."""

        max_lag, _, max_error, mean_error = self._team_alignment_profile(observation)
        return (max_lag, max_error, mean_error)

    def _hazard_queue_offset(
        self,
        *,
        observation: Observation,
        index: int,
        mode: str,
    ) -> tuple[float, float]:
        """Return the single-sided queue offset used for hazard traversal."""

        parsed_mode = parse_mode_label(mode)
        row = index
        side_sign = 1.0 if parsed_mode.behavior.endswith("_left") else -1.0
        spacing = self._formation_spacing(observation)
        longitudinal = -row * max(0.95 * spacing, 1.20 * self.config.vehicle_length)
        lateral_limit = max(observation.road.half_width - 0.95 * self.config.vehicle_width, 0.0)
        lateral = min(
            lateral_limit,
            (0.35 + 0.10 * max(row - 1, 0)) * self.config.vehicle_width,
        )
        return (longitudinal, side_sign * lateral)

    def _recovery_blend(self, observation: Observation) -> float:
        """Return a smooth queue-to-line interpolation factor during recovery."""

        spacing = self._formation_spacing(observation)
        max_lag, max_error, _ = self._team_alignment_metrics(observation)
        lag_ratio = float(np.clip(max_lag / max(2.5 * spacing, 1e-6), 0.0, 1.0))
        error_ratio = float(np.clip(max_error / max(1.8 * spacing, 1e-6), 0.0, 1.0))
        return max(lag_ratio, error_ratio)

    def _leader_recovery_speed_limit(
        self,
        *,
        observation: Observation,
        target_speed: float,
        mode: str,
    ) -> float:
        """Reduce leader speed while the team is still merging back into formation."""

        parsed_mode = parse_mode_label(mode)
        if not parsed_mode.behavior.startswith("recover_"):
            return float(target_speed)

        spacing = self._formation_spacing(observation)
        max_lag, max_lead, max_error, _ = self._team_alignment_profile(observation)
        if max_lag <= 0.35 * spacing and max_error <= 0.40 * spacing:
            return float(target_speed)

        if observation.states[0].x >= observation.goal_x and max_lead > 0.35 * spacing and max_lag <= 0.20 * spacing:
            lead_relief_speed = min(self.target_speed, 1.0 + 0.25 * max_lead)
            return float(np.clip(max(target_speed, lead_relief_speed), self.bounds.speed_min, self.bounds.speed_max))

        lag_scale = spacing / max(spacing + max_lag, 1e-6)
        error_scale = spacing / max(spacing + max_error, 1e-6)
        speed_scale = max(0.18, min(lag_scale, error_scale))
        if observation.states[0].x >= observation.goal_x:
            capped_speed = min(self.target_speed, 0.35 + 0.35 * self.target_speed * speed_scale)
        else:
            capped_speed = min(target_speed, self.target_speed * speed_scale + 0.6)
        return float(np.clip(capped_speed, self.bounds.speed_min, self.bounds.speed_max))

    def _relative_offset_for_mode(
        self,
        *,
        observation: Observation,
        index: int,
        mode: str,
    ) -> tuple[float, float]:
        """Map the topology component of the mode to a relative formation offset."""

        parsed_mode = parse_mode_label(mode)
        if index == 0:
            return observation.desired_offsets[index]

        if parsed_mode.behavior.startswith("recover_"):
            queue_offset = np.asarray(
                self._hazard_queue_offset(
                    observation=observation,
                    index=index,
                    mode=mode,
                ),
                dtype=float,
            )
            line_offset = np.asarray(observation.desired_offsets[index], dtype=float)
            blend = self._recovery_blend(observation)
            recovered_offset = blend * queue_offset + (1.0 - blend) * line_offset
            return (float(recovered_offset[0]), float(recovered_offset[1]))

        if parsed_mode.topology == "line":
            return observation.desired_offsets[index]

        if parsed_mode.behavior != "follow":
            return self._hazard_queue_offset(
                observation=observation,
                index=index,
                mode=mode,
            )

        row = (index + 1) // 2
        lateral_sign = 1.0 if index % 2 == 1 else -1.0
        base_index = min(index, len(observation.desired_offsets) - 1)
        base_spacing = max(abs(observation.desired_offsets[base_index][0]), 1e-6)
        longitudinal = -row * max(0.65 * base_spacing, 1.15 * self.config.vehicle_length)
        lateral_limit = max(observation.road.half_width - 0.75 * self.config.vehicle_width, 0.0)
        lateral = min(lateral_limit, max(1.2 * self.config.vehicle_width, 0.25 * base_spacing + 0.4))
        return (longitudinal, lateral_sign * lateral)

    def _mode_gain_scales(self, mode: str) -> tuple[float, float, float]:
        """Return repulsive/road/speed scaling factors implied by the gain mode."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.gain == "cautious":
            return (1.25, 1.15, 0.80)
        if parsed_mode.gain == "assertive":
            return (1.45, 1.05, 0.70)
        return (1.0, 1.0, 1.0)

    def _mode_adjusted_target_speed(self, target_speed: float, mode: str) -> float:
        """Scale and clamp the speed target using the current gain mode."""

        _, _, speed_scale = self._mode_gain_scales(mode)
        return float(np.clip(speed_scale * target_speed, self.bounds.speed_min, self.bounds.speed_max))

    def _mode_behavior_force(
        self,
        *,
        mode: str,
        road_gain: float,
        index: int,
    ) -> np.ndarray:
        """Return a lateral bias force implied by the behavior component."""

        parsed_mode = parse_mode_label(mode)
        if parsed_mode.behavior == "follow":
            return np.zeros(2, dtype=float)

        if parsed_mode.behavior.startswith("recover_"):
            # Recovery already uses a blended target between the hazard queue and the
            # nominal line formation. Adding a fixed lateral bias here can push a
            # follower toward the wrong road edge when it has drifted to the opposite
            # side during passage clearance, so keep recovery behavior neutral.
            return np.zeros(2, dtype=float)

        if index == 0:
            # The leader uses an explicit lateral bypass target in hazard modes. Keep
            # its extra behavior bias neutral so the target can adapt locally when a
            # staggered blocker makes the previously chosen side infeasible.
            return np.zeros(2, dtype=float)

        direction = 1.0 if parsed_mode.behavior.endswith("_left") else -1.0
        base_gain = 0.35 * road_gain if parsed_mode.behavior.startswith("yield") else 0.70 * road_gain
        return np.asarray([0.0, direction * 0.7 * base_gain], dtype=float)

    def _braking_speed(self, current_x: float, target_x: float) -> float:
        """根据接近终点时的剩余误差构造制动一致的参考速度。

        说明:
            主目标仍然是按纵向剩余距离制动；但当车辆已经接近终点纵向位置时，
            若横向仍明显偏离目标点，就保留一个与横向误差一致的低速爬行量，
            避免 leader 在偏离车道中心或终点编队位置时过早停死。
        """

        remaining_distance = max(target_x - current_x, 0.0)
        max_brake = max(abs(self.bounds.accel_min), 1e-6)
        braking_speed = math.sqrt(max(0.0, 2.0 * max_brake * remaining_distance))
        return min(self.target_speed, braking_speed)

    def _vehicle_velocity(self, state: State) -> np.ndarray:
        """将车辆状态转换为二维速度向量。"""

        return np.asarray(
            [state.speed * math.cos(state.yaw), state.speed * math.sin(state.yaw)],
            dtype=float,
        )

    def _obstacle_velocity(self, obstacle: ObstacleState) -> np.ndarray:
        """将障碍物状态转换为二维速度向量。"""

        return np.asarray(
            [obstacle.speed * math.cos(obstacle.yaw), obstacle.speed * math.sin(obstacle.yaw)],
            dtype=float,
        )

    def _attractive_force(self, state: State, target: np.ndarray) -> np.ndarray:
        """计算目标点吸引力。"""

        position = np.asarray([state.x, state.y], dtype=float)
        return self.config.attraction_gain * (target - position)

    def _road_force(self, state: State, road_gain: float) -> np.ndarray:
        """计算道路边界排斥力。"""

        half_extent_y = 0.5 * self.config.vehicle_width
        upper_clearance = (
            self.road.geometry.half_width
            - (state.y - self.road.geometry.lane_center_y)
            - half_extent_y
        )
        lower_clearance = (
            self.road.geometry.half_width
            + (state.y - self.road.geometry.lane_center_y)
            - half_extent_y
        )
        force_y = 0.0
        margin = self.config.road_influence_margin
        for clearance, direction in ((upper_clearance, -1.0), (lower_clearance, 1.0)):
            if clearance < margin:
                safe_clearance = max(clearance, 1e-3)
                magnitude = road_gain * ((1.0 / safe_clearance) - (1.0 / margin)) / (safe_clearance**2)
                force_y += direction * magnitude
        return np.asarray([0.0, force_y], dtype=float)

    def _interaction_magnitude(
        self,
        clearance: float,
        influence_distance: float,
        repulsive_gain: float,
    ) -> float:
        """根据间距计算 APF 排斥势梯度的尺度。"""

        if clearance >= influence_distance:
            return 0.0
        safe_clearance = max(clearance, 1e-3)
        return repulsive_gain * ((1.0 / safe_clearance) - (1.0 / influence_distance)) / (
            safe_clearance**2
        )

    def _repulsive_direction(self, state: State, other_x: float, other_y: float) -> np.ndarray:
        """返回远离相互作用对象的单位方向。"""

        delta = np.asarray([state.x - other_x, state.y - other_y], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            return np.asarray([0.0, 1.0], dtype=float)
        return delta / norm

    def _dynamic_obstacle_factor(self, state: State, obstacle: ObstacleState) -> float:
        """动态障碍增强因子，默认退化为 1。"""

        del state, obstacle
        return 1.0

    def _obstacle_force(
        self,
        state: State,
        obstacle: ObstacleState,
        repulsive_gain: float,
    ) -> np.ndarray:
        """计算车辆对障碍物的排斥力。"""

        clearance = box_clearance(
            state,
            self.config.vehicle_length,
            self.config.vehicle_width,
            obstacle,
            obstacle.length,
            obstacle.width,
        )
        influence_distance = self.config.obstacle_influence_distance
        magnitude = self._interaction_magnitude(
            clearance=clearance,
            influence_distance=influence_distance,
            repulsive_gain=repulsive_gain,
        )
        if magnitude == 0.0:
            return np.zeros(2, dtype=float)
        direction = self._repulsive_direction(state, obstacle.x, obstacle.y)
        return magnitude * self._dynamic_obstacle_factor(state, obstacle) * direction

    def _peer_force(self, state: State, peer_state: State, repulsive_gain: float) -> np.ndarray:
        """计算车辆之间的最小排斥力。"""

        clearance = box_clearance(
            state,
            self.config.vehicle_length,
            self.config.vehicle_width,
            peer_state,
            self.config.vehicle_length,
            self.config.vehicle_width,
        )
        magnitude = self._interaction_magnitude(
            clearance=clearance,
            influence_distance=self.config.vehicle_influence_distance,
            repulsive_gain=0.35 * repulsive_gain,
        )
        if magnitude == 0.0:
            return np.zeros(2, dtype=float)
        direction = self._repulsive_direction(state, peer_state.x, peer_state.y)
        return magnitude * direction

    def _force_to_action(self, state: State, force: np.ndarray, target_speed: float) -> Action:
        """将连续虚拟力映射为纵向加速度与转角命令。"""

        force_norm = float(np.linalg.norm(force))
        if force_norm <= 1e-9:
            desired_heading = state.yaw
        else:
            desired_heading = math.atan2(float(force[1]), max(float(force[0]), 1e-6))
        heading_error = normalize_angle(desired_heading - state.yaw)
        steer = self.config.heading_gain * heading_error + 0.1 * np.clip(force[1], -2.0, 2.0)
        accel = self.config.speed_gain * (target_speed - state.speed) + 0.08 * np.clip(
            force[0],
            -8.0,
            8.0,
        )
        steer = float(np.clip(steer, self.bounds.steer_min, self.bounds.steer_max))
        accel = float(np.clip(accel, self.bounds.accel_min, self.bounds.accel_max))
        return Action(accel=accel, steer=steer)

    def _sum_peer_forces(
        self,
        observation: Observation,
        index: int,
        repulsive_gain: float,
    ) -> np.ndarray:
        """汇总其他车辆带来的排斥力。"""

        state = observation.states[index]
        total = np.zeros(2, dtype=float)
        for peer_index, peer_state in enumerate(observation.states):
            if peer_index == index:
                continue
            total += self._peer_force(state, peer_state, repulsive_gain=repulsive_gain)
        return total

    def _sum_obstacle_forces(
        self,
        observation: Observation,
        index: int,
        repulsive_gain: float,
    ) -> np.ndarray:
        """汇总障碍物排斥力。"""

        state = observation.states[index]
        total = np.zeros(2, dtype=float)
        for obstacle in observation.obstacles:
            total += self._obstacle_force(state, obstacle, repulsive_gain=repulsive_gain)
        return total


class FormationCruiseController(BaseNominalController):
    """Phase A 的最小巡航控制器。

    说明:
        该实现保留为兼容基线，便于与 Phase C 控制器族对照验证。
    """

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """输出所有车辆的最小巡航控制。"""

        del mode

        leader = observation.states[0]
        leader_target_speed = self._braking_speed(leader.x, observation.goal_x)
        actions: list[Action] = [
            self._force_to_action(
                state=leader,
                force=self._attractive_force(
                    leader,
                    np.asarray([observation.goal_x, observation.road.lane_center_y], dtype=float),
                ) + self._road_force(leader, road_gain=self.config.road_gain),
                target_speed=leader_target_speed,
            )
        ]

        for index in range(1, len(observation.states)):
            current_state = observation.states[index]
            offset_x, offset_y = observation.desired_offsets[index]
            desired_x = leader.x + offset_x
            longitudinal_error = desired_x - current_state.x
            follower_target_speed = min(
                self.bounds.speed_max,
                max(
                    self.bounds.speed_min,
                    leader.speed + self.config.gap_gain * longitudinal_error,
                ),
            )
            target = np.asarray(
                [desired_x, observation.road.lane_center_y + offset_y],
                dtype=float,
            )
            force = self._attractive_force(current_state, target) + self._road_force(
                current_state,
                road_gain=self.config.road_gain,
            )
            actions.append(
                self._force_to_action(
                    state=current_state,
                    force=force,
                    target_speed=follower_target_speed,
                )
            )

        return tuple(actions)


def build_controller(
    config: ControllerConfig,
    bounds: InputBounds,
    road: Road,
    target_speed: float,
    *,
    wheelbase: float | None = None,
    dt: float | None = None,
) -> Controller:
    """根据配置构造具体控制器实现。"""

    if config.kind == "formation_cruise":
        return FormationCruiseController(
            config=config,
            bounds=bounds,
            road=road,
            target_speed=target_speed,
        )
    if config.kind == "apf":
        from apflf.controllers.apf import APFController

        return APFController(config=config, bounds=bounds, road=road, target_speed=target_speed)
    if config.kind == "st_apf":
        from apflf.controllers.apf_st import STAPFController

        return STAPFController(config=config, bounds=bounds, road=road, target_speed=target_speed)
    if config.kind == "apf_lf":
        from apflf.controllers.apf_lf import APFLFController

        return APFLFController(config=config, bounds=bounds, road=road, target_speed=target_speed)
    if config.kind == "adaptive_apf":
        from apflf.controllers.adaptive_apf import AdaptiveAPFController

        return AdaptiveAPFController(
            config=config,
            bounds=bounds,
            road=road,
            target_speed=target_speed,
        )
    if config.kind == "dwa":
        from apflf.controllers.dwa import DWAController

        return DWAController(
            config=config,
            bounds=bounds,
            road=road,
            target_speed=target_speed,
            wheelbase=2.8 if wheelbase is None else wheelbase,
            dt=0.1 if dt is None else dt,
        )
    if config.kind == "orca":
        from apflf.controllers.orca import ORCAController

        return ORCAController(
            config=config,
            bounds=bounds,
            road=road,
            target_speed=target_speed,
            dt=0.1 if dt is None else dt,
        )
    raise ValueError(f"不支持的控制器类型: {config.kind}")
