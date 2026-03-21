"""核心类型定义。

本模块集中定义仿真主链路中的配置对象、状态对象、观测对象与快照对象，
用于保持理论层、代码层与实验导出层的数据语义一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputBounds:
    """车辆执行器与速度边界。"""

    accel_min: float
    accel_max: float
    steer_min: float
    steer_max: float
    speed_min: float
    speed_max: float


@dataclass(frozen=True)
class ExperimentConfig:
    """实验输出相关配置。"""

    name: str
    output_root: str
    save_traj: bool


@dataclass(frozen=True)
class SimulationConfig:
    """数值积分与物理边界配置。"""

    dt: float
    steps: int
    wheelbase: float
    target_speed: float
    bounds: InputBounds


@dataclass(frozen=True)
class ControllerConfig:
    """名义控制器配置。"""

    kind: str
    vehicle_length: float
    vehicle_width: float
    speed_gain: float
    gap_gain: float
    lateral_gain: float
    heading_gain: float
    attraction_gain: float
    repulsive_gain: float
    road_gain: float
    formation_gain: float
    consensus_gain: float
    obstacle_influence_distance: float
    vehicle_influence_distance: float
    road_influence_margin: float
    st_velocity_gain: float
    ttc_gain: float
    ttc_threshold: float
    risk_distance_scale: float
    risk_speed_scale: float
    risk_ttc_threshold: float
    risk_sigmoid_slope: float
    risk_reference: float
    adaptive_alpha: float
    repulsive_gain_min: float
    repulsive_gain_max: float
    road_gain_min: float
    road_gain_max: float
    stagnation_speed_threshold: float
    stagnation_progress_threshold: float
    stagnation_force_threshold: float
    stagnation_steps: int
    stagnation_cooldown_steps: int


@dataclass(frozen=True)
class SafetyConfig:
    """安全滤波配置。

    参数:
        enabled: 是否启用 CBF-QP 安全滤波。
        solver: QP 求解器名称，当前默认 `osqp`。
        safe_distance: 车辆与障碍物/其他车辆的最小安全间距，单位 m。
        barrier_decay: 离散 CBF 衰减系数，越大越保守。
        slack_penalty: 软约束惩罚系数。
        max_slack: 单步允许的最大松弛量。
        road_boundary_margin: 额外道路边界安全裕度，单位 m。
        fallback_brake: fallback 时采用的紧急制动加速度幅值，单位 m/s^2。
        fallback_steer_gain: fallback 时回正/避障转角增益。
    """

    enabled: bool
    solver: str
    safe_distance: float
    barrier_decay: float
    slack_penalty: float
    max_slack: float
    road_boundary_margin: float
    fallback_brake: float
    fallback_steer_gain: float


@dataclass(frozen=True)
class DecisionConfig:
    """模式决策配置。"""

    kind: str
    default_mode: str
    hysteresis_steps: int
    risk_threshold_enter: float
    risk_threshold_exit: float
    clearance_threshold: float
    ttc_threshold: float
    boundary_margin_threshold: float
    lookahead_distance: float
    narrow_passage_margin: float
    stagnation_speed_threshold: float
    stagnation_progress_threshold: float
    stagnation_steps: int


@dataclass(frozen=True)
class RoadConfig:
    """道路几何配置。"""

    length: float
    lane_center_y: float
    half_width: float


@dataclass(frozen=True)
class ObstacleConfig:
    """障碍物配置。"""

    obstacle_id: str
    motion_model: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float


@dataclass(frozen=True)
class ScenarioConfig:
    """场景生成配置。"""

    vehicle_count: int
    spacing: float
    spawn_jitter_std: float
    initial_speed: float
    goal_x: float
    goal_tolerance: float
    road: RoadConfig
    obstacles: tuple[ObstacleConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProjectConfig:
    """项目总配置对象。"""

    experiment: ExperimentConfig
    simulation: SimulationConfig
    controller: ControllerConfig
    safety: SafetyConfig
    decision: DecisionConfig
    scenario: ScenarioConfig

    def to_dict(self) -> dict[str, Any]:
        """将配置对象转换为稳定的字典结构。"""

        return asdict(self)


@dataclass(frozen=True)
class State:
    """车辆状态。"""

    x: float
    y: float
    yaw: float
    speed: float

    def to_array(self) -> tuple[float, float, float, float]:
        """按固定顺序导出状态向量。"""

        return (self.x, self.y, self.yaw, self.speed)


@dataclass(frozen=True)
class Action:
    """车辆控制输入。"""

    accel: float
    steer: float

    def to_array(self) -> tuple[float, float]:
        """按固定顺序导出控制向量。"""

        return (self.accel, self.steer)


@dataclass(frozen=True)
class SafetyFilterResult:
    """安全滤波器输出。

    参数:
        safe_actions: 安全修正后的控制输入。
        correction_norms: 每辆车的最小干预修正范数。
        slack_values: 每辆车对应的软约束松弛量。
        fallback_flags: 每辆车是否触发 fallback。
    """

    safe_actions: tuple[Action, ...]
    correction_norms: tuple[float, ...] = field(default_factory=tuple)
    slack_values: tuple[float, ...] = field(default_factory=tuple)
    fallback_flags: tuple[bool, ...] = field(default_factory=tuple)
    qp_solve_times: tuple[float, ...] = field(default_factory=tuple)
    qp_iterations: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RoadGeometry:
    """运行时道路几何信息。"""

    length: float
    lane_center_y: float
    half_width: float


@dataclass(frozen=True)
class FrenetPose:
    """直线路段上的 Frenet 投影结果。"""

    s: float
    d: float


@dataclass(frozen=True)
class ObstacleState:
    """障碍物运行时状态。"""

    obstacle_id: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float

    def to_numeric_array(self) -> tuple[float, float, float, float, float, float]:
        """导出障碍物数值向量，便于保存轨迹。"""

        return (self.x, self.y, self.yaw, self.speed, self.length, self.width)


@dataclass(frozen=True)
class ScenarioSetup:
    """场景工厂生成的确定性初始化结果。"""

    road: RoadGeometry
    initial_states: tuple[State, ...]
    desired_offsets: tuple[tuple[float, float], ...]
    goal_x: float
    obstacle_configs: tuple[ObstacleConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Observation:
    """世界在离散时刻对上层模块暴露的观测。"""

    step_index: int
    time: float
    states: tuple[State, ...]
    road: RoadGeometry
    goal_x: float
    desired_offsets: tuple[tuple[float, float], ...]
    obstacles: tuple[ObstacleState, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Snapshot:
    """离散步结束后的快照。"""

    step_index: int
    time: float
    mode: str
    states: tuple[State, ...]
    nominal_actions: tuple[Action, ...]
    safe_actions: tuple[Action, ...]
    obstacles: tuple[ObstacleState, ...] = field(default_factory=tuple)
    safety_corrections: tuple[float, ...] = field(default_factory=tuple)
    safety_slacks: tuple[float, ...] = field(default_factory=tuple)
    safety_fallbacks: tuple[bool, ...] = field(default_factory=tuple)
    qp_solve_times: tuple[float, ...] = field(default_factory=tuple)
    qp_iterations: tuple[int, ...] = field(default_factory=tuple)
    step_runtime: float = 0.0
    mode_runtime: float = 0.0
    controller_runtime: float = 0.0
    safety_runtime: float = 0.0
