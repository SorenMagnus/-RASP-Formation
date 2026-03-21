"""车辆动力学模型。"""

from __future__ import annotations

import math

from apflf.env.geometry import normalize_angle
from apflf.utils.types import Action, InputBounds, State


class VehicleDynamics:
    """确定性的运动学自行车模型。

    说明:
        Phase B 在 Phase A 的基础上强化了数值健壮性检查，
        并保持单步积分接口稳定，便于未来控制器与安全层复用。
    """

    def __init__(self, wheelbase: float, bounds: InputBounds) -> None:
        """初始化车辆动力学参数。"""

        if not math.isfinite(wheelbase) or wheelbase <= 0.0:
            raise ValueError("轴距必须是有限正数。")
        self.wheelbase = wheelbase
        self.bounds = bounds

    def clamp_action(self, action: Action) -> Action:
        """将控制输入裁剪到物理边界内。"""

        accel = min(max(action.accel, self.bounds.accel_min), self.bounds.accel_max)
        steer = min(max(action.steer, self.bounds.steer_min), self.bounds.steer_max)
        return Action(accel=accel, steer=steer)

    def step(self, state: State, action: Action, dt: float) -> State:
        """执行一步显式欧拉积分。"""

        self._validate_inputs(state=state, action=action, dt=dt)
        safe_action = self.clamp_action(action)
        next_speed = min(
            max(state.speed + safe_action.accel * dt, self.bounds.speed_min),
            self.bounds.speed_max,
        )
        yaw_rate = state.speed * math.tan(safe_action.steer) / self.wheelbase

        next_x = state.x + state.speed * math.cos(state.yaw) * dt
        next_y = state.y + state.speed * math.sin(state.yaw) * dt
        next_yaw = normalize_angle(state.yaw + yaw_rate * dt)
        return State(x=next_x, y=next_y, yaw=next_yaw, speed=next_speed)

    def _validate_inputs(self, state: State, action: Action, dt: float) -> None:
        """检查积分输入是否可用于稳定计算。"""

        values = (*state.to_array(), *action.to_array(), dt)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("动力学积分输入中存在 NaN 或 Inf。")
        if dt <= 0.0:
            raise ValueError("积分步长 dt 必须为正数。")
