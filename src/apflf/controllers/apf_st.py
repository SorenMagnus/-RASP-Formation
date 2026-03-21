"""时空人工势场控制器。"""

from __future__ import annotations

import math

import numpy as np

from apflf.controllers.apf import APFController
from apflf.utils.types import ObstacleState, State


def spatio_temporal_factor(
    *,
    state: State,
    obstacle: ObstacleState,
    st_velocity_gain: float,
    ttc_gain: float,
    ttc_threshold: float,
) -> float:
    """计算时空势场增强因子。

    说明:
        因子同时考虑相对闭合速度与 TTC，风险越高增强越明显；
        当交互远离碰撞趋势时，该因子回落到 1 附近。
    """

    relative_position = np.asarray([obstacle.x - state.x, obstacle.y - state.y], dtype=float)
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-9:
        return 1.0 + st_velocity_gain + ttc_gain

    line_of_sight = relative_position / distance
    self_velocity = np.asarray(
        [state.speed * math.cos(state.yaw), state.speed * math.sin(state.yaw)],
        dtype=float,
    )
    obstacle_velocity = np.asarray(
        [obstacle.speed * math.cos(obstacle.yaw), obstacle.speed * math.sin(obstacle.yaw)],
        dtype=float,
    )
    closing_speed = max(float(np.dot(self_velocity - obstacle_velocity, line_of_sight)), 0.0)
    if closing_speed <= 1e-9:
        ttc_term = 0.0
    else:
        ttc = distance / closing_speed
        ttc_term = max(0.0, (ttc_threshold - ttc) / ttc_threshold)
    velocity_term = closing_speed / (closing_speed + 1.0)
    return 1.0 + st_velocity_gain * velocity_term + ttc_gain * ttc_term


class STAPFController(APFController):
    """显式考虑相对速度与 TTC 的 ST-APF 控制器。"""

    def _dynamic_obstacle_factor(self, state: State, obstacle: ObstacleState) -> float:
        """以时空项增强障碍物排斥力。"""

        return spatio_temporal_factor(
            state=state,
            obstacle=obstacle,
            st_velocity_gain=self.config.st_velocity_gain,
            ttc_gain=self.config.ttc_gain,
            ttc_threshold=self.config.ttc_threshold,
        )
