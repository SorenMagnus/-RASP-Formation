"""经典人工势场基线控制器。"""

from __future__ import annotations

import numpy as np

from apflf.controllers.base import BaseNominalController
from apflf.utils.types import Action, Observation


class APFController(BaseNominalController):
    """经典 APF 基线。

    理论映射:
        - 吸引项: 驱动车辆向静态终点编队参考点收敛。
        - 排斥项: 对道路边界、障碍物与邻车产生排斥。
        - 输出项: 通过虚拟力映射得到连续控制输入 `u_nom`。
    """

    def compute_actions(self, observation: Observation, mode: str) -> tuple[Action, ...]:
        """输出经典 APF 名义控制。"""

        actions: list[Action] = []
        repulsive_scale, road_scale, _ = self._mode_gain_scales(mode)
        for index, state in enumerate(observation.states):
            target = self._static_goal_target(observation, index, mode)
            attractive_force = self._attractive_force(state, target)
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
            total_force = attractive_force + road_force + obstacle_force + peer_force + behavior_force
            target_speed = self._mode_adjusted_target_speed(
                self._braking_speed(state.x, float(target[0])),
                mode,
            )
            actions.append(
                self._force_to_action(
                    state=state,
                    force=total_force,
                    target_speed=target_speed,
                )
            )
        return tuple(actions)
