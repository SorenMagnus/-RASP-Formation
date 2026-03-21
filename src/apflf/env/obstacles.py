"""静态与动态障碍物模型。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from apflf.env.geometry import normalize_angle
from apflf.utils.types import ObstacleConfig, ObstacleState


class ObstacleModel(ABC):
    """障碍物轨迹模型抽象接口。"""

    def __init__(self, config: ObstacleConfig) -> None:
        """保存障碍物配置。"""

        self.config = config

    @abstractmethod
    def state_at(self, time: float) -> ObstacleState:
        """返回给定时刻的障碍物状态。"""


class StaticObstacle(ObstacleModel):
    """静态障碍物。"""

    def state_at(self, time: float) -> ObstacleState:
        """返回固定障碍物状态。"""

        if not math.isfinite(time) or time < 0.0:
            raise ValueError("障碍物查询时间必须是有限非负数。")
        return ObstacleState(
            obstacle_id=self.config.obstacle_id,
            x=self.config.x,
            y=self.config.y,
            yaw=normalize_angle(self.config.yaw),
            speed=0.0,
            length=self.config.length,
            width=self.config.width,
        )


class ConstantVelocityObstacle(ObstacleModel):
    """匀速直线动态障碍物。"""

    def state_at(self, time: float) -> ObstacleState:
        """返回匀速模型在给定时刻的状态。"""

        if not math.isfinite(time) or time < 0.0:
            raise ValueError("障碍物查询时间必须是有限非负数。")
        distance = self.config.speed * time
        x = self.config.x + distance * math.cos(self.config.yaw)
        y = self.config.y + distance * math.sin(self.config.yaw)
        return ObstacleState(
            obstacle_id=self.config.obstacle_id,
            x=x,
            y=y,
            yaw=normalize_angle(self.config.yaw),
            speed=self.config.speed,
            length=self.config.length,
            width=self.config.width,
        )


def build_obstacle_models(configs: tuple[ObstacleConfig, ...]) -> tuple[ObstacleModel, ...]:
    """根据配置构造障碍物模型。"""

    models: list[ObstacleModel] = []
    for config in configs:
        if config.motion_model == "static":
            models.append(StaticObstacle(config))
        elif config.motion_model == "constant_velocity":
            models.append(ConstantVelocityObstacle(config))
        else:
            raise ValueError(f"不支持的障碍物运动模型: {config.motion_model}")
    return tuple(models)


def sample_obstacles(models: tuple[ObstacleModel, ...], time: float) -> tuple[ObstacleState, ...]:
    """采样给定时刻的所有障碍物状态。"""

    return tuple(model.state_at(time) for model in models)
