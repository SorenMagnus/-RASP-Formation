"""确定性场景工厂。"""

from __future__ import annotations

import numpy as np

from apflf.utils.types import ProjectConfig, RoadGeometry, ScenarioSetup, State


class ScenarioFactory:
    """根据配置与 seed 构建确定性初始场景。"""

    def __init__(self, config: ProjectConfig) -> None:
        """保存项目配置引用。"""

        self.config = config

    def build(self, seed: int) -> ScenarioSetup:
        """生成单次实验的初始化场景。"""

        rng = np.random.default_rng(seed)
        road_cfg = self.config.scenario.road
        road = RoadGeometry(
            length=road_cfg.length,
            lane_center_y=road_cfg.lane_center_y,
            half_width=road_cfg.half_width,
        )

        initial_states: list[State] = []
        desired_offsets: list[tuple[float, float]] = []
        for index in range(self.config.scenario.vehicle_count):
            offset_x = -index * self.config.scenario.spacing
            desired_offsets.append((offset_x, 0.0))
            initial_states.append(
                State(
                    x=offset_x + float(rng.normal(0.0, self.config.scenario.spawn_jitter_std)),
                    y=road.lane_center_y
                    + float(rng.normal(0.0, self.config.scenario.spawn_jitter_std)),
                    yaw=float(rng.normal(0.0, 0.01)),
                    speed=self.config.scenario.initial_speed,
                )
            )

        return ScenarioSetup(
            road=road,
            initial_states=tuple(initial_states),
            desired_offsets=tuple(desired_offsets),
            goal_x=self.config.scenario.goal_x,
            obstacle_configs=self.config.scenario.obstacles,
        )
