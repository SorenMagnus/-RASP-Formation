"""道路几何与边界查询。"""

from __future__ import annotations

from apflf.utils.types import FrenetPose, RoadGeometry


class Road:
    """直线路段道路模型。

    当前实现假定道路中心线与 x 轴平行，因此 Frenet 投影与边界约束都可解析计算。
    这能满足 Phase B 对边界、投影和可测试几何接口的需求，同时为后续扩展保留稳定 API。
    """

    def __init__(self, geometry: RoadGeometry) -> None:
        """构造道路几何对象。"""

        self.geometry = geometry

    def centerline_point(self, s: float) -> tuple[float, float]:
        """返回中心线在弧长坐标 `s` 处的全局位置。"""

        clamped_s = min(max(s, 0.0), self.geometry.length)
        return (clamped_s, self.geometry.lane_center_y)

    def project_point(self, x: float, y: float) -> FrenetPose:
        """将二维点投影到直线路段 Frenet 坐标。"""

        s = min(max(x, 0.0), self.geometry.length)
        d = y - self.geometry.lane_center_y
        return FrenetPose(s=s, d=d)

    def lateral_error(self, y: float) -> float:
        """计算车辆相对道路中心线的横向误差。"""

        return y - self.geometry.lane_center_y

    def boundary_margin(self, y: float, half_extent_y: float = 0.0) -> float:
        """计算到道路边界的剩余余量。

        参数:
            y: 目标点或目标体中心的横向坐标。
            half_extent_y: 目标体在横向方向上的半尺寸，单位 m。
        """

        if half_extent_y < 0.0:
            raise ValueError("横向半尺寸不能为负数。")
        return self.geometry.half_width - (abs(self.lateral_error(y)) + half_extent_y)

    def contains(self, y: float, half_extent_y: float = 0.0) -> bool:
        """判断目标体是否完全位于道路边界内。"""

        return self.boundary_margin(y, half_extent_y=half_extent_y) >= 0.0

    def clamp_lateral_position(self, y: float, half_extent_y: float = 0.0) -> float:
        """将横向位置裁剪到道路可行区域内。"""

        if half_extent_y < 0.0:
            raise ValueError("横向半尺寸不能为负数。")
        max_offset = max(self.geometry.half_width - half_extent_y, 0.0)
        min_y = self.geometry.lane_center_y - max_offset
        max_y = self.geometry.lane_center_y + max_offset
        return min(max(y, min_y), max_y)
