"""几何工具与碰撞原语。"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

from apflf.utils.types import ObstacleState, State

PoseLike: TypeAlias = State | ObstacleState


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""

    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def rotation_matrix(yaw: float) -> np.ndarray:
    """构造二维旋转矩阵。"""

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)


def euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """计算二维欧氏距离。"""

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.hypot(dx, dy)


def oriented_box_corners(pose: PoseLike, length: float, width: float) -> np.ndarray:
    """计算姿态矩形的四个顶点。

    返回:
        形状为 `(4, 2)` 的数组，顶点顺序为逆时针。
    """

    if length <= 0.0 or width <= 0.0:
        raise ValueError("矩形长度与宽度必须为正数。")
    half_length = 0.5 * length
    half_width = 0.5 * width
    local_corners = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    rotated = local_corners @ rotation_matrix(pose.yaw).T
    return rotated + np.asarray([pose.x, pose.y], dtype=float)


def _polygon_axes(polygon: np.ndarray) -> list[np.ndarray]:
    """构造多边形 SAT 分离轴集合。"""

    axes: list[np.ndarray] = []
    for index in range(len(polygon)):
        edge = polygon[(index + 1) % len(polygon)] - polygon[index]
        normal = np.asarray([-edge[1], edge[0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            axes.append(normal / norm)
    return axes


def _project_polygon(polygon: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """将多边形投影到分离轴上。"""

    projection = polygon @ axis
    return float(np.min(projection)), float(np.max(projection))


def polygons_intersect(polygon_a: np.ndarray, polygon_b: np.ndarray, tolerance: float = 1e-9) -> bool:
    """使用 SAT 判断两个凸多边形是否相交。"""

    for axis in [*_polygon_axes(polygon_a), *_polygon_axes(polygon_b)]:
        min_a, max_a = _project_polygon(polygon_a, axis)
        min_b, max_b = _project_polygon(polygon_b, axis)
        if max_a < min_b - tolerance or max_b < min_a - tolerance:
            return False
    return True


def point_to_segment_distance(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """计算点到线段的最短距离。"""

    segment = segment_end - segment_start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - segment_start))
    ratio = float(np.dot(point - segment_start, segment) / denom)
    ratio = min(max(ratio, 0.0), 1.0)
    projection = segment_start + ratio * segment
    return float(np.linalg.norm(point - projection))


def polygon_clearance(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """计算两个凸多边形之间的最小间距。

    说明:
        若多边形相交，则返回 0.0。
    """

    if polygons_intersect(polygon_a, polygon_b):
        return 0.0

    min_distance = math.inf
    for point in polygon_a:
        for index in range(len(polygon_b)):
            candidate = point_to_segment_distance(
                point,
                polygon_b[index],
                polygon_b[(index + 1) % len(polygon_b)],
            )
            min_distance = min(min_distance, candidate)
    for point in polygon_b:
        for index in range(len(polygon_a)):
            candidate = point_to_segment_distance(
                point,
                polygon_a[index],
                polygon_a[(index + 1) % len(polygon_a)],
            )
            min_distance = min(min_distance, candidate)
    return float(min_distance)


def box_clearance(
    pose_a: PoseLike,
    length_a: float,
    width_a: float,
    pose_b: PoseLike,
    length_b: float,
    width_b: float,
) -> float:
    """计算两个姿态矩形之间的最小间距。"""

    polygon_a = oriented_box_corners(pose_a, length_a, width_a)
    polygon_b = oriented_box_corners(pose_b, length_b, width_b)
    return polygon_clearance(polygon_a, polygon_b)


def box_collision(
    pose_a: PoseLike,
    length_a: float,
    width_a: float,
    pose_b: PoseLike,
    length_b: float,
    width_b: float,
    tolerance: float = 1e-9,
) -> bool:
    """判断两个姿态矩形是否发生碰撞。"""

    polygon_a = oriented_box_corners(pose_a, length_a, width_a)
    polygon_b = oriented_box_corners(pose_b, length_b, width_b)
    return polygons_intersect(polygon_a, polygon_b, tolerance=tolerance)
