"""几何与道路边界测试。"""

from __future__ import annotations

import pytest

from apflf.env.geometry import box_clearance, box_collision, oriented_box_corners
from apflf.env.road import Road
from apflf.utils.types import RoadGeometry, State


def test_road_projection_and_boundary_margin() -> None:
    """道路投影、边界余量与裁剪应保持一致。"""

    road = Road(RoadGeometry(length=100.0, lane_center_y=1.0, half_width=3.5))

    projected = road.project_point(x=120.0, y=2.25)

    assert projected.s == pytest.approx(100.0)
    assert projected.d == pytest.approx(1.25)
    assert road.boundary_margin(y=3.0, half_extent_y=0.5) == pytest.approx(1.0)
    assert road.contains(y=3.0, half_extent_y=0.5)
    assert road.contains(y=5.0, half_extent_y=0.6) is False
    assert road.clamp_lateral_position(y=10.0, half_extent_y=0.5) == pytest.approx(4.0)


def test_oriented_box_collision_and_clearance() -> None:
    """姿态矩形的碰撞与间距计算应符合构造几何。"""

    pose_a = State(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    pose_b = State(x=5.0, y=0.0, yaw=0.0, speed=0.0)
    pose_c = State(x=3.5, y=0.0, yaw=0.0, speed=0.0)

    assert box_clearance(pose_a, 4.0, 2.0, pose_b, 4.0, 2.0) == pytest.approx(1.0)
    assert box_collision(pose_a, 4.0, 2.0, pose_b, 4.0, 2.0) is False
    assert box_clearance(pose_a, 4.0, 2.0, pose_c, 4.0, 2.0) == pytest.approx(0.0)
    assert box_collision(pose_a, 4.0, 2.0, pose_c, 4.0, 2.0) is True


def test_oriented_box_corners_rotate_around_pose_center() -> None:
    """旋转矩形顶点应围绕姿态中心对称分布。"""

    pose = State(x=1.0, y=2.0, yaw=0.0, speed=0.0)
    corners = oriented_box_corners(pose, length=4.0, width=2.0)

    assert corners.shape == (4, 2)
    assert corners.mean(axis=0)[0] == pytest.approx(1.0)
    assert corners.mean(axis=0)[1] == pytest.approx(2.0)
