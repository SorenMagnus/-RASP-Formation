"""Leader-Follower 与一致性耦合项。"""

from __future__ import annotations

import numpy as np

from apflf.decision.mode_base import parse_mode_label
from apflf.env.geometry import rotation_matrix
from apflf.utils.types import Observation


class LeaderFollowerMixin:
    """编队与一致性耦合项混入类。"""

    def _desired_global_position(self, observation: Observation, index: int, mode: str) -> np.ndarray:
        """将 leader 坐标系下的期望相对位姿转换到全局坐标。"""

        parsed_mode = parse_mode_label(mode)
        if index > 1 and parsed_mode.behavior != "follow":
            predecessor = observation.states[index - 1]
            predecessor_offset = np.asarray(
                self._relative_offset_for_mode(observation=observation, index=index - 1, mode=mode),
                dtype=float,
            )
            current_offset = np.asarray(
                self._relative_offset_for_mode(observation=observation, index=index, mode=mode),
                dtype=float,
            )
            rotated_offset = rotation_matrix(predecessor.yaw) @ (current_offset - predecessor_offset)
            return np.asarray([predecessor.x, predecessor.y], dtype=float) + rotated_offset

        leader = observation.states[0]
        offset = np.asarray(
            self._relative_offset_for_mode(observation=observation, index=index, mode=mode),
            dtype=float,
        )
        rotated_offset = rotation_matrix(leader.yaw) @ offset
        return np.asarray([leader.x, leader.y], dtype=float) + rotated_offset

    def _formation_force(self, observation: Observation, index: int, mode: str) -> np.ndarray:
        """计算 follower 对 leader 参考队形的误差反馈。"""

        if index == 0:
            return np.zeros(2, dtype=float)
        current_state = observation.states[index]
        desired_position = self._desired_global_position(observation, index, mode)
        current_position = np.asarray([current_state.x, current_state.y], dtype=float)
        return self.config.formation_gain * (desired_position - current_position)

    def _consensus_force(self, observation: Observation, index: int, mode: str) -> np.ndarray:
        """计算链式通信图上的一致性校正力。"""

        if index == 0:
            return np.zeros(2, dtype=float)

        neighbors: list[int] = [0]
        if index - 1 >= 0:
            neighbors.append(index - 1)
        if index + 1 < len(observation.states):
            neighbors.append(index + 1)

        current_state = observation.states[index]
        current_position = np.asarray([current_state.x, current_state.y], dtype=float)
        leader_rotation = rotation_matrix(observation.states[0].yaw)
        consensus = np.zeros(2, dtype=float)
        for neighbor_index in neighbors:
            neighbor_state = observation.states[neighbor_index]
            neighbor_position = np.asarray([neighbor_state.x, neighbor_state.y], dtype=float)
            desired_relative = np.asarray(
                self._relative_offset_for_mode(
                    observation=observation,
                    index=neighbor_index,
                    mode=mode,
                ),
                dtype=float,
            ) - np.asarray(
                self._relative_offset_for_mode(
                    observation=observation,
                    index=index,
                    mode=mode,
                ),
                dtype=float,
            )
            desired_relative_global = leader_rotation @ desired_relative
            consensus += (neighbor_position - current_position) - desired_relative_global
        return self.config.consensus_gain * consensus
