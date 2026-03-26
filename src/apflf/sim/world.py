"""World loop and snapshot recording."""

from __future__ import annotations

from time import perf_counter

from apflf.controllers.base import Controller
from apflf.decision.mode_base import ModeDecisionModule
from apflf.env.dynamics import VehicleDynamics
from apflf.env.obstacles import ObstacleModel, build_obstacle_models, sample_obstacles
from apflf.safety.safety_filter import SafetyFilter
from apflf.utils.types import Observation, ObstacleState, ScenarioSetup, Snapshot, State


class World:
    """Deterministic simulation world."""

    def __init__(
        self,
        *,
        scenario: ScenarioSetup,
        dynamics: VehicleDynamics,
        controller: Controller,
        mode_decision: ModeDecisionModule,
        safety_filter: SafetyFilter,
        dt: float,
    ) -> None:
        self.scenario = scenario
        self.dynamics = dynamics
        self.controller = controller
        self.mode_decision = mode_decision
        self.safety_filter = safety_filter
        self.dt = dt
        self._states = scenario.initial_states
        self._time = 0.0
        self._step_index = 0
        self._snapshots: list[Snapshot] = []
        self._obstacle_models: tuple[ObstacleModel, ...] = build_obstacle_models(
            scenario.obstacle_configs
        )

    @property
    def snapshots(self) -> tuple[Snapshot, ...]:
        return tuple(self._snapshots)

    @property
    def states(self) -> tuple[State, ...]:
        return self._states

    @property
    def obstacle_states(self) -> tuple[ObstacleState, ...]:
        return sample_obstacles(self._obstacle_models, self._time)

    def build_observation(self) -> Observation:
        return Observation(
            step_index=self._step_index,
            time=self._time,
            states=self._states,
            road=self.scenario.road,
            goal_x=self.scenario.goal_x,
            desired_offsets=self.scenario.desired_offsets,
            obstacles=self.obstacle_states,
        )

    def step(self) -> Snapshot:
        step_start = perf_counter()
        observation = self.build_observation()
        mode_start = perf_counter()
        mode = self.mode_decision.select_mode(observation)
        mode_runtime = perf_counter() - mode_start
        controller_start = perf_counter()
        nominal_actions = self.controller.compute_actions(observation=observation, mode=mode)
        controller_runtime = perf_counter() - controller_start
        nominal_diagnostics = self.controller.consume_step_diagnostics()
        safety_start = perf_counter()
        safety_result = self.safety_filter.filter(
            nominal_actions=nominal_actions,
            observation=observation,
        )
        safety_runtime = perf_counter() - safety_start
        next_states = tuple(
            self.dynamics.step(state=state, action=action, dt=self.dt)
            for state, action in zip(self._states, safety_result.safe_actions, strict=True)
        )
        self._time += self.dt
        self._step_index += 1
        obstacle_states = self.obstacle_states
        snapshot = Snapshot(
            step_index=self._step_index,
            time=self._time,
            mode=mode,
            states=next_states,
            nominal_actions=nominal_actions,
            safe_actions=safety_result.safe_actions,
            obstacles=obstacle_states,
            safety_corrections=safety_result.correction_norms,
            safety_slacks=safety_result.slack_values,
            safety_fallbacks=safety_result.fallback_flags,
            qp_solve_times=safety_result.qp_solve_times,
            qp_iterations=safety_result.qp_iterations,
            step_runtime=perf_counter() - step_start,
            mode_runtime=mode_runtime,
            controller_runtime=controller_runtime,
            safety_runtime=safety_runtime,
            nominal_diagnostics=nominal_diagnostics,
        )
        self._states = next_states
        self._snapshots.append(snapshot)
        return snapshot

    def run(self, steps: int) -> tuple[Snapshot, ...]:
        for _ in range(steps):
            self.step()
        return self.snapshots
