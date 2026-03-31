"""Lightweight training environment for the stage-1 RL supervisor."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from apflf.controllers.base import build_controller
from apflf.decision.mode_base import build_mode_decision
from apflf.env.dynamics import VehicleDynamics
from apflf.env.geometry import box_clearance, rotation_matrix
from apflf.env.obstacles import build_obstacle_models, sample_obstacles
from apflf.env.road import Road
from apflf.env.scenarios import ScenarioFactory
from apflf.safety.cbf import boundary_barrier
from apflf.safety.safety_filter import build_safety_filter
from apflf.utils.types import Observation, ProjectConfig, State


class SupervisorTrainingEnv:
    """A white-box rollout environment for `rl_param_only` training."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.factory = ScenarioFactory(config=config)
        self._seed = 0
        self._max_steps = config.simulation.steps
        self._feature_builder = None
        self._fallback_fsm = None
        self._controller = None
        self._safety_filter = None
        self._dynamics = None
        self._scenario = None
        self._road = None
        self._obstacle_models = ()
        self._states: tuple[State, ...] = ()
        self._time = 0.0
        self._step_index = 0
        self._current_observation: Observation | None = None
        self._current_mode = config.decision.default_mode
        self._previous_theta = config.decision.rl.theta.default
        self._previous_theta_delta = (0.0, 0.0, 0.0, 0.0)

    @property
    def observation_dim(self) -> int:
        if self._feature_builder is None:
            self.reset(seed=self._seed)
        assert self._feature_builder is not None
        return self._feature_builder.feature_dim

    @property
    def theta_bounds(self) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        return (self.config.decision.rl.theta.lower, self.config.decision.rl.theta.upper)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._seed = self._seed if seed is None else int(seed)
        self._scenario = self.factory.build(seed=self._seed)
        self._road = Road(self._scenario.road)
        self._obstacle_models = build_obstacle_models(self._scenario.obstacle_configs)
        self._dynamics = VehicleDynamics(
            wheelbase=self.config.simulation.wheelbase,
            bounds=self.config.simulation.bounds,
        )
        self._controller = build_controller(
            config=self.config.controller,
            bounds=self.config.simulation.bounds,
            road=self._road,
            target_speed=self.config.simulation.target_speed,
            wheelbase=self.config.simulation.wheelbase,
            dt=self.config.simulation.dt,
        )
        self._safety_filter = build_safety_filter(
            config=self.config.safety,
            bounds=self.config.simulation.bounds,
            road=self._road,
            wheelbase=self.config.simulation.wheelbase,
            vehicle_length=self.config.controller.vehicle_length,
            vehicle_width=self.config.controller.vehicle_width,
            dt=self.config.simulation.dt,
        )
        fallback_decision_config = replace(self.config.decision, kind="fsm")
        self._fallback_fsm = build_mode_decision(
            config=fallback_decision_config,
            vehicle_length=self.config.controller.vehicle_length,
            vehicle_width=self.config.controller.vehicle_width,
            safe_distance=self.config.safety.safe_distance,
        )
        self._fallback_fsm.reset(self._seed)
        from apflf.rl.features import SupervisorObservationBuilder

        self._feature_builder = SupervisorObservationBuilder(
            vehicle_length=self.config.controller.vehicle_length,
            vehicle_width=self.config.controller.vehicle_width,
            history_length=self.config.decision.rl.observation_history,
            interaction_limit=self.config.decision.rl.interaction_limit,
        )
        self._feature_builder.reset()
        self._states = self._scenario.initial_states
        self._time = 0.0
        self._step_index = 0
        self._previous_theta = self.config.decision.rl.theta.default
        self._previous_theta_delta = (0.0, 0.0, 0.0, 0.0)
        return self._prepare_supervisor_observation()

    def step(
        self,
        theta: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._scenario is None or self._fallback_fsm is None or self._feature_builder is None:
            raise RuntimeError("Call `reset()` before stepping the training environment.")
        assert self._controller is not None
        assert self._dynamics is not None
        assert self._safety_filter is not None
        assert self._current_observation is not None

        projected_theta = self._project_theta(theta)
        nominal_actions = self._controller.compute_actions(
            observation=self._current_observation,
            mode=self._current_mode,
            theta=projected_theta,
        )
        safety_result = self._safety_filter.filter(
            nominal_actions=nominal_actions,
            observation=self._current_observation,
        )
        next_states = tuple(
            self._dynamics.step(state=state, action=action, dt=self.config.simulation.dt)
            for state, action in zip(self._states, safety_result.safe_actions, strict=True)
        )
        next_time = self._time + self.config.simulation.dt
        next_step = self._step_index + 1
        next_observation = Observation(
            step_index=next_step,
            time=next_time,
            states=next_states,
            road=self._scenario.road,
            goal_x=self._scenario.goal_x,
            desired_offsets=self._scenario.desired_offsets,
            obstacles=sample_obstacles(self._obstacle_models, next_time),
        )
        reward_terms = self._reward_terms(
            current_observation=self._current_observation,
            next_observation=next_observation,
            safety_result=safety_result,
            theta=projected_theta,
        )
        reward = float(sum(reward_terms.values()))
        collision, boundary = self._terminal_violations(next_states, next_observation.obstacles)
        reached_goal = next_states[0].x >= self._scenario.goal_x - self.config.scenario.goal_tolerance
        terminated = bool(collision or boundary or reached_goal)
        truncated = bool(next_step >= self._max_steps and not terminated)

        self._feature_builder.observe_feedback(
            safety_corrections=safety_result.correction_norms,
            safety_slacks=safety_result.slack_values,
            safety_fallbacks=safety_result.fallback_flags,
        )
        theta_delta = tuple(
            float(current - previous)
            for current, previous in zip(projected_theta, self._previous_theta, strict=True)
        )
        self._previous_theta = projected_theta
        self._previous_theta_delta = theta_delta
        self._states = next_states
        self._time = next_time
        self._step_index = next_step

        if terminated or truncated:
            next_supervisor_observation = np.zeros(self.observation_dim, dtype=float)
        else:
            next_supervisor_observation = self._prepare_supervisor_observation()

        info = {
            "mode": self._current_mode,
            "reward_terms": reward_terms,
            "reward_total": reward,
            "theta": projected_theta,
            "theta_delta": theta_delta,
            "theta_delta_linf": float(np.max(np.abs(np.asarray(theta_delta, dtype=float)))),
            "collision": collision,
            "boundary_violation": boundary,
            "reached_goal": reached_goal,
            "fallback_events": int(np.count_nonzero(safety_result.fallback_flags)),
            "safety_interventions": int(
                np.count_nonzero(
                    np.asarray(safety_result.correction_norms, dtype=float)
                    > self.config.decision.rl.reward.correction_epsilon
                )
            ),
            "intervention_active_step": float(reward_terms["intervene"] < 0.0),
            "qp_engagement_ratio_step": float(
                np.mean(np.asarray(safety_result.qp_solve_times, dtype=float) > 0.0)
            )
            if safety_result.qp_solve_times
            else 0.0,
            "fallback_ratio_step": float(np.mean(np.asarray(safety_result.fallback_flags, dtype=float)))
            if safety_result.fallback_flags
            else 0.0,
        }
        return next_supervisor_observation, reward, terminated, truncated, info

    def _prepare_supervisor_observation(self) -> np.ndarray:
        assert self._scenario is not None
        assert self._fallback_fsm is not None
        assert self._feature_builder is not None
        self._current_observation = Observation(
            step_index=self._step_index,
            time=self._time,
            states=self._states,
            road=self._scenario.road,
            goal_x=self._scenario.goal_x,
            desired_offsets=self._scenario.desired_offsets,
            obstacles=sample_obstacles(self._obstacle_models, self._time),
        )
        decision = self._fallback_fsm.select(self._current_observation, self._step_index)
        self._current_mode = decision.mode
        return self._feature_builder.build(
            observation=self._current_observation,
            mode=self._current_mode,
            previous_theta=self._previous_theta,
            previous_theta_delta=self._previous_theta_delta,
        )

    def _project_theta(self, theta: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        lower, upper = self.theta_bounds
        return tuple(float(np.clip(value, lo, hi)) for value, lo, hi in zip(theta, lower, upper, strict=True))

    def _reward_terms(
        self,
        *,
        current_observation: Observation,
        next_observation: Observation,
        safety_result,
        theta: tuple[float, float, float, float],
    ) -> dict[str, float]:
        reward_config = self.config.decision.rl.reward
        vehicle_count = max(len(current_observation.states), 1)
        progress = float(next_observation.states[0].x - current_observation.states[0].x)
        current_error = self._formation_error(current_observation)
        next_error = self._formation_error(next_observation)
        theta_delta = np.asarray(theta, dtype=float) - np.asarray(self._previous_theta, dtype=float)
        theta_delta_linf = float(np.max(np.abs(theta_delta))) if theta_delta.size else 0.0
        correction_norms = np.asarray(safety_result.correction_norms, dtype=float)
        slack_values = np.asarray(safety_result.slack_values, dtype=float)
        qp_solve_times = np.asarray(safety_result.qp_solve_times, dtype=float)
        fallback_flags = np.asarray(safety_result.fallback_flags, dtype=bool)
        intervention_active = float(
            np.max(correction_norms) > reward_config.correction_epsilon if correction_norms.size else False
        )
        qp_engagement_ratio = float(np.mean(qp_solve_times > 0.0)) if qp_solve_times.size else 0.0
        fallback_ratio = float(np.count_nonzero(fallback_flags) / vehicle_count) if fallback_flags.size else 0.0
        max_slack = float(np.max(slack_values)) if slack_values.size else 0.0
        collision, boundary = self._terminal_violations(next_observation.states, next_observation.obstacles)
        reached_goal = next_observation.states[0].x >= self._scenario.goal_x - self.config.scenario.goal_tolerance
        return {
            "progress": reward_config.progress_weight * progress,
            "formation": reward_config.formation_weight * (current_error - next_error),
            "intervene": -reward_config.intervention_weight * intervention_active,
            "qp": -reward_config.qp_weight * qp_engagement_ratio,
            "fallback": -reward_config.fallback_weight * fallback_ratio,
            "slack": -reward_config.slack_weight * max_slack,
            "theta_rate": -reward_config.theta_rate_weight * theta_delta_linf,
            "goal": reward_config.goal_reward if reached_goal else 0.0,
            "collision": -reward_config.collision_penalty if collision else 0.0,
            "boundary": -reward_config.boundary_penalty if boundary else 0.0,
        }

    def _formation_error(self, observation: Observation) -> float:
        leader = observation.states[0]
        leader_position = np.asarray([leader.x, leader.y], dtype=float)
        leader_rotation = rotation_matrix(leader.yaw)
        errors: list[float] = []
        for index, state in enumerate(observation.states[1:], start=1):
            desired_position = leader_position + leader_rotation @ np.asarray(
                observation.desired_offsets[index],
                dtype=float,
            )
            errors.append(
                float(np.linalg.norm(desired_position - np.asarray([state.x, state.y], dtype=float)))
            )
        return float(np.mean(errors)) if errors else 0.0

    def _terminal_violations(self, states, obstacles) -> tuple[bool, bool]:
        collision = False
        for index, state in enumerate(states):
            if boundary_barrier(
                state=state,
                road=self._road,
                vehicle_length=self.config.controller.vehicle_length,
                vehicle_width=self.config.controller.vehicle_width,
            ) < 0.0:
                return False, True
            for other_index, other_state in enumerate(states):
                if other_index <= index:
                    continue
                if box_clearance(
                    state,
                    self.config.controller.vehicle_length,
                    self.config.controller.vehicle_width,
                    other_state,
                    self.config.controller.vehicle_length,
                    self.config.controller.vehicle_width,
                ) <= 0.0:
                    collision = True
            for obstacle in obstacles:
                if box_clearance(
                    state,
                    self.config.controller.vehicle_length,
                    self.config.controller.vehicle_width,
                    obstacle,
                    obstacle.length,
                    obstacle.width,
                ) <= 0.0:
                    collision = True
        return collision, False
