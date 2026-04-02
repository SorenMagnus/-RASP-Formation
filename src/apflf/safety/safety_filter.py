"""Safety filter abstractions and the Phase D CBF-QP implementation."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from apflf.env.dynamics import VehicleDynamics
from apflf.env.road import Road
from apflf.safety.cbf import (
    AffineConstraint,
    boundary_barrier,
    linearize_discrete_barrier,
    obstacle_barrier,
    step_with_discrete_dynamics,
)
from apflf.safety.qp_solver import OSQPQPSolver
from apflf.utils.types import Action, InputBounds, Observation, SafetyConfig, SafetyFilterResult, State

LOGGER = logging.getLogger(__name__)


class SafetyFilter(ABC):
    """Abstract interface for nominal-to-safe control filtering."""

    @abstractmethod
    def filter(
        self,
        nominal_actions: tuple[Action, ...],
        observation: Observation,
    ) -> SafetyFilterResult:
        """Return safe actions and optional diagnostics."""


class PassThroughSafetyFilter(SafetyFilter):
    """Phase A-compatible filter that simply forwards nominal actions."""

    def filter(
        self,
        nominal_actions: tuple[Action, ...],
        observation: Observation,
    ) -> SafetyFilterResult:
        if len(nominal_actions) != len(observation.states):
            raise ValueError("The number of nominal actions must match the number of vehicles.")
        zeros = tuple(0.0 for _ in nominal_actions)
        flags = tuple(False for _ in nominal_actions)
        iterations = tuple(0 for _ in nominal_actions)
        return SafetyFilterResult(
            safe_actions=nominal_actions,
            correction_norms=zeros,
            slack_values=zeros,
            fallback_flags=flags,
            qp_solve_times=zeros,
            qp_iterations=iterations,
        )


class CBFQPSafetyFilter(SafetyFilter):
    """Discrete-time CBF-QP safety filter with exact one-step consistency."""

    def __init__(
        self,
        *,
        config: SafetyConfig,
        bounds: InputBounds,
        road: Road,
        wheelbase: float,
        vehicle_length: float,
        vehicle_width: float,
        dt: float,
    ) -> None:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("`dt` must be a finite positive scalar.")
        if wheelbase <= 0.0 or not math.isfinite(wheelbase):
            raise ValueError("`wheelbase` must be a finite positive scalar.")
        self.config = config
        self.bounds = bounds
        self.road = road
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.dt = dt
        self.min_lookahead_steps = max(1, int(math.ceil(0.5 / self.dt)))
        self.max_lookahead_steps = max(
            self.min_lookahead_steps,
            int(math.ceil(2.5 / self.dt)),
        )
        self.dynamics = VehicleDynamics(wheelbase=wheelbase, bounds=bounds)
        self.solver = OSQPQPSolver()

    def filter(
        self,
        nominal_actions: tuple[Action, ...],
        observation: Observation,
    ) -> SafetyFilterResult:
        if len(nominal_actions) != len(observation.states):
            raise ValueError("The number of nominal actions must match the number of vehicles.")

        nominal_actions = tuple(self._clip_action(action) for action in nominal_actions)
        preview_step_count = max(
            self._resolve_preview_steps(state)
            for state in observation.states
        )
        predicted_peer_trajectories = [
            self._rollout_preview_trajectory(
                state=state,
                action=action,
                steps=preview_step_count,
            )
            for state, action in zip(observation.states, nominal_actions, strict=True)
        ]

        safe_actions: list[Action] = []
        correction_norms: list[float] = []
        slack_values: list[float] = []
        fallback_flags: list[bool] = []
        qp_solve_times: list[float] = []
        qp_iterations: list[int] = []

        for index, (state, nominal_action) in enumerate(
            zip(observation.states, nominal_actions, strict=True)
        ):
            (
                safe_action,
                correction_norm,
                slack_value,
                used_fallback,
                solve_time,
                iteration_count,
            ) = self._filter_single_vehicle(
                index=index,
                state=state,
                nominal_action=nominal_action,
                observation=observation,
                predicted_peer_trajectories=tuple(predicted_peer_trajectories),
            )
            safe_actions.append(safe_action)
            correction_norms.append(correction_norm)
            slack_values.append(slack_value)
            fallback_flags.append(used_fallback)
            qp_solve_times.append(solve_time)
            qp_iterations.append(iteration_count)
            predicted_peer_trajectories[index] = self._rollout_preview_trajectory(
                state=state,
                action=safe_action,
                steps=preview_step_count,
            )

        return SafetyFilterResult(
            safe_actions=tuple(safe_actions),
            correction_norms=tuple(correction_norms),
            slack_values=tuple(slack_values),
            fallback_flags=tuple(fallback_flags),
            qp_solve_times=tuple(qp_solve_times),
            qp_iterations=tuple(qp_iterations),
        )

    def _filter_single_vehicle(
        self,
        *,
        index: int,
        state: State,
        nominal_action: Action,
        observation: Observation,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
    ) -> tuple[Action, float, float, bool, float, int]:
        nominal_vector = np.asarray(nominal_action.to_array(), dtype=float)
        constraints: list[AffineConstraint] = []
        fallback_reasons: list[str] = []

        kappa = self._resolve_kappa()
        preview_steps = self._resolve_preview_steps(state)
        if (
            self._candidate_margin(
                state=state,
                action=nominal_action,
                observation=observation,
                ego_index=index,
                predicted_peer_trajectories=predicted_peer_trajectories,
            )
            >= -1e-5
        ):
            return nominal_action, 0.0, 0.0, False, 0.0, 0

        boundary_current = self._boundary_safety_value(state)
        for preview_step in self._preview_indices(preview_steps):
            boundary_linearization = linearize_discrete_barrier(
                state=state,
                nominal_action=nominal_action,
                current_value=boundary_current,
                next_barrier_fn=self._boundary_safety_value,
                dynamics=self.dynamics,
                dt=self.dt,
                bounds=self.bounds,
                kappa=kappa,
                lookahead_steps=preview_step,
                target_scale=self._target_scale(kappa=kappa, preview_step=preview_step),
            )
            self._register_barrier_constraint(
                constraints=constraints,
                barrier=boundary_linearization,
                barrier_name=f"boundary_h{preview_step}",
                fallback_reasons=fallback_reasons,
            )

        for peer_index, peer_state in enumerate(observation.states):
            if peer_index == index:
                continue
            current_value = obstacle_barrier(
                state=state,
                obstacle=peer_state,
                vehicle_length=self.vehicle_length,
                vehicle_width=self.vehicle_width,
                obstacle_length=self.vehicle_length,
                obstacle_width=self.vehicle_width,
                safe_distance=self.config.safe_distance,
            )
            for preview_step in self._preview_indices(preview_steps):
                predicted_peer = predicted_peer_trajectories[peer_index][preview_step - 1]
                linearization = linearize_discrete_barrier(
                    state=state,
                    nominal_action=nominal_action,
                    current_value=current_value,
                    next_barrier_fn=lambda next_state, peer=predicted_peer: obstacle_barrier(
                        state=next_state,
                        obstacle=peer,
                        vehicle_length=self.vehicle_length,
                        vehicle_width=self.vehicle_width,
                        obstacle_length=self.vehicle_length,
                        obstacle_width=self.vehicle_width,
                        safe_distance=self.config.safe_distance,
                    ),
                    dynamics=self.dynamics,
                    dt=self.dt,
                    bounds=self.bounds,
                    kappa=kappa,
                    lookahead_steps=preview_step,
                    target_scale=self._target_scale(kappa=kappa, preview_step=preview_step),
                )
                self._register_barrier_constraint(
                    constraints=constraints,
                    barrier=linearization,
                    barrier_name=f"peer_{peer_index}_h{preview_step}",
                    fallback_reasons=fallback_reasons,
                )

        for obstacle in observation.obstacles:
            current_value = obstacle_barrier(
                state=state,
                obstacle=obstacle,
                vehicle_length=self.vehicle_length,
                vehicle_width=self.vehicle_width,
                obstacle_length=obstacle.length,
                obstacle_width=obstacle.width,
                safe_distance=self.config.safe_distance,
            )
            for preview_step in self._preview_indices(preview_steps):
                predicted_obstacle = self._predict_obstacle_state(
                    obstacle=obstacle,
                    preview_step=preview_step,
                )
                linearization = linearize_discrete_barrier(
                    state=state,
                    nominal_action=nominal_action,
                    current_value=current_value,
                    next_barrier_fn=lambda next_state, obs=predicted_obstacle, obstacle_cfg=obstacle: obstacle_barrier(
                        state=next_state,
                        obstacle=obs,
                        vehicle_length=self.vehicle_length,
                        vehicle_width=self.vehicle_width,
                        obstacle_length=obstacle_cfg.length,
                        obstacle_width=obstacle_cfg.width,
                        safe_distance=self.config.safe_distance,
                    ),
                    dynamics=self.dynamics,
                    dt=self.dt,
                    bounds=self.bounds,
                    kappa=kappa,
                    lookahead_steps=preview_step,
                    target_scale=self._target_scale(kappa=kappa, preview_step=preview_step),
                )
                self._register_barrier_constraint(
                    constraints=constraints,
                    barrier=linearization,
                    barrier_name=f"obstacle_{obstacle.obstacle_id}_h{preview_step}",
                    fallback_reasons=fallback_reasons,
                )

        if fallback_reasons:
            return self._fallback_result(
                state=state,
                nominal_vector=nominal_vector,
                observation=observation,
                ego_index=index,
                predicted_peer_trajectories=predicted_peer_trajectories,
                reason=";".join(fallback_reasons),
            )

        solution = self._solve_qp(nominal_vector=nominal_vector, constraints=constraints)
        if solution.primal is None:
            return self._fallback_result(
                state=state,
                nominal_vector=nominal_vector,
                observation=observation,
                ego_index=index,
                predicted_peer_trajectories=predicted_peer_trajectories,
                reason=f"solver_status={solution.status}",
                qp_solve_time=solution.solve_time,
                qp_iterations=solution.iterations,
            )

        safe_action = self._clip_action(Action(accel=float(solution.primal[0]), steer=float(solution.primal[1])))
        slack_value = float(min(max(solution.primal[2], 0.0), self.config.max_slack))
        correction_norm = float(
            np.linalg.norm(np.asarray(safe_action.to_array(), dtype=float) - nominal_vector)
        )

        verification_reason = self._verify_safe_action(
            state=state,
            safe_action=safe_action,
            observation=observation,
            ego_index=index,
            predicted_peer_trajectories=predicted_peer_trajectories,
        )
        if verification_reason is None and (
            self._candidate_margin(
                state=state,
                action=safe_action,
                observation=observation,
                ego_index=index,
                predicted_peer_trajectories=predicted_peer_trajectories,
            )
            < -1e-5
        ):
            verification_reason = "preview_violation_after_qp"
        if verification_reason is not None:
            return self._fallback_result(
                state=state,
                nominal_vector=nominal_vector,
                observation=observation,
                ego_index=index,
                predicted_peer_trajectories=predicted_peer_trajectories,
                reason=verification_reason,
                qp_solve_time=solution.solve_time,
                qp_iterations=solution.iterations,
            )

        return safe_action, correction_norm, slack_value, False, solution.solve_time, solution.iterations

    def _register_barrier_constraint(
        self,
        *,
        constraints: list[AffineConstraint],
        barrier,
        barrier_name: str,
        fallback_reasons: list[str],
    ) -> None:
        if barrier.constraint is not None:
            constraints.append(barrier.constraint)
            return
        fallback_reasons.append(f"invalid_barrier_{barrier_name}")

    def _solve_qp(
        self,
        *,
        nominal_vector: np.ndarray,
        constraints: list[AffineConstraint],
    ):
        accel_scale = max(self.bounds.accel_max - self.bounds.accel_min, 1e-6)
        steer_scale = max(self.bounds.steer_max - self.bounds.steer_min, 1e-6)
        
        slack_multipliers = [1.0, 10.0, 100.0, 1000.0]
        last_solution = None
        
        for mult in slack_multipliers:
            quadratic_cost = np.diag(
                [
                    1.0 / (accel_scale**2),
                    1.0 / (steer_scale**2),
                    self.config.slack_penalty / mult,
                ]
            )
            linear_cost = np.asarray(
                [
                    -quadratic_cost[0, 0] * nominal_vector[0],
                    -quadratic_cost[1, 1] * nominal_vector[1],
                    0.0,
                ],
                dtype=float,
            )
            rows = [
                np.asarray([1.0, 0.0, 0.0], dtype=float),
                np.asarray([0.0, 1.0, 0.0], dtype=float),
                np.asarray([0.0, 0.0, 1.0], dtype=float),
            ]
            lower_bounds = [self.bounds.accel_min, self.bounds.steer_min, 0.0]
            upper_bounds = [self.bounds.accel_max, self.bounds.steer_max, self.config.max_slack * mult]
            for constraint in constraints:
                rows.append(np.asarray(constraint.coefficients, dtype=float))
                lower_bounds.append(constraint.lower)
                upper_bounds.append(constraint.upper)

            solution = self.solver.solve(
                quadratic_cost=quadratic_cost,
                linear_cost=linear_cost,
                constraint_matrix=np.vstack(rows),
                lower_bounds=np.asarray(lower_bounds, dtype=float),
                upper_bounds=np.asarray(upper_bounds, dtype=float),
            )
            last_solution = solution
            if solution.primal is not None:
                return solution
                
        return last_solution

    def _verify_safe_action(
        self,
        *,
        state: State,
        safe_action: Action,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
    ) -> str | None:
        next_state = step_with_discrete_dynamics(
            state=state,
            action=safe_action,
            dynamics=self.dynamics,
            dt=self.dt,
        )
        boundary_value = self._boundary_safety_value(next_state)
        if boundary_value < -1e-5:
            return "boundary_violation_after_step"

        for peer_index, peer_trajectory in enumerate(predicted_peer_trajectories):
            if peer_index == ego_index:
                continue
            peer_state = peer_trajectory[0]
            if obstacle_barrier(
                state=next_state,
                obstacle=peer_state,
                vehicle_length=self.vehicle_length,
                vehicle_width=self.vehicle_width,
                obstacle_length=self.vehicle_length,
                obstacle_width=self.vehicle_width,
                safe_distance=self.config.safe_distance,
            ) < -1e-5:
                return f"peer_collision_after_step_{peer_index}"

        for obstacle in observation.obstacles:
            predicted_obstacle = self._predict_obstacle_state(obstacle=obstacle, preview_step=1)
            if obstacle_barrier(
                state=next_state,
                obstacle=predicted_obstacle,
                vehicle_length=self.vehicle_length,
                vehicle_width=self.vehicle_width,
                obstacle_length=obstacle.length,
                obstacle_width=obstacle.width,
                safe_distance=self.config.safe_distance,
            ) < -1e-5:
                return f"obstacle_collision_after_step_{obstacle.obstacle_id}"
        return None

    def _fallback_result(
        self,
        *,
        state: State,
        nominal_vector: np.ndarray,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
        reason: str,
        qp_solve_time: float = 0.0,
        qp_iterations: int = 0,
    ) -> tuple[Action, float, float, bool, float, int]:
        fallback_action = self._fallback_action(
            state=state,
            nominal_vector=nominal_vector,
            observation=observation,
            ego_index=ego_index,
            predicted_peer_trajectories=predicted_peer_trajectories,
        )
        correction_norm = float(
            np.linalg.norm(np.asarray(fallback_action.to_array(), dtype=float) - nominal_vector)
        )
        LOGGER.warning("fallback triggered reason=%s step=%s ego_index=%s", reason, observation.step_index, ego_index)
        return fallback_action, correction_norm, 0.0, True, float(max(qp_solve_time, 0.0)), int(max(qp_iterations, 0))

    def _fallback_action(
        self,
        *,
        state: State,
        nominal_vector: np.ndarray,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
    ) -> Action:
        brake_levels = sorted(
            {
                float(np.clip(self.bounds.accel_min, self.bounds.accel_min, self.bounds.accel_max)),
                float(
                    np.clip(
                        -self.config.fallback_brake,
                        self.bounds.accel_min,
                        self.bounds.accel_max,
                    )
                ),
                float(
                    np.clip(
                        -0.5 * self.config.fallback_brake,
                        self.bounds.accel_min,
                        self.bounds.accel_max,
                    )
                ),
                0.0,
            }
        )
        accel_levels = list(brake_levels)
        if state.speed <= 0.25:
            accel_levels.extend(
                [
                    float(np.clip(level, self.bounds.accel_min, self.bounds.accel_max))
                    for level in (0.1, 0.2, 0.3, 0.5)
                ]
            )
        boundary_current = self._boundary_safety_value(state)
        lateral_offset = state.y - self.road.geometry.lane_center_y
        outward_lateral_speed = lateral_offset * state.speed * math.sin(state.yaw)
        boundary_rescue_active = boundary_current <= 0.90 and (
            boundary_current < 0.0 or outward_lateral_speed > 0.05
        )
        if boundary_rescue_active and float(nominal_vector[0]) > 0.0:
            accel_levels.extend(
                [
                    float(np.clip(level, self.bounds.accel_min, self.bounds.accel_max))
                    for level in (
                        0.5,
                        1.0,
                        float(nominal_vector[0]),
                    )
                ]
            )
        accel_levels = sorted(set(accel_levels))
        steer_levels = np.linspace(self.bounds.steer_min, self.bounds.steer_max, num=13, dtype=float)
        best_action: Action | None = None
        best_margin = -math.inf
        best_correction = math.inf
        best_accel = -math.inf
        best_steer_delta = math.inf
        best_action_verification_error: str | None = None
        margin_tolerance = 1e-3
        best_safe_action: Action | None = None
        best_safe_correction = math.inf
        best_safe_accel = -math.inf
        best_safe_steer_delta = math.inf
        best_safe_margin = -math.inf
        best_safe_boundary_action: Action | None = None
        best_safe_boundary_next = -math.inf
        best_safe_boundary_margin = -math.inf
        best_safe_boundary_correction = math.inf
        best_safe_boundary_accel = -math.inf
        best_safe_boundary_steer_delta = math.inf
        best_safe_guided_action: Action | None = None
        best_safe_guided_margin = -math.inf
        best_safe_guided_correction = math.inf
        best_safe_guided_accel = -math.inf
        best_safe_guided_steer_delta = math.inf
        best_verified_action: Action | None = None
        best_verified_margin = -math.inf
        best_verified_correction = math.inf
        best_verified_accel = -math.inf
        best_verified_steer_delta = math.inf
        best_boundary_recovery_action: Action | None = None
        best_boundary_recovery_final = -math.inf
        best_boundary_recovery_margin = -math.inf
        best_boundary_recovery_correction = math.inf
        best_boundary_recovery_accel = math.inf
        best_boundary_recovery_steer_delta = math.inf
        guided_steer_sign = self._preferred_hazard_steer_sign(
            state=state,
            observation=observation,
        )
        near_stop_guided_creep = self._select_near_stop_guided_creep(
            state=state,
            nominal_vector=nominal_vector,
            observation=observation,
            ego_index=ego_index,
            predicted_peer_trajectories=predicted_peer_trajectories,
            accel_levels=tuple(accel_levels),
            steer_levels=tuple(float(level) for level in steer_levels),
            guided_steer_sign=guided_steer_sign,
        )
        guided_mode_active = abs(float(nominal_vector[1])) <= 0.10 and guided_steer_sign != 0.0
        boundary_recovery_tolerance = 0.11

        for accel in accel_levels:
            for steer in steer_levels:
                candidate = Action(accel=float(accel), steer=float(steer))
                next_state = step_with_discrete_dynamics(
                    state=state,
                    action=candidate,
                    dynamics=self.dynamics,
                    dt=self.dt,
                )
                boundary_next = self._boundary_safety_value(next_state)
                margin, boundary_margin, final_boundary_margin = self._candidate_margin_breakdown(
                    state=state,
                    action=candidate,
                    observation=observation,
                    ego_index=ego_index,
                    predicted_peer_trajectories=predicted_peer_trajectories,
                )
                correction = float(
                    np.linalg.norm(np.asarray(candidate.to_array(), dtype=float) - nominal_vector)
                )
                steer_delta = abs(candidate.steer - float(nominal_vector[1]))
                verification_error = self._verify_safe_action(
                    state=state,
                    safe_action=candidate,
                    observation=observation,
                    ego_index=ego_index,
                    predicted_peer_trajectories=predicted_peer_trajectories,
                )
                if boundary_rescue_active and self._is_boundary_only_violation(verification_error):
                    if (
                        final_boundary_margin > best_boundary_recovery_final + boundary_recovery_tolerance
                        or (
                            abs(final_boundary_margin - best_boundary_recovery_final)
                            <= boundary_recovery_tolerance
                            and (
                                boundary_margin > best_boundary_recovery_margin + margin_tolerance
                                or (
                                    abs(boundary_margin - best_boundary_recovery_margin) <= margin_tolerance
                                    and (
                                        correction < best_boundary_recovery_correction - 1e-9
                                        or (
                                            abs(correction - best_boundary_recovery_correction) <= 1e-9
                                            and (
                                                candidate.accel < best_boundary_recovery_accel - 1e-9
                                                or (
                                                    abs(candidate.accel - best_boundary_recovery_accel) <= 1e-9
                                                    and steer_delta
                                                    < best_boundary_recovery_steer_delta - 1e-9
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_boundary_recovery_action = candidate
                        best_boundary_recovery_final = final_boundary_margin
                        best_boundary_recovery_margin = boundary_margin
                        best_boundary_recovery_correction = correction
                        best_boundary_recovery_accel = candidate.accel
                        best_boundary_recovery_steer_delta = steer_delta
                if verification_error is None and margin >= -1e-5:
                    if boundary_rescue_active:
                        if (
                            boundary_next > best_safe_boundary_next + 1e-6
                            or (
                                abs(boundary_next - best_safe_boundary_next) <= 1e-6
                                and (
                                    margin > best_safe_boundary_margin + margin_tolerance
                                    or (
                                        abs(margin - best_safe_boundary_margin) <= margin_tolerance
                                        and (
                                            correction < best_safe_boundary_correction - 1e-9
                                            or (
                                                abs(correction - best_safe_boundary_correction) <= 1e-9
                                                and (
                                                    candidate.accel > best_safe_boundary_accel + 1e-9
                                                    or (
                                                        abs(candidate.accel - best_safe_boundary_accel) <= 1e-9
                                                        and steer_delta < best_safe_boundary_steer_delta - 1e-9
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        ):
                            best_safe_boundary_action = candidate
                            best_safe_boundary_next = boundary_next
                            best_safe_boundary_margin = margin
                            best_safe_boundary_correction = correction
                            best_safe_boundary_accel = candidate.accel
                            best_safe_boundary_steer_delta = steer_delta
                    if (
                        correction < best_safe_correction - 1e-9
                        or (
                            abs(correction - best_safe_correction) <= 1e-9
                            and (
                                candidate.accel > best_safe_accel + 1e-9
                                or (
                                    abs(candidate.accel - best_safe_accel) <= 1e-9
                                    and (
                                        steer_delta < best_safe_steer_delta - 1e-9
                                        or (
                                            abs(steer_delta - best_safe_steer_delta) <= 1e-9
                                            and margin > best_safe_margin + 1e-9
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_safe_action = candidate
                        best_safe_correction = correction
                        best_safe_accel = candidate.accel
                        best_safe_steer_delta = steer_delta
                        best_safe_margin = margin
                    steer_alignment = guided_steer_sign * candidate.steer
                    if guided_mode_active and steer_alignment > 1e-6:
                        if (
                            margin > best_safe_guided_margin + margin_tolerance
                            or (
                                abs(margin - best_safe_guided_margin) <= margin_tolerance
                                and (
                                    correction < best_safe_guided_correction - 1e-9
                                    or (
                                        abs(correction - best_safe_guided_correction) <= 1e-9
                                        and (
                                            candidate.accel > best_safe_guided_accel + 1e-9
                                            or (
                                                abs(candidate.accel - best_safe_guided_accel) <= 1e-9
                                                and steer_delta < best_safe_guided_steer_delta - 1e-9
                                            )
                                        )
                                    )
                                )
                            )
                        ):
                            best_safe_guided_action = candidate
                            best_safe_guided_margin = margin
                            best_safe_guided_correction = correction
                            best_safe_guided_accel = candidate.accel
                            best_safe_guided_steer_delta = steer_delta
                if verification_error is None:
                    if (
                        margin > best_verified_margin + margin_tolerance
                        or (
                            abs(margin - best_verified_margin) <= margin_tolerance
                            and (
                                correction < best_verified_correction - 1e-9
                                or (
                                    abs(correction - best_verified_correction) <= 1e-9
                                    and (
                                        candidate.accel > best_verified_accel + 1e-9
                                        or (
                                            abs(candidate.accel - best_verified_accel) <= 1e-9
                                            and steer_delta < best_verified_steer_delta - 1e-9
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_verified_action = candidate
                        best_verified_margin = margin
                        best_verified_correction = correction
                        best_verified_accel = candidate.accel
                        best_verified_steer_delta = steer_delta
                if (
                    margin > best_margin + margin_tolerance
                    or (
                        abs(margin - best_margin) <= margin_tolerance
                        and (
                            correction < best_correction - 1e-9
                            or (
                                abs(correction - best_correction) <= 1e-9
                                and (
                                    accel > best_accel + 1e-9
                                    or (
                                        abs(accel - best_accel) <= 1e-9
                                        and steer_delta < best_steer_delta - 1e-9
                                    )
                                )
                            )
                        )
                    )
                ):
                    best_action = candidate
                    best_margin = margin
                    best_correction = correction
                    best_accel = accel
                    best_steer_delta = steer_delta
                    best_action_verification_error = verification_error

        if best_safe_boundary_action is not None:
            return self._clip_action(best_safe_boundary_action)
        if best_safe_guided_action is not None:
            if near_stop_guided_creep is not None and best_safe_guided_action.accel <= 0.0:
                return self._clip_action(near_stop_guided_creep)
            return self._clip_action(best_safe_guided_action)
        if best_safe_action is not None:
            if near_stop_guided_creep is not None and best_safe_action.accel <= 0.0:
                return self._clip_action(near_stop_guided_creep)
            return self._clip_action(best_safe_action)
        if near_stop_guided_creep is not None:
            return self._clip_action(near_stop_guided_creep)
        if best_boundary_recovery_action is not None:
            return self._clip_action(best_boundary_recovery_action)
        if (
            best_verified_action is not None
            and best_action_verification_error == "boundary_violation_after_step"
        ):
            return self._clip_action(best_verified_action)
        if best_action is None:
            return self._clip_action(Action(accel=self.bounds.accel_min, steer=0.0))
        return self._clip_action(best_action)

    def _select_near_stop_guided_creep(
        self,
        *,
        state: State,
        nominal_vector: np.ndarray,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
        accel_levels: tuple[float, ...],
        steer_levels: tuple[float, ...],
        guided_steer_sign: float,
    ) -> Action | None:
        """选择低速重定向蠕动动作，只在 exact one-step 安全下允许小幅 preview deficit。"""

        if state.speed > 0.5 or float(nominal_vector[0]) <= 0.0:
            return None

        nominal_steer = float(nominal_vector[1])
        steer_sign = 0.0
        if abs(nominal_steer) > 1e-6:
            steer_sign = math.copysign(1.0, nominal_steer)
        elif guided_steer_sign != 0.0:
            steer_sign = float(math.copysign(1.0, guided_steer_sign))
        if steer_sign == 0.0:
            return None

        epsilon_creep_local = 0.10
        candidate_accels = sorted(
            {
                float(np.clip(level, self.bounds.accel_min, self.bounds.accel_max))
                for level in (
                    *accel_levels,
                    0.1,
                    0.2,
                    0.3,
                    0.5,
                    1.0,
                    float(nominal_vector[0]),
                )
                if level > 0.0
            }
        )
        if not candidate_accels:
            return None

        best_action: Action | None = None
        best_accel = -math.inf
        best_margin = -math.inf
        best_steer_delta = math.inf

        for accel in candidate_accels:
            for steer in steer_levels:
                if steer_sign * steer <= 1e-6:
                    continue
                candidate = Action(accel=float(accel), steer=float(steer))
                verification_error = self._verify_safe_action(
                    state=state,
                    safe_action=candidate,
                    observation=observation,
                    ego_index=ego_index,
                    predicted_peer_trajectories=predicted_peer_trajectories,
                )
                if verification_error is not None:
                    continue
                margin = self._candidate_margin(
                    state=state,
                    action=candidate,
                    observation=observation,
                    ego_index=ego_index,
                    predicted_peer_trajectories=predicted_peer_trajectories,
                )
                if margin < -epsilon_creep_local:
                    continue
                steer_delta = abs(candidate.steer - nominal_steer)
                if (
                    candidate.accel > best_accel + 1e-9
                    or (
                        abs(candidate.accel - best_accel) <= 1e-9
                        and (
                            margin > best_margin + 1e-9
                            or (
                                abs(margin - best_margin) <= 1e-9
                                and steer_delta < best_steer_delta - 1e-9
                            )
                        )
                    )
                ):
                    best_action = candidate
                    best_accel = candidate.accel
                    best_margin = margin
                    best_steer_delta = steer_delta

        if best_action is None:
            return None
        return self._clip_action(best_action)

    def _preferred_hazard_steer_sign(
        self,
        *,
        state: State,
        observation: Observation,
    ) -> float:
        lookahead_distance = max(20.0, 3.0 * state.speed + self.vehicle_length)
        state_front_x = state.x + 0.5 * self.vehicle_length
        relevant_obstacles = []
        for obstacle in observation.obstacles:
            obstacle_rear_x = obstacle.x - 0.5 * obstacle.length
            longitudinal_gap = obstacle_rear_x - state_front_x
            if longitudinal_gap < -self.vehicle_length:
                continue
            if longitudinal_gap > lookahead_distance:
                continue
            relevant_obstacles.append(obstacle)
        if not relevant_obstacles:
            return 0.0

        center_y = observation.road.lane_center_y
        upper_bound = center_y + observation.road.half_width
        lower_bound = center_y - observation.road.half_width
        upper_outer_edge = max(
            (
                obstacle.y + 0.5 * obstacle.width + self.config.safe_distance
                for obstacle in relevant_obstacles
                if obstacle.y >= center_y
            ),
            default=center_y,
        )
        lower_outer_edge = min(
            (
                obstacle.y - 0.5 * obstacle.width - self.config.safe_distance
                for obstacle in relevant_obstacles
                if obstacle.y < center_y
            ),
            default=center_y,
        )
        left_channel = upper_bound - upper_outer_edge
        right_channel = lower_outer_edge - lower_bound
        if left_channel > right_channel + 1e-6:
            return 1.0
        if right_channel > left_channel + 1e-6:
            return -1.0

        nearest = min(
            relevant_obstacles,
            key=lambda obstacle: (obstacle.x - state.x) ** 2 + (obstacle.y - state.y) ** 2,
        )
        return -1.0 if nearest.y >= state.y else 1.0

    def _candidate_margin(
        self,
        *,
        state: State,
        action: Action,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
    ) -> float:
        minimum_margin, _, _ = self._candidate_margin_breakdown(
            state=state,
            action=action,
            observation=observation,
            ego_index=ego_index,
            predicted_peer_trajectories=predicted_peer_trajectories,
        )
        return minimum_margin

    def _candidate_margin_breakdown(
        self,
        *,
        state: State,
        action: Action,
        observation: Observation,
        ego_index: int,
        predicted_peer_trajectories: tuple[tuple[State, ...], ...],
    ) -> tuple[float, float, float]:
        preview_steps = self._resolve_preview_steps(state)
        ego_trajectory = self._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        minimum_margin = math.inf
        minimum_boundary_margin = math.inf
        final_boundary_margin = -math.inf
        for preview_index, next_state in enumerate(ego_trajectory, start=1):
            boundary_margin = self._boundary_safety_value(next_state)
            minimum_boundary_margin = min(minimum_boundary_margin, boundary_margin)
            final_boundary_margin = boundary_margin
            minimum_margin = min(minimum_margin, boundary_margin)
            for peer_index, peer_trajectory in enumerate(predicted_peer_trajectories):
                if peer_index == ego_index:
                    continue
                minimum_margin = min(
                    minimum_margin,
                    obstacle_barrier(
                        state=next_state,
                        obstacle=peer_trajectory[preview_index - 1],
                        vehicle_length=self.vehicle_length,
                        vehicle_width=self.vehicle_width,
                        obstacle_length=self.vehicle_length,
                        obstacle_width=self.vehicle_width,
                        safe_distance=self.config.safe_distance,
                    ),
                )
            for obstacle in observation.obstacles:
                predicted_obstacle = self._predict_obstacle_state(
                    obstacle=obstacle,
                    preview_step=preview_index,
                )
                minimum_margin = min(
                    minimum_margin,
                    obstacle_barrier(
                        state=next_state,
                        obstacle=predicted_obstacle,
                        vehicle_length=self.vehicle_length,
                        vehicle_width=self.vehicle_width,
                        obstacle_length=obstacle.length,
                        obstacle_width=obstacle.width,
                        safe_distance=self.config.safe_distance,
                    ),
                )
        return (
            float(minimum_margin),
            float(minimum_boundary_margin),
            float(final_boundary_margin),
        )

    def _rollout_preview_trajectory(
        self,
        *,
        state: State,
        action: Action,
        steps: int,
    ) -> tuple[State, ...]:
        preview_steps = max(int(steps), 1)
        hold_steps = self._preview_hold_steps(action=action, steps=preview_steps)
        rolled_state = state
        trajectory: list[State] = []
        for preview_index in range(1, preview_steps + 1):
            preview_action = action
            if preview_index > hold_steps:
                preview_action = self._preview_followup_action(
                    reference_action=action,
                    state=rolled_state,
                )
            rolled_state = step_with_discrete_dynamics(
                state=rolled_state,
                action=preview_action,
                dynamics=self.dynamics,
                dt=self.dt,
            )
            trajectory.append(rolled_state)
        return tuple(trajectory)

    def _preview_hold_steps(self, *, action: Action, steps: int) -> int:
        if steps <= 1 or abs(action.steer) <= 1e-6:
            return 1
        return min(2, steps)

    def _preview_followup_action(self, *, reference_action: Action, state: State) -> Action:
        yaw_damping_gain = 1.0
        steer = float(np.clip(-yaw_damping_gain * state.yaw, self.bounds.steer_min, self.bounds.steer_max))
        return Action(accel=reference_action.accel, steer=steer)

    def _resolve_kappa(self) -> float:
        if self.config.barrier_decay <= 1.0:
            return min(max(self.config.barrier_decay, 0.0), 0.95)
        return min(max(self.config.barrier_decay * self.dt, 0.0), 0.95)

    def _resolve_preview_steps(self, state: State) -> int:
        max_brake = max(
            self.config.fallback_brake,
            abs(self.bounds.accel_min),
            1e-6,
        )
        preview_seconds = min(
            2.5,
            max(0.5, 0.4 + state.speed / max_brake),
        )
        preview_steps = int(math.ceil(preview_seconds / self.dt))
        return min(max(preview_steps, self.min_lookahead_steps), self.max_lookahead_steps)

    def _target_scale(self, *, kappa: float, preview_step: int) -> float:
        return float((1.0 - kappa) ** max(preview_step, 1))

    def _preview_indices(self, preview_steps: int) -> tuple[int, ...]:
        anchors = {
            1,
            min(2, preview_steps),
            min(4, preview_steps),
            int(math.ceil(0.5 * preview_steps)),
            int(math.ceil(0.75 * preview_steps)),
            preview_steps,
        }
        return tuple(sorted(step for step in anchors if 1 <= step <= preview_steps))

    def _boundary_safety_value(self, state: State) -> float:
        return boundary_barrier(
            state=state,
            road=self.road,
            vehicle_length=self.vehicle_length,
            vehicle_width=self.vehicle_width,
        ) - self.config.road_boundary_margin

    def _predict_obstacle_state(
        self,
        *,
        obstacle,
        preview_step: int,
    ) -> State:
        preview_time = preview_step * self.dt
        return State(
            x=obstacle.x + preview_time * obstacle.speed * math.cos(obstacle.yaw),
            y=obstacle.y + preview_time * obstacle.speed * math.sin(obstacle.yaw),
            yaw=obstacle.yaw,
            speed=obstacle.speed,
        )

    def _clip_action(self, action: Action) -> Action:
        return Action(
            accel=float(np.clip(action.accel, self.bounds.accel_min, self.bounds.accel_max)),
            steer=float(np.clip(action.steer, self.bounds.steer_min, self.bounds.steer_max)),
        )

    def _is_boundary_only_violation(self, verification_error: str | None) -> bool:
        return verification_error is None or verification_error == "boundary_violation_after_step"


def build_safety_filter(
    *,
    config: SafetyConfig,
    bounds: InputBounds,
    road: Road,
    wheelbase: float,
    vehicle_length: float,
    vehicle_width: float,
    dt: float,
) -> SafetyFilter:
    """Build the configured safety filter implementation."""

    if not config.enabled:
        return PassThroughSafetyFilter()
    if config.solver != "osqp":
        raise ValueError(f"Unsupported safety solver: {config.solver}")
    return CBFQPSafetyFilter(
        config=config,
        bounds=bounds,
        road=road,
        wheelbase=wheelbase,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        dt=dt,
    )
