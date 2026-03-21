"""Phase D CBF helpers based on the exact discrete simulation step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from apflf.env.dynamics import VehicleDynamics
from apflf.env.geometry import box_clearance, oriented_box_corners
from apflf.env.road import Road
from apflf.utils.types import Action, InputBounds, ObstacleState, State

BarrierFunction = Callable[[State], float]


@dataclass(frozen=True)
class AffineConstraint:
    """QP affine constraint row in the form lower <= A z <= upper."""

    coefficients: np.ndarray
    lower: float = -math.inf
    upper: float = math.inf


@dataclass(frozen=True)
class LinearizedBarrier:
    """Linearized one-step discrete barrier information."""

    constraint: AffineConstraint | None
    current_value: float
    nominal_value: float
    target_value: float
    gradient: np.ndarray


def step_with_discrete_dynamics(
    *,
    state: State,
    action: Action,
    dynamics: VehicleDynamics,
    dt: float,
) -> State:
    """Use the same one-step discrete dynamics as the simulator."""

    return dynamics.step(state=state, action=action, dt=dt)


def rollout_with_constant_action(
    *,
    state: State,
    action: Action,
    dynamics: VehicleDynamics,
    dt: float,
    steps: int,
) -> State:
    """Roll out the exact discrete dynamics for a short fixed-action horizon."""

    trajectory = rollout_trajectory_with_constant_action(
        state=state,
        action=action,
        dynamics=dynamics,
        dt=dt,
        steps=steps,
    )
    return trajectory[-1]


def rollout_trajectory_with_constant_action(
    *,
    state: State,
    action: Action,
    dynamics: VehicleDynamics,
    dt: float,
    steps: int,
) -> tuple[State, ...]:
    """Roll out the exact discrete dynamics and keep every intermediate step."""

    rolled_state = state
    history: list[State] = []
    for _ in range(max(int(steps), 1)):
        rolled_state = step_with_discrete_dynamics(
            state=rolled_state,
            action=action,
            dynamics=dynamics,
            dt=dt,
        )
        history.append(rolled_state)
    return tuple(history)


def obstacle_barrier(
    *,
    state: State,
    obstacle: State | ObstacleState,
    vehicle_length: float,
    vehicle_width: float,
    obstacle_length: float,
    obstacle_width: float,
    safe_distance: float,
) -> float:
    """Vehicle-obstacle safety function h_obs(x) = clearance - d_safe."""

    return (
        box_clearance(
            state,
            vehicle_length,
            vehicle_width,
            obstacle,
            obstacle_length,
            obstacle_width,
        )
        - safe_distance
    )


def boundary_barrier(
    *,
    state: State,
    road: Road,
    vehicle_length: float,
    vehicle_width: float,
) -> float:
    """Road-boundary safety function based on the oriented vehicle box."""

    corners = oriented_box_corners(state, vehicle_length, vehicle_width)
    upper_bound = road.geometry.lane_center_y + road.geometry.half_width
    lower_bound = road.geometry.lane_center_y - road.geometry.half_width
    upper_margin = upper_bound - float(np.max(corners[:, 1]))
    lower_margin = float(np.min(corners[:, 1])) - lower_bound
    return min(upper_margin, lower_margin)


def _finite_difference_gradient(
    *,
    state: State,
    nominal_action: Action,
    barrier_fn: BarrierFunction,
    dynamics: VehicleDynamics,
    dt: float,
    bounds: InputBounds,
    lookahead_steps: int = 1,
) -> np.ndarray:
    """Numerically linearize h(f_d(x, u)) with respect to control."""

    nominal = np.asarray(nominal_action.to_array(), dtype=float)
    ranges = np.asarray(
        [
            max(bounds.accel_max - bounds.accel_min, 1e-6),
            max(bounds.steer_max - bounds.steer_min, 1e-6),
        ],
        dtype=float,
    )
    steps = np.maximum(1e-3, 0.01 * ranges)
    lower = np.asarray([bounds.accel_min, bounds.steer_min], dtype=float)
    upper = np.asarray([bounds.accel_max, bounds.steer_max], dtype=float)

    def evaluate(control: np.ndarray) -> float:
        candidate = Action(accel=float(control[0]), steer=float(control[1]))
        next_state = rollout_with_constant_action(
            state=state,
            action=candidate,
            dynamics=dynamics,
            dt=dt,
            steps=lookahead_steps,
        )
        return barrier_fn(next_state)

    gradient = np.zeros(2, dtype=float)
    for index in range(2):
        step = steps[index]
        forward = nominal.copy()
        backward = nominal.copy()
        forward[index] = min(nominal[index] + step, upper[index])
        backward[index] = max(nominal[index] - step, lower[index])
        if abs(forward[index] - backward[index]) <= 1e-9:
            gradient[index] = 0.0
            continue
        gradient[index] = (evaluate(forward) - evaluate(backward)) / (
            forward[index] - backward[index]
        )
    return gradient


def linearize_discrete_barrier(
    *,
    state: State,
    nominal_action: Action,
    current_value: float,
    next_barrier_fn: BarrierFunction,
    dynamics: VehicleDynamics,
    dt: float,
    bounds: InputBounds,
    kappa: float,
    lookahead_steps: int = 1,
    target_scale: float | None = None,
) -> LinearizedBarrier:
    """Build a discrete CBF constraint h(x_{k+1}) >= (1-kappa) h(x_k) - slack."""

    nominal_next_state = rollout_with_constant_action(
        state=state,
        action=nominal_action,
        dynamics=dynamics,
        dt=dt,
        steps=lookahead_steps,
    )
    nominal_value = next_barrier_fn(nominal_next_state)
    gradient = _finite_difference_gradient(
        state=state,
        nominal_action=nominal_action,
        barrier_fn=next_barrier_fn,
        dynamics=dynamics,
        dt=dt,
        bounds=bounds,
        lookahead_steps=lookahead_steps,
    )
    target_value = (1.0 - kappa) * current_value if target_scale is None else target_scale * current_value
    if not math.isfinite(current_value) or not math.isfinite(nominal_value) or not np.all(
        np.isfinite(gradient)
    ):
        return LinearizedBarrier(
            constraint=None,
            current_value=current_value,
            nominal_value=nominal_value,
            target_value=target_value,
            gradient=gradient,
        )

    nominal_control = np.asarray(nominal_action.to_array(), dtype=float)
    rhs = target_value - nominal_value + float(gradient @ nominal_control)
    return LinearizedBarrier(
        constraint=AffineConstraint(
            coefficients=np.asarray([gradient[0], gradient[1], 1.0], dtype=float),
            lower=rhs,
        ),
        current_value=current_value,
        nominal_value=nominal_value,
        target_value=target_value,
        gradient=gradient,
    )
