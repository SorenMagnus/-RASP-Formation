"""Phase D CBF-QP safety filter tests."""

from __future__ import annotations

import numpy as np

from apflf.env.road import Road
from apflf.safety.cbf import (
    boundary_barrier,
    obstacle_barrier,
    rollout_trajectory_with_constant_action,
    step_with_discrete_dynamics,
)
from apflf.safety.safety_filter import CBFQPSafetyFilter
from apflf.utils.types import (
    Action,
    InputBounds,
    Observation,
    ObstacleState,
    RoadGeometry,
    SafetyConfig,
    State,
)


def _make_filter(*, dt: float = 0.2) -> tuple[CBFQPSafetyFilter, Road]:
    road = Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))
    filter_instance = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=0.3,
            slack_penalty=1500.0,
            max_slack=0.5,
            road_boundary_margin=0.0,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.5,
            steer_max=0.5,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=dt,
    )
    return filter_instance, road


def test_safe_control_stays_close_to_nominal_when_nominal_is_already_safe() -> None:
    """If the nominal action is already safe, the filter should barely intervene."""

    safety_filter, road = _make_filter()
    state = State(x=0.0, y=0.0, yaw=0.0, speed=4.0)
    nominal_action = Action(accel=0.2, steer=0.05)
    observation = Observation(
        step_index=0,
        time=0.0,
        states=(state,),
        road=road.geometry,
        goal_x=100.0,
        desired_offsets=((0.0, 0.0),),
        obstacles=(),
    )

    result = safety_filter.filter((nominal_action,), observation)

    safe_action = result.safe_actions[0]
    assert abs(safe_action.accel - nominal_action.accel) <= 1e-4
    assert abs(safe_action.steer - nominal_action.steer) <= 1e-4
    assert result.correction_norms[0] <= 1e-4
    assert result.fallback_flags[0] is False


def test_safe_action_remains_safe_after_one_exact_dynamics_step() -> None:
    """The corrected action should remain safe under the exact simulator step."""

    safety_filter, road = _make_filter(dt=0.15)
    state = State(x=0.0, y=2.15, yaw=0.12, speed=5.0)
    obstacle = ObstacleState(
        obstacle_id="obs_safe",
        x=8.0,
        y=-1.8,
        yaw=0.0,
        speed=0.0,
        length=4.5,
        width=1.8,
    )
    nominal_action = Action(accel=0.3, steer=0.35)
    observation = Observation(
        step_index=0,
        time=0.0,
        states=(state,),
        road=road.geometry,
        goal_x=100.0,
        desired_offsets=((0.0, 0.0),),
        obstacles=(obstacle,),
    )

    result = safety_filter.filter((nominal_action,), observation)
    next_state = step_with_discrete_dynamics(
        state=state,
        action=result.safe_actions[0],
        dynamics=safety_filter.dynamics,
        dt=safety_filter.dt,
    )
    h_bnd = boundary_barrier(
        state=next_state,
        road=road,
        vehicle_length=safety_filter.vehicle_length,
        vehicle_width=safety_filter.vehicle_width,
    )
    h_obs = obstacle_barrier(
        state=next_state,
        obstacle=obstacle,
        vehicle_length=safety_filter.vehicle_length,
        vehicle_width=safety_filter.vehicle_width,
        obstacle_length=obstacle.length,
        obstacle_width=obstacle.width,
        safe_distance=safety_filter.config.safe_distance,
    )

    assert h_bnd >= -1e-5
    assert h_obs >= -1e-5


def test_fallback_is_triggered_for_uncontrollable_case() -> None:
    """If the barrier is uncontrollable in one step, the filter should enter fallback."""

    safety_filter, road = _make_filter()
    state = State(x=0.0, y=3.9, yaw=0.0, speed=0.0)
    nominal_action = Action(accel=0.0, steer=0.0)
    observation = Observation(
        step_index=5,
        time=1.0,
        states=(state,),
        road=road.geometry,
        goal_x=100.0,
        desired_offsets=((0.0, 0.0),),
        obstacles=(),
    )

    result = safety_filter.filter((nominal_action,), observation)

    assert result.fallback_flags[0] is True
    assert result.slack_values[0] == 0.0
    assert result.safe_actions[0].accel <= nominal_action.accel
    assert np.sign(result.safe_actions[0].steer) <= 0.0


def test_preview_rollout_allows_short_steer_then_yaw_damping() -> None:
    """Preview rollout should not treat a brief evasive steer as a constant full-horizon turn."""

    safety_filter, road = _make_filter()
    state = State(
        x=22.2549371,
        y=-0.0051624,
        yaw=-0.0011466,
        speed=7.61649803,
    )
    evasive_action = Action(accel=-2.5, steer=0.218)
    observation = Observation(
        step_index=32,
        time=6.4,
        states=(state,),
        road=RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5),
        goal_x=105.0,
        desired_offsets=((0.0, 0.0),),
        obstacles=(
            ObstacleState("passage_block_1", 44.0, -0.95, 0.0, 0.0, 6.0, 2.5),
            ObstacleState("passage_block_2", 52.0, -0.85, 0.0, 0.0, 6.0, 2.5),
        ),
    )

    preview_steps = safety_filter._resolve_preview_steps(state)
    preview_trajectory = safety_filter._rollout_preview_trajectory(
        state=state,
        action=evasive_action,
        steps=preview_steps,
    )
    constant_trajectory = rollout_trajectory_with_constant_action(
        state=state,
        action=evasive_action,
        dynamics=safety_filter.dynamics,
        dt=safety_filter.dt,
        steps=preview_steps,
    )
    preview_margin = safety_filter._candidate_margin(
        state=state,
        action=evasive_action,
        observation=observation,
        ego_index=0,
        predicted_peer_trajectories=tuple(),
    )

    assert preview_margin > 0.0
    assert preview_trajectory[-1].y < 1.0
    assert constant_trajectory[-1].y > 3.5


def test_fallback_prefers_nominal_side_when_safe_margins_are_nearly_tied() -> None:
    """Regression: avoid flipping to the opposite corridor because of negligible margin differences."""

    road = Road(RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=25,
        time=2.5,
        states=(
            State(17.445980494761894, -0.007341894382046748, -0.00020014372446430784, 8.23824587583754),
            State(7.280183185250037, -0.10759476143040335, 0.04117105365395668, 7.991066221454726),
            State(-0.8378527867052915, -0.07379654481015398, 0.12092065127278806, 7.978446697380708),
        ),
        road=road.geometry,
        goal_x=105.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("passage_block_1", 44.0, -0.95, 0.0, 0.0, 6.0, 2.5),
            ObstacleState("passage_block_2", 52.0, -0.85, 0.0, 0.0, 6.0, 2.5),
        ),
    )
    nominal_actions = (
        Action(0.44940329932996775, 0.0010501273621260172),
        Action(1.3036492403974649, 0.03396906037421492),
        Action(1.3154406707741724, -0.041972856776082215),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[0],
        nominal_vector=np.asarray(nominal_actions[0].to_array(), dtype=float),
        observation=observation,
        ego_index=0,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.accel == 0.0
    assert fallback_action.steer > 0.0


def test_fallback_allows_safe_creep_for_stalled_vehicle() -> None:
    """Regression: a stalled vehicle should be allowed to inch forward when one-step safety permits it."""

    road = Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=110,
        time=11.0,
        states=(
            State(83.38451419, -1.15688573, 0.14064002, 7.1033661),
            State(70.7922537, -1.59067786, -0.02228915, 7.19081943),
            State(25.3248904327929, -1.6293104158734824, -0.021488856388161448, 0.0),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("front", 41.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )
    nominal_actions = (
        Action(2.0, 0.4363323129985824),
        Action(0.0, 0.0),
        Action(2.0, 0.4363323129985824),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[2],
        nominal_vector=np.asarray(nominal_actions[2].to_array(), dtype=float),
        observation=observation,
        ego_index=2,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.accel > 0.0


def test_fallback_allows_small_preview_deficit_for_one_step_safe_creep() -> None:
    """Regression: near-stop steering creep should not freeze when preview deficit is tiny but one-step safety still holds."""

    road = Road(RoadGeometry(length=175.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=80,
        time=8.0,
        states=(
            State(26.680513486535965, -0.9103407023092039, -0.09893089157088175, 0.09948580353732972),
            State(19.813199754973773, -1.7187161949723644, 0.2662272812203703, 0.41037254568747716),
            State(12.891924338118532, 0.7328051713938966, -0.589876430289614, 0.5779897520668271),
        ),
        road=road.geometry,
        goal_x=120.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("dense_static_left", 30.0, 2.0, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_static_right", 32.0, -2.1, 0.0, 0.0, 4.8, 2.0),
            ObstacleState("dense_cross_1", 57.99999999988568, -15.899999999999963, -1.5707963268, 2.8, 4.4, 1.9),
            ObstacleState("dense_cross_2", 67.99999999990202, 13.19999999999997, 1.5707963268, 2.4, 4.4, 1.9),
            ObstacleState("dense_slow", 101.99999999999997, 0.5, 0.0, 3.0, 4.8, 2.0),
        ),
    )
    nominal_actions = (
        Action(2.0, 0.4363323129985824),
        Action(-0.5449169235281253, 0.4363323129985824),
        Action(-0.5998932002505976, -0.4363323129985824),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[0],
        nominal_vector=np.asarray(nominal_actions[0].to_array(), dtype=float),
        observation=observation,
        ego_index=0,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.accel > 0.0
    assert fallback_action.steer > 0.0


def test_fallback_prefers_minimal_intervention_within_safe_candidate_set() -> None:
    """Regression: if multiple preview-safe candidates exist, fallback should not over-brake by default."""

    road = Road(RoadGeometry(length=120.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=70,
        time=7.0,
        states=(
            State(49.04555083879914, -1.1704433270241124, 0.1119938486693477, 7.098581165257745),
            State(36.94951855820324, -0.6337435010661127, -0.25203844818573673, 6.799479845249207),
            State(22.61302477796556, -1.3703699274821612, -0.0026368884945231486, 2.0971060901591088),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("front", 41.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )
    nominal_actions = (
        Action(0.08113506779380453, -0.30290037760139354),
        Action(0.7779321776327714, -0.4363323129985824),
        Action(2.0, 0.1580452233130896),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[2],
        nominal_vector=np.asarray(nominal_actions[2].to_array(), dtype=float),
        observation=observation,
        ego_index=2,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.accel >= 0.0
    assert fallback_action.steer > 0.0


def test_fallback_prefers_exact_step_safe_candidate_when_max_margin_hits_boundary() -> None:
    """Regression: boundary-saving fallback should prefer an exact-step-safe candidate over a boundary-violating one."""

    road = Road(RoadGeometry(length=140.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=114,
        time=11.4,
        states=(
            State(79.23893210598783, -1.7159631189879038, 0.08344981205469086, 5.018567094748913),
            State(71.56030641383898, 1.4879170524138754, 0.37171787293870097, 7.641182680691408),
            State(55.17280367197281, -1.590324064162335, -0.06624992039385516, 7.357512234182319),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("lm_upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lm_lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lm_front", 41.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )
    nominal_actions = (
        Action(-1.190899353112991, 0.31797134093550963),
        Action(-1.1862739116544528, 0.4363323129985824),
        Action(1.1960658683509657, 0.4363323129985824),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[1],
        nominal_vector=np.asarray(nominal_actions[1].to_array(), dtype=float),
        observation=observation,
        ego_index=1,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.steer <= -0.30
    assert (
        safety_filter._verify_safe_action(
            state=observation.states[1],
            safe_action=fallback_action,
            observation=observation,
            ego_index=1,
            predicted_peer_trajectories=predicted_peer_trajectories,
        )
        is None
    )


def test_fallback_uses_boundary_rescue_when_vehicle_is_drifting_outward() -> None:
    """Regression: when a vehicle is already drifting toward the boundary, fallback should bias inward early."""

    road = Road(RoadGeometry(length=140.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=110,
        time=11.0,
        states=(
            State(80.07272688030036, 1.1103517352481794, -0.3558680780199973, 7.556933875585728),
            State(69.87419238981259, 0.9250841474202327, 0.33409412825550167, 7.426868867436576),
            State(60.08056008315048, 2.077826372167588, 0.01905331492655243, 7.195707413573642),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("cross_up", 42.0, -21.6, -1.5707963268, 2.6, 4.4, 2.0),
            ObstacleState("cross_down", 52.0, 17.2, 1.5707963268, 2.2, 4.4, 2.0),
        ),
    )
    nominal_actions = (
        Action(0.9944528974329508, 0.05546574277431531),
        Action(1.4299785813805528, 0.4363323129985824),
        Action(2.0, -0.4363323129985824),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[1],
        nominal_vector=np.asarray(nominal_actions[1].to_array(), dtype=float),
        observation=observation,
        ego_index=1,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.accel > 0.0
    assert fallback_action.steer < 0.0
    assert (
        safety_filter._verify_safe_action(
            state=observation.states[1],
            safe_action=fallback_action,
            observation=observation,
            ego_index=1,
            predicted_peer_trajectories=predicted_peer_trajectories,
        )
        is None
    )


def test_fallback_prefers_boundary_recovery_when_vehicle_is_already_slightly_outside_lane() -> None:
    """Regression: if the oriented box is already slightly out of lane, fallback should pick an inward recovery action."""

    road = Road(RoadGeometry(length=150.0, lane_center_y=0.0, half_width=3.5))
    safety_filter = CBFQPSafetyFilter(
        config=SafetyConfig(
            enabled=True,
            solver="osqp",
            safe_distance=0.5,
            barrier_decay=3.0,
            slack_penalty=1200.0,
            max_slack=2.0,
            road_boundary_margin=0.15,
            fallback_brake=2.5,
            fallback_steer_gain=0.45,
        ),
        bounds=InputBounds(
            accel_min=-2.5,
            accel_max=2.0,
            steer_min=-0.4363323129985824,
            steer_max=0.4363323129985824,
            speed_min=0.0,
            speed_max=12.0,
        ),
        road=road,
        wheelbase=2.8,
        vehicle_length=4.8,
        vehicle_width=1.9,
        dt=0.1,
    )
    observation = Observation(
        step_index=186,
        time=18.6,
        states=(
            State(102.79996302406462, 1.2012854294144868, 0.2183833968956197, 0.0),
            State(121.1434534529706, -2.548431349823056, -0.004798463839100098, 6.015126463225631),
            State(111.96807049048188, -1.5274936162704773, -0.00390750873754353, 5.89745785922767),
        ),
        road=road.geometry,
        goal_x=95.0,
        desired_offsets=((0.0, 0.0), (-8.0, 0.0), (-16.0, 0.0)),
        obstacles=(
            ObstacleState("lm_upper", 27.0, 2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lm_lower", 30.5, -2.75, 0.0, 0.0, 4.5, 1.2),
            ObstacleState("lm_front", 41.5, 2.4, 0.0, 0.0, 4.2, 1.0),
        ),
    )
    nominal_actions = (
        Action(-2.5, 0.4363323129985824),
        Action(-2.5, 0.4363323129985824),
        Action(-2.5, 0.4363323129985824),
    )
    preview_steps = max(safety_filter._resolve_preview_steps(state) for state in observation.states)
    predicted_peer_trajectories = tuple(
        safety_filter._rollout_preview_trajectory(
            state=state,
            action=action,
            steps=preview_steps,
        )
        for state, action in zip(observation.states, nominal_actions, strict=True)
    )

    fallback_action = safety_filter._fallback_action(
        state=observation.states[1],
        nominal_vector=np.asarray(nominal_actions[1].to_array(), dtype=float),
        observation=observation,
        ego_index=1,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )
    hold_action = Action(accel=0.0, steer=0.0)
    _, _, fallback_final_boundary = safety_filter._candidate_margin_breakdown(
        state=observation.states[1],
        action=fallback_action,
        observation=observation,
        ego_index=1,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )
    _, _, hold_final_boundary = safety_filter._candidate_margin_breakdown(
        state=observation.states[1],
        action=hold_action,
        observation=observation,
        ego_index=1,
        predicted_peer_trajectories=predicted_peer_trajectories,
    )

    assert fallback_action.steer > 0.25
    assert fallback_action.accel <= 0.0
    assert fallback_final_boundary > 0.0
    assert fallback_final_boundary > hold_final_boundary + 0.1
