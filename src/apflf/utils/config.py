"""Configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from apflf.utils.types import (
    ControllerConfig,
    DecisionConfig,
    ExperimentConfig,
    InputBounds,
    ObstacleConfig,
    ProjectConfig,
    RoadConfig,
    RLDecisionConfig,
    RLThetaConfig,
    SafetyConfig,
    ScenarioConfig,
    SimulationConfig,
)

SUPPORTED_CONTROLLER_KINDS = {
    "formation_cruise",
    "apf",
    "st_apf",
    "apf_lf",
    "adaptive_apf",
    "dwa",
    "orca",
}
SUPPORTED_DECISION_KINDS = {"static", "fsm", "rl"}
SUPPORTED_OBSTACLE_MODELS = {"static", "constant_velocity"}
SUPPORTED_SAFETY_SOLVERS = {"osqp"}


def _require_mapping(raw: object, section_name: str) -> dict[str, Any]:
    """Assert that a config section is a mapping."""

    if not isinstance(raw, dict):
        raise ValueError(f"Config section `{section_name}` must be a mapping.")
    return raw


def _require_positive(value: float, field_name: str) -> float:
    """Require a finite positive scalar."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"`{field_name}` must be a finite positive scalar, got {value!r}.")
    return value


def _require_non_negative(value: float, field_name: str) -> float:
    """Require a finite non-negative scalar."""

    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"`{field_name}` must be a finite non-negative scalar, got {value!r}.")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config mappings."""

    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_raw_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML and resolve the `extends` inheritance chain."""

    seen = set() if seen is None else seen
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ValueError(f"Cyclic config inheritance detected: {resolved_path}")
    seen.add(resolved_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")

    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")
    extends_value = root.pop("extends", None)
    if extends_value is None:
        return root

    if isinstance(extends_value, str):
        parent_paths = [extends_value]
    elif isinstance(extends_value, list) and all(isinstance(item, str) for item in extends_value):
        parent_paths = extends_value
    else:
        raise ValueError("`extends` must be either a string or a list of strings.")

    merged: dict[str, Any] = {}
    for item in parent_paths:
        parent_path = (resolved_path.parent / item).resolve()
        merged = _deep_merge(merged, _load_raw_config(parent_path, seen=seen.copy()))
    return _deep_merge(merged, root)


def _load_bounds(raw: dict[str, Any]) -> InputBounds:
    """Parse action and speed bounds."""

    accel = raw.get("accel")
    steer_deg = raw.get("steer_deg")
    speed = raw.get("speed")
    if isinstance(accel, list) and len(accel) == 2:
        accel_min, accel_max = float(accel[0]), float(accel[1])
    elif "accel_min" in raw and "accel_max" in raw:
        accel_min = float(raw["accel_min"])
        accel_max = float(raw["accel_max"])
    else:
        raise ValueError(
            "`simulation.bounds` must provide either `accel: [min, max]` or "
            "`accel_min`/`accel_max`."
        )

    if isinstance(steer_deg, list) and len(steer_deg) == 2:
        steer_min = math.radians(float(steer_deg[0]))
        steer_max = math.radians(float(steer_deg[1]))
    elif "steer_min" in raw and "steer_max" in raw:
        steer_min = float(raw["steer_min"])
        steer_max = float(raw["steer_max"])
    else:
        raise ValueError(
            "`simulation.bounds` must provide either `steer_deg: [min, max]` or "
            "`steer_min`/`steer_max`."
        )

    if isinstance(speed, list) and len(speed) == 2:
        speed_min, speed_max = float(speed[0]), float(speed[1])
    elif "speed_min" in raw and "speed_max" in raw:
        speed_min = float(raw["speed_min"])
        speed_max = float(raw["speed_max"])
    else:
        raise ValueError(
            "`simulation.bounds` must provide either `speed: [min, max]` or "
            "`speed_min`/`speed_max`."
        )

    if accel_min >= accel_max:
        raise ValueError("Acceleration bounds must satisfy accel_min < accel_max.")
    if steer_min >= steer_max:
        raise ValueError("Steering bounds must satisfy steer_min < steer_max.")
    if speed_min > speed_max:
        raise ValueError("Speed bounds must satisfy speed_min <= speed_max.")

    return InputBounds(
        accel_min=accel_min,
        accel_max=accel_max,
        steer_min=steer_min,
        steer_max=steer_max,
        speed_min=speed_min,
        speed_max=speed_max,
    )


def _load_obstacles(raw: object) -> tuple[ObstacleConfig, ...]:
    """Parse scenario obstacle configs."""

    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("`scenario.obstacles` must be a list.")

    obstacles: list[ObstacleConfig] = []
    for index, item in enumerate(raw):
        obstacle_raw = _require_mapping(item, f"scenario.obstacles[{index}]")
        motion_model = str(obstacle_raw.get("motion_model", "static"))
        if motion_model not in SUPPORTED_OBSTACLE_MODELS:
            raise ValueError(
                f"`scenario.obstacles[{index}].motion_model` must be one of "
                f"{sorted(SUPPORTED_OBSTACLE_MODELS)}."
            )
        obstacles.append(
            ObstacleConfig(
                obstacle_id=str(obstacle_raw.get("obstacle_id", f"obs_{index:02d}")),
                motion_model=motion_model,
                x=float(obstacle_raw["x"]),
                y=float(obstacle_raw["y"]),
                yaw=float(obstacle_raw.get("yaw", 0.0)),
                speed=_require_non_negative(
                    float(obstacle_raw.get("speed", 0.0)),
                    f"scenario.obstacles[{index}].speed",
                ),
                length=_require_positive(
                    float(obstacle_raw.get("length", 4.5)),
                    f"scenario.obstacles[{index}].length",
                ),
                width=_require_positive(
                    float(obstacle_raw.get("width", 1.8)),
                    f"scenario.obstacles[{index}].width",
                ),
            )
        )
    return tuple(obstacles)


def _load_controller(raw: dict[str, Any]) -> ControllerConfig:
    """Parse controller config."""

    kind = str(raw.get("kind", "formation_cruise"))
    if kind not in SUPPORTED_CONTROLLER_KINDS:
        raise ValueError(f"`controller.kind` must be one of {sorted(SUPPORTED_CONTROLLER_KINDS)}.")

    repulsive_bounds = raw.get("repulsive_gain_bounds")
    road_bounds = raw.get("road_gain_bounds")
    if isinstance(repulsive_bounds, list) and len(repulsive_bounds) == 2:
        repulsive_gain_min = float(repulsive_bounds[0])
        repulsive_gain_max = float(repulsive_bounds[1])
    else:
        repulsive_gain_min = float(raw.get("repulsive_gain_min", 8.0))
        repulsive_gain_max = float(raw.get("repulsive_gain_max", 32.0))
    if isinstance(road_bounds, list) and len(road_bounds) == 2:
        road_gain_min = float(road_bounds[0])
        road_gain_max = float(road_bounds[1])
    else:
        road_gain_min = float(raw.get("road_gain_min", 3.0))
        road_gain_max = float(raw.get("road_gain_max", 15.0))

    if repulsive_gain_min > repulsive_gain_max:
        raise ValueError("`controller.repulsive_gain_bounds` must satisfy min <= max.")
    if road_gain_min > road_gain_max:
        raise ValueError("`controller.road_gain_bounds` must satisfy min <= max.")

    return ControllerConfig(
        kind=kind,
        vehicle_length=_require_positive(
            float(raw.get("vehicle_length", 4.8)),
            "controller.vehicle_length",
        ),
        vehicle_width=_require_positive(
            float(raw.get("vehicle_width", 1.9)),
            "controller.vehicle_width",
        ),
        speed_gain=_require_positive(float(raw.get("speed_gain", 0.8)), "controller.speed_gain"),
        gap_gain=_require_positive(float(raw.get("gap_gain", 0.35)), "controller.gap_gain"),
        lateral_gain=_require_positive(
            float(raw.get("lateral_gain", 0.22)),
            "controller.lateral_gain",
        ),
        heading_gain=_require_positive(
            float(raw.get("heading_gain", 0.65)),
            "controller.heading_gain",
        ),
        attraction_gain=_require_positive(
            float(raw.get("attraction_gain", 1.0)),
            "controller.attraction_gain",
        ),
        repulsive_gain=_require_positive(
            float(raw.get("repulsive_gain", 14.0)),
            "controller.repulsive_gain",
        ),
        road_gain=_require_positive(float(raw.get("road_gain", 8.0)), "controller.road_gain"),
        formation_gain=_require_non_negative(
            float(raw.get("formation_gain", 1.2)),
            "controller.formation_gain",
        ),
        consensus_gain=_require_non_negative(
            float(raw.get("consensus_gain", 0.25)),
            "controller.consensus_gain",
        ),
        obstacle_influence_distance=_require_positive(
            float(raw.get("obstacle_influence_distance", 15.0)),
            "controller.obstacle_influence_distance",
        ),
        vehicle_influence_distance=_require_positive(
            float(raw.get("vehicle_influence_distance", 10.0)),
            "controller.vehicle_influence_distance",
        ),
        road_influence_margin=_require_positive(
            float(raw.get("road_influence_margin", 1.2)),
            "controller.road_influence_margin",
        ),
        st_velocity_gain=_require_non_negative(
            float(raw.get("st_velocity_gain", 0.7)),
            "controller.st_velocity_gain",
        ),
        ttc_gain=_require_non_negative(float(raw.get("ttc_gain", 0.8)), "controller.ttc_gain"),
        ttc_threshold=_require_positive(
            float(raw.get("ttc_threshold", 4.0)),
            "controller.ttc_threshold",
        ),
        risk_distance_scale=_require_positive(
            float(raw.get("risk_distance_scale", 12.0)),
            "controller.risk_distance_scale",
        ),
        risk_speed_scale=_require_positive(
            float(raw.get("risk_speed_scale", 4.0)),
            "controller.risk_speed_scale",
        ),
        risk_ttc_threshold=_require_positive(
            float(raw.get("risk_ttc_threshold", 5.0)),
            "controller.risk_ttc_threshold",
        ),
        risk_sigmoid_slope=_require_positive(
            float(raw.get("risk_sigmoid_slope", 4.0)),
            "controller.risk_sigmoid_slope",
        ),
        risk_reference=float(raw.get("risk_reference", 0.45)),
        adaptive_alpha=_require_non_negative(
            float(raw.get("adaptive_alpha", 1.2)),
            "controller.adaptive_alpha",
        ),
        repulsive_gain_min=repulsive_gain_min,
        repulsive_gain_max=repulsive_gain_max,
        road_gain_min=road_gain_min,
        road_gain_max=road_gain_max,
        stagnation_speed_threshold=_require_non_negative(
            float(raw.get("stagnation_speed_threshold", 0.4)),
            "controller.stagnation_speed_threshold",
        ),
        stagnation_progress_threshold=_require_non_negative(
            float(raw.get("stagnation_progress_threshold", 0.03)),
            "controller.stagnation_progress_threshold",
        ),
        stagnation_force_threshold=_require_non_negative(
            float(raw.get("stagnation_force_threshold", 0.5)),
            "controller.stagnation_force_threshold",
        ),
        stagnation_steps=int(
            _require_positive(
                float(raw.get("stagnation_steps", 6)),
                "controller.stagnation_steps",
            )
        ),
        stagnation_cooldown_steps=int(
            _require_non_negative(
                float(raw.get("stagnation_cooldown_steps", 5)),
                "controller.stagnation_cooldown_steps",
            )
        ),
    )


def _load_safety(raw: dict[str, Any]) -> SafetyConfig:
    """Parse safety-filter config."""

    solver = str(raw.get("solver", "osqp")).lower()
    if solver not in SUPPORTED_SAFETY_SOLVERS:
        raise ValueError(f"`safety.solver` must be one of {sorted(SUPPORTED_SAFETY_SOLVERS)}.")

    def pick_value(*keys: str, default: float) -> float:
        for key in keys:
            if key in raw:
                return float(raw[key])
        return float(default)

    safe_distance = _require_non_negative(
        pick_value("d_safe", "safe_distance", default=0.5),
        "safety.d_safe",
    )
    barrier_decay = pick_value("kappa", "barrier_decay", default=0.3)
    if not math.isfinite(barrier_decay) or barrier_decay < 0.0:
        raise ValueError("`safety.kappa`/`safety.barrier_decay` must be finite and non-negative.")
    slack_penalty = _require_positive(
        pick_value("slack_weight", "slack_penalty", default=1200.0),
        "safety.slack_weight",
    )
    max_slack = _require_non_negative(
        pick_value("max_slack", default=2.0),
        "safety.max_slack",
    )
    road_boundary_margin = _require_non_negative(
        pick_value("road_boundary_margin", default=0.15),
        "safety.road_boundary_margin",
    )
    fallback_brake = _require_non_negative(
        pick_value("fallback_decel", "fallback_brake", default=2.5),
        "safety.fallback_decel",
    )
    fallback_steer_gain = _require_non_negative(
        pick_value("fallback_steer_gain", default=0.45),
        "safety.fallback_steer_gain",
    )

    return SafetyConfig(
        enabled=bool(raw.get("enabled", True)),
        solver=solver,
        safe_distance=safe_distance,
        barrier_decay=barrier_decay,
        slack_penalty=slack_penalty,
        max_slack=max_slack,
        road_boundary_margin=road_boundary_margin,
        fallback_brake=fallback_brake,
        fallback_steer_gain=fallback_steer_gain,
    )


def _load_decision(raw: dict[str, Any]) -> DecisionConfig:
    """Parse mode-decision config."""

    kind = str(raw.get("kind", "fsm")).lower()
    if kind not in SUPPORTED_DECISION_KINDS:
        raise ValueError(f"`decision.kind` must be one of {sorted(SUPPORTED_DECISION_KINDS)}.")

    hysteresis_steps = int(
        _require_positive(
            float(raw.get("hysteresis_steps", 3)),
            "decision.hysteresis_steps",
        )
    )
    risk_threshold_enter = float(raw.get("risk_threshold_enter", 0.55))
    risk_threshold_exit = float(raw.get("risk_threshold_exit", 0.30))
    for field_name, value in (
        ("decision.risk_threshold_enter", risk_threshold_enter),
        ("decision.risk_threshold_exit", risk_threshold_exit),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"`{field_name}` must be finite and lie in [0, 1].")
    if risk_threshold_exit > risk_threshold_enter:
        raise ValueError("`decision.risk_threshold_exit` must not exceed `decision.risk_threshold_enter`.")

    rl_raw = _require_mapping(raw.get("rl", {}), "decision.rl")
    theta_raw = _require_mapping(rl_raw.get("theta", {}), "decision.rl.theta")

    def _load_theta_tuple(
        mapping: dict[str, Any],
        key: str,
        default: tuple[float, float, float, float],
        *,
        non_negative: bool = True,
    ) -> tuple[float, float, float, float]:
        values = mapping.get(key, default)
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f"`decision.rl.theta.{key}` must be a length-4 sequence.")
        parsed = tuple(float(value) for value in values)
        for index, value in enumerate(parsed):
            if not math.isfinite(value):
                raise ValueError(f"`decision.rl.theta.{key}[{index}]` must be finite.")
            if non_negative and value < 0.0:
                raise ValueError(f"`decision.rl.theta.{key}[{index}]` must be non-negative.")
        return parsed

    theta_lower = _load_theta_tuple(theta_raw, "lower", (0.70, 0.70, 0.50, 0.0))
    theta_upper = _load_theta_tuple(theta_raw, "upper", (1.50, 1.50, 1.50, 0.60))
    theta_rate_limit = _load_theta_tuple(theta_raw, "rate_limit", (0.08, 0.08, 0.06, 0.05))
    theta_default = _load_theta_tuple(theta_raw, "default", (1.0, 1.0, 1.0, 0.0))
    for index, (lower, upper, default_value) in enumerate(
        zip(theta_lower, theta_upper, theta_default, strict=True)
    ):
        if lower > upper:
            raise ValueError(f"`decision.rl.theta.lower[{index}]` must not exceed upper.")
        if default_value < lower or default_value > upper:
            raise ValueError(f"`decision.rl.theta.default[{index}]` must lie within bounds.")

    tau_enter = float(rl_raw.get("tau_enter", rl_raw.get("confidence_threshold", 0.55)))
    tau_exit = float(rl_raw.get("tau_exit", max(0.0, tau_enter - 0.10)))

    rl_config = RLDecisionConfig(
        checkpoint_path=str(rl_raw.get("checkpoint_path", "")),
        deterministic_eval=bool(rl_raw.get("deterministic_eval", False)),
        confidence_threshold=tau_enter,
        tau_enter=tau_enter,
        tau_exit=tau_exit,
        ood_threshold=float(rl_raw.get("ood_threshold", 6.0)),
        observation_history=int(
            _require_positive(
                float(rl_raw.get("observation_history", 5)),
                "decision.rl.observation_history",
            )
        ),
        interaction_limit=int(
            _require_positive(
                float(rl_raw.get("interaction_limit", 8)),
                "decision.rl.interaction_limit",
            )
        ),
        theta=RLThetaConfig(
            lower=theta_lower,
            upper=theta_upper,
            rate_limit=theta_rate_limit,
            default=theta_default,
        ),
    )
    if not 0.0 <= rl_config.confidence_threshold <= 1.0:
        raise ValueError("`decision.rl.confidence_threshold` must lie in [0, 1].")
    if not 0.0 < rl_config.tau_enter <= 1.0:
        raise ValueError("`decision.rl.tau_enter` must lie in (0, 1].")
    if not 0.0 <= rl_config.tau_exit < rl_config.tau_enter:
        raise ValueError("`decision.rl.tau_exit` must satisfy 0 <= tau_exit < tau_enter.")
    if rl_config.ood_threshold < 0.0 or not math.isfinite(rl_config.ood_threshold):
        raise ValueError("`decision.rl.ood_threshold` must be finite and non-negative.")

    return DecisionConfig(
        kind=kind,
        default_mode=str(raw.get("default_mode", "topology=line|behavior=follow|gain=nominal")),
        hysteresis_steps=hysteresis_steps,
        risk_threshold_enter=risk_threshold_enter,
        risk_threshold_exit=risk_threshold_exit,
        clearance_threshold=_require_positive(
            float(raw.get("clearance_threshold", 8.0)),
            "decision.clearance_threshold",
        ),
        ttc_threshold=_require_positive(
            float(raw.get("ttc_threshold", 4.0)),
            "decision.ttc_threshold",
        ),
        boundary_margin_threshold=_require_positive(
            float(raw.get("boundary_margin_threshold", 0.75)),
            "decision.boundary_margin_threshold",
        ),
        lookahead_distance=_require_positive(
            float(raw.get("lookahead_distance", 18.0)),
            "decision.lookahead_distance",
        ),
        narrow_passage_margin=_require_non_negative(
            float(raw.get("narrow_passage_margin", 0.5)),
            "decision.narrow_passage_margin",
        ),
        stagnation_speed_threshold=_require_non_negative(
            float(raw.get("stagnation_speed_threshold", 0.45)),
            "decision.stagnation_speed_threshold",
        ),
        stagnation_progress_threshold=_require_non_negative(
            float(raw.get("stagnation_progress_threshold", 0.2)),
            "decision.stagnation_progress_threshold",
        ),
        stagnation_steps=int(
            _require_positive(
                float(raw.get("stagnation_steps", 6)),
                "decision.stagnation_steps",
            )
        ),
        recover_exit_steps=int(
            _require_positive(
                float(raw.get("recover_exit_steps", 4)),
                "decision.recover_exit_steps",
            )
        ),
        rl=rl_config,
    )


def load_config(path: Path) -> ProjectConfig:
    """Load YAML config and convert it into strongly typed dataclasses."""

    root = _load_raw_config(path)

    experiment_raw = _require_mapping(root.get("experiment"), "experiment")
    simulation_raw = _require_mapping(root.get("simulation"), "simulation")
    controller_raw = _require_mapping(root.get("controller"), "controller")
    safety_raw = _require_mapping(root.get("safety", {}), "safety")
    decision_raw = _require_mapping(root.get("decision"), "decision")
    scenario_raw = _require_mapping(root.get("scenario"), "scenario")
    road_raw = _require_mapping(scenario_raw.get("road"), "scenario.road")

    bounds = _load_bounds(_require_mapping(simulation_raw.get("bounds"), "simulation.bounds"))
    experiment = ExperimentConfig(
        name=str(experiment_raw.get("name", "default_experiment")),
        output_root=str(experiment_raw.get("output_root", "outputs")),
        save_traj=bool(experiment_raw.get("save_traj", True)),
    )
    simulation = SimulationConfig(
        dt=_require_positive(float(simulation_raw["dt"]), "simulation.dt"),
        steps=int(_require_positive(float(simulation_raw["steps"]), "simulation.steps")),
        wheelbase=_require_positive(float(simulation_raw["wheelbase"]), "simulation.wheelbase"),
        target_speed=_require_non_negative(
            float(simulation_raw["target_speed"]),
            "simulation.target_speed",
        ),
        bounds=bounds,
    )
    controller = _load_controller(controller_raw)
    safety = _load_safety(safety_raw)
    decision = _load_decision(decision_raw)
    road = RoadConfig(
        length=_require_positive(float(road_raw["length"]), "scenario.road.length"),
        lane_center_y=float(road_raw["lane_center_y"]),
        half_width=_require_positive(float(road_raw["half_width"]), "scenario.road.half_width"),
    )
    scenario = ScenarioConfig(
        vehicle_count=int(
            _require_positive(float(scenario_raw["vehicle_count"]), "scenario.vehicle_count")
        ),
        spacing=_require_positive(float(scenario_raw["spacing"]), "scenario.spacing"),
        spawn_jitter_std=_require_non_negative(
            float(scenario_raw["spawn_jitter_std"]),
            "scenario.spawn_jitter_std",
        ),
        initial_speed=_require_non_negative(
            float(scenario_raw["initial_speed"]),
            "scenario.initial_speed",
        ),
        goal_x=_require_positive(float(scenario_raw["goal_x"]), "scenario.goal_x"),
        goal_tolerance=_require_non_negative(
            float(scenario_raw["goal_tolerance"]),
            "scenario.goal_tolerance",
        ),
        road=road,
        obstacles=_load_obstacles(scenario_raw.get("obstacles")),
    )
    if scenario.initial_speed > simulation.bounds.speed_max:
        raise ValueError("`scenario.initial_speed` cannot exceed the configured speed upper bound.")

    return ProjectConfig(
        experiment=experiment,
        simulation=simulation,
        controller=controller,
        safety=safety,
        decision=decision,
        scenario=scenario,
    )


def compute_config_hash(config: ProjectConfig) -> str:
    """Compute a stable config hash for experiment tracking."""

    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
