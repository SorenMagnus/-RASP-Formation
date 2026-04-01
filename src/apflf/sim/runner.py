"""Batch experiment runner."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import yaml

from apflf.analysis.metrics import (
    build_action_history,
    build_decision_scalar_series,
    build_decision_source_series,
    build_decision_theta_history,
    build_obstacle_history,
    build_nominal_force_history,
    build_nominal_scalar_series,
    build_scalar_history,
    build_state_history,
    compute_run_summary,
)
from apflf.controllers.base import build_controller
from apflf.decision.mode_base import build_mode_decision
from apflf.env.dynamics import VehicleDynamics
from apflf.env.road import Road
from apflf.env.scenarios import ScenarioFactory
from apflf.safety.safety_filter import build_safety_filter
from apflf.sim.world import World
from apflf.utils.config import compute_config_hash
from apflf.utils.logging import get_logger
from apflf.utils.types import ObstacleState, ProjectConfig, Snapshot, State

LOGGER = get_logger(__name__)


def _read_git_commit_hash(repo_root: Path) -> str:
    """Read the current git commit hash if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nogit"
    return result.stdout.strip() or "nogit"


def _resolve_exp_id(config: ProjectConfig, exp_id: str | None) -> str:
    """Resolve a stable experiment output directory name."""

    if exp_id is not None and exp_id.strip():
        return exp_id.strip()
    return f"{config.experiment.name}_{compute_config_hash(config)[:8]}"


def _write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write summary rows to CSV."""

    if not rows:
        raise ValueError("summary.csv requires at least one experiment row.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_artifact(
    *,
    output_path: Path,
    initial_states: tuple[State, ...],
    initial_obstacles: tuple[ObstacleState, ...],
    snapshots: tuple[Snapshot, ...],
    goal_x: float,
    desired_offsets: tuple[tuple[float, float], ...],
    seed: int,
    config_hash: str,
    git_commit: str,
) -> None:
    """Persist a single rollout artifact bundle."""

    states = build_state_history(initial_states=initial_states, snapshots=snapshots)
    obstacles = build_obstacle_history(initial_obstacles=initial_obstacles, snapshots=snapshots)
    nominal_actions = build_action_history(snapshots=snapshots, field_name="nominal_actions")
    safe_actions = build_action_history(snapshots=snapshots, field_name="safe_actions")
    safety_corrections = build_scalar_history(snapshots, "safety_corrections")
    safety_slacks = build_scalar_history(snapshots, "safety_slacks")
    safety_fallbacks = build_scalar_history(snapshots, "safety_fallbacks", dtype=np.bool_)
    qp_solve_times = build_scalar_history(snapshots, "qp_solve_times")
    qp_iterations = build_scalar_history(snapshots, "qp_iterations", dtype=np.int64)
    leader_risk_scores = build_nominal_scalar_series(snapshots, "leader_risk_score")
    leader_base_reference_speeds = build_nominal_scalar_series(snapshots, "leader_base_reference_speed")
    leader_hazard_speed_caps = build_nominal_scalar_series(snapshots, "leader_hazard_speed_cap")
    leader_staggered_speed_caps = build_nominal_scalar_series(snapshots, "leader_staggered_speed_cap")
    leader_release_speed_caps = build_nominal_scalar_series(snapshots, "leader_release_speed_cap")
    leader_target_speeds = build_nominal_scalar_series(snapshots, "leader_target_speed")
    leader_staggered_activations = build_nominal_scalar_series(snapshots, "leader_staggered_activation")
    leader_edge_hold_activations = build_nominal_scalar_series(snapshots, "leader_edge_hold_activation")
    leader_force_attractive = build_nominal_force_history(snapshots, "attractive")
    leader_force_formation = build_nominal_force_history(snapshots, "formation")
    leader_force_consensus = build_nominal_force_history(snapshots, "consensus")
    leader_force_road = build_nominal_force_history(snapshots, "road")
    leader_force_obstacle = build_nominal_force_history(snapshots, "obstacle")
    leader_force_peer = build_nominal_force_history(snapshots, "peer")
    leader_force_behavior = build_nominal_force_history(snapshots, "behavior")
    leader_force_guidance = build_nominal_force_history(snapshots, "guidance")
    leader_force_escape = build_nominal_force_history(snapshots, "escape")
    leader_force_total = build_nominal_force_history(snapshots, "total")
    decision_sources = build_decision_source_series(snapshots)
    decision_confidences = build_decision_scalar_series(snapshots, "confidence")
    decision_confidence_raw = build_decision_scalar_series(snapshots, "confidence_raw")
    decision_effective_tau_enters = build_decision_scalar_series(snapshots, "effective_tau_enter")
    decision_effective_tau_exits = build_decision_scalar_series(snapshots, "effective_tau_exit")
    decision_thetas = build_decision_theta_history(snapshots, "theta")
    decision_theta_deltas = build_decision_theta_history(snapshots, "theta_delta")
    decision_rl_fallbacks = build_decision_scalar_series(snapshots, "rl_fallback", dtype=np.bool_)
    decision_gate_opens = build_decision_scalar_series(snapshots, "gate_open", dtype=np.bool_)
    decision_gate_reasons = np.asarray(
        [snapshot.decision_diagnostics.gate_reason for snapshot in snapshots],
        dtype="<U32",
    )
    decision_theta_clipped = build_decision_scalar_series(snapshots, "theta_clipped", dtype=np.bool_)
    decision_normalized_obs_max_abs = build_decision_scalar_series(
        snapshots,
        "normalized_obs_max_abs",
    )
    step_runtimes = np.asarray([snapshot.step_runtime for snapshot in snapshots], dtype=float)
    mode_runtimes = np.asarray([snapshot.mode_runtime for snapshot in snapshots], dtype=float)
    controller_runtimes = np.asarray([snapshot.controller_runtime for snapshot in snapshots], dtype=float)
    safety_runtimes = np.asarray([snapshot.safety_runtime for snapshot in snapshots], dtype=float)
    times = np.asarray([0.0, *[snapshot.time for snapshot in snapshots]], dtype=float)
    modes = np.asarray([snapshot.mode for snapshot in snapshots], dtype="<U96")
    obstacle_ids = np.asarray([obstacle.obstacle_id for obstacle in initial_obstacles], dtype="<U64")
    np.savez_compressed(
        output_path,
        states=states,
        obstacles=obstacles,
        obstacle_ids=obstacle_ids,
        nominal_actions=nominal_actions,
        safe_actions=safe_actions,
        safety_corrections=safety_corrections,
        safety_slacks=safety_slacks,
        safety_fallbacks=safety_fallbacks,
        qp_solve_times=qp_solve_times,
        qp_iterations=qp_iterations,
        leader_risk_scores=leader_risk_scores,
        leader_base_reference_speeds=leader_base_reference_speeds,
        leader_hazard_speed_caps=leader_hazard_speed_caps,
        leader_staggered_speed_caps=leader_staggered_speed_caps,
        leader_release_speed_caps=leader_release_speed_caps,
        leader_target_speeds=leader_target_speeds,
        leader_staggered_activations=leader_staggered_activations,
        leader_edge_hold_activations=leader_edge_hold_activations,
        leader_force_attractive=leader_force_attractive,
        leader_force_formation=leader_force_formation,
        leader_force_consensus=leader_force_consensus,
        leader_force_road=leader_force_road,
        leader_force_obstacle=leader_force_obstacle,
        leader_force_peer=leader_force_peer,
        leader_force_behavior=leader_force_behavior,
        leader_force_guidance=leader_force_guidance,
        leader_force_escape=leader_force_escape,
        leader_force_total=leader_force_total,
        decision_sources=decision_sources,
        decision_confidences=decision_confidences,
        decision_confidence_raw=decision_confidence_raw,
        decision_effective_tau_enters=decision_effective_tau_enters,
        decision_effective_tau_exits=decision_effective_tau_exits,
        decision_thetas=decision_thetas,
        decision_theta_deltas=decision_theta_deltas,
        decision_rl_fallbacks=decision_rl_fallbacks,
        decision_gate_opens=decision_gate_opens,
        decision_gate_reasons=decision_gate_reasons,
        decision_theta_clipped=decision_theta_clipped,
        decision_normalized_obs_max_abs=decision_normalized_obs_max_abs,
        step_runtimes=step_runtimes,
        mode_runtimes=mode_runtimes,
        controller_runtimes=controller_runtimes,
        safety_runtimes=safety_runtimes,
        times=times,
        modes=modes,
        desired_offsets=np.asarray(desired_offsets, dtype=float),
        goal_x=np.asarray([goal_x], dtype=float),
        seed=np.asarray([seed], dtype=int),
        config_hash=np.asarray([config_hash], dtype="<U128"),
        git_commit=np.asarray([git_commit], dtype="<U64"),
    )


def run_batch(config: ProjectConfig, seeds: list[int], exp_id: str | None = None) -> Path:
    """Run a batch of deterministic experiments."""

    if not seeds:
        raise ValueError("At least one seed is required.")

    repo_root = Path(__file__).resolve().parents[3]
    config_hash = compute_config_hash(config)
    git_commit = _read_git_commit_hash(repo_root)
    resolved_exp_id = _resolve_exp_id(config=config, exp_id=exp_id)
    output_dir = repo_root / config.experiment.output_root / resolved_exp_id
    traj_dir = output_dir / "traj"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    factory = ScenarioFactory(config=config)
    for seed in seeds:
        scenario = factory.build(seed=seed)
        road = Road(scenario.road)
        dynamics = VehicleDynamics(
            wheelbase=config.simulation.wheelbase,
            bounds=config.simulation.bounds,
        )
        controller = build_controller(
            config=config.controller,
            bounds=config.simulation.bounds,
            road=road,
            target_speed=config.simulation.target_speed,
            wheelbase=config.simulation.wheelbase,
            dt=config.simulation.dt,
        )
        mode_decision = build_mode_decision(
            config=config.decision,
            vehicle_length=config.controller.vehicle_length,
            vehicle_width=config.controller.vehicle_width,
            safe_distance=config.safety.safe_distance,
        )
        mode_decision.reset(seed)
        safety_filter = build_safety_filter(
            config=config.safety,
            bounds=config.simulation.bounds,
            road=road,
            wheelbase=config.simulation.wheelbase,
            vehicle_length=config.controller.vehicle_length,
            vehicle_width=config.controller.vehicle_width,
            dt=config.simulation.dt,
        )
        world = World(
            scenario=scenario,
            dynamics=dynamics,
            controller=controller,
            mode_decision=mode_decision,
            safety_filter=safety_filter,
            dt=config.simulation.dt,
        )
        initial_obstacles = world.obstacle_states
        snapshots = world.run(steps=config.simulation.steps)
        summary = compute_run_summary(
            road=road,
            goal_x=scenario.goal_x,
            goal_tolerance=config.scenario.goal_tolerance,
            desired_offsets=scenario.desired_offsets,
            initial_states=scenario.initial_states,
            initial_obstacles=initial_obstacles,
            snapshots=snapshots,
            vehicle_length=config.controller.vehicle_length,
            vehicle_width=config.controller.vehicle_width,
            bounds=config.simulation.bounds,
            dt=config.simulation.dt,
        )
        row: dict[str, object] = {
            "seed": seed,
            "config_hash": config_hash,
            "git_commit": git_commit,
            **summary,
        }
        rows.append(row)
        if config.experiment.save_traj:
            _write_run_artifact(
                output_path=traj_dir / f"seed_{seed:04d}.npz",
                initial_states=scenario.initial_states,
                initial_obstacles=initial_obstacles,
                snapshots=snapshots,
                goal_x=scenario.goal_x,
                desired_offsets=scenario.desired_offsets,
                seed=seed,
                config_hash=config_hash,
                git_commit=git_commit,
            )
        LOGGER.info(
            "finished seed=%s controller=%s leader_final_x=%.3f obstacle_count=%s fallback_events=%s",
            seed,
            config.controller.kind,
            float(row["leader_final_x"]),
            len(initial_obstacles),
            int(row["fallback_events"]),
        )

    _write_summary_csv(rows=rows, output_path=output_dir / "summary.csv")
    return output_dir
