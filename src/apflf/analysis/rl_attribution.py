"""Offline attribution helpers for the stage-1 RL supervisor."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from apflf.sim.replay import ReplayBundle
from apflf.utils.types import DEFAULT_THETA_VECTOR


def _linf(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def summarize_rl_seed(
    bundle: ReplayBundle,
    *,
    confidence_threshold: float | None = None,
    tau_enter: float | None = None,
    tau_exit: float | None = None,
    ood_threshold: float | None = None,
    theta_reference: tuple[float, float, float, float] = DEFAULT_THETA_VECTOR,
    theta_tol: float = 1e-6,
) -> dict[str, object]:
    """Summarize how the RL supervisor behaved on one replayable run."""

    if confidence_threshold is None:
        confidence_threshold = float(bundle.config.decision.rl.confidence_threshold)
    if tau_enter is None:
        tau_enter = float(getattr(bundle.config.decision.rl, "tau_enter", confidence_threshold))
    if tau_exit is None:
        tau_exit = float(getattr(bundle.config.decision.rl, "tau_exit", max(0.0, tau_enter - 0.10)))
    if ood_threshold is None:
        ood_threshold = float(bundle.config.decision.rl.ood_threshold)

    step_count = len(bundle.snapshots)
    if step_count == 0:
        return {
            "num_steps": 0,
            "rl_active_steps": 0,
            "rl_fallback_steps": 0,
            "theta_changed_steps": 0,
            "theta_clip_steps": 0,
            "safety_intervention_steps": 0,
            "safety_fallback_steps": 0,
            "qp_engagement_steps": 0,
            "gate_open_steps": 0,
            "effective_tau_enter_mean": float(tau_enter),
            "effective_tau_enter_min": float(tau_enter),
            "effective_tau_enter_max": float(tau_enter),
            "effective_tau_exit_mean": float(tau_exit),
            "effective_tau_exit_min": float(tau_exit),
            "effective_tau_exit_max": float(tau_exit),
            "dominant_bottleneck": "empty_run",
        }

    source_counts = Counter(snapshot.decision_diagnostics.source for snapshot in bundle.snapshots)
    theta_reference_array = np.asarray(theta_reference, dtype=float)
    theta_linf_from_default: list[float] = []
    theta_delta_linf: list[float] = []
    confidence_raw_values: list[float] = []
    effective_tau_enter_values: list[float] = []
    effective_tau_exit_values: list[float] = []
    low_confidence_fallback_steps = 0
    ood_fallback_steps = 0
    both_gate_fallback_steps = 0
    safety_intervention_steps = 0
    safety_fallback_steps = 0
    qp_engagement_steps = 0
    gate_open_steps = 0
    leader_risk_scores: list[float] = []
    gate_reason_counts: Counter[str] = Counter()
    previous_gate_open = False

    for snapshot in bundle.snapshots:
        diagnostics = snapshot.decision_diagnostics
        theta_linf_from_default.append(
            _linf(np.asarray(diagnostics.theta, dtype=float) - theta_reference_array)
        )
        theta_delta_linf.append(_linf(diagnostics.theta_delta))
        confidence_raw = float(getattr(diagnostics, "confidence_raw", diagnostics.confidence))
        confidence_raw_values.append(confidence_raw)
        effective_tau_enter = float(getattr(diagnostics, "effective_tau_enter", tau_enter))
        effective_tau_exit = float(getattr(diagnostics, "effective_tau_exit", tau_exit))
        effective_tau_enter_values.append(effective_tau_enter)
        effective_tau_exit_values.append(effective_tau_exit)
        leader_risk_scores.append(float(snapshot.nominal_diagnostics.leader_risk_score))
        gate_open = bool(getattr(diagnostics, "gate_open", False))
        gate_open_steps += int(gate_open)
        gate_reason = str(getattr(diagnostics, "gate_reason", "legacy"))
        gate_reason_counts[gate_reason] += 1

        if diagnostics.source == "rl_fallback":
            active_threshold = effective_tau_exit if previous_gate_open else effective_tau_enter
            low_confidence = confidence_raw < active_threshold
            ood = diagnostics.normalized_obs_max_abs > ood_threshold
            if low_confidence and ood:
                both_gate_fallback_steps += 1
            elif low_confidence:
                low_confidence_fallback_steps += 1
            elif ood:
                ood_fallback_steps += 1

        if any(float(value) > 1e-6 for value in snapshot.safety_corrections):
            safety_intervention_steps += 1
        if any(bool(value) for value in snapshot.safety_fallbacks):
            safety_fallback_steps += 1
        if any(float(value) > 0.0 for value in snapshot.qp_solve_times):
            qp_engagement_steps += 1
        previous_gate_open = gate_open

    rl_active_steps = int(source_counts.get("rl", 0))
    rl_fallback_steps = int(source_counts.get("rl_fallback", 0))
    theta_changed_steps = int(sum(value > theta_tol for value in theta_linf_from_default))
    theta_clip_steps = int(sum(snapshot.decision_diagnostics.theta_clipped for snapshot in bundle.snapshots))

    theta_change_ratio = theta_changed_steps / step_count
    rl_fallback_ratio = rl_fallback_steps / step_count
    safety_intervention_ratio = safety_intervention_steps / step_count
    safety_fallback_ratio = safety_fallback_steps / step_count
    qp_engagement_ratio = qp_engagement_steps / step_count

    dominant_bottleneck = "nominal_controller"
    if rl_fallback_ratio >= 0.50:
        dominant_bottleneck = "supervisor_gating"
    elif safety_intervention_ratio >= 0.50 or safety_fallback_ratio >= 0.10:
        dominant_bottleneck = "safety_engagement"
    elif theta_change_ratio <= 0.10:
        dominant_bottleneck = "weak_theta_impact"

    return {
        "num_steps": step_count,
        "rl_active_steps": rl_active_steps,
        "rl_fallback_steps": rl_fallback_steps,
        "rl_active_ratio": rl_active_steps / step_count,
        "rl_fallback_ratio": rl_fallback_ratio,
        "fallback_low_confidence_steps": low_confidence_fallback_steps,
        "fallback_ood_steps": ood_fallback_steps,
        "fallback_both_gate_steps": both_gate_fallback_steps,
        "accepted_enter_steps": int(gate_reason_counts.get("accepted_enter", 0)),
        "accepted_hold_steps": int(gate_reason_counts.get("accepted_hold", 0)),
        "fallback_enter_threshold_steps": int(gate_reason_counts.get("confidence_enter_threshold", 0)),
        "fallback_exit_threshold_steps": int(gate_reason_counts.get("confidence_exit_threshold", 0)),
        "ood_gate_steps": int(gate_reason_counts.get("ood_threshold", 0)),
        "gate_open_steps": gate_open_steps,
        "gate_open_ratio": gate_open_steps / step_count,
        "theta_changed_steps": theta_changed_steps,
        "theta_change_ratio": theta_change_ratio,
        "theta_clip_steps": theta_clip_steps,
        "theta_clip_ratio": theta_clip_steps / step_count,
        "confidence_raw_mean": float(np.mean(confidence_raw_values)),
        "confidence_raw_min": float(np.min(confidence_raw_values)),
        "effective_tau_enter_mean": float(np.mean(effective_tau_enter_values)),
        "effective_tau_enter_min": float(np.min(effective_tau_enter_values)),
        "effective_tau_enter_max": float(np.max(effective_tau_enter_values)),
        "effective_tau_exit_mean": float(np.mean(effective_tau_exit_values)),
        "effective_tau_exit_min": float(np.min(effective_tau_exit_values)),
        "effective_tau_exit_max": float(np.max(effective_tau_exit_values)),
        "theta_linf_from_default_mean": float(np.mean(theta_linf_from_default)),
        "theta_linf_from_default_max": float(np.max(theta_linf_from_default)),
        "theta_delta_linf_mean": float(np.mean(theta_delta_linf)),
        "theta_delta_linf_max": float(np.max(theta_delta_linf)),
        "safety_intervention_steps": safety_intervention_steps,
        "safety_intervention_ratio": safety_intervention_ratio,
        "safety_fallback_steps": safety_fallback_steps,
        "safety_fallback_ratio": safety_fallback_ratio,
        "qp_engagement_steps": qp_engagement_steps,
        "qp_engagement_ratio": qp_engagement_ratio,
        "leader_risk_score_mean": float(np.mean(leader_risk_scores)),
        "leader_risk_score_max": float(np.max(leader_risk_scores)),
        "confidence_threshold": float(confidence_threshold),
        "tau_enter": float(tau_enter),
        "tau_exit": float(tau_exit),
        "ood_threshold": float(ood_threshold),
        "dominant_bottleneck": dominant_bottleneck,
    }


def compare_to_reference_bundle(
    rl_bundle: ReplayBundle,
    reference_bundle: ReplayBundle,
) -> dict[str, object]:
    """Compare an RL run against a seed-aligned white-box reference replay."""

    shared_steps = min(len(rl_bundle.snapshots), len(reference_bundle.snapshots))
    if shared_steps == 0:
        return {
            "shared_steps": 0,
            "nominal_layer_changed": False,
            "leader_target_speed_delta_abs_mean": 0.0,
            "leader_force_total_delta_norm_mean": 0.0,
            "leader_nominal_accel_delta_abs_mean": 0.0,
            "leader_nominal_steer_delta_abs_mean": 0.0,
            "leader_safe_accel_delta_abs_mean": 0.0,
            "leader_safe_steer_delta_abs_mean": 0.0,
            "mode_mismatch_steps": 0,
        }

    leader_target_speed_deltas: list[float] = []
    leader_force_total_delta_norms: list[float] = []
    leader_nominal_accel_deltas: list[float] = []
    leader_nominal_steer_deltas: list[float] = []
    leader_safe_accel_deltas: list[float] = []
    leader_safe_steer_deltas: list[float] = []
    mode_mismatch_steps = 0

    for rl_snapshot, ref_snapshot in zip(
        rl_bundle.snapshots[:shared_steps],
        reference_bundle.snapshots[:shared_steps],
        strict=True,
    ):
        if rl_snapshot.mode != ref_snapshot.mode:
            mode_mismatch_steps += 1
        leader_target_speed_deltas.append(
            abs(
                rl_snapshot.nominal_diagnostics.leader_target_speed
                - ref_snapshot.nominal_diagnostics.leader_target_speed
            )
        )
        leader_force_total_delta_norms.append(
            float(
                np.linalg.norm(
                    np.asarray(rl_snapshot.nominal_diagnostics.leader_force.total, dtype=float)
                    - np.asarray(ref_snapshot.nominal_diagnostics.leader_force.total, dtype=float)
                )
            )
        )
        leader_nominal_accel_deltas.append(
            abs(rl_snapshot.nominal_actions[0].accel - ref_snapshot.nominal_actions[0].accel)
        )
        leader_nominal_steer_deltas.append(
            abs(rl_snapshot.nominal_actions[0].steer - ref_snapshot.nominal_actions[0].steer)
        )
        leader_safe_accel_deltas.append(
            abs(rl_snapshot.safe_actions[0].accel - ref_snapshot.safe_actions[0].accel)
        )
        leader_safe_steer_deltas.append(
            abs(rl_snapshot.safe_actions[0].steer - ref_snapshot.safe_actions[0].steer)
        )

    nominal_layer_changed = any(
        value > 1e-6
        for value in (
            float(np.mean(leader_target_speed_deltas)),
            float(np.mean(leader_force_total_delta_norms)),
            float(np.mean(leader_nominal_accel_deltas)),
            float(np.mean(leader_nominal_steer_deltas)),
        )
    )

    return {
        "shared_steps": shared_steps,
        "mode_mismatch_steps": mode_mismatch_steps,
        "mode_mismatch_ratio": mode_mismatch_steps / shared_steps,
        "nominal_layer_changed": nominal_layer_changed,
        "leader_target_speed_delta_abs_mean": float(np.mean(leader_target_speed_deltas)),
        "leader_target_speed_delta_abs_max": float(np.max(leader_target_speed_deltas)),
        "leader_force_total_delta_norm_mean": float(np.mean(leader_force_total_delta_norms)),
        "leader_force_total_delta_norm_max": float(np.max(leader_force_total_delta_norms)),
        "leader_nominal_accel_delta_abs_mean": float(np.mean(leader_nominal_accel_deltas)),
        "leader_nominal_steer_delta_abs_mean": float(np.mean(leader_nominal_steer_deltas)),
        "leader_safe_accel_delta_abs_mean": float(np.mean(leader_safe_accel_deltas)),
        "leader_safe_steer_delta_abs_mean": float(np.mean(leader_safe_steer_deltas)),
    }


def aggregate_seed_rows(seed_rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-seed attribution rows into one deterministic summary."""

    if not seed_rows:
        return {"num_seeds": 0, "dominant_bottleneck": "empty"}

    aggregated: dict[str, object] = {"num_seeds": len(seed_rows)}
    numeric_keys = [
        key
        for key in seed_rows[0]
        if isinstance(seed_rows[0][key], (int, float, bool)) and key != "seed"
    ]
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in seed_rows], dtype=float)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_max"] = float(np.max(values))
        aggregated[f"{key}_min"] = float(np.min(values))
    bottleneck_counts = Counter(str(row.get("dominant_bottleneck", "unknown")) for row in seed_rows)
    aggregated["dominant_bottleneck"] = bottleneck_counts.most_common(1)[0][0]
    aggregated["bottleneck_counts"] = dict(sorted(bottleneck_counts.items()))
    return aggregated
