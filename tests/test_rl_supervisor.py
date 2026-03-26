"""Tests for the stage-1 RL supervisor integration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from apflf.decision.fsm_mode import FSMModeDecision
from apflf.decision.rl_mode import RLSupervisor
from apflf.env.scenarios import ScenarioFactory
from apflf.rl.policy import ObservationNormalizer, PolicyBundle, PolicyInference
from apflf.sim.replay import load_replay_bundle, read_summary_row
from apflf.sim.runner import run_batch
from apflf.utils.config import load_config
from apflf.utils.types import ModeDecision


class DummySequencePolicy:
    """Deterministic test policy that emits a fixed theta/confidence sequence."""

    def __init__(self, outputs: list[tuple[tuple[float, float, float, float], float]]) -> None:
        self._outputs = outputs
        self._index = 0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._index = 0

    def infer(self, normalized_observation: np.ndarray, *, deterministic: bool) -> PolicyInference:
        del normalized_observation, deterministic
        theta, confidence = self._outputs[min(self._index, len(self._outputs) - 1)]
        self._index += 1
        return PolicyInference(theta=theta, confidence=confidence)


def _default_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def _initial_observation(config):
    scenario = ScenarioFactory(config=config).build(seed=0)
    return scenario, scenario.initial_states, scenario.obstacle_configs


def _make_fsm(config):
    return FSMModeDecision(
        config=config.decision,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        safe_distance=config.safety.safe_distance,
    )


def test_rl_supervisor_without_policy_matches_fsm_exactly() -> None:
    config = _default_config()
    scenario = ScenarioFactory(config=config).build(seed=0)
    from apflf.utils.types import Observation

    observation = Observation(
        step_index=0,
        time=0.0,
        states=scenario.initial_states,
        road=scenario.road,
        goal_x=scenario.goal_x,
        desired_offsets=scenario.desired_offsets,
        obstacles=(),
    )
    pure_fsm = _make_fsm(config)
    wrapped_fsm = _make_fsm(config)
    supervisor = RLSupervisor(
        fallback_fsm=wrapped_fsm,
        policy=None,
        normalizer=ObservationNormalizer.identity(dim=1),
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        ood_threshold=config.decision.rl.ood_threshold,
        deterministic_eval=True,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    expected_mode = pure_fsm.select(observation, 0)
    actual_mode = supervisor.select(observation, 0)

    assert actual_mode.mode == expected_mode.mode
    assert actual_mode.theta == pure_fsm.default_theta()
    assert supervisor.consume_step_diagnostics().source == "fsm"


def test_rl_supervisor_projects_rate_limits_and_fallbacks_on_low_confidence() -> None:
    config = _default_config()
    scenario = ScenarioFactory(config=config).build(seed=0)
    from apflf.utils.types import Observation

    observation = Observation(
        step_index=0,
        time=0.0,
        states=scenario.initial_states,
        road=scenario.road,
        goal_x=scenario.goal_x,
        desired_offsets=scenario.desired_offsets,
        obstacles=(),
    )
    supervisor = RLSupervisor(
        fallback_fsm=_make_fsm(config),
        policy=DummySequencePolicy(
            outputs=[
                ((2.0, 1.9, 1.8, 1.2), 0.95),
                ((0.1, 0.1, 0.1, 0.1), 0.40),
            ]
        ),
        normalizer=ObservationNormalizer(
            mean=np.zeros(108, dtype=float),
            std=np.full(108, 100.0, dtype=float),
        ),
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        ood_threshold=1e6,
        deterministic_eval=True,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    first = supervisor.select(observation, 0)
    first_diag = supervisor.consume_step_diagnostics()
    second = supervisor.select(observation, 1)
    second_diag = supervisor.consume_step_diagnostics()

    assert isinstance(first, ModeDecision)
    assert not hasattr(first, "accel")
    assert first.source == "rl"
    assert first_diag.theta_clipped is True
    for delta, limit in zip(first_diag.theta_delta, config.decision.rl.theta.rate_limit, strict=True):
        assert abs(delta) <= limit + 1e-9
    assert second.source == "rl_fallback"
    assert second.theta == config.decision.rl.theta.default
    assert second_diag.rl_fallback is True


def test_run_batch_with_rl_policy_is_deterministic_and_persists_rl_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _default_config()
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=str(tmp_path), save_traj=True),
        decision=replace(
            config.decision,
            kind="rl",
            rl=replace(
                config.decision.rl,
                checkpoint_path="dummy.ckpt",
                deterministic_eval=True,
            ),
        ),
    )

    def _bundle_factory(*, checkpoint_path: str, theta_config):
        del checkpoint_path, theta_config
        return PolicyBundle(
            policy=DummySequencePolicy(outputs=[((1.7, 1.8, 1.6, 0.9), 0.92)]),
            normalizer=ObservationNormalizer(
                mean=np.zeros(108, dtype=float),
                std=np.full(108, 100.0, dtype=float),
            ),
        )

    monkeypatch.setattr("apflf.decision.rl_mode.load_rl_policy_bundle", _bundle_factory)

    output_a = run_batch(config=config, seeds=[2], exp_id="rl_det_a")
    output_b = run_batch(config=config, seeds=[2], exp_id="rl_det_b")

    summary_a = read_summary_row(output_a, 2)
    summary_b = read_summary_row(output_b, 2)
    assert summary_a["leader_final_x"] == summary_b["leader_final_x"]
    assert summary_a["fallback_events"] == summary_b["fallback_events"]
    assert summary_a["rl_confidence_mean"] == summary_b["rl_confidence_mean"]
    assert summary_a["theta_clip_events"] > 0
    assert "rl_fallback_count" in summary_a
    assert "theta_delta_linf_mean" in summary_a
    assert "theta_delta_linf_max" in summary_a

    bundle_a = load_replay_bundle(output_a, 2)
    bundle_b = load_replay_bundle(output_b, 2)
    theta_a = [snapshot.decision_diagnostics.theta for snapshot in bundle_a.snapshots]
    theta_b = [snapshot.decision_diagnostics.theta for snapshot in bundle_b.snapshots]
    modes_a = [snapshot.mode for snapshot in bundle_a.snapshots]
    modes_b = [snapshot.mode for snapshot in bundle_b.snapshots]

    assert theta_a == theta_b
    assert modes_a == modes_b
    assert bundle_a.snapshots[0].decision_diagnostics.source == "rl"
    assert bundle_a.snapshots[0].decision_diagnostics.theta_clipped is True
