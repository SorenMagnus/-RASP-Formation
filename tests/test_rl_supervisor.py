"""Tests for the stage-1 RL supervisor integration."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apflf.decision.fsm_mode import FSMModeDecision
from apflf.decision.rl_mode import RLSupervisor
from apflf.env.scenarios import ScenarioFactory
from apflf.rl.env import SupervisorTrainingEnv
from apflf.rl.policy import ObservationNormalizer, PolicyBundle, PolicyInference, torch
from apflf.rl.ppo import PPOConfig, PPOTrainer
from apflf.sim.replay import load_replay_bundle, read_summary_row
from apflf.sim.runner import run_batch
from apflf.utils.config import load_config
from apflf.utils.types import ModeDecision, Observation, RLThetaConfig, State


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


class FakeTrainingEnv:
    """Small deterministic environment used to test PPO checkpoint/resume behavior."""

    observation_dim = 5

    def __init__(self, config=None) -> None:
        self.config = config
        self._seed = 0
        self._step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._seed = int(0 if seed is None else seed)
        self._step = 0
        return self._observation()

    def step(
        self,
        theta: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        reward_terms = {
            "progress": 0.15 * self._step + 0.01 * float(self._seed % 11),
            "formation": 0.30 * float(theta[0]),
            "intervene": -0.20 * float(theta[1]),
            "qp": -0.05 * float(self._step % 2),
            "fallback": -0.10 * float((self._seed + self._step) % 2),
            "slack": -0.01 * float(abs(theta[2])),
            "theta_rate": -0.02 * float(abs(theta[3])),
            "goal": 0.0,
            "collision": 0.0,
            "boundary": 0.0,
        }
        reward = float(sum(reward_terms.values()))
        self._step += 1
        terminated = self._step >= 4
        return self._observation(), float(reward), terminated, False, {
            "reward_terms": reward_terms,
            "qp_engagement_ratio_step": 0.5 * float(self._step % 2),
            "fallback_ratio_step": 0.25 * float((self._seed + self._step) % 2),
            "theta_delta_linf": float(np.max(np.abs(np.asarray(theta, dtype=float)))),
        }

    def _observation(self) -> np.ndarray:
        seed_term = float((self._seed % 17) / 17.0)
        step_term = float(self._step)
        return np.asarray(
            [
                seed_term,
                step_term,
                0.10 * step_term * step_term,
                np.sin(self._seed + self._step),
                np.cos(self._seed - self._step),
            ],
            dtype=float,
        )


def _default_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def _require_torch() -> None:
    if torch is None:
        pytest.skip("Torch optional dependency is required for PPO trainer checkpoint tests.")


def _make_fake_trainer(*, total_timesteps: int, steps_per_rollout: int, device: str = "cpu") -> PPOTrainer:
    _require_torch()
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    config = _default_config()
    return PPOTrainer(
        env=FakeTrainingEnv(config=config),
        theta_config=config.decision.rl.theta,
        config=PPOConfig(
            total_timesteps=total_timesteps,
            steps_per_rollout=steps_per_rollout,
            learning_rate=3e-4,
            update_epochs=2,
            minibatch_size=4,
            hidden_sizes=(32, 32),
        ),
        device=device,
    )


def _load_train_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_rl_supervisor.py"
    spec = importlib.util.spec_from_file_location("train_rl_supervisor_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_nested_close(lhs, rhs) -> None:
    if isinstance(lhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_nested_close(lhs[key], rhs[key])
        return
    if isinstance(lhs, (list, tuple)):
        assert len(lhs) == len(rhs)
        for left_item, right_item in zip(lhs, rhs, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    if torch is not None and isinstance(lhs, torch.Tensor):
        assert isinstance(rhs, torch.Tensor)
        assert torch.equal(lhs.cpu(), rhs.cpu())
        return
    if isinstance(lhs, np.ndarray):
        assert isinstance(rhs, np.ndarray)
        assert np.array_equal(lhs, rhs)
        return
    if isinstance(lhs, float):
        assert lhs == pytest.approx(rhs)
        return
    assert lhs == rhs


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


def test_supervisor_training_env_reward_uses_normalized_safety_terms() -> None:
    config = _default_config()
    env = SupervisorTrainingEnv(config=config)
    env.reset(seed=0)
    assert env._current_observation is not None
    current_observation = env._current_observation
    next_states = tuple(
        State(
            x=state.x + 1.0,
            y=state.y,
            yaw=state.yaw,
            speed=state.speed,
        )
        for state in current_observation.states
    )
    next_observation = Observation(
        step_index=current_observation.step_index + 1,
        time=current_observation.time + config.simulation.dt,
        states=next_states,
        road=current_observation.road,
        goal_x=current_observation.goal_x,
        desired_offsets=current_observation.desired_offsets,
        obstacles=current_observation.obstacles,
    )
    safety_safe = SimpleNamespace(
        correction_norms=(1e-7, 0.0, 0.0),
        slack_values=(0.0, 0.0, 0.0),
        fallback_flags=(False, False, False),
        qp_solve_times=(0.0, 0.0, 0.0),
    )
    safety_engaged = SimpleNamespace(
        correction_norms=(1e-4, 0.0, 0.0),
        slack_values=(0.2, 0.0, 0.0),
        fallback_flags=(True, False, False),
        qp_solve_times=(1.0, 0.0, 0.0),
    )

    safe_terms = env._reward_terms(
        current_observation=current_observation,
        next_observation=next_observation,
        safety_result=safety_safe,
        theta=config.decision.rl.theta.default,
    )
    engaged_terms = env._reward_terms(
        current_observation=current_observation,
        next_observation=next_observation,
        safety_result=safety_engaged,
        theta=(1.05, 1.0, 1.0, 0.0),
    )

    assert safe_terms["progress"] == pytest.approx(1.25)
    assert safe_terms["formation"] == pytest.approx(0.0, abs=1e-9)
    assert safe_terms["intervene"] == pytest.approx(0.0)
    assert safe_terms["qp"] == pytest.approx(0.0)
    assert safe_terms["fallback"] == pytest.approx(0.0)
    assert engaged_terms["progress"] == pytest.approx(safe_terms["progress"])
    assert engaged_terms["formation"] == pytest.approx(safe_terms["formation"], abs=1e-9)
    assert engaged_terms["intervene"] == pytest.approx(-0.10)
    assert engaged_terms["qp"] == pytest.approx(-0.20 / 3.0)
    assert engaged_terms["fallback"] == pytest.approx(-0.75 / 3.0)
    assert engaged_terms["slack"] == pytest.approx(-0.02)
    assert engaged_terms["theta_rate"] == pytest.approx(-0.001)
    assert sum(engaged_terms.values()) < sum(safe_terms.values())


def test_supervisor_training_env_reward_adds_terminal_terms(monkeypatch) -> None:
    config = _default_config()
    env = SupervisorTrainingEnv(config=config)
    env.reset(seed=0)
    assert env._current_observation is not None
    current_observation = env._current_observation
    leader_shift = config.scenario.goal_x - current_observation.states[0].x
    next_states = tuple(
        State(
            x=state.x + leader_shift,
            y=state.y,
            yaw=state.yaw,
            speed=state.speed,
        )
        for state in current_observation.states
    )
    next_observation = Observation(
        step_index=current_observation.step_index + 1,
        time=current_observation.time + config.simulation.dt,
        states=next_states,
        road=current_observation.road,
        goal_x=current_observation.goal_x,
        desired_offsets=current_observation.desired_offsets,
        obstacles=current_observation.obstacles,
    )
    monkeypatch.setattr(env, "_terminal_violations", lambda states, obstacles: (True, True))
    safety_result = SimpleNamespace(
        correction_norms=(0.0, 0.0, 0.0),
        slack_values=(0.0, 0.0, 0.0),
        fallback_flags=(False, False, False),
        qp_solve_times=(0.0, 0.0, 0.0),
    )

    reward_terms = env._reward_terms(
        current_observation=current_observation,
        next_observation=next_observation,
        safety_result=safety_result,
        theta=config.decision.rl.theta.default,
    )

    assert reward_terms["goal"] == pytest.approx(config.decision.rl.reward.goal_reward)
    assert reward_terms["collision"] == pytest.approx(-config.decision.rl.reward.collision_penalty)
    assert reward_terms["boundary"] == pytest.approx(-config.decision.rl.reward.boundary_penalty)


def test_torch_beta_policy_uses_variance_calibrated_confidence() -> None:
    _require_torch()
    from apflf.rl.policy import TorchBetaPolicy

    class UniformBetaNetwork:
        def distribution_and_value(self, observations):
            del observations
            alpha = torch.ones((1, 4), dtype=torch.float32)
            beta = torch.ones((1, 4), dtype=torch.float32)
            return torch.distributions.Beta(alpha, beta), torch.zeros(1, dtype=torch.float32)

    class ConcentratedBetaNetwork:
        def distribution_and_value(self, observations):
            del observations
            alpha = torch.full((1, 4), 9.0, dtype=torch.float32)
            beta = torch.full((1, 4), 9.0, dtype=torch.float32)
            return torch.distributions.Beta(alpha, beta), torch.zeros(1, dtype=torch.float32)

    theta_config = RLThetaConfig()
    lower = np.asarray(theta_config.lower, dtype=float)
    upper = np.asarray(theta_config.upper, dtype=float)
    expected_theta = tuple(float(value) for value in (lower + upper) * 0.5)

    uniform_policy = TorchBetaPolicy(network=UniformBetaNetwork(), theta_config=theta_config)
    concentrated_policy = TorchBetaPolicy(network=ConcentratedBetaNetwork(), theta_config=theta_config)

    uniform = uniform_policy.infer(np.zeros(8, dtype=float), deterministic=True)
    concentrated = concentrated_policy.infer(np.zeros(8, dtype=float), deterministic=True)

    assert uniform.theta == pytest.approx(expected_theta)
    assert uniform.confidence == pytest.approx(0.0)
    assert 0.0 <= concentrated.confidence <= 1.0
    assert concentrated.confidence > uniform.confidence


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
        tau_enter=config.decision.rl.tau_enter,
        tau_exit=config.decision.rl.tau_exit,
        tau_enter_start=config.decision.rl.tau_enter_start,
        tau_exit_start=config.decision.rl.tau_exit_start,
        gate_warmup_timesteps=config.decision.rl.gate_warmup_timesteps,
        ood_threshold=config.decision.rl.ood_threshold,
        deterministic_eval=True,
        checkpoint_timesteps_done=None,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    expected_mode = pure_fsm.select(observation, 0)
    actual_mode = supervisor.select(observation, 0)

    assert actual_mode.mode == expected_mode.mode
    assert actual_mode.theta == pure_fsm.default_theta()
    diagnostics = supervisor.consume_step_diagnostics()
    assert diagnostics.source == "fsm"
    assert diagnostics.gate_open is False
    assert diagnostics.gate_reason == "policy_missing"
    assert diagnostics.effective_tau_enter == pytest.approx(config.decision.rl.tau_enter)
    assert diagnostics.effective_tau_exit == pytest.approx(config.decision.rl.tau_exit)


def test_rl_supervisor_applies_hysteresis_gate_and_exact_fallback() -> None:
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
                ((2.0, 1.9, 1.8, 1.2), 0.60),
                ((1.5, 1.4, 1.3, 0.2), 0.50),
                ((0.1, 0.1, 0.1, 0.1), 0.44),
            ]
        ),
        normalizer=ObservationNormalizer(
            mean=np.zeros(108, dtype=float),
            std=np.full(108, 100.0, dtype=float),
        ),
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        tau_enter=0.55,
        tau_exit=0.45,
        tau_enter_start=config.decision.rl.tau_enter_start,
        tau_exit_start=config.decision.rl.tau_exit_start,
        gate_warmup_timesteps=config.decision.rl.gate_warmup_timesteps,
        ood_threshold=1e6,
        deterministic_eval=True,
        checkpoint_timesteps_done=None,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    first = supervisor.select(observation, 0)
    first_diag = supervisor.consume_step_diagnostics()
    second = supervisor.select(observation, 1)
    second_diag = supervisor.consume_step_diagnostics()
    third = supervisor.select(observation, 2)
    third_diag = supervisor.consume_step_diagnostics()

    assert isinstance(first, ModeDecision)
    assert not hasattr(first, "accel")
    assert first.source == "rl"
    assert first_diag.confidence_raw == pytest.approx(0.60)
    assert first_diag.gate_open is True
    assert first_diag.gate_reason == "accepted_enter"
    assert first_diag.effective_tau_enter == pytest.approx(0.55)
    assert first_diag.effective_tau_exit == pytest.approx(0.45)
    assert first_diag.theta_clipped is True
    for delta, limit in zip(first_diag.theta_delta, config.decision.rl.theta.rate_limit, strict=True):
        assert abs(delta) <= limit + 1e-9
    assert second.source == "rl"
    assert second_diag.gate_open is True
    assert second_diag.gate_reason == "accepted_hold"
    assert second_diag.confidence_raw == pytest.approx(0.50)
    assert second_diag.effective_tau_enter == pytest.approx(0.55)
    assert second_diag.effective_tau_exit == pytest.approx(0.45)
    assert third.source == "rl_fallback"
    assert third.theta == config.decision.rl.theta.default
    assert third_diag.rl_fallback is True
    assert third_diag.gate_open is False
    assert third_diag.gate_reason == "confidence_exit_threshold"
    assert third_diag.effective_tau_enter == pytest.approx(0.55)
    assert third_diag.effective_tau_exit == pytest.approx(0.45)


def test_rl_supervisor_ood_forces_fallback_even_with_high_confidence() -> None:
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
        policy=DummySequencePolicy(outputs=[((1.2, 1.2, 1.2, 0.1), 0.99)]),
        normalizer=ObservationNormalizer(
            mean=np.zeros(108, dtype=float),
            std=np.full(108, 1e-3, dtype=float),
        ),
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        tau_enter=0.55,
        tau_exit=0.45,
        tau_enter_start=config.decision.rl.tau_enter_start,
        tau_exit_start=config.decision.rl.tau_exit_start,
        gate_warmup_timesteps=config.decision.rl.gate_warmup_timesteps,
        ood_threshold=6.0,
        deterministic_eval=True,
        checkpoint_timesteps_done=None,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    decision = supervisor.select(observation, 0)
    diagnostics = supervisor.consume_step_diagnostics()

    assert decision.source == "rl_fallback"
    assert decision.theta == config.decision.rl.theta.default
    assert diagnostics.rl_fallback is True
    assert diagnostics.gate_open is False
    assert diagnostics.gate_reason == "ood_threshold"
    assert diagnostics.confidence_raw == pytest.approx(0.99)
    assert diagnostics.normalized_obs_max_abs > supervisor.ood_threshold
    assert diagnostics.effective_tau_enter == pytest.approx(0.55)
    assert diagnostics.effective_tau_exit == pytest.approx(0.45)


def test_rl_supervisor_training_thresholds_anneal_monotonically() -> None:
    thresholds = [
        RLSupervisor._annealed_threshold(
            start=0.25,
            final=0.55,
            timesteps_done=timesteps_done,
            warmup_timesteps=20_000,
        )
        for timesteps_done in (0, 5_000, 10_000, 20_000, 40_000)
    ]
    exit_thresholds = [
        RLSupervisor._annealed_threshold(
            start=0.15,
            final=0.45,
            timesteps_done=timesteps_done,
            warmup_timesteps=20_000,
        )
        for timesteps_done in (0, 5_000, 10_000, 20_000, 40_000)
    ]

    assert thresholds[0] == pytest.approx(0.25)
    assert exit_thresholds[0] == pytest.approx(0.15)
    assert thresholds[-2] == pytest.approx(0.55)
    assert thresholds[-1] == pytest.approx(0.55)
    assert exit_thresholds[-2] == pytest.approx(0.45)
    assert exit_thresholds[-1] == pytest.approx(0.45)
    assert thresholds == sorted(thresholds)
    assert exit_thresholds == sorted(exit_thresholds)
    for tau_enter, tau_exit in zip(thresholds, exit_thresholds, strict=True):
        assert 0.0 <= tau_exit < tau_enter <= 1.0


def test_rl_supervisor_uses_annealed_thresholds_only_for_non_deterministic_runtime() -> None:
    config = _default_config()
    scenario = ScenarioFactory(config=config).build(seed=0)
    observation = Observation(
        step_index=0,
        time=0.0,
        states=scenario.initial_states,
        road=scenario.road,
        goal_x=scenario.goal_x,
        desired_offsets=scenario.desired_offsets,
        obstacles=(),
    )
    normalizer = ObservationNormalizer(
        mean=np.zeros(108, dtype=float),
        std=np.full(108, 100.0, dtype=float),
    )
    policy = DummySequencePolicy(outputs=[((1.1, 1.1, 1.1, 0.1), 0.30)])

    training_runtime = RLSupervisor(
        fallback_fsm=_make_fsm(config),
        policy=policy,
        normalizer=normalizer,
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        tau_enter=0.55,
        tau_exit=0.45,
        tau_enter_start=0.25,
        tau_exit_start=0.15,
        gate_warmup_timesteps=20_000,
        ood_threshold=1e6,
        deterministic_eval=False,
        checkpoint_timesteps_done=0,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )
    deterministic_runtime = RLSupervisor(
        fallback_fsm=_make_fsm(config),
        policy=DummySequencePolicy(outputs=[((1.1, 1.1, 1.1, 0.1), 0.30)]),
        normalizer=normalizer,
        constraints=config.decision.rl.theta,
        confidence_threshold=config.decision.rl.confidence_threshold,
        tau_enter=0.55,
        tau_exit=0.45,
        tau_enter_start=0.25,
        tau_exit_start=0.15,
        gate_warmup_timesteps=20_000,
        ood_threshold=1e6,
        deterministic_eval=True,
        checkpoint_timesteps_done=0,
        vehicle_length=config.controller.vehicle_length,
        vehicle_width=config.controller.vehicle_width,
        observation_history=config.decision.rl.observation_history,
        interaction_limit=config.decision.rl.interaction_limit,
    )

    assert training_runtime._runtime_gate_thresholds() == pytest.approx((0.25, 0.15))
    assert deterministic_runtime._runtime_gate_thresholds() == pytest.approx((0.55, 0.45))

    training_decision = training_runtime.select(observation, 0)
    training_diag = training_runtime.consume_step_diagnostics()
    deterministic_decision = deterministic_runtime.select(observation, 0)
    deterministic_diag = deterministic_runtime.consume_step_diagnostics()

    assert training_decision.source == "rl"
    assert training_diag.gate_open is True
    assert training_diag.gate_reason == "accepted_enter"
    assert training_diag.effective_tau_enter == pytest.approx(0.25)
    assert training_diag.effective_tau_exit == pytest.approx(0.15)
    assert deterministic_decision.source == "rl_fallback"
    assert deterministic_diag.gate_open is False
    assert deterministic_diag.gate_reason == "confidence_enter_threshold"
    assert deterministic_diag.effective_tau_enter == pytest.approx(0.55)
    assert deterministic_diag.effective_tau_exit == pytest.approx(0.45)


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
    assert bundle_a.snapshots[0].decision_diagnostics.confidence_raw == pytest.approx(
        bundle_a.snapshots[0].decision_diagnostics.confidence
    )
    assert bundle_a.snapshots[0].decision_diagnostics.gate_open is True
    assert bundle_a.snapshots[0].decision_diagnostics.gate_reason == "accepted_enter"
    assert bundle_a.snapshots[0].decision_diagnostics.theta_clipped is True
    assert bundle_a.snapshots[0].decision_diagnostics.effective_tau_enter == pytest.approx(0.55)
    assert bundle_a.snapshots[0].decision_diagnostics.effective_tau_exit == pytest.approx(0.45)


def test_run_batch_persists_annealed_gate_thresholds_for_training_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "o"
    config = _default_config()
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=str(output_root), save_traj=True),
        decision=replace(
            config.decision,
            kind="rl",
            rl=replace(
                config.decision.rl,
                checkpoint_path="dummy.ckpt",
                deterministic_eval=False,
            ),
        ),
    )

    def _bundle_factory(*, checkpoint_path: str, theta_config):
        del checkpoint_path, theta_config
        return PolicyBundle(
            policy=DummySequencePolicy(outputs=[((1.1, 1.1, 1.1, 0.1), 0.30)]),
            normalizer=ObservationNormalizer(
                mean=np.zeros(108, dtype=float),
                std=np.full(108, 100.0, dtype=float),
            ),
            checkpoint_timesteps_done=0,
        )

    monkeypatch.setattr("apflf.decision.rl_mode.load_rl_policy_bundle", _bundle_factory)

    output_dir = run_batch(config=config, seeds=[2], exp_id="tau")
    bundle = load_replay_bundle(output_dir, 2)
    diagnostics = bundle.snapshots[0].decision_diagnostics

    assert diagnostics.source == "rl"
    assert diagnostics.gate_open is True
    assert diagnostics.effective_tau_enter == pytest.approx(0.25)
    assert diagnostics.effective_tau_exit == pytest.approx(0.15)


def test_ppo_trainer_checkpoint_payload_contains_resume_state(tmp_path: Path) -> None:
    trainer = _make_fake_trainer(total_timesteps=4, steps_per_rollout=4)
    latest_path = tmp_path / "latest.pt"

    trainer.train(seed=7, periodic_checkpoint_path=latest_path)

    assert latest_path.exists()
    payload = torch.load(latest_path, map_location="cpu")
    required_keys = {
        "state_dict",
        "obs_dim",
        "action_dim",
        "hidden_sizes",
        "normalizer_mean",
        "normalizer_std",
        "optimizer_state_dict",
        "obs_stats_count",
        "obs_stats_mean",
        "obs_stats_m2",
        "timesteps_done",
        "rollout_seed_next",
        "initial_seed",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "trainer_config",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["timesteps_done"] == 4
    assert payload["rollout_seed_next"] == 8
    assert payload["initial_seed"] == 7


def test_ppo_trainer_logs_reward_diagnostics() -> None:
    trainer = _make_fake_trainer(total_timesteps=4, steps_per_rollout=4)

    logs = trainer.train(seed=7)

    assert len(logs) == 1
    rollout_log = logs[0]
    for key in (
        "reward_progress_mean",
        "reward_form_mean",
        "reward_intervene_mean",
        "reward_qp_mean",
        "reward_fallback_mean",
        "reward_slack_mean",
        "reward_theta_rate_mean",
        "qp_engagement_ratio_mean",
        "fallback_ratio_mean",
        "theta_delta_linf_mean",
    ):
        assert key in rollout_log
    assert rollout_log["reward_progress_mean"] > 0.0
    assert rollout_log["reward_intervene_mean"] <= 0.0
    assert rollout_log["qp_engagement_ratio_mean"] >= 0.0
    assert rollout_log["fallback_ratio_mean"] >= 0.0
    assert rollout_log["theta_delta_linf_mean"] > 0.0


def test_ppo_trainer_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    continuous = _make_fake_trainer(total_timesteps=12, steps_per_rollout=4)
    continuous.train(seed=11)

    latest_path = tmp_path / "latest.pt"
    staged = _make_fake_trainer(total_timesteps=4, steps_per_rollout=4)
    staged.train(seed=11, periodic_checkpoint_path=latest_path)

    resumed = _make_fake_trainer(total_timesteps=12, steps_per_rollout=4)
    resumed.train(seed=11, periodic_checkpoint_path=latest_path, resume_from=latest_path)

    assert resumed.timesteps_done == continuous.timesteps_done == 12
    assert resumed.rollout_seed_next == continuous.rollout_seed_next
    _assert_nested_close(continuous.network.state_dict(), resumed.network.state_dict())
    _assert_nested_close(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert resumed.obs_stats.count == pytest.approx(continuous.obs_stats.count)
    assert np.array_equal(resumed.obs_stats.mean, continuous.obs_stats.mean)
    assert np.array_equal(resumed.obs_stats.m2, continuous.obs_stats.m2)
    assert resumed.logs == continuous.logs


def test_train_script_cli_writes_main_and_latest_without_resume(tmp_path: Path, monkeypatch, capsys) -> None:
    _require_torch()
    train_script = _load_train_script_module()
    config = _default_config()
    output_path = tmp_path / "main.pt"

    monkeypatch.setattr(train_script, "load_config", lambda _: config)
    monkeypatch.setattr(train_script, "SupervisorTrainingEnv", FakeTrainingEnv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rl_supervisor.py",
            "--config",
            "unused.yaml",
            "--seed",
            "5",
            "--total-timesteps",
            "4",
            "--steps-per-rollout",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
    )

    assert train_script.main() == 0
    assert output_path.exists()
    assert output_path.with_name("latest.pt").exists()
    captured = capsys.readouterr()
    assert "[ppo] start" in captured.out
    assert "[ppo] rollout_start" in captured.out
    assert "[ppo] rollout_done" in captured.out
    assert "timesteps_done=4/4" in captured.out


def test_train_script_cli_resume_continues_timesteps(tmp_path: Path, monkeypatch, capsys) -> None:
    _require_torch()
    train_script = _load_train_script_module()
    config = _default_config()
    output_path = tmp_path / "main.pt"
    latest_path = output_path.with_name("latest.pt")

    monkeypatch.setattr(train_script, "load_config", lambda _: config)
    monkeypatch.setattr(train_script, "SupervisorTrainingEnv", FakeTrainingEnv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rl_supervisor.py",
            "--config",
            "unused.yaml",
            "--seed",
            "9",
            "--total-timesteps",
            "4",
            "--steps-per-rollout",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
    )
    assert train_script.main() == 0
    capsys.readouterr()

    first_payload = torch.load(latest_path, map_location="cpu")
    first_mtime = latest_path.stat().st_mtime

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rl_supervisor.py",
            "--config",
            "unused.yaml",
            "--seed",
            "9",
            "--total-timesteps",
            "8",
            "--steps-per-rollout",
            "4",
            "--device",
            "cpu",
            "--resume-from",
            str(latest_path),
            "--output",
            str(output_path),
        ],
    )
    assert train_script.main() == 0

    captured = capsys.readouterr()
    second_payload = torch.load(latest_path, map_location="cpu")
    assert second_payload["timesteps_done"] == 8
    assert second_payload["timesteps_done"] > first_payload["timesteps_done"]
    assert latest_path.stat().st_mtime >= first_mtime
    assert "[ppo] resume" in captured.out
    assert "[ppo] rollout_done" in captured.out
    assert "timesteps_done=8/8" in captured.out
