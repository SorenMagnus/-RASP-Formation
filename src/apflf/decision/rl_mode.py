"""RL supervisor wrapper that preserves exact FSM fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apflf.decision.mode_base import ModeDecisionModule
from apflf.rl.features import SupervisorObservationBuilder
from apflf.rl.policy import ObservationNormalizer, PolicyBundle, PolicyProtocol, load_policy_bundle
from apflf.utils.types import (
    DecisionDiagnostics,
    ModeDecision,
    RLThetaConfig,
    ZERO_THETA_VECTOR,
)


def load_rl_policy_bundle(
    *,
    checkpoint_path: str,
    theta_config: RLThetaConfig,
) -> PolicyBundle:
    """Load an RL policy bundle from a checkpoint path when available."""

    return load_policy_bundle(checkpoint_path=checkpoint_path, theta_config=theta_config)


@dataclass(frozen=True)
class ThetaProjectionResult:
    theta: tuple[float, float, float, float]
    clipped: bool


class RLSupervisor(ModeDecisionModule):
    """High-level theta supervisor with exact white-box FSM fallback."""

    def __init__(
        self,
        *,
        fallback_fsm: ModeDecisionModule,
        policy: PolicyProtocol | None,
        normalizer: ObservationNormalizer,
        constraints: RLThetaConfig,
        confidence_threshold: float,
        tau_enter: float,
        tau_exit: float,
        tau_enter_start: float,
        tau_exit_start: float,
        gate_warmup_timesteps: int,
        ood_threshold: float,
        deterministic_eval: bool,
        checkpoint_timesteps_done: int | None,
        vehicle_length: float,
        vehicle_width: float,
        observation_history: int,
        interaction_limit: int,
    ) -> None:
        self.fallback_fsm = fallback_fsm
        self.policy = policy
        self.normalizer = normalizer
        self.constraints = constraints
        self.confidence_threshold = float(confidence_threshold)
        self.tau_enter = float(tau_enter)
        self.tau_exit = float(tau_exit)
        self.tau_enter_start = float(tau_enter_start)
        self.tau_exit_start = float(tau_exit_start)
        self.gate_warmup_timesteps = int(gate_warmup_timesteps)
        self.ood_threshold = float(ood_threshold)
        self.deterministic_eval = bool(deterministic_eval)
        self.checkpoint_timesteps_done = (
            None if checkpoint_timesteps_done is None else int(checkpoint_timesteps_done)
        )
        if not 0.0 < self.tau_enter <= 1.0:
            raise ValueError("`tau_enter` must lie in (0, 1].")
        if not 0.0 <= self.tau_exit < self.tau_enter:
            raise ValueError("`tau_exit` must satisfy 0 <= tau_exit < tau_enter.")
        if not 0.0 < self.tau_enter_start <= 1.0:
            raise ValueError("`tau_enter_start` must lie in (0, 1].")
        if not 0.0 <= self.tau_exit_start < self.tau_enter_start:
            raise ValueError("`tau_exit_start` must satisfy 0 <= tau_exit_start < tau_enter_start.")
        if self.tau_enter_start > self.tau_enter:
            raise ValueError("`tau_enter_start` must not exceed `tau_enter`.")
        if self.tau_exit_start > self.tau_exit:
            raise ValueError("`tau_exit_start` must not exceed `tau_exit`.")
        if self.gate_warmup_timesteps <= 0:
            raise ValueError("`gate_warmup_timesteps` must be positive.")
        if self.checkpoint_timesteps_done is not None and self.checkpoint_timesteps_done < 0:
            raise ValueError("`checkpoint_timesteps_done` must be non-negative when provided.")
        self.feature_builder = SupervisorObservationBuilder(
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            history_length=observation_history,
            interaction_limit=interaction_limit,
        )
        self._previous_theta = constraints.default
        self._previous_theta_delta = ZERO_THETA_VECTOR
        self._gate_open = False
        self._step_diagnostics = DecisionDiagnostics()

    def reset(self, seed: int | None = None) -> None:
        self.fallback_fsm.reset(seed)
        self.feature_builder.reset()
        if self.policy is not None:
            self.policy.reset(seed)
        self._previous_theta = self.constraints.default
        self._previous_theta_delta = ZERO_THETA_VECTOR
        self._gate_open = False
        self._step_diagnostics = DecisionDiagnostics()

    def default_theta(self) -> tuple[float, float, float, float]:
        return self.constraints.default

    def observe_feedback(
        self,
        *,
        safety_corrections: tuple[float, ...],
        safety_slacks: tuple[float, ...],
        safety_fallbacks: tuple[bool, ...],
    ) -> None:
        self.feature_builder.observe_feedback(
            safety_corrections=safety_corrections,
            safety_slacks=safety_slacks,
            safety_fallbacks=safety_fallbacks,
        )

    def consume_step_diagnostics(self) -> DecisionDiagnostics:
        diagnostics = self._step_diagnostics
        self._step_diagnostics = DecisionDiagnostics()
        return diagnostics

    def select(self, observation, step: int) -> ModeDecision:
        fallback_decision = self.fallback_fsm.select(observation, step)
        if self.policy is None:
            return self._finalize_decision(
                mode=fallback_decision.mode,
                theta=fallback_decision.theta,
                source="fsm",
                confidence=1.0,
                confidence_raw=1.0,
                rl_fallback=False,
                gate_open=False,
                gate_reason="policy_missing",
                theta_clipped=False,
                normalized_obs_max_abs=0.0,
            )

        raw_observation = self.feature_builder.build(
            observation=observation,
            mode=fallback_decision.mode,
            previous_theta=self._previous_theta,
            previous_theta_delta=self._previous_theta_delta,
        )
        normalized_observation = self._normalize_observation(raw_observation)
        normalized_obs_max_abs = float(np.max(np.abs(normalized_observation))) if normalized_observation.size else 0.0

        inference = self.policy.infer(
            normalized_observation,
            deterministic=self.deterministic_eval,
        )
        confidence_raw = float(np.clip(inference.confidence, 0.0, 1.0))
        projected = self._project_theta(inference.theta)
        rate_limited = self._rate_limit(projected.theta)
        theta_clipped = projected.clipped or any(
            abs(a - b) > 1e-12 for a, b in zip(projected.theta, rate_limited, strict=True)
        )

        if normalized_obs_max_abs > self.ood_threshold:
            self._gate_open = False
            return self._finalize_decision(
                mode=fallback_decision.mode,
                theta=fallback_decision.theta,
                source="rl_fallback",
                confidence=confidence_raw,
                confidence_raw=confidence_raw,
                rl_fallback=True,
                gate_open=False,
                gate_reason="ood_threshold",
                theta_clipped=theta_clipped,
                normalized_obs_max_abs=normalized_obs_max_abs,
            )

        tau_enter, tau_exit = self._runtime_gate_thresholds()
        active_threshold = tau_exit if self._gate_open else tau_enter
        if confidence_raw < active_threshold:
            gate_reason = (
                "confidence_exit_threshold" if self._gate_open else "confidence_enter_threshold"
            )
            self._gate_open = False
            return self._finalize_decision(
                mode=fallback_decision.mode,
                theta=fallback_decision.theta,
                source="rl_fallback",
                confidence=confidence_raw,
                confidence_raw=confidence_raw,
                rl_fallback=True,
                gate_open=False,
                gate_reason=gate_reason,
                theta_clipped=theta_clipped,
                normalized_obs_max_abs=normalized_obs_max_abs,
            )

        gate_reason = "accepted_hold" if self._gate_open else "accepted_enter"
        self._gate_open = True
        return self._finalize_decision(
            mode=fallback_decision.mode,
            theta=rate_limited,
            source="rl",
            confidence=confidence_raw,
            confidence_raw=confidence_raw,
            rl_fallback=False,
            gate_open=True,
            gate_reason=gate_reason,
            theta_clipped=theta_clipped,
            normalized_obs_max_abs=normalized_obs_max_abs,
        )

    @staticmethod
    def _annealed_threshold(*, start: float, final: float, timesteps_done: int, warmup_timesteps: int) -> float:
        if timesteps_done >= warmup_timesteps:
            return float(final)
        progress = max(0.0, 1.0 - float(timesteps_done) / float(warmup_timesteps))
        return float(final - (final - start) * progress)

    def _runtime_gate_thresholds(self) -> tuple[float, float]:
        if self.deterministic_eval or self.checkpoint_timesteps_done is None:
            return (self.tau_enter, self.tau_exit)
        timesteps_done = max(self.checkpoint_timesteps_done, 0)
        tau_enter = self._annealed_threshold(
            start=self.tau_enter_start,
            final=self.tau_enter,
            timesteps_done=timesteps_done,
            warmup_timesteps=self.gate_warmup_timesteps,
        )
        tau_exit = self._annealed_threshold(
            start=self.tau_exit_start,
            final=self.tau_exit,
            timesteps_done=timesteps_done,
            warmup_timesteps=self.gate_warmup_timesteps,
        )
        return (tau_enter, tau_exit)

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        if self.normalizer.mean.shape == (1,) and self.normalizer.std.shape == (1,):
            return observation
        if self.normalizer.mean.shape != observation.shape or self.normalizer.std.shape != observation.shape:
            raise ValueError(
                "RL observation normalizer shape does not match the supervisor feature vector. "
                "Re-export the checkpoint with matching feature settings."
            )
        return self.normalizer.normalize(observation)

    def _project_theta(self, theta: tuple[float, float, float, float]) -> ThetaProjectionResult:
        if len(theta) != 4:
            raise ValueError("RL supervisor theta proposals must be length-4.")
        projected: list[float] = []
        clipped = False
        for value, lower, upper in zip(theta, self.constraints.lower, self.constraints.upper, strict=True):
            clipped_value = float(np.clip(float(value), lower, upper))
            projected.append(clipped_value)
            if abs(clipped_value - float(value)) > 1e-12:
                clipped = True
        return ThetaProjectionResult(theta=tuple(projected), clipped=clipped)

    def _rate_limit(self, theta: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        limited: list[float] = []
        for previous, target, limit in zip(
            self._previous_theta,
            theta,
            self.constraints.rate_limit,
            strict=True,
        ):
            delta = float(np.clip(target - previous, -limit, limit))
            limited.append(float(previous + delta))
        return tuple(limited)

    def _finalize_decision(
        self,
        *,
        mode: str,
        theta: tuple[float, float, float, float],
        source: str,
        confidence: float,
        confidence_raw: float,
        rl_fallback: bool,
        gate_open: bool,
        gate_reason: str,
        theta_clipped: bool,
        normalized_obs_max_abs: float,
    ) -> ModeDecision:
        theta_delta = tuple(
            float(current - previous)
            for current, previous in zip(theta, self._previous_theta, strict=True)
        )
        self._previous_theta = tuple(float(value) for value in theta)
        self._previous_theta_delta = theta_delta
        self._step_diagnostics = DecisionDiagnostics(
            source=source,
            confidence=float(confidence),
            confidence_raw=float(confidence_raw),
            theta=self._previous_theta,
            theta_delta=theta_delta,
            rl_fallback=bool(rl_fallback),
            gate_open=bool(gate_open),
            gate_reason=str(gate_reason),
            theta_clipped=bool(theta_clipped),
            normalized_obs_max_abs=float(normalized_obs_max_abs),
        )
        return ModeDecision(
            mode=mode,
            theta=self._previous_theta,
            source=source,
            confidence=float(confidence),
        )
