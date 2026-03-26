"""Mode decision interfaces, helpers, and builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from apflf.utils.types import (
    DEFAULT_THETA_VECTOR,
    DecisionConfig,
    DecisionDiagnostics,
    ModeDecision,
    Observation,
)

DEFAULT_MODE_LABEL = "topology=line|behavior=follow|gain=nominal"
SUPPORTED_MODE_TOPOLOGIES = {"line", "diamond"}
SUPPORTED_MODE_BEHAVIORS = {
    "follow",
    "yield_left",
    "yield_right",
    "escape_left",
    "escape_right",
    "recover_left",
    "recover_right",
}
SUPPORTED_MODE_GAINS = {"nominal", "cautious", "assertive"}


@dataclass(frozen=True)
class ParsedMode:
    """Structured discrete mode command."""

    topology: str = "line"
    behavior: str = "follow"
    gain: str = "nominal"

    def to_label(self) -> str:
        """Serialize the discrete mode into a stable label."""

        return compose_mode_label(
            topology=self.topology,
            behavior=self.behavior,
            gain=self.gain,
        )


def compose_mode_label(*, topology: str, behavior: str, gain: str) -> str:
    """Build a stable topology/behavior/gain mode label."""

    return f"topology={topology}|behavior={behavior}|gain={gain}"


def parse_mode_label(mode: str) -> ParsedMode:
    """Parse a stable mode label and fall back to safe defaults on legacy labels."""

    normalized = (mode or "").strip()
    if not normalized or normalized == "nominal_follow":
        return ParsedMode()

    tokens = {"topology": "line", "behavior": "follow", "gain": "nominal"}
    for item in normalized.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", maxsplit=1)
        tokens[key.strip().lower()] = value.strip().lower()

    topology = tokens["topology"]
    behavior = tokens["behavior"]
    gain = tokens["gain"]
    if topology not in SUPPORTED_MODE_TOPOLOGIES:
        topology = "line"
    if behavior not in SUPPORTED_MODE_BEHAVIORS:
        behavior = "follow"
    if gain not in SUPPORTED_MODE_GAINS:
        gain = "nominal"
    return ParsedMode(topology=topology, behavior=behavior, gain=gain)


class ModeDecisionModule(ABC):
    """Discrete mode decision interface."""

    def reset(self, seed: int | None = None) -> None:
        """Reset any internal state used by the decision module."""

        del seed

    @abstractmethod
    def select(self, observation: Observation, step: int) -> ModeDecision:
        """Select a structured decision from the current observation."""

    def select_mode(self, observation: Observation) -> str:
        """Backward-compatible string-only mode selection."""

        return self.select(observation, observation.step_index).mode

    def observe_feedback(
        self,
        *,
        safety_corrections: tuple[float, ...],
        safety_slacks: tuple[float, ...],
        safety_fallbacks: tuple[bool, ...],
    ) -> None:
        """Consume previous-step safety feedback for stateful decision modules."""

        del safety_corrections, safety_slacks, safety_fallbacks

    def consume_step_diagnostics(self) -> DecisionDiagnostics:
        """Return step diagnostics for persistence and reset the local cache."""

        return DecisionDiagnostics()

    @abstractmethod
    def default_theta(self) -> tuple[float, float, float, float]:
        """Return the exact theta that reproduces the white-box baseline."""

        return DEFAULT_THETA_VECTOR


class StaticModeDecision(ModeDecisionModule):
    """Mode selector that always returns the configured default mode."""

    def __init__(self, default_mode: str) -> None:
        self.default_mode = parse_mode_label(default_mode).to_label()

    def select(self, observation: Observation, step: int) -> ModeDecision:
        del observation, step
        return ModeDecision(
            mode=self.default_mode,
            theta=self.default_theta(),
            source="static",
            confidence=1.0,
        )

    def default_theta(self) -> tuple[float, float, float, float]:
        return DEFAULT_THETA_VECTOR


def build_mode_decision(
    *,
    config: DecisionConfig,
    vehicle_length: float,
    vehicle_width: float,
    safe_distance: float,
) -> ModeDecisionModule:
    """Build the configured mode decision module."""

    if config.kind == "static":
        return StaticModeDecision(default_mode=config.default_mode)
    if config.kind == "fsm":
        from apflf.decision.fsm_mode import FSMModeDecision

        return FSMModeDecision(
            config=config,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            safe_distance=safe_distance,
        )
    if config.kind == "rl":
        from apflf.decision.fsm_mode import FSMModeDecision
        from apflf.decision.rl_mode import RLSupervisor, load_rl_policy_bundle

        fallback_fsm = FSMModeDecision(
            config=config,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            safe_distance=safe_distance,
        )
        policy_bundle = load_rl_policy_bundle(
            checkpoint_path=config.rl.checkpoint_path,
            theta_config=config.rl.theta,
        )
        return RLSupervisor(
            fallback_fsm=fallback_fsm,
            policy=policy_bundle.policy,
            normalizer=policy_bundle.normalizer,
            constraints=config.rl.theta,
            confidence_threshold=config.rl.confidence_threshold,
            ood_threshold=config.rl.ood_threshold,
            deterministic_eval=config.rl.deterministic_eval,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            observation_history=config.rl.observation_history,
            interaction_limit=config.rl.interaction_limit,
        )
    raise ValueError(f"Unsupported decision module: {config.kind}")
