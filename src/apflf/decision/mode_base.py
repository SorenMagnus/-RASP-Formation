"""Mode decision interfaces, helpers, and builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from apflf.utils.types import DecisionConfig, Observation

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

    @abstractmethod
    def select_mode(self, observation: Observation) -> str:
        """Select a discrete mode from the current observation."""


class StaticModeDecision(ModeDecisionModule):
    """Mode selector that always returns the configured default mode."""

    def __init__(self, default_mode: str) -> None:
        self.default_mode = parse_mode_label(default_mode).to_label()

    def select_mode(self, observation: Observation) -> str:
        del observation
        return self.default_mode


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
    raise ValueError(f"Unsupported decision module: {config.kind}")
