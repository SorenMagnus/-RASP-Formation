"""Optional Torch policy and normalization helpers for the RL supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from apflf.utils.types import RLThetaConfig

try:  # pragma: no cover - exercised only when torch is installed.
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - default test environment does not require torch.
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class PolicyInference:
    """Typed policy output consumed by the RL supervisor."""

    theta: tuple[float, float, float, float]
    confidence: float


class PolicyProtocol(Protocol):
    """Minimal inference contract shared by dummy and Torch-backed policies."""

    def reset(self, seed: int | None = None) -> None:
        """Reset any internal state used by the policy."""

    def infer(self, normalized_observation: np.ndarray, *, deterministic: bool) -> PolicyInference:
        """Predict a bounded theta proposal and confidence from a normalized observation."""


@dataclass(frozen=True)
class ObservationNormalizer:
    """Affine observation normalization used by training and evaluation."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        return (observation - self.mean) / np.maximum(self.std, 1e-6)

    @classmethod
    def identity(cls, dim: int) -> ObservationNormalizer:
        return cls(mean=np.zeros(dim, dtype=float), std=np.ones(dim, dtype=float))


@dataclass(frozen=True)
class PolicyBundle:
    """Checkpoint-loading result used by the decision builder."""

    policy: PolicyProtocol | None
    normalizer: ObservationNormalizer


class ConstantThetaPolicy:
    """Simple deterministic theta source used in tests and warm starts."""

    def __init__(
        self,
        theta: tuple[float, float, float, float],
        *,
        confidence: float = 1.0,
    ) -> None:
        self._theta = tuple(float(value) for value in theta)
        self._confidence = float(np.clip(confidence, 0.0, 1.0))

    def reset(self, seed: int | None = None) -> None:
        del seed

    def infer(self, normalized_observation: np.ndarray, *, deterministic: bool) -> PolicyInference:
        del normalized_observation, deterministic
        return PolicyInference(theta=self._theta, confidence=self._confidence)


if nn is not None:  # pragma: no branch

    class TorchBetaActorCritic(nn.Module):
        """Shared-trunk Beta actor-critic used by the custom PPO backend."""

        def __init__(
            self,
            *,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: tuple[int, ...] = (128, 128),
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.alpha_head = nn.Linear(in_dim, action_dim)
            self.beta_head = nn.Linear(in_dim, action_dim)
            self.value_head = nn.Linear(in_dim, 1)

        def distribution_and_value(self, observations: torch.Tensor) -> tuple[torch.distributions.Beta, torch.Tensor]:
            hidden = self.trunk(observations)
            alpha = F.softplus(self.alpha_head(hidden)) + 1.0
            beta = F.softplus(self.beta_head(hidden)) + 1.0
            value = self.value_head(hidden).squeeze(-1)
            return torch.distributions.Beta(alpha, beta), value


class TorchBetaPolicy:
    """Checkpoint-backed continuous theta policy with deterministic eval support."""

    def __init__(
        self,
        *,
        network: TorchBetaActorCritic,
        theta_config: RLThetaConfig,
    ) -> None:
        if torch is None:  # pragma: no cover - guarded by builder, kept defensive here.
            raise RuntimeError("TorchBetaPolicy requires the optional `torch` dependency.")
        self.network = network
        self.theta_config = theta_config

    def reset(self, seed: int | None = None) -> None:
        if seed is None or torch is None:
            return
        torch.manual_seed(int(seed))

    def infer(self, normalized_observation: np.ndarray, *, deterministic: bool) -> PolicyInference:
        if torch is None:  # pragma: no cover - guarded by builder.
            raise RuntimeError("TorchBetaPolicy requires the optional `torch` dependency.")

        observation_tensor = torch.as_tensor(
            np.asarray(normalized_observation, dtype=np.float32)[None, :],
            dtype=torch.float32,
        )
        with torch.no_grad():
            distribution, _ = self.network.distribution_and_value(observation_tensor)
            sample01 = distribution.mean if deterministic else distribution.sample()
            theta_lower = torch.as_tensor(self.theta_config.lower, dtype=torch.float32)
            theta_upper = torch.as_tensor(self.theta_config.upper, dtype=torch.float32)
            theta = theta_lower + sample01.squeeze(0) * (theta_upper - theta_lower)
            entropy = distribution.entropy().squeeze(0)
            normalized_entropy = torch.exp(torch.clamp(entropy, max=0.0))
            confidence = torch.clamp(1.0 - normalized_entropy.mean(), 0.0, 1.0)
        return PolicyInference(
            theta=tuple(float(value) for value in theta.cpu().numpy()),
            confidence=float(confidence.cpu().item()),
        )


def load_policy_bundle(
    *,
    checkpoint_path: str,
    theta_config: RLThetaConfig,
) -> PolicyBundle:
    """Load a Torch PPO policy bundle when a checkpoint is available."""

    if not checkpoint_path:
        return PolicyBundle(policy=None, normalizer=ObservationNormalizer.identity(dim=1))

    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.exists():
        return PolicyBundle(policy=None, normalizer=ObservationNormalizer.identity(dim=1))
    if torch is None:  # pragma: no cover - only triggered when RL extra is missing.
        raise RuntimeError(
            "Loading an RL checkpoint requires the optional `torch` dependency. "
            "Install with `pip install .[rl]`."
        )

    payload = torch.load(resolved_path, map_location="cpu")
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload.get("action_dim", 4))
    hidden_sizes = tuple(int(value) for value in payload.get("hidden_sizes", (128, 128)))
    network = TorchBetaActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()

    normalizer_mean = np.asarray(payload.get("normalizer_mean", np.zeros(obs_dim, dtype=float)), dtype=float)
    normalizer_std = np.asarray(payload.get("normalizer_std", np.ones(obs_dim, dtype=float)), dtype=float)
    if normalizer_mean.shape != (obs_dim,) or normalizer_std.shape != (obs_dim,):
        raise ValueError("RL checkpoint normalizer statistics do not match `obs_dim`.")

    return PolicyBundle(
        policy=TorchBetaPolicy(network=network, theta_config=theta_config),
        normalizer=ObservationNormalizer(mean=normalizer_mean, std=normalizer_std),
    )
