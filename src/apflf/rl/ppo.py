"""Custom Torch PPO backend for the stage-1 RL supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apflf.rl.policy import TorchBetaActorCritic, torch
from apflf.utils.types import RLThetaConfig

if torch is not None:  # pragma: no branch
    from torch import optim


@dataclass(frozen=True)
class PPOConfig:
    """Training hyperparameters for the custom PPO backend."""

    total_timesteps: int = 20_000
    steps_per_rollout: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_loss_coef: float = 0.50
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_sizes: tuple[int, ...] = (128, 128)


@dataclass
class RunningObservationStats:
    """Simple running mean/std statistics used for observation normalization."""

    dim: int

    def __post_init__(self) -> None:
        self.count = 1e-4
        self.mean = np.zeros(self.dim, dtype=float)
        self.m2 = np.ones(self.dim, dtype=float)

    @property
    def std(self) -> np.ndarray:
        variance = np.maximum(self.m2 / max(self.count, 1.0), 1e-6)
        return np.sqrt(variance)

    def update(self, observations: np.ndarray) -> None:
        observations = np.asarray(observations, dtype=float)
        if observations.ndim == 1:
            observations = observations[None, :]
        batch_mean = np.mean(observations, axis=0)
        batch_var = np.var(observations, axis=0)
        batch_count = observations.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        self.m2 = (
            self.m2
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / total_count
        )
        self.mean = new_mean
        self.count = total_count

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=float)
        return (observations - self.mean) / np.maximum(self.std, 1e-6)


class PPOTrainer:
    """A compact PPO trainer for the stage-1 continuous-theta supervisor."""

    def __init__(
        self,
        *,
        env,
        theta_config: RLThetaConfig,
        config: PPOConfig,
        device: str = "cpu",
    ) -> None:
        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        self.env = env
        self.theta_config = theta_config
        self.config = config
        self.device = torch.device(device)
        self.obs_stats = RunningObservationStats(dim=self.env.observation_dim)
        self.network = TorchBetaActorCritic(
            obs_dim=self.env.observation_dim,
            action_dim=4,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.theta_lower = torch.as_tensor(theta_config.lower, dtype=torch.float32, device=self.device)
        self.theta_upper = torch.as_tensor(theta_config.upper, dtype=torch.float32, device=self.device)

    def train(self, *, seed: int = 0) -> list[dict[str, float]]:
        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        logs: list[dict[str, float]] = []
        timesteps = 0
        rollout_seed = int(seed)
        while timesteps < self.config.total_timesteps:
            batch = self._collect_rollout(seed=rollout_seed)
            rollout_seed += 1
            timesteps += int(batch["observations"].shape[0])
            logs.append(self._update(batch))
        return logs

    def save_checkpoint(self, path: Path) -> None:
        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        payload = {
            "state_dict": self.network.state_dict(),
            "obs_dim": self.env.observation_dim,
            "action_dim": 4,
            "hidden_sizes": self.config.hidden_sizes,
            "normalizer_mean": self.obs_stats.mean,
            "normalizer_std": self.obs_stats.std,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def _collect_rollout(self, *, seed: int) -> dict[str, np.ndarray]:
        observations: list[np.ndarray] = []
        normalized_observations: list[np.ndarray] = []
        action01s: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []
        obs = self.env.reset(seed=seed)

        for _ in range(self.config.steps_per_rollout):
            self.obs_stats.update(obs)
            normalized_obs = self.obs_stats.normalize(obs)
            obs_tensor = torch.as_tensor(normalized_obs[None, :], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                distribution, value = self.network.distribution_and_value(obs_tensor)
                action01 = distribution.sample()
                log_prob = distribution.log_prob(action01).sum(dim=-1)
            theta = self._theta_from_unit_box(action01.squeeze(0).cpu().numpy())
            next_obs, reward, terminated, truncated, _ = self.env.step(theta)
            done = bool(terminated or truncated)

            observations.append(np.asarray(obs, dtype=float))
            normalized_observations.append(np.asarray(normalized_obs, dtype=float))
            action01s.append(action01.squeeze(0).cpu().numpy())
            log_probs.append(float(log_prob.cpu().item()))
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(float(value.squeeze(0).cpu().item()))

            obs = self.env.reset(seed=seed + len(observations)) if done else next_obs

        self.obs_stats.update(obs)
        normalized_last_obs = self.obs_stats.normalize(obs)
        last_obs_tensor = torch.as_tensor(normalized_last_obs[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_value = self.network.distribution_and_value(last_obs_tensor)
        advantages, returns = self._gae(
            rewards=np.asarray(rewards, dtype=float),
            dones=np.asarray(dones, dtype=float),
            values=np.asarray(values, dtype=float),
            last_value=float(last_value.squeeze(0).cpu().item()),
        )
        return {
            "observations": np.asarray(observations, dtype=float),
            "normalized_observations": np.asarray(normalized_observations, dtype=float),
            "action01s": np.asarray(action01s, dtype=float),
            "log_probs": np.asarray(log_probs, dtype=float),
            "returns": returns,
            "advantages": advantages,
            "values": np.asarray(values, dtype=float),
        }

    def _theta_from_unit_box(self, action01: np.ndarray) -> tuple[float, float, float, float]:
        theta = np.asarray(self.theta_config.lower, dtype=float) + np.asarray(action01, dtype=float) * (
            np.asarray(self.theta_config.upper, dtype=float) - np.asarray(self.theta_config.lower, dtype=float)
        )
        return tuple(float(value) for value in theta)

    def _gae(
        self,
        *,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=float)
        gae = 0.0
        next_value = last_value
        for index in reversed(range(rewards.size)):
            mask = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * next_value * mask - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / max(np.std(advantages), 1e-6)
        return advantages.astype(float), returns.astype(float)

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(batch["normalized_observations"], dtype=torch.float32, device=self.device)
        actions01 = torch.as_tensor(batch["action01s"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        update_count = 0
        indices = np.arange(observations.shape[0])

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, indices.size, self.config.minibatch_size):
                batch_indices = indices[start : start + self.config.minibatch_size]
                obs_batch = observations[batch_indices]
                action_batch = actions01[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                return_batch = returns[batch_indices]
                advantage_batch = advantages[batch_indices]

                distribution, value = self.network.distribution_and_value(obs_batch)
                log_prob = distribution.log_prob(action_batch).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                ratio = torch.exp(log_prob - old_log_prob_batch)
                surrogate1 = ratio * advantage_batch
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantage_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = torch.mean((return_batch - value) ** 2)
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_loss_total += float(policy_loss.detach().cpu().item())
                value_loss_total += float(value_loss.detach().cpu().item())
                entropy_total += float(entropy.detach().cpu().item())
                update_count += 1

        return {
            "policy_loss": policy_loss_total / max(update_count, 1),
            "value_loss": value_loss_total / max(update_count, 1),
            "entropy": entropy_total / max(update_count, 1),
        }
