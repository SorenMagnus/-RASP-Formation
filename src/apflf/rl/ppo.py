"""Custom Torch PPO backend for the stage-1 RL supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

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

    def load_state(self, *, count: float, mean: np.ndarray, m2: np.ndarray) -> None:
        """Restore running statistics from a serialized checkpoint payload."""

        self.count = float(count)
        self.mean = np.asarray(mean, dtype=float).copy()
        self.m2 = np.asarray(m2, dtype=float).copy()


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
        self.initial_seed: int | None = None
        self.timesteps_done = 0
        self.rollout_seed_next = 0
        self.logs: list[dict[str, float]] = []
        self._session_start_time = 0.0

    def train(
        self,
        *,
        seed: int = 0,
        periodic_checkpoint_path: Path | None = None,
        resume_from: Path | None = None,
    ) -> list[dict[str, float]]:
        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        self._session_start_time = time.perf_counter()
        if resume_from is not None:
            self.load_checkpoint(Path(resume_from), expected_seed=seed)
            self._emit_progress(
                event="resume",
                checkpoint_path=Path(resume_from),
            )
        else:
            self._initialize_training_state(seed=seed)
            self._emit_progress(event="start")

        if self.config.total_timesteps < self.timesteps_done:
            raise ValueError(
                "Requested `total_timesteps` is smaller than the checkpoint's completed timesteps."
            )

        while self.timesteps_done < self.config.total_timesteps:
            rollout_index = self._rollout_index()
            self._emit_progress(
                event="rollout_start",
                rollout_index=rollout_index,
                rollout_seed=self.rollout_seed_next,
            )
            batch = self._collect_rollout(seed=self.rollout_seed_next)
            rollout_log = self._update(batch)
            batch_steps = int(batch["observations"].shape[0])
            self.timesteps_done += batch_steps
            self.rollout_seed_next += 1
            self.logs.append(rollout_log)
            if periodic_checkpoint_path is not None:
                self.save_checkpoint(Path(periodic_checkpoint_path))
            self._emit_progress(
                event="rollout_done",
                rollout_index=rollout_index,
                rollout_seed=self.rollout_seed_next - 1,
                batch_steps=batch_steps,
                rollout_log=rollout_log,
                checkpoint_path=periodic_checkpoint_path,
            )
        self._emit_progress(event="complete")
        return list(self.logs)

    def save_checkpoint(self, path: Path) -> None:
        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        if self.initial_seed is None:
            raise RuntimeError("Trainer state has not been initialized; call `train()` before saving.")
        payload = {
            "state_dict": self.network.state_dict(),
            "obs_dim": self.env.observation_dim,
            "action_dim": 4,
            "hidden_sizes": self.config.hidden_sizes,
            "normalizer_mean": self.obs_stats.mean,
            "normalizer_std": self.obs_stats.std,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_stats_count": float(self.obs_stats.count),
            "obs_stats_mean": self.obs_stats.mean.copy(),
            "obs_stats_m2": self.obs_stats.m2.copy(),
            "timesteps_done": int(self.timesteps_done),
            "rollout_seed_next": int(self.rollout_seed_next),
            "initial_seed": int(self.initial_seed),
            "logs": [dict(item) for item in self.logs],
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "trainer_config": self._trainer_config_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(self, path: Path, *, expected_seed: int) -> None:
        """Restore a rollout-boundary training state from a checkpoint payload."""

        if torch is None:  # pragma: no cover - requires optional dependency.
            raise RuntimeError("PPOTrainer requires the optional `torch` dependency.")
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {resolved_path}")
        payload = torch.load(resolved_path, map_location="cpu")
        self._validate_checkpoint_payload(payload, expected_seed=expected_seed)
        self.network.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._move_optimizer_state_to_device()
        self.obs_stats.load_state(
            count=float(payload["obs_stats_count"]),
            mean=np.asarray(payload["obs_stats_mean"], dtype=float),
            m2=np.asarray(payload["obs_stats_m2"], dtype=float),
        )
        self.timesteps_done = int(payload["timesteps_done"])
        self.rollout_seed_next = int(payload["rollout_seed_next"])
        self.initial_seed = int(payload["initial_seed"])
        self.logs = [dict(item) for item in payload.get("logs", [])]
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload.get("torch_cuda_rng_state_all", []))

    def _initialize_training_state(self, *, seed: int) -> None:
        self.initial_seed = int(seed)
        self.timesteps_done = 0
        self.rollout_seed_next = int(seed)
        self.logs = []
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _trainer_config_dict(self) -> dict[str, object]:
        return {
            "steps_per_rollout": int(self.config.steps_per_rollout),
            "gamma": float(self.config.gamma),
            "gae_lambda": float(self.config.gae_lambda),
            "clip_ratio": float(self.config.clip_ratio),
            "value_loss_coef": float(self.config.value_loss_coef),
            "entropy_coef": float(self.config.entropy_coef),
            "learning_rate": float(self.config.learning_rate),
            "update_epochs": int(self.config.update_epochs),
            "minibatch_size": int(self.config.minibatch_size),
            "hidden_sizes": tuple(int(value) for value in self.config.hidden_sizes),
            "device_type": self.device.type,
        }

    def _validate_checkpoint_payload(self, payload: dict[str, object], *, expected_seed: int) -> None:
        expected_obs_dim = int(self.env.observation_dim)
        if int(payload.get("obs_dim", -1)) != expected_obs_dim:
            raise ValueError("Checkpoint observation dimension does not match the current trainer.")
        if int(payload.get("action_dim", -1)) != 4:
            raise ValueError("Checkpoint action dimension does not match the current trainer.")
        hidden_sizes = tuple(int(value) for value in payload.get("hidden_sizes", ()))
        if hidden_sizes != tuple(int(value) for value in self.config.hidden_sizes):
            raise ValueError("Checkpoint hidden sizes do not match the current trainer.")
        initial_seed = int(payload.get("initial_seed", -1))
        if initial_seed != int(expected_seed):
            raise ValueError("Checkpoint seed does not match the requested resume seed.")
        saved_config = dict(payload.get("trainer_config", {}))
        if saved_config != self._trainer_config_dict():
            raise ValueError("Checkpoint trainer configuration does not match the current trainer.")
        if int(payload.get("timesteps_done", -1)) < 0:
            raise ValueError("Checkpoint `timesteps_done` must be non-negative.")

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _rollout_index(self) -> int:
        if self.initial_seed is None:
            return 0
        return int(self.rollout_seed_next - self.initial_seed + 1)

    def _emit_progress(
        self,
        *,
        event: str,
        rollout_index: int | None = None,
        rollout_seed: int | None = None,
        batch_steps: int | None = None,
        rollout_log: dict[str, float] | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        elapsed_s = max(time.perf_counter() - self._session_start_time, 0.0)
        total_timesteps = max(int(self.config.total_timesteps), 1)
        progress_pct = 100.0 * min(self.timesteps_done / total_timesteps, 1.0)
        message_parts = [
            f"[ppo] {event}",
            f"device={self.device.type}",
            f"seed={self.initial_seed if self.initial_seed is not None else 'unset'}",
            f"timesteps_done={self.timesteps_done}/{self.config.total_timesteps}",
            f"progress={progress_pct:.2f}%",
            f"elapsed_s={elapsed_s:.1f}",
        ]
        if rollout_index is not None:
            message_parts.append(f"rollout_index={rollout_index}")
        if rollout_seed is not None:
            message_parts.append(f"rollout_seed={rollout_seed}")
        if batch_steps is not None:
            message_parts.append(f"batch_steps={batch_steps}")
        if checkpoint_path is not None:
            message_parts.append(f"checkpoint={Path(checkpoint_path)}")
        if rollout_log is not None:
            message_parts.extend(
                [
                    f"policy_loss={rollout_log['policy_loss']:.6f}",
                    f"value_loss={rollout_log['value_loss']:.6f}",
                    f"entropy={rollout_log['entropy']:.6f}",
                    f"reward_total_mean={rollout_log.get('reward_total_mean', 0.0):.6f}",
                    f"qp_engagement_ratio_mean={rollout_log.get('qp_engagement_ratio_mean', 0.0):.6f}",
                    f"fallback_ratio_mean={rollout_log.get('fallback_ratio_mean', 0.0):.6f}",
                ]
            )
        print(" ".join(message_parts), flush=True)

    def _collect_rollout(self, *, seed: int) -> dict[str, object]:
        observations: list[np.ndarray] = []
        normalized_observations: list[np.ndarray] = []
        action01s: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []
        step_infos: list[dict[str, object]] = []
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
            next_obs, reward, terminated, truncated, info = self.env.step(theta)
            done = bool(terminated or truncated)

            observations.append(np.asarray(obs, dtype=float))
            normalized_observations.append(np.asarray(normalized_obs, dtype=float))
            action01s.append(action01.squeeze(0).cpu().numpy())
            log_probs.append(float(log_prob.cpu().item()))
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(float(value.squeeze(0).cpu().item()))
            step_infos.append(dict(info))

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
        reward_array = np.asarray(rewards, dtype=float)
        return {
            "observations": np.asarray(observations, dtype=float),
            "normalized_observations": np.asarray(normalized_observations, dtype=float),
            "action01s": np.asarray(action01s, dtype=float),
            "log_probs": np.asarray(log_probs, dtype=float),
            "returns": returns,
            "advantages": advantages,
            "values": np.asarray(values, dtype=float),
            "rollout_stats": self._summarize_rollout_infos(step_infos=step_infos, rewards=reward_array),
        }

    def _summarize_rollout_infos(
        self,
        *,
        step_infos: list[dict[str, object]],
        rewards: np.ndarray,
    ) -> dict[str, float]:
        reward_key_map = {
            "progress": "reward_progress_mean",
            "formation": "reward_form_mean",
            "intervene": "reward_intervene_mean",
            "qp": "reward_qp_mean",
            "fallback": "reward_fallback_mean",
            "slack": "reward_slack_mean",
            "theta_rate": "reward_theta_rate_mean",
            "goal": "reward_goal_mean",
            "collision": "reward_collision_mean",
            "boundary": "reward_boundary_mean",
        }
        aggregates: dict[str, float] = {value: 0.0 for value in reward_key_map.values()}
        aggregates.update(
            {
                "reward_total_mean": float(np.mean(rewards)) if rewards.size else 0.0,
                "qp_engagement_ratio_mean": 0.0,
                "fallback_ratio_mean": 0.0,
                "theta_delta_linf_mean": 0.0,
            }
        )
        if not step_infos:
            return aggregates

        for info in step_infos:
            reward_terms = info.get("reward_terms", {})
            if isinstance(reward_terms, dict):
                for reward_key, aggregate_key in reward_key_map.items():
                    aggregates[aggregate_key] += float(reward_terms.get(reward_key, 0.0))
            aggregates["qp_engagement_ratio_mean"] += float(info.get("qp_engagement_ratio_step", 0.0))
            aggregates["fallback_ratio_mean"] += float(info.get("fallback_ratio_step", 0.0))
            aggregates["theta_delta_linf_mean"] += float(info.get("theta_delta_linf", 0.0))

        step_count = float(len(step_infos))
        for key in reward_key_map.values():
            aggregates[key] /= step_count
        aggregates["qp_engagement_ratio_mean"] /= step_count
        aggregates["fallback_ratio_mean"] /= step_count
        aggregates["theta_delta_linf_mean"] /= step_count
        return aggregates

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

    def _update(self, batch: dict[str, object]) -> dict[str, float]:
        rollout_stats = dict(batch.pop("rollout_stats", {}))
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
            **rollout_stats,
        }
