from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from apflf.analysis.rl_attribution import (
    aggregate_seed_rows,
    compare_to_reference_bundle,
    summarize_rl_seed,
)
from apflf.rl.policy import ObservationNormalizer, PolicyBundle, PolicyInference
from apflf.sim.replay import load_replay_bundle
from apflf.sim.runner import run_batch
from apflf.utils.config import load_config


class DummySequencePolicy:
    """Deterministic policy used to build replayable RL attribution fixtures."""

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


def _load_script_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run_pair(
    tmp_path: Path,
    monkeypatch,
    *,
    deterministic_eval: bool = True,
    checkpoint_timesteps_done: int | None = None,
    confidence: float = 0.92,
) -> tuple[Path, Path]:
    config = _default_config()
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=str(tmp_path), save_traj=True),
    )
    reference_dir = run_batch(config=config, seeds=[2], exp_id="no_rl_case")

    rl_config = replace(
        config,
        decision=replace(
            config.decision,
            kind="rl",
            rl=replace(
                config.decision.rl,
                checkpoint_path="dummy.ckpt",
                deterministic_eval=deterministic_eval,
            ),
        ),
    )

    def _bundle_factory(*, checkpoint_path: str, theta_config):
        del checkpoint_path, theta_config
        return PolicyBundle(
            policy=DummySequencePolicy(outputs=[((1.7, 1.8, 1.6, 0.9), confidence)]),
            normalizer=ObservationNormalizer(
                mean=np.zeros(108, dtype=float),
                std=np.full(108, 100.0, dtype=float),
            ),
            checkpoint_timesteps_done=checkpoint_timesteps_done,
        )

    monkeypatch.setattr("apflf.decision.rl_mode.load_rl_policy_bundle", _bundle_factory)
    rl_dir = run_batch(config=rl_config, seeds=[2], exp_id="rl_case")
    return reference_dir, rl_dir


def test_rl_attribution_summarizes_and_compares_seed(monkeypatch, tmp_path: Path) -> None:
    reference_dir, rl_dir = _make_run_pair(tmp_path, monkeypatch)
    reference_bundle = load_replay_bundle(reference_dir, 2)
    rl_bundle = load_replay_bundle(rl_dir, 2)

    summary = summarize_rl_seed(rl_bundle)
    comparison = compare_to_reference_bundle(rl_bundle, reference_bundle)
    aggregate = aggregate_seed_rows([{"seed": 2, **summary, **comparison}])

    assert summary["num_steps"] > 0
    assert summary["confidence_raw_mean"] > 0.0
    assert summary["gate_open_steps"] > 0
    assert summary["accepted_enter_steps"] > 0
    assert summary["effective_tau_enter_mean"] == pytest.approx(0.55)
    assert summary["effective_tau_exit_mean"] == pytest.approx(0.45)
    assert summary["theta_changed_steps"] > 0
    assert summary["theta_clip_steps"] > 0
    assert summary["dominant_bottleneck"] in {
        "supervisor_gating",
        "safety_engagement",
        "weak_theta_impact",
        "nominal_controller",
    }
    assert comparison["shared_steps"] > 0
    assert comparison["nominal_layer_changed"] is True
    assert comparison["leader_target_speed_delta_abs_mean"] > 0.0
    assert aggregate["num_seeds"] == 1
    assert rl_bundle.snapshots[0].decision_diagnostics.confidence_raw == pytest.approx(
        rl_bundle.snapshots[0].decision_diagnostics.confidence
    )
    assert rl_bundle.snapshots[0].decision_diagnostics.gate_open is True
    assert rl_bundle.snapshots[0].decision_diagnostics.gate_reason == "accepted_enter"
    assert rl_bundle.snapshots[0].decision_diagnostics.effective_tau_enter == pytest.approx(0.55)
    assert rl_bundle.snapshots[0].decision_diagnostics.effective_tau_exit == pytest.approx(0.45)


def test_rl_attribution_uses_persisted_annealed_gate_thresholds(monkeypatch, tmp_path: Path) -> None:
    reference_dir, rl_dir = _make_run_pair(
        tmp_path,
        monkeypatch,
        deterministic_eval=False,
        checkpoint_timesteps_done=0,
        confidence=0.30,
    )
    del reference_dir
    rl_bundle = load_replay_bundle(rl_dir, 2)

    summary = summarize_rl_seed(rl_bundle)

    assert rl_bundle.snapshots[0].decision_diagnostics.effective_tau_enter == pytest.approx(0.25)
    assert rl_bundle.snapshots[0].decision_diagnostics.effective_tau_exit == pytest.approx(0.15)
    assert summary["effective_tau_enter_mean"] == pytest.approx(0.25)
    assert summary["effective_tau_exit_mean"] == pytest.approx(0.15)
    assert summary["gate_open_steps"] > 0


def test_analyze_s5_rl_attribution_script_writes_outputs(monkeypatch, tmp_path: Path, capsys) -> None:
    reference_dir, rl_dir = _make_run_pair(tmp_path, monkeypatch)
    script = _load_script_module("analyze_s5_rl_attribution", "scripts/analyze_s5_rl_attribution.py")
    output_dir = tmp_path / "analysis_bundle"

    exit_code = script.main(
        [
            "--rl-run-dir",
            str(rl_dir),
            "--reference-run-dir",
            str(reference_dir),
            "--output-dir",
            str(output_dir),
            "--as-json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["num_seeds"] == 1
    assert report["effective_tau_enter_mean_mean"] == pytest.approx(0.55)
    assert report["effective_tau_exit_mean_mean"] == pytest.approx(0.45)
    assert (output_dir / "seed_attribution.csv").exists()
    assert (output_dir / "aggregate.json").exists()
    assert (output_dir / "attribution_overview.pdf").exists()
