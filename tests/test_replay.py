"""Tests for replaying saved rollout artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from apflf.sim.replay import compare_summary_dicts, load_replay_bundle, read_summary_row, recompute_summary
from apflf.sim.runner import run_batch
from apflf.utils.config import load_config


def test_replay_recomputes_saved_summary(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            output_root=str(tmp_path),
            save_traj=True,
        ),
    )
    output_dir = run_batch(config=config, seeds=[3], exp_id="replay_case")

    replay_summary = recompute_summary(output_dir, 3)
    saved_summary = read_summary_row(output_dir, 3)
    compared_saved = {
        key: value
        for key, value in saved_summary.items()
        if key not in {"seed", "config_hash", "git_commit"}
    }
    mismatches = compare_summary_dicts(compared_saved, replay_summary)

    assert not mismatches

    bundle = load_replay_bundle(output_dir, 3)
    diagnostics = bundle.snapshots[0].nominal_diagnostics
    assert diagnostics.leader_target_speed > 0.0
    assert diagnostics.leader_risk_score >= 0.0

    force_sum = (
        np.asarray(diagnostics.leader_force.attractive, dtype=float)
        + np.asarray(diagnostics.leader_force.formation, dtype=float)
        + np.asarray(diagnostics.leader_force.consensus, dtype=float)
        + np.asarray(diagnostics.leader_force.road, dtype=float)
        + np.asarray(diagnostics.leader_force.obstacle, dtype=float)
        + np.asarray(diagnostics.leader_force.peer, dtype=float)
        + np.asarray(diagnostics.leader_force.behavior, dtype=float)
        + np.asarray(diagnostics.leader_force.guidance, dtype=float)
        + np.asarray(diagnostics.leader_force.escape, dtype=float)
    )
    assert np.allclose(force_sum, np.asarray(diagnostics.leader_force.total, dtype=float))
