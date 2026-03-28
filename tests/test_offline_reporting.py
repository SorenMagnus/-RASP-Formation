from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_offline_paper_bundle(tmp_path: Path) -> None:
    module = _load_module("build_offline_paper_bundle", "scripts/build_offline_paper_bundle.py")

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    rl_dir = tmp_path / "rl_eval"
    rl_dir.mkdir()

    metrics_row_headers = {
        "seed": 0,
        "leader_goal_error": 10.0,
        "time_to_goal": 12.0,
        "mean_speed": 3.5,
        "leader_path_length_ratio": 1.02,
        "min_ttc": 1.8,
        "min_obstacle_clearance": 0.7,
        "collision_count": 0,
        "boundary_violation_count": 0,
        "fallback_ratio": 0.1,
        "terminal_formation_error": 0.4,
    }
    _write_summary_csv(
        baseline_dir / "summary.csv",
        [metrics_row_headers, {**metrics_row_headers, "seed": 1, "leader_goal_error": 11.0}],
    )
    _write_summary_csv(
        rl_dir / "summary.csv",
        [{**metrics_row_headers, "leader_goal_error": 8.0}, {**metrics_row_headers, "seed": 1, "leader_goal_error": 7.5}],
    )

    (baseline_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "s5_dense_multi_agent"},
                "controller": {"kind": "adaptive_apf"},
                "decision": {"kind": "fsm", "rl": {"checkpoint_path": ""}},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (rl_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "s5_dense_multi_agent"},
                "controller": {"kind": "adaptive_apf"},
                "decision": {
                    "kind": "rl",
                    "rl": {"checkpoint_path": "outputs/rl_train_s5_param_only/checkpoints/main.pt"},
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "bundle"
    exit_code = module.main(
        [
            "--run-dir",
            str(baseline_dir),
            "--run-dir",
            str(rl_dir),
            "--output-dir",
            str(output_dir),
            "--reference-method",
            "no_rl",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "all_runs.csv").exists()
    assert (output_dir / "tables" / "main_results.csv").exists()
    assert (output_dir / "tables" / "run_inventory.csv").exists()
    assert (output_dir / "figures" / "metric_overview.pdf").exists()

    inventory = list(csv.DictReader((output_dir / "tables" / "run_inventory.csv").open("r", encoding="utf-8")))
    methods = {row["method"] for row in inventory}
    assert methods == {"no_rl", "rl_param_only"}


def test_report_rl_training_status_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    torch = pytest.importorskip("torch")
    module = _load_module("report_rl_training_status", "scripts/report_rl_training_status.py")

    train_dir = tmp_path / "rl_train"
    checkpoints_dir = train_dir / "checkpoints"
    logs_dir = train_dir / "logs"
    checkpoints_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    torch.save(
        {
            "timesteps_done": 1024,
            "rollout_seed_next": 3,
            "trainer_config": {"total_timesteps": 200000},
        },
        checkpoints_dir / "latest.pt",
    )
    (logs_dir / "main_stdout.log").write_text(
        "\n".join(
            [
                "[ppo] rollout_done device=cuda seed=0 timesteps_done=512/200000 progress=0.26% elapsed_s=100.0 rollout_index=1 rollout_seed=0 batch_steps=512 checkpoint=latest.pt",
                "[ppo] rollout_done device=cuda seed=0 timesteps_done=1024/200000 progress=0.51% elapsed_s=200.0 rollout_index=2 rollout_seed=1 batch_steps=512 checkpoint=latest.pt",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "main_stderr.log").write_text("fallback triggered\n", encoding="utf-8")
    (logs_dir / "supervisor.log").write_text("checkpoint available; safe_to_shutdown=True\n", encoding="utf-8")

    exit_code = module.main(["--train-dir", str(train_dir), "--as-json"])
    assert exit_code == 0

    report = json.loads(capsys.readouterr().out)
    assert report["timesteps_done"] == 1024
    assert report["rollout_seed_next"] == 3
    assert report["safe_to_shutdown"] is True
    assert report["latest_rollout"]["rollout_index"] == 2
    assert report["estimated_remaining_s"] is not None
