"""Analyze S5 `rl_param_only` outputs against a white-box reference run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.analysis.rl_attribution import (  # noqa: E402
    aggregate_seed_rows,
    compare_to_reference_bundle,
    summarize_rl_seed,
)
from apflf.analysis.stats import read_summary_csv, write_csv_rows  # noqa: E402
from apflf.sim.replay import load_replay_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze stage-1 RL attribution on replayable S5 outputs.")
    parser.add_argument("--rl-run-dir", required=True, help="Run directory for the RL benchmark output.")
    parser.add_argument(
        "--reference-run-dir",
        default="",
        help="Optional white-box reference run directory, typically the matching `no_rl` output.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed subset. Defaults to all seeds found in the RL summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional destination directory. Defaults to `<rl-run-dir>/analysis/rl_attribution`.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the aggregate attribution report as JSON.",
    )
    return parser


def _summary_rows_by_seed(run_dir: Path) -> dict[int, dict[str, object]]:
    rows = read_summary_csv(run_dir / "summary.csv")
    return {int(row["seed"]): row for row in rows}


def _plot_overview(seed_rows: list[dict[str, object]], output_path: Path) -> None:
    if not seed_rows:
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.5, "No attribution rows available.", ha="center", va="center")
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return

    seeds = [int(row["seed"]) for row in seed_rows]
    rl_fallback_ratio = [float(row["rl_fallback_ratio"]) for row in seed_rows]
    theta_change_ratio = [float(row["theta_change_ratio"]) for row in seed_rows]
    safety_intervention_ratio = [float(row["safety_intervention_ratio"]) for row in seed_rows]
    leader_final_x_delta = [float(row.get("leader_final_x_delta", 0.0)) for row in seed_rows]
    target_speed_delta = [float(row.get("leader_target_speed_delta_abs_mean", 0.0)) for row in seed_rows]

    x_positions = np.arange(len(seeds), dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    width = 0.24
    axes[0].bar(x_positions - width, rl_fallback_ratio, width=width, label="rl fallback ratio")
    axes[0].bar(x_positions, theta_change_ratio, width=width, label="theta change ratio")
    axes[0].bar(
        x_positions + width,
        safety_intervention_ratio,
        width=width,
        label="safety intervention ratio",
    )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([str(seed) for seed in seeds])
    axes[0].set_ylabel("Ratio")
    axes[0].set_title("RL Supervisor Attribution by Seed")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(seeds, leader_final_x_delta, marker="o", label="leader_final_x delta")
    axes[1].plot(seeds, target_speed_delta, marker="s", label="target speed delta |mean|")
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Delta")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rl_run_dir = Path(args.rl_run_dir).resolve()
    reference_run_dir = Path(args.reference_run_dir).resolve() if args.reference_run_dir else None
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (rl_run_dir / "analysis" / "rl_attribution").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rl_summary_by_seed = _summary_rows_by_seed(rl_run_dir)
    reference_summary_by_seed = (
        _summary_rows_by_seed(reference_run_dir) if reference_run_dir is not None else {}
    )
    seeds = sorted(rl_summary_by_seed) if args.seeds is None else sorted(args.seeds)

    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        if seed not in rl_summary_by_seed:
            continue
        rl_bundle = load_replay_bundle(rl_run_dir, seed)
        seed_row: dict[str, object] = {"seed": seed}
        seed_row.update(summarize_rl_seed(rl_bundle))
        rl_summary = rl_summary_by_seed[seed]
        seed_row["leader_final_x"] = float(rl_summary["leader_final_x"])
        seed_row["fallback_events"] = int(rl_summary["fallback_events"])
        seed_row["safety_interventions"] = int(rl_summary["safety_interventions"])
        if reference_run_dir is not None and seed in reference_summary_by_seed:
            reference_bundle = load_replay_bundle(reference_run_dir, seed)
            reference_summary = reference_summary_by_seed[seed]
            seed_row.update(compare_to_reference_bundle(rl_bundle, reference_bundle))
            seed_row["reference_leader_final_x"] = float(reference_summary["leader_final_x"])
            seed_row["leader_final_x_delta"] = float(rl_summary["leader_final_x"]) - float(
                reference_summary["leader_final_x"]
            )
            seed_row["fallback_events_delta"] = int(rl_summary["fallback_events"]) - int(
                reference_summary["fallback_events"]
            )
            seed_row["safety_interventions_delta"] = int(rl_summary["safety_interventions"]) - int(
                reference_summary["safety_interventions"]
            )
        seed_rows.append(seed_row)

    aggregate = aggregate_seed_rows(seed_rows)
    write_csv_rows(seed_rows, output_dir / "seed_attribution.csv")
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot_overview(seed_rows, output_dir / "attribution_overview.pdf")

    if args.as_json:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
