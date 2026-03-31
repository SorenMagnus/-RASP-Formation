"""Re-export paper figures and tables from an existing reproduce output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.analysis.export import export_paper_artifacts  # noqa: E402
from apflf.analysis.stats import (  # noqa: E402
    DEFAULT_METRICS,
    pairwise_compare_to_reference,
    read_summary_csv,
    summarize_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper tables/figures from an existing output dir.")
    parser.add_argument("--input-dir", required=True, help="Directory produced by scripts/reproduce_paper.py.")
    parser.add_argument(
        "--reference-method",
        default="no_rl",
        help="Reference method used for pairwise comparison tables.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    raw_rows = read_summary_csv(input_dir / "all_runs.csv")
    summary_rows = summarize_experiments(
        raw_rows,
        group_keys=("scenario", "method"),
        metrics=DEFAULT_METRICS,
    )
    comparison_rows = pairwise_compare_to_reference(
        raw_rows,
        reference_method=args.reference_method,
        metrics=DEFAULT_METRICS,
    )
    export_paper_artifacts(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        output_dir=input_dir,
    )
    print(input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
