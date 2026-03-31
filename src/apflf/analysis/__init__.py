"""Analysis package exports."""

from __future__ import annotations

from apflf.analysis.stats import (
    DEFAULT_METRICS,
    aggregate_metric,
    pairwise_compare_to_reference,
    read_summary_csv,
    summarize_experiments,
    write_csv_rows,
)


def export_paper_artifacts(*args, **kwargs):
    """Lazy-export the paper artifact builder to avoid replay/export import cycles."""

    from apflf.analysis.export import export_paper_artifacts as _export_paper_artifacts

    return _export_paper_artifacts(*args, **kwargs)

__all__ = [
    "DEFAULT_METRICS",
    "aggregate_metric",
    "export_paper_artifacts",
    "pairwise_compare_to_reference",
    "read_summary_csv",
    "summarize_experiments",
    "write_csv_rows",
]
