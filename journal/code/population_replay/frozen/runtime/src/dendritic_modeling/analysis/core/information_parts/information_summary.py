"""Text summary formatting for information-analysis results."""

from __future__ import annotations

from typing import Any

_SUMMARY_METRIC_SECTIONS = (
    ("basic_mi", "Basic MI with C:", True),
    ("pairwise_mi", "\nPairwise MI:", False),
    ("conditional_mi", "\nConditional MI given C:", False),
    ("soma_coupling_mi", "\nBranch-to-owning-soma MI:", False),
    ("gaussian_fisher", "\nGaussian/Fisher proxy:", False),
)


def format_information_summary(results: dict[str, Any]) -> str:
    """Generate the legacy text summary for information-analysis results."""
    summary_lines = [
        "Information Analysis Summary",
        f"Method: {results['method']}",
        f"Samples: {results['n_samples']}",
        "",
    ]

    for category, title, skip_normalized in _SUMMARY_METRIC_SECTIONS:
        _append_metric_section(
            summary_lines,
            results,
            category=category,
            title=title,
            skip_normalized=skip_normalized,
        )

    _append_pid_section(summary_lines, results)
    return "\n".join(summary_lines)


def _append_metric_section(
    summary_lines: list[str],
    results: dict[str, Any],
    *,
    category: str,
    title: str,
    skip_normalized: bool,
) -> None:
    if category not in results:
        return

    summary_lines.append(title)
    for key, value in results[category].items():
        if skip_normalized and key.endswith("_normalized"):
            continue
        summary_lines.append(f"  {key}: {value:.4f}")


def _append_pid_section(summary_lines: list[str], results: dict[str, Any]) -> None:
    if "pid" not in results:
        return

    summary_lines.append("\nPID Results:")
    for pid_key, pid_values in results["pid"].items():
        summary_lines.append(f"  {pid_key}:")
        for metric, value in pid_values.items():
            summary_lines.append(f"    {metric}: {value:.4f}")


__all__ = ["format_information_summary"]
