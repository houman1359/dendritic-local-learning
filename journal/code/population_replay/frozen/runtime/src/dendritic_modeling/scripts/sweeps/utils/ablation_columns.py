"""Canonical ablation column naming, flattening, and parsing.

This module is the single source of truth for how ablation results are
represented as flat DataFrame columns in the sweep pipeline.

Internal ablation schema (canonical nested)::

    {
        "method": {                        # "lesion" | "shuffle" | "mean_clamp"
            "target": {                    # "all_synapses" | "excitation" | "inhibition" | "upstream"
                module_name: {
                    "depth": int,
                    "accuracy_drop": float,
                    "auc_drop": float,
                    ...
                }
            }
        }
    }

Flat sweep column format::

    ablation_{method}_{target}_{metric}_depth{N}

After seed aggregation the suffixes ``_mean`` / ``_std`` are appended.

Module-level sweep column format::

    ablation_module__{method}__{target}__{metric}__{module_slug}__depth{N}

The standard ``ablation_...`` columns are depth-aggregated summaries intended
for sweep plots. Module columns preserve per-module identity for more detailed
analysis and diagnostics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_METHODS = frozenset({"lesion", "shuffle", "mean_clamp"})
KNOWN_TARGETS = frozenset({"all_synapses", "excitation", "inhibition", "upstream"})
KNOWN_METRICS = frozenset(
    {
        "accuracy_drop",
        "auc_drop",
        "categorical_loglikelihood_drop",
        "mse_drop",
        "cosine_similarity_drop",
        "pred_label_mi_bits_drop",
    }
)

# Regex for parsing aggregated ablation columns.
# The target group is anchored to the known set so that multi-word methods
# (mean_clamp) and multi-word metrics (categorical_loglikelihood_drop) are
# handled correctly.
_ABLATION_COL_RE = re.compile(
    r"^ablation_"
    r"(?!module__)"  # reject module-level columns
    r"(?P<method>.+?)"
    r"_(?P<target>" + "|".join(sorted(KNOWN_TARGETS)) + r")"
    r"_(?P<metric>.+?)"
    r"_depth(?P<depth>\d+)"
    r"(?:_(?P<stat>mean|std))?$"
)

_ABLATION_MODULE_COL_RE = re.compile(
    r"^ablation_module__"
    r"(?P<method>.+?)"
    r"__(?P<target>" + "|".join(sorted(KNOWN_TARGETS)) + r")"
    r"__(?P<metric>.+?)"
    r"__(?P<module>[a-z0-9_]+)"
    r"__depth(?P<depth>\d+)"
    r"(?:_(?P<stat>mean|std))?$"
)


# ---------------------------------------------------------------------------
# Metric key normalization
# ---------------------------------------------------------------------------


def normalize_metric_key(raw_key: str) -> str:
    """Normalize a raw metric key to its canonical form.

    ``"accuracy drop"`` -> ``"accuracy_drop"``
    ``"categorical_loglikelihood drop"`` -> ``"categorical_loglikelihood_drop"``
    ``"accuracy"`` -> ``"accuracy_drop"`` (adds _drop if missing)
    """
    key = str(raw_key).strip().replace(" ", "_").replace("-", "_")
    if not key.endswith("_drop"):
        key = f"{key}_drop"
    return key


def sanitize_module_key(raw_key: str) -> str:
    """Normalize a module name into a stable, filename-safe slug."""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw_key).strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return key or "module"


def _iter_ablation_entries(nested: dict):
    """Yield canonical ablation entries from nested results."""
    for method, targets in nested.items():
        if not isinstance(targets, dict):
            continue
        for target, modules in targets.items():
            if target not in KNOWN_TARGETS or not isinstance(modules, dict):
                continue
            for module_name, metrics in modules.items():
                if not isinstance(metrics, dict):
                    continue
                depth = int(metrics.get("depth", 0))
                module_slug = sanitize_module_key(module_name)
                for metric_key, value in metrics.items():
                    if metric_key in {"depth", "layer_name", "ablation_method"}:
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    yield (
                        method,
                        target,
                        module_name,
                        module_slug,
                        depth,
                        normalize_metric_key(metric_key),
                        float(value),
                    )


# ---------------------------------------------------------------------------
# Flatten nested ablation results to sweep columns
# ---------------------------------------------------------------------------


def flatten_ablation_results(nested: dict) -> dict[str, float]:
    """Flatten canonical nested ablation results to sweep-ready flat columns.

    Input schema::

        {
            "lesion": {
                "excitation": {
                    "module_name": {"depth": 0, "accuracy_drop": 0.05, ...}
                }
            }
        }

    Output::

        {
            "ablation_lesion_excitation_accuracy_drop_depth0": 0.05,
            ...
        }
    """
    # Multiple modules may share a depth. Export the explicit mean over those
    # modules rather than silently overwriting one with another.
    buckets: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for (
        method,
        target,
        _module_name,
        _module_slug,
        depth,
        metric_key,
        value,
    ) in _iter_ablation_entries(nested):
        buckets[(method, target, metric_key, depth)].append(value)

    flat: dict[str, float] = {}
    for (method, target, metric_key, depth), values in buckets.items():
        col = f"ablation_{method}_{target}_{metric_key}_depth{depth}"
        flat[col] = float(sum(values) / len(values))
    return flat


def flatten_ablation_module_results(nested: dict) -> dict[str, float]:
    """Flatten canonical nested ablation results to module-preserving columns."""
    flat: dict[str, float] = {}
    for (
        method,
        target,
        _module_name,
        module_slug,
        depth,
        metric_key,
        value,
    ) in _iter_ablation_entries(nested):
        col = (
            "ablation_module__"
            f"{method}__{target}__{metric_key}__{module_slug}__depth{depth}"
        )
        flat[col] = value
    return flat


# ---------------------------------------------------------------------------
# Parse flat ablation column names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedAblationColumn:
    """Parsed components of an ablation sweep column name."""

    method: str
    target: str
    metric: str
    depth: int
    stat: str | None = None  # "mean", "std", or None (raw)

    @property
    def base_col(self) -> str:
        """Column name without stat suffix."""
        return f"ablation_{self.method}_{self.target}_{self.metric}_depth{self.depth}"

    @property
    def full_col(self) -> str:
        """Full column name including stat suffix if present."""
        base = self.base_col
        if self.stat:
            return f"{base}_{self.stat}"
        return base


@dataclass(frozen=True)
class ParsedAblationModuleColumn:
    """Parsed components of a module-level ablation sweep column name."""

    method: str
    target: str
    metric: str
    module: str
    depth: int
    stat: str | None = None

    @property
    def base_col(self) -> str:
        return (
            "ablation_module__"
            f"{self.method}__{self.target}__{self.metric}__{self.module}__depth{self.depth}"
        )

    @property
    def full_col(self) -> str:
        base = self.base_col
        if self.stat:
            return f"{base}_{self.stat}"
        return base


def parse_ablation_column(col: str) -> ParsedAblationColumn | None:
    """Parse an ablation column name into its components.

    Returns ``None`` if the column does not match the canonical pattern.

    Examples::

        >>> parse_ablation_column("ablation_lesion_excitation_accuracy_drop_depth0_mean")
        ParsedAblationColumn(method='lesion', target='excitation',
                             metric='accuracy_drop', depth=0, stat='mean')

        >>> parse_ablation_column("ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2_std")
        ParsedAblationColumn(method='mean_clamp', target='upstream',
                             metric='pred_label_mi_bits_drop', depth=2, stat='std')
    """
    m = _ABLATION_COL_RE.match(col)
    if m is None:
        return None
    return ParsedAblationColumn(
        method=m.group("method"),
        target=m.group("target"),
        metric=m.group("metric"),
        depth=int(m.group("depth")),
        stat=m.group("stat"),
    )


def parse_ablation_module_column(col: str) -> ParsedAblationModuleColumn | None:
    """Parse a module-level ablation column name into its components."""
    m = _ABLATION_MODULE_COL_RE.match(col)
    if m is None:
        return None
    return ParsedAblationModuleColumn(
        method=m.group("method"),
        target=m.group("target"),
        metric=m.group("metric"),
        module=m.group("module"),
        depth=int(m.group("depth")),
        stat=m.group("stat"),
    )


def find_ablation_columns(
    columns: list[str],
    *,
    stat: str | None = "mean",
) -> list[ParsedAblationColumn]:
    """Find and parse all ablation columns, optionally filtered by stat suffix.

    Args:
        columns: DataFrame column names.
        stat: If not None, only return columns matching this stat suffix.
              Pass None to return all parsed ablation columns.
    """
    results = []
    for col in columns:
        parsed = parse_ablation_column(col)
        if parsed is None:
            continue
        if stat is not None and parsed.stat != stat:
            continue
        results.append(parsed)
    return results


def find_ablation_module_columns(
    columns: list[str],
    *,
    stat: str | None = "mean",
) -> list[ParsedAblationModuleColumn]:
    """Find and parse all module-level ablation columns."""
    results = []
    for col in columns:
        parsed = parse_ablation_module_column(col)
        if parsed is None:
            continue
        if stat is not None and parsed.stat != stat:
            continue
        results.append(parsed)
    return results


def discover_ablation_dimensions(
    columns: list[str],
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Discover all (methods, targets, metrics, depths) from ablation columns.

    Returns four sorted lists. Only considers ``_mean`` columns to avoid
    double-counting.
    """
    parsed = find_ablation_columns(columns, stat="mean")
    methods: set[str] = set()
    targets: set[str] = set()
    metrics: set[str] = set()
    depths: set[int] = set()
    for p in parsed:
        methods.add(p.method)
        targets.add(p.target)
        metrics.add(p.metric)
        depths.add(p.depth)
    return (
        sorted(methods),
        sorted(targets),
        sorted(metrics),
        sorted(depths),
    )


def discover_ablation_module_dimensions(
    columns: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[int]]:
    """Discover all (methods, targets, metrics, modules, depths) from module columns."""
    parsed = find_ablation_module_columns(columns, stat="mean")
    methods: set[str] = set()
    targets: set[str] = set()
    metrics: set[str] = set()
    modules: set[str] = set()
    depths: set[int] = set()
    for p in parsed:
        methods.add(p.method)
        targets.add(p.target)
        metrics.add(p.metric)
        modules.add(p.module)
        depths.add(p.depth)
    return (
        sorted(methods),
        sorted(targets),
        sorted(metrics),
        sorted(modules),
        sorted(depths),
    )
