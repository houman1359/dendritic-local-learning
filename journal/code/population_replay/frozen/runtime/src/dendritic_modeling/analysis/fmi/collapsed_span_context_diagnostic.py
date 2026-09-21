"""Causal sampled-prefix context diagnostic for collapsed-span captures.

The collapsed-span target need not be a function of the hidden state at one
sampled token.  This module tests the smallest useful alternative: whether a
running mean of *earlier sampled rows from the same source sequence* improves
prediction of the captured correction.  Three exactly capacity-matched ridge
probes are compared:

``token_only``
    Two fixed random views of the current token input.
``aligned_causal_prefix``
    One fixed view of the current token and one view of the causal sampled
    prefix mean.
``cross_group_shuffled_prefix``
    The same architecture as the aligned arm, but its prefix vector is taken
    from a different content-addressed source group by a deterministic,
    hash-fixed permutation.

Models are fit only on fit groups.  Gate-validation tensor values are not
indexed, and no endpoint is computed there.  All models and hyperparameters
are fixed before the untouched profile tensors are indexed; profile is the
sole endpoint.  Inference is paired at the source-group level and uses a
deterministic percentile bootstrap over source groups.

The v1 capture producer preserves sampled rows in canonical traversal order,
but a capture does not encode or verify the teacher's attention mask.  Callers
must therefore explicitly attest that within-group capture order is causal.
The report records this as an external assertion rather than a software-
verified fact.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.analysis.fmi.collapsed_span import tensor_content_sha256
from dendritic_modeling.analysis.fmi.collapsed_span_surrogate_zoo import (
    CachedSpanCapture,
    RidgeProbeConfig,
    SurrogateSplitConfig,
    _fit_ridge,
    _partition_manifest,
    canonical_json_sha256,
    reproduce_registered_partitions,
)
from dendritic_modeling.analysis.statistics import bootstrap_mean_interval, stable_seed

REPORT_SCHEMA = "dendritic_collapsed_span_causal_context_diagnostic/v1"
EVIDENCE_STATUS = "exploratory_profile_endpoint"
CAUSAL_ROW_ORDER_ATTESTATION = (
    "within_each_source_group_capture_rows_are_strictly_increasing_"
    "causal_sequence_positions"
)
CLAIM_BOUNDARY = (
    "Exploratory paired source-group diagnostic on a cached capture. It tests "
    "whether an aligned mean of earlier sampled inputs predicts the captured "
    "span correction better than capacity-matched token-only and cross-group "
    "shuffled-context controls. It does not establish that full token history, "
    "a particular sequence model, or a deployable contextual replacement is "
    "optimal. The software verifies row ordering, partition isolation, feature "
    "construction, and content identities; causal masking and causal-position "
    "semantics are externally attested and are not recoverable from capture v1."
)
PRIMARY_DELTA_R2_CONTRASTS = (
    "aligned_minus_token_delta_r2",
    "aligned_minus_shuffled_delta_r2",
)


@dataclass(frozen=True)
class CausalContextDiagnosticConfig:
    """Frozen controls for one causal-context diagnostic."""

    causal_row_order_attestation: str
    split: SurrogateSplitConfig = field(default_factory=SurrogateSplitConfig)
    projected_input_features: int = 256
    output_rank: int = 128
    ridge: float = 1.0e-3
    projection_seed: int = 40_701
    fit_seed: int = 40_709
    shuffle_seed: int = 40_721
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 40_729
    confidence: float = 0.95
    minimum_eligible_groups: int = 2
    evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.causal_row_order_attestation != CAUSAL_ROW_ORDER_ATTESTATION:
            raise ValueError(
                "causal_row_order_attestation must exactly equal the published "
                "CAUSAL_ROW_ORDER_ATTESTATION; capture v1 cannot infer causality"
            )
        if int(self.projected_input_features) < 2:
            raise ValueError("projected_input_features must be at least two")
        if int(self.output_rank) < 0:
            raise ValueError("output_rank must be nonnegative")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if int(self.bootstrap_draws) < 1:
            raise ValueError("bootstrap_draws must be positive")
        if not math.isclose(float(self.confidence), 0.95, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "confidence must equal 0.95 for the preregistered two-sided rule"
            )
        if int(self.minimum_eligible_groups) < 2:
            raise ValueError("minimum_eligible_groups must be at least two")
        if int(self.evaluation_batch_size) < 1:
            raise ValueError("evaluation_batch_size must be positive")


@dataclass(frozen=True)
class CausalPrefixFeatures:
    """Prefix means and their exact earlier-sampled-row counts."""

    means: torch.Tensor
    history_counts: torch.Tensor
    eligible_index: torch.Tensor
    eligible_groups: torch.Tensor
    excluded_groups: torch.Tensor


def _require_canonical_group_order(group_ids: torch.Tensor) -> None:
    """Fail if rows from one source group are interleaved in capture order."""

    if group_ids.ndim != 1 or group_ids.dtype != torch.int64:
        raise TypeError("group_ids must be a one-dimensional int64 tensor")
    for group in torch.unique(group_ids, sorted=True):
        index = torch.nonzero(group_ids == group, as_tuple=False).reshape(-1)
        expected = torch.arange(int(index[0]), int(index[-1]) + 1, dtype=torch.int64)
        if not torch.equal(index, expected):
            raise ValueError(
                "capture rows for a source group are interleaved; capture v1 "
                "does not carry absolute positions, so causal order is ambiguous"
            )


def causal_sampled_prefix_means(
    inputs: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    minimum_eligible_groups: int = 2,
) -> CausalPrefixFeatures:
    """Return means of strictly earlier sampled rows within each source group.

    The current row is added to the running sum only *after* its prefix is
    written.  A group's first sampled row has history count zero and is omitted
    from ``eligible_index`` rather than being assigned an ambiguous sentinel.
    This operation consumes inputs and group ids only; targets are not an
    argument and cannot enter feature construction.
    """

    values = inputs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    groups = group_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if values.ndim != 2 or tuple(groups.shape) != (int(values.shape[0]),):
        raise ValueError("inputs must be [rows, features] with one group id per row")
    _require_canonical_group_order(groups)
    means = torch.zeros_like(values)
    counts = torch.zeros(int(values.shape[0]), dtype=torch.int64)
    eligible_groups: list[int] = []
    excluded_groups: list[int] = []
    for group in torch.unique(groups, sorted=True):
        index = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
        if int(index.numel()) < 2:
            excluded_groups.append(int(group))
            continue
        eligible_groups.append(int(group))
        running = torch.zeros(int(values.shape[1]), dtype=torch.float32)
        for ordinal, row in enumerate(index.tolist()):
            if ordinal:
                means[row] = running / float(ordinal)
                counts[row] = int(ordinal)
            running.add_(values[row])
    if len(eligible_groups) < int(minimum_eligible_groups):
        raise ValueError(
            "too few source groups contain at least two sampled rows for a "
            "paired causal-prefix endpoint"
        )
    eligible = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
    return CausalPrefixFeatures(
        means=means,
        history_counts=counts,
        eligible_index=eligible,
        eligible_groups=torch.tensor(eligible_groups, dtype=torch.int64),
        excluded_groups=torch.tensor(excluded_groups, dtype=torch.int64),
    )


def _hash_order_key(*values: Any) -> bytes:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def hash_fixed_cross_group_permutation(
    group_ids: torch.Tensor,
    *,
    source_group_sha256: Mapping[int, str],
    seed: int,
    partition_label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Permute rows bijectively so every context comes from another group.

    Groups are ordered by their content hash and the declared seed.  Their
    contiguous row blocks are then circularly shifted by the largest group
    size.  This is a cross-group derangement whenever no group owns more than
    half the rows; otherwise the routine refuses to create a distribution-
    changing with-replacement control.
    """

    groups = group_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if groups.ndim != 1 or int(groups.numel()) < 4:
        raise ValueError("shuffle needs at least four eligible rows")
    unique = [int(value) for value in torch.unique(groups, sorted=True)]
    if len(unique) < 2:
        raise ValueError("cross-group shuffle needs at least two source groups")
    missing = set(unique) - {int(value) for value in source_group_sha256}
    if missing:
        raise ValueError(f"source hashes are missing groups {sorted(missing)}")
    group_order = sorted(
        unique,
        key=lambda group: _hash_order_key(
            int(seed), partition_label, source_group_sha256[group]
        ),
    )
    rows_by_group = {
        group: torch.nonzero(groups == group, as_tuple=False).reshape(-1).tolist()
        for group in group_order
    }
    maximum = max(len(rows) for rows in rows_by_group.values())
    total = int(groups.numel())
    if maximum * 2 > total:
        raise ValueError(
            "a source group owns more than half the eligible rows; no bijective "
            "cross-group shuffle exists"
        )
    destination_order = [row for group in group_order for row in rows_by_group[group]]
    donor_order = destination_order[maximum:] + destination_order[:maximum]
    permutation = torch.empty(total, dtype=torch.int64)
    for destination, donor in zip(destination_order, donor_order):
        permutation[destination] = int(donor)
    donor_groups = groups[permutation]
    if bool((donor_groups == groups).any()):
        raise RuntimeError("hash-fixed shuffle failed its cross-group invariant")
    if not torch.equal(permutation.sort().values, torch.arange(total)):
        raise RuntimeError("hash-fixed shuffle is not a row-level permutation")
    pair_counts: dict[tuple[int, int], int] = {}
    for recipient, donor in zip(groups.tolist(), donor_groups.tolist()):
        key = (int(recipient), int(donor))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    manifest = {
        "algorithm": "content_hash_group_blocks_circular_shift_by_max_group_size/v1",
        "seed": int(seed),
        "partition_label": str(partition_label),
        "rows": total,
        "groups": len(unique),
        "largest_group_rows": int(maximum),
        "bijective": True,
        "every_donor_group_differs_from_recipient": True,
        "permutation_sha256": tensor_content_sha256(permutation),
        "recipient_group_ids_sha256": tensor_content_sha256(groups),
        "donor_group_ids_sha256": tensor_content_sha256(donor_groups),
        "group_pair_counts": [
            {"recipient_group": recipient, "donor_group": donor, "rows": count}
            for (recipient, donor), count in sorted(pair_counts.items())
        ],
    }
    return permutation, manifest


def _fixed_projection(input_dim: int, output_dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        int(input_dim),
        int(output_dim),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(int(output_dim))


def _arm_features(
    token_inputs: torch.Tensor,
    aligned_context: torch.Tensor,
    shuffled_context: torch.Tensor,
    *,
    primary_projection: torch.Tensor,
    secondary_projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    primary = token_inputs.float() @ primary_projection
    return {
        "token_only": torch.cat(
            (primary, token_inputs.float() @ secondary_projection), dim=1
        ).contiguous(),
        "aligned_causal_prefix": torch.cat(
            (primary, aligned_context.float() @ secondary_projection), dim=1
        ).contiguous(),
        "cross_group_shuffled_prefix": torch.cat(
            (primary, shuffled_context.float() @ secondary_projection), dim=1
        ).contiguous(),
    }


def _module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_content_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _predict(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            stop = min(int(features.shape[0]), start + int(batch_size))
            predictions.append(
                model(features[start:stop].to(device=device, dtype=torch.float32))
                .float()
                .cpu()
            )
    return torch.cat(predictions, dim=0)


def _arm_sse(
    predictions: Mapping[str, torch.Tensor], targets: torch.Tensor
) -> dict[str, float]:
    return {
        arm: float((prediction.double() - targets.double()).square().sum())
        for arm, prediction in predictions.items()
    }


def preregistered_context_support_decision(
    paired_source_group_inference: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen conjunction without consulting gate-validation."""

    lower_bounds = {}
    for name in PRIMARY_DELTA_R2_CONTRASTS:
        summary = paired_source_group_inference.get(name)
        if not isinstance(summary, Mapping):
            raise ValueError(f"paired inference is missing primary contrast {name!r}")
        if not math.isclose(
            float(summary.get("confidence", float("nan"))),
            0.95,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"primary contrast {name!r} is not a 95% interval")
        lower = float(summary.get("lower", float("nan")))
        if not math.isfinite(lower):
            raise ValueError(f"primary contrast {name!r} has a non-finite lower bound")
        lower_bounds[name] = lower
    criteria = {name: lower > 0.0 for name, lower in lower_bounds.items()}
    contextual_support = all(criteria.values())
    return {
        "rule": (
            "Declare contextual support only when the lower bound of the "
            "two-sided 95% paired source-group bootstrap interval is strictly "
            "above zero for BOTH primary delta-R2 contrasts: aligned causal "
            "prefix minus token-only AND aligned causal prefix minus "
            "cross-group shuffled prefix."
        ),
        "selection_uses_gate_validation": False,
        "confidence": 0.95,
        "interval_sidedness": "two_sided",
        "conjunction_required": True,
        "lower_bounds": lower_bounds,
        "criteria_passed": criteria,
        "all_criteria_passed": contextual_support,
        "machine_verdict": (
            "contextual_support" if contextual_support else "no_contextual_support"
        ),
        "claim_language": (
            "The paired profile endpoint supports additional predictive "
            "information in aligned sampled-prefix context beyond both "
            "capacity-matched controls."
            if contextual_support
            else "The preregistered conjunction was not met; this diagnostic "
            "provides no support for additional predictive information from "
            "aligned sampled-prefix context."
        ),
        "scope_warning": (
            "Even a positive verdict is contextual predictive support, not proof "
            "of a causal mechanism, full-history sufficiency, or an optimal "
            "deployable replacement."
        ),
    }


def _paired_profile_endpoint(
    predictions: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    source_group_sha256: Mapping[int, str],
    bootstrap_draws: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    arms = tuple(predictions)
    if set(arms) != {
        "token_only",
        "aligned_causal_prefix",
        "cross_group_shuffled_prefix",
    }:
        raise ValueError("profile endpoint received an unexpected arm set")
    mean = target_mean.detach().to(device="cpu", dtype=torch.float64)
    y = targets.detach().to(device="cpu", dtype=torch.float64)
    groups = group_ids.detach().to(device="cpu", dtype=torch.int64)
    group_rows = []
    contrast_values: dict[str, list[float]] = {
        "aligned_minus_token_delta_r2": [],
        "aligned_minus_shuffled_delta_r2": [],
        "token_minus_aligned_sse": [],
        "shuffled_minus_aligned_sse": [],
        "aligned_relative_sse_reduction_vs_token": [],
        "aligned_relative_sse_reduction_vs_shuffled": [],
    }
    for group in torch.unique(groups, sorted=True):
        index = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
        group_y = y[index]
        sst = float((group_y - mean).square().sum())
        if not math.isfinite(sst) or sst <= 1.0e-12:
            raise ValueError("profile source-group target variance is degenerate")
        sse = _arm_sse(
            {arm: prediction[index] for arm, prediction in predictions.items()},
            group_y,
        )
        r2 = {arm: float(1.0 - value / sst) for arm, value in sse.items()}
        token_gain = sse["token_only"] - sse["aligned_causal_prefix"]
        shuffle_gain = sse["cross_group_shuffled_prefix"] - sse["aligned_causal_prefix"]
        values = {
            "aligned_minus_token_delta_r2": r2["aligned_causal_prefix"]
            - r2["token_only"],
            "aligned_minus_shuffled_delta_r2": r2["aligned_causal_prefix"]
            - r2["cross_group_shuffled_prefix"],
            "token_minus_aligned_sse": token_gain,
            "shuffled_minus_aligned_sse": shuffle_gain,
            "aligned_relative_sse_reduction_vs_token": token_gain
            / max(sse["token_only"], 1.0e-12),
            "aligned_relative_sse_reduction_vs_shuffled": shuffle_gain
            / max(sse["cross_group_shuffled_prefix"], 1.0e-12),
        }
        for key, value in values.items():
            contrast_values[key].append(float(value))
        group_rows.append(
            {
                "group_id": int(group),
                "source_sha256": source_group_sha256[int(group)],
                "eligible_rows": int(index.numel()),
                "sst_against_fit_target_mean": sst,
                "sse": sse,
                "r2_against_fit_target_mean": r2,
                "paired_contrasts": values,
            }
        )
    group_count = len(group_rows)
    if group_count < 2:
        raise ValueError("paired group inference needs at least two profile groups")
    inferential = {}
    for name, values in contrast_values.items():
        interval = bootstrap_mean_interval(
            values,
            draws=int(bootstrap_draws),
            seed=stable_seed(int(bootstrap_seed), REPORT_SCHEMA, name),
            confidence=float(confidence),
            require_n=group_count,
            finite_policy="raise",
        )
        inferential[name] = interval.to_dict()
    decision = preregistered_context_support_decision(inferential)
    aggregate_sst = float((y - mean).square().sum())
    aggregate_sse = _arm_sse(predictions, y)
    return {
        "experimental_unit": "content_addressed_source_group",
        "group_count": group_count,
        "eligible_rows": int(y.shape[0]),
        "target_coordinates_per_row": int(y.shape[1]),
        "aggregate_sst_against_fit_target_mean": aggregate_sst,
        "aggregate_sse": aggregate_sse,
        "aggregate_r2_against_fit_target_mean": {
            arm: float(1.0 - sse / max(aggregate_sst, 1.0e-12))
            for arm, sse in aggregate_sse.items()
        },
        "source_group_rows": group_rows,
        "paired_source_group_inference": inferential,
        "primary_contrasts": list(PRIMARY_DELTA_R2_CONTRASTS),
        "preregistered_conjunctive_decision": decision,
        "positive_direction": (
            "Positive means the aligned causal-prefix arm has lower SSE and "
            "higher R2 than its named comparator."
        ),
        "interval_method": (
            "Deterministic paired percentile bootstrap of the source-group "
            "macro mean; rows and output coordinates are not treated as "
            "independent replicates."
        ),
    }


def _prefix_manifest(
    features: CausalPrefixFeatures,
    *,
    partition_index: torch.Tensor,
    partition_group_ids: torch.Tensor,
) -> dict[str, Any]:
    eligible_global_index = partition_index[features.eligible_index]
    return {
        "rows_total": int(partition_index.numel()),
        "rows_with_strictly_earlier_sampled_history": int(
            features.eligible_index.numel()
        ),
        "eligible_groups": int(features.eligible_groups.numel()),
        "excluded_singleton_groups": int(features.excluded_groups.numel()),
        "eligible_group_ids_sha256": tensor_content_sha256(features.eligible_groups),
        "excluded_group_ids_sha256": tensor_content_sha256(features.excluded_groups),
        "eligible_local_indices_sha256": tensor_content_sha256(features.eligible_index),
        "eligible_capture_indices_sha256": tensor_content_sha256(eligible_global_index),
        "eligible_group_rows_sha256": tensor_content_sha256(
            partition_group_ids[features.eligible_index]
        ),
        "history_counts_sha256": tensor_content_sha256(
            features.history_counts[features.eligible_index]
        ),
        "minimum_history_count": int(
            features.history_counts[features.eligible_index].min()
        ),
        "maximum_history_count": int(
            features.history_counts[features.eligible_index].max()
        ),
    }


def diagnose_cached_span_causal_context(
    capture: CachedSpanCapture,
    *,
    config: CausalContextDiagnosticConfig,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the frozen three-arm diagnostic with profile as the sole endpoint."""

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    boundary = capture.boundary
    _require_canonical_group_order(boundary.row_group_ids)
    partitions = reproduce_registered_partitions(boundary, config.split)
    partition_manifest = _partition_manifest(boundary, partitions)
    partition_manifest["fit"]["values_used_for_model_fit"] = True
    partition_manifest["gate_validation"].update(
        {
            "values_used_for_model_fit_or_metrics": False,
            "reserved_role": "unused_in_this_single_frozen_diagnostic",
        }
    )
    partition_manifest["fingerprint_profile"].update(
        {
            "values_used_for_model_fit": False,
            "values_used_for_endpoint_only_after_all_models_fitted": True,
            "reserved_role": "sole_paired_context_diagnostic_endpoint",
        }
    )

    # FIT ONLY.  No gate or profile tensor values are indexed before every arm
    # below has been fitted and content-hashed.
    fit_index = partitions.fit_index
    fit_x_all = boundary.inputs[fit_index]
    fit_groups_all = boundary.row_group_ids[fit_index]
    fit_prefix = causal_sampled_prefix_means(
        fit_x_all,
        fit_groups_all,
        minimum_eligible_groups=config.minimum_eligible_groups,
    )
    fit_eligible = fit_prefix.eligible_index
    fit_x = fit_x_all[fit_eligible]
    fit_y = boundary.targets[fit_index[fit_eligible]]
    fit_groups = fit_groups_all[fit_eligible]
    fit_shuffle, fit_shuffle_manifest = hash_fixed_cross_group_permutation(
        fit_groups,
        source_group_sha256=boundary.source_group_sha256,
        seed=config.shuffle_seed,
        partition_label="fit",
    )
    fit_context = fit_prefix.means[fit_eligible]
    fit_shuffled_context = fit_context[fit_shuffle]
    total_features = int(config.projected_input_features)
    primary_features = total_features // 2
    secondary_features = total_features - primary_features
    primary_projection = _fixed_projection(
        int(boundary.inputs.shape[1]),
        primary_features,
        seed=config.projection_seed,
    )
    secondary_projection = _fixed_projection(
        int(boundary.inputs.shape[1]),
        secondary_features,
        seed=int(config.projection_seed) + 1,
    )
    fit_arm_features = _arm_features(
        fit_x,
        fit_context,
        fit_shuffled_context,
        primary_projection=primary_projection,
        secondary_projection=secondary_projection,
    )
    ridge_config = RidgeProbeConfig(
        input_features=total_features,
        output_rank=config.output_rank,
        ridge=config.ridge,
        seed=config.fit_seed,
    )
    models: dict[str, nn.Module] = {}
    model_details: dict[str, Any] = {}
    for arm in (
        "token_only",
        "aligned_causal_prefix",
        "cross_group_shuffled_prefix",
    ):
        model, details = _fit_ridge(
            fit_arm_features[arm],
            fit_y,
            config=ridge_config,
            device=target_device,
        )
        models[arm] = model
        model_details[arm] = {
            **details,
            "fitted_state_sha256": _module_state_sha256(model),
            "fit_rows": int(fit_y.shape[0]),
            "fit_groups": int(torch.unique(fit_groups).numel()),
        }
    accounting = {arm: model_details[arm]["accounting"] for arm in model_details}
    reference_accounting = accounting["token_only"]
    matched_keys = (
        "active_parameter_values",
        "trainable_parameter_values",
        "learned_parameter_values",
        "stored_values",
        "stored_bytes",
        "inference_macs_per_row",
    )
    if any(
        accounting[arm][key] != reference_accounting[key]
        for arm in accounting
        for key in matched_keys
    ):
        raise RuntimeError("the three fitted probes are not exactly capacity matched")
    fitted_state_manifest_sha256 = canonical_json_sha256(
        {arm: details["fitted_state_sha256"] for arm, details in model_details.items()}
    )

    # PROFILE SOLE ENDPOINT.  This is deliberately the first indexing of any
    # profile input or target tensor value in the diagnostic.
    profile_index = partitions.profile_index
    profile_x_all = boundary.inputs[profile_index]
    profile_groups_all = boundary.row_group_ids[profile_index]
    profile_prefix = causal_sampled_prefix_means(
        profile_x_all,
        profile_groups_all,
        minimum_eligible_groups=config.minimum_eligible_groups,
    )
    profile_eligible = profile_prefix.eligible_index
    profile_x = profile_x_all[profile_eligible]
    profile_y = boundary.targets[profile_index[profile_eligible]]
    profile_groups = profile_groups_all[profile_eligible]
    profile_shuffle, profile_shuffle_manifest = hash_fixed_cross_group_permutation(
        profile_groups,
        source_group_sha256=boundary.source_group_sha256,
        seed=config.shuffle_seed,
        partition_label="profile_endpoint",
    )
    profile_context = profile_prefix.means[profile_eligible]
    profile_arm_features = _arm_features(
        profile_x,
        profile_context,
        profile_context[profile_shuffle],
        primary_projection=primary_projection,
        secondary_projection=secondary_projection,
    )
    predictions = {
        arm: _predict(
            models[arm],
            profile_arm_features[arm],
            device=target_device,
            batch_size=config.evaluation_batch_size,
        )
        for arm in models
    }
    target_means = {
        tensor_content_sha256(model.target_mean) for model in models.values()
    }
    if len(target_means) != 1:
        raise RuntimeError("capacity-matched arms have different fit target means")
    output_bases = {
        (
            None
            if model.output_basis is None
            else tensor_content_sha256(model.output_basis)
        )
        for model in models.values()
    }
    if len(output_bases) != 1:
        raise RuntimeError("capacity-matched arms have different fitted output bases")
    endpoint = _paired_profile_endpoint(
        predictions,
        profile_y,
        profile_groups,
        target_mean=models["token_only"].target_mean,
        source_group_sha256=boundary.source_group_sha256,
        bootstrap_draws=config.bootstrap_draws,
        bootstrap_seed=config.bootstrap_seed,
        confidence=config.confidence,
    )

    feature_state_values = int(
        primary_projection.numel() + secondary_projection.numel()
    )
    report = {
        "schema": REPORT_SCHEMA,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "selection_policy": (
            "All three arms, feature widths, seeds, ridge strength, output rank, "
            "and contrasts are fixed in the configuration. Gate-validation is "
            "not inspected and profile is the sole endpoint. No arm is promoted "
            "to an FMI prescription by this diagnostic."
        ),
        "capture": {
            "path": capture.path,
            "bytes": int(capture.file_bytes),
            "file_sha256": capture.file_sha256,
            "semantic_content_sha256": boundary.content_sha256,
            "calibration_sha256": boundary.calibration_sha256,
            "span_key": boundary.spec.key,
            "rows": int(boundary.inputs.shape[0]),
            "input_dim": int(boundary.inputs.shape[1]),
            "output_dim": int(boundary.targets.shape[1]),
        },
        "config": asdict(config),
        "causal_semantics": {
            "external_attestation": config.causal_row_order_attestation,
            "external_attestation_sha256": canonical_json_sha256(
                {
                    "capture_semantic_content_sha256": boundary.content_sha256,
                    "assertion": config.causal_row_order_attestation,
                }
            ),
            "software_verified": [
                "rows belonging to each source group are contiguous",
                "each aligned prefix uses only lower capture-row indices from the same group",
                "the current row is excluded from its prefix mean",
                "fit and profile context construction never crosses partitions",
                "cross-group controls are bijective and every donor group differs",
                "targets are not an input to context feature construction",
            ],
            "not_software_verified_from_capture_v1": [
                "the originating model used a causal attention mask",
                "capture traversal order equals strictly increasing token position",
                "unsampled earlier tokens are represented by the sampled-prefix mean",
            ],
        },
        "partitions": partition_manifest,
        "fit_prefix": _prefix_manifest(
            fit_prefix,
            partition_index=fit_index,
            partition_group_ids=fit_groups_all,
        ),
        "profile_prefix": _prefix_manifest(
            profile_prefix,
            partition_index=profile_index,
            partition_group_ids=profile_groups_all,
        ),
        "negative_control_permutations": {
            "fit": fit_shuffle_manifest,
            "profile_endpoint": profile_shuffle_manifest,
            "gate_validation": "not_constructed_or_indexed",
        },
        "feature_extractor": {
            "schema": "capacity_matched_token_and_context_random_views/v1",
            "primary_token_features": primary_features,
            "secondary_features": secondary_features,
            "total_features_per_arm": total_features,
            "primary_projection_sha256": tensor_content_sha256(primary_projection),
            "secondary_projection_sha256": tensor_content_sha256(secondary_projection),
            "fixed_projection_values_per_arm": feature_state_values,
            "fixed_projection_bytes_per_arm": feature_state_values
            * int(primary_projection.element_size()),
            "projection_macs_per_row_per_arm": int(boundary.inputs.shape[1])
            * total_features,
            "context_preprocessing_boundary": (
                "The aligned arm additionally maintains an input-dimensional "
                "running sum and count per active sequence. This nonlearned state "
                "and arithmetic are reported separately and are not hidden in "
                "the equal surrogate parameter/MAC accounting."
            ),
        },
        "models": model_details,
        "capacity_match_verification": {
            "exact_for_fitted_model_accounting": True,
            "exact_for_fixed_projection_values_bytes_and_macs": True,
            "matched_fields": list(matched_keys),
            "shared_output_basis_and_fit_target_mean_verified": True,
            "fitted_state_manifest_sha256_before_profile_access": (
                fitted_state_manifest_sha256
            ),
        },
        "profile_endpoint": endpoint,
        "leakage_audit": {
            "targets_available_to_feature_constructor": False,
            "current_or_future_rows_in_aligned_prefix": False,
            "cross_source_context_in_aligned_arm": False,
            "cross_partition_context": False,
            "gate_tensor_values_indexed": False,
            "profile_tensor_values_first_indexed_after_all_models_fitted": True,
            "profile_is_sole_reported_endpoint": True,
        },
        "device": str(target_device),
        "report_content_sha256": None,
    }
    report["report_content_sha256"] = canonical_json_sha256(
        {**report, "report_content_sha256": None}
    )
    return report


__all__ = [
    "CAUSAL_ROW_ORDER_ATTESTATION",
    "CLAIM_BOUNDARY",
    "EVIDENCE_STATUS",
    "PRIMARY_DELTA_R2_CONTRASTS",
    "REPORT_SCHEMA",
    "CausalContextDiagnosticConfig",
    "CausalPrefixFeatures",
    "causal_sampled_prefix_means",
    "diagnose_cached_span_causal_context",
    "hash_fixed_cross_group_permutation",
    "preregistered_context_support_decision",
]
