"""Builders for neuroscience-inspired sequence datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    build_explicit_or_deferred_split,
    deferred_split,
    positive_input_encoding,
)
from dendritic_modeling.datasets.sequence.neuro_tasks import (
    GainModulatedContextualComparisonDataset,
    OculomotorDelayedResponseDataset,
    RomoDelayComparisonDataset,
    StringerV1Dataset,
    _resolve_stringer_session_file,
)


def build_odr_generalization(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "n_targets": kwargs.get("n_targets", 8),
        "input_tuning_dim": kwargs.get("input_tuning_dim", 16),
        "fixation_duration": kwargs.get("fixation_duration", 6),
        "cue_duration": kwargs.get("cue_duration", 8),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "go_duration": kwargs.get("go_duration", 8),
        "tuning_kappa": kwargs.get("tuning_kappa", 2.0),
        "cue_noise_std": kwargs.get("cue_noise_std", 0.05),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_explicit_or_deferred_split(
        dataset_cls=OculomotorDelayedResponseDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def build_romo_delay_comparison(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "frequency_values": tuple(
            kwargs.get("frequency_values", (2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        ),
        "stimulus_duration": kwargs.get("stimulus_duration", 20),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "response_duration": kwargs.get("response_duration", 8),
        "noise_std": kwargs.get("noise_std", 0.08),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
        "tuning_dim": kwargs.get("tuning_dim", 8),
        "tuning_sigma": kwargs.get("tuning_sigma", None),
        "pair_sampling": kwargs.get("pair_sampling", "independent_unequal_v1"),
    }
    return build_explicit_or_deferred_split(
        dataset_cls=RomoDelayComparisonDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def build_gain_modulated_contextual_comparison(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 24000)
    common_defaults = {
        "n_pathways": kwargs.get("n_pathways", 4),
        "pathway_dim": kwargs.get("pathway_dim", 4),
        "cue_duration": kwargs.get("cue_duration", 8),
        "stimulus_duration": kwargs.get("stimulus_duration", 16),
        "min_delay": kwargs.get("min_delay", 20),
        "max_delay": kwargs.get("max_delay", 80),
        "response_duration": kwargs.get("response_duration", 8),
        "amplitude_values": tuple(
            kwargs.get(
                "amplitude_values",
                (0.45, 0.6, 0.75, 0.9, 1.05, 1.2),
            )
        ),
        "comparison_margin": kwargs.get("comparison_margin", 0.15),
        "global_gain_min": kwargs.get("global_gain_min", 0.85),
        "global_gain_max": kwargs.get("global_gain_max", 1.15),
        "pathway_gain_min": kwargs.get("pathway_gain_min", 0.8),
        "pathway_gain_max": kwargs.get("pathway_gain_max", 1.2),
        "gain_drift_std": kwargs.get("gain_drift_std", 0.03),
        "noise_std": kwargs.get("noise_std", 0.08),
        "cue_noise_std": kwargs.get("cue_noise_std", 0.02),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 1.5),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 4),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_explicit_or_deferred_split(
        dataset_cls=GainModulatedContextualComparisonDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def build_stringer_v1(
    _dataset_name: str,
    data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    data_dir = kwargs.get("data_dir", data_path)
    if data_dir is None:
        raise ValueError(
            "stringer_v1 requires data_dir (path to directory with "
            "gratings_drifting_GT3_2019_04_05_1.npy)"
        )
    n_neurons = kwargs.get("n_neurons", 500)
    neuron_selection_scope = kwargs.get("neuron_selection_scope", "all_trials")
    n_classes = kwargs.get("n_classes", 8)
    normalize = kwargs.get("normalize", True)
    seed = kwargs.get("label_noise_seed", 42)
    session_id = kwargs.get("session_id", None)
    if session_id is not None:
        grating_file, session_idx = _resolve_stringer_session_file(
            data_dir, str(session_id)
        )
    else:
        session_idx = kwargs.get("session_idx", 20)
        grating_file = kwargs.get("grating_file", StringerV1Dataset._GRATING_FILE)
    running_quantile = kwargs.get("running_quantile", 0.5)
    running_state = kwargs.get("running_state", "all")
    train_running_state = kwargs.get("train_running_state", running_state)
    test_running_state = kwargs.get("test_running_state", running_state)
    state_holdout_fraction = float(kwargs.get("state_holdout_fraction", 0.0))
    if not (0.0 <= state_holdout_fraction < 1.0):
        raise ValueError(
            f"state_holdout_fraction must be in [0, 1), got {state_holdout_fraction}"
        )
    gain_group_count = kwargs.get("gain_group_count", 1)
    gain_grouping = kwargs.get("gain_grouping", "preferred_class")
    poisson_sampling = kwargs.get("poisson_sampling", False)
    poisson_rate_scale = kwargs.get("poisson_rate_scale", 1.0)
    running_residualize = kwargs.get("running_residualize", False)
    running_residualize_clip = kwargs.get("running_residualize_clip", True)
    gain_seed = kwargs.get("gain_seed", seed)

    def state_kwargs(prefix: str, state: str) -> dict[str, Any]:
        return {
            "running_state": state,
            "running_quantile": running_quantile,
            "gain_group_count": kwargs.get(
                f"{prefix}gain_group_count", gain_group_count
            ),
            "gain_grouping": kwargs.get(f"{prefix}gain_grouping", gain_grouping),
            "global_gain_min": kwargs.get(
                f"{prefix}global_gain_min", kwargs.get("global_gain_min", 1.0)
            ),
            "global_gain_max": kwargs.get(
                f"{prefix}global_gain_max", kwargs.get("global_gain_max", 1.0)
            ),
            "group_gain_min": kwargs.get(
                f"{prefix}group_gain_min", kwargs.get("group_gain_min", 1.0)
            ),
            "group_gain_max": kwargs.get(
                f"{prefix}group_gain_max", kwargs.get("group_gain_max", 1.0)
            ),
            "gain_seed": kwargs.get(f"{prefix}gain_seed", gain_seed),
            "poisson_sampling": kwargs.get(
                f"{prefix}poisson_sampling", poisson_sampling
            ),
            "poisson_rate_scale": kwargs.get(
                f"{prefix}poisson_rate_scale", poisson_rate_scale
            ),
            "running_residualize": kwargs.get(
                f"{prefix}running_residualize", running_residualize
            ),
            "running_residualize_clip": kwargs.get(
                f"{prefix}running_residualize_clip", running_residualize_clip
            ),
        }

    common = {
        "data_dir": data_dir,
        "n_neurons": n_neurons,
        "neuron_selection_scope": neuron_selection_scope,
        "n_classes": n_classes,
        "normalize": normalize,
        "seed": seed,
        "session_idx": session_idx,
        "grating_file": grating_file,
    }

    use_state_specific_split = (
        train_running_state != "all" or test_running_state != "all"
    )
    if use_state_specific_split:

        def state_set(state: str) -> set[str]:
            return {"low", "high"} if state == "all" else {state}

        if state_set(train_running_state) & state_set(test_running_state):
            raise ValueError(
                "Stringer V1 state-specific train/test requests must be "
                "disjoint to avoid trial overlap. Use train_running_state='low' "
                "with test_running_state='high' (or the reverse), or set both "
                "states to 'all' for the standard fixed split."
            )
        # Optionally reserve a deterministic same-state endpoint that is not
        # passed to the training/validation splitter.  This supports a genuine
        # mechanism-by-state interaction: the frozen model can be evaluated on
        # both an untouched training-state holdout and the disjoint target
        # state.  The complementary holdout is reconstructed with
        # ``split='test'`` and the same ``1-state_holdout_fraction`` fraction.
        state_train_fraction = 1.0 - state_holdout_fraction
        train_full = StringerV1Dataset(
            split="train" if state_holdout_fraction > 0.0 else "all",
            train_fraction=state_train_fraction,
            **common,
            **state_kwargs("train_", train_running_state),
        )
        test_selection_indices = (
            train_full._neuron_indices
            if neuron_selection_scope == "train_split"
            else None
        )
        test_ds = StringerV1Dataset(
            split="all",
            **common,
            **state_kwargs("test_", test_running_state),
            normalization_stats=getattr(train_full, "normalization_stats", None),
            neuron_selection_indices=test_selection_indices,
        )
        if test_selection_indices is not None and not np.array_equal(
            train_full._neuron_indices, test_ds._neuron_indices
        ):
            raise RuntimeError(
                "Stringer V1 state-specific test set did not preserve the "
                "training-state neuron selection"
            )
        train_full.state_holdout_fraction = state_holdout_fraction
        train_full.state_train_fraction = state_train_fraction
        test_ds.state_holdout_fraction = state_holdout_fraction
        test_ds.state_train_fraction = state_train_fraction
        # Name-level disjointness (above) is necessary but not sufficient: the
        # low/high running-speed masks are both boundary-inclusive, so at the
        # default running_quantile=0.5 (or with ties in a zero-inflated running
        # trace) the realized "low" and "high" trial sets can still overlap.
        # Enforce disjointness on the actual selected trial indices.
        train_idx = set(np.asarray(train_full._split_indices).ravel().tolist())
        test_idx = set(np.asarray(test_ds._split_indices).ravel().tolist())
        overlap = train_idx & test_idx
        if overlap:
            raise ValueError(
                f"Stringer V1 state-specific split produced {len(overlap)} trial(s) "
                f"in both train (running_state={train_running_state!r}) and test "
                f"(running_state={test_running_state!r}). The running-speed thresholds "
                f"are inclusive, so trials tied at the quantile boundary fall into both "
                f"bands. Use a running_quantile < 0.5 so the low/high bands are strictly "
                f"separated (e.g. running_quantile=0.25 for bottom/top quartiles)."
            )
        return deferred_split(train_full, test_ds)

    # Fixed 90/10 train/test split; train_valid_split handled by factory below.
    test_fraction = 0.1
    train_full = StringerV1Dataset(
        split="train",
        train_fraction=1.0 - test_fraction,
        **common,
        **state_kwargs("train_", train_running_state),
    )
    test_ds = StringerV1Dataset(
        split="test",
        train_fraction=1.0 - test_fraction,
        **common,
        **state_kwargs("test_", test_running_state),
    )
    return deferred_split(train_full, test_ds)


NEURO_SEQUENCE_BUILDERS = {
    "gain_modulated_contextual_comparison": (
        build_gain_modulated_contextual_comparison
    ),
    "gain_modulated_contextual_comparison_positive": (
        build_gain_modulated_contextual_comparison
    ),
    "odr_generalization": build_odr_generalization,
    "odr_generalization_positive": build_odr_generalization,
    "romo_delay_comparison": build_romo_delay_comparison,
    "romo_delay_comparison_positive": build_romo_delay_comparison,
    "stringer_v1": build_stringer_v1,
}


__all__ = [
    "NEURO_SEQUENCE_BUILDERS",
    "build_gain_modulated_contextual_comparison",
    "build_odr_generalization",
    "build_romo_delay_comparison",
    "build_stringer_v1",
]
