"""Builders for signal forecasting and dynamical sequence datasets."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    build_deferred_train_test_split,
    deferred_split,
    test_size,
)
from dendritic_modeling.datasets.sequence.signal_tasks import (
    LorenzSequenceDataset,
    MultiFrequencyDataset,
    MultiSineForecastDataset,
    SwitchingLDSDataset,
)


def build_multi_frequency(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 200),
        "n_frequencies": kwargs.get("n_frequencies", 5),
        "base_freq": kwargs.get("base_freq", 0.01),
        "noise_std": kwargs.get("noise_std", 0.1),
    }
    return build_deferred_train_test_split(
        dataset_cls=MultiFrequencyDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=1000,
    )


def build_multi_sine_forecast(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 12000)
    common = {
        "seq_len": kwargs.get("seq_len", 160),
        "rollout_horizon": kwargs.get("rollout_horizon", 160),
        "prediction_mode": kwargs.get("prediction_mode", "all"),
        "n_components": kwargs.get("n_components", 3),
        "frequency_bank": kwargs.get("frequency_bank", [1.0, 2.5, 5.5, 9.0, 14.0]),
        "amplitude_min": kwargs.get("amplitude_min", 0.3),
        "amplitude_max": kwargs.get("amplitude_max", 1.0),
        "envelope_scale": kwargs.get("envelope_scale", 0.15),
        "envelope_freq_min": kwargs.get("envelope_freq_min", 0.15),
        "envelope_freq_max": kwargs.get("envelope_freq_max", 0.6),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.02),
        "seed": kwargs.get("seed", 42),
        "normalize_signal": kwargs.get("normalize_signal", True),
    }
    return deferred_split(
        MultiSineForecastDataset(n_samples=n_train, **common),
        MultiSineForecastDataset(
            n_samples=test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


def build_lorenz_sequence(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 50),
        "dt": kwargs.get("dt", 0.01),
        "sigma": kwargs.get("sigma", 10.0),
        "rho": kwargs.get("rho", 28.0),
        "beta": kwargs.get("beta", 8.0 / 3.0),
        "rollout_horizon": kwargs.get("rollout_horizon", 50),
        "prediction_mode": kwargs.get("prediction_mode", "last"),
        "observation_mode": kwargs.get("observation_mode", "full"),
        "observation_dim": kwargs.get("observation_dim", 3),
        "observation_seed": kwargs.get("observation_seed", kwargs.get("seed", 42)),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.0),
        "burn_in_steps": kwargs.get("burn_in_steps", 200),
        "seed": kwargs.get("seed", 42),
        "integration_method": kwargs.get("integration_method", "euler"),
        "normalize_states": kwargs.get("normalize_states", True),
        "normalize_observations": kwargs.get("normalize_observations", True),
        "normalization_steps": kwargs.get("normalization_steps", 10000),
        "initial_state_scale": kwargs.get("initial_state_scale", 0.1),
    }
    return deferred_split(
        LorenzSequenceDataset(n_samples=n_train, **common),
        LorenzSequenceDataset(
            n_samples=test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


def build_switching_lds(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 12000)
    common = {
        "seq_len": kwargs.get("seq_len", 96),
        "latent_dim": kwargs.get("latent_dim", 4),
        "observation_dim": kwargs.get("observation_dim", 2),
        "n_modes": kwargs.get("n_modes", 3),
        "rollout_horizon": kwargs.get("rollout_horizon", 80),
        "prediction_mode": kwargs.get("prediction_mode", "all"),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.03),
        "transition_noise_std": kwargs.get("transition_noise_std", 0.02),
        "min_mode_duration": kwargs.get("min_mode_duration", 36),
        "max_mode_duration": kwargs.get("max_mode_duration", 96),
        "burn_in_steps": kwargs.get("burn_in_steps", 24),
        "seed": kwargs.get("seed", 42),
        "normalize_states": kwargs.get("normalize_states", True),
        "normalize_observations": kwargs.get("normalize_observations", True),
        "normalization_samples": kwargs.get("normalization_samples", 4096),
        "stats_seed": kwargs.get("stats_seed", 1234),
        "initial_state_scale": kwargs.get("initial_state_scale", 0.35),
    }
    return deferred_split(
        SwitchingLDSDataset(n_samples=n_train, **common),
        SwitchingLDSDataset(
            n_samples=test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


SIGNAL_SEQUENCE_BUILDERS = {
    "lorenz": build_lorenz_sequence,
    "lorenz_sequence": build_lorenz_sequence,
    "multi_frequency": build_multi_frequency,
    "multi_sine_forecast": build_multi_sine_forecast,
    "switching_lds": build_switching_lds,
}


__all__ = [
    "SIGNAL_SEQUENCE_BUILDERS",
    "build_lorenz_sequence",
    "build_multi_frequency",
    "build_multi_sine_forecast",
    "build_switching_lds",
]
