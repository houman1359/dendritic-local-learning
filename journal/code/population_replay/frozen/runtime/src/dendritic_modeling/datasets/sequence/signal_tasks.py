"""Sequence dataset task implementations."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultiFrequencyDataset(Dataset):
    """Multi-frequency classification task.

    Input is a mixture of sinusoids at different frequencies.
    Target: class index based on dominant frequency.

    Args:
        n_samples: Number of samples.
        seq_len: Number of timesteps.
        n_frequencies: Number of frequency classes.
        base_freq: Base frequency (Hz).
        noise_std: Standard deviation of additive noise.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 200,
        n_frequencies: int = 5,
        base_freq: float = 0.01,
        noise_std: float = 0.1,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_frequencies = n_frequencies

        t = torch.linspace(0, 1, seq_len).unsqueeze(0)  # [1, seq_len]
        freqs = [(i + 1) * base_freq for i in range(n_frequencies)]

        self.inputs = torch.zeros(n_samples, seq_len, 1)
        self.targets = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            # Choose dominant frequency
            dominant = torch.randint(0, n_frequencies, (1,)).item()
            self.targets[i] = dominant

            # Build signal: dominant + weaker harmonics + noise
            signal = torch.zeros(1, seq_len)
            for f_idx, freq in enumerate(freqs):
                amplitude = 1.0 if f_idx == dominant else 0.2
                phase = torch.rand(1) * 2 * math.pi
                signal += amplitude * torch.sin(
                    2 * math.pi * freq * t * seq_len + phase
                )

            signal += noise_std * torch.randn(1, seq_len)
            self.inputs[i, :, 0] = signal.squeeze()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MultiSineForecastDataset(Dataset):
    """Forecast a quasi-periodic 1D signal built from multiple sinusoids.

    Each sample is a smooth mixture of frequencies drawn from a shared bank,
    optionally modulated by a slow envelope. The task is next-step prediction
    from a context window, with autonomous rollout evaluated by feeding model
    predictions back as the next observation.

    Args:
        n_samples: Number of sequences.
        seq_len: Length of the context window given to the model.
        rollout_horizon: Number of future steps reserved for autonomous rollout.
        prediction_mode: "all" for per-timestep next-step targets or "last"
            for only the next sample after the full context.
        n_components: Number of sinusoidal components in each sequence.
        frequency_bank: Frequencies (cycles over the full normalized interval).
        amplitude_min: Minimum component amplitude.
        amplitude_max: Maximum component amplitude.
        envelope_scale: Strength of a slow multiplicative modulation.
        envelope_freq_min: Minimum envelope frequency.
        envelope_freq_max: Maximum envelope frequency.
        observation_noise_std: Noise level added to observations.
        seed: RNG seed for sample generation.
        normalize_signal: Whether to z-score signals using dataset statistics.
    """

    def __init__(
        self,
        n_samples: int = 12000,
        seq_len: int = 160,
        rollout_horizon: int = 160,
        prediction_mode: str = "all",
        n_components: int = 3,
        frequency_bank: list[float] | None = None,
        amplitude_min: float = 0.3,
        amplitude_max: float = 1.0,
        envelope_scale: float = 0.15,
        envelope_freq_min: float = 0.15,
        envelope_freq_max: float = 0.6,
        observation_noise_std: float = 0.02,
        seed: int = 42,
        normalize_signal: bool = True,
    ):
        super().__init__()

        if prediction_mode not in {"all", "last"}:
            raise ValueError(
                f"prediction_mode must be 'all' or 'last', got {prediction_mode}"
            )
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_noise_std = observation_noise_std
        self.state_dim = 1
        self.observation_dim = 1

        if frequency_bank is None:
            frequency_bank = [1.0, 2.5, 5.5, 9.0, 14.0]
        if len(frequency_bank) < n_components:
            raise ValueError(
                "frequency_bank must contain at least n_components entries"
            )
        self.frequency_bank = tuple(float(f) for f in frequency_bank)

        total_steps = seq_len + rollout_horizon + 1
        self.total_steps = total_steps
        time = np.linspace(0.0, 1.0, total_steps, dtype=np.float32)
        rng = np.random.default_rng(seed)

        signals = np.zeros((n_samples, total_steps), dtype=np.float32)
        for sample_idx in range(n_samples):
            chosen = rng.choice(
                np.asarray(self.frequency_bank),
                size=n_components,
                replace=False,
            )
            amplitudes = rng.uniform(amplitude_min, amplitude_max, size=n_components)
            phases = rng.uniform(0.0, 2.0 * math.pi, size=n_components)

            signal = np.zeros_like(time)
            for freq, amp, phase in zip(chosen, amplitudes, phases):
                signal += amp * np.sin(2.0 * math.pi * freq * time + phase)

            if envelope_scale > 0.0:
                env_freq = rng.uniform(envelope_freq_min, envelope_freq_max)
                env_phase = rng.uniform(0.0, 2.0 * math.pi)
                envelope = 1.0 + envelope_scale * np.sin(
                    2.0 * math.pi * env_freq * time + env_phase
                )
                signal *= envelope

            # Small smooth bias term broadens the family without making the task
            # chaotic or unstable.
            bias = rng.uniform(-0.15, 0.15)
            signal += bias
            signals[sample_idx] = signal.astype(np.float32)

        states_raw = torch.from_numpy(signals).unsqueeze(-1)  # [N, T, 1]
        if normalize_signal:
            mean = states_raw.mean(dim=(0, 1), keepdim=True)
            std = states_raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        else:
            mean = torch.zeros(1, 1, 1, dtype=states_raw.dtype)
            std = torch.ones(1, 1, 1, dtype=states_raw.dtype)

        states = (states_raw - mean) / std
        observations = states.clone()
        if observation_noise_std > 0.0:
            observations = observations + observation_noise_std * torch.randn_like(
                observations
            )

        self.state_mean = mean
        self.state_std = std
        self.full_states_raw = states_raw
        self.full_states = states
        self.full_observations = observations
        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = states[:, 1 : seq_len + 1]
        else:
            self.targets = states[:, seq_len]

    def denormalize_state_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.state_std.to(value.device) + self.state_mean.to(
            value.device
        )

    def states_to_observations(
        self, states_raw: torch.Tensor, add_noise: bool = False
    ) -> torch.Tensor:
        observations = (
            states_raw - self.state_mean.to(states_raw.device)
        ) / self.state_std.to(states_raw.device)
        if add_noise and self.observation_noise_std > 0.0:
            observations = observations + self.observation_noise_std * torch.randn_like(
                observations
            )
        return observations

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LorenzSequenceDataset(Dataset):
    """Lorenz attractor dynamics prediction (regression benchmark).

    By default this behaves like the original one-step benchmark:
    input = [seq_len, 3] state history and target = [3] next-step state.
    It also supports a stronger setting for recurrent regression:
    - partial or projected observations
    - per-timestep next-state targets for ``output_mode="all"``
    - stored future trajectory for post-hoc autonomous rollout evaluation

    Args:
        n_samples: Number of trajectory segments.
        seq_len: Length of each segment.
        dt: Integration timestep.
        sigma, rho, beta: Lorenz system parameters.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 50,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        rollout_horizon: int = 50,
        prediction_mode: str = "last",
        observation_mode: str = "full",
        observation_dim: int = 3,
        observation_seed: int | None = None,
        observation_noise_std: float = 0.0,
        burn_in_steps: int = 200,
        seed: int = 42,
        test_seed: int | None = None,
        integration_method: str = "euler",
        normalize_states: bool = True,
        normalize_observations: bool = True,
        normalization_steps: int = 10000,
        initial_state_scale: float = 0.1,
    ):
        if prediction_mode not in {"last", "all"}:
            raise ValueError(
                f"prediction_mode must be 'last' or 'all', got '{prediction_mode}'"
            )
        if observation_mode not in {"full", "partial", "linear"}:
            raise ValueError(
                "observation_mode must be one of {'full', 'partial', 'linear'}, "
                f"got '{observation_mode}'"
            )
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if integration_method not in {"euler", "rk4"}:
            raise ValueError(
                f"integration_method must be 'euler' or 'rk4', got '{integration_method}'"
            )

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_mode = observation_mode
        self.observation_dim = 3 if observation_mode == "full" else observation_dim
        self.observation_seed = seed if observation_seed is None else observation_seed
        self.observation_noise_std = observation_noise_std
        self.burn_in_steps = burn_in_steps
        self.seed = seed if test_seed is None else test_seed
        self.integration_method = integration_method
        self.normalize_states = normalize_states
        self.normalize_observations = normalize_observations
        self.normalization_steps = normalization_steps
        self.initial_state_scale = initial_state_scale
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        total_future_steps = seq_len + rollout_horizon
        generator = torch.Generator().manual_seed(self.seed)

        self.observation_matrix = self._build_observation_matrix(
            mode=observation_mode,
            observation_dim=self.observation_dim,
            seed=self.observation_seed,
        )
        self.state_mean, self.state_std = self._compute_state_stats(
            n_steps=normalization_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )
        self.obs_mean, self.obs_std = self._compute_observation_stats(
            n_steps=normalization_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )

        self.full_states_raw = self._generate_trajectories(
            n_samples=n_samples,
            total_steps=total_future_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )
        self.full_states = self.normalize_state_tensor(self.full_states_raw)
        observations = self.states_to_observations(
            self.full_states_raw,
            add_noise=observation_noise_std > 0.0,
            generator=generator,
        )

        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = self.full_states[:, 1 : seq_len + 1]
        else:
            self.targets = self.full_states[:, seq_len]

    def _lorenz_derivative(self, state: torch.Tensor) -> torch.Tensor:
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack((dx, dy, dz), dim=-1)

    def _integrate_step(self, state: torch.Tensor) -> torch.Tensor:
        if self.integration_method == "euler":
            return state + self.dt * self._lorenz_derivative(state)

        k1 = self._lorenz_derivative(state)
        k2 = self._lorenz_derivative(state + 0.5 * self.dt * k1)
        k3 = self._lorenz_derivative(state + 0.5 * self.dt * k2)
        k4 = self._lorenz_derivative(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _sample_initial_states(
        self, n_samples: int, generator: torch.Generator
    ) -> torch.Tensor:
        center = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        noise = (
            torch.randn(n_samples, 3, generator=generator) * self.initial_state_scale
        )
        return center.unsqueeze(0) + noise

    def _generate_trajectories(
        self,
        n_samples: int,
        total_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        states = self._sample_initial_states(n_samples=n_samples, generator=generator)
        for _ in range(burn_in_steps):
            states = self._integrate_step(states)

        trajectories = torch.zeros(n_samples, total_steps + 1, 3, dtype=torch.float32)
        trajectories[:, 0] = states
        for step_idx in range(total_steps):
            states = self._integrate_step(states)
            trajectories[:, step_idx + 1] = states
        return trajectories

    def _reference_trajectory(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        state = self._sample_initial_states(n_samples=1, generator=generator).squeeze(0)
        for _ in range(burn_in_steps):
            state = self._integrate_step(state)
        trajectory = torch.zeros(n_steps + 1, 3, dtype=torch.float32)
        trajectory[0] = state
        for step_idx in range(n_steps):
            state = self._integrate_step(state)
            trajectory[step_idx + 1] = state
        return trajectory

    def _build_observation_matrix(
        self, mode: str, observation_dim: int, seed: int
    ) -> torch.Tensor:
        if mode == "full":
            return torch.eye(3, dtype=torch.float32)

        if mode == "partial":
            if observation_dim > 3:
                raise ValueError(
                    f"partial observation_dim must be <= 3, got {observation_dim}"
                )
            basis = torch.eye(3, dtype=torch.float32)
            return basis[:, :observation_dim]

        generator = torch.Generator().manual_seed(seed + 17)
        matrix = torch.randn(3, observation_dim, generator=generator)
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return matrix.to(dtype=torch.float32)

    def _compute_state_stats(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self._reference_trajectory(
            n_steps=n_steps, burn_in_steps=burn_in_steps, generator=generator
        )
        mean = reference.mean(dim=0)
        std = reference.std(dim=0).clamp(min=1e-8)
        return mean, std

    def _compute_observation_stats(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self._reference_trajectory(
            n_steps=n_steps, burn_in_steps=burn_in_steps, generator=generator
        )
        projected = torch.matmul(reference, self.observation_matrix)
        mean = projected.mean(dim=0)
        std = projected.std(dim=0).clamp(min=1e-8)
        return mean, std

    def normalize_state_tensor(self, states_raw: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states_raw
        return (states_raw - self.state_mean) / self.state_std

    def denormalize_state_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states
        return states * self.state_std + self.state_mean

    def normalize_observation_tensor(
        self, observations_raw: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations_raw
        return (observations_raw - self.obs_mean) / self.obs_std

    def denormalize_observation_tensor(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations
        return observations * self.obs_std + self.obs_mean

    def states_to_observations(
        self,
        states_raw: torch.Tensor,
        add_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        observations_raw = torch.matmul(states_raw, self.observation_matrix)
        observations = self.normalize_observation_tensor(observations_raw)
        if add_noise and self.observation_noise_std > 0.0:
            noise = torch.randn(
                observations.shape,
                generator=generator,
                dtype=observations.dtype,
                device=observations.device,
            )
            observations = observations + self.observation_noise_std * noise
        return observations

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SwitchingLDSDataset(Dataset):
    """Partially observed switching linear dynamical system benchmark.

    The latent dynamics follow a mode-dependent linear transition:
        x_{t+1} = A_{m_t} x_t + eps_t
        y_t = C x_t + eta_t

    Modes switch only every ``min_mode_duration`` to ``max_mode_duration`` steps,
    making this a standard switching-LDS setting with a slow hidden regime and
    faster within-mode linear state evolution. The benchmark is intended to test
    whether models can simultaneously track a slow latent mode and a fast latent
    state under partial observation.
    """

    def __init__(
        self,
        n_samples: int = 12000,
        seq_len: int = 96,
        latent_dim: int = 4,
        observation_dim: int = 2,
        n_modes: int = 3,
        rollout_horizon: int = 80,
        prediction_mode: str = "all",
        observation_noise_std: float = 0.03,
        transition_noise_std: float = 0.02,
        min_mode_duration: int = 36,
        max_mode_duration: int = 96,
        burn_in_steps: int = 24,
        seed: int = 42,
        test_seed: int | None = None,
        normalize_states: bool = True,
        normalize_observations: bool = True,
        normalization_samples: int = 4096,
        stats_seed: int = 1234,
        initial_state_scale: float = 0.35,
    ):
        super().__init__()
        if prediction_mode not in {"last", "all"}:
            raise ValueError(
                f"prediction_mode must be 'last' or 'all', got '{prediction_mode}'"
            )
        if latent_dim != 4:
            raise ValueError(
                f"SwitchingLDSDataset currently expects latent_dim=4, got {latent_dim}"
            )
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if n_modes < 2:
            raise ValueError(f"n_modes must be >= 2, got {n_modes}")

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.n_modes = n_modes
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std
        self.min_mode_duration = max(1, min_mode_duration)
        self.max_mode_duration = max(self.min_mode_duration, max_mode_duration)
        self.burn_in_steps = max(0, burn_in_steps)
        self.seed = seed if test_seed is None else test_seed
        self.normalize_states = normalize_states
        self.normalize_observations = normalize_observations
        self.normalization_samples = normalization_samples
        self.stats_seed = stats_seed
        self.initial_state_scale = initial_state_scale

        self.transition_matrices = self._build_transition_matrices(n_modes)
        self.observation_matrix = self._build_observation_matrix(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            seed=stats_seed + 17,
        )

        stats_generator = torch.Generator().manual_seed(stats_seed)
        reference_states, _ = self._simulate(
            n_samples=normalization_samples,
            total_steps=seq_len + rollout_horizon,
            generator=stats_generator,
        )
        reference_observations = self._states_to_observations(
            reference_states, add_noise=False
        )
        self.state_mean = reference_states.mean(dim=(0, 1))
        self.state_std = reference_states.std(dim=(0, 1)).clamp(min=1e-8)
        self.obs_mean = reference_observations.mean(dim=(0, 1))
        self.obs_std = reference_observations.std(dim=(0, 1)).clamp(min=1e-8)

        generator = torch.Generator().manual_seed(self.seed)
        total_steps = seq_len + rollout_horizon
        self.full_states_raw, self.full_modes = self._simulate(
            n_samples=n_samples,
            total_steps=total_steps,
            generator=generator,
        )
        self.full_states = self.normalize_state_tensor(self.full_states_raw)
        observations = self.states_to_observations(
            self.full_states_raw,
            add_noise=observation_noise_std > 0.0,
            generator=generator,
        )

        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = self.full_states[:, 1 : seq_len + 1]
        else:
            self.targets = self.full_states[:, seq_len]

        self.switch_mask = self.full_modes[:, 1:] != self.full_modes[:, :-1]
        self.time_since_switch = self._compute_time_since_switch(self.full_modes)

    def _rotation_block(self, radius: float, angle: float) -> torch.Tensor:
        return torch.tensor(
            [
                [radius * math.cos(angle), -radius * math.sin(angle)],
                [radius * math.sin(angle), radius * math.cos(angle)],
            ],
            dtype=torch.float32,
        )

    def _build_transition_matrices(self, n_modes: int) -> torch.Tensor:
        fast_blocks = [
            self._rotation_block(radius=0.985, angle=0.33),
            self._rotation_block(radius=0.975, angle=-0.24),
            self._rotation_block(radius=0.992, angle=0.11),
        ]
        slow_blocks = [
            torch.tensor([[0.996, 0.018], [0.0, 0.986]], dtype=torch.float32),
            torch.tensor([[0.989, -0.021], [0.012, 0.993]], dtype=torch.float32),
            torch.tensor([[0.995, 0.0], [0.024, 0.984]], dtype=torch.float32),
        ]
        slow_to_fast = [
            torch.tensor([[0.09, -0.03], [0.02, 0.05]], dtype=torch.float32),
            torch.tensor([[-0.07, 0.05], [0.06, -0.02]], dtype=torch.float32),
            torch.tensor([[0.03, 0.08], [-0.05, 0.04]], dtype=torch.float32),
        ]
        fast_to_slow = [
            torch.tensor([[0.012, 0.0], [0.0, 0.015]], dtype=torch.float32),
            torch.tensor([[0.008, -0.006], [0.005, 0.01]], dtype=torch.float32),
            torch.tensor([[0.004, 0.009], [-0.008, 0.006]], dtype=torch.float32),
        ]

        matrices = []
        for mode_idx in range(n_modes):
            fast = fast_blocks[mode_idx % len(fast_blocks)]
            slow = slow_blocks[mode_idx % len(slow_blocks)]
            sf = slow_to_fast[mode_idx % len(slow_to_fast)]
            fs = fast_to_slow[mode_idx % len(fast_to_slow)]
            top = torch.cat((fast, sf), dim=1)
            bottom = torch.cat((fs, slow), dim=1)
            matrix = torch.cat((top, bottom), dim=0)
            eigvals = torch.linalg.eigvals(matrix)
            spectral_radius = eigvals.abs().max().real.item()
            if spectral_radius >= 0.998:
                matrix = matrix * (0.998 / spectral_radius)
            matrices.append(matrix)
        return torch.stack(matrices, dim=0)

    def _build_observation_matrix(
        self, latent_dim: int, observation_dim: int, seed: int
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(latent_dim, observation_dim, generator=generator)
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return matrix.to(dtype=torch.float32)

    def _sample_initial_states(
        self, n_samples: int, generator: torch.Generator
    ) -> torch.Tensor:
        return (
            torch.randn(n_samples, self.latent_dim, generator=generator)
            * self.initial_state_scale
        )

    def _generate_mode_sequences(
        self, n_samples: int, total_steps: int, generator: torch.Generator
    ) -> torch.Tensor:
        modes = torch.zeros(n_samples, total_steps + 1, dtype=torch.long)
        for sample_idx in range(n_samples):
            current_mode = torch.randint(
                0, self.n_modes, (1,), generator=generator
            ).item()
            t = 0
            while t <= total_steps:
                block = torch.randint(
                    self.min_mode_duration,
                    self.max_mode_duration + 1,
                    (1,),
                    generator=generator,
                ).item()
                end = min(total_steps + 1, t + block)
                modes[sample_idx, t:end] = current_mode
                if end >= total_steps + 1:
                    break
                next_mode = torch.randint(
                    0, self.n_modes - 1, (1,), generator=generator
                ).item()
                current_mode = next_mode if next_mode < current_mode else next_mode + 1
                t = end
        return modes

    def _simulate(
        self, n_samples: int, total_steps: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode_sequences = self._generate_mode_sequences(
            n_samples=n_samples,
            total_steps=total_steps + self.burn_in_steps,
            generator=generator,
        )

        states = self._sample_initial_states(n_samples=n_samples, generator=generator)
        trajectories = torch.zeros(
            n_samples,
            total_steps + self.burn_in_steps + 1,
            self.latent_dim,
            dtype=torch.float32,
        )
        trajectories[:, 0] = states

        for step_idx in range(total_steps + self.burn_in_steps):
            active_modes = mode_sequences[:, step_idx]
            transition = self.transition_matrices[active_modes]
            process_noise = (
                torch.randn(n_samples, self.latent_dim, generator=generator)
                * self.transition_noise_std
            )
            states = torch.einsum("nij,nj->ni", transition, states) + process_noise
            trajectories[:, step_idx + 1] = states

        start = self.burn_in_steps
        end = start + total_steps + 1
        return trajectories[:, start:end], mode_sequences[:, start:end]

    def _states_to_observations(
        self,
        states_raw: torch.Tensor,
        *,
        add_noise: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        observations = states_raw @ self.observation_matrix
        if add_noise and self.observation_noise_std > 0.0:
            noise = torch.randn(
                observations.shape, generator=generator, dtype=observations.dtype
            )
            observations = observations + self.observation_noise_std * noise
        return observations

    def _compute_time_since_switch(self, modes: torch.Tensor) -> torch.Tensor:
        steps = torch.zeros_like(modes, dtype=torch.long)
        for sample_idx in range(modes.shape[0]):
            current = 0
            for step_idx in range(1, modes.shape[1]):
                if modes[sample_idx, step_idx] != modes[sample_idx, step_idx - 1]:
                    current = 0
                else:
                    current += 1
                steps[sample_idx, step_idx] = current
        return steps

    def normalize_state_tensor(self, states_raw: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states_raw
        return (states_raw - self.state_mean) / self.state_std

    def denormalize_state_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states
        return states * self.state_std + self.state_mean

    def normalize_observation_tensor(
        self, observations_raw: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations_raw
        return (observations_raw - self.obs_mean) / self.obs_std

    def denormalize_observation_tensor(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations
        return observations * self.obs_std + self.obs_mean

    def states_to_observations(
        self,
        states_raw: torch.Tensor,
        *,
        add_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        observations_raw = self._states_to_observations(
            states_raw, add_noise=add_noise, generator=generator
        )
        return self.normalize_observation_tensor(observations_raw)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


__all__ = [
    "LorenzSequenceDataset",
    "MultiFrequencyDataset",
    "MultiSineForecastDataset",
    "SwitchingLDSDataset",
]
