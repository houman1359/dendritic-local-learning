"""Sequence dataset task implementations."""

from __future__ import annotations

import logging
import math
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HierarchicalTemporalDataset(Dataset):
    """Context-gated multi-timescale classification task.

    Tests whether a model can jointly perform:
    1. **Slow memory** -- maintain a context cue over the full sequence (τ~125)
    2. **Medium integration** -- identify block-level patterns (τ~25)
    3. **Fast filtering** -- denoise individual timesteps (τ~5)
    4. **Multiplicative gating** -- context determines which blocks matter

    Structure per sample:
    - *Context phase* (``context_duration`` steps): present one of ``n_contexts``
      context cues as a one-hot vector.
    - *Integration phase* (``n_blocks x block_duration`` steps): sequence of
      blocks.  Each block has a type drawn uniformly from ``n_block_types``.
      The per-timestep input is the block's prototype vector + Gaussian noise.
    - Classification: each context ``c`` maps to a target block type ``t_c``.
      Class is determined by whether the target type count exceeds the
      expected count (``n_blocks / n_block_types``).

    Samples are generated with balanced classes (50/50) by first choosing
    the target class then constructing a block sequence consistent with it.

    The task is designed so that DendriNet's geometric tau spacing
    (125, 25, 5) aligns with the three timescales of the task.

    Args:
        n_samples: Number of samples.
        n_contexts: Number of context cues.
        n_block_types: Number of distinct block types.
        n_blocks: Number of blocks in the integration phase.
        context_duration: Steps for which the context cue is shown.
        block_duration: Steps per block.
        signal_dim: Dimensionality of the input signal.
        noise_std: Standard deviation of fast Gaussian noise.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_contexts: int = 2,
        n_block_types: int = 4,
        n_blocks: int = 8,
        context_duration: int = 20,
        block_duration: int = 25,
        signal_dim: int = 16,
        noise_std: float = 0.5,
    ):
        self.n_samples = n_samples
        self.n_contexts = n_contexts
        self.n_block_types = n_block_types
        self.n_blocks = n_blocks
        self.context_duration = context_duration
        self.block_duration = block_duration
        self.signal_dim = signal_dim
        self.noise_std = noise_std
        self.total_len = context_duration + n_blocks * block_duration
        self.input_dim = signal_dim

        # Build fixed block prototypes: [n_block_types, signal_dim]
        rng = np.random.RandomState(0)
        prototypes = rng.randn(n_block_types, signal_dim).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.prototypes = torch.from_numpy(prototypes)

        # Build context cue vectors: [n_contexts, signal_dim]
        ctx_vecs = rng.randn(n_contexts, signal_dim).astype(np.float32)
        ctx_vecs /= np.linalg.norm(ctx_vecs, axis=1, keepdims=True)
        self.context_vectors = torch.from_numpy(ctx_vecs)

        # Map each context to a target block type
        self.context_to_target = torch.arange(n_contexts) % n_block_types
        # Threshold: target type must appear more than expected by chance
        self.threshold = n_blocks // n_block_types + 1

        # Generate all samples with balanced classes
        self.inputs = torch.zeros(n_samples, self.total_len, signal_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, dtype=torch.long)
        self.block_types = torch.zeros(n_samples, n_blocks, dtype=torch.long)

        for i in range(n_samples):
            ctx = torch.randint(0, n_contexts, (1,)).item()
            self.contexts[i] = ctx
            target_type = self.context_to_target[ctx].item()

            # Decide class first (balanced 50/50)
            label = i % 2
            self.targets[i] = label

            # Build block sequence consistent with the label
            block_seq = self._sample_block_seq(
                target_type, label, n_blocks, n_block_types, self.threshold
            )
            self.block_types[i] = block_seq

            # Context phase
            for t in range(context_duration):
                self.inputs[i, t] = self.context_vectors[ctx]

            # Integration phase
            for b in range(n_blocks):
                bt = block_seq[b].item()
                t_start = context_duration + b * block_duration
                for t in range(t_start, t_start + block_duration):
                    self.inputs[i, t] = self.prototypes[bt]

            # Add fast noise to entire sequence
            self.inputs[i] += torch.randn(self.total_len, signal_dim) * noise_std

    @staticmethod
    def _sample_block_seq(
        target_type: int,
        label: int,
        n_blocks: int,
        n_block_types: int,
        threshold: int,
    ) -> torch.Tensor:
        """Sample a block sequence with the correct number of target blocks."""
        non_target_types = [t for t in range(n_block_types) if t != target_type]
        for _ in range(1000):
            if label == 1:
                # Target type must appear >= threshold times
                n_target = torch.randint(threshold, n_blocks + 1, (1,)).item()
            else:
                # Target type must appear < threshold times
                n_target = torch.randint(0, threshold, (1,)).item()
            seq = torch.full((n_blocks,), target_type, dtype=torch.long)
            # Fill remaining positions with non-target types
            other_indices = torch.randperm(n_blocks)[n_target:]
            for idx in other_indices:
                seq[idx] = non_target_types[
                    torch.randint(0, len(non_target_types), (1,)).item()
                ]
            # Shuffle
            seq = seq[torch.randperm(n_blocks)]
            return seq
        # Fallback
        return torch.randint(0, n_block_types, (n_blocks,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ContextDependentDecisionDataset(Dataset):
    """Context-dependent integration task inspired by Mante et al. (2013).

    Two sensory channels (e.g. "color" and "motion") are presented simultaneously
    as noisy streams.  A brief context cue at sequence onset indicates which
    channel is task-relevant.  The model must integrate the relevant channel
    over a variable delay and report the sign of the integrated evidence.

    This tests:
    1. **Context gating** -- selectively route one channel through dendritic
       pathways while ignoring the other.
    2. **Noisy integration** -- accumulate evidence over time using slow traces.
    3. **Interference resistance** -- the irrelevant channel provides a strong
       distractor that must be suppressed.

    Input structure per timestep:
        [context_cue (2d) | channel_A (signal_dim) | channel_B (signal_dim)]
    Total input_dim = 2 + 2 * signal_dim

    Target: binary (0/1) -- sign of integrated evidence in the relevant channel.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        signal_dim: int = 8,
        context_duration: int = 10,
        integration_duration: int = 50,
        coherence_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
        noise_std: float = 1.0,
        variable_delay: bool = True,
        min_integration: int = 20,
        max_integration: int = 80,
        delay_duration: int = 0,
        min_delay: int | None = None,
        max_delay: int | None = None,
        conflict_fraction: float = 0.0,
        distractor_burst_prob: float = 0.0,
        distractor_burst_scale: float = 2.0,
        distractor_burst_duration: int = 5,
        cue_dropout_prob: float = 0.0,
        cue_keep_duration: int = 1,
        late_context_switch_prob: float = 0.0,
        context_signal_scale: float = 1.0,
        return_seq_lengths: bool = False,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.signal_dim = signal_dim
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 2 + (4 if self.positive_input_encoding else 2) * signal_dim
        self.context_duration = context_duration
        self.integration_duration = integration_duration
        self.variable_integration = bool(variable_delay)
        self.min_integration = min_integration
        self.max_integration = max_integration
        self.delay_duration = delay_duration
        self.min_delay = delay_duration if min_delay is None else min_delay
        self.max_delay = delay_duration if max_delay is None else max_delay
        self.return_seq_lengths = bool(
            return_seq_lengths
            or self.variable_integration
            or self.min_delay != self.max_delay
        )

        max_integration_len = (
            max_integration if self.variable_integration else integration_duration
        )
        max_delay_len = self.max_delay
        self.seq_len = context_duration + max_integration_len + max_delay_len

        self.inputs = torch.zeros(n_samples, self.seq_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.seq_lengths = torch.full((n_samples,), self.seq_len, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, dtype=torch.long)
        self.final_contexts = torch.zeros(n_samples, dtype=torch.long)
        self.integration_lengths = torch.zeros(n_samples, dtype=torch.long)
        self.delay_lengths = torch.zeros(n_samples, dtype=torch.long)
        self.switch_steps = torch.full((n_samples,), -1, dtype=torch.long)
        self.conflict_flags = torch.zeros(n_samples, dtype=torch.bool)
        self.coherence_a = torch.zeros(n_samples, dtype=torch.float32)
        self.coherence_b = torch.zeros(n_samples, dtype=torch.float32)
        self.sign_a = torch.zeros(n_samples, dtype=torch.float32)
        self.sign_b = torch.zeros(n_samples, dtype=torch.float32)
        self.relevant_scores = torch.zeros(n_samples, dtype=torch.float32)

        coherence_tensor = torch.tensor(coherence_levels, dtype=torch.float32)
        cue_keep_duration = max(1, min(cue_keep_duration, context_duration))
        distractor_burst_duration = max(1, distractor_burst_duration)

        for i in range(n_samples):
            initial_context = torch.randint(0, 2, (1,)).item()
            integration_len = (
                torch.randint(min_integration, max_integration + 1, (1,)).item()
                if self.variable_integration
                else integration_duration
            )
            delay_len = (
                torch.randint(self.min_delay, self.max_delay + 1, (1,)).item()
                if self.max_delay > self.min_delay
                else self.max_delay
            )
            seq_len = context_duration + integration_len + delay_len

            conflict = torch.rand(1).item() < conflict_fraction
            switch_trial = (
                integration_len >= 4 and torch.rand(1).item() < late_context_switch_prob
            )
            switch_step = -1
            if switch_trial:
                switch_low = max(1, integration_len // 3)
                switch_high = max(switch_low + 1, integration_len - 1)
                switch_step = torch.randint(switch_low, switch_high, (1,)).item()

            sign_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            if conflict:
                sign_b = -sign_a
            else:
                sign_b = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            coh_a = coherence_tensor[
                torch.randint(0, len(coherence_levels), (1,))
            ].item()
            coh_b = coherence_tensor[
                torch.randint(0, len(coherence_levels), (1,))
            ].item()

            self.contexts[i] = initial_context
            self.final_contexts[i] = (
                1 - initial_context if switch_step >= 0 else initial_context
            )
            self.integration_lengths[i] = integration_len
            self.delay_lengths[i] = delay_len
            self.seq_lengths[i] = seq_len
            self.switch_steps[i] = switch_step
            self.conflict_flags[i] = conflict
            self.coherence_a[i] = coh_a
            self.coherence_b[i] = coh_b
            self.sign_a[i] = sign_a
            self.sign_b[i] = sign_b

            cue = torch.zeros(context_duration, 2)
            cue[:, initial_context] = context_signal_scale
            if torch.rand(1).item() < cue_dropout_prob:
                cue[cue_keep_duration:] = 0.0
            self.inputs[i, :context_duration, :2] = cue

            burst_start = None
            if (
                integration_len >= distractor_burst_duration
                and torch.rand(1).item() < distractor_burst_prob
            ):
                burst_start = torch.randint(
                    0,
                    integration_len - distractor_burst_duration + 1,
                    (1,),
                ).item()

            relevant_score = 0.0
            for t in range(integration_len):
                active_context = initial_context
                if switch_step >= 0 and t >= switch_step:
                    active_context = 1 - initial_context
                    abs_t = context_duration + t
                    self.inputs[i, abs_t, :2] = 0.0
                    self.inputs[i, abs_t, active_context] = context_signal_scale

                a_signal = sign_a * coh_a + noise_std * torch.randn(signal_dim)
                b_signal = sign_b * coh_b + noise_std * torch.randn(signal_dim)

                if (
                    burst_start is not None
                    and burst_start <= t < burst_start + distractor_burst_duration
                ):
                    burst_sign = -sign_a if active_context == 0 else -sign_b
                    burst = distractor_burst_scale * max(coh_a, coh_b) * burst_sign
                    if active_context == 0:
                        b_signal = b_signal + burst
                    else:
                        a_signal = a_signal + burst

                abs_t = context_duration + t
                if self.positive_input_encoding:
                    a_pos = torch.relu(a_signal)
                    a_neg = torch.relu(-a_signal)
                    b_pos = torch.relu(b_signal)
                    b_neg = torch.relu(-b_signal)
                    self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_pos
                    self.inputs[i, abs_t, 2 + signal_dim : 2 + 2 * signal_dim] = a_neg
                    self.inputs[
                        i,
                        abs_t,
                        2 + 2 * signal_dim : 2 + 3 * signal_dim,
                    ] = b_pos
                    self.inputs[i, abs_t, 2 + 3 * signal_dim :] = b_neg
                else:
                    self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_signal
                    self.inputs[i, abs_t, 2 + signal_dim :] = b_signal

                if active_context == 0:
                    relevant_score += sign_a * coh_a
                else:
                    relevant_score += sign_b * coh_b

            self.relevant_scores[i] = relevant_score
            self.targets[i] = int(relevant_score > 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.return_seq_lengths:
            seq_lengths = self.seq_lengths[idx]
            if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
                seq_lengths = int(seq_lengths.item())
        else:
            seq_lengths = None
        return self.inputs[idx], self.targets[idx], seq_lengths


class SwitchingContextDecisionDataset(Dataset):
    """Continuous evidence integration under slowly switching context.

    This task is designed to separate models that can simultaneously maintain
    a slow latent context and perform fast evidence integration. Each sample is
    a fixed-length sequence partitioned into long context blocks. Within a
    block, the relevant context stays constant while the local evidence target
    changes every ``trial_duration`` steps. The model must emit the correct
    binary decision at every timestep.

    Input structure per timestep:
        [context_hint (2d) | channel_A (signal_dim) | channel_B (signal_dim)]

    Target per timestep:
        binary decision (0/1) given by the sign of the context-relevant
        evidence segment active at that timestep.
    """

    def __init__(
        self,
        n_samples: int = 16000,
        signal_dim: int = 8,
        seq_len: int = 360,
        trial_duration: int = 12,
        min_context_duration: int = 90,
        max_context_duration: int = 150,
        coherence_levels: tuple[float, ...] = (0.03, 0.06, 0.12, 0.24),
        noise_std: float = 1.1,
        conflict_fraction: float = 0.75,
        distractor_burst_prob: float = 0.35,
        distractor_burst_scale: float = 2.0,
        distractor_burst_duration: int = 3,
        context_cue_duration: int = 10,
        switch_cue_duration: int = 4,
        context_signal_scale: float = 1.0,
        switch_signal_scale: float = 0.35,
        min_blocks: int = 2,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.signal_dim = signal_dim
        self.seq_len = seq_len
        self.trial_duration = max(1, trial_duration)
        self.min_context_duration = max(self.trial_duration, min_context_duration)
        self.max_context_duration = max(self.min_context_duration, max_context_duration)
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 2 + (4 if self.positive_input_encoding else 2) * signal_dim

        coherence_tensor = torch.tensor(coherence_levels, dtype=torch.float32)
        context_cue_duration = max(1, context_cue_duration)
        switch_cue_duration = max(1, switch_cue_duration)
        distractor_burst_duration = max(1, distractor_burst_duration)
        min_blocks = max(2, min_blocks)

        self.inputs = torch.zeros(n_samples, seq_len, self.input_dim)
        self.targets = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.time_since_switch = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.switch_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.cue_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.conflict_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.relevant_signal = torch.zeros(n_samples, seq_len, dtype=torch.float32)
        self.irrelevant_signal = torch.zeros(n_samples, seq_len, dtype=torch.float32)

        for i in range(n_samples):
            block_lengths = []
            remaining = seq_len
            while remaining > 0:
                min_block = min(self.min_context_duration, remaining)
                max_block = min(self.max_context_duration, remaining)
                if len(block_lengths) + 1 < min_blocks:
                    # Reserve enough room for the remaining required blocks.
                    reserve = self.min_context_duration * (
                        min_blocks - len(block_lengths) - 1
                    )
                    max_block = max(min_block, min(max_block, remaining - reserve))
                block_len = (
                    torch.randint(min_block, max_block + 1, (1,)).item()
                    if max_block > min_block
                    else min_block
                )
                block_lengths.append(block_len)
                remaining -= block_len
            if len(block_lengths) < min_blocks:
                # Fallback to equal partitions when clipping produced too few blocks.
                base = seq_len // min_blocks
                block_lengths = [base] * min_blocks
                block_lengths[-1] += seq_len - sum(block_lengths)

            current_context = torch.randint(0, 2, (1,)).item()
            t_start = 0

            for block_idx, block_len in enumerate(block_lengths):
                t_end = min(seq_len, t_start + block_len)
                cue_duration = min(
                    context_cue_duration if block_idx == 0 else switch_cue_duration,
                    t_end - t_start,
                )
                cue_scale = (
                    context_signal_scale if block_idx == 0 else switch_signal_scale
                )

                self.contexts[i, t_start:t_end] = current_context
                self.time_since_switch[i, t_start:t_end] = torch.arange(
                    0, t_end - t_start, dtype=torch.long
                )
                self.switch_mask[i, t_start] = True
                self.cue_mask[i, t_start : t_start + cue_duration] = True
                self.inputs[i, t_start : t_start + cue_duration, current_context] = (
                    cue_scale
                )

                trial_t = t_start
                while trial_t < t_end:
                    seg_end = min(t_end, trial_t + self.trial_duration)
                    conflict = torch.rand(1).item() < conflict_fraction
                    sign_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    sign_b = (
                        -sign_a
                        if conflict
                        else (1.0 if torch.rand(1).item() > 0.5 else -1.0)
                    )
                    coh_a = coherence_tensor[
                        torch.randint(0, len(coherence_tensor), (1,))
                    ].item()
                    coh_b = coherence_tensor[
                        torch.randint(0, len(coherence_tensor), (1,))
                    ].item()

                    burst_start = None
                    seg_len = seg_end - trial_t
                    if (
                        seg_len >= distractor_burst_duration
                        and torch.rand(1).item() < distractor_burst_prob
                    ):
                        burst_start = torch.randint(
                            0, seg_len - distractor_burst_duration + 1, (1,)
                        ).item()

                    for local_t, abs_t in enumerate(range(trial_t, seg_end)):
                        a_signal = sign_a * coh_a + noise_std * torch.randn(signal_dim)
                        b_signal = sign_b * coh_b + noise_std * torch.randn(signal_dim)

                        if (
                            burst_start is not None
                            and burst_start
                            <= local_t
                            < burst_start + distractor_burst_duration
                        ):
                            burst_sign = -sign_a if current_context == 0 else -sign_b
                            burst = (
                                distractor_burst_scale * max(coh_a, coh_b) * burst_sign
                            )
                            if current_context == 0:
                                b_signal = b_signal + burst
                            else:
                                a_signal = a_signal + burst

                        if self.positive_input_encoding:
                            a_pos = torch.relu(a_signal)
                            a_neg = torch.relu(-a_signal)
                            b_pos = torch.relu(b_signal)
                            b_neg = torch.relu(-b_signal)
                            self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_pos
                            self.inputs[
                                i, abs_t, 2 + signal_dim : 2 + 2 * signal_dim
                            ] = a_neg
                            self.inputs[
                                i, abs_t, 2 + 2 * signal_dim : 2 + 3 * signal_dim
                            ] = b_pos
                            self.inputs[i, abs_t, 2 + 3 * signal_dim :] = b_neg
                        else:
                            self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_signal
                            self.inputs[i, abs_t, 2 + signal_dim :] = b_signal
                        self.conflict_mask[i, abs_t] = conflict

                        if current_context == 0:
                            self.targets[i, abs_t] = int(sign_a > 0)
                            self.relevant_signal[i, abs_t] = sign_a * coh_a
                            self.irrelevant_signal[i, abs_t] = sign_b * coh_b
                        else:
                            self.targets[i, abs_t] = int(sign_b > 0)
                            self.relevant_signal[i, abs_t] = sign_b * coh_b
                            self.irrelevant_signal[i, abs_t] = sign_a * coh_a

                    trial_t = seg_end

                t_start = t_end
                current_context = 1 - current_context

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TimescaleGeneralizationDataset(Dataset):
    """Train on one temporal regime, test on another.

    A simple delayed match-to-sample task where the delay duration is drawn
    from a specified range.  By creating separate train/test datasets with
    non-overlapping delay ranges, we test timescale *generalization* rather
    than interpolation.

    This directly tests whether learnable tau / hierarchical timescales
    provide compositional temporal generalization.

    Input structure:
        [stimulus phase | delay (zeros) | probe phase]
    Target: binary match/non-match.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        stimulus_dim: int = 16,
        stimulus_duration: int = 10,
        min_delay: int = 10,
        max_delay: int = 100,
        probe_duration: int = 10,
        positive_input_encoding: bool = False,
        noise_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.positive_input_encoding = bool(positive_input_encoding)
        max_seq_len = stimulus_duration + max_delay + probe_duration
        self.max_seq_len = max_seq_len

        inputs = torch.zeros(n_samples, max_seq_len, stimulus_dim)
        seq_lengths = torch.zeros(n_samples, dtype=torch.long)

        delays = torch.randint(min_delay, max_delay + 1, (n_samples,))
        matches = torch.randint(0, 2, (n_samples,))

        # Generate all templates and probes at once
        if self.positive_input_encoding:
            templates = torch.rand(n_samples, stimulus_dim)
            templates = (
                templates / templates.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
            )
            alt_probes = torch.rand(n_samples, stimulus_dim)
            alt_probes = (
                alt_probes / alt_probes.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
            )
        else:
            templates = torch.randn(n_samples, stimulus_dim)
            templates = templates / templates.norm(dim=1, keepdim=True) * 2.0
            alt_probes = torch.randn(n_samples, stimulus_dim)
            alt_probes = alt_probes / alt_probes.norm(dim=1, keepdim=True) * 2.0
        probes = torch.where(matches.unsqueeze(1).bool(), templates, alt_probes)

        # Stimulus noise and probe noise
        stim_noise = noise_std * torch.randn(n_samples, stimulus_duration, stimulus_dim)
        probe_noise = noise_std * torch.randn(n_samples, probe_duration, stimulus_dim)

        # Fill stimulus phase (same for all samples)
        inputs[:, :stimulus_duration] = templates.unsqueeze(1) + stim_noise
        if self.positive_input_encoding:
            inputs[:, :stimulus_duration].clamp_(min=0.0)

        # Fill probe phase (varies by delay)
        for i in range(n_samples):
            d = delays[i].item()
            total_len = stimulus_duration + d + probe_duration
            probe_start = stimulus_duration + d
            inputs[i, probe_start:total_len] = probes[i].unsqueeze(0) + probe_noise[i]
            if self.positive_input_encoding:
                inputs[i, probe_start:total_len].clamp_(min=0.0)
            seq_lengths[i] = total_len

        targets = matches
        self.inputs = inputs
        self.targets = targets
        self.seq_lengths = seq_lengths

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.seq_lengths[idx]


class ReadySetGoDataset(Dataset):
    """Interval-reproduction task with an autonomous post-Set Go response.

    A Ready pulse and a Set pulse are separated by a sampled interval.  After
    Set, the model must reproduce that interval and place a single Go event at
    ``set_step + production_scale * sample_interval``.  The training target is
    a normalized Gaussian distribution over time, intended for the
    ``temporal_event_ce`` loss.  This avoids the severe background imbalance of
    independent per-timestep event classification.

    Inputs are three non-negative channels: Ready, Set, and task-irrelevant
    distractor pulses.  Set time is sampled independently of the interval and
    Ready is placed one interval earlier.  Therefore neither absolute Set time
    nor construction order identifies the interval.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        sample_intervals: tuple[int, ...] = (40, 50, 60, 70, 80, 90, 100),
        production_scale: float = 1.0,
        set_min_step: int = 110,
        set_max_step: int = 130,
        pulse_duration: int = 2,
        target_sigma: float = 2.0,
        post_go_duration: int = 20,
        sequence_duration: int | None = None,
        cue_gain_min: float = 0.7,
        cue_gain_max: float = 1.3,
        input_noise_std: float = 0.05,
        mean_distractors_per_trial: float = 1.0,
        distractor_amplitude: float = 1.0,
        distractor_pulse_duration: int = 2,
        positive_input_encoding: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, Integral)
            or n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")
        if not sample_intervals or any(
            isinstance(interval, bool) or not isinstance(interval, Integral)
            for interval in sample_intervals
        ):
            raise TypeError("sample_intervals must contain integers")
        intervals = tuple(int(interval) for interval in sample_intervals)
        if any(interval <= 0 for interval in intervals):
            raise ValueError("sample_intervals must contain positive integers")
        if len(set(intervals)) != len(intervals):
            raise ValueError("sample_intervals must not contain duplicates")
        if not math.isfinite(production_scale) or production_scale <= 0.0:
            raise ValueError("production_scale must be finite and positive")
        integer_options = {
            "set_min_step": set_min_step,
            "set_max_step": set_max_step,
            "pulse_duration": pulse_duration,
            "post_go_duration": post_go_duration,
            "distractor_pulse_duration": distractor_pulse_duration,
            "seed": seed,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in integer_options.values()
        ):
            raise TypeError("step, duration, and seed options must be integers")
        if set_min_step < max(intervals) or set_max_step < set_min_step:
            raise ValueError(
                "expected max(sample_intervals) <= set_min_step <= set_max_step"
            )
        if pulse_duration <= 0 or distractor_pulse_duration <= 0:
            raise ValueError("pulse durations must be positive")
        if min(intervals) < pulse_duration:
            raise ValueError("sample intervals must prevent Ready/Set pulse overlap")
        if not math.isfinite(target_sigma) or target_sigma <= 0.0:
            raise ValueError("target_sigma must be finite and positive")
        minimum_post_go = math.ceil(4.0 * target_sigma)
        if post_go_duration < minimum_post_go:
            raise ValueError(
                "post_go_duration must cover at least four target standard "
                f"deviations ({minimum_post_go} steps)"
            )
        if (
            not math.isfinite(cue_gain_min)
            or not math.isfinite(cue_gain_max)
            or cue_gain_min <= 0.0
            or cue_gain_max < cue_gain_min
        ):
            raise ValueError("cue gain bounds must be positive and ordered")
        if not math.isfinite(input_noise_std) or input_noise_std < 0.0:
            raise ValueError("input_noise_std must be finite and non-negative")
        if (
            not math.isfinite(mean_distractors_per_trial)
            or mean_distractors_per_trial < 0.0
        ):
            raise ValueError(
                "mean_distractors_per_trial must be finite and non-negative"
            )
        if not math.isfinite(distractor_amplitude) or distractor_amplitude < 0.0:
            raise ValueError("distractor_amplitude must be finite and non-negative")

        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.sample_interval_values = torch.tensor(intervals, dtype=torch.long)
        self.production_scale = float(production_scale)
        self.set_min_step = int(set_min_step)
        self.set_max_step = int(set_max_step)
        self.pulse_duration = int(pulse_duration)
        self.target_sigma = float(target_sigma)
        self.post_go_duration = int(post_go_duration)
        self.cue_gain_min = float(cue_gain_min)
        self.cue_gain_max = float(cue_gain_max)
        self.input_noise_std = float(input_noise_std)
        self.mean_distractors_per_trial = float(mean_distractors_per_trial)
        self.distractor_amplitude = float(distractor_amplitude)
        self.distractor_pulse_duration = int(distractor_pulse_duration)
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 3

        maximum_interval = max(intervals)
        maximum_production = round(self.production_scale * maximum_interval)
        minimum_production = round(self.production_scale * min(intervals))
        if minimum_production < self.pulse_duration:
            raise ValueError("production intervals must extend beyond the Set pulse")
        required_duration = (
            self.set_max_step + maximum_production + self.post_go_duration + 1
        )
        if sequence_duration is None:
            sequence_duration = required_duration
        if isinstance(sequence_duration, bool) or not isinstance(
            sequence_duration, Integral
        ):
            raise TypeError("sequence_duration must be an integer or None")
        if sequence_duration < required_duration:
            raise ValueError(
                "sequence_duration is too short for the largest Ready-Set-Go trial: "
                f"need at least {required_duration}, got {sequence_duration}"
            )
        self.sequence_duration = int(sequence_duration)
        if self.distractor_pulse_duration > self.sequence_duration:
            raise ValueError("distractor pulse cannot exceed sequence duration")

        self.inputs = torch.zeros(
            self.n_samples, self.sequence_duration, self.input_dim
        )
        self.targets = torch.zeros(self.n_samples, self.sequence_duration)
        self.sample_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.production_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.ready_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.set_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.go_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.cue_gains = torch.zeros(self.n_samples)
        self.distractor_mask = torch.zeros(
            self.n_samples, self.sequence_duration, dtype=torch.bool
        )

        generator = torch.Generator().manual_seed(self.seed)
        balanced_interval_indices = torch.arange(self.n_samples) % len(intervals)
        balanced_interval_indices = balanced_interval_indices[
            torch.randperm(self.n_samples, generator=generator)
        ]
        time = torch.arange(self.sequence_duration, dtype=torch.float32)
        for sample_idx in range(self.n_samples):
            interval_idx = int(balanced_interval_indices[sample_idx])
            sample_interval = intervals[interval_idx]
            production_interval = round(self.production_scale * sample_interval)
            set_step = torch.randint(
                self.set_min_step,
                self.set_max_step + 1,
                (1,),
                generator=generator,
            ).item()
            ready_step = set_step - sample_interval
            go_step = set_step + production_interval
            gain = (
                torch.empty(1)
                .uniform_(
                    self.cue_gain_min,
                    self.cue_gain_max,
                    generator=generator,
                )
                .item()
            )

            self.sample_intervals[sample_idx] = sample_interval
            self.production_intervals[sample_idx] = production_interval
            self.ready_steps[sample_idx] = ready_step
            self.set_steps[sample_idx] = set_step
            self.go_steps[sample_idx] = go_step
            self.cue_gains[sample_idx] = gain
            self.inputs[
                sample_idx, ready_step : ready_step + self.pulse_duration, 0
            ] = gain
            self.inputs[sample_idx, set_step : set_step + self.pulse_duration, 1] = gain

            n_distractors = int(
                torch.poisson(
                    torch.tensor(self.mean_distractors_per_trial),
                    generator=generator,
                ).item()
            )
            if n_distractors > 0 and self.distractor_amplitude > 0.0:
                latest_start = self.sequence_duration - self.distractor_pulse_duration
                for _ in range(n_distractors):
                    start = torch.randint(
                        0, latest_start + 1, (1,), generator=generator
                    ).item()
                    stop = start + self.distractor_pulse_duration
                    self.inputs[sample_idx, start:stop, 2] += self.distractor_amplitude
                    self.distractor_mask[sample_idx, start:stop] = True

            if self.input_noise_std > 0.0:
                self.inputs[sample_idx] += self.input_noise_std * torch.randn(
                    self.sequence_duration,
                    self.input_dim,
                    generator=generator,
                )
            if self.positive_input_encoding:
                self.inputs[sample_idx].clamp_(min=0.0)

            target = torch.exp(
                -0.5 * ((time - float(go_step)) / self.target_sigma).square()
            )
            self.targets[sample_idx] = target / target.sum().clamp_min(1e-12)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class ContextualReadySetGoDataset(Dataset):
    """Context-selected interval reproduction with two competing cue streams.

    Each cue geometry is emitted as a counterfactual pair.  The two members
    have bit-identical Ready--Set streams, gains, distractors, and input noise,
    but opposite one-hot contexts and different target Go times.  Thus the
    context pulse is the only input that identifies which stream controls the
    target.  Neither stream receives a Go input, and the two implied Go times
    are always distinct.

    The seven input channels are Ready-A, Set-A, Ready-B, Set-B, context-A,
    context-B, and task-irrelevant distractor pulses.  Setting
    ``positive_input_encoding=True`` clamps additive noise to preserve a
    non-negative encoding.  Targets are normalized Gaussian distributions over
    the fixed sequence horizon and are compatible with ``temporal_event_ce``.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        sample_intervals: tuple[int, ...] = (40, 50, 60, 70, 80, 90, 100),
        production_scale: float = 1.0,
        set_min_step: int = 130,
        set_max_step: int = 260,
        pulse_duration: int = 2,
        context_start_step: int = 0,
        context_duration: int = 10,
        context_gain: float = 1.0,
        minimum_competing_go_separation: int = 10,
        target_sigma: float = 2.0,
        post_go_duration: int = 20,
        sequence_duration: int | None = None,
        cue_gain_min: float = 0.7,
        cue_gain_max: float = 1.3,
        input_noise_std: float = 0.05,
        mean_distractors_per_trial: float = 1.0,
        distractor_amplitude: float = 1.0,
        distractor_pulse_duration: int = 2,
        positive_input_encoding: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, Integral)
            or n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")
        if n_samples % 2:
            raise ValueError(
                "contextual Ready-Set-Go requires an even n_samples for "
                "counterfactual context pairs"
            )
        if not sample_intervals or any(
            isinstance(interval, bool) or not isinstance(interval, Integral)
            for interval in sample_intervals
        ):
            raise TypeError("sample_intervals must contain integers")
        intervals = tuple(int(interval) for interval in sample_intervals)
        if len(intervals) < 2:
            raise ValueError("contextual Ready-Set-Go requires at least two intervals")
        if any(interval <= 0 for interval in intervals):
            raise ValueError("sample_intervals must contain positive integers")
        if len(set(intervals)) != len(intervals):
            raise ValueError("sample_intervals must not contain duplicates")
        if not math.isfinite(production_scale) or production_scale <= 0.0:
            raise ValueError("production_scale must be finite and positive")

        integer_options = {
            "set_min_step": set_min_step,
            "set_max_step": set_max_step,
            "pulse_duration": pulse_duration,
            "context_start_step": context_start_step,
            "context_duration": context_duration,
            "minimum_competing_go_separation": minimum_competing_go_separation,
            "post_go_duration": post_go_duration,
            "distractor_pulse_duration": distractor_pulse_duration,
            "seed": seed,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in integer_options.values()
        ):
            raise TypeError(
                "step, duration, separation, and seed options must be integers"
            )
        if set_min_step < max(intervals) or set_max_step < set_min_step:
            raise ValueError(
                "expected max(sample_intervals) <= set_min_step <= set_max_step"
            )
        if pulse_duration <= 0 or context_duration <= 0:
            raise ValueError("cue and context durations must be positive")
        if distractor_pulse_duration <= 0:
            raise ValueError("distractor_pulse_duration must be positive")
        if min(intervals) < pulse_duration:
            raise ValueError("sample intervals must prevent Ready/Set pulse overlap")
        earliest_ready = set_min_step - max(intervals)
        if (
            context_start_step < 0
            or context_start_step + context_duration > earliest_ready
        ):
            raise ValueError(
                "context pulse must end no later than the earliest possible Ready pulse"
            )
        if minimum_competing_go_separation <= 0:
            raise ValueError("minimum_competing_go_separation must be positive")
        if not math.isfinite(context_gain) or context_gain <= 0.0:
            raise ValueError("context_gain must be finite and positive")
        if not math.isfinite(target_sigma) or target_sigma <= 0.0:
            raise ValueError("target_sigma must be finite and positive")
        minimum_post_go = math.ceil(4.0 * target_sigma)
        if post_go_duration < minimum_post_go:
            raise ValueError(
                "post_go_duration must cover at least four target standard "
                f"deviations ({minimum_post_go} steps)"
            )
        if (
            not math.isfinite(cue_gain_min)
            or not math.isfinite(cue_gain_max)
            or cue_gain_min <= 0.0
            or cue_gain_max < cue_gain_min
        ):
            raise ValueError("cue gain bounds must be positive and ordered")
        if not math.isfinite(input_noise_std) or input_noise_std < 0.0:
            raise ValueError("input_noise_std must be finite and non-negative")
        if (
            not math.isfinite(mean_distractors_per_trial)
            or mean_distractors_per_trial < 0.0
        ):
            raise ValueError(
                "mean_distractors_per_trial must be finite and non-negative"
            )
        if not math.isfinite(distractor_amplitude) or distractor_amplitude < 0.0:
            raise ValueError("distractor_amplitude must be finite and non-negative")

        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.sample_interval_values = torch.tensor(intervals, dtype=torch.long)
        self.production_scale = float(production_scale)
        self.set_min_step = int(set_min_step)
        self.set_max_step = int(set_max_step)
        self.pulse_duration = int(pulse_duration)
        self.context_start_step = int(context_start_step)
        self.context_duration = int(context_duration)
        self.context_gain = float(context_gain)
        self.minimum_competing_go_separation = int(minimum_competing_go_separation)
        self.target_sigma = float(target_sigma)
        self.post_go_duration = int(post_go_duration)
        self.cue_gain_min = float(cue_gain_min)
        self.cue_gain_max = float(cue_gain_max)
        self.input_noise_std = float(input_noise_std)
        self.mean_distractors_per_trial = float(mean_distractors_per_trial)
        self.distractor_amplitude = float(distractor_amplitude)
        self.distractor_pulse_duration = int(distractor_pulse_duration)
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 7

        maximum_production = round(self.production_scale * max(intervals))
        minimum_production = round(self.production_scale * min(intervals))
        if minimum_production < self.pulse_duration:
            raise ValueError("production intervals must extend beyond the Set pulse")
        required_duration = (
            self.set_max_step + maximum_production + self.post_go_duration + 1
        )
        if sequence_duration is None:
            sequence_duration = required_duration
        if isinstance(sequence_duration, bool) or not isinstance(
            sequence_duration, Integral
        ):
            raise TypeError("sequence_duration must be an integer or None")
        if sequence_duration < required_duration:
            raise ValueError(
                "sequence_duration is too short for the largest contextual "
                f"Ready-Set-Go trial: need at least {required_duration}, "
                f"got {sequence_duration}"
            )
        self.sequence_duration = int(sequence_duration)
        if self.distractor_pulse_duration > self.sequence_duration:
            raise ValueError("distractor pulse cannot exceed sequence duration")

        # Fail before sampling if any scheduled ordered interval pair has no
        # Set-time pair with sufficiently distinct implied Go times.
        for first_interval in intervals:
            first_production = round(self.production_scale * first_interval)
            for second_interval in intervals:
                if second_interval == first_interval:
                    continue
                second_production = round(self.production_scale * second_interval)
                first_go_bounds = (
                    self.set_min_step + first_production,
                    self.set_max_step + first_production,
                )
                second_go_bounds = (
                    self.set_min_step + second_production,
                    self.set_max_step + second_production,
                )
                maximum_separation = max(
                    abs(first_go_bounds[0] - second_go_bounds[1]),
                    abs(first_go_bounds[1] - second_go_bounds[0]),
                )
                if maximum_separation < self.minimum_competing_go_separation:
                    raise ValueError(
                        "minimum_competing_go_separation is infeasible for "
                        "the configured intervals and Set-time window"
                    )

        self.inputs = torch.zeros(
            self.n_samples, self.sequence_duration, self.input_dim
        )
        self.targets = torch.zeros(self.n_samples, self.sequence_duration)
        self.context_indices = torch.zeros(self.n_samples, dtype=torch.long)
        self.stream_intervals = torch.zeros(self.n_samples, 2, dtype=torch.long)
        self.stream_production_intervals = torch.zeros(
            self.n_samples, 2, dtype=torch.long
        )
        self.stream_ready_steps = torch.zeros(self.n_samples, 2, dtype=torch.long)
        self.stream_set_steps = torch.zeros(self.n_samples, 2, dtype=torch.long)
        self.stream_go_steps = torch.zeros(self.n_samples, 2, dtype=torch.long)
        self.stream_cue_gains = torch.zeros(self.n_samples, 2)
        self.sample_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.production_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.ready_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.set_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.go_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.cue_gains = torch.zeros(self.n_samples)
        self.competing_go_separations = torch.zeros(self.n_samples, dtype=torch.long)
        self.distractor_mask = torch.zeros(
            self.n_samples, self.sequence_duration, dtype=torch.bool
        )
        self.pair_ids = torch.full((self.n_samples,), -1, dtype=torch.long)
        self.counterfactual_indices = torch.full(
            (self.n_samples,), -1, dtype=torch.long
        )
        self.is_counterfactual_paired = True
        self.n_counterfactual_pairs = self.n_samples // 2

        generator = torch.Generator().manual_seed(self.seed)
        n_pairs = self.n_counterfactual_pairs
        pair_slots = torch.randperm(self.n_samples, generator=generator).reshape(
            n_pairs, 2
        )
        schedule_order = torch.randperm(n_pairs, generator=generator)
        time = torch.arange(self.sequence_duration, dtype=torch.float32)
        set_candidates = tuple(range(self.set_min_step, self.set_max_step + 1))

        def sample_set_pair(stream_productions: list[int]) -> tuple[int, int]:
            # Rejection from two independent uniforms is exchangeable and
            # uniform over valid ordered Set-time pairs.  The deterministic
            # fallback prevents pathological runtimes near the feasibility edge.
            for _ in range(128):
                candidates = torch.randint(
                    self.set_min_step,
                    self.set_max_step + 1,
                    (2,),
                    generator=generator,
                ).tolist()
                separation = abs(
                    candidates[0]
                    + stream_productions[0]
                    - candidates[1]
                    - stream_productions[1]
                )
                if separation >= self.minimum_competing_go_separation:
                    return int(candidates[0]), int(candidates[1])

            valid_pairs = tuple(
                (first, second)
                for first in set_candidates
                for second in set_candidates
                if abs(first + stream_productions[0] - second - stream_productions[1])
                >= self.minimum_competing_go_separation
            )
            choice = int(
                torch.randint(0, len(valid_pairs), (1,), generator=generator).item()
            )
            return valid_pairs[choice]

        for pair_id in range(n_pairs):
            schedule_index = int(schedule_order[pair_id])
            first_interval_index = schedule_index % len(intervals)
            offset = 1 + (schedule_index // len(intervals)) % (len(intervals) - 1)
            second_interval_index = (first_interval_index + offset) % len(intervals)
            stream_intervals = [
                intervals[first_interval_index],
                intervals[second_interval_index],
            ]
            stream_productions = [
                round(self.production_scale * interval) for interval in stream_intervals
            ]
            stream_sets = list(sample_set_pair(stream_productions))
            stream_readies = [
                set_step - interval
                for set_step, interval in zip(
                    stream_sets, stream_intervals, strict=True
                )
            ]
            stream_goes = [
                set_step + production
                for set_step, production in zip(
                    stream_sets, stream_productions, strict=True
                )
            ]
            gains = (
                torch.empty(2)
                .uniform_(
                    self.cue_gain_min,
                    self.cue_gain_max,
                    generator=generator,
                )
                .tolist()
            )
            go_separation = abs(stream_goes[1] - stream_goes[0])

            base_input = torch.zeros(self.sequence_duration, self.input_dim)
            for stream in range(2):
                ready_channel = 2 * stream
                set_channel = ready_channel + 1
                ready_step = stream_readies[stream]
                set_step = stream_sets[stream]
                gain = gains[stream]
                base_input[
                    ready_step : ready_step + self.pulse_duration,
                    ready_channel,
                ] = gain
                base_input[
                    set_step : set_step + self.pulse_duration,
                    set_channel,
                ] = gain

            base_distractor_mask = torch.zeros(self.sequence_duration, dtype=torch.bool)
            n_distractors = int(
                torch.poisson(
                    torch.tensor(self.mean_distractors_per_trial),
                    generator=generator,
                ).item()
            )
            if n_distractors > 0 and self.distractor_amplitude > 0.0:
                latest_start = self.sequence_duration - self.distractor_pulse_duration
                for _ in range(n_distractors):
                    start = int(
                        torch.randint(
                            0,
                            latest_start + 1,
                            (1,),
                            generator=generator,
                        ).item()
                    )
                    stop = start + self.distractor_pulse_duration
                    base_input[start:stop, 6] += self.distractor_amplitude
                    base_distractor_mask[start:stop] = True

            shared_noise = None
            if self.input_noise_std > 0.0:
                shared_noise = self.input_noise_std * torch.randn(
                    self.sequence_duration,
                    self.input_dim,
                    generator=generator,
                )

            first_slot, second_slot = (int(value) for value in pair_slots[pair_id])
            slots = (first_slot, second_slot)
            for selected_stream, sample_idx in enumerate(slots):
                sample_input = base_input.clone()
                context_channel = 4 + selected_stream
                sample_input[
                    self.context_start_step : self.context_start_step
                    + self.context_duration,
                    context_channel,
                ] = self.context_gain
                if shared_noise is not None:
                    sample_input += shared_noise
                if self.positive_input_encoding:
                    sample_input.clamp_(min=0.0)

                self.inputs[sample_idx] = sample_input
                self.context_indices[sample_idx] = selected_stream
                self.stream_intervals[sample_idx] = torch.tensor(stream_intervals)
                self.stream_production_intervals[sample_idx] = torch.tensor(
                    stream_productions
                )
                self.stream_ready_steps[sample_idx] = torch.tensor(stream_readies)
                self.stream_set_steps[sample_idx] = torch.tensor(stream_sets)
                self.stream_go_steps[sample_idx] = torch.tensor(stream_goes)
                self.stream_cue_gains[sample_idx] = torch.tensor(gains)
                self.sample_intervals[sample_idx] = stream_intervals[selected_stream]
                self.production_intervals[sample_idx] = stream_productions[
                    selected_stream
                ]
                self.ready_steps[sample_idx] = stream_readies[selected_stream]
                self.set_steps[sample_idx] = stream_sets[selected_stream]
                self.go_steps[sample_idx] = stream_goes[selected_stream]
                self.cue_gains[sample_idx] = gains[selected_stream]
                self.competing_go_separations[sample_idx] = go_separation
                self.distractor_mask[sample_idx] = base_distractor_mask
                self.pair_ids[sample_idx] = pair_id

                target = torch.exp(
                    -0.5
                    * (
                        (time - float(stream_goes[selected_stream])) / self.target_sigma
                    ).square()
                )
                self.targets[sample_idx] = target / target.sum().clamp_min(1e-12)

            self.counterfactual_indices[first_slot] = second_slot
            self.counterfactual_indices[second_slot] = first_slot

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


__all__ = [
    "ContextDependentDecisionDataset",
    "ContextualReadySetGoDataset",
    "HierarchicalTemporalDataset",
    "ReadySetGoDataset",
    "SwitchingContextDecisionDataset",
    "TimescaleGeneralizationDataset",
]
