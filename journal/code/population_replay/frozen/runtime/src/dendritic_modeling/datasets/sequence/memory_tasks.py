"""Sequence dataset task implementations."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SequentialMNISTDataset(Dataset):
    """Sequential MNIST: processes image pixel-by-pixel or row-by-row.

    Args:
        mode: "pixel" for pixel-by-pixel (seq_len=784, dim=1) or
              "row" for row-by-row (seq_len=28, dim=28).
        train: If True, training set; else test set.
        data_path: Path to MNIST data directory.
        normalize: Whether to normalize to [0, 1].
    """

    def __init__(
        self,
        mode: str = "pixel",
        train: bool = True,
        data_path: str | None = None,
        normalize: bool = True,
    ):
        from torchvision import datasets, transforms

        transform = (
            transforms.ToTensor()
            if normalize
            else transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
            )
        )
        root = data_path or "./data"
        mnist = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        self.mode = mode
        images = mnist.data.float() / 255.0 if normalize else mnist.data.float()
        self.labels = mnist.targets

        if mode == "pixel":
            # [N, 28, 28] -> [N, 784, 1]
            self.sequences = images.reshape(-1, 784, 1)
        elif mode == "row":
            # [N, 28, 28] -> [N, 28, 28]
            self.sequences = images
        else:
            raise ValueError(f"mode must be 'pixel' or 'row', got '{mode}'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class CopyTaskDataset(Dataset):
    """Copy task: memorize a sequence and reproduce it after a delay.

    Input: [total_len, n_symbols + 2] where total_len = seq_len + delay + seq_len.
    Target: [total_len] with ignore_index=-100 for non-output positions,
            symbol indices for the output phase (last seq_len timesteps).

    Compatible with both output_mode="all" (many-to-many with ignore_index)
    and output_mode="last" (use only final timestep, though seq_len>1 needs "all").

    For training with output_mode="all", use CrossEntropyLoss(ignore_index=-100).

    Args:
        n_samples: Number of samples to generate.
        seq_len: Length of the sequence to memorize.
        delay: Number of blank timesteps between input and output.
        n_symbols: Number of distinct symbols (excluding blank and trigger).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 10,
        delay: int = 50,
        n_symbols: int = 8,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.delay = delay
        self.n_symbols = n_symbols
        self.total_len = seq_len + delay + seq_len
        self.input_dim = n_symbols + 2  # symbols + blank + trigger

        # Pre-generate all samples
        self.inputs = torch.zeros(n_samples, self.total_len, self.input_dim)
        # Full-length targets: -100 (ignore) for non-output, symbols for output phase
        self.targets = torch.full(
            (n_samples, self.total_len), self.IGNORE_INDEX, dtype=torch.long
        )

        for i in range(n_samples):
            symbols = torch.randint(0, n_symbols, (seq_len,))
            # Input phase: one-hot symbols
            for t, s in enumerate(symbols):
                self.inputs[i, t, s] = 1.0
            # Delay phase: blank (all zeros, could add blank channel)
            # Trigger marker at start of output phase
            self.inputs[i, seq_len + delay, n_symbols + 1] = 1.0
            # Output phase targets: symbol indices
            output_start = seq_len + delay
            self.targets[i, output_start : output_start + seq_len] = symbols

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class AddingProblemDataset(Dataset):
    """Adding problem: sum two marked entries in a long sequence.

    Input: [seq_len, 2] -- first channel is random uniform, second is mask.
    Target: scalar (sum of the two marked entries).

    Args:
        n_samples: Number of samples.
        seq_len: Sequence length.
    """

    def __init__(self, n_samples: int = 10000, seq_len: int = 200):
        self.n_samples = n_samples
        self.seq_len = seq_len

        self.inputs = torch.zeros(n_samples, seq_len, 2)
        self.targets = torch.zeros(n_samples, 1)

        for i in range(n_samples):
            # Random values in [0, 1)
            self.inputs[i, :, 0] = torch.rand(seq_len)
            # Mark two random positions
            marks = torch.randperm(seq_len)[:2]
            self.inputs[i, marks[0], 1] = 1.0
            self.inputs[i, marks[1], 1] = 1.0
            # Target is sum of the marked values
            self.targets[i] = self.inputs[i, marks[0], 0] + self.inputs[i, marks[1], 0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class VariableDelayDMSDataset(Dataset):
    """Variable-delay delayed match-to-sample task.

    Stimulus presented, delay period, then match/non-match probe.
    Target: binary (match=1, non-match=0).

    Args:
        n_samples: Number of samples.
        stimulus_dim: Dimensionality of the stimulus.
        stimulus_duration: Number of timesteps for stimulus presentation.
        min_delay: Minimum delay in timesteps.
        max_delay: Maximum delay in timesteps.
        probe_duration: Number of timesteps for probe presentation.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        stimulus_dim: int = 16,
        stimulus_duration: int = 10,
        min_delay: int = 10,
        max_delay: int = 100,
        probe_duration: int = 10,
    ):
        self.n_samples = n_samples
        self.stimulus_dim = stimulus_dim
        self.stimulus_duration = stimulus_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.probe_duration = probe_duration
        self.max_total_len = stimulus_duration + max_delay + probe_duration
        self.input_dim = stimulus_dim + 1  # stimulus + timing channel

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
            self.delays[i] = delay
            total_len = stimulus_duration + delay + probe_duration

            # Generate stimulus
            stimulus = torch.randn(stimulus_dim)
            stimulus = stimulus / stimulus.norm()

            # Stimulus phase
            for t in range(stimulus_duration):
                self.inputs[i, t, :stimulus_dim] = stimulus

            # Timing channel: ramp during delay
            for t in range(stimulus_duration, stimulus_duration + delay):
                self.inputs[i, t, stimulus_dim] = (t - stimulus_duration) / delay

            # Probe phase: match or non-match (50/50)
            is_match = torch.rand(1).item() > 0.5
            if is_match:
                probe = stimulus
                self.targets[i] = 1
            else:
                probe = torch.randn(stimulus_dim)
                probe = probe / probe.norm()
                self.targets[i] = 0

            for t in range(
                stimulus_duration + delay, min(total_len, self.max_total_len)
            ):
                self.inputs[i, t, :stimulus_dim] = probe

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        delays = self.delays[idx]
        if delays.dim() == 0:
            # Single index: return scalar seq_len
            seq_len = self.stimulus_duration + delays.item() + self.probe_duration
        else:
            # Slice/list index: return tensor of seq_lengths
            seq_len = self.stimulus_duration + delays + self.probe_duration
        return self.inputs[idx], self.targets[idx], seq_len


__all__ = [
    "AddingProblemDataset",
    "CopyTaskDataset",
    "SequentialMNISTDataset",
    "VariableDelayDMSDataset",
]
