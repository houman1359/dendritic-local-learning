"""Paired arithmetic-composition datasets."""

import torch
from torch.utils.data import Dataset, TensorDataset

from .indexing import SliceSafeDataset


class PairedArithmeticDataset(Dataset):
    """Pair two examples and predict an arithmetic composition of their labels.

    Inputs are emitted either as flattened concatenations or as side-by-side images.
    Optional context can be appended as a one-hot vector (flattened mode) or as
    constant context planes (image mode). This keeps the task usable both for
    plain MLP-style inputs and for future conv-stem experiments.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        *,
        task_mode: str = "sum_mod10",
        flatten: bool = True,
        include_context: bool = False,
        context_dim: int = 2,
        context_seed: int = 0,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.left_indices = left_indices.long()
        self.right_indices = right_indices.long()
        self.task_mode = str(task_mode)
        self.flatten = bool(flatten)
        self.include_context = bool(include_context)
        self.context_dim = int(context_dim)

        if self.include_context:
            generator = torch.Generator().manual_seed(int(context_seed))
            self.context = torch.randint(
                0,
                self.context_dim,
                (len(self.left_indices),),
                generator=generator,
                dtype=torch.long,
            )
        else:
            self.context = None

    def __len__(self) -> int:
        return int(self.left_indices.numel())

    def _compose_input(
        self, left_x: torch.Tensor, right_x: torch.Tensor, context_idx: int | None
    ) -> torch.Tensor:
        if self.flatten:
            pair_input = torch.cat([left_x.reshape(-1), right_x.reshape(-1)], dim=0)
            if context_idx is not None:
                ctx = torch.zeros(self.context_dim, dtype=pair_input.dtype)
                ctx[context_idx] = 1.0
                pair_input = torch.cat([pair_input, ctx], dim=0)
            return pair_input

        # Side-by-side image layout for future convolutional frontends.
        if left_x.ndim >= 2:
            pair_input = torch.cat([left_x, right_x], dim=-1)
        else:
            pair_input = torch.cat([left_x, right_x], dim=0)

        if context_idx is None:
            return pair_input

        if pair_input.ndim == 3:
            _, height, width = pair_input.shape
            ctx_planes = torch.zeros(
                self.context_dim, height, width, dtype=pair_input.dtype
            )
            ctx_planes[context_idx].fill_(1.0)
            return torch.cat([pair_input, ctx_planes], dim=0)

        ctx = torch.zeros(self.context_dim, dtype=pair_input.dtype)
        ctx[context_idx] = 1.0
        return torch.cat([pair_input.reshape(-1), ctx], dim=0)

    def _compose_label(
        self,
        left_label: torch.Tensor,
        right_label: torch.Tensor,
        context_idx: int | None,
    ) -> torch.Tensor:
        yl = int(left_label.item())
        yr = int(right_label.item())

        if self.task_mode == "sum_mod10":
            target = (yl + yr) % 10
        elif self.task_mode == "diff_mod10":
            target = (yl - yr) % 10
        elif self.task_mode == "context_sum_diff":
            if context_idx is None:
                raise ValueError("context_sum_diff requires context indices.")
            target = (yl + yr) % 10 if context_idx == 0 else (yl - yr) % 10
        else:
            raise ValueError(f"Unknown paired arithmetic mode: {self.task_mode}")

        return torch.tensor(target, dtype=torch.long)

    def __getitem__(self, idx):
        li = int(self.left_indices[idx].item())
        ri = int(self.right_indices[idx].item())
        context_idx = (
            int(self.context[idx].item()) if self.context is not None else None
        )

        left_x = self.inputs[li]
        right_x = self.inputs[ri]
        left_y = self.labels[li]
        right_y = self.labels[ri]

        x = self._compose_input(left_x, right_x, context_idx)
        y = self._compose_label(left_y, right_y, context_idx)
        return x, y


def _make_pair_indices(
    n_examples: int,
    *,
    shuffle_iterations: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic left/right pair indices across repeated shuffles."""
    half = n_examples // 2
    generator = torch.Generator().manual_seed(int(seed))
    left_batches = []
    right_batches = []

    for _ in range(max(1, int(shuffle_iterations))):
        perm = torch.randperm(n_examples, generator=generator)
        left_batches.append(perm[:half])
        right_batches.append(perm[half : half + half])

    return torch.cat(left_batches, dim=0), torch.cat(right_batches, dim=0)


def _build_paired_arithmetic_dataset(
    dataset: TensorDataset,
    *,
    shuffle_iterations: int = 1,
    task_mode: str = "sum_mod10",
    flatten: bool = True,
    include_context: bool = False,
    context_seed: int = 0,
    pair_seed: int = 0,
) -> Dataset:
    """Convert a TensorDataset into a paired arithmetic-composition dataset."""
    inputs, labels = dataset.tensors
    left_indices, right_indices = _make_pair_indices(
        len(dataset), shuffle_iterations=shuffle_iterations, seed=pair_seed
    )
    return SliceSafeDataset(
        PairedArithmeticDataset(
            inputs,
            labels,
            left_indices,
            right_indices,
            task_mode=task_mode,
            flatten=flatten,
            include_context=include_context,
            context_seed=context_seed,
        )
    )


def _build_iterative_modulo10_dataset(
    dataset: Dataset,
    *,
    shuffle_iterations: int,
    pair_seed: int,
) -> TensorDataset:
    """Build the deterministic split-half modulo-10 pairing dataset."""
    generator = torch.Generator().manual_seed(int(pair_seed))
    indices = torch.randperm(len(dataset), generator=generator)
    n_pairs = len(dataset) // 2
    left_indices = indices[:n_pairs]
    right_indices = indices[n_pairs : n_pairs + n_pairs]

    left_inputs, left_labels = dataset[left_indices]
    right_inputs, right_labels = dataset[right_indices]

    inputs = []
    labels = []
    for _ in range(int(shuffle_iterations)):
        inputs.append(torch.cat([left_inputs, right_inputs], dim=-1))
        labels.append((left_labels + right_labels) % 10)

        left_perm = torch.randperm(left_inputs.shape[0], generator=generator)
        left_inputs = left_inputs[left_perm]
        left_labels = left_labels[left_perm]

        right_perm = torch.randperm(right_inputs.shape[0], generator=generator)
        right_inputs = right_inputs[right_perm]
        right_labels = right_labels[right_perm]

    return TensorDataset(torch.cat(inputs, dim=0), torch.cat(labels, dim=0))


__all__ = [
    "PairedArithmeticDataset",
    "_build_iterative_modulo10_dataset",
    "_build_paired_arithmetic_dataset",
    "_make_pair_indices",
]
