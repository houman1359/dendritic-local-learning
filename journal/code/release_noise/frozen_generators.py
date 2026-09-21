"""Frozen generator bodies; see ORIGINS.json for source identity and scope.

These symbols preserve their original implementation. The legacy ambiguous name
is not the public dispatcher; callers must choose an explicit generator below.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split


def load_mnist_as_datasets(**kwargs):
    # The installable committed core ships the loader and its dependencies.
    from dendritic_modeling.datasets.standard_datasets import load_mnist_as_datasets as loader
    return loader(**kwargs)


class NoisyLineDataset(Dataset):
    """
    line dataset but add random Gaussian noise
    """

    def __init__(self, n_samples=10000, image_size=10, noise_level=0.2):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.noise_level = noise_level
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            noise = np.random.normal(0, self.noise_level, size=img.shape)
            img += noise.astype(np.float32)
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _make_fixed_noise_projection(
    input_dim: int,
    latent_dim: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a fixed random projection that induces correlated input noise."""
    generator = torch.Generator().manual_seed(int(seed))
    projection = torch.randn(input_dim, latent_dim, generator=generator, dtype=dtype)
    projection = projection / projection.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return projection


def _add_projected_gaussian_noise(
    x: torch.Tensor,
    projection: torch.Tensor,
    *,
    sigma_task: float,
    seed: int,
    gain_multiplier: float = 1.0,
    clamp_inputs: bool = True,
) -> torch.Tensor:
    """Add fixed-projection correlated Gaussian noise to a tensor dataset."""
    generator = torch.Generator().manual_seed(int(seed))
    latent = torch.randn(
        x.shape[0], projection.shape[1], generator=generator, dtype=x.dtype
    )
    noise = float(sigma_task) * (latent @ projection.to(dtype=x.dtype).T)
    x_noisy = (x + noise) * float(gain_multiplier)
    if clamp_inputs:
        x_noisy = x_noisy.clamp(0.0, 1.0)
    return x_noisy


def load_noise_resilience_mnist(
    *,
    train_valid_split=0.8,
    flatten=True,
    normalize=False,
    task_data_path=None,
    sigma_task: float = 1.5,
    noise_latent_dim: int = 50,
    projection_seed: int = 0,
    train_noise_seed: int = 1,
    valid_noise_seed: int = 2,
    test_noise_seed: int = 3,
    train_gain: float = 1.0,
    valid_gain: float = 1.0,
    test_gain: float = 1.0,
    clamp_inputs: bool = True,
):
    """MNIST with fixed low-rank projected Gaussian channel noise.

    The same random projection is reused across splits. Independent latent
    Gaussian draws are projected through that fixed matrix to create
    structured, correlated corruption while keeping the task deterministic
    under the provided seeds.
    """
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
    )

    x_train, y_train = train_ds.tensors
    x_valid, y_valid = valid_ds.tensors
    x_test, y_test = test_ds.tensors

    input_dim = int(x_train.shape[1]) if x_train.ndim == 2 else int(x_train[0].numel())
    projection = _make_fixed_noise_projection(
        input_dim=input_dim,
        latent_dim=int(noise_latent_dim),
        seed=int(projection_seed),
        dtype=x_train.dtype,
    )

    x_train_noisy = _add_projected_gaussian_noise(
        x_train,
        projection,
        sigma_task=float(sigma_task),
        seed=int(train_noise_seed),
        gain_multiplier=float(train_gain),
        clamp_inputs=bool(clamp_inputs),
    )
    x_valid_noisy = _add_projected_gaussian_noise(
        x_valid,
        projection,
        sigma_task=float(sigma_task),
        seed=int(valid_noise_seed),
        gain_multiplier=float(valid_gain),
        clamp_inputs=bool(clamp_inputs),
    )
    x_test_noisy = _add_projected_gaussian_noise(
        x_test,
        projection,
        sigma_task=float(sigma_task),
        seed=int(test_noise_seed),
        gain_multiplier=float(test_gain),
        clamp_inputs=bool(clamp_inputs),
    )

    return {
        "train": TensorDataset(x_train_noisy, y_train.clone()),
        "valid": TensorDataset(x_valid_noisy, y_valid.clone()),
        "test": TensorDataset(x_test_noisy, y_test.clone()),
    }
