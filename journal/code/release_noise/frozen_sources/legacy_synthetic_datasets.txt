# synthetic_datasets.py

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

# Import standard dataset functions from the main package
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dendritic_modeling.datasets.standard_datasets import (
    get_data_directory,
    load_mnist_as_datasets,
    load_cifar10_as_datasets,
    load_imagenet_as_datasets,
    load_mnist_modulo10,
)

########################################
# Original tasks: line, center_surround, orientation_bars, multi_xor, plus standard MNIST
########################################


class LineDataset(Dataset):
    def __init__(self, n_samples=10000, image_size=10):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
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
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CenterSurroundDataset(Dataset):
    def __init__(self, n_samples=10000, image_size=15):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        radius = self.image_size // 4
        cx = cy = self.image_size // 2
        for _ in range(self.n_samples):
            X.append(
                np.zeros((self.image_size, self.image_size), dtype=np.float32).flatten()
            )
            label = np.random.choice([0, 1, 2])
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            if label == 0:
                val = np.random.rand() * 0.2 + 0.4
                img[:] = val
            elif label == 1:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        if dist < radius:
                            img[i, j] = 1.0
                        else:
                            img[i, j] = 0.0
            else:
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        if dist < radius:
                            img[i, j] = 0.0
                        else:
                            img[i, j] = 1.0
            X[-1] = img.flatten()
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OrientationBarsDataset(Dataset):
    def __init__(self, n_samples=10000, image_size=16, n_orient_classes=8):
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_orient_classes = n_orient_classes
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            label = np.random.randint(0, self.n_orient_classes)
            angle_deg = (180 / self.n_orient_classes) * label
            angle_rad = np.deg2rad(angle_deg)
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            cx = cy = self.image_size // 2
            for r in range(-cx, cx):
                x = int(cx + r * np.cos(angle_rad))
                y = int(cy + r * np.sin(angle_rad))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    img[y, x] = 1.0
            X.append(img.flatten())
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MultiXORDataset(Dataset):
    def __init__(self, n_samples=10000, n_bits=8):
        super().__init__()
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        X, Y = [], []
        for _ in range(self.n_samples):
            bits = np.random.randint(0, 2, size=self.n_bits).astype(np.int32)
            y1 = bits[0] ^ bits[1]
            y2 = bits[2] ^ bits[3] ^ bits[4]
            label = (y1 << 1) + y2
            X.append(bits)
            Y.append(label)
        return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_mnist_modulo10(
    shuffle_iterations=2,
    train_valid_split=1,
    flatten=False,
    normalize=False,
    cache_dir=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        data_path=cache_dir,  # Legacy parameter
        output_dir=output_dir,  # Legacy parameter
        task_data_path=task_data_path,
        outputs_dir=outputs_dir,
    )
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    for key, ds in data.items():
        ix = torch.randperm(len(ds))
        ix0 = ix[: int(len(ds) / 2)]
        ix1 = ix[int(len(ds) / 2) :]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]
        assert label0.shape[0] == label1.shape[0]

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim=-1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0])
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0])
            input1 = input1[ix1]
            label1 = label1[ix1]

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        data[key] = TensorDataset(inputs, labels)

    return data


class InfoShuntingDataset(Dataset):
    """
    half = Gaussian, half = Uniform; label=0 or 1
    """

    def __init__(self, n_samples=20000, input_dim=10):
        super().__init__()
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.n_samples // 2
        gauss = np.random.normal(0, 1, size=(half, self.input_dim)).astype(np.float32)
        unif = np.random.uniform(
            -1, 1, size=(self.n_samples - half, self.input_dim)
        ).astype(np.float32)
        Xfull = np.concatenate([gauss, unif], axis=0)
        Yfull = np.concatenate([np.zeros(half), np.ones(self.n_samples - half)])
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        Xfull = Xfull[idx]
        Yfull = Yfull[idx]
        return torch.tensor(Xfull), torch.tensor(Yfull, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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


class SparseMNISTDataset(Dataset):
    """
    binarize MNIST
    """

    def __init__(self, base_mnist, threshold=0.3):
        super().__init__()
        self.base_mnist = base_mnist
        self.threshold = threshold

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        arr = np.array(img, dtype=np.float32) / 255.0
        bin_arr = (arr > self.threshold).astype(np.float32)
        return torch.tensor(bin_arr.flatten(), dtype=torch.float32), label


class ContextualFigureGroundMNIST(Dataset):
    """
    left half random noise, right half original digit
    """

    def __init__(self, base_mnist, noise_level=0.1):
        super().__init__()
        self.base_mnist = base_mnist
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, label = self.base_mnist[idx]
        img_arr = np.array(img, dtype=np.float32)
        w = 28
        new_img = np.random.rand(w, w) * self.noise_level
        boundary = w // 2
        new_img[:, boundary:] = img_arr[:, boundary:]
        return torch.tensor(new_img.flatten(), dtype=torch.float32), label


class MultiTaskLineDataset(Dataset):
    """
    stub: half data is line dataset (0,1,2), second half inverts labels
    """

    def __init__(self, total_size=12000, image_size=10):
        super().__init__()
        self.total_size = total_size
        self.image_size = image_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.total_size // 2
        Xlist, Ylist = [], []
        for _ in range(half):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            Xlist.append(img.flatten())
            Ylist.append(label)
        for _ in range(self.total_size - half):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            label = np.random.choice([0, 1, 2])
            if label == 1:
                row = np.random.randint(0, self.image_size)
                img[row, :] = 1.0
            elif label == 2:
                col = np.random.randint(0, self.image_size)
                img[:, col] = 1.0
            new_label = {0: 0, 1: 2, 2: 1}[label]
            Xlist.append(img.flatten())
            Ylist.append(new_label + 3)
        Xarr = np.stack(Xlist).astype(np.float32)
        Yarr = np.array(Ylist, dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CompositionalMNIST(Dataset):
    """
    merges two MNIST digits horizontally => label = digit1*10+digit2
    """

    def __init__(self, base_mnist, n_samples=20000):
        super().__init__()
        self.base_mnist = base_mnist
        self.n_samples = n_samples
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        length = len(self.base_mnist)
        Xlist = []
        Ylist = []
        for _ in range(self.n_samples):
            idx1 = np.random.randint(0, length)
            idx2 = np.random.randint(0, length)
            img1, lbl1 = self.base_mnist[idx1]
            img2, lbl2 = self.base_mnist[idx2]
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            comp = np.concatenate([arr1, arr2], axis=1)
            Xlist.append(comp.flatten())
            Ylist.append(lbl1 * 10 + lbl2)
        Xarr = np.stack(Xlist).astype(np.float32)
        Yarr = np.array(Ylist, dtype=np.int64)
        return torch.tensor(Xarr), torch.tensor(Yarr)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_cifar10_as_datasets(
    train_valid_split=1.0,
    flatten=False,
    normalize=False,
    data_path=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    """
    Load the CIFAR-10 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        data_path (str, optional): Path to the data directory
        output_dir (str, optional): Output directory from config to store datasets

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )

    transform = transforms.Compose(tf_list)

    cifar_dir = get_data_directory(
        "cifar", data_path, output_dir, task_data_path, outputs_dir
    )

    train_dataset = datasets.CIFAR10(
        root=cifar_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=cifar_dir, train=False, download=True, transform=transform
    )

    def convert_to_tensor(dataset):
        data = []
        labels = []
        for img, label in dataset:
            data.append(img)
            labels.append(label)
        data_tensor = torch.stack(data)
        label_tensor = torch.tensor(labels)
        if flatten:
            data_tensor = data_tensor.view(data_tensor.size(0), -1)
        return data_tensor, label_tensor

    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )

        x_train, y_train = convert_to_tensor(train_dataset)
        x_valid, y_valid = convert_to_tensor(valid_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)

        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    else:
        x_train, y_train = convert_to_tensor(train_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        # Create a separate validation dataset with the same data as train
        # to avoid reference issues
        valid_ds = torch.utils.data.TensorDataset(x_train.clone(), y_train.clone())
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


########################################
# The single get_unified_datasets(...) function
########################################


def apply_random_flip(images, labels, setAll=False, setValue=0):
    """
    This function modifies images and labels in-place to flip the label
    or shift it. The 'first 3 pixels' logic sets a code:
      pixel0=10 => normal
      pixel1=10 => flip to 9-digit
      pixel2=10 => shift by -1 mod 10
    """
    images_flip = images.clone()
    labels_flip = labels.clone()
    if setAll:
        # set all flips to a single mode
        if setValue == 0:
            # normal
            images_flip[:, 0] = 10
            images_flip[:, 1] = 0
            images_flip[:, 2] = 0
        elif setValue == 1:
            # 9 - label
            labels_flip = 9 - labels_flip
            images_flip[:, 0] = 0
            images_flip[:, 1] = 10
            images_flip[:, 2] = 0
        else:
            # label-1 mod 10
            labels_flip = (labels_flip - 1) % 10
            images_flip[:, 0] = 0
            images_flip[:, 1] = 0
            images_flip[:, 2] = 10
        return images_flip, labels_flip, None
    else:
        # random assignment
        randFlips = torch.randint(
            0, 3, (images_flip.shape[0],), device=images_flip.device
        )
        # mode=0 => normal, mode=1 => flip 9-lbl, mode=2 => lbl-1 mod10
        # set the first pixel code
        images_flip[randFlips == 0, 0] = 10
        images_flip[randFlips == 0, 1] = 0
        images_flip[randFlips == 0, 2] = 0

        labels_flip[randFlips == 1] = 9 - labels_flip[randFlips == 1]
        images_flip[randFlips == 1, 0] = 0
        images_flip[randFlips == 1, 1] = 10
        images_flip[randFlips == 1, 2] = 0

        labels_flip[randFlips == 2] = (labels_flip[randFlips == 2] - 1) % 10
        images_flip[randFlips == 2, 0] = 0
        images_flip[randFlips == 2, 1] = 0
        images_flip[randFlips == 2, 2] = 10

        return images_flip, labels_flip, randFlips


class RandomFlipMNISTDataset(Dataset):
    """
    Loads standard MNIST and applies the apply_random_flip() transform
    to each sample in __getitem__ if random_flip=True.
    This modifies the first 3 pixels and possibly flips/shifts the label.
    """

    def __init__(
        self,
        split="train",
        transform=None,
        data_path=None,
        output_dir=None,
        random_flip=True,
        setAll=None,
        setValue=0,
        task_data_path=None,
        outputs_dir=None,
    ):
        """
        split: 'train', 'valid', or 'test'
        random_flip: if False, no flipping is applied
        setAll: if an integer in [0,1,2], then all flips set to that mode
        setValue: same as above
        """
        super().__init__()
        self.transform = transform
        self.random_flip = random_flip
        self.setAll = setAll
        self.setValue = setValue
        # load full MNIST or we can do partial
        mnist_dir = get_data_directory(
            "mnist", data_path, output_dir, task_data_path, outputs_dir
        )
        if split == "test":
            base_ds = datasets.MNIST(root=mnist_dir, train=False, download=True)
        else:
            base_ds = datasets.MNIST(root=mnist_dir, train=True, download=True)

        # For simplicity, we'll just store the entire split in memory
        data_tensors = []
        label_tensors = []
        for img, lbl in base_ds:
            arr = np.array(img, dtype=np.float32)
            arr = arr.reshape(-1)
            data_tensors.append(arr)
            label_tensors.append(lbl)
        X = torch.tensor(np.stack(data_tensors))
        Y = torch.tensor(label_tensors, dtype=torch.long)

        # If user wants a valid split, we do that
        if split == "train":
            # we'll take first 50k
            self.X = X[:50000]
            self.Y = Y[:50000]
        elif split == "valid":
            # next 10k
            self.X = X[50000:]
            self.Y = Y[50000:]
        else:  # test
            self.X = X
            self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.random_flip:
            if self.setAll is not None:
                x, y, _ = apply_random_flip(
                    x.unsqueeze(0), y.unsqueeze(0), setAll=True, setValue=self.setAll
                )
                x = x.squeeze(0)
                y = y.squeeze(0)
            else:
                x, y, _ = apply_random_flip(
                    x.unsqueeze(0), y.unsqueeze(0), setAll=False
                )
                x = x.squeeze(0)
                y = y.squeeze(0)
        return x, y


class PoissonGeneratorDataset(Dataset):
    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        rate_multiplier: float = 1.0,
    ):
        self.base_inputs = base_inputs
        self.base_labels = base_labels
        self.rate_multiplier = rate_multiplier

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        input = self.base_inputs[idx]
        label = self.base_labels[idx]
        scaled_input = input * self.rate_multiplier
        noisy_scaled_input = torch.poisson(scaled_input)
        noisy_input = noisy_scaled_input / self.rate_multiplier
        return noisy_input, label


def load_poisson_generator_datasets(
    base_dataset: str,
    rate_multiplier: float = 1,
    train_valid_split: float = 0.8,
    flatten: bool = False,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_data_path: Optional[str] = None,
    outputs_dir: Optional[str] = None,
):
    """
    Generate a dataset of Poisson-distributed samples from a base dataset.

    Args:
        base_dataset: The base dataset to sample from
        rate_multiplier: The rate multiplier for the Poisson distribution
        train_valid_split: The proportion of samples to use for training vs validation
        flatten: Whether to flatten the samples
        data_path: The path to save the generated dataset
        output_dir: The output directory to save the generated dataset

    Returns:
        A dataset of Poisson-distributed samples
    """
    # Load the base dataset
    if base_dataset == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            data_path=data_path,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    elif base_dataset == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            data_path=data_path,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    else:
        raise ValueError(f"Unknown base dataset: {base_dataset}")

    train_ds = PoissonGeneratorDataset(*train_ds[:], rate_multiplier=rate_multiplier)
    valid_ds = PoissonGeneratorDataset(*valid_ds[:], rate_multiplier=rate_multiplier)
    test_ds = PoissonGeneratorDataset(*test_ds[:], rate_multiplier=rate_multiplier)

    return train_ds, valid_ds, test_ds


class OrthoNet(torch.nn.Module):
    """Orthogonal sigmoid network (Invertible network)"""

    def __init__(self, d, nb_layers, nonlinearity="sigmoid"):
        super().__init__()
        for i in range(nb_layers):
            # register as we don't want to update it
            self.register_buffer(f"Q_{i}", self.rand_ortho(d))
        self.nb_layer = nb_layers
        if nonlinearity is None:
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise ValueError(f"Invalid nonlinearity: {nonlinearity}")

    def forward(self, x):
        for i in range(self.nb_layer):
            Q = getattr(self, f"Q_{i}")
            x = self.nonlinearity(x @ Q)
        return x

    def rand_ortho(self, d):
        """Generate a random orthogonal matrix of dimension d"""
        Q, R = torch.linalg.qr(torch.randn(d, d))
        return Q


class OrthoNetRepeat(torch.nn.Module):
    """Orthogonal sigmoid network with repeated layer weights plus noise"""

    def __init__(self, d, nb_layers, nonlinearity=None, noise_std=0.1):
        super().__init__()
        # Generate the first layer's orthogonal matrix
        base_Q = self.rand_ortho(d)
        self.register_buffer("Q_0", base_Q)

        # Create subsequent layers by adding noise to the first layer
        for i in range(1, nb_layers):
            noise = torch.randn_like(base_Q) * noise_std
            noisy_Q = base_Q + noise
            # Re-orthogonalize the matrix
            Q, R = torch.linalg.qr(noisy_Q)
            self.register_buffer(f"Q_{i}", Q)

        self.nb_layer = nb_layers
        if nonlinearity is None:
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise ValueError(f"Invalid nonlinearity: {nonlinearity}")

    def forward(self, x):
        for i in range(self.nb_layer):
            Q = getattr(self, f"Q_{i}")
            x = self.nonlinearity(x @ Q)
        return x

    def rand_ortho(self, d):
        """Generate a random orthogonal matrix of dimension d"""
        Q, R = torch.linalg.qr(torch.randn(d, d))
        return Q


class OrthoNetDataset(Dataset):
    """
    Dataset that generates data using invertible nonlinear transformations.
    Input data is uniformly random in [-1,1]^d, transformed through an OrthoNet network.
    Labels are the original inputs, making this an inverse problem dataset.
    """

    def __init__(
        self,
        n_samples=10000,
        input_dim=10,
        n_layers=5,
        nonlinearity="sigmoid",
        positive_input=True,
        seed=None,
        repeat_layers=False,
        noise_std=0.1,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.positive_input = positive_input
        self.repeat_layers = repeat_layers
        self.noise_std = noise_std
        self.seed = seed

        if seed is not None:
            self.set_seed(seed)

        self.data, self.labels = self._generate_data()

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate_data(self):
        # Generate random input data in [-1,1]^d
        if self.positive_input:
            X = torch.rand(self.n_samples, self.input_dim)
        else:
            X = 2 * torch.rand(self.n_samples, self.input_dim) - 1

        # Create and apply transformation
        if self.repeat_layers:
            F = OrthoNetRepeat(
                self.input_dim, self.n_layers, self.nonlinearity, self.noise_std
            )
        else:
            F = OrthoNet(self.input_dim, self.n_layers, self.nonlinearity)

        with torch.no_grad():
            Y = F(X)

        return Y, X

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # if isinstance(idx, slice):
        #     return self.data[idx], self.labels[idx]
        return self.data[idx], self.labels[idx]


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    data_path=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    """
    Load the ImageNet dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        data_path (str, optional): Path to the data directory (legacy)
        output_dir (str, optional): Output directory from config to store datasets (legacy)
        task_data_path (str, optional): Path from task.data_path config
        outputs_dir (str, optional): Directory from outputs.dir config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Check if we have a direct path to existing ImageNet data
    if task_data_path and os.path.exists(os.path.join(task_data_path, "train")):
        # Use the direct path if train directory exists
        imagenet_dir = task_data_path
    else:
        # Use the standard get_data_directory function
        imagenet_dir = get_data_directory(
            "imagenet", data_path, output_dir, task_data_path, outputs_dir
        )

    # ImageNet preprocessing
    tf_list = [transforms.ToTensor()]
    if normalize:
        # ImageNet normalization values
        tf_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    # Add resize transform for ImageNet
    tf_list.insert(0, transforms.Resize((224, 224)))
    transform = transforms.Compose(tf_list)

    try:
        # Try to load ImageNet datasets
        train_dataset = datasets.ImageNet(
            root=imagenet_dir, split="train", transform=transform
        )
        test_dataset = datasets.ImageNet(
            root=imagenet_dir, split="val", transform=transform
        )
    except Exception as e:
        # If ImageNet is not available, create a warning and use a subset of data
        print(f"Warning: Could not load ImageNet from {imagenet_dir}: {e}")
        print("Creating dummy ImageNet-like dataset for testing...")

        # Create dummy data with ImageNet dimensions
        if flatten:
            input_dim = 224 * 224 * 3  # 150528
        else:
            input_dim = (3, 224, 224)

        # Create synthetic data with ImageNet-like properties
        train_size = 100000  # Smaller for testing
        test_size = 10000

        if flatten:
            train_data = torch.randn(train_size, input_dim)
            test_data = torch.randn(test_size, input_dim)
        else:
            train_data = torch.randn(train_size, 3, 224, 224)
            test_data = torch.randn(test_size, 3, 224, 224)

        train_labels = torch.randint(0, 1000, (train_size,))
        test_labels = torch.randint(0, 1000, (test_size,))

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

    # Split training data into train and validation
    if hasattr(train_dataset, "__len__"):
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )
    else:
        # Fallback for when we can't determine size
        valid_dataset = test_dataset

    # Convert to tensor datasets if needed and flatten if requested
    def process_dataset(dataset):
        if flatten and hasattr(dataset, "__getitem__"):
            # Convert to flattened tensor dataset
            data_list = []
            label_list = []
            # Sample a subset for memory efficiency
            subset_size = min(len(dataset), 50000)  # Limit size for memory
            for i in range(subset_size):
                data, label = dataset[i]
                if isinstance(data, torch.Tensor):
                    if len(data.shape) > 1:  # If not already flattened
                        data = data.view(-1)
                data_list.append(data)
                label_list.append(label)

            if data_list:
                data_tensor = torch.stack(data_list)
                label_tensor = torch.tensor(label_list)
                return TensorDataset(data_tensor, label_tensor)

        return dataset

    if flatten:
        train_dataset = process_dataset(train_dataset)
        valid_dataset = process_dataset(valid_dataset)
        test_dataset = process_dataset(test_dataset)

    return train_dataset, valid_dataset, test_dataset


def generate_synthetic_data(num_samples=100, num_features=10):
    data = torch.randn(num_samples, num_features)
    labels = torch.randint(0, 2, (num_samples,))
    return TensorDataset(data, labels)


def get_unified_datasets(task_cfg, train_cfg, output_dir=None, outputs_cfg=None):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration
        train_cfg: Training configuration
        output_dir: Output directory from config to store datasets (legacy)
        outputs_cfg: Outputs configuration containing dir

    Returns:
        dict: Dictionary with 'train', 'valid', and 'test' datasets
    """
    try:
        dataset_name = task_cfg["dataset"]
        task_data_path = task_cfg.get("data_path", None)
        p = task_cfg.get("parameters", {})
    except TypeError:
        dataset_name = task_cfg.dataset
        task_data_path = getattr(task_cfg, "data_path", None)
        p = task_cfg.parameters

    # Extract outputs directory from outputs config
    try:
        outputs_dir = outputs_cfg.get("dir", None) if outputs_cfg else None
    except (TypeError, AttributeError):
        outputs_dir = getattr(outputs_cfg, "dir", None) if outputs_cfg else None

    # If task_data_path is empty string, set it to None
    if task_data_path == "":
        task_data_path = None

    # If outputs_dir is empty string, set it to None
    if outputs_dir == "":
        outputs_dir = None

    # Legacy fallback: If both task_data_path and outputs_dir are None, and output_dir is not provided,
    # use the outputs directory in project root instead of current working directory
    if task_data_path is None and outputs_dir is None and output_dir is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

    train_valid_split = getattr(train_cfg, "train_valid_split", 1.0)
    train_size = p.get("train_size", 8000)
    valid_size = p.get("valid_size", 1000)
    test_size = p.get("test_size", 1000)
    image_size = p.get("image_size", 784)
    flatten = p.get("flatten", True)
    normalize = p.get("normalize", False)
    base_dataset = p.get("base_dataset", "mnist")
    rate_multiplier = p.get("rate_multiplier", 1.0)
    n_orient_classes = p.get("n_orient_classes", 8)
    n_bits = p.get("n_bits", 8)
    total = train_size + valid_size + test_size

    if dataset_name == "line":
        full_ds = LineDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "center_surround":
        full_ds = CenterSurroundDataset(n_samples=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "orientation_bars":
        full_ds = OrientationBarsDataset(
            n_samples=total, image_size=image_size, n_orient_classes=n_orient_classes
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "multi_xor":
        full_ds = MultiXORDataset(n_samples=total, n_bits=n_bits)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "random_flip_mnist":
        # Our newly added dataset that uses RandomFlipMNISTDataset
        # We'll treat train vs valid vs test accordingly
        train_dataset = RandomFlipMNISTDataset(
            split="train",
            data_path=None,
            output_dir=output_dir,
            random_flip=True,
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        valid_dataset = RandomFlipMNISTDataset(
            split="valid",
            data_path=None,
            output_dir=output_dir,
            random_flip=True,
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        test_dataset = RandomFlipMNISTDataset(
            split="test",
            data_path=None,
            output_dir=output_dir,
            random_flip=True,
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        return train_dataset, valid_dataset, test_dataset
    elif dataset_name == "mnist_modulo10":
        data = load_mnist_modulo10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            cache_dir=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
    elif dataset_name == "info_shunting":
        full_ds = InfoShuntingDataset(n_samples=total, input_dim=10)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "noise_resilience":
        full_ds = NoisyLineDataset(
            n_samples=total, image_size=image_size, noise_level=0.2
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "feature_selectivity":
        # Create temporary directory path for datasets that need direct MNIST access
        temp_data_path = None
        if task_data_path and task_data_path.strip():
            temp_data_path = os.path.join(task_data_path, "datasets", "mnist")
        elif outputs_dir and outputs_dir.strip():
            temp_data_path = os.path.join(outputs_dir, "datasets", "mnist")
        elif output_dir:
            temp_data_path = os.path.join(output_dir, "datasets", "mnist")
        else:
            temp_data_path = os.path.join("data", "mnist")
        os.makedirs(temp_data_path, exist_ok=True)

        base_mnist = datasets.MNIST(root=temp_data_path, train=True, download=True)
        full_ds = SparseMNISTDataset(base_mnist, threshold=0.3)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "context_gating":
        # Create temporary directory path for datasets that need direct MNIST access
        temp_data_path = None
        if task_data_path and task_data_path.strip():
            temp_data_path = os.path.join(task_data_path, "datasets", "mnist")
        elif outputs_dir and outputs_dir.strip():
            temp_data_path = os.path.join(outputs_dir, "datasets", "mnist")
        elif output_dir:
            temp_data_path = os.path.join(output_dir, "datasets", "mnist")
        else:
            temp_data_path = os.path.join("data", "mnist")
        os.makedirs(temp_data_path, exist_ok=True)

        base_mnist = datasets.MNIST(root=temp_data_path, train=True, download=True)
        full_ds = ContextualFigureGroundMNIST(base_mnist, noise_level=0.1)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "learning_dynamics":
        full_ds = MultiTaskLineDataset(total_size=total, image_size=image_size)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "hierarchical_processing":
        # Create temporary directory path for datasets that need direct MNIST access
        temp_data_path = None
        if task_data_path and task_data_path.strip():
            temp_data_path = os.path.join(task_data_path, "datasets", "mnist")
        elif outputs_dir and outputs_dir.strip():
            temp_data_path = os.path.join(outputs_dir, "datasets", "mnist")
        elif output_dir:
            temp_data_path = os.path.join(output_dir, "datasets", "mnist")
        else:
            temp_data_path = os.path.join("data", "mnist")
        os.makedirs(temp_data_path, exist_ok=True)

        base_mnist = datasets.MNIST(root=temp_data_path, train=True, download=True)
        full_ds = CompositionalMNIST(base_mnist, n_samples=total)
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "cifar10":
        return load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    elif dataset_name == "poisson_generator":
        train_ds, valid_ds, test_ds = load_poisson_generator_datasets(
            base_dataset=base_dataset,
            rate_multiplier=rate_multiplier,
            train_valid_split=train_valid_split,
            flatten=flatten,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "orthonet":
        input_dim = p.get("input_dim", 10)
        n_layers = p.get("n_layers", 5)
        nonlinearity = p.get("nonlinearity", "sigmoid")
        repeat_layers = p.get("repeat_layers", False)
        noise_std = p.get("noise_std", 0.1)
        seed = getattr(train_cfg, "seed", 0)
        full_ds = OrthoNetDataset(
            n_samples=total,
            input_dim=input_dim,
            n_layers=n_layers,
            nonlinearity=nonlinearity,
            seed=seed,
            repeat_layers=repeat_layers,
            noise_std=noise_std,
        )
        return random_split(full_ds, [train_size, valid_size, test_size])
    elif dataset_name == "synthetic_correlated":
        # Note: synthetic_correlated dataset moved to experiments/synaptic_correlation/
        # Import locally if needed for correlation experiments
        try:
            # Try importing from local experiments directory first
            from synthetic_correlated import SyntheticCorrelatedDataset
        except ImportError:
            try:
                # Try importing from full path
                from experiments.synaptic_correlation.synthetic_correlated import (
                    SyntheticCorrelatedDataset,
                )
            except ImportError:
                raise ImportError(
                    "synthetic_correlated dataset is now located in experiments/synaptic_correlation/. "
                    "This dataset is specialized for correlation analysis experiments. "
                    "Run from experiments/synaptic_correlation/ directory or add it to PYTHONPATH."
                )

        input_dim = p.get("input_dim", 300)
        correlation_range = (p.get("corr_min", -0.1), p.get("corr_max", 0.1))
        mean_magnitude = p.get("mean_magnitude", 0.5)
        seed = getattr(train_cfg, "seed", 0)

        train_ds = SyntheticCorrelatedDataset(
            n_samples=train_size // 2,  # Per class
            input_dim=input_dim,
            correlation_range=correlation_range,
            random_seed=seed,
            apply_sigmoid=True,
            mean_magnitude=mean_magnitude,
        )
        valid_ds = SyntheticCorrelatedDataset(
            n_samples=valid_size // 2,
            input_dim=input_dim,
            correlation_range=correlation_range,
            random_seed=seed + 1 if seed is not None else None,
            apply_sigmoid=True,
            mean_magnitude=mean_magnitude,
        )
        test_ds = SyntheticCorrelatedDataset(
            n_samples=test_size // 2,
            input_dim=input_dim,
            correlation_range=correlation_range,
            random_seed=seed + 2 if seed is not None else None,
            apply_sigmoid=True,
            mean_magnitude=mean_magnitude,
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "imagenet":
        return load_imagenet_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
