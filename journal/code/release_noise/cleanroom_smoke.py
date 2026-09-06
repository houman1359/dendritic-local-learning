"""Download-free smoke of an installed release, run outside the source checkout."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    args = parser.parse_args()
    release = args.release_root.resolve()
    adapter = release / 'article_analysis/code/release_noise'
    sys.path.insert(0, str(adapter))
    import numpy as np
    import torch
    import torchvision
    import wandb
    import google.protobuf
    import dendritic_modeling
    from torch.utils.data import TensorDataset
    from dendritic_modeling.datasets.standard_datasets import get_unified_datasets
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import branch_dynamics
    from dendritic_modeling.scripts.training.train_experiments import cli_main
    package = Path(dendritic_modeling.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), f'Package escaped clean venv: {package}'
    assert sys.prefix != sys.base_prefix, 'Use an isolated virtual environment'
    assert 'include-system-site-packages = false' in (Path(sys.prefix)/'pyvenv.cfg').read_text()
    import frozen_generators as frozen
    import generators
    try:
        generators.get_unified_datasets({'dataset':'noise_resilience'})
    except ValueError as error:
        assert 'Ambiguous' in str(error)
    else:
        raise AssertionError('Ambiguous historical task was silently interpreted')
    task = {'dataset':'legacy_noisy_lines','parameters': {
        'seed':42,'image_size':8,'train_size':12,'valid_size':4,'test_size':4}}
    splits = get_unified_datasets(task)
    assert [len(s) for s in splits] == [12,4,4]
    assert splits[0][0][0].shape == (64,)
    clean = tuple(TensorDataset(torch.linspace(0,1,784).repeat(n,1),torch.arange(n)%10)
                  for n in (6,3,4))
    frozen.load_mnist_as_datasets = lambda **kwargs: clean
    noisy = generators.projected_noise_mnist()
    assert [s.tensors[0].shape[1] for s in noisy] == [784]*3
    assert all(torch.equal(a.tensors[1],b.tensors[1]) for a,b in zip(clean,noisy))
    assert all(0 <= a.tensors[0].min() and a.tensors[0].max() <= 1 for a in noisy)
    assert not torch.equal(clean[0].tensors[0],noisy[0].tensors[0])
    versions = {name:importlib.metadata.version(name) for name in
                ['numpy','torch','torchvision','wandb','protobuf','scipy','pytest']}
    print(json.dumps({'status':'passed','isolated_venv':True,'package_source':'installed wheel in venv',
        'versions':versions,'checks':['core branch dynamics import','training entrypoint import',
        'installed core to released dataset hook','ambiguous task rejection','legacy flattened shape',
        'projected corruption bounds and unchanged labels'],
        'scope':'Tiny synthetic image tensors; no MNIST download, retraining or raw biological data'},indent=2))


if __name__ == '__main__':
    main()
