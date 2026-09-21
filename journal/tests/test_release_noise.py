"""Generator identity, frozen-body integrity and portable dispatch checks."""
from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1] / 'code/release_noise'
sys.path.insert(0, str(ROOT))
import frozen_generators as frozen
import generators


def test_frozen_bodies_match_original_source_segments():
    origins = json.loads((ROOT / 'ORIGINS.json').read_text())
    delivered = {node.name: ast.get_source_segment((ROOT/'frozen_generators.py').read_text(), node)
                 for node in ast.parse((ROOT/'frozen_generators.py').read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for record in origins['symbols']:
        source = (ROOT / record['source_file']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == record['source_sha256']
        segment = delivered[record['symbol']]
        assert hashlib.sha256(segment.encode()).hexdigest() == record['symbol_sha256']


def test_explicit_legacy_generator_replays_original_body_and_split():
    torch.manual_seed(71)
    np.random.seed(71)
    original = frozen.NoisyLineDataset(n_samples=24, image_size=7, noise_level=0.2)
    expected = torch.utils.data.random_split(original, [16,4,4])
    result = generators.legacy_noisy_lines(seed=71, image_size=7,
                                         train_size=16, valid_size=4, test_size=4)
    for got, wanted in zip(result, expected):
        assert got.indices == wanted.indices
        assert torch.equal(got.dataset.data, wanted.dataset.data)
        assert torch.equal(got.dataset.labels, wanted.dataset.labels)
    assert original.data.shape == (24, 49)
    assert set(original.labels.tolist()) == {0,1,2}
    assert original.data.min() < 0  # legacy images are signed and unclipped


def test_projected_noise_formula_split_seeds_and_labels(monkeypatch):
    clean = [TensorDataset(torch.linspace(0,1,784).repeat(n,1), torch.arange(n)%10)
             for n in (5,3,4)]
    monkeypatch.setattr(frozen, 'load_mnist_as_datasets', lambda **kwargs: tuple(clean))
    result = generators.projected_noise_mnist()
    p = torch.randn(784,50,generator=torch.Generator().manual_seed(0))
    p = p / p.norm(dim=0,keepdim=True).clamp(min=1e-8)
    for split, (observed, baseline) in enumerate(zip(result,clean),1):
        x,y = baseline.tensors
        z = torch.randn(len(x),50,generator=torch.Generator().manual_seed(split))
        expected = (x + 1.5*(z@p.T)).clamp(0,1)
        assert torch.equal(observed.tensors[0], expected)
        assert torch.equal(observed.tensors[1], y)
    assert torch.allclose(p.norm(dim=0),torch.ones(50))


def test_ambiguous_key_is_rejected_and_explicit_hook_runs():
    with pytest.raises(ValueError, match='Ambiguous'):
        generators.get_unified_datasets({'dataset':'noise_resilience'})
    from experiments.data_generation.synthetic_datasets import get_unified_datasets
    result = get_unified_datasets({'dataset':'noise_resilience', 'parameters': {
        'release_generator':'legacy_noisy_lines', 'seed':9, 'image_size':4,
        'train_size':8,'valid_size':2,'test_size':2}})
    assert [len(split) for split in result] == [8,2,2]
