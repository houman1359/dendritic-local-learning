"""Regression for the actual sorted-key JSON tree roundtrip."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bridge import adaptive_tree, task, tree_payload, forward, domain
from run_serialization_fix import ORIGINAL_DECODER,decode_numeric_order


def test_sorted_json_roundtrip_reproduces_then_corrects_node_order_failure():
    original,_=adaptive_tree(task(820910,0),"adaptive")
    payload=json.loads(json.dumps(tree_payload(original),sort_keys=True))
    with pytest.raises(KeyError):
        ORIGINAL_DECODER(payload)
    restored=decode_numeric_order(payload)
    assert original.children==restored.children
    assert original.descendants==restored.descendants
    assert original.parent==restored.parent
    weights=np.random.default_rng(43).normal(size=(7,4))
    np.testing.assert_array_equal(forward(domain(),original,weights),forward(domain(),restored,weights))
