"""Reproduce the helper mismatch, then execute a complete paired update."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_end_to_end_fix import experiment,compatible_adaptive,ORIGINAL_ADAPTIVE
from bridge import task,tree_payload


def test_oracle_return_adapter_preserves_tree_and_complete_update(monkeypatch):
    seed=721000;family=3;coeff=task(seed,family)
    tree,_=ORIGINAL_ADAPTIVE(coeff,"estimated_dp")
    choice={"tree":tree_payload(tree),"sample_sha256":"contract_fixture"}
    cfg={"fixed_candidate":"balanced_p0"}
    monkeypatch.setattr(experiment,"adaptive_tree",ORIGINAL_ADAPTIVE)
    with pytest.raises(ValueError,match="not enough values"):
        experiment.train(seed,family,choice,cfg)
    ordinary,_=ORIGINAL_ADAPTIVE(coeff,"oracle_dp")
    adapted,_,unused=compatible_adaptive(coeff,"oracle_dp")
    assert tree_payload(ordinary)==tree_payload(adapted) and unused is None
    monkeypatch.setattr(experiment,"adaptive_tree",compatible_adaptive)
    # Execute the full initialization/evaluation/first Adam update, retaining
    # ordinary ranges for topology packing. No scientific result is saved.
    monkeypatch.setattr(experiment,"range",lambda *args: range(1,2) if args==(1,1025) else range(*args),raising=False)
    rows,weights,audit=experiment.train(seed,family,choice,cfg)
    assert len(rows)==12 and {row["step"] for row in rows}=={0,1}
    assert all(row["status"]=="completed" for row in rows)
    assert weights.shape==(6,7,4) and np.isfinite(weights).all()
    assert max(abs(weights).flat)<=2
    assert audit["label_independent_initialization"]
