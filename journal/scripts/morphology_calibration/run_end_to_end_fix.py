#!/usr/bin/env python3
"""Pre-training compatibility correction for the oracle helper return arity."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("frozen_calibration_end_to_end",HERE/"end_to_end.py")
assert spec and spec.loader
experiment=importlib.util.module_from_spec(spec);sys.modules[spec.name]=experiment
spec.loader.exec_module(experiment)
ORIGINAL_ADAPTIVE=experiment.adaptive_tree


def compatible_adaptive(coeff,name):
    tree,bound=ORIGINAL_ADAPTIVE(coeff,name)
    # The frozen train() ignores both auxiliary return values. Supplying the
    # unused third value preserves the exact returned tree and bound.
    return (tree,bound,None) if name=="oracle_dp" else (tree,bound)


def activate():
    experiment.adaptive_tree=compatible_adaptive


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    amendment=json.loads((experiment.OUT/"runtime_amendment.json").read_text())
    assert amendment["runtime_fix_sha256"]==experiment.sha(Path(__file__).resolve())
    assert amendment["protocol_sha256"]==experiment.sha(experiment.OUT/"protocol.json")
    assert amendment["selection_seal_sha256"]==experiment.sha(experiment.OUT/"selection_seal.json")
    activate();experiment.run_seed(args.index)
