#!/usr/bin/env python3
"""Explicit sibling-runner dispatch plus the pre-outcome numerical-order decoder."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

import bridge

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_morphology_calibration_run", HERE/"run.py")
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
ORIGINAL_DECODER = bridge.tree_from_payload


def decode_numeric_order(payload):
    corrected = dict(payload)
    corrected["children"] = dict(sorted(payload["children"].items(), key=lambda item: int(item[0])))
    return ORIGINAL_DECODER(corrected)


def activate():
    bridge.tree_from_payload = decode_numeric_order
    run.tree_from_payload = decode_numeric_order


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True)
    args=parser.parse_args()
    assert run.OUT == HERE.parents[1]/"source_data/morphology_calibration"
    amendment=json.loads((run.OUT/"serialization_dispatch_amendment.json").read_text())
    assert amendment["runtime_fix_sha256"]==run.sha(Path(__file__).resolve())
    assert amendment["protocol_sha256"]==run.sha(run.OUT/"protocol.json")
    assert amendment["selection_seal_sha256"]==run.sha(run.OUT/"confirmatory_selection_seal.json")
    activate()
    run.confirm(args.index)
