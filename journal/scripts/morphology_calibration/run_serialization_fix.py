#!/usr/bin/env python3
"""Audited pre-outcome JSON decoder correction; frozen selectors stay unchanged.

JSON sort_keys orders string node IDs lexically. Restore numerical/topological
order before passing a saved tree to the original decoder. This changes no
tree, candidate choice, fitting rule, dataset, seed or endpoint definition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import bridge
import run

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
    amendment=json.loads((run.OUT/"serialization_amendment.json").read_text())
    assert amendment["runtime_fix_sha256"]==run.sha(Path(__file__).resolve())
    assert amendment["protocol_sha256"]==run.sha(run.OUT/"protocol.json")
    assert amendment["selection_seal_sha256"]==run.sha(run.OUT/"confirmatory_selection_seal.json")
    activate()
    run.confirm(args.index)
