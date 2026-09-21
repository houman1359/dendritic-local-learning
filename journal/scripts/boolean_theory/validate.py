#!/usr/bin/env python3
"""Run focused tests and independently replay every exported coefficient certificate."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "source_data/boolean_theory")
    args = parser.parse_args()
    start = time.perf_counter()
    test = Path(__file__).with_name("test_audit.py")
    result = subprocess.run([sys.executable, "-m", "pytest", str(test), "-q"], text=True, capture_output=True)
    (args.out / "pytest_output.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    truth = list(csv.DictReader((args.out / "truth_tables.csv").open()))
    certs = json.loads((args.out / "exact_constructions.json").read_text())
    max_error = 0.0
    # Parse the tiny exported tree grammar locally: no import of audit.py.
    def parse(text):
        if len(text) == 1:
            return "abcd".index(text)
        level = 0
        for pos, char in enumerate(text[1:-1], start=1):
            if char == "(": level += 1
            if char == ")": level -= 1
            if char == "," and level == 0:
                return parse(text[1:pos]), parse(text[pos+1:-1])
        raise ValueError(text)
    def node_name(t):
        return "abcd"[t] if isinstance(t,int) else "("+node_name(t[0])+","+node_name(t[1])+")"
    def forward(t, params, x):
        if isinstance(t,int): return x[t]
        l, r = forward(t[0],params,x), forward(t[1],params,x)
        a,b,c,d = params[node_name(t)]
        return a+b*l+c*r+d*l*r
    for cert in certs:
        tree = parse(cert["tree"])
        params = cert["normalized_node_coefficients"]
        assert max(abs(v) for row in params.values() for v in row) <= 2
        assert Fraction(cert["max_normalized_coefficient_squared_rational"]) <= 4
        for row in truth:
            if row["family"] != cert["family"]: continue
            x = [float(row["x_"+letter]) for letter in "abcd"]
            error = abs(forward(tree,params,x)-float(row["target_normalized"]))
            max_error = max(max_error,error)
    assert max_error < 1e-13
    report = dict(test_return_code=result.returncode, pytest_output=result.stdout.strip(),
        exported_normalized_constructions_replayed=len(certs), patterns_per_construction=16,
        maximum_absolute_normalized_truth_table_error=max_error,
        validation_runtime_seconds=time.perf_counter()-start,
        validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_script_sha256=hashlib.sha256(Path(__file__).with_name("audit.py").read_bytes()).hexdigest(),
        test_script_sha256=hashlib.sha256(test.read_bytes()).hexdigest())
    (args.out / "test_validation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
