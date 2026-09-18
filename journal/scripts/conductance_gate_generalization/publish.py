"""Publish a complete fresh-seed cohort as compact source tables and SI table."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import numpy as np

import experiment as ex

JOURNAL = Path(__file__).resolve().parents[2]


def write_once_or_verify(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == content, f"Refusing to replace different output: {path}"
    else:
        path.write_text(content)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().replace("\r\n", "\n")


def tex_number(value):
    if value < 0.001:
        exponent = int(math.floor(math.log10(value)))
        return rf"${value / 10**exponent:.3g}\times10^{{{exponent}}}$"
    return f"${value:.3g}$"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    protocol = json.loads((root / "fresh_protocol.json").read_text())
    summary = json.loads((root / "fresh_summary.json").read_text())
    assert summary["protocol_sha256"] == ex.digest(root / "fresh_protocol.json")
    for path, digest in protocol["source_sha256"].items():
        assert ex.digest(path) == digest
    endpoints, raw_hashes = [], {}
    for seed in protocol["fresh_seeds"]:
        for task in ex.TASKS:
            path = root / "results" / f"seed_{seed}_{task}.json"
            data = json.loads(path.read_text())
            assert data["protocol_sha256"] == summary["protocol_sha256"]
            raw_hashes[path.name] = ex.digest(path)
            state_path = path.with_suffix(".npz")
            raw_hashes[state_path.name] = ex.digest(state_path)
            with np.load(state_path) as archive:
                parameters = archive["selected_parameters"]
            tx, ty, _ = ex.dataset(seed, "test", 4096, task)
            replay = ex.evaluate(parameters, tx, ty, np.var(ty))
            np.testing.assert_allclose(replay, [r["test_nmse"] for r in data["endpoints"]],
                                       rtol=1e-11, atol=1e-13)
            for row in data["endpoints"]:
                row = dict(row)
                row["fixed_development_rate"] = row["rate"] == protocol["fixed_rates"][task][row["rule"]]
                endpoints.append(row)
    assert len(endpoints) == 900
    assert sum(r["fixed_development_rate"] for r in endpoints) == 300
    out = JOURNAL / "source_data/curated_publication"
    for name, rows in [("endpoints", endpoints), ("summary", summary["rows"]),
                       ("contrasts", summary["paired_contrasts"])]:
        write_once_or_verify(out / f"gate_generalization_{name}.csv", csv_text(rows))
    metadata = dict(protocol=protocol, raw_json_sha256=raw_hashes,
                    summary_sha256=ex.digest(root / "fresh_summary.json"),
                    low_level_implementation="scripts/conductance_local_gate/model.py",
                    experiment="scripts/conductance_gate_generalization/experiment.py",
                    checkpoint_replay="All 900 selected states reproduce reported held-out NMSE",
                    inference="Descriptive paired-seed intervals; no equivalence claim")
    write_once_or_verify(out / "gate_generalization_provenance.json", json.dumps(metadata, indent=2) + "\n")
    names = {"exact": "Exact path", "unit_broadcast": "Unit broadcast",
             "proportional_gate": "Proportional distal gate", "resistance_gate": "Relative-resistance gate",
             "swapped_gate": "Swapped gate"}
    values = {(r["task"], r["rule"]): r["mean"] for r in summary["rows"]}
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Local gating with graded context and independent sensory targets.
Mean held-out NMSE over twenty fresh paired seeds at development-selected
learning rates and validation-selected states within 4,096 updates. All rules
train the same seven-compartment conductance neuron and affine readout.
The graded teacher is model-generated; selection and mixture targets are
specified independently. Lower is better. Similar means do not establish
equivalence. The graded-mixture error remains appreciable even with exact
credit. Complete per-seed outcomes, other rates, bound contacts and descriptive
paired intervals are in the accompanying source tables; Section~S5 defines
the generators and rules.}
\label{tab:gate_generalization}
\small
\begin{tabular}{lrrr}
\toprule
Rule & Graded teacher & Sensory selection & Graded mixture \\
\midrule
"""
    for rule in ex.RULES:
        tex += names[rule] + " & " + " & ".join(tex_number(values[t, rule]) for t in ex.TASKS) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_once_or_verify(JOURNAL / "supplementary/curated/si_gate_generalization_table.tex", tex)
    print("Published 900 endpoint rows (300 fixed-rate), 15 means, 9 paired contrasts, and SI table.")


if __name__ == "__main__":
    main()
