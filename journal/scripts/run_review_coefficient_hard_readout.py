#!/usr/bin/env python3
"""Exploratory readout diagnostic designed AFTER soft-encoder outcomes.

Reuse the identical fitted softmax encoder and paired task data, but choose
the maximum-probability subtree instead of delivering its soft probability
mixture. No classifier refitting, endpoint tuning or seed selection occurs.
This changes the coefficient readout's sparsity and noise sensitivity, not
the available route span. Original soft-encoder results remain the primary
frozen experiment and are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_review_coefficient_encoder as base

OUT=base.ROOT/"source_data/review_coefficient_hard_readout"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int);parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args();cfg=json.loads(base.CONFIG.read_text())
    cfg["scope"]=("Exploratory follow-up specified after observing the soft-encoder results. "
        "The identical learned context estimator uses a deterministic maximum-probability subtree readout. "
        "Original seeds and task data are paired sensitivity units, not a new independent replication. "
        "The encoder is unchanged; no task outcome selects its readout parameters. " + cfg["scope"])
    old_prediction=base.coefficient_predictions
    def hard(cues,w,delay):
        probability=old_prediction(cues,w,delay)
        return np.eye(4)[probability.argmax(axis=1)]
    base.coefficient_predictions=hard
    base.OUT=OUT
    if args.aggregate:
        base.aggregate(cfg)
        report=json.loads((OUT/"report.json").read_text())
        # Reusing the soft experiment's condition does not create a new
        # primary family after the hard-readout choice was made post hoc.
        contrasts=pd.read_csv(OUT/"paired_contrasts.csv")
        contrasts=contrasts.drop(columns=["primary_family_holm_p"],errors="ignore")
        contrasts["analysis_status"]="exploratory_paired_sensitivity"
        contrasts.to_csv(OUT/"paired_contrasts.csv",index=False,float_format="%.12g")
        reference=report.pop("primary_contrasts",[])
        for row in reference:
            row.pop("primary_family_holm_p",None)
            row["analysis_status"]="exploratory_paired_sensitivity"
        report["reference_condition_contrasts"]=reference
        report["multiplicity_scope"]=("No confirmatory family or Holm-adjusted "
            "primary P values are assigned to this exploratory readout analysis. "
            "Unadjusted signed-rank P values and paired intervals are descriptive.")
        report.update(exploratory_after_soft_outcomes=True,readout="maximum_probability_subtree",
                      wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        (OUT/"README.md").write_text("# Exploratory hard selection of learned route coefficients\n\n"+
            cfg["scope"]+"\n\nThe twenty original seeds and fitted encoders are reused; "
            "this is a paired sensitivity analysis, not a fresh replication. "
            "The 256-trial, noise-SD0.5, zero-delay condition is retained as a reference "
            "to the primary soft experiment, not a new primary family. "
            "All intervals and unadjusted P values are descriptive; no exploratory "
            "row is assigned a primary-family Holm adjustment. Numerical learning "
            "outcomes and the original soft experiment are unchanged.\n")
    else:
        assert args.seed in cfg["seeds"]
        base.run(args.seed,cfg,OUT/"runs")
        path=OUT/"runs"/f"seed_{args.seed}.json";record=json.loads(path.read_text())
        record.update(exploratory_after_soft_outcomes=True,readout="maximum_probability_subtree",
                      wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path.write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":main()
