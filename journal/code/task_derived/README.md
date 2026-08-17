# CAVE/DANDI measured-response pipeline

The code is copied from `drafts/dendritic-credit-routing`; origin hashes and
portable edits are listed in `../../reproducibility/origin_manifest.tsv`.

Expected layout relative to the journal directory:

```text
external_data/
  microns_morphology/                    # skeleton/synapse downloads
  microns_dandi_trial_manifest/
    microns_dandi_trial_manifest.csv
reproduced_results/
  microns_morphology_credit/             # segment_metrics.csv, mapped_synapses.csv.gz
  microns_functional_partner_manifest/
  microns_functional_partner_responses/
  microns_task_derived_credit_learning/
```

From the journal directory, the principal stages are:

```bash
python code/task_derived/build_functional_partner_manifest.py
python code/task_derived/extract_functional_partner_responses.py --help
python code/task_derived/analyze_functional_topology.py --help
python code/task_derived/summarize_functional_topology_cohort.py
python code/task_derived/run_task_derived_credit_learning.py \
  --channels 4 --replicates 10 --seed 20260721 \
  --test-fraction 0.2 --steps 1200 \
  --learning-rate 0.015 --weight-decay 0.001
```

The extraction wrapper is called once per target/session/scan listed in
`../../reproducibility/task_derived_primary_targets.csv`; use `--help` for its
required identifiers. It streams public NWB assets from DANDI. Building the
functional partner manifest queries CAVE and may require credentials.

`analyze_functional_topology.py` is also called once per extracted target with
`--extract-dir`. `summarize_functional_topology_cohort.py` then treats the
seven postsynaptic targets, rather than their 356 dependent partner pairs, as
the cohort replication units.

The task model uses one connected presynaptic partner per site, a tanh branch
nonlinearity, full-batch Adam, and exact readout/bias gradients for every
feedback method. Only the site/input-weight error is projected. Dictionary
coefficients are least-squares oracles, so the experiment tests capacity and
sufficiency rather than a biological encoder for the coefficients.

For the direct topology analysis, functional similarity is Pearson
correlation between vectors of condition-mean raw fluorescence over the 136
conditions having at least two trials. Repeat reliability uses alternating
occurrences of each repeated condition and a Spearman correlation between the
resulting condition-mean vectors. The controlled ancestry statistic rank
transforms similarity, shared-path fraction, `log1p` Euclidean distance and
`log1p` absolute soma-path-length difference; it residualizes the first two
separately on an intercept and the ranked controls, then correlates their
residuals.

All mapped contacts from one partner contribute to partner-level totals. A
multi-contact partner nevertheless supplies one feature and one modeled site,
assigned to the segment with greatest summed contact size (synapse count breaks
ties). The primary task analysis uses no repeat-reliability threshold and no
manual-only filter.
