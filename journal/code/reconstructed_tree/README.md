# Exact reconstructed-tree analysis sources

These five Python files originate from
`drafts/dendritic-credit-routing/analysis`. Four remain byte-identical. The
focal-perturbation script has one documented extension:

- `analyze_microns_morphology_credit.py`: maps synapses, compresses skeletons,
  constructs normalized electrical proxies, and generates structural fields;
- `test_morphology_compression_robustness.py`: repeats dictionary-capacity
  analyses over nested Monte Carlo streams;
- `analyze_credit_routing_capacity.py`: converts reconstruction residuals to
  credit capture and cell-level inference;
- `run_focal_shunting_credit_perturbation.py`: passive-matrix adjoint,
  shunting/current-matched interventions, soma clamp, relations, and numerical
  checks. An optional conductance-system injection hook supports the exact
  factor-freeze control; its default execution path is unchanged;
- `verify_credit_capture_bound.py`: numerical audit of the projected-gradient
  one-step bound.

The scripts retain their original `PROJECT = parents[1]` default convention.
Their origin hashes and the focal-script modification are recorded in
`reproducibility/origin_manifest.tsv`. In this relocated directory, pass
inputs and outputs explicitly. For example:

```bash
python code/reconstructed_tree/run_focal_shunting_credit_perturbation.py \
  --segments reproduced_results/microns_morphology_credit/segment_metrics.csv \
  --outdir reproduced_results/microns_focal_shunting_credit
```

Run scripts from this directory, or add it to `PYTHONPATH`, so imports of
`analyze_microns_morphology_credit` resolve locally. CAVE authentication and
raw MICrONS downloads are outside this source bundle.
