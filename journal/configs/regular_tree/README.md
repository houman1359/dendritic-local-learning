# Regular-tree configurations

These files preserve the artificial-tree experiments inherited from
`drafts/dendritic-local-learning`.

## Archived protocols

- `feedback_definition_extra10_seeds_portable.yaml`: seeds 47-56, shunting
  and raw-additive cores, matched-width/scalar-fallback and neuron-indexed
  ancestry-shared feedback, three-factor local learning, 180 epochs.
- `exact_transport_factorial_5seed_portable.yaml`: seeds 42-46, shunting core,
  3F/5F by local/backpropagated decoder factorial, exact path transport,
  200 epochs.
- `backprop_reference_5seed_portable.yaml`: seeds 42-46, matched shunting
  backpropagation reference, 200 epochs.

Only private output and data paths were changed in the portable YAML files.
All numerical fields are unchanged. The scheduler account remains visible as
historical provenance and can be overridden at submission time.

`archived_resolved/` contains one fully resolved JSON example for the shunting
and additive feedback cores, the 3F/local exact-transport cell, and the
backpropagation reference. Only private data/result paths were replaced.
Resolved files expose defaults that are absent from the concise sweep YAMLs,
including deterministic settings, optimizer betas, gate policies, data-loader
tuning, and the raw-additive alias resolution.

## Results and limitations

- `reported_feedback_definition_15seed.csv` contains all 60 accuracy rows for
  seeds 42-56. Seeds 47-56 link to complete archived run directories. Seeds
  42-46 have only their source-data rows; their original configs and
  checkpoints were not recoverable.
- `reported_exact_transport_factorial.csv` and
  `reported_backprop_reference.csv` are byte-identical copies of the grouped
  archived result tables.
- `feedback_definition_extra10_frozen_manifest.json` is the untouched frozen
  manifest from the complete July extension. It records a dirty worktree, so
  its source-file hashes, not the commit alone, are the operative provenance.

The exact-transport and backpropagation protocols share a ReLU core output and
200 epochs. The feedback-definition protocol uses an identity core output and
180 epochs. They answer different controls and must not be presented as one
fully crossed experiment.

See `../reruns/README.md` for the prepared clean 15-seed replacement cohort.

## Original arXiv/NeurIPS regime controls

`arxiv_regimes/` contains sanitized, hash-verified sweep configurations for
the rule-family, local-mismatch, noise-feedback-rank and CIFAR-10 control
panels. Their output directories are replaced by `./results`; numerical
settings are unchanged. Corresponding frozen result tables are in
`../../source_data/regular_tree_regimes/`. They can be re-exported from a
sibling `dendritic-local-learning` checkout with
`make refresh-regular-tree-source`.
