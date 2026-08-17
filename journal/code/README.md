# Analysis code included with the journal package

- `regular_tree/`: standalone raw-additive equations and finite-difference
  validation. Full regular-tree training uses the repository package under
  `src/dendritic_modeling` and the sweeps in `configs/regular_tree/`.
- `reconstructed_tree/`: source copies for morphology reconstruction,
  structural capacity, focal shunting, and the projected descent-bound check.
  Four are byte-identical; the focal script adds a documented injection hook
  for the exact factor-freeze control. Origins and hashes are in
  `../reproducibility/origin_manifest.tsv`.
- `task_derived/`: portable copies of the CAVE/DANDI join and measured-response
  branch-model pipeline. Only project-relative defaults and one minimal local
  morphology-helper import differ from their recorded origins.

The reconstructed-tree scripts retain their original directory-relative
defaults. Supply paths explicitly when running them from this package; see
`reconstructed_tree/README.md`.
