# Single-point accessibility boundary result

Completed: 12 August 2026.  Status: passed; selected point frozen for a
fresh-seed confirmatory replication.

At signal delta 0.80, mean held-out accuracies are 0.6085, 0.6737 and 0.9227
for D1 `[8]`, D2 `[2,3]` and D3 `[2,1,2]`.  Every individual seed exceeds
0.57, all depth means lie in [0.60,0.95], and D3 minus D1 is +0.3147 and
+0.3136 in the two seeds.  The frozen boundary selector passes.

This result is exploratory because signal 0.80 was chosen after the preceding
ladder narrowly missed its D1 threshold.  It is not used for inference.  The
selected operating point is tested next with ten disjoint paired seeds and
prespecified topology, sensor, transport and additive controls.

Provenance:

- Run:
  `nonlinear_physical_depth_runs/journal_exploratory_physical_depth_boundary_bp_20260812004401`
- Frozen manifest SHA256:
  `941479867d0d7c730dc3011fdb1155a599b1ddd096cd333e98bb76ddbb06349d`
- Row source: `source_data/nonlinear_physical_depth_boundary/bp_seed_rows.csv`
