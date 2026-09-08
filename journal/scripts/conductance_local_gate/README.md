# Prospective conductance local-gate follow-up

The experiment lives in `journal/source_data/conductance_local_gate/`. Read its
`protocol.json`, `protocol_freeze.json`, `METHODS.md` and `RESULTS.md` first.
The hypothesis follows explicitly disclosed panel exploratory evidence; only the
new twenty seed blocks enter the follow-up's scientific summaries. Main Fig. 5
and supplementary panels A–D use those fresh blocks. Supplementary panel E
explicitly retains the preceding cohort's gradient-cancellation diagnostic.

`model.py` independently implements the released seven-compartment circuit.
Its local gradient branches never call `exact_path`; an explicit sentinel test
checks this. The same local eligibilities are used by every rule. Oracle source
access remains explicit in the exact and projected comparator names.

Run from the `journal/` directory in a working copy. NumPy and pandas suffice
for training/replay; pytest runs validation; matplotlib generates figures. Exact
package versions and scheduler jobs are recorded per seed. Set all BLAS/OpenMP
thread counts to one. On the Harvard cluster use Slurm compute nodes for runs.
Do not execute from a checksummed release staging directory.

The prospective worker is `worker.sh`. Scientific runs enforce a reachable Git
commit containing the frozen protocol and verify all listed source hashes.
Completed seed outcomes are immutable. `worker_validation.sh` runs the excluded
historical replay; `worker_report.sh` refuses incomplete cohorts or a failed
historical replay. It checks every recorded artifact hash before analysis.

For standalone reviewer use, `portable_replay.py` does not require a Git
repository and writes only a new separate output directory. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B scripts/conductance_local_gate/portable_replay.py \
  --seed 2026090801 --task opposed_strong \
  --rule hard_distal_unit_proximal --rate 0.03 --steps 4096 \
  --output /tmp/local-gate-replay-seed-2026090801
```

The replay checks its result against canonical files when available. To run the
short excluded implementation check, use seed `2026090799` and `--steps 4`.
The full run outputs contain initial parameters, all saved parameter arrays,
optimizer state at the final update, input/initialization/checkpoint hashes,
paired endpoint views, fixed-step outcomes and bound-contact counters.

The native figure builder writes a six-panel `local_gate_primary.pdf` and the
complete rate/budget sheet `local_gate_all_rates.pdf`. Canonical main-manuscript
figure mapping belongs to the root integration workflow, not these scripts.

Use `python -B scripts/conductance_local_gate/figure.py` to redraw only figures
from verified numerical summaries. It writes current `figure_provenance.json`
without changing the protocol or summary tables. The initial
`summaries/completeness_audit.json` retains historical render hashes; current
figures are authenticated separately by `figure_provenance.json` and
`completion_gate.json`. Run
`python -B scripts/conductance_local_gate/validate_completion.py --check` for a
read-only check of the complete cohort, current figures, and retained replay
evidence. Original expected hashes remain intact in a sanitized release;
`portable_contract.py` accepts changed bytes only through a declared verified
original-to-released hash chain.

The exact source and execution history, including the corrected timing of the
analysis-helper snapshot, is described in
`source_data/conductance_local_gate/REPRODUCIBILITY.md`.
