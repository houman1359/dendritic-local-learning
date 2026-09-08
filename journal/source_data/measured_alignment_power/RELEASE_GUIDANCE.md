# Release integration

The complete scientific Source Data package should include this directory's `inputs/` (including the byte-identical original pair/contact CSVs and original analysis definitions), `input_manifest.json`, `input_audit.json`, `PROTOCOL.md`, `protocol_freeze.json`, `runs/chunk_*.npz`, paired chunk metadata JSONs, calibration audit, all summary CSV/JSONs, independent arithmetic audit, and figure provenance. The NPZ chunks contain simulated scan/target statistics and exact P values, not model weights. They are required to recompute every curve without repeating the Monte Carlo simulation.

Preserve original source paths and hashes in the input manifest. The original pair-level data were outside journal Source Data; the newly archived copies are required for independent reproduction of the empirical statistic, reliability calibration and shared-recording geometry. Inputs occupy approximately 907KB. Exclude `slurm_logs/`, `__pycache__/`, `.pyc`, `.pytest_cache/` and `report_execution.txt` from scientific release bundles. Scheduling records may remain as small execution provenance.

Software should include `scripts/measured_alignment_power/{prepare_inputs.py,model.py,run.py,test_model.py,report.py,audit_results.py,portable_replay.py,test_portable_replay.py,worker.sh}`, `code/release_noise/release_hashes.py`, and the usual journal figure style dependencies. The protocol freezes the scientific model/runner/input preparation; later report rendering does not alter simulation choices. No simulation launch is needed for a manuscript/figure rebuild. The portable replay wrapper preserves all original frozen hash expectations and authenticates any declared path translations through the software/Source Data release sidecars. Preserve those sidecars during restoration; do not replace a registered original hash with a released hash.

Stable source figure: `source_data/measured_alignment_power/figures/measured_alignment_sensitivity.pdf`.
Assigned canonical SI figure filename: `figures/supplementary/figure_S56_measured_alignment_power.pdf`.
Stable labels: `note:measured_alignment_power` and `fig:supp_measured_alignment_power`.
The manuscript owner assigns the final SI number and copies/inputs the supplied TeX snippets; this analysis has not edited global manuscript or bundle files.
