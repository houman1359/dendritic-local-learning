# When dendritic structure helps local credit assignment

This directory is the self-contained package for the canonical, standalone
journal article. It integrates the load-bearing theory, simulations and
controls developed during the conference project with the subsequent
topology, conductance and biological-boundary experiments. It assumes no prior
paper. The active target is a **Nature Communications Article**; any related
conference or preprint status will be disclosed at submission but does not
determine the article's scientific scope.

The paper is organized around one conditional claim:

> Dendritic trees transform limited neuron-level teaching signals into
> structured synaptic credit. Conductance controls route gain, topology
> controls which synapses can share credit, and useful learning requires the
> resulting routes to align with task credit.

The package distinguishes sixteen levels of evidence:

1. exact mathematical identities for conductance trees;
2. controlled learning experiments in regular artificial trees, including a
   complete 640-run historical depth-by-feedback factorial (480 input-valid
   runs), a 120-checkpoint input-valid credit-to-loss diagnostic, a 120-run
   retained bandwidth-matched routing control, a detached 320-run clean
   exact-transport/backpropagation audit, a 240-run retained spatial-topology
   boundary control, and a 160-run retained fixed-contact depth control;
3. a frozen 50-seed stochastic quadratic phase experiment testing spectral
   alignment, hierarchy-depth matching, projection denoising and branch
   reliability, plus operator and trained partition-residual reanalyses of the
   completed 2,700-fit factorial;
4. a fresh 270-fit exact-resource positive-rate physical-depth experiment with
   backpropagation, LocalCA, alignment, shuffled-sensor, reversed-placement and
   raw-additive controls;
5. 200 paired point--dendrite and BP--local-credit fits separating serial
   composition, point-network capacity, coordinate restriction, optimizer
   effects and path specificity;
6. a 90-fit prospective interpolation across three intermediate task--sensor
   alignment doses, joined to the frozen endpoint cohorts;
7. a preregistered 60-fit Fashion-MNIST replication of the scalar,
   neuron-indexed and exact-path feedback ladder;
8. a frozen 220-fit literal grouped-point and independent-H2 control that
   replicates the serial-composition crossover under BP and LocalCA;
9. a separate frozen 50-seed trained positive-conductance mechanism test with
   fixed oracle shunts and an exact state clamp;
10. a fresh 50-seed adaptive-conductance test estimating branch reliability
    from paired noisy local credit observations;
11. model-based analyses and perturbations on eight reconstructed MICrONS trees,
   including independent reciprocal-cable and physical-unit controls;
12. a disjoint 47-cell public-v661 sensitivity cohort from the same mouse,
   including a frozen irregular-tree wavelet analysis against isotropic and
   ancestry-permuted controls;
13. analyses of measured MICrONS visual responses;
14. a synapse-resolved inhibitory census with presynaptic-axon and 3D controls;
15. a controlled sufficiency test that rotates exact task credit into or out of
   a fixed reconstructed-tree routing subspace; and
16. a retrospective six-animal consistency test of signed neuron identity.

Modeled gradients on reconstructed anatomy are not described as measurements
of biological learning. Negative alignment results are retained because they
define the boundary of the theory.

## Files

- `main.tex`: journal manuscript, including Methods and declarations.
- `supplementary/supplementary.tex`: supplementary theory and controls.
- `main_with_supplementary.pdf`: combined reading copy containing the complete
  main Article followed by all Supplementary Information.
- `references.bib`: shared bibliography.
- `figures/main/`: canonical assets compiled as main Figures 1--8.
- `figures/supplementary/`: canonical assets compiled as S1--S26.
- `figures/generated/`: internal descriptive-name outputs from figure scripts;
  these are never referenced by LaTeX or included in the Overleaf bundle.
- `figures/README.md`: authoritative figure, panel, and asset map.
- `source_data/`: numerical source data organized by figure and panel.
- `scripts/`: figure generation and journal-specific analyses.
- `configs/regular_tree/`: archived artificial-tree sweeps, representative
  resolved configurations, frozen manifest, and result tables.
- `configs/reruns/`: frozen full 15-seed replacement sweeps for the feedback
  comparison. The two 30-task arrays completed and passed the prospective
  design, completeness, checkpoint, and hash checks on 31 July 2026. Their
  current-code results now supply Figure 2.
- `configs/task_derived/`: complete machine-readable specification of the
  measured-response branch model.
- `code/`: additive reference equations, exact reconstructed-tree source
  scripts, and a portable CAVE/DANDI task-derived analysis pipeline.
- `reproducibility/`: origin hashes, hardware accounting, seven-target and
  full eight-cell cohort manifests, archive boundaries, rerun records, and the
  exact conference-to-journal figure-lineage map.
- `analysis/EVIDENCE_LEDGER.md`: claim-to-evidence and provenance audit.
- `analysis/AUGUST_2026_CONFIRMATORY_RESULTS.md`: consolidated numerical
  report for the new theory, trained-address, focal-conductance, functional
  boundary and clean implementation studies.
- `analysis/EXPERIMENT_CONTRACT.md`: frozen analyses and decision rules.
- `analysis/alignment_controlled_results.md`: controlled topology--task
  alignment result and its scope.
- `analysis/JOURNAL_READINESS.md`: remaining submission requirements.
- `analysis/NATURE_COMMUNICATIONS_PROGRAM.md`: target story, figure plan and
  evidence boundary.
- `submission/`: cover letter, reporting checklists, Source Data, and the
  checksummed reviewer software release.

## Build

From this directory:

```bash
make figures
make paper
make supplement
make combined
make overleaf-bundle
make reproducibility
make software-release
make audit
```

The manuscript build requires a standard TeX distribution with pdfLaTeX and
BibTeX. Figure generation requires Python 3 with NumPy, pandas, SciPy,
Matplotlib, and seaborn. The current working manuscript contains nine numbered
main figures. `make figures` regenerates the publication-facing journal figures.
Audited conference-era generator snapshots and source tables are retained under
`scripts/inherited_neurips/` and `source_data/inherited_neurips/`. The prospective
figure command reads the audited run and checkpoint tables already packaged in
`source_data/prospective_learning/`; the checkpoint collector is kept separate
because it re-evaluates the frozen models. The unified draft has nine numbered
main figures and twenty-six supplementary figures. Each main figure is one vector
PDF with a single consecutive panel sequence; expanded diagnostics remain in
Supplementary Information.
`make overleaf-bundle` writes the current allow-listed package and ZIP under
`submission/`; see `OVERLEAF_README.md` before uploading.
The inherited arXiv/NeurIPS regular-tree tables are already frozen in this
repository. Authors with access to the sibling source project can verify and
refresh those exports with `make refresh-regular-tree-source`.
The public-v661 figure command
also refreshes its panel tables and exclusion manifest.
`make software-release` exports a clean Git-HEAD copy of the complete training
implementation, adds explicitly allow-listed journal analysis materials,
sanitizes historical machine-local defaults, and validates both file-level
checksums and the final ZIP. Its outputs are
`submission/software_release/` and
`submission/Dendritic_credit_assignment_software.zip`.

## Current evidence and release boundary

The structural result is supported in the original eight-cell pilot and in a
frozen 47-cell cohort that is disjoint by stable nucleus identifier. The
larger cohort uses historical MICrONS minnie65 version 661 reconstructions and
direct presynaptic coarse E/I calls. It is a same-mouse sensitivity analysis,
not an independent-animal replication or an in vivo learning experiment.
Focal-gradient factor decomposition and the alignment-controlled experiment
are complete and have panel-level source data.

The current 15-seed regular-tree feedback table is a fully archived clean
cohort with all 60 configurations, checkpoints, logs, result hashes, and
scheduler records retained. It prospectively replaced the earlier mixed
archive, which remains available only as an audit artifact. The larger
prospective experiment contains 640 audited training runs; an outcome-
independent input rule retains 480. Its publication diagnostic evaluates
feedback geometry and one-step loss change at 120 matched backpropagation
checkpoints. A bandwidth-matched routing cohort retains 120 valid runs and
isolates correct neuron-to-tree assignment. The complete 400-run inhibitory-
dose family is excluded because its prespecified cross-core endpoint depends
on signed synthetic inputs driving positive-conductance shunting cells. A
240-run valid spatial-topology control identifies a forward coverage effect
shared by backpropagation, and a 160-run valid fixed-contact control shows that
scalar feedback amplifies the optimization cost of increasing depth. The
detached 320-run audit independently compares exact transport with
backpropagation under valid transfers; all artifacts and finite stage-complete
checkpoints passed, and all four depth-averaged intervals include zero. The
global mean difference is -0.0545 percentage points across 160 pairs, reported
as agreement rather than formal equivalence. These analyses supply Figure 2 and
Supplementary Figures S7--S9; the trained 2,700-fit address factorial supplies
Figure 3. The point--dendrite controls, alignment interpolation and H2
hierarchy replication and H4 saturation test supply Figure 5, the immutable-source
H2/H3 audit supplies Supplementary Figure S26, and the Fashion-MNIST replication
extends Figure 2. Stable public
identifiers are recorded for MICrONS minnie65 materialization 1822, the v661
static release, and each DANDI asset. The DANDI dataset remains a draft
version, and the in-development synapse-target proxy remains secondary until
it has a public release identifier.

See `analysis/EVIDENCE_LEDGER.md` for the claim-level record and
`analysis/JOURNAL_READINESS.md` for unresolved submission gates. The complete
1,840-run historical ledger preserves all executions and records 640 input-
invalid exclusions without using outcomes. Only validity-qualified identity,
routing, spatial-topology and fixed-budget conclusions enter the paper.

## Source projects and publication overlap

The mathematical foundation and regular-tree experiments were first developed
in the archived `../neurips` project. Reconstructed-tree and MICrONS analyses
were first developed in `../../dendritic-credit-routing`. The present article
is the sole scientific authority and states every load-bearing result directly;
the source repositories establish provenance rather than a prerequisite
reading order. Any arXiv or conference status must be disclosed in the cover
letter.

A second related preprint, arXiv:2607.24990, uses the same broader DendriNet
framework to study forward population readout. It is cited and disclosed but
is not part of the journal article's backward credit-routing evidence.
