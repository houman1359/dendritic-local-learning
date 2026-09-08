# Dendritic morphology as a dictionary for local credit assignment

This directory contains the canonical manuscript prepared as a *Nature Communications* Article. Its central question is which spatial distinctions a task's learning signals must preserve. The main sequence progresses from neuron identity on image tasks to branch selection under conflict, ancestry grouping under hierarchical distractors, and learned credit geometry under higher-order interactions. Morphology supplies spatial feedback patterns; local conductance can change their gains.

Nine main figures follow this task-to-credit argument. A matched pairwise/quartic comparison tests fixed spatial credit on the same trees, inputs and initializations, and a context-dependent conductance task tests local inhibitory gating with explicit proximal-credit controls. Physical-depth experiments examine the dependence on optimizer and training budget. Reconstructed arbors test the capacity of anatomy-derived routes beyond a shared broadcast. The final figure separates empirical ancestry–response similarity from an offline transfer-geometry diagnostic. The measured responses do not establish endogenous dendritic credit assignment.

The Supplementary Information contains 35 figures, S1–S35. It retains the representability and morphology-estimation analyses, the unsuccessful prospective initialization-based selector, Boolean controls, alternative optimizers and expanded biological/model diagnostics. These support the credit argument without replacing it with a general morphology-selection claim. The current display map is in [figures/README.md](figures/README.md).

## Files and reproducibility

- `main.tex` and `supplementary/supplementary.tex` are the canonical text sources; their local TeX inputs are required for complete builds.
- `main.pdf`, `supplementary/supplementary.pdf` and `main_with_supplementary.pdf` are generated reading copies. Their page counts must be taken from the final compiled files.
- `figures/main/` and `figures/supplementary/` contain numbered publication assets. Native components and older assets elsewhere are not additional numbered figures.
- `source_data/` contains retained numerical evidence. `source_data/credit_first_provenance/` supplies the current panel map; the package manifest records each source's original path, released path and checksum.
- `scripts/`, `code/`, `configs/` and `reproducibility/` contain implementations, protocols, checks and provenance boundaries.
- `submission/` contains editorial documents and generated release bundles.

The new matched credit study is implemented in `scripts/credit_rule_bridge/`; saved states and complete outcomes are in `source_data/credit_rule_bridge/`. Independent field, gauge and replay checks are in `scripts/credit_resolution_bridge/`. The six-arm MNIST controls are in `image_ladder_controls`; the separately frozen monotonic and opponent-tuning conductance studies are in `conductance_credit_demand`. The common-broadcast anatomy controls, depth-budget extension and ancestry-gain calculation have dedicated `anatomy_commonmode`, `physical_depth_budget` and `shunt_ancestry_gain` folders. Development choices, fresh tests and descriptive diagnostics remain identified in their protocols and result records.

The historical key `noise_resilience` refers to two different generators. The clean exact/BP rerun used three-class noisy-line images; projected-noise MNIST is the intended protocol for separate prospective cohorts, whose executed nested generator bytes were not pinned. Some inherited aggregates remain unresolved. See `source_data/release_task_identity/README.md` and the explicit release adapter in `code/release_noise/`.

## Build and release

From this directory:

```bash
make figures
make paper
make supplement
make combined
make audit
```

`rebuild_final_publication_figures.py` renders the current nine main and 35 supplementary figures from retained evidence. Historical builder numbers can differ from publication numbers; running an old compositor directly can restore an obsolete layout. Figure generation uses Python, NumPy, pandas, SciPy, Matplotlib and PyMuPDF. Manuscripts require pdfLaTeX/BibTeX. Biological reanalysis additionally requires the upstream data/cache access documented by each pipeline.

After the text, figures, panel provenance and Source Data are finalized, commit the scientific inputs and rebuild the software and other bundles in dependency order. [RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md) describes committed-source packaging, the isolated installation smoke and the explicit noise-generator choices. [OVERLEAF_README.md](OVERLEAF_README.md) describes the manuscript project. Existing archives can be stale even when their checksums are valid.

[The environment guide](code/release_noise/ENVIRONMENTS.md) distinguishes the
original scientific executions, historical source exports, supported CPU replay
and publication rendering. It includes a study-to-environment map and the
resolved CPU installation recipe. Public package CI, figure reconstruction and
full experimental replay establish different things.

To restore Source Data, use `manifest.tsv`'s `original_source` field, not the display folder name. Supporting evidence now stored under `Methods/retained_evidence/` still belongs at its recorded `source_data/` path for analysis. The software package includes `code/release_noise/restore_source_data.py`, which checks hashes and refuses to substitute display-filtered tables for complete sources.

## Prior dissemination and author information

The conductance-tree foundation, LocalCA rule and early regular-tree experiments were previously disseminated in arXiv:2607.03556. The forward population-readout preprint arXiv:2607.24990 is distinct related work. Their relationship to this Article is documented in `submission/extension_statement.md`. This project is being prepared as the sole journal paper; the recorded NeurIPS decision status and author declarations belong in the submission documents. Local compilation and packaging do not submit the manuscript.
