# NeurIPS-to-journal figure lineage audit

The journal paper is an extension of the NeurIPS/arXiv study, not an
independent redraw of it. This audit fixes the visual and computational
lineage of every inherited display.

## Exact inheritance

- Figure 1 panels a-c use the original Figure 1 functions byte for byte.
- Figure 2a reproduces the submitted exact-gradient diagnostic with the same
  two metrics, data, jitter, markers, medians, and log scale.
- Figure 2e imports the original path-gain panel callable directly and uses the
  same five paired-seed values.
- Supplementary Figures S1-S3 are the unchanged final NeurIPS/arXiv Figures
  3-5. Their PDF and PNG assets are byte-identical. Exact generator snapshots
  and the original numerical plotting inputs are stored with the journal.

## Journal extensions

The prospective learning, reconstructed-morphology, MICrONS, perturbation,
measured-response, animal-reanalysis, and conditional-alignment panels are new.
They use the unchanged NeurIPS style module as a single source of truth for
color, typography, line weights, panel labels, geometry, and export settings.
Supplementary Figure S4 is an expanded replot of the regular-tree evidence and
is labeled as such rather than described as an exact reproduction.

All ten main and eleven supplementary figures are authored on the canonical
7.2-inch canvas and included at full text width. This keeps nominal type,
axis-weight, and marker scales consistent in the compiled article. The visual
audit inspected every current PNG together, not only the inherited panels; no
panel overlaps or clipped labels were found. All submitted PDFs remain the
publication assets, so plots and schematics are vector rather than dependent
on raster resolution.

## Frozen hashes

- `scripts/neurips_style.py`: `47b62db8656dd576f2c52a17cddb170a4be35f9a5fb7cacbe2cc2f646b76da46`
- `scripts/figure1_neurips_components.py`: `f57439109e3d6e411b1a05abeab86d8c7a994eed6ce21215f3d2f978052e1926`
- `scripts/inherited_neurips/generate_neurips_figures.py`: `d03020396906e4ea59db2464dcfd1ad3a33da213d9c7cc970f637cd55e94223d`
- `scripts/inherited_neurips/generate_revision_figures.py`: `ded3e36ea4684443b4e5f5f6e7a181c50793f9a9653e209cf40d5ec920ad510b`
- `scripts/inherited_neurips/generate_theory_diagnostics_figures.py`: `213610d30ebaebfd7e7417da7b3787221bc8a578bb2d4da9f9a9f82780aede80`
- Supplementary Figure S1 PDF: `88c0fe5290fc4b07539af46388f1ba150ce09c30a3b236881b14f153869c327f`
- Supplementary Figure S2 PDF: `90e4d2378aa8579eeb4f7d6bf47034ea31545730710ad4aee00c6df21cec031a`
- Supplementary Figure S3 PDF: `b51c005792b7165a94f02d661820e1ae7df104304be440d21e1e3e59258431fc`

The automated audit fails if any frozen hash changes or if a production figure
generator bypasses the shared style module.
