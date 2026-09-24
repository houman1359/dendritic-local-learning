# checkpoint_computation — SI clarity pass 2026-09-23

- Source key: direct (sheet pasted as built; no registry entry)
- Builder: `scripts/review_completion/checkpoint_figure.py` (no argparse; `main()` writes the fixed path)
- Output: `figures/supplementary/curated/checkpoint_computation.pdf` (git-tracked)
- Reproducibility gate: PASS with the repository interpreter
  `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python`
  (matplotlib 3.10.9, the producer recorded in the committed PDF). The unchanged builder
  rewrote the output byte-identically (sha256 fb8e2ecd9e1d099385db3353941f13c9b4cfce80defde2b318629cc57a49ba2b = the `git show HEAD:` blob).
  - The system `python3` (matplotlib 3.10.6) gave different bytes (0f8df241…). The only
    difference was the PDF Producer string. Text, words, 3,343 paths and the 150-dpi
    raster were identical. The HEAD blob was restored at once with `git show`.
  - The final output was built with the `.venv` interpreter.
  - Saved original: `si_pass/renders/checkpoint_computation_orig.pdf`. Before render:
    `si_pass/renders/checkpoint_computation_before.png`.
- New output sha256: 9fe2e02093d3339831751e35c80e719599f849c22c844b74ab73344e1a8e0704
  (518.4 x 440.0 pt, unchanged size).
- Checks:
  - Strict canvas audit: 0 violations.
  - `letter_ink_audit.py`: 0 problems.
  - Marks: all 3,343 paths kept, same count per colour, fill and width class.
- Side outputs of the builder, both gitignored, both rewritten on every run:
  - `source_data/checkpoint_computation/checkpoint_plotted.csv`: byte-identical to the
    2026-09-22 copy in `submission/nature_source_data/Supplementary_Figure_22/`
    (sha256 04e023d3…).
  - `source_data/checkpoint_computation/checkpoint_provenance.json`: now records the
    edited generator's sha256 and the current sha256s of the shared modules.
- The shared drawing modules were NOT edited:
  - `scripts/conductance_local_gate/figure.py` (`interaction`, `cancellation`)
  - `scripts/review_completion/population_figure.py` (`alignment_panel`)
  - The two labels are removed from the axes after the shared calls return
    (`drop_texts`, which asserts each label exists exactly once). The tick-label spacing
    is set the same way (`spread_ticklabels`).

## Removed

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `zero difference` (label of the dashed zero line) | ADD TO CAPTION. The A sentences name the hairlines, diamonds and whiskers but not the dashed rule. |
| B | `no cancellation` (label of the dashed line at one) | caption already states it: 'One indicates no cancellation.' |

## Caption additions

In `supplementary/curated/si_11_checkpoint_computation.tex`, append to the second
\textbf{A} sentence, which ends "…means and paired 95\% seed-bootstrap intervals.":

`; the dashed line marks zero`

The sentence then reads: `Hairlines connect the twenty paired seeds; diamonds and whiskers show means and paired 95\% seed-bootstrap intervals; the dashed line marks zero.`

## Kept

- `Shunting`, `Forward controls` (C): group labels for the two halves of the strip plot.
- `1.7×`, `87×`, `6.6×` (C): the effect-size annotations that carry the panel's result. The
  caption says 'labels give broadcast/gate ratios of means'.
- `exact credit`, `calibrated broadcast` (B): direct series labels.
- `within context`, `cross context` (D): compact marker key.
- `Target`, `Resistance h`, `Augmented hf′`, `Exact` (E–H): short condition labels for four
  heat maps that are otherwise identical.
- Tick labels (`4,096`, `16,384`, `Initial`, `Trained`, `Ordinary`, `Stress 3`, `Stress 3 rate 0.1`,
  `Tonic stress 3`, `Current stress 3`, `Exact`, `Broadcast`, `Resistance gate h`, `Wrong branch`,
  `Uniform RMS`).
- Axis labels, already in sentence case: `(Broadcast − gate) NMSE: opposed − aligned`,
  `Window (updates)`, `Distal gradient retained fraction`, `NMSE`, `Alignment (cosine)`,
  `Feature 1`, `Feature 2`, `Response`.

## Layout changes

- None to the canvas, rows, axes or letters. Both removed labels sat inside their axes.
- D: `Broadcast` is shifted 2 pt left of its tick and `Resistance` / `gate h` 2 pt right, so
  the two labels no longer run together. They were 0.9 pt apart and read as
  "BroadcastResistance"; they are now 4.9 pt apart. `Resistance` to `Wrong` is now 4.4 pt.
  This moves labels only. No tick, datum or scale changes.
