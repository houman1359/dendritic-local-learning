# image_generalization (panels C and D of N4C) — SI text pass ledger (2026-09-23)

- Source key: N4C, panels C and D only (`('N4C','CD')`, printed as panels A and B of the SI sheet)
- Builder: `scripts/build_supplementary_figure_s04_native.py --output figures/supplementary/figure_cifar_ladders_native.pdf --confirmatory-analysis-dir source_data/cifar10_additive_feedback_ladder_confirmatory --shunting-analysis-dir source_data/cifar10_shunting_feedback_ladder_confirmatory --allow-convergence-flags` (it also writes `figure_cifar_ladders_native.png`)
- Output: `figures/supplementary/figure_cifar_ladders_native.pdf`
- Reproducibility gate: PASS with the right interpreter.
  - `python3` (matplotlib 3.10.6) differed by one byte, the PDF Producer string "v3.10.6" against "v3.10.9". The original PDF and PNG were restored.
  - `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python` (matplotlib 3.10.9, with scipy) reproduced the PDF (sha256 f24e2fd8…fc31ff3c) and the PNG byte for byte.
- Edited: **no**. The builder, PDF and PNG are byte-identical to their pre-pass state.
- Audits of the unchanged PDF: strict 0 violations; `letter_ink_audit.py` 0 problems.

## Removed

Nothing. The text in C and D is:
- titles `Shunting CIFAR-10` and `Raw-additive CIFAR-10`
- band label `BP ±1 pp` (in both panels)
- y label `CIFAR-10 accuracy` (C only)
- tick labels `strict` / `scalar`, `neuron-` / `specific`, `exact` / `path`, `BP`

## Caption additions

None.

## Kept

- `Shunting CIFAR-10`, `Raw-additive CIFAR-10`: condition labels of three words or fewer, with no verbs or findings, that tell the two otherwise identical panels apart. The dataset name also separates them from the Fashion-MNIST panel beside them on the SI sheet.
- `BP ±1 pp`: the direct label of the grey reference band. Its meaning does not follow from the axis, and without it the band needs the caption to be read. The caption already states "Grey bands mark $\pm1$ percentage point around each backpropagation mean; the band alone does not establish equivalence."
- `CIFAR-10 accuracy` and the tick labels: axis text.

## Convergence qualifier

No qualifier or convergence note appears on the C/D artwork; the builder draws none. The caption carries it: "Shunting results are descriptive: one scalar run triggered the convergence criterion at 400 epochs; all twenty seeds remain included." Nothing moved.

## Layout changes

None.
