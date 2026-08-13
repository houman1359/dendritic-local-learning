# Reproducibility Notes

These notes give a compact map from the manuscript to the repository contents.
They are intentionally protocol-level for the current preprint. The public
release after publication will identify an immutable tag, the pinned
environment, end-to-end launch scripts, and expected summary hashes.

Run commands from the repository root:

```bash
cd drafts/dendritic-local-learning/neurips
```

## What Is Included

- Core dendritic model and LocalCA implementation.
- Training and diagnostic scripts for the reported experiment families.
- Representative configuration files for the main architectures and controls.
- Figure-generation scripts for the main and supplementary figures.
- Tests covering key dataset and implementation utilities.

## Main Experiment Families

- Exact-gradient reconstruction and gradient-fidelity diagnostics.
- Layer-soma factorial diagnostic separating soma-error reuse from within-tree
  path transport.
- Fixed-forward-state factor decomposition separating local resistance,
  backward resistance, driving force, and parent transfer derivatives.
- Norm-matched finite-update checks comparing LocalCA and exact-gradient loss
  changes on the same held-out batches.
- Fifteen-seed identity-transfer and feedback-definition replications.
- Fifteen-seed 3F architecture-by-reactivation-initialization factorial.
- Path-gain, exact-error rank, and implemented-feedback fidelity diagnostics.
- Post-training inhibitory-conductance interventions.
- Transported-error oracle and feedback-construction controls.
- Matched-capacity MNIST, Fashion-MNIST, and figure-ground MNIST performance.
- Supplementary stress tests: morphology, feedback noise, CIFAR-10 compact
  controls, FA/DFA, cue routing, and low-rank feedback.

## Figure Regeneration

The following commands regenerate the manuscript figures from the summaries
currently present in the repository:

```bash
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_figure1_schematic.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_theory_diagnostics_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_neurips_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_mechanistic_audit_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_alignment_norm_dynamics_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_cifar_sweep_comparison_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_morphology_ie_regime_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_inhibition_causality_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_cue_routing_figures.py
```

These are figure/summarization entry points, not yet a frozen from-scratch
reproduction artifact. The public release after publication will provide a
single-script path from raw training launches to checked summaries.

## Rebuttal Follow-up Diagnostics

The fixed-state and one-step analyses consume completed sweep directories:

```bash
PYTHONPATH=../../../src:$PYTHONPATH python scripts/measure_fixed_state_factorial.py \
  --sweep-dir /path/to/completed/sweep \
  --output-dir analysis/fixed_state_factorial \
  --split test --batch-size 128

PYTHONPATH=../../../src:$PYTHONPATH python scripts/measure_norm_matched_one_step.py \
  --sweep-dir /path/to/completed/sweep \
  --output-csv analysis/norm_matched_one_step.csv \
  --split test --batch-size 128

python scripts/summarize_fixed_state_followups.py \
  --fixed-state-csv analysis/fixed_state_factorial/fixed_state_factorial_checkpoints.csv \
  --one-step-csv analysis/norm_matched_one_step.csv \
  --output-dir figures/data/fixed_state_followups
```

The fixed-state variants are diagnostic hybrids: they reuse one recorded
forward state and are not interpreted as self-consistent trained models.

The stage-resolved error-field audit keeps separate feedforward dendritic
populations separate and averages captured-energy fractions per checkpoint:

```bash
PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_error_field_decomposition.py \
  --device cuda --split train valid test
```

This writes `figures/data/error_field_decomposition_runs.csv` and
`figures/data/error_field_decomposition_summary.csv`. The default manifest is
the five matched MNIST checkpoints per core plus the five shunting
noise-resilience checkpoints used in the rebuttal audit. Repeat `--run-dir` to
analyze another explicit checkpoint set.

The trained inhibitory operating point and conductance-stage finite-difference
calibration are reproduced with:

```bash
PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_inhibitory_operating_point.py \
  --device cuda --split test
```

The calibration verifies the algebraic descendant-only path-gain ratio and
measures the range of the first-order approximation. It is not presented as an
independent empirical discovery.

The fixed-forward-state backward-only dose response and the coupling-depth
audit are reproduced with:

```bash
PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_backward_inhibition_dose_response.py \
  --dataset mnist --split train valid test \
  --output-prefix figures/data/backward_inhibition_dose_response_mnist

PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_backward_inhibition_dose_response.py \
  --dataset noise_resilience --split train valid test \
  --output-prefix figures/data/backward_inhibition_dose_response_noise

PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_effective_dendritic_depth.py
```

Removal fractions above one in the dose-response script are nonphysical
sensitivity extensions. The output records when the counterfactual
conductance floor is active. The depth audit reports both nominal and effective
stages and distinguishes an initializer-floor route from learned pruning.

To connect field geometry with parameter updates and finite-step progress:

```bash
PYTHONPATH=../../../src:$PYTHONPATH \
  python scripts/measure_feedback_learning_relevance.py \
  --split train valid test \
  --output-prefix figures/data/feedback_learning_relevance
```

This diagnostic holds checkpoints and batches fixed while comparing global
scalar, submitted hybrid, ancestry-shared, exact-soma ancestry-shared, and
exact-transport fields. Oracle conditions are analysis bounds, not proposed
biological feedback rules.

The extended identity-transfer and feedback-definition sweeps use
`configs/sweeps/sweep_rebuttal_identity_3f_mnist_extra10seed.yaml` and
`configs/sweeps/sweep_rebuttal_feedback_definition_3f_mnist_extra10seed.yaml`.
The matched initialization-policy factorial uses
`configs/sweeps/sweep_rebuttal_3f_init_policy_factorial_mnist_15seed.yaml`.
The source-backed figure summaries include
`figures/data/revision_exact_transport_factorial_grouped.csv` for the
five-seed 3F/5F exact-error panel and
`figures/data/feedback_definition_replication/feedback_definition_details.csv`
for the fifteen-seed neuron-wise-feedback intervention.
Main Figure 5's recovered error-source control uses the twelve local-decoder runs in
`figures/data/local_mismatch_recheck_runs.csv`; the figure script computes its
means and sample standard deviations directly rather than using the hard-coded
values in the arXiv-era plotting code.
After training, the tracked summaries are produced with:

```bash
python scripts/summarize_identity_transfer_replication.py \
  --diagnostic-root /path/to/original/diagnostics /path/to/extension/diagnostics \
  --sweep-root /path/to/original/sweep /path/to/extension/sweep \
  --output-dir figures/data/identity_transfer_replication

python scripts/summarize_feedback_definition_control.py \
  --factorial-root /path/to/completed/feedback_sweep \
  --details-csv figures/data/feedback_definition_mnist_3f.csv \
  --diagnostic-csv /path/to/original/branch_gradient_checkpoint_summary.csv \
                   /path/to/extension/branch_gradient_checkpoint_summary.csv \
  --output-dir figures/data/feedback_definition_replication

python scripts/summarize_init_policy_factorial.py \
  --sweep-root /path/to/completed/init_policy_factorial_sweep \
  --expected-seeds 15 \
  --output-dir figures/data/init_policy_factorial_3f_mnist_15seed
```

The matched resource profile runs the same resolved configuration once with
LocalCA and once with standard backpropagation:

```bash
PYTHONPATH=../../../src:$PYTHONPATH python scripts/profile_training_resources.py \
  /path/to/resolved_config.yaml \
  --output-dir /path/to/profile/local_ca \
  --learning-strategy local_ca --profile-scope end_to_end --seed 42

PYTHONPATH=../../../src:$PYTHONPATH python scripts/profile_training_resources.py \
  /path/to/resolved_config.yaml \
  --output-dir /path/to/profile/backprop \
  --learning-strategy standard --profile-scope end_to_end --seed 42
```

The manuscript-level export is
`figures/data/training_resource_profile.csv`. It reports process-wide peaks
with identical hooks and final evaluation; it is not an optimized kernel or
hardware-energy benchmark. Use `--profile-scope training` to reset and measure
the CUDA peak at the main training phase alone.

## Environment

The repository currently includes dependency specifications at:

- `requirements.txt`
- `pyproject.toml`
- `setup/environment.yml`

The public release will identify the exact environment file and commit used for
the public artifact.

## Build Commands

```bash
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
```

After building, scan logs for unresolved citations, references, overfull boxes,
and fatal errors:

```bash
rg "Undefined|Warning: Citation|Warning: Reference|Overfull|Error|Fatal" *.log
```
