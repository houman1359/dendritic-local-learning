# Conductance-gate generalization: exploratory follow-up

This is separate from the frozen Figure 5 cohort. It must not silently replace
its results or be described as a preregistered replication.

## Questions and tasks

1. **Graded teacher:** retain the 24-conductance, seven-compartment teacher but
   sample context continuously, `c ~ Uniform(0,1)`, and give the two subtrees
   independent latent features. This tests sensitivity to assumptions of the
   original binary, duplicated-input task; it remains model-matched.
2. **Sensory selection:** independent features `z1,...,z4 ~ Uniform(-2,2)` feed
   the two subtrees. The externally specified target is
   `y = (1-c) [tanh(z1)+tanh(z2)]/2 - c [tanh(z3)+tanh(z4)]/2`, with binary `c`.
   It tests learning context-selected, oppositely tuned sensory readouts.
3. **Graded sensory mixture:** the same external target with continuous `c`.
   It tests whether inhibitory selection generalizes to simultaneous, weighted
   contributions. This target is not guaranteed representable by the neuron.

Positive inputs are `exp(z)` and `exp(-z)`. Proximal inhibitory activities are
`a0=10c` and `a1=10(1-c)`; all rules see the same inputs and forward neuron.
Every rule trains all 24 conductances and the same affine scalar readout.
These are steady-state computation analogues, not models of a particular brain
area, a temporal integration task, or evidence of endogenous teaching signals.

## Comparisons

All updates multiply the same exact *local* eligibility by somatic error and
a delivery factor. Exact paths and unit broadcast are reference conditions.
The distal-only gates are:

- Proportional: `1/(1+gI*a)`, the continuous shape tested with binary context in
  Figure 5.
- Relative resistance: `(1+sum(child couplings))/(1+sum(child couplings)+gI*a)`.
  This is the parent's resistance relative to its uninhibited value; it omits
  both child-specific coupling and soma-to-parent transmission factors.
- Swapped: the proportional rule using the other branch's inhibitory activity.
  This is an information-matched misassignment control, not another local
  mechanism (it requires the other branch's activity).

Proximal and somatic factors remain one; inhibitory conductances are not frozen.
Exact paths must never be evaluated by a local-gate gradient. Sentinel tests
and a 26-parameter central-difference gradient test enforce these distinctions.

## Pilot and decision rules

Three paired seeds, three rates (0.01, 0.03, 0.1), five rules, three tasks:
135 trajectories, each 4,096 Adam steps. Train/validation/test sizes are
2,048/1,024/4,096; minibatches 128. Shared initial states and minibatch streams
pair rules within seeds. Log conductances are bounded at [-7,7]; bound contacts
and gradient clipping are reported. Readout weights are unconstrained.

All rates and endpoints are retained. Validation chooses states and, only for
this exploratory pilot, rates within each seed. Test labels never select a
state or rate. NMSE divides by each split's target variance. Inspect absolute
exact-path error before interpreting a local-vs-exact gap: failed exact fits
or poor representability cannot establish a credit-assignment deficit.

Do not make inferential claims from three seeds. If warranted, choose rates on
these development seeds, fix outcomes and contrasts, and evaluate at least
20 fresh paired seeds. Report nulls and failures. A larger conductance network
and a reciprocal reconstructed neuron would test separate generalization steps;
the pilot does not establish either.

Run only on compute nodes. The worker reads a source snapshot and writes to
`kempner_project_b`; it uses NumPy, no GPU and no W&B. Keep the published
Figure 5 source and its numerical tables unchanged.

## Completed fresh-seed follow-up

The pilot is complete. `followup.py freeze --root RUN_ROOT` selected one rate
per task/rule using mean validation loss over the three development seeds,
then wrote an internally timestamped source-hashed protocol before fresh runs.
Twenty new paired seeds (2026091800–2026091819) completed all three tasks,
five rules and three rates: 900 additional trajectories. Primary summaries
use the fixed development-selected rate, not a fresh per-seed rate choice.
`followup.py run --root RUN_ROOT --seed SEED` validates the source freeze;
`followup.py report --root RUN_ROOT` refuses an incomplete cohort and reports
all three tasks with descriptive paired-seed intervals.

`publish.py --root RUN_ROOT` replays all 900 selected parameter states before
writing compact source tables and Supplementary Table S13. It does not
overwrite differing outputs. The main text and Supplementary Section S5
report this cohort separately from Figure 5's original curves. No population,
spiking or reconstructed-neuron training is claimed.

The relative-resistance gate improved over broadcast in all twenty seeds of
each task. The original proportional gate was worse than broadcast on graded
mixtures, and exact credit also retained appreciable mixture error. Bounds,
finite training duration, and different development-selected rates limit
interpretation. All outcomes, including this negative result, are retained.
