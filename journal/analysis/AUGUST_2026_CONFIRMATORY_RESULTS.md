# August 2026 confirmatory results

## Scientific verdict

The new programme supplies enough distinct material for a journal extension,
provided the paper makes the conditional result its central claim.  The data
do not support a universal dendritic or shunting advantage.  They support a
more specific hierarchy: neuron-level signals provide coordinates, ancestry
provides structured candidate addresses, conductance regulates route gain,
and learning benefits only when the task requires fields that those routes can
express.  The matched point and flat controls are essential: they locate the
benefit in an address resource rather than in dendritic material alone.

## Theory added and checked

- A single implicit-adjoint identity now covers directed artificial trees,
  reciprocal passive cables and differentiable active steady states:
  `J_V^T q = grad_V L` and
  `partial L / partial g_i = -q^T partial F / partial g_i`.
- The general nonsymmetric operator uses the inverse transpose; the familiar
  Green's-function expression follows only when the passive conductance matrix
  is symmetric.
- The weighted capture/descent argument is stated in the consistent transformed
  coordinate `bar w = W^{-1/2} w`.  It is a result about a specified
  preconditioned update, not a generic claim about synaptic area.
- Address dimension, route coherence and off-route leakage formalize the
  resource supplied by nested ancestry routes.  Exact transport remains an
  information upper bound.
- The Sherman--Morrison focal-shunt corollary connects selectivity to the local
  input resistance, Green's-function column and perturbation conductance.
- A stochastic credit-operator theorem now separates retained task signal,
  finite-step gain and admitted gradient noise. It proves the projected
  stochastic crossover, point-partition residual, representation/coefficient
  error split, static-gain span invariance, branch-reliability shrinkage and
  passive-shunt no-sign-reversal boundary.

### Credit signal--noise phase

The locked 50-seed quadratic programme produced all 10,800 prespecified
seed--condition rows. At full task--tree alignment and `K=4`, ancestry exceeded
random rank-matched spectral capture by 0.5520 (95% interval
0.5353--0.5686; 50/50 seeds). Each task hierarchy `H=1,2,3,4` had minimum mean
loss at matching routed depth `D=H`; matched depth reduced loss by 0.0957
relative to the best mismatch (0.0782--0.1134 after averaging task depths
within seed; 48/50 independent seeds; two-sided Wilcoxon
$P=3.00\times10^{-13}$). Full
depth was full rank and identical to stochastic backpropagation.

All 25 signal/noise condition means obeyed the analytical rule that a common
projection helps only when rejected noise exceeds discarded signal.
Reliability-aligned gains exceeded the best global gain by 1.2415 at maximum
heterogeneity (1.1160--1.3728; 50/50 seeds), but an explicit point gate with
the same gains matched exactly. Static nonzero route gains preserved capture
to `1.33e-15` while changing Gram conditioning.

A secondary initialization reanalysis reconstructed 64 minibatches per seed
for the completed 2,700-fit factorial. Across 540 dendritic route conditions,
the signal--noise--curvature utility predicted observed one-step progress with
Spearman `rho=0.937` (seed-block interval 0.934--0.941) and final accuracy with
`rho=0.916` (0.905--0.929), outperforming spectral capture alone. This is a
fixed-state theory validation and existing-data diagnostic, not a trained
positive-conductance reliability result or superiority to exact full-batch BP.

### State-matched positive-conductance reliability (superseded pilot)

The numerical results below are retained for provenance, but the experiment
used the $c=1$ reliability attenuation with a $c=0.5$ step. Publication
inference now uses the fresh, prospectively frozen correction in
`source_data/positive_conductance_reliability_step_consistent/`.

A separately locked experiment trained positive excitatory conductances in an
eight-branch model with nonnegative input rates. Fixed nonnegative shunts were
paired with an oracle compensating current that held each branch voltage
identical while retaining the smaller physical input resistance in the local
eligibility. All 2,250 rows across 50 confirmatory seeds completed; the maximum
state mismatch was `2.22e-16`, point/additive equivalence errors were zero and
the maximum relative finite-difference error was `2.13e-8`.

At maximum reliability heterogeneity, aligned shunting improved one-step test-
loss reduction over the best global shunt by `0.001913` (95% interval
`0.001789--0.002038`; 50/50 seeds). Its mean one-step advantage over no shunt
was `0.000947` (`0.000114--0.001914`), but only 28/50 pairs favored aligned and
the Wilcoxon test was not significant (`P=0.249`). After 40 updates, aligned
shunting reduced final loss relative to global by `0.019052` and also beat
shuffled and anti-aligned shunts, but its loss was `0.003810` higher than the
unshunted noisy rule. Exact clean backpropagation remained best, and an
explicit point gate matched aligned shunting exactly.

Interpretation: positive conductance can physically realize the oracle
reliability factor and its aligned-versus-global ordering under a state clamp.
The experiment does not learn shunt placement, show a dendrite-exclusive
operation or establish sustained superiority over unshunted local learning.

## Completed trained and mechanistic experiments

### Continuous path-demand boundary

The frozen Fashion-MNIST branch-conflict experiment completed all 2,400
confirmatory fits across 20 paired seeds, branch counts `B=2,4,8`, eight nested
conflict doses and five routing conditions. Every branch received a nonzero
view, while a downstream context gate selected one branch for the somatic
readout. At zero conflict, correct-path minus neuron-shared accuracy was
`+0.08`, `+0.01` and `-0.16` percentage points for `B=2,4,8`, respectively,
all within the prespecified two-point equivalence margin. At full conflict the
effects were `+34.90`, `+58.20` and `+57.40` points.

The seed-wise accuracy-advantage slope on conflict was positive in 20/20 seeds
at every branch count: means `0.2456`, `0.6478` and `0.7503`, with 95% paired-
seed intervals `0.2105--0.2801`, `0.6376--0.6586` and `0.7382--0.7622`.
Chance crossings of the mean shared-credit curves were `0.964`, `0.668` and
`0.573`, close to the analytic shared-mode boundaries `1`, `2/3` and `4/7`.
The seed-wise transition order `B=8 < B=4 < B=2` held in all 20 pairs.
Analytic backpropagation and the explicitly gated grouped-point condition were
identical to correct routing; the maximum analytic-versus-autograd gradient
error was `3.43e-7`.

Interpretation: the experiment supplies the missing continuous existence test.
Branch-selective information becomes useful when off-route eligibilities
conflict, and its failure boundary is predicted by branch count. It does not
show that biological dendrites are uniquely necessary: the selector can be
carried by a routed teaching coefficient or by an equivalent branch-local
eligibility gate, and the mechanism-matched task does not estimate prevalence
in natural learning.

### Within-neuron address hierarchy

The shallow credit-reversal phase used ten paired seeds and 80 frozen fits.
Correct two-subtree routing reached 0.8063 held-out accuracy versus 0.7624 for
one neuron-shared coordinate, 0.1963 for a fixed within-neuron derangement and
0.5847 for a random dense rank-two field.  Correct-minus-shared was +4.39
percentage points in 10/10 seeds; correct-minus-deranged was +61.00 points in
10/10.  Correct routing eliminated the context-switch forgetting seen under a
shared coordinate.  Exact transport, analytic backpropagation and an explicitly
gated point implementation matched the correct route exactly.

The complete frozen factorial contained 2,700 fits, 20 paired seeds, four
feedback budgets, five parameter-matched representations and six route
families.  Correct-ancestry accuracy was 0.1866, 0.4486, 0.8010 and 0.8100 at
`K = 1, 2, 4, 8`.  Against the best matched non-anatomical control selected
within seed, ancestry lost at `K=1,2`, won at `K=4` by 0.0127 (95% paired
bootstrap interval 0.0059--0.0197; 15/20 seeds; two-sided Wilcoxon
`P=0.0048`) and tied at `K=8`.  The task-matched tree exceeded a degree- and
depth-matched rewired tree by 0.2315 at `K=2` and 0.0502 at `K=4`, positive in
20/20 seeds in both cases.  Dendritic, gated-point, flat-compartment and
grouped-point implementations were numerically identical when supplied with
the same fields.

Interpretation: this is a positive, bandwidth-dependent structured-address
result.  It is not evidence that a dendritic implementation is uniquely more
powerful than a point model that is explicitly given the same parameter groups
and routing coordinates.

### Exact-resource nonlinear physical depth

The production-model bridge fixes 64 somata, eight nonsomatic branch units per
soma, 14,336 active contacts, 21,760 candidate mask slots, 66,178 trainable
parameters and 2,944 persistent state scalars while changing physical depth
from D1 `[8]` through D2 `[2,3]` to D3 `[2,1,2]`. The task multiplies a distal
positive-rate class signal by fine, coarse and global gains and supplies three
local inhibitory sensor blocks.

Across fresh paired seeds 10200--10209, aligned shunting BP accuracy was
0.6096, 0.6752 and 0.9182 at D1--D3. D3 minus D1 was +30.86 percentage points
(95% paired-seed bootstrap interval 30.45--31.30; 10/10 positive; exact
sign-flip `P=0.00195`). Zero-alignment and trial-shuffled controls had depth
effects +0.02 and -0.01 points. The corrected width-preserving reversed-tree
control had -0.43 points (-0.67 to -0.15), so the aligned-minus-reversed
interaction was +31.28 points (30.67--31.82; 10/10). Raw additive BP fell from
0.5733 at D1 to 0.5190 at D3.

The complete LocalCA cohort shows aligned D3-minus-D1 effects of +16.95 points
under one shared-soma coordinate and +27.90 points under exact path transport,
both positive in 10/10 seeds. Exact-resource reversal removed the depth effect:
the aligned-minus-reversed interactions were +17.72 points (17.27--18.18) and
+28.68 points (28.07--29.27), respectively, both positive in 10/10 seeds. The
first cyclic placement was invalidated solely by the prospective
candidate-slot equality gate and was replaced before inference.

A read-only replay of all 30 aligned BP checkpoints reproduced the selected
autograd conductance-gradient vector under exact path transport (minimum cosine
0.9999999999994). Shared-soma cosine fell from 0.9457 at D1 to 0.7990 at D2
and 0.7674 at D3; the paired D3-minus-D1 change was -0.1784 (-0.2082 to
-0.1490; 0/10 positive). Thus task-aligned divisive stages can add useful
forward computation, while path-specific credit becomes more important as
physical depth differentiates branch state. This remains a calibrated
mechanism-matched task, not a dendrite-exclusive expressivity claim.

### Literal grouped point and independent H2 replication

A final frozen matrix added 220 fits. Sixty H3 fits replaced every serial child
aggregator by a parameter-matched direct-to-soma projection while preserving
the local branch modules, masks and initialized conductances. The grouped-point
model stayed flat across depth. Serial-minus-grouped-point was +30.87 points at
aligned D3 (30.48--31.31; 10/10 seeds) and -0.44 after placement reversal; the
architecture-by-placement interaction was +31.31 points. The earlier grouped
star differed from the literal point model by only +0.06 points at aligned D3
and failed the frozen directional gate.

The remaining 160 fits used a fresh H2 generator, exact-resource D1 `[8]` and
D2 `[4,1]` morphologies, two sensor tiers and seeds 10300--10309. Serial BP
gained +30.84 points when aligned and lost -1.56 after reversal, producing a
+32.40-point interaction. Grouped-point BP stayed flat. Shared-soma LocalCA
retained +21.20 points and exact-path LocalCA +31.38 points; their placement
interactions were +22.93 and +33.36 points. Every positive primary contrast
was positive in 10/10 seeds. All 220 fits passed completeness, finite-metric,
seed, no-fallback and exact-resource gates. This is a hierarchy-depth
replication within the same calibrated task family, not an independent dataset
or evidence that natural tasks generally favor serial trees.

### Input-valid prospective learning

An outcome-independent audit considered 1,840 historical executions and
retained 1,000.  It excluded 640 signed-input positive-conductance shunting
runs, including the entire 400-run inhibitory-dose family, without using an
accuracy or gradient outcome.  In the retained central cohort, neuron-indexed
feedback exceeded scalar feedback by 4.92--6.99 percentage points on MNIST and
9.76--14.16 points in the additive noise task; all 120 paired comparisons were
positive.  Correct neuron-to-tree ownership added 0.15--0.70 points at matched
bandwidth and was favored in 58/60 paired seeds; five of six condition tests
survived Benjamini--Hochberg correction.

At 120 retained backpropagation checkpoints, mean dendritic-gradient cosine
was 0.027 for scalar and 0.586 for neuron-indexed feedback.  The corresponding
fields were descent directions at 75/120 and 120/120 checkpoints.  Gradient
cosine predicted retained norm-matched one-step progress with Spearman
`rho=0.893` (95% checkpoint-bootstrap interval 0.860--0.923).

The additive fixed-contact control retained 160 runs with 16 terminal branches
and 960--968 contacts per soma.  Depth four underperformed depth one in 10/10
seeds for every feedback condition: -7.92 points for scalar, -1.06 for
neuron-indexed, and -1.37 for exact transport or backpropagation.  The 240-run
valid spatial-map control improved learning by 0.36--2.57 points, including
under backpropagation and on a randomly projected task; it is therefore a
forward sparse-coverage result, not a credit-routing effect.

### Focal conductance and active steady states

The passive sensitivity matrix contained 16,362 site--condition--intervention
rows across 101 sites in eight cells.  At fixed 0.5 nS, shunting-minus-additive
localization was 0.0104, 0.0073 and -0.0039 at membrane resistances 300, 1,000
and 15,000 ohm cm2.  At high resistance, distributed background conductance
raised the input-conductance-normalized contrast from -0.0004 to 0.1210 and
0.2587.  All 8,181 focal-shunt rows attenuated descendant gradient magnitude;
none enhanced or reversed its sign.

The active ensemble accepted 64 Na/K/Ca/HCN/NMDA channel draws in each of eight
reconstructed cells: 512 stable nonlinear equilibria and 38,784
site--draw--condition rows.  At unit input-conductance dose, the
shunting-minus-additive localization contrast was 0.3817 (95% cell-bootstrap
interval 0.3556--0.4091), positive in 8/8 cells (`P=0.0078`).  Descendant
energy fell to 0.330 of baseline; every modeled descendant gradient was
attenuated and none was enhanced or sign-flipped.  This is a positive local-
Jacobian steady-state result, not a simulation of channel kinetics or spikes.

### Reconstructed-tree task alignment

All 13 deterministically eligible visual-response scans nested in seven target
cells retained the null boundary: partial shared-path effect -0.0485 (95%
target-bootstrap interval -0.2237--0.1183; 3/7 targets positive;
`P=0.8125`).  A separate full compressed-tree analysis retained segment-level
states and exact adjoints for 520 fits.  Mean normalized test MSE was 0.7877 for
exact transport, 0.8321 for topology, 0.8333 for random routes and 0.8391 for
site shuffling.  Topology-minus-shuffle was 0.0070 (-0.0207--0.0321), and
topology-minus-random was 0.0013 (-0.0090--0.0113).  These analyses do not
establish endogenous morphology--task alignment. The archived NWB trial
stream has only approximately 0.1-s intertrial gaps, so a pretrial baseline
or post-trial lag window would reuse the adjacent stimulus; we retained the
frozen full-trial endpoint and the existing repeat-reliability sensitivities
rather than introduce a contaminated response definition.

## Detached exact-transport/backpropagation audit

The locked audit completed all 320 runs and 160 exact/backpropagation pairs
from detached tracked-clean root commit `74792ca` and nested data commit
`4f3612a`. Every resolved configuration, seed record, final metric, newest log
and finite stage-complete checkpoint passed; there were no transport-shape
fallbacks or critical accepted-log matches. Forty-two runs used the audited
batchwise terminal-metric recomputation after full-dataset materialization
exceeded device memory, and every accepted endpoint was finite.

After averaging depths one through four within seed, exact-minus-
backpropagation accuracy was +0.005 percentage points for additive MNIST (95%
paired-seed bootstrap interval -0.028 to +0.041; 4/10 positive; descriptive
two-sided Wilcoxon `P=1.000`), -0.013 for shunting MNIST (-0.047 to +0.022;
4/10; `P=0.492`), -0.112 for additive noise resilience (-0.465 to +0.247;
4/10; `P=0.625`), and -0.098 for shunting noise resilience with a ReLU input
transfer (-0.653 to +0.390; 7/10; `P=1.000`). Across all 160 raw method pairs,
the mean difference was -0.0545 points and the largest absolute pair
difference was 5.90 points. Every interval includes zero. This is evidence of
close training-level agreement at the condition-average scale, not a formal
equivalence test or an identity claim.

## Fresh raw-additive CIFAR-10 feedback ladder

The configuration-limited provisional additive baseline was replaced by an
independently frozen 20-seed ladder at the validation-selected operating point.
All 80 fits passed source, pairing, calibration, checkpoint and convergence
audits. Mean test accuracies were 34.47% for strict scalar, 50.87% for
neuron-specific, 50.01% for exact path and 50.12% for matched BP.
Neuron-specific minus scalar was +16.393 percentage points (95% paired interval
15.878--16.908; 20/20 positive; Holm-adjusted directional
`P=5.57e-24`). Exact path minus neuron-specific was -0.859 points
(-1.171 to -0.547; 2/20 positive), so the prespecified path-resolution
promotion gate failed. Exact path and BP were formally equivalent within the
frozen +/-1-point margin (`P_TOST=1.58e-5`). The result supports a harder-data
neuron-identity bandwidth claim, not a CIFAR benefit of finer dendritic path
transport.

## Paper and presentation placement

- Main Figure 1: point neuron to dendritic coordinate/address/gain hierarchy
  and the general adjoint theorem.
- Main Figure 2: neuronal-coordinate bandwidth, ownership, fixed-budget depth,
  the Fashion-MNIST replication and the branch-conflict transition overview.
- Main Figure 3: branch-conflict task and analytic boundary followed by the
  complete 2,700-fit nested-address bandwidth factorial.
- Main Figure 4: stochastic credit phase, hierarchy-depth crossover,
  reliability controls and frozen-factorial utility reanalysis.
- Main Figure 5: exact-resource physical depth, point controls, BP--LocalCA
  decomposition, alignment dose, literal grouped point and H2 replication.
- Main Figures 6--9: nonlinear-depth saturation and task-family boundaries,
  reconstructed route capacity, focal conductance and the measured-alignment/
  animal boundary.
- Supplementary Figures S1--S29 retain the derivations, audit controls,
  robustness analyses, active/full-tree details and quantitative inventories.
- Supplementary Figure S4D reports the fresh raw-additive CIFAR-10 ladder; the
  38.98% provisional cohort remains excluded.
- The combined manuscript contains the Supplementary Information in the same
  128-page PDF. The current workshop package contains a 26-slide technical core
  and seven backup slides in a 34-page, native-vector PDF, with speaker notes and
  15/30/45-minute cut maps.

## Submission boundary

The scientific package is being prepared for Nature Communications.  It must
not be described as submitted while the NeurIPS exclusivity/status question is
unresolved.  The remaining non-computational gates are final authorship and
CRediT approval, immutable archive identifiers, current journal forms and the
submission-day related-work disclosure.
