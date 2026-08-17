# Nature Communications revision plan after full-manuscript review

Date frozen: 2026-08-10

## Editorial decision

Retain the cautious title, *Dendritic topology and conductance organize local
credit assignment*, and target **Nature Communications** after the NeurIPS
decision. The current evidence is rigorous enough for a serious journal
manuscript. A completed frozen phase-1 experiment now shows that trained
networks benefit from assigning two task-dependent coordinates to the correct
sibling subtrees, while an equivalent gated point implementation matches that
benefit. The evidence therefore supports an address-resource claim, not a
uniquely dendritic superiority claim.

The revision distinguishes four spatial levels everywhere:

1. global scalar or sign;
2. neuronal identity / coordinate ownership;
3. within-tree address;
4. route gain.

The existing prospective derangement tests level 2, not level 3. Exact
transport spans levels 2--4 as an information upper bound.

## Implemented immediately from existing evidence

| Review item | Action | Location |
|---|---|---|
| General theory was split between a directed path product and reciprocal cable | Added the implicit-adjoint theorem, the local conductance corollary, and the inverse-transpose qualification | Main theory; Supplementary S1 |
| Weighted-coordinate algebra | Defined `bar(w)=W^{-1/2}w`, stated the original-coordinate preconditioned step, and restricted the descent interpretation to exact gradients in the transformed coordinates | Main capture result; Supplementary capture theory |
| Tree ownership was described as a dendritic address effect | Renamed the Results claim and figure language; explicitly states that the control does not assign different coordinates within one tree | Main Results and Figure 3; talk |
| Address lacked a formal definition | Defined a feedback-dictionary column as an address and separated point, depth, ancestry, and dense upper-bound dictionaries | Main topology result |
| Figure 1C was overloaded | Replaced it with a four-level spatial-credit hierarchy; retained the high-quality inherited A/B visual language | Figure 1 generator and caption |
| Main prospective figure lacked the breadth expected of a journal article | Reorganized it as a legible nine-panel 3-by-3 display: identity, exact/backprop agreement, ownership, clean implementation, fixed-budget depth, trained subtree address, gradient geometry and switch interference | Figure 3 generator and caption |
| Focal effect was called descendant-specific / redistributed | Changed to descendant-enriched / preferentially changes; clarified reciprocal off-route effects | Main focal result and talk |
| Somatic voltage clamp terminology | Changed to state-matching compensatory current injection and stated that its derivative is not a continuously adjusted clamp | Main and Supplementary Methods; talk |
| Focal Shapley cross-reference | Corrected to Supplementary Section S7 | Main focal result |
| Major-branch response model was overinterpreted | Reframed as a coarse major-branch surrogate; states that sites on one major branch share task-dependent error and fine topology is unresolved | Main functional result and talk |
| P+ test was called a sign test | Changed to exact one-sided random-sign permutation | Main animal result; Supplementary table |
| Signed-mode energy lacked uncertainty | Added aggregation definition, individual-animal fractions, animal bootstrap, and leave-one-animal-out sensitivity | Analysis script, Source Data summary, main and Supplementary text |
| Four-/five-factor terms were called branch-level | Defined them as slowly estimated stage-level scalar preconditioners and changed notation to stage index | Main and Supplementary Methods |
| Scalar baseline was underspecified | Added the exact mean-or-signed-maximum reduction with epsilon and identified fallback-rate logging as a required clean-rerun output | Supplementary S2 and feedback table |
| Miscellaneous consistency errors | Corrected FG to figure-ground MNIST, deleted stale structural-plasticity language, labeled selected-site tests one-sided, removed negative zero, and explained the bootstrap/Wilcoxon discrepancy | Supplementary text and tables |

## Essential new-data work before submission

### Gate A: decisive trained within-neuron address experiment --- complete

The phase-1 K=2 test was followed by a 2,700-fit factorial with 20 paired
seeds, $K\in\{1,2,4,8\}$, five matched representations, six route families,
a degree/depth-matched rewired tree and exact/backpropagation references.
Correct ancestry loses to the best unrestricted matched control at $K=1,2$,
wins at $K=4$ by 1.27 percentage points, and ties at full rank. It exceeds the
rewired tree at $K=2,4$ in 20/20 seeds. Dendritic, gated-point, flat and
grouped-point implementations agree exactly when supplied with the same
routes. This is the intended structured-address resource result, not a unique
dendritic-superiority result.

### Gate B: clean reruns of central inherited analyses --- complete

A 320-run exact-transport/backpropagation factorial completed from detached,
tracked-clean commit `74792ca`, with frozen configurations and source hashes,
ten paired seeds, two tasks, two cores and four depths. All checkpoints, logs
and endpoints passed. The four depth-averaged exact-minus-backpropagation
intervals include zero; the global mean across 160 pairs is -0.0545 percentage
points, with a 5.90-point maximum individual discrepancy. This is reported as
agreement, not equivalence. The measured-response rerun was replaced by the
stronger complete-tree analysis below, with exact input, script and output
hashes.

### Gate C: measured-response sensitivity --- primary ambiguity resolved

Every deterministically eligible scan was retained, giving 13 scans nested in
seven targets. The topology screen remains null. A separate 520-fit analysis
places mapped partners on the complete compressed tree, retains intermediate
states and exact segment-level adjoints, and matches restricted dictionary
rank within run; topology routes do not reliably improve over random or
site-shuffled routes. The NWB trials have only approximately 0.1-s intertrial
gaps, so a pretrial baseline or post-trial lag window would reuse the adjacent
stimulus rather than provide an independent response definition. We therefore
retain the archived full-trial raw-fluorescence endpoint and the existing
reliability sensitivities instead of manufacturing a contaminated baseline.

### Gate D: physical focal-shunt boundary --- complete

The frozen eight-cell passive analysis crosses
relative-local, fixed-absolute-nS and input-conductance-normalized doses with
three membrane resistances and three distributed-background levels. It reports
signed attenuation/enhancement, sign flips, descendant energy, local input
resistance and the electrotonic selectivity criterion. The active steady-state
extension adds Na, K, Ca, HCN and NMDA current families across 512 accepted
equilibria. The unit-dose shunt-minus-additive contrast is 0.382, positive in
8/8 cells, with attenuation and no sign flips. This remains a local-Jacobian
steady-state result rather than a kinetic-channel or spiking simulation.

### Gate E: step-consistent reliability mechanism --- complete

The original positive-conductance pilot paired the $\eta=1/L$ reliability
gain with an $\eta=0.5/L$ update. A prospectively frozen correction used 50
fresh seeds and the general fixed-step optimum. Independently computed
physical-shunt, point-gate and state-clamped unattenuated paths passed direct
finite-difference and trajectory gates. Aligned shunting improves over the
step-consistent global, shuffled and anti-aligned controls; its final
difference from no shunt is null. The manuscript now describes a
state-dependent approximation, not exact block-scalar realization.

### Gate F: same-span learning and spectral theory --- complete

A 4,800-row, 50-seed rate experiment compares orthonormal, raw nested and
statically scaled nested coordinates with the identical rank-eight projector.
The preregistered unconditional Haar advantage is falsified at low data and
reverses at high data. An exact modal bias--variance theorem predicts the
crossover, while a Gram-preconditioned control removes it. A deterministic
reanalysis of the frozen spectral phase adds the Ky--Fan capacity bound and
exact affine alignment thresholds. The passive theory now also bounds the
spectral norm of a focal inverse-conductance perturbation.

## Figure and manuscript sequence after new results

1. Figure 1: five-panel framework, adjoint theorem and four-level hierarchy.
2. Figure 2: exact factorization and feedback-identity foundation.
3. Figure 3: nine-panel prospective hierarchy spanning neuron identity,
   ownership, clean-source/depth boundaries and the completed K=2 subtree test.
4. Figure 4: complete trained subtree-address factorial.
5. Figure 5: reconstructed-tree address capacity and its matched controls.
6. Figure 6: inhibitory route anatomy and robustness controls.
7. Figure 7: focal mechanism, rank-one theory, descendant-enriched effect and
   physical boundary; the broader passive matrix is Supplementary Figure S11.
8. Figure 8: active focal sensitivity and complete-tree response boundary.
9. Figure 9: coarse measured-response and all-scan topology boundary.
10. Figure 10: constructive alignment control and independent signed neuronal
   coordinates, with their evidential roles explicitly separated.

This ten-figure, eleven-supplementary-figure organization retains the richer
journal evidence rather than optimizing prematurely for page count.

Legacy NeurIPS assets remain frozen for provenance, but journal displays should
be regenerated in the common journal style rather than reproduced merely
because they are byte-identical.

## Talk logic

The 10-minute story now follows:

`credit assignment -> backpropagation reference -> biological alternatives -> point-neuron limit -> general adjoint -> neuron identity -> ownership boundary -> candidate addresses -> conductance gain -> task alignment -> decisive subtree experiment`.

The talk says explicitly that the 58/60 validity-qualified comparison is neuron-to-tree
ownership, the measured-response result remains a biological boundary, and
the complete address factorial establishes a bandwidth-dependent resource
value while matched point/flat emulations prevent a unique material claim.

## Submission rule

Retain the current cautious title. Gate A is positive only at an intermediate
task-aligned budget and the same routed fields work in matched point/flat
implementations. The abstract should therefore foreground structured address
resources, the ownership bottleneck and electrotonic boundary rather than
claiming universal or uniquely dendritic superiority.
