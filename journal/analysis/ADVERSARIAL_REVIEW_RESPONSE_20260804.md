# Disposition of the adversarial journal review

> **Historical record, superseded for inference on 10 August 2026.** The
> outcome-independent input-validity audit excludes every signed synthetic-
> noise/positive-conductance shunting run. Use
> `source_data/prospective_input_validity/` and `analysis/EVIDENCE_LEDGER.md`
> for publication claims. In particular, the historical 78/80 routing count
> becomes 58/60 among valid pairs, and the inhibitory-dose family is excluded.

This record maps every concern in the eight-dimension review to the current
analysis, display item and claim boundary. It distinguishes corrections that
changed the scientific interpretation from additions that strengthen an
existing result.

## Headline blockers

| Review concern | Action and result | Current presentation |
|---|---|---|
| The original MICrONS route-capacity target was generated from the same ancestry kernel used as the morphology dictionary. | Retained that calculation only as an explicitly model-matched compression diagnostic. Added an independent target: the finite-difference focal-shunt response operator of the exact reciprocal cable model. At eight channels, morphology capture was 0.458, versus 0.363 for random paths and 0.249 for row-shuffled paths; morphology exceeded both in 8/8 cells. The differences from depth bins (0.014; 5/8; $p=0.844$) and degree/depth-matched surrogate trees (0.035; 6/8; $p=0.078$) were not reliable. | Main Fig. 5i,j; Results and Methods; Supplementary Table “Independent-generator route capacity.” The paper now claims useful sparse support beyond random or reassigned routes, not evidence for fine topology beyond depth and degree. |
| The task-derived figure selectively omitted feedback families and one channel budget. | Rebuilt the figure and text to report all seven feedback families and all four budgets. At four channels, normalized held-out error ranked exact 0.778, dense oracle 0.781, depth bins 0.791, scalar 0.801, random 0.834, morphology 0.837 and ancestry shuffle 0.842. Within-target capture predicted error at one and eight channels, but the morphology-minus-shuffle learning contrast was not established at any budget. | Main Fig. 9d–h; full Results paragraph; complete Source Data. This is presented as the topology–task alignment boundary. |
| Focal-shunt localization depended on the normalized cable regime. | Added physical-unit cable calculations using reconstructed radii and lengths and varied $R_a$ and $R_m$. At $R_a=150\,\Omega\,\mathrm{cm}$ and $R_m=15{,}000\,\Omega\,\mathrm{cm}^2$, the original-cohort contrast was negligible and the v661 effect was small. The effect recovered as membrane conductance increased, reaching 0.0674 in the original cohort at $R_m=300\,\Omega\,\mathrm{cm}^2$. | Main Fig. 7i; Results; full parameter table in the supplement. The abstract and Discussion now state electrotonic-regime dependence rather than a universal anatomical effect. |
| Neuron-indexed feedback changed both bandwidth and the destination of each signal. | Added a separately frozen 160-run control that keeps all neuron-indexed coordinate values and bandwidth fixed but applies a fixed derangement of the neuron-to-tree map. Correct addressing improved 78/80 paired seeds, by 0.15–0.72 percentage points across conditions. This is smaller than the 4.92–14.16 point scalar-to-neuron-indexed gain. | Main Fig. 3g,h. The paper now attributes most of the original gain to coordinate bandwidth and a smaller reproducible component to correct routing. |

## Major issues

| Review concern | Resolution |
|---|---|
| The alignment experiment was constructed rather than empirical evidence about the recorded cells. | Moved to Supplementary Fig. S2 and labeled throughout as a constructed, oracle-coefficient sufficiency test. |
| Inhibitory co-targeting ignored repeated contacts from the same axon and local 3D geometry. | Added aggregation over 92 presynaptic axons and joint 3D/path matching. Tree-distance clustering survived axon aggregation ($p=0.0008$); shared-path and descendant-overlap endpoints did not. No endpoint survived joint 3D/path matching. The biological claim is now limited to coarse co-targeting and inhibitory-class spatial scale. |
| The first-order-current-matched additive perturbation is not an exact factor-matched control. | Promoted an exact factor-freeze comparison. Full shunting and the driving-force-only state share the same post-shunt voltage and driving force, so their difference isolates adjoint transport. Localization increased by 0.0786, positive in 8/8 cells. The additive perturbation remains a descriptive intervention, not the causal reference. |
| Directly typed and all-input route capacities were juxtaposed despite different composition and rank. | The directly typed result is labeled a separate sensitivity analysis and is not compared numerically with the all-input estimate. |
| Inhibitory counts and identity/proofreading language were inaccurate. | Replaced pre-mapping counts with 114,775 input contacts and 2,637 inhibitory contacts from 114 cells. The manuscript states that the stable-ID join does not imply unchanged segmentation and that the packaged table lacks a per-target proofreading-status field. |
| A sign-flip value was labeled as a sign test. | Corrected the test name and audited all inferential labels. |
| Core dendritic and MICrONS precedents were missing. | Added Rall (1962), Gidon and Segev (2012), and Ding et al. (2025), with the latter used to distinguish within-tree organization from broader like-to-like connectivity. |
| Figure 1 panel labels were clipped. | Rebuilt and visually inspected the complete main and supplementary figure set at manuscript scale. |

## Additional evidence added after the review

The revised manuscript also contains two prospective results that sharpen the
central mechanism without inflating its scope.

1. A 400-run inhibitory-dose factorial shows a 22.11 percentage-point increase
   in the shunting-minus-additive contrast from zero to 40 inhibitory contacts
   under scalar feedback (95% paired-seed interval 19.94–24.44; 10/10 pairs).
   The corresponding interactions are near zero or negative with
   neuron-indexed, exact and backpropagated credit. Shunting therefore helps in
   a severe low-information regime, not in general.
2. A 320-run fixed spatial-connectivity test improves learning under local
   rules and backpropagation alike. Its branches also cover 336 unique inputs,
   versus 276.2 for independently sampled random branches. It is therefore
   reported as a useful forward-connectivity prior, not evidence for routed
   credit.

## Current scientific position

The strongest defensible chain is now:

1. exact conductance-tree gradients separate into synapse-local eligibility
   and a transported compartment error;
2. neuronal identity is the main feedback bottleneck in the tested regular
   trees, and correct tree assignment adds a smaller bandwidth-matched effect;
3. inhibition creates an exact path-gain control and selectively improves
   learning when feedback is severely restricted and the operating point is
   inhibition sensitive;
4. reconstructed ancestry provides sparse candidate routes, but the evidence
   specific to fine topology is bounded by depth/degree controls;
5. inhibitory anatomy provides access at coarse spatial and cell-class scales,
   not a demonstrated endogenous teaching map;
6. measured visual responses do not establish alignment of those routes with
   the tested task, while the constructed control shows why alignment would
   matter if it were present.

This supports a Nature Communications Article. It does not presently support a
claim of observed in-vivo branch-level credit routing, which would require an
independent multi-animal, branch-resolved learning dataset or a causal
experiment.
