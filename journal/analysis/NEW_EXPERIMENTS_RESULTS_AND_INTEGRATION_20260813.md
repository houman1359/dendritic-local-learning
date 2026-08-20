# New experiments: results, interpretation and manuscript integration

Date: 13 August 2026

This ledger records the reviewer-motivated experiments added during the final
scientific revision. It separates what was run, what it showed, where it is
used in the article and what remains untested. Percentage-point effects are
paired by seed.

## Completed recent cohorts

| Experiment | Runs | Principal result | Scientific interpretation | Article integration |
|---|---:|---|---|---|
| Resource-identical serial tree versus all-active grouped star | 60 primary fits within the 200-fit control programme | At D1 the models were equal (-0.005 pp). The aligned serial tree exceeded the star by 6.54 pp at D2 and 30.81 pp at D3; the D3 alignment interaction was 31.24 pp, all 10/10 positive. Under reversed placement the serial contrast was -0.43 pp. | The aligned depth effect requires ordered child-to-parent divisive composition; it is not explained by the number of branch modules, contacts, parameters or persistent states. | Main Results, Figure 5G-I; Supplementary Methods and Results, "Same-task point, nonserial and credit-coordinate controls"; Source Data `point_dendrite_credit_controls/`. |
| Active- and total-parameter-matched point MLPs | 20 fits within the same programme | Accuracies were 0.9861 and 0.9902. The point models exceeded serial D3 by 6.79 and 7.20 pp, respectively, in 10/10 seeds. | The dendritic tree has no absolute expressivity or benchmark advantage here. Its defensible role is a task-aligned structured inductive and wiring resource. | Main Results, Figure 5L; abstract and Discussion; Supplementary control section; Source Data `point_dendrite_credit_controls/`. |
| Soma-broadcast autograd and optimizer-matched BP--LocalCA ladder | 120 primary/amended fits within the same programme | Under the BP optimizer, full BP exceeded soma broadcast by only 0.64 pp at D3. With the LocalCA optimizer, soma-broadcast autograd and shared LocalCA differed by -0.06 pp (interval crosses zero). Optimizer choice accounted for 13.47 pp; exact path LocalCA recovered 10.96 pp over shared LocalCA; full BP remained 3.09 pp above path LocalCA. | The tested local eligibility expression is not the main bottleneck after matching coordinate and optimizer. Most of the raw BP--local separation is attributable to optimizer regime and within-tree teaching-coordinate specificity. | Main Results, Figure 5J-K; Discussion; Supplementary control section; Source Data `point_dendrite_credit_controls/`. |
| Physical-depth alignment interpolation | 90 new fits plus the frozen endpoint cohorts | D3-minus-D1 was 0.02, 0.21, 0.82, 3.13 and 30.86 pp at alignment alpha 0, 0.25, 0.50, 0.75 and 1.00. All ten within-seed slopes were positive, but 27.73 pp of the change occurred in the final 0.75-to-1 step. | The depth effect is a steep alignment-dependent transition, not a generic linear benefit of increasingly aligned dendrites. This is an interpolation of the existing task, not an independent hierarchy replication. | Main Results, Figure 5M-O; Supplementary Methods and Results, "Alignment-dose interpolation"; Source Data `physical_alignment_dose/`. |
| Literal grouped-point H3 control | 60 new fits | The aligned serial tree exceeded the literal grouped-point/direct-to-soma model by 30.87 pp at D3 (30.48--31.31; 10/10 seeds), whereas the contrast was -0.44 pp after sensor-tier reversal. The architecture-by-placement interaction was +31.31 pp. Grouped-point D3-minus-D1 was -0.008 pp. | The H3 result requires ordered divisive composition; it is not an artifact of the earlier grouped-star implementation. The grouped star and literal grouped point differed by only +0.06 pp at aligned D3 and did not pass the frozen positive gate. | Abstract; Main Results and Figure 5P-R; Supplementary Methods and Results, "Literal grouped-point and independent H2 hierarchy tests"; Source Data `remaining_physical_experiments/`. |
| Independent H2 hierarchy replication | 160 new fits | Serial BP gained +30.84 pp from D1 to D2 when aligned and lost -1.56 pp after reversal (interaction +32.40 pp). Grouped-point BP stayed flat (+0.023 and -0.021 pp; interaction +0.044 pp). Shared LocalCA retained +21.20 pp and path LocalCA +31.38 pp; their placement interactions were +22.93 and +33.36 pp. Every positive primary contrast was positive in 10/10 seeds and had exact sign-flip P=0.001953. | The serial crossover replicates at a second hierarchy depth and survives local credit. Path-resolved transport nearly recovers BP; a neuron-shared coordinate retains a smaller but substantial effect. This remains a replication within the same calibrated synthetic task family, not evidence of prevalence on natural tasks. | Abstract; Main Results and Figure 5S-U; Discussion; Supplementary Methods and Results; Source Data `remaining_physical_experiments/`. |
| Frozen H4 physical-depth factorial | 360 intended fits, including 90 construction-only D3 replacements | Aligned serial BP was +25.63 pp at D4 versus D1 but -1.46 pp at D4 versus D3, falsifying one-to-one depth tracking. Grouped point stayed flat; the D4 architecture-by-placement interaction was +25.91 pp. Shared LocalCA gained +5.27 pp from D3 to D4; path LocalCA gained +0.60 pp but failed the 8/10 positive-pair gate. | Task-aligned serial composition remains valuable relative to shallow and point controls, but the benefit saturates. Added depth is not itself a monotone computational advantage, and restricted-credit optimization can move differently from exact BP. | Main Results and Figure 6A-D; Discussion; Supplementary H4 section; Source Data `physical_depth_h4_factorial/`. |
| Immutable-source H2/H3 replication | 430 same-seed reruns | All validity gates passed. Clean-source H2/H3 aligned BP gains were +30.84/+30.86 pp and architecture-by-placement interactions were +32.43/+31.27 pp. Median absolute historical change was 0.030 pp; every load-bearing direction and claim gate was retained. | The earlier H2/H3 conclusions do not depend on the historical dirty-diff execution environment. This is source concordance, not independent task or biological replication. | Main Results; Supplementary Fig. S26 and methods; Source Data `physical_depth_clean_source_replication/`. |
| Fashion-MNIST feedback ladder | 60 fits | Neuron-indexed feedback exceeded scalar feedback by 4.46 pp in shunting trees and 3.78 pp in additive trees, 10/10 positive in both. Exact path transport added -0.11 and +0.13 pp beyond neuron identity, neither a reliable improvement. | The large scalar-to-neuronal-coordinate bandwidth effect generalizes beyond MNIST. Fine within-tree transport is not automatically useful on a standard image task. | Main Results, Figure 2P-Q; Supplementary artificial-tree methods; Source Data `fashion_feedback_ladder/`. |
| Adaptive local conductance reliability | 1,050 outcomes, 50 fresh paired seeds | At maximum heterogeneity, adaptive local shunts lowered final loss by 0.01622 versus adaptive global and 0.00553 versus shuffled placement, but were worse than no shunt by 0.00425 and worse than the fixed initial oracle by 0.00345. The independent point gate matched within $4.00\times10^{-15}$. | Local noisy credit observations recover relative branch ordering, not a sustained advantage over unattenuated learning. The operation remains point-emulable and uses paired probes plus an exact state clamp. | Main credit-phase Results; Supplementary Fig. S24, methods and Table; Source Data `adaptive_conductance_reliability/`. |
| Irregular-tree wavelet scale analysis | 47 primary plus eight secondary reconstructed cells; 256 nested column permutations per cell | In the 47-cell cohort, coarse route energy was enriched 2.351-fold over isotropic coordinate noise (47/47), but actual-minus-permuted coarse energy was only 0.0345 (interval -0.0025--0.0713; 31/47; $P=0.078$). The original eight-cell excess was positive. | Real unbalanced trees provide coarse multiscale coordinates, but the fine-anatomy-specific excess seen in the pilot does not replicate. This is structural capacity, not measured credit use. | Main MICrONS Results; Supplementary Fig. S25 and Methods; Source Data `irregular_tree_wavelets/`. |

All 1,360 recent trained-network fits passed the stated finite-output, seed,
checkpoint and resource audits. The point/dendrite programme contains 200 fits in total: 140
in the primary extension and 60 in the optimizer-matched amendment. The final
reviewer-extension programme adds 220 independently audited fits: 60 literal
H3 grouped-point controls and 160 H2 hierarchy conditions.

## Other major experimental additions already used in the article

| Experiment | Scale | Result and use |
|---|---:|---|
| Trained subtree-address factorial | 2,700 fits, 20 paired seeds | Correct ancestry beat the best matched non-anatomical control only at K=4 (+1.27 pp, BH-adjusted P=0.0064) and lost at the smaller budgets; it beat the depth-matched rewired tree at K=2 and K=4. Used in Figure 3 and the theory-validation analysis. |
| Credit-operator phase tests | Four fixed-state screens, 50 seeds | The utility bound predicts route/depth outcomes, including the matched-depth advantage (0.0957 loss units, 48/50 seeds) and trained-outcome rank correlations of 0.937 and 0.916. Used in Figure 4 and the abstract. |
| Nonlinear physical-depth endpoints | Confirmatory ten-seed cohorts | Aligned shunting BP gained 30.86 pp from D1 to D3; independent, shuffled and reversed controls removed the gain, while raw-additive depth worsened. LocalCA retained 16.95 pp with a shared coordinate and 27.90 pp with exact path transport. Used in Figure 5A-F. |
| State-matched positive-conductance reliability | 50 seeds across shunt levels and controls | Reliability-aligned shunts improved the immediate credit step and final loss over the best global shunt, but not final loss over unshunted noisy learning. The explicit point gate matched the operation. Used as a bounded conductance result in the main theory/Discussion and Supplementary Figure S13. |
| Fixed-budget depth control | 320 runs | In the valid additive cells, D4 underperformed D1 in all 40 paired contrasts; exact transport and BP each lost 1.37 pp. Used as the negative depth arm in Figure 2L, the abstract and Discussion. |
| Active-conductance focal-shunting extension | 512 accepted steady states | Descendant-enriched attenuation persisted under exact local linearization without modeled gradient sign reversal. Used in Figure 8 and the conductance boundary argument. |
| Complete-tree measured-response learning | 520 fits over 13 eligible scans | No reliable morphology-specific learning advantage appeared for the tested visual-response objective. Used as the closing null in Figure 9 and the abstract. |

## Remaining experimental boundaries

1. The trained conductance-network H4 crossover is complete and falsifies
   one-to-one depth tracking: D4 remains useful relative to D1 but falls below
   D3 under exact BP. The most informative remaining extension is a second
   task family or a route-budget/alignment crossover, not simply deeper trees.
2. Standard image tasks already provide an informative negative boundary:
   fixed-budget depth reduced MNIST accuracy, and exact within-tree transport
   added no reliable benefit beyond neuron identity on Fashion-MNIST. What is
   still absent is a naturalistic task with an independently defined latent
   hierarchy and learned, rather than supplied, sensor placement. That would
   require a new discrete or differentiable placement method, not a small
   control within the current implementation.
3. An independent second-animal connectomic replication remains unavailable
   in the current public structural cohort. The primary 47-cell cohort is
   disjoint from the original eight cells but belongs to the same MICRONS
   animal; the manuscript states that limitation explicitly.
4. Fully autonomous kinetic learning of route gain or inhibitory placement
   remains open. The adaptive local reliability experiment removes oracle
   signal/noise energies, but still grants paired teaching probes, the
   optimizer step fraction and an exact state clamp. Removing those aids would
   require specifying and validating a new biological learning mechanism.

The two highest-leverage reviewer controls---the literal grouped-point H3
emulation and a fresh H2 hierarchy with BP and LocalCA---are therefore no
longer pending. The H4 extension now supplies the broader trained $D\times H$
test. Beyond it, the next genuinely distinct study would need an externally
specified task and learned sensor placement rather than another isolated
endpoint in the same calibrated family.

## Overall conclusion from the additions

The new experiments do not support the broad statement that dendrites learn
better than point neurons. They support the more precise result that a
tree-structured, serial divisive computation can be a useful inductive and
credit-addressing resource when task factors align with its hierarchy.
Neuron identity supplies the dominant feedback bandwidth; within-tree address
then matters on tasks requiring different updates within one arbor. Flexible
point networks can reproduce or exceed the computation, but doing so requires
explicit grouping/gating or unconstrained capacity. This conditional boundary
is the empirical counterpart of the credit-operator phase theory.
