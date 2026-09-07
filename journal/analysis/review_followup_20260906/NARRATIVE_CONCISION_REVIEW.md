# Read-only narrative concision review

Proposed replacements are ready to copy; `main.tex` was not changed. Counts use the publication audit's prose-word function and exclude equations and citations. The exact old/new strings are in `concision_replacements.json`.

| Replacement | Before | After | Saved |
|---|---:|---:|---:|
| Interaction Results: full prose before figure | 830 | 620 | 210 |
| Discussion: full prose | 675 | 429 | 246 |
| Anatomy opening | 73 | 45 | 28 |
| Anatomy common-mode design: two paragraphs | 163 | 101 | 62 |
| Anatomy cohort limitations | 95 | 64 | 31 |
| Anatomy labeling and cost | 94 | 51 | 43 |
| Branch-selection opening | 83 | 43 | 40 |
| Branch-selection equivalence after update equation | 59 | 43 | 16 |
| Branch-selection conclusion | 99 | 72 | 27 |
| Introduction conductance paragraph | 77 | 56 | 21 |
| Introduction final progression paragraph | 85 | 54 | 31 |
| Measured-response closing | 73 | 44 | 29 |

**Total reduction: 784 words.**

The interaction rewrite preserves every reported NMSE, confidence interval, capture value, cohort count, task definition and scope distinction. The Discussion keeps the task-to-credit spine, the conductance bridge, representability as prerequisite, the physical-depth accuracy/loss reversal, anatomy gain mechanism, empirical null and a concrete biological test. The explicit 20/20 initialization-selector failure remains untouched in the image Results.

## Replacements

### Interaction Results: full prose before figure (save 210 words)

```latex
Interactions across branches can change credit's sign and magnitude between examples. We compared targets with identical input-sensitivity spectra on the same representationally compatible tree, separating credit delivery from forward capacity.

Eight independent inputs $x_i$ were equally likely to be $-1$ or $+1$. Each branch computed $u=a+bu_L+cu_R+du_Lu_R$ with four trainable coefficients. This multi-affine unit is affine in either child while the other is fixed. Pairwise targets summed four disjoint products, $f_M=\frac12\sum_{(i,j)\in M}s_{ij}x_ix_j$; quartic targets combined complementary quartets, $f_A=\frac12(s_A\prod_{i\in A}x_i+s_{\bar A}\prod_{i\notin A}x_i)$. Here $M$ partitions inputs into pairs, $A$ is one quartet, $\bar A$ its complement, and signs $s$ were independently sampled. Within each seed, targets shared a compatible balanced tree, input assignment, initial weights and examples (Fig.~\ref{fig:prospective}A). Both have input-sensitivity second moment $\E[\nabla_xf\nabla_xf^{\mathsf T}]=I_8/4$, giving identical strengths of independent sensitivity directions.

Compatibility has an exact criterion. At each subtree boundary, arrange interaction coefficients into a matrix indexed by products of inputs inside and outside the subtree. Removing the constant inside row must leave rank at most one, because only one scalar leaves the branch. This condition is necessary and sufficient across all boundaries; discarded singular values bound representation error (Methods; Supplementary Section~S4). It constrains forward capacity, not the learning rule. Boolean examples and the structure census appear in Supplementary Figs.~S36--S37 and S43--S44.

All credit rules used the same exact local eligibilities and scalar output error. We compared exact paths with unit broadcast and a fixed six-site profile calibrated from each student's mean path derivatives over 256 unlabeled initialization examples. Three development seeds selected learning rates with equal budgets; twenty fresh paired seeds tested the frozen choices. A separate nested target summed products over prefixes of two, four, six and eight inputs on its own compatible tree.

After 1,024 Adam steps, exact/calibrated held-out normalized mean squared errors (NMSE) were 0.0317/0.0236 for pairwise targets, 0.1516/0.9813 for quartic targets and 0.0230/0.7690 for nested targets (Fig.~\ref{fig:prospective}B--D). NMSE divides squared prediction error by clean target variance. The predefined quartic-minus-pairwise difference in calibrated-minus-exact gaps was 0.838 (95\% paired seed-bootstrap interval, 0.719--0.931), positive in all twenty seeds; a common rate gave 0.843 (0.727--0.935; Fig.~\ref{fig:prospective}E). Some exact quartic fits retained appreciable error at this fixed budget. Equal label noise gave different NMSE floors because pairwise and quartic target variances were one and one-half.

The trained fields reveal a distinction missed by the input spectrum. Let $\bm q(x)$ contain the six nonsomatic derivatives of root output with respect to branch output; exact branch credit is scalar output error times $\bm q$. Allowing an oracle amplitude per example, the best fixed profile captured 99.7\% of exact-trained path-field energy for pairwise targets, but 46.1\% for quartic and 44.8\% for nested targets (Fig.~\ref{fig:prospective}F). Loss-error weighting, eligibility weighting, site-normalized spectra and replay of 600 earlier fits preserved this separation (Supplementary Section~S4). Uniform-profile capture did not: a nearly one-dimensional field can have a signed, nonuniform direction. Positive fixed-profile scaling largely cancels in Adam, explaining calibrated/sign-only agreement; it does not explain the slower pairwise learning with changing exact paths.

Thus input-matched targets developed different credit spectra and tolerance of fixed profiles. At fixed weights and inputs, however, path derivatives are target-independent: their spectrum describes learned states and coordinates, not a task-only initialization diagnostic. Failure of the tested profiles supports example-dependent spatial credit without establishing that six independent channels or exact gradients are necessary.

Additional controls delimit the result. Fixed-profile Adam learned a compatible positive-conductance teacher accurately; broadcast learned aligned XOR-of-AND, whereas parity favored exact credit under the tested optimizer (Supplementary Figs.~S42--S44). Noisy-query interaction estimation also yielded useful trees, retained as supporting structure-design evidence in Supplementary Figs.~S41--S42. Interaction structure therefore motivates specific credit comparisons rather than a universal monotonic resolution law.
```

### Discussion: full prose (save 246 words)

```latex
Useful credit resolution depends on the task and learner. Morphology supplies delivery patterns, teaching mechanisms supply their coefficients, and conductance regulates their gains. Image learning benefits mainly from preserving neuronal identity, including with random between-neuron feedback. Contextual conflict requires branch selection, whether supplied by feedback or local eligibility gating. Hierarchical distractors favor ancestry at a bandwidth predicted by the generator's coefficients. These results distinguish useful information from its physical carrier.

The interaction comparison shows why input rank alone is insufficient. Pairwise and quartic targets share input-sensitivity spectra and a compatible tree but develop different credit spectra and tolerance of fixed profiles, despite initial calibration and common-rate controls. Spatial direction matters alongside dimension: even a nearly one-dimensional field can be poorly captured by uniform broadcast. These are learned-state properties, not retrospective support for the failed initialization selector. Intermediate dictionaries, adaptive profiles and alternative dynamics remain plausible solutions.

Conductance trees connect this distinction to a dendritic computation. Simple credit learns some compatible teachers accurately, whereas opposing branch tuning under contextual inhibition benefits strongly from routed credit; three supplied spatial patterns with oracle coefficients suffice in the tested circuit. Forward compatibility remains essential: the scalar multi-affine cut criterion connects input grouping to hierarchical tensor descriptions and supports structure estimation from noisy examples \citep{hackbusch2009tensor,grasedyck2010hierarchical,cohen2016tensor}. Serial composition likewise helps remove distributed nuisance factors, but longer training reverses the LocalCA accuracy advantage of exact credit while preserving its lower cross-entropy. Task rank, physical depth and credit resolution therefore describe different constraints.

Reconstructed arbors provide ancestry capacity beyond a common broadcast, including in the disjoint 47-cell cohort; degree--depth surrogates explain much of that capacity. Focal shunting supplies a biophysical link by reweighting the baseline-weighted ancestry partition, with additional local changes from synaptic driving forces. Cable state determines their spatial selectivity. This concerns sensitivity transport, distinct from inhibition that computes errors or regulates forward gain and load \citep{rossbroich2023disinhibitory,galloni2026cellular,greedy2026celltype,safaai2026gainload}.

Measured ancestry--response similarity showed no preferential alignment in the small recorded cohort. The offline learning comparison chiefly tests preservation of an almost fixed transfer profile and offers limited evidence about endogenous route use. A direct experiment would compare plasticity below a focal inhibitory site with electrotonically matched off-route sites while matching somatic voltage, first-order current and outcome signal. Changing conductance state would test the predicted separation between shared and ancestry-selective effects \citep{francioni2026vectorized,cichon2015branch,bloss2016structured}.

The framework complements somatic prediction, segregated teaching, interneuron-mediated errors and burst plasticity \citep{urbanczik2014dendritic,guerguiev2017segregated,sacramento2018dendritic,payeur2021burst}, alongside cable-learning and engineered gating models \citep{bicknell2021synaptic,chavlis2025dendrites,sezener2021gated,iyer2022activedendrites,lv2025dendritic}. Spikes, temporal eligibility, recurrence and channel kinetics may change the available profiles and coefficient estimation \citep{bittner2017btsp,magee2020plasticity}. The organizing question remains which task-relevant distinctions the arbor and learning mechanism can jointly preserve.
```

### Anatomy opening (save 28 words)

```latex
MICrONS links reconstructed anatomy and connectivity to visual responses \citep{microns2025}. We used observed skeletons, soma positions and mapped contact locations and areas to model conductances, voltages and gradient fields, asking whether measured geometry supplies useful delivery patterns. This tests anatomical capacity rather than observed teaching signals.
```

### Anatomy common-mode design: two paragraphs (save 62 words)

```latex
The fields were gradient changes induced by small focal shunts in reciprocal cables, independently of dictionary selection. In the original eight cells, broadcast captured 31.4\% of field energy, near the 32.6\% rank-one ceiling but leaving most energy outside it. Subtree routes lacked a constant component whereas depth bins could broadcast, confounding common and spatial capacity. We therefore gave every family one budgeted broadcast column and $K-1$ spatial profiles, measuring total and residual-field capture and recording actual rank and nonzero count. The protocol was fixed before extension to 47 disjoint cells from the same mouse and a reconstructed second mouse (Fig.~\ref{fig:topology}).
```

### Anatomy cohort limitations (save 31 words)

```latex
The original eight cells retained the same ordering (Fig.~\ref{fig:topology}D). In the Pinky second mouse, eight of ten qualifying cells supported the eight-column comparison. Their 9--13 input-bearing sites put ancestry near the residual-capture ceiling, so lower-budget comparisons and exclusions remain separate from the larger arbors. Earlier route-generated positive controls test imposed alignment, distinct from these independent fields (Supplementary Figs.~S10, S20 and S27).
```

### Anatomy labeling and cost (save 43 words)

```latex
Direct E/I labels covered 4.71\% of disjoint-cohort inputs; the original cohort's 73.0\% direct-versus-proxy agreement concerned jointly labeled contacts. Nonzero site-by-route coefficients are a delivery-cost proxy, not feedback synapses or metabolic expenditure (Fig.~\ref{fig:topology}E). Geometry, labeling and compression sensitivities are in Methods and Supplementary Figs.~S12, S25 and S33.
```

### Branch-selection opening (save 40 words)

```latex
To isolate branch-selective information, we used one logistic output with $B\in\{2,4,8\}$ parallel branch weight vectors and no internal forward compartments or conductance nonlinearity (Fig.~\ref{fig:branchconflict}A). Context selected the branch determining the output, while every branch received an image and formed an ungated local eligibility.
```

### Branch-selection equivalence after update equation (save 16 words)

```latex
The indicator $\mathbf{1}[\cdot]$ is one when its condition holds, zero otherwise; $1/B$ matches scale without supplying branch information. A feedback selector and a branch-local eligibility gate give the same update (Fig.~\ref{fig:branchconflict}B), testing access to selection rather than a need for independently computed external errors.
```

### Branch-selection conclusion (save 27 words)

```latex
The shared rule reached chance at lower conflict as branch count increased, consistent with the predicted shift (Fig.~\ref{fig:branchconflict}F); five two-branch seeds never crossed in the sampled range. Theoretical, mean-curve and individual-seed crossings are distinct. Gradient alignment also changed during learning (Methods; Supplementary Fig.~S29D--F). Credit reaching inactive branches can therefore harm learning; dendrites offer one possible selection site. A separate two-stream task tests learning and interference effects (Supplementary Fig.~S19).
```

### Introduction conductance paragraph (save 21 words)

```latex
Conductance changes the gains of anatomical delivery routes. Excitatory and inhibitory contacts open nonnegative conductances whose reversal potentials determine current; inhibition can lower input resistance and attenuate other inputs even when its own current is small \citep{koch1983nonlinear,holt1997shunting,chance2002gain,gidon2012inhibition,lovettbarron2012regulation}. Channel-amplitude changes rescale a profile, whereas altered cable transport can reshape how local voltage affects somatic output and task error.
```

### Introduction final progression paragraph (save 31 words)

```latex
We compare tasks requiring neuronal identity, branch selection, ancestry grouping or example-dependent credit across interacting branches. Morphology supplies candidate patterns and conductance regulates their expression. Forward compatibility anchors the learning comparisons; measured responses test the anatomical proposal. The question is which distinctions useful credit must preserve, rather than whether finer feedback is always better.
```

### Measured-response closing (save 29 words)

```latex
Input-removal and noise controls test dependence on recorded partners (Supplementary Fig.~S34); imposed-alignment fields establish learning capacity when targets lie in the anatomical span (Supplementary Fig.~S5). Neither tests endogenous teaching. The empirical ancestry-similarity test and modeled transfer diagnostic therefore bound distinct biological claims.
```
