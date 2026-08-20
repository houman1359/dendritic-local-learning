# Eight-reviewer revision matrix

Last updated: 19 August 2026

This is the implementation ledger for the 12 August eight-reviewer report.
It distinguishes completed repairs, active analyses, new prospective evidence,
and author/external actions that cannot be completed from the repository.

## Editorial decision and central frame

- **Venue:** Nature Communications; Nature Computational Science remains a
  methods-forward fallback.
- **Advance:** the stochastic credit-operator signal--noise phase theory and
  its quantitative validation. Anatomy, conductance and depth experiments map
  the theory's boundary rather than support a categorical dendritic advantage.
- **Submission form:** the current priority is a scientifically complete
  seven-section, eight-figure manuscript. The authors set an eventual working
  target of at most approximately 8,000 narrative words; evidence and negative
  results will not be removed merely to meet the shorter journal guidance.

## Submission blockers

| Finding | State | Required action |
|---|---|---|
| Broken/dead bibliography keys and missing load-bearing literature | Completed | The bibliography now contains 56 cited records, including all load-bearing works requested in the review. `scripts/audit_citations.py` reports zero undefined and zero unused keys. |
| Placeholder Code Availability | External action | The local release is prepared; an author must publish it and mint the Zenodo DOI. The manuscript must then name URL, DOI and MIT license. |
| Cover letter and NeurIPS dual-track status | Local revision completed; author action remains | The cover letter is now a theory-first significance letter and the related-work statement treats the journal Article as the canonical standalone account. Authors must insert the final NeurIPS status, satisfy exclusivity, and approve the sibling-manuscript disclosures before submission. |
| Length, display count and stale checklist | Figure inventory completed; length remains advisory | Results now follow eight scientific subsections and use eight single-file numbered figures without continued floats. The checklist records eight main figures and Supplementary Figs. S1--S24. Narrative length remains deliberately advisory during scientific revision, as requested by the authors. |

## Scientific and inferential repairs

| Finding | State | Implementation |
|---|---|---|
| Laminar shortfall was overstated as leading-eigenspace energy | Completed | Supplement gives the exact leading-minus-trailing decomposition, the leading-energy upper bound and the top-$K$ containment equality condition; executable tests remain in `tests/test_laminar_optimality.py`. |
| Inverse-operator norm used to explain spatial selectivity | Completed | Discussion attributes selectivity to the focal Green's-function column and length constant, restricting the inverse norm to global perturbation magnitude. |
| Matched-depth Wilcoxon pooled 200 non-independent rows | Completed | Four task-depth contrasts are averaged within each of 50 seeds before bootstrap and Wilcoxon inference; corrected result: 0.0957, 95% CI 0.0782--0.1134, 48/50 seeds, two-sided $P=3.00\times10^{-13}$. |
| $K$-budget multiplicity | Completed | The four-budget scan is BH corrected; the $K=4$ result survives ($P_{\mathrm{raw}}=0.00482$, $P_{\mathrm{BH}}=0.00643$). |
| Seed-count rationale and inferential-unit audit | Completed | Methods now gives the role-based seed rationale; `source_data/inferential_units.csv` maps every headline endpoint to its independent and nested units. |
| $D=H$ presented as discovery | Completed in text | It is explicitly a controlled validation because the generator places signal through $H$ and finer-scale noise beyond it. |

## Highest-leverage new experiments

The H3 controls use the already frozen hierarchical gain--load task, operating
point and paired seeds 10200--10209. The second-hierarchy arm uses ten fresh
paired seeds 10300--10309. No task or optimizer retuning is allowed.

1. **Resource-identical serial-tree versus all-active grouped-point/star
   control.** The star retains the same branch modules, masks, trainable
   parameters, active contacts and initial state dictionary, but removes
   serial child-to-parent composition. D1 is an equality control; D2--D3 test
   whether the 31-point result requires serial divisive composition.
2. **Active-parameter-matched unstructured point MLP.** This answers the weaker
   capacity-matched point-versus-tree question without granting the point model
   the tree's grouped state.
3. **Autograd soma-broadcast control.** Standard eligibility derivatives are
   retained, but the loss derivative at every non-somatic compartment is
   replaced by its owning soma derivative. Comparing it with full BP,
   shared-soma LocalCA and exact-path LocalCA separates coordinate restriction,
   eligibility approximation and path transport.

Positive architecture claims require paired intervals excluding zero and at
least 8/10 seed signs. Dendrite specificity additionally requires an
aligned-minus-reversed interaction. A null star contrast means the result is
about grouped divisive computation rather than serial dendritic depth and will
be reported that way.

**Completed outcome (200 new fits).** The serial tree and exact-resource star
are equal at D1; the serial tree leads by 6.54 points at D2 and 30.81 points at
D3 under alignment, with a 31.24-point D3 alignment interaction. Active- and
total-parameter-matched point MLPs beat serial D3 by 6.79 and 7.20 points,
respectively, ruling out an unconstrained expressivity advantage. Full BP loses
only 0.64 points when restricted to soma-broadcast coordinates under a common
BP optimizer. A transparently post-outcome optimizer amendment shows no
detectable eligibility residual between matched broadcast and shared LocalCA
($-0.06$ points), while optimizer groups account for 13.47 points and path
specificity for 10.96 points. All numerical and resource gates passed.

4. **Physical-depth alignment dose response (complete; 90 new fits).** The same
   endpoint generator is interpolated at
   $\alpha\in\{0.25,0.50,0.75\}$ for D1--D3 and ten paired seeds, with the
   existing $\alpha=0,1$ endpoint cohorts retained unchanged. The primary test
   is the within-seed slope of the D3-minus-D1 depth effect with alignment.
   This is a prospective interpolation, not an independent task hierarchy or
   a second biological replication. All artifact and resource gates passed.
   The depth benefit was 0.02, 0.21, 0.82, 3.13 and 30.86 points across the
   five alignment levels; the positive mean slope (25.84 points per unit
   alignment; 10/10 seeds) therefore summarizes a strongly nonlinear curve
   with a steep transition between $\alpha=0.75$ and $1$.

5. **Second-dataset feedback ladder (complete; 60 new fits).** Fashion-MNIST
   was substituted without retuning the regular-tree architecture or training
   protocol. Neuron-indexed feedback improved over scalar fallback by 4.46
   points in shunting trees and 3.78 points in additive trees, with 10/10
   positive paired seeds in both; exact path transport added no reliable gain
   beyond neuron identity. This supports the bandwidth pillar, not topology.

6. **Literal grouped-point and second hierarchy (complete; 220 new fits).**
   A direct-to-soma grouped-point implementation now replaces every serial
   child aggregator with a parameter-matched projection while preserving local
   branch modules, masks and initialized conductances. It remains flat across
   depth on H3, whereas serial-minus-point is +30.87 points at aligned D3 and
   -0.44 after reversal (interaction +31.31). A fresh H2 hierarchy independently
   gives +30.84 points for aligned serial BP and -1.56 after reversal
   (interaction +32.40); grouped-point BP stays flat. Shared and path LocalCA
   retain +21.20 and +31.38 points, with +22.93 and +33.36 placement
   interactions. All ten positive primary effects pass the frozen interval
   and 8/10-sign gates; all 220 fits pass resource and artifact audits. This
   closes the reviewer's literal grouped-point control and second-hierarchy
   requests within the calibrated task family, not across an independent task.

## Claim and writing repairs

- Pair 14.2-fold model-matched capture per wire with the approximately
  2.7-fold density-matched anatomy-specific factor everywhere.
- Qualify the 31-point contrast as a calibrated, mechanism-matched existence
  proof and pair it with the fixed-budget negative depth result.
- State the $R_m=15{,}000$ focal-shunting null at first mechanistic mention;
  reserve "permissive regime" for experimentally grounded high-conductance
  conditions.
- Promote the MICrONS null and fixed-budget depth cost to headline findings.
- Consolidate audit vocabulary and provenance in Methods; use one term each
  for neuron identity, ownership, ancestry route, scalar feedback and capture.
- Add a boxed theory scope statement: one-step geometry predicts within-family
  progress and large contrasts; trajectory-accrued between-family advantages
  are outside its guarantee.

## Figure programme

The eight-figure order is now: framework; coordinates/ownership; subtree
address; credit-phase theory; physical depth and point controls; MICrONS
topology plus wiring-normalized capture; focal shunting plus its active-channel
extension; and measured-response nulls, complete-tree null, imposed-alignment
rescue and the alignment-by-bandwidth phase plane. The six-animal signed-
coordinate analysis is Supplementary Fig. S17 because it validates neuron
identity but does not test within-tree routing. QC and provenance panels remain
available in SI. Continued displays preserve the complete panel set while
keeping eight numbered main figures.

The workshop deck was used as a visual-logic audit. Its reusable
coordinate--address--gain hierarchy now sets the vocabulary of Figure 1; its
wiring-economics panel motivated promotion of capture per wire into Figure 6;
and its closing phase plane is now Figure 8O. Slide-only categorical wording
was not imported into the manuscript.

## Author/external dependencies

- Public repository URL and versioned Zenodo DOI.
- NeurIPS decision and final exclusivity/reuse wording.
- Final author approval for related-work disclosures, declarations and cover
  letter.
- A second-animal connectomic cohort remains desirable but is not represented
  as complete or promised for the initial submission.
