# Deep literature, terminology, novelty and presentation audit

## Manuscript

*When dendritic structure helps local credit assignment*  
Canonical source: `journal/main.tex`; compiled manuscript: `journal/main.pdf`  
Audience: authors preparing a Nature Communications submission  
Research date: 23 August 2026

## Scope and assumptions

This audit assesses the current main manuscript, its compiled figures and its
61-entry bibliography against primary literature available through 23 August
2026 and current Nature Communications instructions. It focuses on scientific
positioning, terminology, novelty and professional presentation. It does not
re-run the experiments, validate every numerical result, re-review the full
Supplementary Information, or constitute journal peer review.

The intended central claim is interpreted as: dendritic trees are a
conditional, bandwidth-limited substrate for routing already available
neuron-specific teaching coordinates; their value depends on task alignment,
noise, address resolution and conductance state. It is not interpreted as a
claim that dendrites uniquely implement gradients or generally outperform
point models or backpropagation.

## Executive answer

The manuscript is scientifically coherent, unusually candid about nulls and
controls, and broadly consistent with the literature. Its strongest defensible
novelty is the **restricted credit-routing framework and its quantitative
boundary map**: a complete arbor is treated as a structured dictionary for a
compartment-level error field, and a signal–noise utility predicts when added
route structure helps, ties or hurts. The intermediate-bandwidth result,
matched non-anatomical controls, point emulations, cross-task depth boundary,
two-animal modeled-capacity direction and measured-response null make this a
substantive contribution.

The paper should not, however, be presented as the first derivation of
dendritic gradients, the first use of dendrites for credit assignment, the
discovery of neuron-specific teaching signals, or the discovery that shunting
controls gain. Those components have strong precedents. The current text mostly
respects that distinction, but four passages still blur it: the full-arbor gap
in the Introduction, the phrase “phase theory,” the abstract’s “endogenous task
credit,” and the framing of the Francioni reanalysis as an independent test.

The compiled figures are visually consistent and substantially professional.
The editorial defects identified in the initial audit have now been repaired:
the main narrative is about 7,855 mechanically counted words; every legend is
below 350 words; Figure 9 uses a clearly defined paired standardized effect;
its negative boundary is cohort-specific; the figure README is synchronized;
and the duplicate limitations sentence is removed. Code is available to
editors and reviewers through the prepared archive. A public repository and
versioned archival DOI, and the final disposition of the related NeurIPS
submission, remain author-facing actions before journal submission.

## Novelty audit

### Strong and defensible novelty

1. **Complete-arbor credit routing as a restricted representation problem.**
   Prior dendritic credit models usually send feedback to an apical/basal pair
   or another small set of named compartments. The manuscript instead asks
   which compartment-error fields a complete tree can express under a feedback
   budget. Segregated-dendrite, dendritic-microcircuit, burst and target-
   propagation models establish the broader field, but do not supply this
   alignment-by-bandwidth boundary
   ([Guerguiev et al. 2017](https://elifesciences.org/articles/22901),
   [Sacramento et al. 2018](https://papers.nips.cc/paper_files/paper/2018/hash/1dc3a89d0d440ba31729b0ba74b93a33-Abstract.html),
   [Payeur et al. 2021](https://www.nature.com/articles/s41593-021-00857-x),
   [Galloni et al. 2026](https://www.sciencedirect.com/science/article/pii/S2211124726002378)).

2. **A unified restricted-routing utility tied to trained outcomes.** The
   smoothness bound itself is elementary, but its use to combine retained
   task signal, representation error, coefficient error, finite-step gain and
   admitted noise into one dendritic-routing prediction is useful. The
   quantitative validation against 540 routed conditions and the explanation
   of nulls are more novel than the inequality alone.

3. **The intermediate-bandwidth and task-alignment boundary.** The result that
   nested subtree routes lose at low rank, win modestly at an interior rank and
   tie at full rank is a specific, falsifiable result. Feedback alignment has
   long established that useful feedback requires a positive relationship with
   the target gradient, but not this complete-arbor bandwidth allocation
   problem ([Lillicrap et al. 2016](https://www.nature.com/articles/ncomms13276)).

4. **A controlled boundary map rather than a one-sided success story.** The
   standard-calibration shunting null, fixed-resource depth costs, point
   equivalences, task-mismatch reversals and measured-response null are not
   weaknesses under the conditional framing. They are the empirical content
   of the theory.

5. **Separation of three meanings of depth.** Distinguishing route resolution,
   forward nonlinear composition and transport cost is conceptually helpful.
   The physical-depth experiments are best described as calibrated existence
   proofs and boundary tests, not a generic advantage of deeper dendrites.

### Established components that must not carry the novelty claim

1. **Exact differentiation and dendritic credit sensitivity.** Fixed-point
   adjoints descend from recurrent backpropagation
   ([Pineda 1987](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.2229)).
   Schiess and colleagues derived a supervised dendritic plasticity rule with
   error propagation inside active dendrites
   ([Schiess et al. 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004638)).
   Most importantly, Bicknell and Häusser explicitly used cable-theory
   variational equations to solve a synaptic credit problem across a detailed
   pyramidal morphology and derived a local approximation
   ([Bicknell & Häusser 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8691952/)).
   The present exact conductance identity and directed-tree path product can be
   valuable technical foundations, but “exact dendritic gradients” cannot be
   the paper’s first-in-field claim.

2. **Neuron-specific teaching coordinates.** Feedback alignment already sends
   neuron-specific teaching vectors through non-symmetric feedback
   ([Lillicrap et al. 2016](https://www.nature.com/articles/ncomms13276)).
   Francioni and colleagues now report causal, signed/vectorized dendritic
   instructive signals in six animals
   ([Francioni et al. 2026](https://www.nature.com/articles/s41586-026-10190-7)).
   The manuscript correctly says that the scalar-to-neuron result is a bandwidth
   effect rather than a new scaling law. Keep that honesty.

3. **Dendritic segregation and local error computation.** Several established
   models use dendritic compartments, interneurons, bursts or prospective
   dynamics for local credit. Generalized Latent Equilibrium also uses
   dendritic morphology and approximated adjoint variables in a local
   spatiotemporal framework
   ([Ellenberger et al. 2025](https://pubmed.ncbi.nlm.nih.gov/41453943/)).

4. **Shunting and reliability-weighted conductance.** Spatial domains of
   dendritic inhibition and high-conductance gain modulation are established
   ([Gidon & Segev 2012](https://pubmed.ncbi.nlm.nih.gov/22841317/),
   [Chance et al. 2002](https://pubmed.ncbi.nlm.nih.gov/12194875/)). Conductance-
   based dendrites have also been derived as reliability-weighted Bayesian cue
   integrators ([Jordan et al. 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012047)).
   The manuscript-specific delta is backward route-gain control and its
   electrotonic boundary, not gain modulation itself.

5. **Anatomy as a constraint on credit.** Connectomic work in songbird already
   links spine/shaft organization to local reinforcement credit
   ([Kornfeld et al., updated 2025 preprint](https://pubmed.ncbi.nlm.nih.gov/41279705/)).
   A June 2026 zebrafish preprint shows developmental alignment between
   instructive inputs and downstream function, and a separate 2026 preprint
   proposes a signed-XOR connectomic error motif
   ([Meier et al. 2026](https://www.biorxiv.org/content/10.64898/2026.06.24.734069v1),
   [Peña Fernández et al. 2026](https://doi.org/10.64898/2026.06.05.730322)).
   These do not duplicate the complete-arbor route-span analysis, but they
   should be acknowledged as concurrent, preprint-level neighbors.

## The most important literature-positioning repair

The current Introduction says that most approaches use a neuronal coordinate
or a few named compartments, followed by “What a full dendritic arbor
contributes after credit reaches a neuron remains unresolved.” The first clause
is broadly fair; the second is too broad because Bicknell and Häusser already
addressed synaptic credit in a detailed arbor. Citing that work only late in the
Discussion under structured plasticity is not enough.

Recommended replacement:

> Most circuit-level learning models deliver a neuron-specific teaching signal
> to one or a few named compartments. Detailed-arbor rules have instead
> optimized single-neuron output using cable sensitivities. What remains
> unresolved is whether a complete arbor provides a useful, bandwidth-limited
> routing basis for teaching coordinates derived from circuit-level error.

Then cite Bicknell and Häusser in the Introduction and again near the exact
gradient. State the delta explicitly: their work learns a single neuron’s
forward input–output computation using detailed cable sensitivities; this paper
studies compression and routing of a circuit-derived teaching field over the
arbor, with alignment, bandwidth and noise controls.

## Terminology audit

### Terms that are clear and should be retained

- **Local eligibility × transported error.** This is intuitive and consistent
  with three-factor learning language. The paper correctly calls exact
  transport an oracle rather than a proposed biological mechanism.
- **Neuron-specific teaching coordinate.** This aligns well with the
  literature’s “vectorized instructive signal.”
- **Route resolution versus physical stage count.** The distinction between
  `D_r` and `D_p` is clear. D1–D4 are explicitly defined as one to four serial
  physical stages and not cell types. This earlier ambiguity is resolved.
- **Point equivalence.** Repeatedly stating that an explicitly gated point
  model must match the same route/gain operation is scientifically important.
- **Qualified capture definitions.** The box distinguishing field, spectral,
  checkpoint and wiring-normalized capture is helpful, even though the family
  remains cognitively dense.

### Terms that should change

1. **“Phase theory.”** I did not find this as an established term in the close
   credit-assignment literature. The manuscript has to disclaim twice that it
   does not mean a thermodynamic or temporal phase. That is a sign that the term
   creates avoidable friction. The actual result is a finite signal–noise
   crossover/regime map. Use **“signal–noise theory of restricted credit
   routing”** or **“credit-routing regime theory.”** Keep “phase diagram” only
   for the alignment-by-bandwidth panel if desired. Rename “phase utility” to
   **“operator utility”** or **“guaranteed-progress score.”** The current title,
   *When dendritic structure helps local credit assignment*, should stay.

2. **“Neuron identity.”** In neuroscience this can imply cell type or
   developmental identity. The result is neuron-indexed/vectorized feedback,
   so prefer **“neuron-specific teaching coordinate”** in headings and first
   mentions. If “identity” remains, define it as index-specific feedback, not
   cell identity.

3. **“Ownership.”** This is understandable after definition but not standard.
   Prefer **“coordinate-to-arbor assignment”** or **“neuron-to-tree mapping”**;
   optionally give “ownership” once in parentheses.

4. **“Ancestry routes.”** In a biological paper, “ancestry” can be read as
   developmental lineage. Prefer **“nested subtree routes”** or
   **“subtree-ancestry routes”** and use one term consistently in text, figures
   and Source Data.

5. **“Credit field.”** This is useful but can be confused with an extracellular
   electric field. At first use, define it explicitly as the vector of exact
   compartment errors (or eligibility-weighted synaptic updates) over an arbor.
   Do not alternate those two estimands under the same unqualified term.

6. **`q_n` versus `∂L/∂V_n`.** The mathematics is correct and the text
   explains where input resistance is placed, but the main narrative makes the
   reader carry two nearly synonymous exact-error quantities. Use one primary
   transported-error symbol in the main text; move the convention conversion to
   Methods or a parenthetical after the first derivation.

7. **LocalCA.** It is properly defined, but 37 uses in the main source make an
   implementation label feel like a field-standard algorithm. After its first
   definition, use “shared-coordinate local rule” and “exact-path local rule”
   in scientific prose where that is clearer.

## Two evidence-language corrections

### Measured MICRONS responses

The abstract says that measured visual responses show no morphology-specific
alignment with “endogenous task credit.” The analysis actually constructs a
response-prediction objective and derives task gradients from measured
responses; it does not directly measure an endogenous teaching signal.

Use:

> measured visual responses show no morphology-specific alignment with
> response-derived task gradients in the analyzed MICRONS cohort.

Similarly, Figure 9G currently says “no endogenous task credit in vivo.” That
is overbroad and conflicts with the adjacent Francioni result. Replace it with
**“no endogenous within-arbor route use established”** or **“no
morphology-specific route alignment detected in the MICRONS response cohort.”**

### Francioni reanalysis

The manuscript says, “We tested this assumption using published source data”
and later calls the result independent support. Francioni et al. already
designed the causal P+/P− comparison and concluded that the signal is
vectorized. Recomputing the published P+/P− contrast is a reanalysis, not an
independent biological test. The new contribution is the signed-versus-common
basis decomposition and its integration into the paper’s hierarchy.

Use:

> We reanalyzed the published six-animal contrasts to express the reported
> vectorized instructive signal in our signed-versus-common coordinate basis.

Then say it is **consistent with** the first level of the framework, not an
independent validation of its existence. Francioni’s causal perturbation and
primary biological result should receive explicit credit
([Francioni et al. 2026](https://www.nature.com/articles/s41586-026-10190-7)).

## Professional presentation audit

### What is already strong

- The seven-word conditional title is accurate and non-hyped.
- The 195-word abstract is within the journal’s 200-word guidance.
- Nine main figures and 61 references fit the journal’s general limits.
- Figures 1–8 use a consistent visual grammar, semantic colors and typography.
  Figure 1 is now a broad-audience framework rather than a wall of equations.
- Figure 4 already contains the appropriate main sequence: operator utility,
  spectral alignment, hierarchy–resolution crossover, projection boundary,
  reliability gain, validation and the integrated regime plane. Same-span
  conditioning is correctly supplementary.
- Figure 9 no longer carries the global phase plane, and its P+/P− signed-mode
  decomposition is clearer than the earlier coarse surrogate-heavy version.
- The bibliography audit passes: 61 cited entries, no undefined keys and no
  unused entries.
- The manuscript reports inferential units, calibrated conditions, nulls,
  point emulations and limitations with unusual care.

### Resolution status of the original editorial findings

1. **Length and hierarchy — resolved.** Introduction + Results + Discussion
   are now about 7,855 mechanically counted words, below the agreed 8,000-word
   ceiling and close enough to the journal’s ideal 5,000 for initial submission.
   The theory, boundary tests and biological hand-off now form the visible
   hierarchy. Official guidance says Methods are typically under 3,000 words
   ([Nature Communications Article guidance](https://www.nature.com/ncomms/submit/article)).

2. **Figure legends — resolved.** Every main-figure legend is now below 350
   words (range 147–204), with rendering/provenance details moved out of the
   legends. The journal asks legends to be short and method-light
   ([submission guidance](https://www.nature.com/ncomms/submit/how-to-submit)).

3. **Figure 9B effect metrics — resolved.** Every target-level contrast is now
   divided by that row’s across-target standard deviation, and the target
   bootstrap recomputes the complete standardized estimand. The axis and legend
   define this common scale explicitly.

4. **Figure 9G evidence language — resolved.** The panel now says that
   morphology-specific alignment and within-arbor route use are not established;
   the text limits the null to the analyzed MICRONS response-derived objective.

5. **Audit vocabulary — resolved in the narrative.** Execution chronology and
   source guards have been consolidated in Methods; Results lead with the
   scientific findings and retain only concise design/provenance statements.

6. **Repository-level cleanup — resolved.** The duplicate sentence is removed;
   the figure README now matches the compiled panels for all nine figures; and
   the figure provenance manifest contains hashes of the rebuilt assets.

7. **Code Availability — internally resolved, public release still external.**
   The manuscript states that the complete MIT-licensed reviewer archive will
   accompany submission and that the same version will receive a public URL and
   permanent archival record. The public repository and DOI must still be
   created by an author before final submission/publication
   ([official submission guidance](https://www.nature.com/ncomms/submit/how-to-submit)).

8. **Related-manuscript disclosure — wording resolved, disposition external.**
   The cover letter says the NeurIPS manuscript is currently under review and
   that the journal paper will not be submitted during concurrent consideration.
   At submission, replace that sentence with the actual rejected/withdrawn
   status and supply the related manuscript plus extension statement
   ([official policy](https://www.nature.com/ncomms/submit/how-to-submit)).

## Recommended claim architecture

Use one sentence repeatedly, in slightly adapted form:

> We use standard adjoint differentiation to define the exact compartment-error
> field, then ask when a bandwidth-limited, tree-structured approximation to
> that field improves local learning.

Organize novelty under three claims:

1. **Theory:** restricted routes help when retained task signal exceeds
   representation, gain and noise costs; the resulting utility predicts both
   immediate progress and trained outcomes.
2. **Boundary:** neuron-specific feedback supplies most useful bandwidth;
   nested subtree routes add a small benefit only at intermediate bandwidth
   and adequate task alignment; physical depth helps only under matched
   nonlinear hierarchy and saturates.
3. **Biological hand-off:** reconstructed arbors can support sparse candidate
   routes and focal conductance can modulate route gain, but measured MICRONS
   responses do not establish endogenous within-tree use. The theory therefore
   yields a focal-inhibition plasticity prediction rather than claiming an
   observed biological mechanism.

This is strong enough for a Nature Communications argument. It is more
credible than a categorical “dendrites solve credit assignment” story and more
distinct from both the detailed-arbor forward-learning literature and the
neuron-specific instructive-signal literature.

## Prioritized revision list

### Implemented before scientific circulation

1. **Done:** moved Bicknell & Häusser into the Introduction and exact-gradient section;
   rewrite the full-arbor gap as a circuit-derived routing gap.
2. **Done:** replaced paper-level “phase theory/phase utility” with
   “restricted-routing signal–noise theory/operator utility”; keep “phase
   diagram” only for the synthesis plot if desired.
3. **Done:** replaced “endogenous task credit” with “response-derived task gradients” in
   the abstract and MICRONS sections.
4. **Done:** reframed the Francioni arm as a published-data coordinate decomposition, not
   an independent test of neuron-specific signals.
5. **Done:** standardized “neuron-specific coordinate,” “coordinate-to-arbor assignment,”
   and “nested subtree route” across text, captions and plots.

### Implemented submission-package repairs

6. **Done:** reduced the narrative below the agreed ~8,000-word interim target; cut
   repeated numeric summaries and execution-language from Results/Discussion.
7. **Done:** brought every legend below 350 words.
8. **Done:** standardized Figure 9B and corrected Figure 9G’s negative claim.
9. **Done:** removed the duplicate and synchronized the figure README.
10. **Partly external:** reviewer code access and disclosure wording are ready;
    the author must mint the public archive/DOI and enter the final NeurIPS status.

### Literature update immediately before upload

11. **Done:** added the two June 2026 preprints on developmental
    alignment and signed-XOR connectomic routing, clearly labeled as preprints.
12. Re-run a date-bounded search because this field is moving quickly; a July
    2026 dendritic predictive-coding article already adds another temporal,
    compartmental approach, although it does not challenge the complete-arbor
    routing claim
    ([Li et al. 2026](https://www.sciencedirect.com/science/article/pii/S0893608026008531)).

## Final verdict

**Literature consistency:** strong after one major citation/positioning repair.  
**Terminology:** mathematically coherent, but “phase theory,” “neuron identity,”
“ownership,” and “ancestry routes” should be normalized to less ambiguous
language.  
**Novelty:** real and potentially Nature Communications-level when centered on
restricted complete-arbor credit routing and quantitative boundary prediction;
not defensible if centered on exact dendritic gradients, vectorized teaching
signals or generic shunting gain.  
**Professionalism:** the editorial hierarchy, legends, terminology, Figure 9
semantics and repository bookkeeping now pass the automated and visual checks.
The scientific package is circulation-ready. Public archival release and the
final related-manuscript disposition remain external submission prerequisites.

## Limitations and confidence

This review used primary research and official journal guidance, with very
recent preprints labeled as such. Searches covered exact-gradient dendritic
learning, vectorized feedback, compartmental credit algorithms, shunting and
conductance, connectomic/anatomical credit, and 2026 adjacent work. Repeated
query families converged on the same close precedents and did not reveal a
direct prior for the complete-arbor alignment-by-bandwidth theory. Literature
search cannot prove uniqueness, and the two newest anatomy/alignment sources
have not been peer reviewed. Confidence is high for the terminology and
positioning recommendations, moderate-to-high for the bounded novelty claim,
and deliberately limited regarding experimental correctness because the runs
were not independently reproduced in this audit.

## Claim-to-source ledger

| Claim | Evidence |
|---|---|
| Fixed random feedback can carry useful neuron-specific teaching vectors | [Lillicrap et al. 2016](https://www.nature.com/articles/ncomms13276) |
| Detailed-morphology cable sensitivities and local synaptic learning predate this manuscript | [Bicknell & Häusser 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8691952/) |
| Active-dendrite error backpropagation predates this manuscript | [Schiess et al. 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004638) |
| Dendritic compartments are established components of biologically motivated deep-credit models | [Guerguiev et al. 2017](https://elifesciences.org/articles/22901), [Payeur et al. 2021](https://www.nature.com/articles/s41593-021-00857-x), [Galloni et al. 2026](https://www.sciencedirect.com/science/article/pii/S2211124726002378) |
| Vectorized dendritic instructive signals were causally reported in six animals | [Francioni et al. 2026](https://www.nature.com/articles/s41586-026-10190-7) |
| Dendritic inhibition and background conductance have established spatial/gain effects | [Gidon & Segev 2012](https://pubmed.ncbi.nlm.nih.gov/22841317/), [Chance et al. 2002](https://pubmed.ncbi.nlm.nih.gov/12194875/) |
| Conductance-based dendrites have an established forward reliability-weighting theory | [Jordan et al. 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012047) |
| Connectomic/anatomical constraints on credit have adjacent precedents | [Kornfeld et al. 2025 preprint](https://pubmed.ncbi.nlm.nih.gov/41279705/), [Meier et al. 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.06.24.734069v1), [Peña Fernández et al. 2026 preprint](https://doi.org/10.64898/2026.06.05.730322) |
| Nature Communications suggests ~5,000 main-text words, <=200 abstract words, <=350 words per legend, <=70 references and up to 10 display items | [Official Article guidance](https://www.nature.com/ncomms/submit/article) |
| Initial submission is flexible, but central code must be available for review and related under-review manuscripts disclosed | [Official submission guidance](https://www.nature.com/ncomms/submit/how-to-submit) |

## Search summary

Search families included: exact dendritic gradient/cable learning; full-arbor
credit assignment; feedback alignment and neuron-specific teaching vectors;
dendritic error, burst and target-propagation models; conductance reliability
and focal inhibition; connectomic substrates of credit; “phase theory,”
“credit operator,” “credit field,” subtree “address” and “routing”; recent 2026
dendritic-credit papers; and official Nature Communications format, code and
overlap policies. Primary papers and official journal pages were used for the
material findings.
