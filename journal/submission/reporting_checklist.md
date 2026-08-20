# Nature Communications submission and reporting checklist

Status key: `[x]` complete in the current draft; `[~]` drafted but requires a final author or output check; `[ ]` not yet complete.

## Article format and editorial package

- [x] Article title is concise, declarative, contains no punctuation and is under 15 words.
- [x] Abstract is one paragraph and remains below the journal's 200-word
      guideline; the automated format audit records the exact current count.
- [x] Main narrative is organized as Introduction, Results, Discussion and Methods.
- [x] The working article contains nine numbered main figures, each compiled
      from one vector PDF. Expanded controls are Supplementary Figures S1--S26.
- [x] A standalone Supplementary Information manuscript compiles.
- [x] A cover letter and standalone related-work/overlap statement are drafted.
- [ ] Confirm the final corresponding author, postal address, email and journal-portal metadata.
- [ ] Confirm the final author order and CRediT contributions with all authors.
- [ ] Update the NeurIPS status and prior-publication disclosure on the day of submission.
- [x] Checked the current Nature Communications aims, Article instructions and
      submission resources on 11 August 2026; recheck on the upload date.

## Study design and replication units

- [x] Artificial-network replication unit is the independent initialization/training seed.
- [x] MICrONS structural and focal-perturbation replication unit is the reconstructed cell.
- [x] Measured-response replication unit is the postsynaptic target cell.
- [x] The inhibitory-census replication unit is the postsynaptic target cell;
      contacts and connections are nested within 20 targets.
- [x] The external signed-credit replication unit is the animal (n=6).
- [x] Sites, synapses, stimulus pairs, batches, epochs and Monte Carlo streams are not counted as independent biological replicates.
- [x] Sample sizes are stated in the main text and supplement: 1,840 historical prospective/follow-up executions with 1,000 input-valid runs retained; a detached 320-run exact/backpropagation audit; 120 retained diagnostic checkpoints; 2,700 trained subtree-address fits at 20 paired seeds; 200 same-task point--dendrite and BP--local-credit fits, 90 physical-alignment interpolation fits, 220 literal-point/second-hierarchy fits, a 360-fit H4 factorial, 430 same-seed immutable-source H2/H3 reruns and 60 Fashion-MNIST ladder fits at ten paired seeds; 1,050 adaptive-reliability outcomes at 50 fresh paired seeds; 15 paired seeds for the earlier feedback experiment; five seeds for the earlier exact-transport factorial; eight pilot morphology cells; 512 accepted active-conductance equilibria; 47 disjoint v661 routing cells; 45 v661 focal cells with 235 sites; 40 v661 relation-control cells with 230 sites; 20 inhibitory-census targets with 402 multi-clump connections; 13 eligible functional scans nested in seven targets; 520 complete-tree fits; and six animals in the external signed-credit reanalysis.
- [x] No new animals or human participants were used; biological data are from public MICrONS/DANDI and publisher-supplied source-data resources.
- [x] The one-animal and selected-cell limitations of MICrONS are stated.
- [x] The focal endpoint and site-selection rule are described as frozen before outcome inspection.
- [x] The dated analysis contract and claim-to-evidence ledger are included and
      hashed in the reviewer software archive.

## Inclusion, exclusion, randomization and blinding

- [x] Morphology mapping threshold is stated: nearest dendritic cable within 5 micrometres.
- [x] Focal sites require at least three descendant and three comparison sites and are selected by a depth-stratified rule without gradient outcomes.
- [x] Functional joins, reliability thresholds, complete-stimulus holdouts and manual/automatic match tiers are described.
- [x] Analysis exclusions are rule based rather than chosen by outcome.
- [x] Paired seeds or initializations are used where the scientific comparison permits them.
- [x] Blinding is not claimed. Most endpoints are deterministic computational quantities; outcome-free selection and frozen analysis code serve the relevant bias-control role.
- [~] Cell- and target-level machine-readable accounting is complete for the original eight-cell pilot, seven-target functional cohort, all 55 v661 candidates, and the complete 20-target inhibitory overlap. The final archive should additionally retain any low-level mapped-synapse and candidate-site rejection tables produced during regeneration.
- [x] The v661 sensitivity cohort reports all 55 frozen candidates, exclusion of the eight original pilot nuclei, endpoint eligibility, retrieval status and exclusion reasons before its 47-cell outcomes.

## Statistics

- [x] All central comparisons state the inferential unit.
- [x] Central effects include paired differences and 95% intervals.
- [x] Small paired-cell comparisons use exact two-sided Wilcoxon signed-rank tests.
- [x] Hierarchical intervals resample cells before nested Monte Carlo streams or sites.
- [x] Target-level intervals resample postsynaptic targets.
- [x] Null results are described as a lack of reliable evidence rather than proof of equality.
- [x] Exact identities and implementation checks report numerical errors instead of inferential tests.
- [x] No (P) value based on (n=3) is used as central evidence.
- [~] Confirm exact test implementations, bootstrap replicate counts and random seeds in a panel-level statistics manifest.
- [x] Prespecified single endpoints are reported without omnibus multiplicity correction; the six validity-qualified routing contrasts additionally report Benjamini--Hochberg-adjusted decisions. Channel sweeps, parameter grids and reliability thresholds are labeled secondary sensitivity analyses.

## Artificial-network reporting

- [x] Morphology, rule, feedback field, optimizer matching and seed counts are described for central comparisons.
- [x] Three-factor local eligibility is distinguished from four- and five-factor empirical preconditioners.
- [x] Exact transport and fitted projections are explicitly labeled oracles.
- [x] The additive architecture is defined and not described as a conductance model.
- [x] The manuscript does not claim SOTA performance or superiority in memory or latency.
- [x] Frozen machine-readable sweeps specify seeds, feedback, morphology, gate policy, optimizer, schedule, batch size, decoder and selection settings. The clean 15-seed replacement cohort supplies Figure 2. The complete historical programme passed artifact audit; an outcome-independent rule retains 480/640 central runs, 120/160 routing runs, 240/320 spatial-topology runs, 160/320 fixed-budget runs and 120/160 diagnostic checkpoints, while excluding the complete 400-run inhibitory-dose family. The detached 320-run clean exact/backpropagation audit passed every artifact and finite stage-complete checkpoint gate, and the 2,700-fit subtree-address factorial supplies the within-neuron control.
- [x] The clean cohort retains resolved configurations, checkpoints, logs, H200 task timing, software identity, and source/output hashes for all 60 runs.
- [x] The prospective cohort retains resolved configurations, checkpoints,
      learning curves, held-out metrics, resource records, frozen manifests,
      and a source-equivalence audit for all 640 runs.
- [ ] Complete a final ML reproducibility checklist using the journal's current template if provided.

## MICrONS and functional-data reporting

- [x] The current analysis identifies CAVE datastack `minnie65_public`, materialization 1822 and every table used.
- [x] Stable nucleus-to-root resolution is described.
- [x] Skeleton compression, synapse mapping and parent-tree validation are described.
- [x] Direct presynaptic E/I types are distinguished from the target-structure proxy.
- [x] Proxy concordance and the direct-type-only sensitivity analysis are reported.
- [x] Passive electrical quantities are described as normalized sensitivity weights, not fitted biophysical conductances.
- [x] Functional input responses are not called teaching signals.
- [x] Complete stimulus identities are held out in task-derived analyses.
- [~] Stable identifiers are recorded for MICrONS materialization 1822, the official minnie65 v661 static release and tables, primary dataset DOI `10.1038/s41586-025-08790-w`, DANDI Dandiset `000402`, and each of the seven DANDI assets. DANDI is still version `draft`, so an immutable dataset version or DOI remains pending.
- [~] `synapse_target_predictions_ssa_v2` is identified as an in-development proxy table. Confirm a public release identifier before submission; otherwise retain proxy-based results only as a documented secondary analysis.

## Figure and source-data integrity

- [x] Every quantitative panel in the nine numbered main figures and twenty-six supplementary figures has machine-readable source tables with plotted values and unit identifiers; conceptual panels are programmatic. The 360-fit H4 factorial and 430-fit immutable-source replication passed their frozen artifact and resource gates; Supplementary Figure S23 reconstructs all 2,700 route-factorial fits, Supplementary Figure S24 reports all 1,050 adaptive-reliability outcomes, Supplementary Figure S25 reports the irregular-tree analysis and Supplementary Figure S26 reports H2/H3 source concordance.
- [x] The panel-level provenance manifest records source path, SHA-256 hash,
      generator and inferential unit for Figures 1--8 and Supplementary Figures S1--S26; hashes include the final alignment-dose, Fashion-MNIST, detailed morphology, focal and measured-response panels, trained partition residual, adaptive reliability, irregular-tree wavelets and the clean-source replication.
- [x] Plotting scripts read the frozen source tables or write and then read the deterministic analysis outputs; numerical result labels are not maintained as an independent hand-entered source.
- [~] Final figures use embedded Type 1 or TrueType fonts and a consistent,
      color-accessible palette; the existing renders and the alignment-dose
      continuation passed automated layout/font checks and visual inspection.
- [x] Panel labels, legend definitions, in-text callouts and sample sizes were
      audited after the final source replacement.
- [x] Underlying cell-, target-, seed-, site- or stream-level numerical data are retained for every main empirical figure; large stream tables may be compressed.
- [x] The current audit excludes withdrawn pooled metrics, and the stale-value and unmatched-cohort search was repeated after the clean Figure 2 decision.

## Data availability

- [x] The draft identifies MICrONS and DANDI as public upstream resources.
- [x] Authentication tokens and externally hosted raw caches are excluded from redistribution.
- [x] Derived manifests, endpoint exclusion logs and non-restricted source data are organized locally for release.
- [ ] Deposit the final derived-data package in a stable repository and add its DOI/accession.
- [x] The journal-formatted Source Data package was regenerated for the final
      eight-main-figure, twenty-five-supplementary-figure layout. Its allow-listed
      archive includes the H3 literal grouped-point control, the
      independent H2 BP/LocalCA cohort, paired contrasts and completeness audit;
      the ZIP integrity check passes.
- [ ] State any access conditions for large or restricted upstream data exactly as required by the provider.

## Code availability and reproducibility

- [x] Analysis scripts and tests are in the project workspace.
- [x] Numerical checks cover exact gradients, tree construction, projection identities and focal finite differences.
- [x] A validated local reviewer package contains portable reference equations,
      frozen configurations, exact source copies, the task-derived pipeline,
      analysis contracts, origin hashes, and source-data metadata.
- [ ] Create an immutable release or private reviewer archive from the final commit.
- [ ] Record the repository URL, commit hash, environment lock file and archive DOI.
- [ ] Run the full manuscript build, test suite and submission audit from a clean checkout.
- [x] Automated release scans found no credentials, absolute private paths,
      unpublished data, scheduler logs, checkpoints, or large caches.

## Ethics, consent and competing interests

- [x] No new animal or human experimentation was conducted by this study.
- [~] Confirm that reuse of public MICrONS/DANDI data requires no additional local ethical approval statement beyond citation of the source study.
- [x] A no-competing-interests statement is present.
- [ ] Obtain final approval of the competing-interests statement from every author.
- [ ] Check whether any code, patent, consulting or funding relationship requires disclosure.

## Funding and acknowledgements

- [x] The Chan Zuckerberg Initiative Foundation support for establishing the Kempner Institute is acknowledged.
- [x] The MICrONS Consortium and CAVE/DANDI maintainers are acknowledged.
- [ ] Verify the exact funder name, grant or gift language and grant numbers with institutional records.
- [ ] Add any omitted computing allocations, data-resource awards and individual fellowships.

## Prior dissemination and authorship

- [x] arXiv:2607.03556 and the NeurIPS 2026 submission are disclosed in the draft cover letter.
- [x] The related-work statement identifies the integrated conference-stage
      foundation and the evidence added in the standalone journal Article.
- [x] Maceo Richards is retained as an author for reused first-paper theory and experiments.
- [ ] Update the earlier-work status after the NeurIPS decision.
- [ ] Run a final text- and figure-overlap audit and cite the proceedings article if accepted.
- [ ] Confirm that all reused content satisfies conference and journal copyright policies.

## Computational assistance disclosure

- [x] The manuscript contains a transparent draft disclosure covering code assistance, consistency checks and language editing.
- [ ] Reconstruct the verified record of tool use with all authors.
- [ ] Adjust the wording to the Nature Portfolio policy current on the submission date.
- [ ] Ensure the authors independently verify every numerical result, citation and scientific statement.

## Final sign-off

- [ ] All author names, affiliations and contribution statements approved.
- [ ] All numerical claims traced to source data and independently checked.
- [x] All references were verified against publisher or indexed bibliographic records.
- [x] The main manuscript and supplement compile without errors, and the final
  Source Data archive opens cleanly after integration of the fixed-budget
  cohort and eight-display figure order.
- [ ] No unresolved placeholders remain in submission files.
- [ ] Cover letter statements about exclusivity, approval, competing interests and availability are true on the day of submission.
