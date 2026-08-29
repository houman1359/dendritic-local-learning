# Author actions before Nature Communications submission

This document separates tasks that can be checked from the repository from decisions and declarations that only the authors can make. It is an internal sign-off list, not a file to upload unless the submission portal asks for it.

## Hard submission gates

- [ ] **Resolve concurrent consideration.** The NeurIPS 2026 submission is currently in discussion. Before submitting this Article to *Nature Communications*, the authors must decide whether to wait for the NeurIPS decision or formally withdraw the conference submission, and must retain written confirmation of the chosen status. Update the cover letter and related-work statement with the exact status on the day of journal submission. Do not leave language saying that this will be decided later.
- [ ] **Confirm the journal's current prior-publication and exclusivity rules.** Disclose arXiv:2607.03556, the NeurIPS submission or proceedings article as applicable, and arXiv:2607.24990. If the conference paper is accepted, cite it, quantify the scientific and textual overlap and added evidence, and confirm that its license permits any reused material.
- [ ] **Approve authorship.** All authors must approve the title, author order, affiliations, corresponding authors, CRediT statement, acknowledgements, funding statement, competing-interests statement, and submitted files. Maceo Richards remains an author because the integrated Article retains theory and experiments from the earlier work.
- [x] **Finish the journal-package evidence freeze.** The clean Figure 2 cohort,
      all figures, Source Data, provenance hashes, and strict audit are complete.
      The clean-commit repeat and reviewer-archive rebuild are recorded below.
- [ ] **Create immutable data and code records.** Deposit the final derived source data and software release in an appropriate stable repository. Add the public or reviewer-accessible URL, immutable version or commit, DOI/accession when available, license, and access conditions to the Data availability and Code availability statements.
- [ ] **Resolve the MICrONS proxy-table status.** Confirm a citable public release and access terms for `synapse_target_predictions_ssa_v2`. If that cannot be done, retain proxy-derived results only as clearly labeled secondary analyses and ensure that the direct-type public analysis carries the principal robustness claim.

## Author and portal metadata

- [ ] Enter each author's full legal publication name, email, ORCID, affiliation and country exactly as approved by that author.
- [ ] Confirm the corresponding author or authors and supply complete postal addresses, telephone details if requested, and current institutional email addresses.
- [ ] Approve the article type as **Article** and the primary target as **Nature Communications**.
- [ ] Confirm that the title in the portal exactly matches `main.tex`.
- [ ] Paste the abstract from the final manuscript and verify the portal character handling.
- [ ] Supply concise editor-facing subject classifications and keywords. Suggested starting concepts are dendritic computation, biologically plausible learning, credit assignment, shunting inhibition, neuronal morphology and neuroAI.
- [ ] Write a short, non-promotional significance statement if requested by the portal; do not claim state-of-the-art accuracy, biological learning observed in vivo, or independent-animal replication.
- [ ] If requested, provide suggested and opposed reviewers with institutional email addresses, relevant expertise, conflict checks and a brief factual rationale. The authors must select these names.
- [ ] Identify any related manuscripts, preprints, conference submissions, patents, datasets or software records requested by the portal.

## Manuscript and scientific sign-off

- [ ] Confirm the final corresponding-author block and affiliations in `main.tex`.
- [ ] Confirm that the abstract, main text, Methods, figure legends and Supplementary Information describe the same frozen cohorts, sample sizes, estimands and figure numbering.
- [ ] Verify every numerical statement against the final source-data tables and have a second author independently check the load-bearing claims.
- [ ] Verify that the primary inferential unit is the training seed, reconstructed cell or postsynaptic target, as appropriate, and that nested sites, streams and stimulus splits are not presented as independent biological replicates.
- [ ] Confirm the scope language: the MICrONS analyses test modeled credit-routing capacity and perturbation sensitivity, not endogenous learning in the animal; the 47-cell cohort is disjoint by nucleus but comes from the same mouse.
- [ ] Confirm that exact transport, fitted coefficients and dense projections are labeled as information oracles wherever used.
- [ ] Run a final overlap review against arXiv:2607.03556 and any proceedings version. Cite reused foundations and replace avoidable verbatim overlap.
- [ ] Check every citation against the final bibliographic record, including spelling, year, journal, volume, pages or article number and DOI where appropriate.
- [x] All main and supplementary figures were inspected at journal size; their
      labels remain legible, fonts are embedded, adjacent panels do not overlap,
      and the legends and in-text callouts agree with the rendered panels.

## Data, code and reproducibility

- [x] The current `submission/Source_Data.zip` includes the prospective,
      physical-depth, point--dendrite, alignment-dose, Fashion-MNIST, literal
      grouped-point, independent H2, task-family, phase-plane,
      wiring-efficiency and two-animal structural-capacity data. Its manifest
      matches the nine-main-figure and twenty-nine-supplementary-figure layout.
- [x] The integrated Article, CIFAR-10 confirmation, figures and workshop source
      are committed, and the reviewer software archive has been rebuilt from a
      clean commit. Verify its supplied SHA-256 sidecar immediately before
      upload.
- [x] Recorded the source commit, package versions, accelerator model, random
      seeds, run manifests, and checkpoint or output hashes for the reported
      artificial-network results.
- [ ] Confirm that all public-source identifiers are current: MICrONS `minnie65_public`, materialization 1822, the official v661 release, the source-study DOI, DANDI Dandiset 000402 and the seven assets used here.
- [ ] Replace the draft DANDI version with an immutable version or DOI if one becomes available; otherwise state the live status accurately and provide asset identifiers.
- [x] Repository tests, numerical checks, manuscript builds and strict main/SI
      submission audits were repeated from the clean committed source. The
      release metadata records that commit, and each rebuilt archive has a
      SHA-256 sidecar.
- [x] Freeze the current technical package only after all required files and provenance hashes are final, then build it with `python scripts/build_submission_bundle.py`. Rebuild the bundle after any author-day metadata or disclosure changes.

## Declarations requiring author confirmation

- [ ] Confirm that no new animal or human experimentation was performed and that reuse of the cited public datasets requires no additional local approval or consent statement.
- [ ] Obtain a competing-interests declaration from every author and update the manuscript if any relevant patent, consulting, equity, funding or software interest exists.
- [ ] Verify the exact Chan Zuckerberg Initiative Foundation support language, any grant or gift identifiers, computing allocations, fellowships and data-resource acknowledgements with institutional records.
- [ ] Confirm the author-contribution statement with all authors using the journal's requested taxonomy.
- [ ] Review the computational-assistance disclosure against the Nature Portfolio policy in force on the submission date. All authors remain responsible for, and must independently verify, the scientific content.

## Official forms and upload set

- [ ] Download fresh copies of the applicable forms from the official *Nature Communications* resources page and complete them as described in `OFFICIAL_FORMS_REQUIRED.md`.
- [ ] Open and save Nature's smart PDF forms using Adobe Acrobat Reader; confirm that saved answers remain visible after closing and reopening each file.
- [ ] Upload the main manuscript, Supplementary Information, figure files, Source Data, code/software material or reviewer link, cover letter, reporting forms, and related-work disclosure in the categories requested by the portal.
- [ ] Open every uploaded PDF and archive from the portal preview, not only from the local filesystem.
- [ ] Confirm that the final cover letter no longer contains future-tense placeholders about exclusivity, code, data or author approval.

## Final author declaration

- [ ] All authors have read the exact submitted version.
- [ ] All authors approve submission to *Nature Communications*.
- [ ] The manuscript is not simultaneously under consideration elsewhere under the policies applicable on the submission date.
- [ ] Every required disclosure is complete and accurate.
