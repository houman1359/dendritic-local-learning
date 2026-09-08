# Nature Communications submission workspace

This directory prepares *Dendritic morphology as a dictionary for local credit assignment* as an Article. The current sequence contains nine main figures and 56 supplementary figures. It follows the task-to-credit argument: neuronal coordinates, branch selection, ancestry grouping, interaction order and learned credit geometry, conductance credit, physical depth, anatomy beyond broadcast, ancestry-partition gain and measured responses.

- `cover_letter.md` and `editorial_summary.md` describe the scientific contribution.
- `extension_statement.md` records prior dissemination and related-work overlap.
- `reporting_checklist.md`, `AUTHOR_ACTIONS.md` and `OFFICIAL_FORMS_REQUIRED.md` track submission information and author declarations.
- `official_forms/` contains the retained official forms and draft technical answers; pending author fields remain identified there.
- Source Data, software, Overleaf and submission ZIPs are generated outputs. Their manifests, source commits and SHA-256 records identify a specific build.

The supplement retains the capacity/estimation, Boolean and prospective-selection analyses. S45 contains expanded image diagnostics, S46 the update-utility material and S47 the weak-channel linearization check. S48 retains normalized shunt doses, S49 the expanded MNIST controls, and S50–S52 the conductance-task and expanded-rate controls. The main figures include both algebraic and conductance credit comparisons, common-broadcast anatomy controls and physical-depth budget analysis. The authoritative figure map is `../figures/README.md`.

Finalize the manuscript, figure assets and panel-level provenance before packaging. Compile both manuscripts and derive page counts from those final PDFs; this README does not freeze provisional page counts. Build Source Data from the current display inventory, commit the scientific/release inputs, then build the committed software release and the Overleaf/submission bundles in their dependency order. Consult `../RELEASE_WORKFLOW.md` for the software installation check and the `../Makefile` for build targets. Internal revision logs, unrelated project files and presentation decks do not belong in submission bundles.

All bundles must contain the same manuscript version, sequence of nine main and 56 supplementary figures, numerical sources and declarations. The presence of an older ZIP, or its valid checksum, does not establish that it matches current files. Supporting Source Data moved to `Methods/retained_evidence/` retain their original analysis locations in the manifest's `original_source` field.

This is the sole journal-paper project. The authors' recorded conference-submission status, approvals, declarations, official forms and immutable data/code access identifiers must agree with the actual submission records. Preparing these files does not submit the manuscript.

S53 retains all local-gate rules and rates; S54–S55 extend every algebraic credit rule to the longer budgets; S56 quantifies measured-response sensitivity conditional on repeat reliability and the observed sampling. Main Fig. 5 uses twenty new local-gate seed blocks; main Fig. 6 plots accuracy and cross-entropy gaps together with stopping status.
