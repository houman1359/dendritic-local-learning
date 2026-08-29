# Nature Communications submission workspace

This directory contains the active preparation materials for a **Nature
Communications Article**. The manuscript may be held until the NeurIPS 2026
decision; it must not be submitted concurrently unless the journals'
applicable policies and the exact conference status permit it.

Active materials:

- `cover_letter.md`: Nature Communications cover letter;
- `editorial_summary.md`: concise editor-facing scientific summary;
- `extension_statement.md`: standalone related-work and overlap disclosure;
- `reporting_checklist.md`: internal scientific and reporting audit;
- `AUTHOR_ACTIONS.md`: decisions and declarations requiring author approval;
- `OFFICIAL_FORMS_REQUIRED.md`: current Nature Portfolio forms;
- `Source_Data.zip`: current generated source-data archive;
- `../main_with_supplementary.pdf`: combined main Article and Supplementary
  Information reading copy.

The August 20--21 `nature_communications_bundle/`, software, Overleaf and
combined submission archives predate the current CIFAR-10, figure and text
integration and must not be submitted. Rebuild them only from the final clean
commit after the author-side gates in `AUTHOR_ACTIONS.md` are resolved.

Build the active bundle only after compiling the paper and supplement,
refreshing Source Data and resolving author-day metadata:

```bash
make submission-bundle
```

The builder writes `Nature_Communications_Submission.zip` and its SHA-256
digest. Archived Nature Neuroscience preparation is historical and is not an
active target.
