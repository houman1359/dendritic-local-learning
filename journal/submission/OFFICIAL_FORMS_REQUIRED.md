# Official Nature Communications forms and instructions

Official *Nature Communications* resources for author verification before submission:

- Article instructions: <https://www.nature.com/ncomms/submit/article>
- Submission resources and downloadable forms: <https://www.nature.com/ncomms/submit/resources>

Original official PDFs and their download hashes are retained in `official_forms/`. Prepared technical answers for every applicable field are in `official_forms/COMPLETION_ANSWERS.md`; author-only fields remain pending. The machine-learning form has a clearly labeled populated draft, `official_forms/DRAFT_machine-learning-checklist_technical.pdf`, with saved widget values checked and all three rendered pages visually inspected in PyMuPDF. Its validation JSON records the source and answer-document versions used. Author-controlled fields remain pending. The software/reporting PDFs are XFA forms whose transfer remains pending. None of the forms has been validated in Adobe Acrobat Reader. Download fresh copies immediately before submission if the journal has updated them. The journal can revise forms and portal requirements without changing this repository. The local `reporting_checklist.md` is a project audit and does **not** replace an official Nature Portfolio form.

## Forms applicable to this Article

### Code and Software Submission Checklist

**Applicability:** required for this study because custom software is central to the theoretical checks, artificial-network experiments, morphology analyses and figure generation.

The corresponding author should verify at minimum:

- repository or archive location, immutable version and license;
- installation and environment instructions;
- hardware and software requirements;
- entry points connecting reported results to code and configurations;
- random seeds and expected outputs;
- tests and numerical validation;
- access arrangements during peer review; and
- removal of credentials, private filesystem paths and non-redistributable data.

### Machine Learning Checklist

**Applicability:** required or strongly applicable because the paper trains and evaluates machine-learning models and compares learning rules.

Complete the journal's current form using the frozen experimental record. Report the model architecture and dendritic morphology, datasets and splits, optimizer and schedules, feedback modes, initialization and gate policies, seeds, model-selection rules, computational resources, uncertainty and statistical units. Clearly distinguish mechanistic tests from benchmark claims. The paper does not claim state-of-the-art accuracy, memory or latency.

### Life Sciences Reporting Summary

**Applicability:** required or strongly applicable because the Article analyzes neuronal reconstructions, synapses and measured visual responses from public life-science datasets.

State that no new animals or human participants were used. Identify the source studies and public resources, inclusion and exclusion rules, biological replication unit, nested observations, randomization or outcome-free selection, lack of blinding where appropriate, sample sizes, statistical tests, software, and data identifiers. Distinguish the two MICrONS mice: the original minnie65 pilot, disjoint 47-cell sensitivity, functional and focal analyses come from one mouse; the Pinky structural-capacity analysis supplies a second mouse. Two mice provide a directional structural check, not population-level animal inference. The external six-animal signed-coordinate reanalysis is supplementary and does not measure dendritic credit gradients.

## How to complete Nature's smart PDFs

Nature's resources page warns that its reporting-summary and checklist PDFs use interactive fields. Use the current desktop version of Adobe Acrobat Reader rather than a browser PDF viewer or macOS Preview. After completing each form:

1. save it under a versioned filename;
2. close and reopen it in Adobe Acrobat Reader;
3. confirm that every response is still visible;
4. print or export a flattened review copy if useful, while retaining the editable original; and
5. have the corresponding author approve the final saved form.

Do not commit completed forms containing personal contact details or confidential editor/reviewer information to a public repository.

## Other portal declarations to prepare

These may be portal fields rather than downloadable forms, but they require explicit author confirmation:

- authorship and CRediT contributions;
- corresponding-author contact information;
- funding and acknowledgements;
- competing interests;
- ethics and consent for reuse of public data;
- data availability and access restrictions;
- code availability, license and reviewer access;
- related manuscripts and preprints;
- prior dissemination and text or figure reuse;
- suggested or opposed reviewers, if requested; and
- confirmation of exclusive consideration.

The last item cannot be inferred from the repository. The author reports that the NeurIPS 2026 submission is expected not to be accepted but has not received an official rejection. Nature Communications is the intended sole publication; the pending conference status is not treated as confirmed exclusivity. The authors must confirm and document the current status before journal submission.

## Format checks from the official Article instructions

The current draft is designed for the *Nature Communications* Article format. Before upload, verify the current limits on the official Article page. The local format audit retains approximately 5,000 words for Introduction, Results and Discussion as advisory journal guidance. The author requested a roughly 8,000-word working main text, so length remains an editorial consideration rather than a silently satisfied submission requirement. Check title, abstract, display-item and reference guidance on the official page before upload. The present Article has nine main figures and 34 supplementary figures. Portal validation and the current journal page take precedence over this note.
