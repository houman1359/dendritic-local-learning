# Official Nature Communications forms and instructions

Checked against the current official *Nature Communications* author resources available on 9 August 2026:

- Article instructions: <https://www.nature.com/ncomms/submit/article>
- Submission resources and downloadable forms: <https://www.nature.com/ncomms/submit/resources>

Always download fresh copies immediately before submission. The journal can revise forms and portal requirements without changing this repository. The local `reporting_checklist.md` is a project audit and does **not** replace an official Nature Portfolio form.

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

State that no new animals or human participants were used. Identify the source studies and public resources, inclusion and exclusion rules, biological replication unit, nested observations, randomization or outcome-free selection, lack of blinding where appropriate, sample sizes, statistical tests, software, and data identifiers. Make clear that all MICrONS cells come from one animal and that the 47-cell cohort is disjoint-cell robustness rather than independent-animal replication.

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

The last item cannot be inferred from the repository. Because the NeurIPS 2026 submission is active in the current record, the authors must resolve and document its status before journal submission.

## Format checks from the official Article instructions

The current draft is designed for the *Nature Communications* Article format. Before upload, verify the current limits on the official Article page. The present instructions specify a title of no more than 15 words, an abstract of no more than 200 words, a main-text guideline of approximately 5,000 words for Introduction, Results and Discussion, up to 10 display items, and up to 70 references. Portal validation and the current journal page take precedence over this note.
