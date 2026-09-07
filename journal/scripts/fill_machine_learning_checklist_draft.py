"""Fill verified technical ML-checklist widgets; never sign or attest access."""
from pathlib import Path
import hashlib
import json
import re
import fitz

ROOT = Path(__file__).resolve().parents[1]
FORMS = ROOT / "submission/official_forms"
ORIGINAL = FORMS / "machine-learning-checklist.pdf"
DRAFT = FORMS / "DRAFT_machine-learning-checklist_technical.pdf"
VALIDATION = FORMS / "DRAFT_machine-learning-checklist_validation.json"

# Xrefs were matched to the official item text, position and button export state,
# then checked against a rendered original page. They are not inferred from the
# repeated generic field names (e.g., "Yes-0"). The source hash gate below also
# binds this mapping to the downloaded original.
TEXT = {
    476: "Prepared source: Dendritic_credit_assignment_software.zip; upload pending.",
    478: "Not applicable: research source code, no standalone compiled application.",
    480: "Synthetic generators, test scripts and README in prepared reviewer archive.",
    482: "Installation/run instructions in reviewer archive README and experiment READMEs.",
    485: "Not applicable: no externally pretrained models.",
    126: "Methods; SI S4-S10; Figs. S33-S52: cohort, finite-label, fixed-template and model-class limits.",
    130: "Methods and SI S5-S10; executable preprocessing and response-baseline audit.",
    134: "Methods: anatomy/response joins; SI S6, S8 and S10; nested target/scan units.",
    136: "Conductance, linear, multi-affine trees (SI S1/S4/S5).",
    144: "Methods; SI S4/S5/S10: separate task seeds, calibration and stimulus identities.",
    149: "Mechanistic research: rotations; randomized interactions; seven fixed Boolean templates; observed-input visual responses.",
    152: "SI S4: sealed selections; SI S10: nested training-only tuning and leakage audit.",
    156: "Analytic gradients, explicit route dictionaries and controlled ablations; SI S1-S5.",
    160: "Methods and SI: accuracy, loss/NMSE, capture, update match, costed regret, selectivity.",
    166: "MNIST, Fashion-MNIST and CIFAR-10; Methods and SI S5.",
    170: "Figs. 1-9/S31/S34-S52: broadcast, rank, decoder-only, point/ridge and privileged target-informed references.",
    175: "No state-of-the-art accuracy or deployment claim; comparisons isolate mechanisms, information, optimization and resource constraints.",
    178: "Feedback/topology/optimizer/shunt; input grouping, exact/broadcast credit; cue and mapping controls.",
}

# Checkboxes on page1 and exactly one button of each resolved yes/no group.
SELECTED = {
    475: "1 source code prepared in submission archive",
    479: "1 test generators/data and replication instructions prepared",
    481: "1 installation/run README prepared",
    488: "2A data sources: yes",
    124: "2C dataset biases: yes",
    128: "2D preprocessing: yes",
    132: "2E combining sources: yes",
    139: "3B separate Model Card: no",
    140: "3C train/validation/test separation where fitted: yes",
    142: "3D split method stated: yes",
    147: "3E deployment-like split claim: no",
    150: "3F leakage avoidance: yes",
    154: "3G mechanistic interpretability: yes",
    158: "4A metrics: yes",
    162: "4B nested response-baseline cross-validation: yes",
    164: "4C community benchmarks: yes",
    168: "4D simple baselines: yes",
    173: "4E state-of-the-art benchmarking: no",
    176: "4F ablations: yes",
    182: "5A recorded hardware information: yes",
    184: "5B recorded compute costs: yes",
}

PENDING = {
    "author_identity_and_date": "Corresponding-author and author-update fields remain blank.",
    "reviewer_access": "Prepared local archive is not an actual uploaded or authorized public access link.",
    "public_identifiers": "Public DOI/version/access assurances require author completion; no identifier invented.",
    "2B_all_datasets_public": "Both buttons unselected: public benchmark/generator sources coexist with explicit biological/proxy access limits.",
    "4G_fully_independent_dataset": "Both buttons unselected: simulation/image replications and second-mouse structural checks have distinct scopes; author must resolve form-wide wording.",
    "signoff_exclusivity_ethics": "No author approval, exclusive-submission, conflict, funding or ethics attestation made.",
    "reader_verification": "Saved/reopened with PyMuPDF only. Adobe Acrobat Reader and corresponding-author verification remain pending.",
    "XFA_forms": "Software and Reporting Summary forms remain original; technical responses are in COMPLETION_ANSWERS.md.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original_hash = sha(ORIGINAL)
    assert original_hash == "bf23bd9bffb3b0f5f6650ceaec66c5dd49b59ea105ea96617ad04180d41f4a14"
    inventory = json.loads((FORMS / "machine_learning_widget_inventory.json").read_text())
    expected = {r["xref"]: r for r in inventory}
    document = fitz.open(ORIGINAL)
    assert len(document) == 3 and document.is_form_pdf
    changed = []
    for page_num, page in enumerate(document):
        for widget in page.widgets() or []:
            ref = expected[widget.xref]
            assert ref["page"] == page_num+1 and ref["name"] == widget.field_name
            assert ref["type"] == widget.field_type_string
            if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                # Clear preprinted example responses, including identity/date,
                # without substituting a generated identity or approval date.
                widget.field_value = TEXT.get(widget.xref, "")
                if not widget.field_value:
                    # PyMuPDF's widget update skips a falsy empty value. Clear
                    # both PDF entries explicitly so old example answers do
                    # not silently return when the saved form is reopened.
                    document.xref_set_key(widget.xref, "V", "()")
                    document.xref_set_key(widget.xref, "DV", "()")
                widget.text_font = "Helv"
                widget.text_fontsize = 7 if widget.xref == 136 else 8
                widget.text_color = (0.05, 0.15, 0.38)
                widget.update()
                changed.append(dict(page=page_num+1, xref=widget.xref,
                    name=widget.field_name, expected_value=widget.field_value,
                    action="technical_text" if widget.xref in TEXT else "clear_placeholder_pending_or_unused"))
            elif widget.xref in SELECTED:
                widget.field_value = widget.on_state()
                widget.update()
                changed.append(dict(page=page_num+1, xref=widget.xref,
                    name=widget.field_name, expected_value=widget.field_value,
                    action=SELECTED[widget.xref]))
        page.wrap_contents()
        page.insert_text((55, 20), "DRAFT - technical answers only; author and final access verification pending",
                         fontsize=8, fontname="hebo", color=(.55, .12, .12))
    metadata = dict(document.metadata)
    metadata["title"] = "DRAFT technical answers - Nature Portfolio Machine Learning Checklist"
    metadata["subject"] = "Author identity, approval, access identifiers and Reader verification pending"
    document.set_metadata(metadata)
    document.save(DRAFT, deflate=True)
    document.close()
    assert sha(ORIGINAL) == original_hash
    reopened = fitz.open(DRAFT)
    retained = {w.xref: (i+1, w) for i, p in enumerate(reopened) for w in (p.widgets() or [])}
    for item in changed:
        page, widget = retained[item["xref"]]
        assert page == item["page"] and widget.field_name == item["name"]
        expected_value = item["expected_value"]
        if item["xref"] in SELECTED:
            # PDF name escapes are decoded on reopening (e.g., #20 -> space).
            expected_value = re.sub(r"#([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), expected_value)
        assert widget.field_value == expected_value, (item, widget.field_value)
        item["expected_reopened_value"] = expected_value
        if item["xref"] in SELECTED:
            appearance_state = reopened.xref_get_key(item["xref"], "AS")[1]
            assert appearance_state not in ("/Off", "null"), item
            item["retained_appearance_state"] = appearance_state
        item["reopened_value_matches"] = True
    for ref in (472, 473):
        assert retained[ref][1].field_value == ""
    for ref in (490, 491, 180, 181):
        assert retained[ref][1].field_value in ("Off", "")
    assert len(retained) == len(expected)
    for i, page in enumerate(reopened):
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(f"/tmp/ml_checklist_DRAFT_page{i+1}.png")
    validation = dict(status="passed_PyMuPDF_reopen_only_not_author_approval", original_unchanged=True,
        original_sha256=original_hash, draft_sha256=sha(DRAFT), script_sha256=sha(__file__),
        answer_document_sha256=sha(FORMS / "COMPLETION_ANSWERS.md"),
        original_pages=3, draft_pages=len(reopened), original_widgets=len(expected), retained_widgets=len(retained),
        selected_technical_buttons=len(SELECTED), populated_technical_text_fields=len(TEXT),
        all_requested_values_retained=True, draft_header_present_all_pages=all("DRAFT - technical answers only" in p.get_text() for p in reopened),
        pending=PENDING, field_checks=changed,
        scope="DRAFT generated by faithful AcroForm widget mapping. Prepared archives exist locally; no upload, public deposition, author sign-off or Reader compatibility is certified.")
    VALIDATION.write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({k:v for k,v in validation.items() if k not in ("pending", "field_checks")}, indent=2))


if __name__ == "__main__":
    main()
