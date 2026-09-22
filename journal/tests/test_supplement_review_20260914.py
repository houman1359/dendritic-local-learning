"""Keep reviewed SI captions and figure keys faithful to their frozen analyses."""
from pathlib import Path
import ast
import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest


J = Path(__file__).resolve().parents[1]
SCRIPTS = J / "scripts"
CONSOLIDATION = SCRIPTS / "supplement_consolidation"
REVIEWED = (
    "mechanistic_chain", "utility_signal_noise", "input_coverage_depth",
    "ancestry_coefficients", "physical_architecture", "physical_optimizer",
    "anatomy_preprocessing", "shunt_replication", "measured_transfer_geometry",
    "morphology_estimation",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def specification():
    return load_module("reviewed_si_specification", CONSOLIDATION / "specification.py")


@pytest.fixture(scope="module")
def captions(specification):
    return {row[0]: row[4] for row in specification.FIGURES}


@pytest.mark.parametrize("ident", REVIEWED)
def test_reviewed_caption_survives_regeneration(specification, ident):
    # Load only the pure caption-cleaning function, not the PDF builder's
    # module-level source registry. This check must never rebuild assets.
    tree = ast.parse((CONSOLIDATION / "build.py").read_text())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "clean_caption")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "clean_caption", "exec"), scope)
    _, module, _, title, caption = next(row for row in specification.FIGURES if row[0] == ident)
    expected = scope["clean_caption"](r"\textbf{" + title + "} " + caption)
    if ident in specification.CAPTION_APPEND:
        expected = expected.rstrip() + " " + specification.CAPTION_APPEND[ident]
    tex = (J / "supplementary/curated" / (module + "_figures.tex")).read_text()
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S)
    block = next(b for b in blocks if r"\label{fig:si_" + ident + "}" in b)
    actual = block.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].strip()
    assert actual == expected + "}"
    cache = json.loads((J / "configs/supplement_consolidation/captions.json").read_text())
    assert cache[ident] == expected


def test_uncertainty_and_bound_scope(captions):
    cap = captions["mechanistic_chain"]
    assert "five paired seeds" in cap and "Points are retained runs" in cap
    assert "unresolved execution or normalization settings" in cap
    assert "excluded from the displayed evidence" in cap
    assert "one sample standard deviation" in cap
    assert "below 0.02" not in cap and "0.6--0.9" not in cap
    cap = captions["utility_signal_noise"]
    assert r"bound-optimal nonnegative step $\eta^*$" in cap
    assert "smoothness along the update segments" in cap and "zero updates" in cap
    assert r"\textbf{E} diagnoses actual trained checkpoints" in cap
    assert r"\textbf{F} imposes route alignment and uses oracle projections" in cap


def test_morphology_endpoint_and_alignment_scope(captions):
    cap = captions["input_coverage_depth"]
    for token in ("MNIST", "Each seed averages shunting and raw-additive models", "executed generator is unresolved"):
        assert token in cap
    cap = captions["ancestry_coefficients"]
    assert "held-out accuracy" in cap and "0.82" in cap and "0.64" in cap
    assert "utility" not in cap
    cap = captions["physical_architecture"]
    assert "secant of 5.84" in cap and "all five levels" in cap and "final jump" in cap
    cap = captions["physical_optimizer"]
    assert "restarted from the original seeds" in cap and "continuations" not in cap


def test_s24_palette_matches_s22_arm_roles():
    tree = ast.parse((SCRIPTS / "build_supplementary_figures_s17_s20_native.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "f_rows" for t in n.targets))
    hues = {ast.literal_eval(call.args[0]): ast.literal_eval(call.args[2])
            for call in node.value.elts}
    assert hues == {
        "h2_alignment_interaction__serial_bp": "ink",
        "h2_alignment_interaction__grouped_bp": "point_mlp",
        "h2_alignment_interaction__shared_local": "local",
        "h2_alignment_interaction__path_local": "bp",
    }


def test_equal_cell_weights_are_not_contact_weights(captions):
    cap = captions["anatomy_preprocessing"]
    assert "bootstrap intervals weight cells equally" in cap
    assert "unequal weight" not in cap
    frame = pd.read_csv(J / "source_data/review_morphology_uncertainty/label_missingness.csv")
    frame = frame[frame.grouping.eq("compartment")]
    for stratum in ("soma", "internal", "terminal"):
        group = frame[frame.stratum.eq(stratum)]
        assert len(group) == 8
        assert not np.isclose(group.direct_fraction.mean(), group.n_direct.sum() / group.n_contacts.sum())


def test_calibration_residual_and_labels_use_clipped_target(captions, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = load_module("reviewed_si_calibration", SCRIPTS / "build_supplementary_figure_measured_transfer_geometry_native.py")
    audit = pd.read_csv(J / "source_data/measured_alignment_power/reliability_calibration_audit.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        stats = module.panel_calibration(ax, audit)
        assert stats["beyond"] == 5
        assert "calibration target" in ax.get_xlabel() and "calibration target" in ax.get_ylabel()
        raw = audit.simulated_mean_split_half_spearman - audit.measured_split_half_spearman
        assert int((raw.abs() > 1.96 * audit.simulated_se).sum()) == 7
        assert r"$r_+=\max(r_{\rm measured},0)$" in captions["measured_transfer_geometry"]
        assert "differ from the calibration target" in captions["measured_transfer_geometry"]
    finally:
        plt.close(fig)


def test_si_prose_does_not_publish_production_notes(captions, specification):
    assert "lies near the zero rule" in captions["shunt_replication"]
    assert "animal_credit_reanalysis" not in specification.CAPTION_APPEND
    assert "animal_credit_reanalysis" not in captions
    assert r"\label{tab:animal_credit}" in (J / "supplementary/curated/si_tables_retained.tex").read_text()
    source = (SCRIPTS / "build_supplementary_figure_morphology_estimation_native.py").read_text()
    assert "pt here, not drawn" not in source
    assert "lower confidence bounds" in captions["morphology_estimation"]


def test_inhibitory_crops_keep_axis_text_and_exclude_next_row(specification):
    import fitz

    with fitz.open(J / "figures/supplementary/figure_S12_panels_A-J.pdf") as doc:
        spans = [span for block in doc[0].get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]]
    axis_labels = [fitz.Rect(span["bbox"]) for span in spans
                   if span["text"] == "one-sided Wilcoxon p"]
    next_letters = [fitz.Rect(span["bbox"]) for span in spans
                    if span["text"] in ("I", "J")]
    assert len(axis_labels) == 3 and len(next_letters) == 2
    for panel, label in zip("DEF", sorted(axis_labels, key=lambda r: r.x0)):
        crop = fitz.Rect(specification.PANEL_BOUNDS[("S12", panel)])
        assert crop.contains(label)
        assert crop.y1 >= label.y1 + 0.6
        assert crop.y1 < min(letter.y0 for letter in next_letters) - 1.0
