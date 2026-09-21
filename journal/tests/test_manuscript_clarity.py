"""Protect the scientific distinctions clarified in the manuscript review."""
import ast
from pathlib import Path
import re

J = Path(__file__).resolve().parents[1]


def function_source(path, name):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def test_measured_partner_schematic_matches_excitatory_recording_cohort():
    source = function_source(
        J / "scripts/credit_first_figures/build_measured.py", "panel_statistic")
    contacts = [node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "contact"]
    assert len(contacts) == 3
    for node in contacts:
        kind = next(k.value for k in node.keywords if k.arg == "kind")
        assert ast.literal_eval(kind) == "exc"


def test_population_panel_has_its_own_learning_rule_key():
    source = (J / "scripts/inhibitory_selection/figure_panels.py").read_text()
    assert "h.legend(" in source
    for label in ("Exact BP", "Unit broadcast", "Relative-resistance gate"):
        assert label in source
    assert "Within-neuron broadcast" not in source


def test_figure_one_derivative_names_its_readout_coordinate():
    source = (J / "scripts/credit_first_figures/build_framework.py").read_text()
    assert '[("δ", "u"), " = ∂L/∂", ("z", "u")]' in source


def test_tasks_are_defined_before_population_outcomes():
    source = (J / "main.tex").read_text()
    target = source.index(r"y=\sum_{b=0}^3c_b(-1)^b")
    severity = source.index(r"[1+(s-1)(1-c_b)]z_{bj}")
    gate = source.index(r"h_{up}=D_{up}^0/(D_{up}^0+g_{up}^{\rm I}a_b^{\rm I})")
    result = source.index("Resistance gating helped most under strong distractors.")
    assert target < severity < gate < result
    assert "parent nonlinearity and target changed together" in source


def test_capture_and_context_cancellation_are_different_definitions():
    source = (J / "main.tex").read_text()
    assert r"\mathcal C(A;\bm\delta)=\frac{\|P_A\bm\delta\|^2}{\|\bm\delta\|^2}" in source
    assert r"\|\sum_c p_c\bm v_c\|/\sum_c p_c\|\bm v_c\|" in source
    assert "one indicating no cancellation" in source


def test_supplement_explains_shared_path_and_nmse_normalizations():
    measured = (J / "supplementary/curated/si_09_measured.tex").read_text()
    assert "shorter of their two soma-to-contact path lengths" in measured
    assert "geometric-mean" in measured and "positive semidefiniteness" in measured
    guide = (J / "supplementary/curated/si_notation.tex").read_text()
    assert "held-out error" in guide and "training-mean predictor" in guide
    assert "main Fig.~7 fixes" in guide and "Fig.~S9 fixes sixteen terminals" in guide


def test_all_current_si_figures_have_a_caption_and_panel_letters():
    import fitz
    count = 0
    from tex_sources import tex_sources
    for path in tex_sources(J / "supplementary/supplementary.tex"):
        for block in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", path.read_text(), re.S):
            asset = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", block).group(1)
            assert r"\caption{" in block
            with fitz.open(J / "figures" / asset) as pdf:
                words = {w[4] for w in pdf[0].get_text("words")}
                assert "A" in words, asset
            count += 1
    from build_submission_bundle import SUPPLEMENTARY_FIGURES
    assert count == len(SUPPLEMENTARY_FIGURES) == 38
