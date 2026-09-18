"""Caption artifacts must use the same definitions as the compiled manuscript."""
import ast
import json
from pathlib import Path
import re

import pytest

JOURNAL = Path(__file__).resolve().parents[1]


def manuscript_caption(label):
    source = (JOURNAL / "main.tex").read_text()
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", source, re.S)
    figure = next(block for block in figures if "\\label{" + label + "}" in block)
    return figure[figure.index(r"\caption{"):figure.index(r"\label{")].strip()


@pytest.mark.parametrize("builder,constant,label", [
    ("build_ancestry.py", "CAPTION", "fig:subtreefactorial"),
    ("build_anatomy.py", "CAPTION", "fig:topology"),
    ("build_restored_main.py", "FIG4_CAPTION", "fig:prospective"),
    ("build_restored_main.py", "F6_CAPTION", "fig:physicaldepth"),
])
def test_current_caption_constant_matches_manuscript(builder, constant, label):
    caption = manuscript_caption(label)
    tree = ast.parse((JOURNAL / "scripts/credit_first_figures" / builder).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == constant
                              for target in node.targets))
    assert ast.literal_eval(assignment.value).strip() == caption


def test_context_task_caption_distinguishes_task_from_image_source():
    cfg = json.loads((JOURNAL / "configs/path_necessity/"
                     "credit_conflict_fashion_confirmatory.json").read_text())
    caption = manuscript_caption("fig:branchconflict")
    panel_a = caption.split(r"\textbf{A},", 1)[1].split(r"\textbf{B},", 1)[0]
    assert cfg["dataset"]["name"] == "FashionMNIST"
    assert cfg["dataset"]["classes"] == [0, 6]
    assert cfg["task"]["selected_view_defines_label"]
    assert "Context-gated binary branch selection" in panel_a
    assert "Fashion-MNIST classes 0 and 6" in panel_a
    assert "$B=4$ schematic" in panel_a
    assert "$x_y$ and $x_{1-y}$" in panel_a
    assert "target- and opposite-class inputs" in panel_a


def test_context_task_caption_keeps_synthetic_forgetting_cohort_separate():
    caption = manuscript_caption("fig:branchconflict")
    panel_e = caption.split(r"\textbf{E},", 1)[1].split(r"\textbf{F--H},", 1)[0]
    assert "Separate synthetic two-stream task" in panel_e
    assert "mixed-context acquisition and context-1-only training" in panel_e
    assert "$n=10$ paired" in panel_e
    assert "Fashion-MNIST" not in panel_e
    assert "binary chance (50\\%)" in caption


def test_physical_depth_caption_snapshot_matches_manuscript():
    snapshot = JOURNAL / "figures/provenance/structure_restoration_20260908/figure_06_caption.md"
    assert snapshot.read_text().strip() == manuscript_caption("fig:physicaldepth")
