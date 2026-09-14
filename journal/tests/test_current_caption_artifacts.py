"""Caption artifacts must use the same definitions as the compiled manuscript."""
import ast
from pathlib import Path
import re

import pytest

JOURNAL = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("builder,label", [
    ("build_ancestry.py", "fig:subtreefactorial"),
    ("build_anatomy.py", "fig:topology"),
])
def test_current_caption_constant_matches_manuscript(builder, label):
    source = (JOURNAL / "main.tex").read_text()
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", source, re.S)
    figure = next(block for block in figures if "\\label{" + label + "}" in block)
    caption = figure[figure.index(r"\caption{"):figure.index(r"\label{")].strip()
    tree = ast.parse((JOURNAL / "scripts/credit_first_figures" / builder).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "CAPTION"
                              for target in node.targets))
    assert ast.literal_eval(assignment.value).strip() == caption
