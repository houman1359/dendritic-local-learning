from __future__ import annotations

import ast
import re
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
MAIN = JOURNAL / "main.tex"

MAIN_PANEL_INVENTORY = {
    "fig:framework": "abcdefg",
    "fig:branchconflict": "abcdefgh",
    "fig:subtreefactorial": "abcdef",
    "fig:prospective": "abcdefg",
    "fig:conductancecredit": "abcdefg",
    "fig:conductancepopulation": "abcdef",
    "fig:physicaldepth": "abcdefgh",
    "fig:topology": "abcdefgh",
    "fig:focal": "abcdefgh",
    "fig:boundary": "abcdef",
}

FIGURE_BLOCK = re.compile(
    r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL
)
PANEL_REF = re.compile(
    r"\\ref\{(?P<label>fig:[^}]+)\}"
    r"(?P<panels>[A-Ma-m](?:(?:--|,)[A-Ma-m])*)"
)
EQUATION_LABEL = re.compile(r"\\label\{(?P<label>eq:[^}]+)\}")
EQUATION_REF = re.compile(r"\\(?:ref|eqref)\{(?P<label>eq:[^}]+)\}")


def _expand_panels(specification: str) -> list[str]:
    panels: list[str] = []
    for item in specification.lower().split(","):
        if "--" in item:
            start, stop = item.split("--", maxsplit=1)
            panels.extend(chr(code) for code in range(ord(start), ord(stop) + 1))
        else:
            panels.append(item)
    return panels


def test_every_main_panel_is_cited_in_results_in_first_use_order() -> None:
    source = MAIN.read_text(encoding="utf-8")
    results = source.split(r"\section*{Results}", maxsplit=1)[1].split(
        r"\section*{Discussion}", maxsplit=1
    )[0]
    narrative = FIGURE_BLOCK.sub("", results)

    cited: dict[str, list[str]] = {label: [] for label in MAIN_PANEL_INVENTORY}
    for match in PANEL_REF.finditer(narrative):
        label = match.group("label")
        if label not in cited:
            continue
        for panel in _expand_panels(match.group("panels")):
            if panel not in cited[label]:
                cited[label].append(panel)

    for label, expected in MAIN_PANEL_INVENTORY.items():
        assert cited[label] == list(expected), (
            f"{label} first cites panels {''.join(cited[label])}; expected {expected}"
        )


def test_panel_reference_case_matches_current_artwork() -> None:
    source = MAIN.read_text(encoding="utf-8")
    lower_case_reference = re.search(r"\\ref\{fig:[^}]+\}[a-g]", source)
    assert lower_case_reference is None, (
        "main artwork and captions use uppercase panel labels; found a "
        "lowercase panel citation"
    )


def test_equations_are_not_referenced_before_their_display() -> None:
    source = MAIN.read_text(encoding="utf-8")
    label_positions = {
        match.group("label"): match.start() for match in EQUATION_LABEL.finditer(source)
    }
    premature = []
    for match in EQUATION_REF.finditer(source):
        label = match.group("label")
        if label in label_positions and match.start() < label_positions[label]:
            premature.append(label)
    assert not premature, f"equations referenced before display: {premature}"


# ---------------------------------------------------------------------------
# DECISIONS G1: literal supplementary citations, both directions.
# ---------------------------------------------------------------------------

SUPPLEMENT_AUX = JOURNAL / "supplementary" / "supplementary.aux"
SUPPLEMENT_SPEC = JOURNAL / "scripts" / "supplement_consolidation" / "specification.py"

SI_LABEL = re.compile(r"\\newlabel\{(?P<label>fig:si_[^}]+)\}\{\{S(?P<number>\d+)\}")
CITE_HEAD = re.compile(r"Figs?\.~")
CITE_FIRST = re.compile(r"S(\d+)(?:[A-H](?:,\s*[A-H])*)?(?:--[A-H])?")
CITE_RANGE = re.compile(r"--S(\d+)")
CITE_SEP = re.compile(r"(?:,\s*|[\s~]+and[\s~]+)(?=S\d)")


def _supplement_figure_numbers() -> set[int]:
    """Figure numbers that exist in the compiled supplement."""
    if SUPPLEMENT_AUX.is_file():
        aux = SUPPLEMENT_AUX.read_text(encoding="utf-8", errors="replace")
        numbers = {int(m.group("number")) for m in SI_LABEL.finditer(aux)}
        if numbers:
            return numbers
    # Fall back on the registry the supplement is built from.
    tree = ast.parse((JOURNAL / "scripts/build_submission_bundle.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "SUPPLEMENTARY_FIGURES"
                              for target in node.targets))
    figures = ast.literal_eval(assignment.value)
    assert len(set(figures)) == len(figures)
    count = len(figures)
    assert count, "cannot determine the supplement's figure inventory"
    return set(range(1, count + 1))


def test_supplement_inventory_without_compiled_aux(monkeypatch, tmp_path) -> None:
    # LaTeX also briefly empties this file at the start of a rebuild. The
    # fallback must read the actual tuple registry, not a retired API name.
    empty_aux = tmp_path / "supplementary.aux"
    empty_aux.touch()
    monkeypatch.setitem(globals(), "SUPPLEMENT_AUX", empty_aux)
    assert _supplement_figure_numbers() == set(range(1, 38))


def _cited_supplementary_figures(text: str) -> set[int]:
    """Every SI figure number cited literally from ``text``."""
    cited: set[int] = set()
    for head in CITE_HEAD.finditer(text):
        index = head.end()
        while True:
            first = CITE_FIRST.match(text, index)
            if first is None:
                break
            number = int(first.group(1))
            cited.add(number)
            index = first.end()
            span = CITE_RANGE.match(text, index)
            if span is not None:
                cited.update(range(number, int(span.group(1)) + 1))
                index = span.end()
            separator = CITE_SEP.match(text, index)
            if separator is None:
                break
            index = separator.end()
    return cited


def test_every_cited_supplementary_figure_exists() -> None:
    cited = _cited_supplementary_figures(MAIN.read_text(encoding="utf-8"))
    available = _supplement_figure_numbers()
    missing = sorted(cited - available)
    assert not missing, (
        f"main.tex cites supplementary figures that the supplement does not "
        f"contain: {['S%d' % n for n in missing]}"
    )


def test_every_supplementary_figure_is_cited_from_main() -> None:
    cited = _cited_supplementary_figures(MAIN.read_text(encoding="utf-8"))
    available = _supplement_figure_numbers()
    uncited = sorted(available - cited)
    assert not uncited, (
        f"supplementary figures never cited by number from main.tex: "
        f"{['S%d' % n for n in uncited]}"
    )


def test_main_uses_only_literal_supplementary_citations() -> None:
    """G1 bans xr-style cross-document references in ``main.tex``."""
    source = MAIN.read_text(encoding="utf-8")
    assert r"\ref{SI-fig:" not in source
    assert r"\ref{fig:si_" not in source
