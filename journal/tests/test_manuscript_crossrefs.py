from __future__ import annotations

import re
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
MAIN = JOURNAL / "main.tex"

MAIN_PANEL_INVENTORY = {
    "fig:framework": "abcde",
    "fig:feedback": "abcdefg",
    "fig:creditphase": "abcdefgh",
    "fig:branchconflict": "abcdef",
    "fig:subtreefactorial": "abcdefg",
    "fig:physicaldepth": "abcdefg",
    "fig:topology": "abcdefg",
    "fig:focal": "abcdefg",
    "fig:boundary": "abcdefg",
}

FIGURE_BLOCK = re.compile(
    r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL
)
PANEL_REF = re.compile(
    r"\\ref\{(?P<label>fig:[^}]+)\}"
    r"(?P<panels>[A-Ha-h](?:(?:--|,)[A-Ha-h])*)"
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
