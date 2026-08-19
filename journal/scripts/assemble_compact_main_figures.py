#!/usr/bin/env python3
"""Assemble one coherent vector asset for each numbered main figure.

The analysis scripts intentionally emit modular panel blocks.  This compositor
selects the load-bearing blocks, preserves their vector content, renumbers the
panels and writes the eight publication-facing PDFs.  Omitted audits remain in
Supplementary Information or in the modular provenance assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "figures" / "main"
SUPP = ROOT / "figures" / "supplementary"
FONT_REGULAR = Path("/usr/share/fonts/urw-base35/NimbusSans-Regular.otf")
FONT_BOLD = Path("/usr/share/fonts/urw-base35/NimbusSans-Bold.otf")

if not FONT_REGULAR.is_file() or not FONT_BOLD.is_file():
    raise FileNotFoundError(
        "The vector compositor requires embedded Nimbus Sans fonts from urw-base35"
    )


@dataclass(frozen=True)
class Panel:
    filename: str
    row: int
    col: int
    nrows: int
    ncols: int


@dataclass(frozen=True)
class Slot:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


SOURCE_START = {
    "figure_02_panels_G-O.pdf": "G",
    "figure_02_panels_P-Q.pdf": "P",
    "figure_05_panels_G-L.pdf": "G",
    "figure_05_panels_M-O.pdf": "M",
    "figure_05_panels_P-U.pdf": "P",
    "figure_06_panels_K-L.pdf": "K",
    "figure_07_panels_J-M.pdf": "J",
    "figure_08_panels_I-J.pdf": "I",
    "figure_08_panels_K-N.pdf": "K",
    "figure_08_panel_O.pdf": "O",
}


def panel(filename: str, letter: str, nrows: int, ncols: int) -> Panel:
    start = SOURCE_START.get(filename, "A")
    index = ord(letter.upper()) - ord(start)
    return Panel(filename, index // ncols, index % ncols, nrows, ncols)


def copy_page(source: str, destination: str) -> None:
    src = fitz.open(MAIN / source)
    out = fitz.open()
    out.insert_pdf(src)
    out.set_metadata({})
    out.save(MAIN / destination, garbage=4, deflate=True)


def _source_rect(page: fitz.Page, spec: Panel) -> fitz.Rect:
    width = page.rect.width / spec.ncols
    height = page.rect.height / spec.nrows
    return fitz.Rect(
        spec.col * width,
        spec.row * height,
        (spec.col + 1) * width,
        (spec.row + 1) * height,
    )


def _slot_rect(slot: Slot, rows: int, cols: int, width: float, height: float) -> fitz.Rect:
    margin = 4.0
    gutter = 3.0
    cell_w = (width - 2 * margin - (cols - 1) * gutter) / cols
    cell_h = (height - 2 * margin - (rows - 1) * gutter) / rows
    x0 = margin + slot.col * (cell_w + gutter)
    y0 = margin + slot.row * (cell_h + gutter)
    x1 = x0 + slot.colspan * cell_w + (slot.colspan - 1) * gutter
    y1 = y0 + slot.rowspan * cell_h + (slot.rowspan - 1) * gutter
    return fitz.Rect(x0, y0, x1, y1)


def compose(
    destination: Path,
    panels: list[Panel],
    titles: list[str],
    *,
    rows: int,
    cols: int,
    height: float,
    slots: list[Slot] | None = None,
) -> None:
    width = 518.4
    if slots is None:
        slots = [Slot(i // cols, i % cols) for i in range(len(panels))]
    if len(slots) != len(panels) or len(titles) != len(panels):
        raise ValueError("Each panel must have one destination slot")

    opened: dict[str, fitz.Document] = {}
    out = fitz.open()
    page_out = out.new_page(width=width, height=height)
    for index, (spec, slot, title) in enumerate(zip(panels, slots, titles, strict=True)):
        source = opened.setdefault(spec.filename, fitz.open(MAIN / spec.filename))
        source_page = source[0]
        target = _slot_rect(slot, rows, cols, width, height)
        clip = _source_rect(source_page, spec)
        page_out.show_pdf_page(
            target,
            source,
            0,
            clip=clip,
            keep_proportion=True,
            overlay=True,
        )

        # Redraw a uniform publication heading without rasterizing the plot.
        # When a wide source panel is fitted into a taller slot its old title
        # starts below the slot boundary, so clear both the fitted-source and
        # destination heading bands.
        scale = min(target.width / clip.width, target.height / clip.height)
        shown_x = target.x0 + (target.width - clip.width * scale) / 2
        shown_y = target.y0 + (target.height - clip.height * scale) / 2
        shown_width = clip.width * scale
        source_heading_box = fitz.Rect(
            shown_x, shown_y, shown_x + shown_width, shown_y + 28
        )
        page_out.draw_rect(
            source_heading_box, color=None, fill=(1, 1, 1), overlay=True
        )
        heading_box = fitz.Rect(target.x0, target.y0, target.x1, target.y0 + 28)
        page_out.draw_rect(heading_box, color=None, fill=(1, 1, 1), overlay=True)
        page_out.insert_text(
            fitz.Point(target.x0 + 3, target.y0 + 12),
            chr(ord("A") + index),
            fontsize=9.5,
            fontname="dllbold",
            fontfile=str(FONT_BOLD),
            color=(0.06, 0.06, 0.06),
            overlay=True,
        )
        page_out.insert_textbox(
            fitz.Rect(target.x0 + 22, target.y0 + 3, target.x1 - 2, target.y0 + 16),
            title,
            fontsize=7.3,
            fontname="dllregular",
            fontfile=str(FONT_REGULAR),
            color=(0.08, 0.08, 0.08),
            align=fitz.TEXT_ALIGN_CENTER,
            overlay=True,
        )

    # Clear the narrow inter-row gutters after all placements.  Matplotlib
    # panel letters can sit a few points outside their nominal grid cell and
    # otherwise survive as tiny fragments between composed rows.
    margin = 4.0
    gutter = 3.0
    cell_height = (height - 2 * margin - (rows - 1) * gutter) / rows
    for row in range(1, rows):
        prior_bottom = margin + row * cell_height + (row - 1) * gutter
        next_top = prior_bottom + gutter
        page_out.draw_rect(
            fitz.Rect(0, prior_bottom - 2, width, next_top + 2),
            color=None,
            fill=(1, 1, 1),
            overlay=True,
        )
    page_out.draw_rect(
        fitz.Rect(0, height - margin - 2, width, height),
        color=None,
        fill=(1, 1, 1),
        overlay=True,
    )

    out.set_metadata({})
    out.save(destination, garbage=4, deflate=True)


def main() -> None:
    # Figures 1, 3 and 4 were already designed as single coherent canvases.
    copy_page("figure_01_panels_A-E.pdf", "figure_01.pdf")
    copy_page("figure_03_panels_A-F.pdf", "figure_03.pdf")
    copy_page("figure_04_panels_A-I.pdf", "figure_04.pdf")

    compose(
        MAIN / "figure_02.pdf",
        [
            panel("figure_02_panels_A-F.pdf", "B", 2, 3),
            panel("figure_02_panels_A-F.pdf", "C", 2, 3),
            panel("figure_02_panels_A-F.pdf", "D", 2, 3),
            panel("figure_02_panels_G-O.pdf", "G", 3, 3),
            panel("figure_02_panels_G-O.pdf", "J", 3, 3),
            panel("figure_02_panels_G-O.pdf", "M", 3, 3),
            panel("figure_02_panels_P-Q.pdf", "P", 1, 2),
            panel("figure_02_panels_P-Q.pdf", "Q", 1, 2),
        ],
        [
            "Neuron-indexed feedback", "Gradient alignment", "Exact error transport",
            "MNIST identity gain", "Matched routing", "Within-neuron routing",
            "Second-dataset ladder", "Replicated bottleneck",
        ],
        rows=3,
        cols=3,
        height=458,
    )

    compose(
        MAIN / "figure_05.pdf",
        [
            *[panel("figure_05_panels_A-F.pdf", letter, 2, 3) for letter in "ABCDEF"],
            panel("figure_05_panels_G-L.pdf", "G", 2, 3),
            panel("figure_05_panels_G-L.pdf", "H", 2, 3),
            panel("figure_05_panels_G-L.pdf", "J", 2, 3),
            panel("figure_05_panels_G-L.pdf", "K", 2, 3),
        ],
        [
            "Matched-resource depth", "Nested divisive task", "Backprop depth test",
            "Prespecified contrasts", "Local credit transport", "Divisive control",
            "Architecture controls", "Serial composition", "Credit-coordinate ladder",
            "BP--local decomposition",
        ],
        rows=3,
        cols=4,
        height=430,
    )

    compose(
        MAIN / "figure_06.pdf",
        [
            panel("figure_06_panels_A-J.pdf", letter, 2, 3)
            for letter in "ABCDEF"
        ]
        + [
            panel("figure_06_panels_K-L.pdf", "K", 1, 2),
            panel("figure_06_panels_K-L.pdf", "L", 1, 2),
        ],
        [
            "Reconstructed tree", "Ancestry addresses", "Sparse route capacity",
            "Model-field controls", "Reciprocal cable field", "Topology controls",
            "Wire efficiency at eight channels", "Wiring-normalized capture",
        ],
        rows=3,
        cols=3,
        height=430,
    )

    compose(
        MAIN / "figure_07.pdf",
        [
            panel("figure_07_panels_A-I.pdf", letter, 2, 3)
            for letter in "ABCDEF"
        ]
        + [
            panel("figure_07_panels_J-M.pdf", letter, 1, 4)
            for letter in "JKLM"
        ],
        [
            "Matched focal shunt", "Tree-relation selectivity", "Within-cell controls",
            "Dose response", "Adjoint transport", "Electrotonic limit",
            "Active channels", "Dose response", "Cellwise contrast", "Signed outcome",
        ],
        rows=3,
        cols=4,
        height=430,
    )

    # Figure 8 ends with the unifying phase plane, which spans two columns.
    compose(
        MAIN / "figure_08.pdf",
        [
            *[panel("figure_08_panels_A-H.pdf", letter, 2, 3) for letter in "ABCDEF"],
            panel("figure_08_panels_I-J.pdf", "I", 1, 2),
            panel("figure_08_panels_I-J.pdf", "J", 1, 2),
            panel("../supplementary/figure_S17_panels_A-D.pdf", "B", 1, 4),
            panel("../supplementary/figure_S17_panels_A-D.pdf", "C", 1, 4),
            panel("figure_08_panel_O.pdf", "O", 1, 1),
        ],
        [
            "Measured cohort", "Structure--function boundary", "Task-field capture",
            "Held-out learning", "Controlled alignment", "Capture vs progress",
            "Full-tree task", "Anatomy boundary", "Signed neuron pairs", "Signed mode",
            "Alignment x bandwidth",
        ],
        rows=3,
        cols=4,
        height=458,
        slots=[
            Slot(0, 0), Slot(0, 1), Slot(0, 2), Slot(0, 3),
            Slot(1, 0), Slot(1, 1), Slot(1, 2), Slot(1, 3),
            Slot(2, 0), Slot(2, 1), Slot(2, 2, colspan=2),
        ],
    )

    # The physical-depth dose response and second-hierarchy replication remain
    # visible as one compact supplementary figure rather than continued main
    # panels.
    compose(
        SUPP / "figure_S18_panels_A-K.pdf",
        [
            panel("figure_05_panels_G-L.pdf", "I", 2, 3),
            panel("figure_05_panels_G-L.pdf", "L", 2, 3),
            *[panel("figure_05_panels_M-O.pdf", letter, 1, 3) for letter in "MNO"],
            *[panel("figure_05_panels_P-U.pdf", letter, 2, 3) for letter in "PQRSTU"],
        ],
        [
            "Serial vs grouped star", "Flexible point controls", "Alignment dose",
            "Depth benefit", "Dose contrasts", "H3 models", "H3 outcomes",
            "H3 interaction", "H2 hierarchy", "H2 local learning", "H2 contrasts",
        ],
        rows=3,
        cols=4,
        height=458,
    )

    compose(
        SUPP / "figure_S19_panels_A-I.pdf",
        [panel("figure_02_panels_G-O.pdf", letter, 3, 3) for letter in "GHIJKLMNO"],
        [
            "MNIST identity", "Noise-task identity", "Exact vs BP", "Ownership",
            "Clean exact vs BP", "Fixed-budget depth", "Two-stream learning",
            "Capture vs progress", "Context forgetting",
        ],
        rows=3,
        cols=3,
        height=458,
    )

    print("Assembled eight compact main figures and Supplementary Figures S18--S19.")


if __name__ == "__main__":
    main()
