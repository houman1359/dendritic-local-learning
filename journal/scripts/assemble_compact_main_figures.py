#!/usr/bin/env python3
"""Assemble one coherent vector asset for each numbered main figure.

The analysis scripts intentionally emit modular panel blocks.  This compositor
selects the load-bearing blocks, preserves their vector content, renumbers the
panels and writes the nine publication-facing PDFs.  Omitted audits remain in
Supplementary Information or in the modular provenance assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "figures" / "main"
SUPP = ROOT / "figures" / "supplementary"
COMPONENTS = ROOT / "figures" / "components"

# A main figure that has been rebuilt as ONE native full-width canvas is
# emitted verbatim: page in, page out, scale 1.0.  Scaling a pre-rendered
# sub-block into a grid slot is what destroyed the journal type scale (each
# block took a different scale factor, so one nominal 7.6 pt tick label
# printed at anything between 4.3 and 6.9 pt across the figure set).  Drop
# ``figures/components/main_figure_NN_native.pdf`` next to this script's
# other components and figure NN takes the unscaled path instead.
NATIVE_TEMPLATE = "main_figure_{:02d}_native.pdf"
FONT_REGULAR = Path("/usr/share/fonts/urw-base35/NimbusSans-Regular.otf")
FONT_BOLD = Path("/usr/share/fonts/urw-base35/NimbusSans-Bold.otf")

# Composed sheets carry the same panel-letter size as every native sheet, so a
# reader paging through the supplement meets one lettering scale throughout.
PANEL_LETTER_PT = 10.5

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
    erase_heading: bool = True
    erase_phrases: tuple[str, ...] = ()
    # Re-expand the measured clip top by this many points AFTER the
    # heading-band clamp, for the rare source panel whose rotated y label
    # rises into its own title row (the compose-time white-out still covers
    # any re-included heading spans, which are erased span-by-span).
    pad_top_pt: float = 0.0


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
    "figure_07_panels_K-L.pdf": "K",
    "figure_08_panels_J-M.pdf": "J",
    "figure_09_panels_I-J.pdf": "I",
    "figure_09_panels_K-N.pdf": "K",
    "figure_09_panel_O.pdf": "O",
}


def panel(
    filename: str,
    letter: str,
    nrows: int,
    ncols: int,
    *,
    erase_heading: bool = True,
    erase_phrases: tuple[str, ...] = (),
    pad_top_pt: float = 0.0,
) -> Panel:
    start = SOURCE_START.get(filename, "A")
    index = ord(letter.upper()) - ord(start)
    return Panel(
        filename,
        index // ncols,
        index % ncols,
        nrows,
        ncols,
        erase_heading,
        erase_phrases,
        pad_top_pt,
    )


def copy_page(source: str, destination: str, *,
              keep_metadata: bool = False) -> None:
    src = fitz.open(MAIN / source)
    out = fitz.open()
    out.insert_pdf(src)
    metadata: dict[str, str] = {}
    if keep_metadata:
        # A native canvas carries its panel-geometry manifest in the PDF
        # keywords; ``figure_canvas.audit_native_pdf`` needs it to verify
        # per-row panel heights, so it must survive the copy.
        keywords = (src.metadata or {}).get("keywords") or ""
        if keywords.strip():
            metadata["keywords"] = keywords
    out.set_metadata(metadata)
    out.save(MAIN / destination, garbage=4, deflate=True, no_new_id=True)


def native_component(number: int) -> Path:
    """Path a natively built full-width canvas must occupy to be used."""
    return COMPONENTS / NATIVE_TEMPLATE.format(int(number))


def emit_native(number: int) -> bool:
    """Emit ``figure_NN.pdf`` from its native canvas; report whether it ran."""
    source = native_component(number)
    if not source.is_file():
        return False
    copy_page(f"../components/{source.name}", f"figure_{int(number):02d}.pdf",
              keep_metadata=True)
    print(f"  figure_{int(number):02d}.pdf <- {source.name} (native, scale 1.0)")
    return True


def _destination_figure_number(destination: Path) -> int | None:
    """The main-figure number a compose destination stands for, if any."""
    destination = Path(destination)
    if destination.parent != MAIN:
        return None
    stem = destination.stem
    if not stem.startswith("figure_"):
        return None
    tail = stem[len("figure_"):]
    return int(tail) if tail.isdigit() else None


_RASTER_ZOOM = 150.0 / 72.0
_INK_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_BOUND_CACHE: dict[tuple[str, int, int], tuple[list[float], list[float]]] = {}
_CONTENT_CACHE: dict[Panel, fitz.Rect] = {}


def _ink_profiles(page: fitz.Page, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Row/column ink occupancy of a source page at raster resolution."""
    if key not in _INK_CACHE:
        matrix = fitz.Matrix(_RASTER_ZOOM, _RASTER_ZOOM)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, len(img) // pix.height)[:, : pix.width]
        ink = img < 250
        _INK_CACHE[key] = (ink.any(axis=1), ink.any(axis=0))
    return _INK_CACHE[key]


def _snap_boundary(
    nominal: float,
    has_ink: np.ndarray,
    *,
    prefer_forward: bool,
    window_pt: float = 16.0,
    limit: float | None = None,
) -> float:
    """Move an internal grid cut to the nearest blank corridor.

    Uniform-grid cuts land mid-glyph whenever a panel's x label, y label or
    letter overhangs its nominal cell.  Snapping every cut into an ink-free
    corridor keeps each glyph whole and assigns it to exactly one panel.
    Row cuts pass ``prefer_forward=True``: the first corridor *below* the
    nominal cut wins, so a bottom x label stays with the panel above it.
    Column cuts pass ``prefer_forward=False`` and take the nearest corridor,
    which sits in the inter-panel gutter on whichever side of the overhanging
    label (x label from the left panel, y label from the right panel).
    """
    n = len(has_ink)
    pixel = int(round(nominal * _RASTER_ZOOM))
    if pixel <= 0 or pixel >= n:
        return nominal
    if not has_ink[pixel] and limit is None:
        return nominal
    window = int(window_pt * _RASTER_ZOOM)
    lo = max(0, pixel - window)
    hi = min(n, pixel + window + 1)
    # A corridor must be at least ~2.5 pt of contiguous whitespace so the cut
    # cannot thread the gap between two letters or two words of one label.
    min_run = max(2, int(round(2.5 * _RASTER_ZOOM)))
    corridors: list[tuple[int, int]] = []  # (center pixel, signed offset)
    run_start = None
    for index in range(lo, hi + 1):
        blank = index < hi and not has_ink[index]
        if blank and run_start is None:
            run_start = index
        elif not blank and run_start is not None:
            if index - run_start >= min_run:
                center = (run_start + index - 1) // 2
                corridors.append((center, center - pixel))
            run_start = None
    if limit is not None:
        # The next row's panel letters start at ``limit``: the cut must stay
        # above them, or a letter band migrates into the panel above.
        limit_pixel = int(limit * _RASTER_ZOOM)
        corridors = [c for c in corridors if c[0] < limit_pixel]
        if not has_ink[pixel] and pixel < limit_pixel:
            return nominal
    if not corridors:
        return nominal
    if prefer_forward:
        forward = [c for c in corridors if c[1] > 0]
        chosen = min(forward, key=lambda c: c[1]) if forward else max(
            corridors, key=lambda c: c[1]
        )
    else:
        chosen = min(corridors, key=lambda c: abs(c[1]))
    return chosen[0] / _RASTER_ZOOM


def _panel_letter_rects(
    page: fitz.Page, key: str, nrows: int, ncols: int
) -> dict[tuple[int, int], fitz.Rect]:
    """Locate each panel's bold letter to anchor the grid boundaries."""
    start = SOURCE_START.get(Path(key).name, "A")
    spans: list[tuple[str, fitz.Rect]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if len(text) == 1 and text.isupper() and span.get("size", 0) >= 9.5:
                    spans.append((text, fitz.Rect(span["bbox"])))
    cell_w = page.rect.width / ncols
    cell_h = page.rect.height / nrows
    positions: dict[tuple[int, int], fitz.Rect] = {}
    for row in range(nrows):
        for column in range(ncols):
            letter = chr(ord(start) + row * ncols + column)
            origin = (column * cell_w, row * cell_h)
            candidates = [
                rect
                for text, rect in spans
                if text == letter
                and abs(rect.x0 - origin[0]) < 0.6 * cell_w
                and abs(rect.y0 - origin[1]) < 0.6 * cell_h
            ]
            if candidates:
                positions[(row, column)] = min(
                    candidates,
                    key=lambda rect: (rect.x0 - origin[0]) ** 2
                    + (rect.y0 - origin[1]) ** 2,
                )
    return positions


def _grid_bounds(
    page: fitz.Page, key: str, nrows: int, ncols: int
) -> tuple[list[float], list[float]]:
    """Ink-aware row/column boundaries for one source page's panel grid."""
    cache_key = (key, nrows, ncols)
    if cache_key not in _BOUND_CACHE:
        row_ink, col_ink = _ink_profiles(page, key)
        letters = _panel_letter_rects(page, key, nrows, ncols)
        height = page.rect.height
        width = page.rect.width
        rows = [0.0]
        for index in range(1, nrows):
            tops = [
                letters[(index, column)].y0
                for column in range(ncols)
                if (index, column) in letters
            ]
            limit = (min(tops) - 1.0) if tops else None
            cut = _snap_boundary(
                index * height / nrows,
                row_ink,
                prefer_forward=True,
                limit=limit,
            )
            if limit is not None:
                # Never cut through or below the next row's panel letters.
                cut = min(cut, limit)
            rows.append(cut)
        rows.append(height)
        cols = [0.0]
        for index in range(1, ncols):
            cols.append(
                _snap_boundary(index * width / ncols, col_ink, prefer_forward=False)
            )
        cols.append(width)
        _BOUND_CACHE[cache_key] = (rows, cols)
    return _BOUND_CACHE[cache_key]


def _source_rect(page: fitz.Page, spec: Panel) -> fitz.Rect:
    rows, cols = _grid_bounds(page, spec.filename, spec.nrows, spec.ncols)
    return fitz.Rect(
        cols[spec.col],
        rows[spec.row],
        cols[spec.col + 1],
        rows[spec.row + 1],
    )


def _content_rect(page: fitz.Page, spec: Panel, cell: fitz.Rect) -> fitz.Rect:
    """Return the visible-content bounds of one modular source panel.

    The analysis sheets deliberately reserve generous cell margins.  Carrying
    those margins into a multi-panel journal figure made the actual axes small
    and produced the large white bands visible in the previous assembly.  We
    rasterize only to *measure* the ink bounds; the content copied below stays
    vector.  Source letters/titles are ignored because the compositor supplies
    one consistent heading band.
    """
    if spec in _CONTENT_CACHE:
        return _CONTENT_CACHE[spec]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(_RASTER_ZOOM, _RASTER_ZOOM),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=cell,
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, len(img) // pix.height)[:, : pix.width]
    ink = img < 247

    heading_spans = _title_spans(page, cell) if spec.erase_heading else []
    ignored = list(heading_spans)
    for phrase in spec.erase_phrases:
        ignored.extend(
            rect for rect in page.search_for(phrase) if cell.intersects(rect)
        )
    for rect in ignored:
        local = rect & cell
        x0 = max(0, int(np.floor((local.x0 - cell.x0 - 2.0) * _RASTER_ZOOM)))
        x1 = min(
            pix.width,
            int(np.ceil((local.x1 - cell.x0 + 2.0) * _RASTER_ZOOM)),
        )
        y0 = max(0, int(np.floor((local.y0 - cell.y0 - 1.5) * _RASTER_ZOOM)))
        y1 = min(
            pix.height,
            int(np.ceil((local.y1 - cell.y0 + 1.5) * _RASTER_ZOOM)),
        )
        ink[y0:y1, x0:x1] = False

    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        result = cell
    else:
        pad = 2.5
        result = fitz.Rect(
            max(cell.x0, cell.x0 + xs.min() / _RASTER_ZOOM - pad),
            max(cell.y0, cell.y0 + ys.min() / _RASTER_ZOOM - pad),
            min(cell.x1, cell.x0 + (xs.max() + 1) / _RASTER_ZOOM + pad),
            min(cell.y1, cell.y0 + (ys.max() + 1) / _RASTER_ZOOM + pad),
        )
        if heading_spans and spec.erase_heading:
            # Do not let antialiased fragments of the removed source heading
            # survive on the crop edge.  Plot ink starts well below this band;
            # rotated y labels were deliberately excluded by _title_spans.
            heading_bottom = max(rect.y1 for rect in heading_spans)
            result.y0 = max(result.y0, min(result.y1 - 1.0, heading_bottom + 1.5))
        if spec.pad_top_pt:
            result.y0 = max(cell.y0, result.y0 - spec.pad_top_pt)
    _CONTENT_CACHE[spec] = result
    return result


def _title_spans(page: fitz.Page, clip: fitz.Rect) -> list[fitz.Rect]:
    """Text boxes of the source panel's letter/title inside ``clip``'s top.

    Panel letters (10.5 pt bold) and titles (8.8 pt regular) are the only
    journal-style horizontal text at or above 8.2 pt in the top strip of a
    panel; tick labels (7.6 pt) and annotations (7.2 pt) stay below the
    threshold, and rotated y labels are excluded by line direction.  Each
    span is erased individually: a single union band would also swallow the
    top of a long y label that shares the strip's height in a different
    column.
    """
    strip = fitz.Rect(clip.x0, clip.y0, clip.x1, min(clip.y1, clip.y0 + 30.0))
    rects: list[fitz.Rect] = []
    layout = page.get_text("dict", clip=strip)
    for block in layout.get("blocks", []):
        for line in block.get("lines", []):
            direction = line.get("dir", (1.0, 0.0))
            if abs(direction[0]) < 0.9:
                # Rotated text is a y label whose top enters the strip, not a
                # heading; erasing it would truncate the label.
                continue
            for span in line.get("spans", []):
                if span.get("size", 0.0) < 8.2:
                    continue
                span_rect = fitz.Rect(span["bbox"])
                if span_rect.y0 < clip.y0 - 1.0:
                    continue
                rects.append(span_rect)
    return rects


def _slot_rect(
    slot: Slot,
    rows: int,
    cols: int,
    width: float,
    height: float,
    row_heights: list[float] | None = None,
) -> fitz.Rect:
    margin = 2.5
    gutter = 7.0
    cell_w = (width - 2 * margin - (cols - 1) * gutter) / cols
    x0 = margin + slot.col * (cell_w + gutter)
    x1 = x0 + slot.colspan * cell_w + (slot.colspan - 1) * gutter
    if row_heights is None:
        cell_h = (height - 2 * margin - (rows - 1) * gutter) / rows
        y0 = margin + slot.row * (cell_h + gutter)
        y1 = y0 + slot.rowspan * cell_h + (slot.rowspan - 1) * gutter
    else:
        # Explicit per-row heights: rows holding source panels of different
        # aspect need different heights, or the sheet pays every row at the
        # tallest panel's height and overflows the LaTeX text block.
        offsets = [margin]
        for row_height in row_heights:
            offsets.append(offsets[-1] + row_height + gutter)
        y0 = offsets[slot.row]
        y1 = offsets[slot.row + slot.rowspan] - gutter
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
    row_heights: list[float] | None = None,
    gutter_clear_above: float = 10.0,
) -> None:
    number = _destination_figure_number(destination)
    if number is not None and emit_native(number):
        # Rebuilt natively: keep the sub-block recipe below as the documented
        # fallback, but never scale it into a grid slot again.
        return
    width = 518.4
    if slots is None:
        slots = [Slot(i // cols, i % cols) for i in range(len(panels))]
    if len(slots) != len(panels) or len(titles) != len(panels):
        raise ValueError("Each panel must have one destination slot")
    if row_heights is not None:
        if len(row_heights) != rows:
            raise ValueError("row_heights must name one height per grid row")
        # The sheet height is fully determined by the explicit rows.
        height = 2 * 2.5 + sum(row_heights) + (rows - 1) * 7.0

    opened: dict[str, fitz.Document] = {}
    out = fitz.open()
    page_out = out.new_page(width=width, height=height)
    for index, (spec, slot, title) in enumerate(
        zip(panels, slots, titles, strict=True)
    ):
        # Sources normally live in figures/main; the reduced regular-tree
        # boundary sheet recomposes panels of an inherited supplementary
        # asset, so fall back to figures/supplementary for those.
        source_path = MAIN / spec.filename
        if not source_path.is_file():
            source_path = SUPP / spec.filename
        source = opened.setdefault(spec.filename, fitz.open(source_path))
        source_page = source[0]
        target = _slot_rect(slot, rows, cols, width, height, row_heights)
        cell = _source_rect(source_page, spec)
        clip = _content_rect(source_page, spec, cell)
        content_target = fitz.Rect(
            target.x0,
            target.y0 + 16.0,
            target.x1,
            target.y1,
        )
        page_out.show_pdf_page(
            content_target,
            source,
            0,
            clip=clip,
            keep_proportion=True,
            overlay=True,
        )

        # Redraw a uniform publication heading without rasterizing the plot.
        # Erase exactly the source panel's own letter/title band (measured
        # from the embedded text) rather than a fixed-height strip, so top
        # tick labels and top data markers survive the recomposition.
        scale = min(
            content_target.width / clip.width,
            content_target.height / clip.height,
        )
        shown_x = content_target.x0 + (content_target.width - clip.width * scale) / 2
        shown_y = content_target.y0 + (content_target.height - clip.height * scale) / 2
        if spec.erase_heading:
            for span_rect in _title_spans(source_page, cell):
                if not clip.intersects(span_rect):
                    continue
                heading_box = fitz.Rect(
                    max(content_target.x0, shown_x + (span_rect.x0 - 2.0 - clip.x0) * scale),
                    content_target.y0,
                    min(content_target.x1, shown_x + (span_rect.x1 + 2.0 - clip.x0) * scale),
                    shown_y + (span_rect.y1 + 1.5 - clip.y0) * scale,
                )
                page_out.draw_rect(
                    heading_box, color=None, fill=(1, 1, 1), overlay=True
                )
        # A few legacy modular panels carry letter-dependent cross-references
        # (for example, “key as in A”).  Remove those explicitly when the
        # final compositor supplies a self-contained key elsewhere; never
        # white out a broad fixed band that could cover data.
        for phrase in spec.erase_phrases:
            for phrase_rect in source_page.search_for(phrase):
                if not cell.intersects(phrase_rect) or not clip.intersects(phrase_rect):
                    continue
                phrase_box = fitz.Rect(
                    shown_x + (phrase_rect.x0 - 1.5 - clip.x0) * scale,
                    shown_y + (phrase_rect.y0 - 1.0 - clip.y0) * scale,
                    shown_x + (phrase_rect.x1 + 1.5 - clip.x0) * scale,
                    shown_y + (phrase_rect.y1 + 1.0 - clip.y0) * scale,
                )
                page_out.draw_rect(
                    phrase_box, color=None, fill=(1, 1, 1), overlay=True
                )
        page_out.insert_text(
            fitz.Point(target.x0 + 1.5, target.y0 + 11.0),
            chr(ord("A") + index),
            fontsize=PANEL_LETTER_PT,
            fontname="dllbold",
            fontfile=str(FONT_BOLD),
            color=(0.06, 0.06, 0.06),
            overlay=True,
        )
        page_out.insert_textbox(
            fitz.Rect(target.x0 + 19, target.y0 + 2, target.x1 - 1, target.y0 + 14),
            title,
            fontsize=7.6,
            fontname="dllregular",
            fontfile=str(FONT_REGULAR),
            color=(0.08, 0.08, 0.08),
            align=fitz.TEXT_ALIGN_CENTER,
            overlay=True,
        )

    # Ink-aware source clipping keeps every glyph whole inside its own panel,
    # so no destination gutter or margin whiteout is needed: those bands used
    # to cut bottom x labels mid-glyph while hiding leaked letter fragments.
    del gutter_clear_above

    out.set_metadata({})
    out.save(destination, garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    # Figure 1 is already designed as a single coherent canvas.
    if not emit_native(1):
        copy_page("figure_01_panels_A-E.pdf", "figure_01.pdf")

    compose(
        MAIN / "figure_02.pdf",
        [
            panel("figure_02_panels_A-F.pdf", "B", 2, 3),
            panel("figure_02_panels_A-F.pdf", "C", 2, 3),
            panel("figure_02_panels_G-O.pdf", "G", 3, 3),
            panel("figure_02_panels_G-O.pdf", "J", 3, 3),
            panel("figure_02_panels_G-O.pdf", "M", 3, 3),
            panel("figure_02_panels_P-Q.pdf", "P", 1, 2),
            panel("figure_02_panels_P-Q.pdf", "Q", 1, 2),
        ],
        [
            "Neuron-indexed feedback",
            "Gradient alignment",
            "Identity gain across depth",
            "Matched ownership",
            "Within-tree credit reversal",
            "Fashion-MNIST ladder",
            "Replicated bottleneck",
        ],
        rows=3,
        cols=6,
        height=392,
        slots=[
            Slot(0, 0, colspan=2),
            Slot(0, 2, colspan=2),
            Slot(0, 4, colspan=2),
            Slot(1, 0, colspan=3),
            Slot(1, 3, colspan=3),
            Slot(2, 0, colspan=3),
            Slot(2, 3, colspan=3),
        ],
        row_heights=[118.0, 124.0, 126.0],
    )

    # Keep the route dictionary adjacent to the measurements, but at the same
    # visual weight as the evidence rather than as a slide-like banner.
    compose(
        MAIN / "figure_03.pdf",
        [
            panel("figure_03_panels_A-F.pdf", "A", 2, 3),
            panel("figure_03_panels_A-F.pdf", "B", 2, 3),
            panel("figure_03_panels_A-F.pdf", "C", 2, 3),
            panel("figure_03_panels_A-F.pdf", "D", 2, 3),
            panel("figure_03_panels_A-F.pdf", "E", 2, 3),
            panel("figure_03_panels_A-F.pdf", "F", 2, 3),
        ],
        [
            "Address bandwidth",
            "Learning across bandwidth",
            "Best-control contrast",
            "Task–topology alignment",
            "Representation match",
            "Capture and learning",
        ],
        rows=2,
        cols=3,
        height=325,
        row_heights=[145.0, 161.0],
    )

    # The alignment-by-bandwidth plane is the theory's most compact boundary
    # map, so it closes Figure 4 at full width rather than competing with ten
    # heterogeneous panels in the biological-boundary figure.  Same-span and
    # conditioning diagnostics remain in Supplementary Figure S14.
    compose(
        MAIN / "figure_04.pdf",
        [
            panel("figure_04_panels_A-I.pdf", "A", 3, 3),
            panel("figure_04_panels_A-I.pdf", "B", 3, 3),
            panel("figure_04_panels_A-I.pdf", "I", 3, 3),
            panel("figure_04_panels_A-I.pdf", "D", 3, 3),
            panel("figure_04_panels_A-I.pdf", "E", 3, 3),
            panel("figure_04_panels_A-I.pdf", "F", 3, 3),
            panel(
                "../components/main_phase_plane_clean.pdf",
                "A",
                1,
                1,
                erase_heading=False,
            ),
        ],
        [
            "Credit-operator utility",
            "Spectral alignment",
            "Predictive utility",
            "Route-resolution crossover",
            "Projection boundary",
            "Reliability gains",
            "Alignment × bandwidth",
        ],
        rows=3,
        cols=6,
        height=455,
        slots=[
            Slot(0, 0, colspan=2),
            Slot(0, 2, colspan=2),
            Slot(0, 4, colspan=2),
            Slot(1, 0, colspan=2),
            Slot(1, 2, colspan=2),
            Slot(1, 4, colspan=2),
            Slot(2, 0, colspan=6),
        ],
        row_heights=[140.0, 140.0, 169.0],
    )

    compose(
        MAIN / "figure_05.pdf",
        [
            panel(
                "../components/figure5_schematics.pdf",
                "A",
                1,
                2,
                erase_heading=False,
            ),
            panel(
                "../components/main_physical_task_schematic.pdf",
                "A",
                1,
                1,
                erase_heading=False,
            ),
            *[panel("figure_05_panels_A-F.pdf", letter, 2, 3) for letter in "CDEF"],
            panel(
                "../components/figure5_architecture_schematic.pdf",
                "A",
                1,
                3,
                erase_heading=False,
            ),
            panel("figure_05_panels_G-L.pdf", "H", 2, 3),
        ],
        [
            "Matched-resource depth",
            "Nested divisive task",
            "Backprop depth test",
            "Prespecified contrasts",
            "Local credit transport",
            "Divisive control",
            "Architecture controls",
            "Serial composition",
        ],
        rows=3,
        cols=6,
        height=388,
        slots=[
            Slot(0, 0, colspan=3),
            Slot(0, 3, colspan=3),
            Slot(1, 0, colspan=2),
            Slot(1, 2, colspan=2),
            Slot(1, 4, colspan=2),
            Slot(2, 0, colspan=2),
            Slot(2, 2, colspan=2),
            Slot(2, 4, colspan=2),
        ],
        row_heights=[100.0, 132.0, 132.0],
    )

    # Figure 6 is authored as one coherent final-size canvas.  Its heatmaps
    # and aligned labels depend on shared row/column geometry, so splitting it
    # into heterogeneous legacy source sheets would undo the redesign.
    if not emit_native(6):
        copy_page("../components/main_figure6_redesigned.pdf", "figure_06.pdf")

    figure_07_panels = [
        panel(
            "../components/main_mapped_reconstruction.pdf",
            "A",
            1,
            1,
            erase_heading=False,
        ),
        panel(
            "../components/main_ancestry_addresses.pdf",
            "A",
            1,
            1,
            erase_heading=False,
        ),
        panel("figure_07_panels_A-J.pdf", "E", 2, 3),
        panel("figure_07_panels_A-J.pdf", "C", 2, 3),
        panel(
            "../components/main_wire_efficiency.pdf",
            "A",
            1,
            1,
            erase_heading=False,
        ),
        panel("figure_07_panels_K-L.pdf", "L", 1, 2),
        panel(
            "../components/main_cross_animal.pdf",
            "A",
            1,
            1,
            erase_heading=False,
        ),
    ]
    figure_07_titles = [
        "Mapped reconstruction",
        "Ancestry addresses",
        "Reciprocal cable field",
        "Sparse route capacity",
        "Wire efficiency at eight channels",
        "Wiring-normalized capture",
        "Independent-animal direction",
    ]
    compose(
        MAIN / "figure_07.pdf",
        figure_07_panels,
        figure_07_titles,
        rows=3,
        cols=6,
        height=396,
        slots=[
            Slot(0, 0, colspan=2),
            Slot(0, 2, colspan=2),
            Slot(0, 4, colspan=2),
            Slot(1, 0, colspan=3),
            Slot(1, 3, colspan=3),
            Slot(2, 0, colspan=3),
            Slot(2, 3, colspan=3),
        ],
        row_heights=[112.0, 127.0, 133.0],
    )

    compose(
        MAIN / "figure_08.pdf",
        [
            panel(
                "../components/main_focal_schematic.pdf",
                "A",
                1,
                1,
                erase_heading=False,
            ),
            panel("figure_08_panels_A-I.pdf", "B", 2, 3),
            panel("figure_08_panels_A-I.pdf", "D", 2, 3),
            panel("figure_08_panels_A-I.pdf", "E", 2, 3),
            panel("figure_08_panels_A-I.pdf", "F", 2, 3),
            panel("../components/active_dose_main.pdf", "A", 1, 2),
            panel("../components/active_dose_main.pdf", "B", 1, 2),
        ],
        [
            "Matched focal shunt",
            "Tree-relation selectivity",
            "Dose response",
            "Adjoint transport",
            "Electrotonic regime",
            "Active dose response",
            "Cellwise contrast",
        ],
        rows=3,
        cols=6,
        height=406,
        slots=[
            Slot(0, 0, colspan=2),
            Slot(0, 2, colspan=2),
            Slot(0, 4, colspan=2),
            Slot(1, 0, colspan=3),
            Slot(1, 3, colspan=3),
            Slot(2, 0, colspan=3),
            Slot(2, 3, colspan=3),
        ],
        row_heights=[116.0, 132.0, 134.0],
    )

    # Figure 9 is the empirical boundary: measured-response nulls, imposed
    # alignment, signed animal coordinates and the complete-tree learning
    # test.  The quantitative synthesis plane now belongs to Figure 4.
    compose(
        MAIN / "figure_09.pdf",
        [
            panel(
                "../components/main_structure_function_summary.pdf",
                "A",
                1,
                1,
                erase_heading=False,
            ),
            panel("figure_09_panels_A-H.pdf", "C", 2, 3),
            panel("figure_09_panels_A-H.pdf", "D", 2, 3),
            panel("figure_09_panels_A-H.pdf", "E", 2, 3),
            panel("../components/animal_pairs_wide.pdf", "A", 1, 1),
            panel("figure_09_panels_I-J.pdf", "I", 1, 2),
            panel("figure_09_panels_I-J.pdf", "J", 1, 2),
        ],
        [
            "Structure–function boundary",
            "Task-field capture",
            "Held-out learning",
            "Controlled alignment",
            "Signed contrast, six animals",
            "Full-tree task",
            "Anatomy boundary",
        ],
        rows=3,
        cols=6,
        height=404,
        slots=[
            Slot(0, 0, colspan=2),
            Slot(0, 2, colspan=2),
            Slot(0, 4, colspan=2),
            Slot(1, 0, colspan=3),
            Slot(1, 3, colspan=3),
            Slot(2, 0, colspan=3),
            Slot(2, 3, colspan=3),
        ],
        row_heights=[118.0, 127.0, 135.0],
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
            "Serial vs grouped star",
            "Flexible point controls",
            "Alignment dose",
            "Depth benefit",
            "Dose contrasts",
            # Titles name the placement regime each source panel actually
            # shows (P/Q/T were "H=3 aligned hierarchy", "H=3 reversed
            # placement", "H=2 reversed placement").  The earlier
            # models/outcomes and "local learning" retitles erased the
            # aligned-versus-reversed contrast the caption relies on, and J
            # includes serial and grouped-point BP, so "local learning" was
            # simply wrong.
            "H3 aligned hierarchy",
            "H3 reversed placement",
            "H3 interaction",
            "H2 aligned hierarchy",
            "H2 reversed placement",
            "H2 contrasts",
        ],
        rows=3,
        cols=4,
        height=458,
    )

    compose(
        SUPP / "figure_S19_panels_A-I.pdf",
        [
            # Source panels I and K carry the rotated label "exact transport
            # - BP (pp)", whose closing bracket rises into the heading band;
            # without the re-expansion the sheet composed a truncated "(pp".
            panel(
                "figure_02_panels_G-O.pdf", letter, 3, 3,
                pad_top_pt=4.0 if letter in "IK" else 0.0,
                # Re-expanding the clip also re-admits each source panel's own
                # heading, so erase it span-by-span.
                erase_phrases=(
                    ("Initial exact/BP comparison",) if letter == "I"
                    else ("Version-locked exact/BP",) if letter == "K"
                    else ()
                ),
            )
            for letter in "GHIJKLMNO"
        ],
        [
            "MNIST identity",
            "Noise-task identity",
            "Exact vs BP",
            "Ownership",
            "Version-locked exact vs BP",
            "Fixed-budget depth",
            "Two-stream learning",
            "Capture vs progress",
            "Context forgetting",
        ],
        rows=3,
        cols=3,
        height=458,
    )

    # The reduced regular-tree boundary sheet (S04 panels A-C) is now drawn
    # natively by scripts/build_supplementary_figure_s04_native.py from the
    # frozen regime tables; the archived nine-panel asset remains on disk for
    # provenance.

    # Duplication-trimmed detail sheets: each keeps only the panels that do
    # not replot a main-figure display, recomposed bit-identically from the
    # full synced sheets (which remain on disk as the archived sources).
    compose(
        SUPP / "figure_S17_panels_A-B.pdf",
        [panel("figure_S17_panels_A-D.pdf", "A", 1, 4),
         panel("figure_S17_panels_A-D.pdf", "D", 1, 4)],
        [
            "Signed causal mapping",
            "Neuron-level residuals",
        ],
        rows=1,
        cols=2,
        height=205,
    )

    compose(
        SUPP / "figure_S20_panels_A-B.pdf",
        [Panel("figure_S20_panels_A-J.pdf", 0, 2, 4, 3, pad_top_pt=6.5,
               erase_phrases=("Domains versus depth",)),
         # The 4x2 reading of the irregular source grid leaves panel J's
         # letter inside this cell's whitespace valley; white it out.
         Panel("figure_S20_panels_A-J.pdf", 2, 1, 4, 2,
               erase_phrases=("J",))],
        [
            "Route support",
            "Direct-type capture",
        ],
        rows=1,
        cols=2,
        height=140,
    )

    compose(
        SUPP / "figure_S21_panels_A-D.pdf",
        [
            # Source panel G's rotated "localization difference" label rises
            # into its heading band and composed as "localization differenc".
            panel(
                "figure_S21_panels_A-I.pdf", L, 3, 3,
                pad_top_pt=4.0 if L == "G" else 0.0,
            )
            for L in "CEFG"
        ],
        [
            "Within-cell controls",
            "Focal-site depth",
            "Scale and reversal sensitivity",
            "Direct presynaptic types",
        ],
        rows=2,
        cols=2,
        height=330,
    )

    compose(
        SUPP / "figure_S27_panels_A-C.pdf",
        [Panel("figure_S27_panels_A-D.pdf", 0, 0, 2, 2),
         Panel("figure_S27_panels_A-D.pdf", 0, 1, 2, 2, pad_top_pt=8.0),
         Panel("figure_S27_panels_A-D.pdf", 1, 1, 2, 2)],
        [
            "Pinky v185 cohort",
            "Residual across budgets",
            "Capture per coefficients",
        ],
        rows=2,
        cols=2,
        height=430,
    )

    compose(
        SUPP / "figure_S28_panels_A-B.pdf",
        [
            panel("figure_05_panels_G-L.pdf", "J", 2, 3),
            panel("figure_05_panels_G-L.pdf", "K", 2, 3),
        ],
        [
            "Credit-coordinate ladder",
            "BP–local decomposition",
        ],
        rows=1,
        cols=2,
        height=220,
    )

    print("Assembled nine compact main figures and Supplementary Figures S17, S18, S19, S20, S21, S27 and S28.")


if __name__ == "__main__":
    main()
