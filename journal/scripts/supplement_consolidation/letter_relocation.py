"""Keep every panel letter of a curated sheet above and left of its panel's ink.

Review pass 2026-09-23.  Sheets pasted whole from native renders keep the
letters their builders placed, and eleven of them had a letter whose right
edge ran past a y-axis title or whose top sat below a panel's first line of
ink.  The fix moves only the letter glyphs; no panel content is touched.

Measured on the raster of the composed page: every bold letter glyph is
blanked, and inside each panel rectangle (minus any strip it shares with a
neighbouring rectangle, where a neighbour's label column can intrude) the
first ink row and column are that panel's top and left ink.  A letter moves
up until its top is ``TOP_PT`` above that row and left until its right edge
is ``LEFT_PT`` left of that column.  Letters whose tops lie within
``GROUP_PT`` of each other move by one common dy, and letters whose left
edges lie within ``GROUP_PT`` by one common dx, so row and column alignment
survive the move.  No letter may land on ink: a group's move is cut back to
the largest step that keeps every member ``CLEAR_PT`` clear of all marks on
the page, so a letter hemmed in by a neighbour's title moves only partway.
"""
from __future__ import annotations

import fitz
import numpy as np

TOP_PT, LEFT_PT, GROUP_PT, REACH_PT, EDGE_PT, CLEAR_PT = 2.6, 3.6, 12.0, 25.0, 0.3, 1.0


def _letters(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if (len(text) == 1 and "A" <= text <= "Z" and span["size"] >= 8
                        and ("bold" in span["font"].lower() or span["flags"] & 16)):
                    out.append(span)
    return out


def _gap(a, b):
    dx = max(a.x0 - b.x1, 0.0, b.x0 - a.x1)
    dy = max(a.y0 - b.y1, 0.0, b.y0 - a.y1)
    return (dx * dx + dy * dy) ** 0.5


def _groups(items, key):
    """Chains of letters whose ``key`` coordinates lie within GROUP_PT."""
    ordered = sorted(items, key=key)
    groups, current = [], []
    for item in ordered:
        if current and key(item) - key(current[-1]) > GROUP_PT:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def native_regions(source_layout):
    """Exact panel content boxes of a native-canvas sheet pasted whole.

    The manifest's ``letter_content_bbox`` is each panel's axes plus its own
    labels (top-left coordinates), which excludes sheet-wide keys that the
    letter-derived crop rectangles swallow.
    """
    if not source_layout or source_layout.get("schema") != "native-canvas/1":
        return {}
    return {p["letter"]: p["letter_content_bbox"]
            for p in source_layout.get("panels", [])
            if p.get("letter") and p.get("letter_content_bbox")}


def relocate_letters(page, panelmap, fontfile, *, zoom=4.0, regions=None):
    """Move offending letters on ``page``; return one record per moved glyph.

    ``regions`` (letter -> rect) replaces a panel's crop rectangle with its
    exact content box when the source is a native sheet pasted whole.
    """
    recs = [r for r in panelmap if r.get("panel")]
    regions = regions or {}
    rects = [fitz.Rect(regions.get(r["panel"], r["target_rect"])) & page.rect
             for r in recs]
    glyphs = _letters(page)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                          colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).copy()
    for span in glyphs:
        x0, y0, x1, y1 = span["bbox"]
        img[max(int((y0 - 0.6) * zoom), 0):int((y1 + 0.6) * zoom) + 1,
            max(int((x0 - 0.6) * zoom), 0):int((x1 + 0.6) * zoom) + 1] = 255
    ink = img < 245
    need = []
    for rec, box in zip(recs, rects):
        name = rec["panel"]
        # a sheet carries each panel letter once; distance only breaks ties
        owned = sorted((s for s in glyphs if s["text"].strip() == name),
                       key=lambda s: _gap(fitz.Rect(s["bbox"]), box))
        if not owned or box.is_empty:
            continue
        span = owned[0]
        sub = ink[int(box.y0 * zoom):int(box.y1 * zoom),
                  int(box.x0 * zoom):int(box.x1 * zoom)].copy()
        for other in rects:
            if other is box:
                continue
            shared = box & other
            if shared.is_empty or shared.width <= 0 or shared.height <= 0:
                continue
            sub[int((shared.y0 - box.y0) * zoom):int((shared.y1 - box.y0) * zoom) + 1,
                int((shared.x0 - box.x0) * zoom):int((shared.x1 - box.x0) * zoom) + 1] = False
        if not sub.any():
            continue
        top = box.y0 + np.argmax(sub.any(axis=1)) / zoom
        left = box.x0 + np.argmax(sub.any(axis=0)) / zoom
        x0, y0, x1, _ = span["bbox"]
        need.append(dict(span=span, letter=name, x0=x0, y0=y0,
                         dx=max(0.0, x1 - (left - LEFT_PT)),
                         dy=max(0.0, y0 - (top - TOP_PT))))
    def clear(n, dx, dy):
        """Is the glyph box, moved by (-dx, -dy) and padded, free of ink?"""
        x0, y0, x1, y1 = n["span"]["bbox"]
        if x0 - dx < EDGE_PT or y0 - dy < EDGE_PT:
            return False
        window = ink[max(int((y0 - dy - CLEAR_PT) * zoom), 0):int((y1 - dy + CLEAR_PT) * zoom) + 1,
                     max(int((x0 - dx - CLEAR_PT) * zoom), 0):int((x1 - dx + CLEAR_PT) * zoom) + 1]
        return not window.any()

    def largest(group, step, axis):
        """Largest common move <= step that keeps every member clear."""
        value = step
        while value > 0.05:
            if all(clear(n, value if axis == "x" else n["dx"],
                         value if axis == "y" else n["dy"]) for n in group):
                return value
            value -= 0.25
        return 0.0

    for n in need:
        n["dx"] = n["dx"] if n["dx"] > 0.05 else 0.0
    for group in _groups(need, key=lambda n: n["y0"]):
        want = max(n["dy"] for n in group)
        for n in group:
            n["dy"], n["dx0"] = want, n["dx"]
            n["dx"] = 0.0
        dy = largest(group, want, "y")
        for n in group:
            n["dy"] = dy
            n["dx"] = n.pop("dx0")
    for group in _groups(need, key=lambda n: n["x0"]):
        dx = largest(group, max(n["dx"] for n in group), "x")
        for n in group:
            n["dx"] = dx
    moves = [n for n in need if n["dx"] > 0.05 or n["dy"] > 0.05]
    if not moves:
        return []
    for n in moves:
        x0, y0, x1, y1 = n["span"]["bbox"]
        # a small central box: only this glyph's own bbox can reach it
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        page.add_redact_annot(fitz.Rect(cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0),
                              fill=None)
    page.apply_redactions(images=0, graphics=0, text=0)
    page.insert_font(fontname="PanelSans", fontfile=fontfile)
    records = []
    for n in moves:
        span = n["span"]
        colour = span["color"]
        rgb = (((colour >> 16) & 255) / 255.0, ((colour >> 8) & 255) / 255.0,
               (colour & 255) / 255.0)
        ox, oy = span["origin"]
        page.insert_text((ox - n["dx"], oy - n["dy"]), n["letter"],
                         fontname="PanelSans", fontsize=span["size"], color=rgb)
        records.append(dict(letter=n["letter"], dx_pt=round(-n["dx"], 2),
                            dy_pt=round(-n["dy"], 2)))
    return records
