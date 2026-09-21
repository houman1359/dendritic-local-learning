"""Check actual PDF letters against full, source-declared panel ownership.

Missing layout metadata is unverified, never a pass. Shared legends and
spanning panels must be assigned by the builder, not guessed by proximity.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
TOL_PT = 0.6
LEFT_GAP_PT = 3.4
TOP_CLEAR_PT = 5.4


def panel_geometry(metadata):
    schema = metadata.get("schema")
    if schema == "native-vector-reflow/1":
        child = metadata.get("letter_layout") or metadata.get("source_layout")
        if not child:
            raise ValueError("missing complete panel-letter layout metadata")
        return panel_geometry(child)
    if schema == "panel-letter-layout/1":
        if not metadata.get("panels"):
            raise ValueError("no declared panel letters; cannot verify")
        return metadata["panels"]
    if schema != "native-canvas/1":
        raise ValueError("missing complete panel-letter layout metadata")
    height = float(metadata["height_pt"])
    panels = []
    for rec in metadata["panels"]:
        if not rec.get("letter"):
            continue
        required = ("tx0_pt", "tx1_pt", "ty0_pt", "ty1_pt")
        if any(key not in rec for key in required):
            raise ValueError(f"panel {rec['letter']}: full content bounds missing")
        panels.append({"letter": rec["letter"], "row": rec["row"],
                       "column": rec["col"],
                       "content_bbox": rec.get("letter_content_bbox",
                            [rec["tx0_pt"], height-rec["ty1_pt"],
                             rec["tx1_pt"], height-rec["ty0_pt"]])})
    if not panels:
        raise ValueError("no declared panel letters; cannot verify")
    return panels


def letter_spans(page):
    found = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                bold = "bold" in span["font"].lower() or span["flags"] & 16
                if len(text) == 1 and "A" <= text <= "Z" and bold and span["size"] >= 8:
                    found.setdefault(text, []).append(span)
    return found


def check(path, dpi=300.0):
    """Return failures; dpi is retained for compatibility with the old audit."""
    del dpi
    with fitz.open(path) as doc:
        try:
            metadata = json.loads(doc.metadata.get("keywords") or "{}")
            panels = panel_geometry(metadata)
        except (ValueError, KeyError, TypeError) as exc:
            return [f"UNVERIFIED: {exc}"]
        page = doc[0]; found = letter_spans(page)
        bad = []; rows = {}; columns = {}; letters = []
        for rec in panels:
            name = rec["letter"]; candidates = found.get(name, [])
            if len(candidates) != 1:
                bad.append(f"panel {name}: expected one letter, found {len(candidates)}")
                continue
            lb = fitz.Rect(candidates[0]["bbox"])
            content = fitz.Rect(rec["content_bbox"])
            left = content.x0 - lb.x1; top = content.y0 - lb.y0
            if left < LEFT_GAP_PT-TOL_PT:
                bad.append(f"panel {name}: left clearance {left:.2f}pt < {LEFT_GAP_PT}pt")
            if top < TOP_CLEAR_PT-TOL_PT:
                bad.append(f"panel {name}: top clearance {top:.2f}pt < {TOP_CLEAR_PT}pt")
            expanded = page.rect + (-TOL_PT, -TOL_PT, TOL_PT, TOL_PT)
            if not expanded.contains(lb):
                bad.append(f"panel {name}: letter outside page")
            if not expanded.contains(content):
                bad.append(f"panel {name}: content outside page")
            rows.setdefault(str(rec["row"]), []).append((name, lb.y0))
            columns.setdefault(str(rec["column"]), []).append((name, lb.x0))
            letters.append((name, lb, content))
        for axis, groups in (("row", rows), ("column", columns)):
            for key, group in groups.items():
                spread = max(v for _, v in group)-min(v for _, v in group)
                if spread > TOL_PT:
                    bad.append(f"{axis} {key}: letter spread {spread:.2f}pt "
                               f"({','.join(n for n, _ in group)})")
        for name, lb, _ in letters:
            for other, _, content in letters:
                if name != other:
                    overlap = lb & content
                    if overlap.width > 1 and overlap.height > 1:
                        # A decorated union includes empty corners (notably
                        # beside colorbars). Require another actual text mark
                        # in that corner before calling it a collision.
                        for block in page.get_text('dict')['blocks']:
                            for line in block.get('lines', []):
                                for span in line['spans']:
                                    box = fitz.Rect(span['bbox'])
                                    if box == lb:
                                        continue
                                    hit = box & lb
                                    if hit.width > 1 and hit.height > 1:
                                        bad.append(f"letter {name}: overlaps text "
                                                   f"{span['text']!r}")
        return bad


def publication_paths(main_only=False):
    """Use the submission allowlist, including separately rendered SI figures."""
    from build_submission_bundle import MAIN_FIGURES, SUPPLEMENTARY_FIGURES
    selected = MAIN_FIGURES if main_only else MAIN_FIGURES + SUPPLEMENTARY_FIGURES
    return [ROOT / "figures" / relative for relative in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths] or publication_paths(args.main_only)
    total = 0
    for path in paths:
        bad = check(path); total += len(bad)
        print(f"{path.name}: " + ("ok" if not bad else "; ".join(bad)))
    print(f"\n{len(paths)} figures checked; letter-placement problems: {total}")
    if args.strict and total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
