"""Horizontal clearance between side-by-side panels, per main figure.

The row-separation audit measures the bands BETWEEN rows; this one measures
the bands between neighbours WITHIN a row, over a y window that reaches above
the axes so panel letters and titles are included -- that is where neighbours
actually collide, not at the axes edges.
"""
import sys, glob
sys.path.insert(0, "scripts")
import fitz, numpy as np
from figure_canvas import _page_ink, _read_manifest

FLOOR_PT = 6.0        # a neighbour gap below this reads as touching
HEAD_PT = 10.0        # reach above the axes box: the title band only.
#                       Panel letters sit higher and are measured by
#                       audit_letter_alignment.py; including them here
#                       conflates a letter-to-its-own-label sliver with a
#                       real collision between neighbours.
LABEL_DROP_PT = 20.0  # how far the row above's x labels hang below it


def gaps(path):
    """Clearance between the FULL extents of side-by-side panels, in points.

    Each panel's extent -- axes plus ticks, labels and title -- is recorded in
    the manifest at build time as ``tx0_pt``/``tx1_pt``.  Ownership is the
    whole point: a rendered page cannot say which panel a given mark belongs
    to, so measuring the raster reports the slivers inside one panel's own
    label cluster as though they were the space between panels.  A negative
    number here is a genuine overlap: the two panels' extents intersect.
    """
    doc = fitz.open(path)
    man = _read_manifest(doc)
    if not man:
        return []
    rows = {}
    for p in man["panels"]:
        rows.setdefault(int(p.get("row", 0)), []).append(p)
    out = []
    for r in sorted(rows):
        panels = sorted(rows[r], key=lambda p: int(p.get("col", 0)))
        for a, b in zip(panels, panels[1:]):
            if "tx1_pt" not in a or "tx0_pt" not in b:
                continue
            out.append((r, a.get("name"), b.get("name"),
                        float(b["tx0_pt"]) - float(a["tx1_pt"])))
    return out


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    bad = 0
    for p in sorted(glob.glob("figures/main/figure_[0-9][0-9].pdf")):
        res = gaps(p)
        tight = [x for x in res if x[3] < FLOOR_PT]
        bad += len(tight)
        print(f"{p.split('/')[-1]:16s} " + "  ".join(
            f"{a}|{b}=" + ("  n/a" if g == float("inf") else f"{g:4.1f}")
            + ('!' if g < FLOOR_PT else '')
            for _, a, b, g in res))
    print(f"\nneighbour gaps below {FLOOR_PT} pt: {bad}")
    if strict and bad:
        raise SystemExit(1)
