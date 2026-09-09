"""Panel letters: on the module grid, and clear of every panel's ink.

Two checks per main figure.

2026-09-08 spec upgrade: the letter's x is the MODULE COLUMN origin minus the
canvas's one letter offset, not a single hand-set ``LETTER_HOME_PT`` for the
row-leading letters, so the first check now reads the letter grid out of the
manifest and asserts that every letter of a column shares an x (the same
contract ``figure_canvas --audit`` enforces as ``letter-grid``).  The second
check is unchanged and is the one that bites: a column-0 panel whose y labels
are wider than the left margin pushes ink into the letter column, and the
figure's ``MARGINS.left`` has to grow until it does not.
"""
import sys, glob
sys.path.insert(0, "scripts")
import fitz, numpy as np
from figure_canvas import (_page_ink, _read_manifest, ALIGN_TOL_PT,
                           LETTER_GAP_PT, PANEL_LABEL_PT)

LETTER_W_PT = 0.72 * PANEL_LABEL_PT    # bold cap advance at the letter size
TOL_PT = 0.6


def check(path, dpi=300.0):
    doc = fitz.open(path)
    man = _read_manifest(doc)
    if not man:
        return []
    ink, zoom = _page_ink(doc[0], dpi)
    H = float(man["height_pt"])
    bad = []
    letters = man.get("letters") or []
    by_col = {}
    for rec in letters:
        by_col.setdefault(int(rec.get("col", 0)), []).append(rec)
    for col, group in sorted(by_col.items()):
        xs = [float(r["x_pt"]) for r in group]
        if len(xs) > 1 and max(xs) - min(xs) > ALIGN_TOL_PT:
            names = ", ".join(f"{r['letter']}@{float(r['x_pt']):.1f}"
                              for r in group)
            bad.append(f"module column {col} letters spread "
                       f"{max(xs) - min(xs):.1f}pt ({names})")
    home = min((float(r["x_pt"]) for r in letters), default=None)
    left_edge = np.where(ink.any(axis=0))[0][0] / zoom
    if home is not None and left_edge < home - TOL_PT:
        bad.append(f"ink at {left_edge:.1f}pt, left of the leftmost letter "
                   f"{home:.1f}pt")
    column_end = (home if home is not None else 0.0) \
        + LETTER_W_PT + LETTER_GAP_PT
    for rec in man["panels"]:
        if int(rec.get("col", 0)) != 0:
            continue
        t = H - (float(rec["y0_pt"]) + float(rec["h_pt"]))
        b = H - float(rec["y0_pt"])
        xs = np.where(ink[int((t + 2) * zoom):int(b * zoom)].any(axis=0))[0]
        if len(xs) and xs.min() / zoom < column_end - TOL_PT:
            bad.append(f"panel {rec.get('name')} ink at {xs.min() / zoom:.1f}pt "
                       f"intrudes into the letter column (ends {column_end:.1f}pt)")
    return bad


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    total = 0
    for p in sorted(glob.glob("figures/main/figure_[0-9][0-9].pdf")):
        bad = check(p)
        total += len(bad)
        print(f"{p.split('/')[-1]:16s} " +
              ("ok" if not bad else "; ".join(bad)))
    print(f"\nletter-alignment problems: {total}")
    if strict and total:
        raise SystemExit(1)
