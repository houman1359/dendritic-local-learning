"""Row-leading panel letters: one shared left edge, and the leftmost ink.

Two checks per main figure.  The letters that lead each row must sit at the
same x -- the canvas pins them to ``LETTER_HOME_PT`` -- and nothing may sit
left of them, so the page reads down a single margin rule.  The second check
is the one that bites: a leading panel whose y labels are wider than the left
margin pushes ink into the letter column, and the figure's ``MARGINS.left``
has to grow until it does not.
"""
import sys, glob
sys.path.insert(0, "scripts")
import fitz, numpy as np
from figure_canvas import (_page_ink, _read_manifest, LETTER_HOME_PT,
                           LETTER_GAP_PT)

LETTER_W_PT = 7.4      # bold cap advance at the panel-label size
TOL_PT = 0.6


def check(path, dpi=300.0):
    doc = fitz.open(path)
    man = _read_manifest(doc)
    if not man:
        return []
    ink, zoom = _page_ink(doc[0], dpi)
    H = float(man["height_pt"])
    bad = []
    left_edge = np.where(ink.any(axis=0))[0][0] / zoom
    if left_edge < LETTER_HOME_PT - TOL_PT:
        bad.append(f"ink at {left_edge:.1f}pt, left of the letter home "
                   f"{LETTER_HOME_PT:.1f}pt")
    column_end = LETTER_HOME_PT + LETTER_W_PT + LETTER_GAP_PT
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
