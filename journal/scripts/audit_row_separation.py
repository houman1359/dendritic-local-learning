"""Real ink-to-ink separation between adjacent panel rows, per main figure.

Row bands come from the panels' final axes boxes (post reserve-lock), not the
nominal grid, and columns owned by a panel whose rowspan crosses the boundary
are masked out -- that panel's ink legitimately runs through the gutter.
"""
import sys, glob
sys.path.insert(0, "scripts")
import fitz, numpy as np
from figure_canvas import _page_ink, _read_manifest

LABEL_BITE_PT = 46.0
FLOOR_PT = 8.5          # 3 mm


def gaps(path, dpi=300.0):
    doc = fitz.open(path)
    man = _read_manifest(doc)
    if not man:
        return None
    ink, zoom = _page_ink(doc[0], dpi)
    W = ink.shape[1]
    H = float(man["height_pt"])
    nrows = len(man["row_h_pt"])
    # axes-box band of each row, measured DOWN from the page top
    band = {}
    for p in man["panels"]:
        r0 = int(p.get("row", 0)); rs = int(p.get("rowspan", 1) or 1)
        if rs != 1:
            continue
        t = H - (float(p["y0_pt"]) + float(p["h_pt"])); b = H - float(p["y0_pt"])
        lo, hi = band.get(r0, (t, b))
        band[r0] = (min(lo, t), max(hi, b))
    out = []
    for i in range(nrows - 1):
        if i not in band or i + 1 not in band:
            out.append(float("nan")); continue
        keep = np.ones(W, dtype=bool)
        for p in man["panels"]:
            r0 = int(p.get("row", 0)); rs = int(p.get("rowspan", 1) or 1)
            if r0 <= i and r0 + rs - 1 >= i + 1:
                a = int(max(0, (float(p["x0_pt"]) - LABEL_BITE_PT) * zoom))
                b = int(min(W, (float(p["x0_pt"]) + float(p["w_pt"]) + 8) * zoom))
                keep[a:b] = False
        s = max(0, int((band[i][1] - 6.0) * zoom))
        e = min(int((band[i + 1][0] + 6.0) * zoom), ink.shape[0] - 1)
        # A column inked down essentially the whole band is a connector or a
        # bracket drawn ACROSS the gutter on purpose, not a row crowding its
        # neighbour: it would read as a zero gap everywhere it appears, so it
        # is masked exactly as a row-spanning panel is.
        span = ink[s:e + 1]
        if span.shape[0]:
            keep &= span.mean(axis=0) < 0.9
        rows_ink = ink[:, keep].any(axis=1)
        best = run = 0
        for has in rows_ink[s:e + 1]:
            run = 0 if has else run + 1
            best = max(best, run)
        out.append(best / zoom)
    return out


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    bad = 0
    for p in sorted(glob.glob("figures/main/figure_[0-9][0-9].pdf")):
        g = gaps(p)
        if g is None:
            continue
        print(f"{p.split('/')[-1]:16s} " + "  ".join(
            f"r{i}|r{i+1}={v:5.1f}pt({v * 25.4 / 72:4.2f}mm)"
            f"{'  <-LOW' if v < FLOOR_PT else '       '}"
            for i, v in enumerate(g)))
        bad += sum(1 for v in g if v < FLOOR_PT)
    print(f"\nboundaries below {FLOOR_PT} pt (3 mm): {bad}")
