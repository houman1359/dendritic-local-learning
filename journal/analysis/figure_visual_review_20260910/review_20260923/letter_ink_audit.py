"""Independent panel-letter check measured on the rendered PDF ink.

Every text span and vector path is assigned to the nearest panel axes box from
the embedded native-canvas manifest (unlettered axes belong to the preceding
lettered panel in the same grid row).  A letter passes when it sits above and
to the left of ALL ink it owns: letter top at least TOP_PT above the panel's
topmost ink and letter right edge at least LEFT_PT left of its leftmost ink.
Letters sharing a grid row must share a top edge; letters sharing a grid
column must share a left edge.
"""
import json, sys
from pathlib import Path
import fitz

TOP_PT, LEFT_PT, TOL = 2.0, 3.0, 0.6


def letters_of(page):
    out = {}
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines', []):
            for s in line['spans']:
                t = s['text'].strip()
                if len(t) == 1 and 'A' <= t <= 'Z' and ('bold' in s['font'].lower() or s['flags'] & 16) and s['size'] >= 8:
                    out.setdefault(t, []).append(fitz.Rect(s['bbox']))
    return out


def ink_of(page, letter_rects):
    items = []
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines', []):
            for s in line['spans']:
                if not s['text'].strip():
                    continue
                r = fitz.Rect(s['bbox'])
                if any(r == lr for lr in letter_rects):
                    continue
                items.append(('text', s['text'].strip()[:40], r))
    for d in page.get_drawings():
        r = fitz.Rect(d['rect'])
        if r.is_empty and r.width == 0 and r.height == 0:
            continue
        # ignore invisible (unfilled, unstroked) and page-sized background rects
        if d.get('fill') is None and d.get('color') is None:
            continue
        if r.width > page.rect.width * 0.98 and r.height > page.rect.height * 0.98:
            continue
        if d.get('fill') in ((1, 1, 1), [1, 1, 1]) and d.get('color') is None:
            continue
        items.append(('path', d.get('type', ''), r))
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            items.append(('image', str(img[0]), fitz.Rect(r)))
    return items


def dist(r, p):
    dx = max(r.x0 - p.x1, 0, p.x0 - r.x1)
    dy = max(r.y0 - p.y1, 0, p.y0 - r.y1)
    return (dx * dx + dy * dy) ** .5


def audit_reflow(page, meta, zoom=4.0):
    """Reflowed sheets (supplement): each panel owns its target rectangle.

    Measured on the RASTER: vector paths in these sheets can bundle strokes
    from several panels into one item, so their boxes are not usable.  The
    page is rendered, every bold letter glyph is blanked, and within each
    panel rectangle the first ink row and column are the panel's top and left
    ink.  The letter is the glyph named for the panel nearest its rectangle
    (inside it or just above-left).  Letters whose tops lie within 12 pt form
    a row and must share a top edge; left edges within 12 pt form a column.
    """
    import numpy as np
    recs = [r for r in meta['panels'] if r.get('panel')]
    rects = [fitz.Rect(r['target_rect']) & page.rect for r in recs]
    found = letters_of(page)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    for rr in (r for v in found.values() for r in v):
        x0, y0 = int((rr.x0 - 0.6) * zoom), int((rr.y0 - 0.6) * zoom)
        x1, y1 = int((rr.x1 + 0.6) * zoom) + 1, int((rr.y1 + 0.6) * zoom) + 1
        img[max(y0, 0):y1, max(x0, 0):x1] = 255
    ink = img < 245
    problems, geom = [], {}
    for r, box in zip(recs, rects):
        L = r['panel']
        cands = sorted((c for c in found.get(L, []) if dist(c, box) <= 25.0),
                       key=lambda c: dist(c, box))
        if not cands:
            problems.append(f'{L}: no letter glyph within 25 pt of its rectangle'); continue
        lb = cands[0]
        sub = ink[int(box.y0 * zoom):int(box.y1 * zoom), int(box.x0 * zoom):int(box.x1 * zoom)].copy()
        # strips shared with a neighbouring rectangle belong to neither panel:
        # a neighbour's column of labels can run a few points into this one
        for other in rects:
            if other is box:
                continue
            ov = box & other
            if ov.is_empty or ov.width <= 0 or ov.height <= 0:
                continue
            sub[int((ov.y0 - box.y0) * zoom):int((ov.y1 - box.y0) * zoom) + 1,
                int((ov.x0 - box.x0) * zoom):int((ov.x1 - box.x0) * zoom) + 1] = False
        if not sub.any():
            problems.append(f'{L}: no ink'); continue
        top = box.y0 + np.argmax(sub.any(axis=1)) / zoom
        left = box.x0 + np.argmax(sub.any(axis=0)) / zoom
        top_gap, left_gap = top - lb.y0, left - lb.x1
        geom[L] = dict(top_gap=round(top_gap, 1), left_gap=round(left_gap, 1),
                       x=round(lb.x0, 1), y=round(lb.y0, 1))
        if top_gap < TOP_PT - TOL:
            problems.append(f'{L}: letter top only {top_gap:.1f} pt above panel ink')
        if left_gap < LEFT_PT - TOL:
            problems.append(f'{L}: letter right edge only {left_gap:.1f} pt left of panel ink')
    for name, key in (('row', 'y'), ('column', 'x')):
        items = sorted(geom.items(), key=lambda kv: kv[1][key])
        groups, cur = [], []
        for L, g in items:
            if cur and g[key] - cur[-1][1][key] > 12.0:
                groups.append(cur); cur = []
            cur.append((L, g))
        if cur:
            groups.append(cur)
        for grp in groups:
            vals = [g[key] for _, g in grp]
            if len(grp) > 1 and max(vals) - min(vals) > TOL:
                problems.append(f'{name}: letters {"".join(L for L, _ in grp)} spread {max(vals) - min(vals):.1f} pt')
    return geom, problems


def audit(path):
    doc = fitz.open(path); page = doc[0]; H = page.rect.height
    meta = json.loads(doc.metadata.get('keywords') or '{}')
    if meta.get('schema') == 'native-vector-reflow/1':
        return audit_reflow(page, meta)
    if meta.get('schema') != 'native-canvas/1':
        return None, [f'no native-canvas manifest ({meta.get("schema")})']
    recs = meta['panels']
    boxes = []
    for r in recs:
        top = H - (r['y0_pt'] + r['h_pt'])
        boxes.append(fitz.Rect(r['x0_pt'], top, r['x0_pt'] + r['w_pt'], top + r['h_pt']))
    owner = []
    for i, r in enumerate(recs):
        if r.get('letter'):
            owner.append(r['letter']); continue
        prev = [j for j, q in enumerate(recs) if q['row'] == r['row'] and q['col'] < r['col'] and q.get('letter')]
        owner.append(recs[max(prev, key=lambda j: recs[j]['col'])]['letter'] if prev else None)
    found = letters_of(page)
    letter_rects = [rr for v in found.values() for rr in v]
    ink = ink_of(page, letter_rects)
    union, worst = {}, {}
    for kind, label, r in ink:
        j = min(range(len(recs)), key=lambda k: dist(r, boxes[k]))
        L = owner[j]
        if L is None:
            continue
        u = union.get(L)
        union[L] = fitz.Rect(r) if u is None else u | r
        for side, val in (('top', r.y0), ('left', r.x0)):
            key = (L, side)
            if key not in worst or val < worst[key][0]:
                worst[key] = (val, kind, label)
    problems, rows = [], {}
    geom = {}
    for r in recs:
        L = r.get('letter')
        if not L:
            continue
        cands = found.get(L, [])
        if len(cands) != 1:
            problems.append(f'{L}: {len(cands)} letter glyphs'); continue
        lb = cands[0]; u = union.get(L)
        if u is None:
            problems.append(f'{L}: no ink'); continue
        top_gap = u.y0 - lb.y0; left_gap = u.x0 - lb.x1
        geom[L] = dict(top_gap=round(top_gap, 1), left_gap=round(left_gap, 1),
                       top_ink=worst[(L, 'top')][1:], left_ink=worst[(L, 'left')][1:],
                       row=r['row'], col=r['col'], x=round(lb.x0, 1), y=round(lb.y0, 1))
        if top_gap < TOP_PT - TOL:
            problems.append(f'{L}: letter top only {top_gap:.1f} pt above panel ink (topmost: {worst[(L,"top")][1]} {worst[(L,"top")][2]!r})')
        if left_gap < LEFT_PT - TOL:
            problems.append(f'{L}: letter right edge only {left_gap:.1f} pt left of panel ink (leftmost: {worst[(L,"left")][1]} {worst[(L,"left")][2]!r})')
    by_row, by_col = {}, {}
    for L, g in geom.items():
        by_row.setdefault(g['row'], []).append((L, g['y']))
        by_col.setdefault(g['col'], []).append((L, g['x']))
    for name, groups in (('row', by_row), ('column', by_col)):
        for k, grp in groups.items():
            vals = [v for _, v in grp]
            if max(vals) - min(vals) > TOL:
                problems.append(f'{name} {k}: letters {"".join(L for L,_ in grp)} spread {max(vals)-min(vals):.1f} pt')
    return geom, problems


if __name__ == '__main__':
    total = 0
    for p in sys.argv[1:]:
        geom, probs = audit(p)
        total += len(probs)
        print(f'== {Path(p).name}: {"ok" if not probs else str(len(probs)) + " problem(s)"}')
        for x in probs:
            print('   ', x)
        if geom and '-v' in sys.argv[0:1]:
            pass
    print('total problems', total)
