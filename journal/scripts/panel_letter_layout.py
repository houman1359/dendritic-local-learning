"""Full decorated-panel lettering for retained Matplotlib figure builders.

This save-time adapter uses explicit ownership groups, including colorbars and
multiple axes within a panel. It preserves numerical artists, limits, fonts
and page size. Only panel placement and lettering are adjusted.
"""
from __future__ import annotations
import json
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from journal_style import PANEL_LABEL_PT, SANS_FAMILY


def finish_panel_letters(fig, groups):
    """Return PDF metadata; groups are (letter, row, column, [owned axes])."""
    names = {name for name, _, _, _ in groups}
    for artist in list(fig.findobj(Text)):
        weight = artist.get_fontweight()
        bold = weight in ('bold', 'semibold', 'demibold', 'heavy', 'black')
        bold = bold or isinstance(weight, (int, float)) and weight >= 600
        if artist.get_text().strip() in names and bold and artist.get_fontsize() >= 8:
            artist.remove()
    width, height = fig.get_size_inches()*72
    scale = 72/fig.dpi

    def bounds(axes, renderer):
        return Bbox.union([a.get_tightbbox(renderer) for a in axes
                           if a.get_visible()])

    # Leave enough room at the left page edge for the widest letter and gap.
    for _ in range(4):
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        deficits = {}
        for _, _, col, axes in groups:
            left = bounds(axes, renderer).x0*scale
            deficits[col] = max(deficits.get(col, 0), 16.0-left)
        if max(deficits.values(), default=0) <= .05:
            break
        moved = set()
        for _, _, col, axes in groups:
            delta = max(0, deficits[col])/width
            for ax in axes:
                if id(ax) in moved or delta <= 0:
                    continue
                box = ax.get_position()
                ax.set_position([box.x0+delta, box.y0, box.width-delta, box.height])
                moved.add(id(ax))
    # A reserved top strip keeps the first row's labels on the page.
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    top = max(bounds(axes, renderer).y1*scale for _, _, _, axes in groups)
    shift = max(0, top+9.0-height)
    if shift:
        for ax in fig.axes:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0-shift/height, box.width, box.height])
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    col_left = {}; row_top = {}
    for _, row, col, axes in groups:
        box = bounds(axes, renderer)
        col_left[col] = min(col_left.get(col, float('inf')), box.x0*scale-12.0)
        row_top[row] = max(row_top.get(row, 0), box.y1*scale+7.0)
    for name, row, col, _ in groups:
        fig.text(col_left[col]/width, row_top[row]/height, name,
                 ha='left', va='top', fontweight='bold', fontsize=PANEL_LABEL_PT,
                 family=SANS_FAMILY, color='#202427')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    records = []
    for name, row, col, axes in groups:
        box = bounds(axes, renderer)
        records.append({'letter':name, 'row':row, 'column':col,
                        'content_bbox':[box.x0*scale, height-box.y1*scale,
                                        box.x1*scale, height-box.y0*scale]})
    return {'CreationDate':None, 'ModDate':None,
            'Keywords':json.dumps({'schema':'panel-letter-layout/1',
                                  'panels':records}, separators=(',', ':'))}
