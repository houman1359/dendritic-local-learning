"""Regression coverage for the all-panel (not first-column-only) contract."""
import json
from pathlib import Path
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_panel_letter_geometry import check, panel_geometry, publication_paths


def test_missing_metadata_is_unverified(tmp_path):
    path=tmp_path/'missing.pdf'
    with fitz.open() as doc:
        doc.new_page(width=300,height=200)
        doc.save(path)
    assert check(path) and 'UNVERIFIED' in check(path)[0]


def test_second_column_clearance_is_checked(tmp_path):
    path=tmp_path/'second.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=300,height=200)
        page.insert_text((10,20),'A',fontname='hebo',fontsize=9)
        page.insert_text((160,20),'B',fontname='hebo',fontsize=9)
        doc.set_metadata({'keywords':json.dumps({
            'schema':'panel-letter-layout/1','panels':[
                {'letter':'A','row':0,'column':0,'content_bbox':[25,28,120,100]},
                {'letter':'B','row':0,'column':1,'content_bbox':[157,28,290,100]},
            ]})})
        doc.save(path)
    assert any('panel B: left clearance' in issue for issue in check(path))


def test_native_composition_keeps_source_ownership():
    source={'schema':'panel-letter-layout/1','panels':[
        {'letter':'A','row':0,'column':0,'content_bbox':[20,30,80,90]}]}
    wrapped={'schema':'native-vector-reflow/1','source_layout':source}
    assert panel_geometry(wrapped)==source['panels']


def test_publication_inventory_includes_every_supplement():
    assert len(publication_paths(main_only=True))==10
    assert len(publication_paths())==46


def test_all_publication_panel_letters():
    failures={str(path.relative_to(ROOT)):check(path)
              for path in publication_paths()}
    assert not {path:issues for path,issues in failures.items() if issues}


def test_all_main_row_ink_gaps():
    import math
    from audit_row_separation import FLOOR_PT, gaps

    for path in publication_paths(main_only=True):
        measured = gaps(path)
        assert measured is not None, f"Missing row metadata: {path}"
        assert all(math.isfinite(gap) and gap >= FLOOR_PT for gap in measured), (
            path, measured)


def test_subplot_size_and_whole_panel_letter_bounds_are_distinct(tmp_path):
    from figure_canvas import NativeCanvas
    import matplotlib.pyplot as plt
    canvas=NativeCanvas(2.5,1)
    canvas.panel('A',0,0,6)
    canvas.panel('A_second_axis',0,6,6,letter='')
    path=tmp_path/'group.pdf'
    try:
        canvas.save(path,png=False,quiet=True)
        with fitz.open(path) as doc:
            metadata=json.loads(doc.metadata['keywords'])
        first=metadata['panels'][0]
        assert first['tx1_pt'] < first['letter_content_bbox'][2]-50
        assert panel_geometry(metadata)[0]['content_bbox']==first['letter_content_bbox']
        assert check(path)==[]
    finally:
        plt.close(canvas.fig)
