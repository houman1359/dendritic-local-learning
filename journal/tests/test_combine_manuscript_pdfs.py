"""Named destinations collide across papers and use PDF rather than MuPDF axes."""
from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from combine_manuscript_pdfs import combine_pdfs


def make_source(path, title, raw_y):
    doc = fitz.open()
    doc.new_page(width=400, height=600)
    doc.new_page(width=400, height=600)
    doc[0].insert_text((30, 30), title)
    doc[1].insert_text((40, 600-raw_y+10), "named target")
    # Both input documents deliberately define exactly the same destination.
    destination = f"[{doc.page_xref(1)} 0 R /XYZ 40 {raw_y} 0]"
    doc.xref_set_key(doc.pdf_catalog(), "Names", f"<< /Dests << /Names [(shared.name) {destination}] >> >>")
    doc[0].insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(30, 50, 140, 65),
                        "page": 1, "to": fitz.Point(40, 600-raw_y)})
    page = doc.reload_page(doc[0])
    ref = page.get_links()[0]["xref"]
    doc.xref_set_key(ref, "A", "<< /S /GoTo /D (shared.name) >>")
    page.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(30, 80, 140, 95),
                      "page": 1, "to": fitz.Point(75, 123)})
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(30, 110, 140, 125),
                      "uri": "https://example.org/"+title})
    doc.set_toc([[1, title, 2]])
    outline = doc.get_toc(False)[0][3]["xref"]
    doc.xref_set_key(outline, "A", "<< /S /GoTo /D (shared.name) >>")
    doc.save(path)
    doc.close()


def test_colliding_named_destinations_offsets_coordinates_and_outlines(tmp_path):
    a, b, output = [tmp_path / p for p in ("a.pdf", "b.pdf", "combined.pdf")]
    make_source(a, "main", 510)
    make_source(b, "supplement", 170)
    report = combine_pdfs([a, b], output, tmp_path / "audit.json")
    assert report["internal_links"] == 4 and report["uri_links"] == 2
    assert report["bookmarks"] == 2 and report["input_pdfs_unchanged"]
    with fitz.open(output) as doc:
        for source_page, target_page, named_y, uri in [(0, 1, 90, "main"), (2, 3, 430, "supplement")]:
            links = sorted(doc[source_page].get_links(), key=lambda l: l["from"].y0)
            assert [l["kind"] for l in links] == [fitz.LINK_GOTO, fitz.LINK_GOTO, fitz.LINK_URI]
            assert links[0]["page"] == target_page
            assert tuple(links[0]["to"]) == pytest.approx((40, named_y))
            assert links[1]["page"] == target_page
            assert tuple(links[1]["to"]) == pytest.approx((75, 123))
            assert links[2]["uri"] == "https://example.org/"+uri
        toc = doc.get_toc(False)
        assert [row[:3] for row in toc] == [[1, "main", 2], [1, "supplement", 4]]
        assert tuple(toc[0][3]["to"]) == pytest.approx((40, 90))
        assert tuple(toc[1][3]["to"]) == pytest.approx((40, 430))


def test_does_not_overwrite_input(tmp_path):
    a = tmp_path / "a.pdf"
    make_source(a, "main", 510)
    original = a.read_bytes()
    with pytest.raises(ValueError, match="output must not replace"):
        combine_pdfs([a], a)
    assert a.read_bytes() == original
