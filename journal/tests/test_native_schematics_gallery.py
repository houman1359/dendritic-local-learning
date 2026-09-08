"""The shared schematic glyph library must emit only journal tokens.

Builds the helper gallery (every Frame helper family and the four
figure-specific compositions on one native canvas) into a temporary
directory and asserts the strict PDF audit reports no violation, so a
change to any helper that introduces a foreign type size, stroke weight,
text collision or layout defect fails here before it reaches a figure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("fitz")

SPEC = importlib.util.spec_from_file_location(
    "native_schematics_gallery", SCRIPTS / "native_schematics_gallery.py")
GALLERY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GALLERY)


@pytest.fixture(scope="module")
def gallery_pdf(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gallery") / "native_schematics_gallery.pdf"
    problems = GALLERY.build(out, png=False, quiet=True)
    assert out.is_file()
    assert not problems, problems
    return out


def test_gallery_passes_strict_audit(gallery_pdf: Path) -> None:
    from figure_canvas import audit_native_pdf

    report = audit_native_pdf(gallery_pdf, strict=True, report=True)
    assert not report.violations, [str(v) for v in report.violations]


def test_gallery_draws_every_helper_family(gallery_pdf: Path) -> None:
    import fitz

    page = fitz.open(gallery_pdf)[0]
    text = page.get_text()
    for needle in ("Anatomy glyphs", "Four delivery modes",
                   "Dictionary triplet", "Cards and badges",
                   "Teacher and chance", "Partition, shunt", "Local gate tree",
                   "Stage pair", "Operator tree pair", "Measured boundary",
                   "teacher", "chance", "oracle", "local rule", "control",
                   "descendants", "uninhibited", "inhibited", "global"):
        assert needle in text, needle


def test_palette_aliases_add_no_hue() -> None:
    from journal_style import COLORS

    assert COLORS["gate"] == COLORS["inh"]
    assert COLORS["credit_ink"] == COLORS["ink"]
    assert COLORS["scalar"] == COLORS["local"]


def test_balanced_tree_topology_and_point_sizing() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from figure_canvas import NativeCanvas, Margins
    from native_schematics import Frame

    canvas = NativeCanvas(3.0, 1, margins=Margins(), lock_reserves=False,
                          letters=False)
    wide = canvas.panel("wide", 0, 0, 7, schematic=True, lock=False)
    narrow = canvas.panel("narrow", 0, 7, 4, schematic=True, lock=False)
    for ax in (wide, narrow):
        f = Frame(ax)
        nodes = f.balanced_tree((0.0, 0.0, 1.0, 1.0), mode="forward",
                                input_labels=[("x", str(i)) for i in range(1, 9)])
        assert nodes.route("T4") == ["S", "J1", "JL", "JLR", "T4"]
        assert nodes.terminals_under("JR") == ["T5", "T6", "T7", "T8"]
        assert nodes.at_depth(4) == ["JLL", "JLR", "JRL", "JRR"]
        # every glyph is sized in points: the soma disc is the same size in
        # both slots even though the tree geometry is not
        soma = [p for p in ax.patches if p.get_facecolor()[:3] == matplotlib.colors.to_rgb("#E8873C")][0]
        assert abs(soma.width * f.w_pt - 6.0) < 1e-6
        f.credit_delivery(nodes, mode="exact", alpha_tags=True)
        f.credit_delivery(nodes, mode="scalar")
        f.credit_delivery(nodes, mode="subtree", K=4)
        f.credit_delivery(nodes, mode="neuron")
        f.gate(nodes["JR"], closed=True, descendants=["JR"], nodes=nodes,
               node="JR")
        with pytest.raises(ValueError):
            f.credit_delivery(nodes, mode="teleport")
        with pytest.raises(ValueError):
            f.badge((0.5, 0.5), "hero")
    matplotlib.pyplot.close(canvas.fig)
