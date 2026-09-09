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
        # the soma is found by its TOKEN, never by a literal hex: the anatomy
        # register moved on 2026-09-08 and a hardcoded hue would pin the test
        # to a palette the gate has since replaced.
        from journal_style import COLORS
        soma_rgb = matplotlib.colors.to_rgb(COLORS["soma"])
        soma = [p for p in ax.patches
                if p.get_facecolor()[:3] == soma_rgb][0]
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


# ── 2026-09-08 spec upgrade: the library contract the builders rely on ────
def test_typeface_is_helvetica_class_and_no_dejavu(gallery_pdf: Path) -> None:
    """The gallery embeds the journal face and nothing from the fallbacks."""
    import fitz
    from journal_style import FORBIDDEN_FONT_MARKERS, SANS_FAMILY, SANS_STACK

    assert SANS_STACK[-1] == "DejaVu Sans"      # fallback, never the face
    embedded = [str(f[3]).split("+")[-1]
                for f in fitz.open(gallery_pdf)[0].get_fonts(full=False)]
    assert embedded, "no font embedded"
    for name in embedded:
        assert not any(bad.lower() in name.lower()
                       for bad in FORBIDDEN_FONT_MARKERS), name
        assert SANS_FAMILY.replace(" ", "") in name.replace(" ", ""), name


def test_type_scale_is_three_sizes_with_a_seven_point_floor() -> None:
    import figure_canvas as F
    import journal_style as J

    assert F.PT_TOKENS == (7.0, 8.0, 9.0)
    assert (J.PT_SMALL, J.PT_ANNOT, J.PT_LEGEND, J.PT_TICK) == (7.0,) * 4
    assert (J.PT_LABEL, J.PT_TITLE) == (8.0, 8.0)
    assert J.PANEL_LABEL_PT == 9.0 and J.PT_FLOOR == 7.0
    assert [J.snap_pt(v) for v in (6.8, 7.2, 7.4, 7.6, 8.4, 8.8, 10.5)] == \
        [7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0]


def test_stroke_rule_exempts_only_closed_filled_paths() -> None:
    import figure_canvas as F

    assert F.DECORATIVE_LW_PT == 1.35
    assert F.LW_TOKENS == (0.55, 0.7, 0.85, 0.95, 1.25)


def test_palette_gate_passes_and_fixes_the_reviewed_pairs() -> None:
    import journal_style as J

    report = J.palette_gate()
    assert not report["failures"], report["failures"]
    reviewed = {("shunting", "dend"): 5.9, ("bp", "inh"): 6.9,
                ("scalar", "soma"): 6.5, ("additive", "exc"): 8.3,
                ("point_mlp", "mute"): 3.1}
    for (series, anatomy), before in reviewed.items():
        row = [r for r in report["rows"]
               if r["series"] == series and r["anatomy"] == anatomy][0]
        assert row["normal"] >= 15.0, (series, anatomy, row["normal"], before)
        assert row["cvd"] >= 10.0, (series, anatomy, row["cvd"])


def test_tint_patch_is_a_fill_not_a_fat_stroke() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from journal_style import LW_HAIR, tint_patch

    fig, ax = plt.subplots(figsize=(2, 2))
    patch = tint_patch(ax, ("ribbon", [[(0.1, 0.1), (0.5, 0.6), (0.9, 0.2)]],
                            10.0), color="shunting")
    assert patch is not None
    assert patch.get_linewidth() == LW_HAIR
    assert patch.get_facecolor()[3] > 0          # it is filled
    plt.close(fig)


def test_letters_lock_to_the_module_column() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from figure_canvas import ALIGN_TOL_PT, NativeCanvas

    canvas = NativeCanvas(4.0, 2)
    for row in (0, 1):
        canvas.panel(f"l{row}", row, 0, 6)
        canvas.panel(f"r{row}", row, 6, 6)
    canvas.axes["r0"].set_ylabel("a very long y axis label indeed")
    assert canvas.align_letters() == []
    xs = {}
    for item in canvas._letters:
        xs.setdefault(item["col"], []).append(item["art"].get_position()[0])
    for col, values in xs.items():
        spread = (max(values) - min(values)) * canvas.width_pt
        assert spread <= ALIGN_TOL_PT, (col, spread)
    matplotlib.pyplot.close(canvas.fig)


def test_forest_centres_labels_and_always_draws_the_seed_fan() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from figure_canvas import NativeCanvas

    canvas = NativeCanvas(3.2, 1)
    ax = canvas.panel("f", 0, 0, 12)
    out = canvas.forest(ax, [
        {"label": "MNIST", "mean": 0.09, "lo": 0.04, "hi": 0.14,
         "seeds": [0.05, 0.09, 0.13], "n": 3},
        {"label": "CIFAR-10", "mean": 0.02, "lo": -0.02, "hi": 0.06,
         "seeds": [0.0, 0.02, 0.05], "n": 3},
    ], value_label="Accuracy gain (pp)")
    assert out["gutter_pt"] > 0 and "n = 3" in out["tag_text"]
    labels = {a.get_text(): a for a in ax.texts}
    assert labels["MNIST"].get_va() == "center"      # ON the row, not above
    fans = [ln for ln in ax.lines if ln.get_linestyle() == "None"
            and ln.get_markersize() < 4.0]
    assert len(fans) == 2, "the seed fan is not optional"
    matplotlib.pyplot.close(canvas.fig)


# ── 2026-09-08 glyph language: the rules, not only the shapes ─────────────
def _frame(width=12, height=3.0):
    import matplotlib
    matplotlib.use("Agg")
    from figure_canvas import Margins, NativeCanvas
    from native_schematics import Frame

    canvas = NativeCanvas(height, 1, margins=Margins(), lock_reserves=False,
                          letters=False)
    ax = canvas.panel("p", 0, 0, width, schematic=True, lock=False)
    return canvas, Frame(ax)


def test_every_tree_helper_draws_the_soma_lowest() -> None:
    """GLYPH RULE (a): the soma is the lowest node, in every tree helper."""
    import matplotlib
    from native_schematics import (Frame, SchematicRuleError,
                                   require_soma_lowest)

    canvas, f = _frame()
    balanced = f.balanced_tree((0.0, 0.0, 0.45, 1.0), mode="forward")
    sites = f.site_tree((0.5, 0.0, 0.45, 1.0))          # orient defaults 'up'
    for nodes in (balanced, sites):
        assert nodes.orient == "up"
        assert nodes.soma[1] <= min(xy[1] for xy in nodes.values()) + 1e-9
        require_soma_lowest(nodes, f)                    # does not raise
    # a tree drawn the other way up is refused, with the offenders named
    upside_down = type(sites)(sites)
    upside_down.soma = (0.5, 1.0)
    with pytest.raises(SchematicRuleError):
        require_soma_lowest(upside_down, f)
    matplotlib.pyplot.close(canvas.fig)


def test_site_tree_orient_right_is_legacy_and_matrices_take_a_strip() -> None:
    """A matrix aligns to a vertical site STRIP, not to a sideways tree."""
    import matplotlib
    import numpy as np
    from native_schematics import ORIENTATION_LEGACY, SchematicRuleError

    canvas, f = _frame()
    before = len(ORIENTATION_LEGACY)
    legacy = f.site_tree((0.0, 0.0, 0.3, 1.0), orient="right")
    assert legacy.row_axis == "y"                      # still alignable
    assert len(ORIENTATION_LEGACY) == before + 1       # and recorded as legacy
    upright = f.site_tree((0.35, 0.0, 0.2, 1.0))
    assert upright.row_axis == "x"
    strip = f.site_strip((0.6, 0.0, 0.06, 1.0))
    assert strip.row_axis == "y" and strip.kind == "site_strip"
    assert strip.order[:2] == [0, 3] and strip.pitch_pt >= 6.0
    A = np.eye(12)[:, :3]
    with pytest.raises(SchematicRuleError):
        f.dictionary_product((0.7, 0.0, 0.28, 1.0), A, np.ones(3),
                             align_to=upright)
    matplotlib.pyplot.close(canvas.fig)


def test_every_composition_carries_exactly_one_delta0_per_soma() -> None:
    """GLYPH RULE (b), asserted for all COMPOSITIONS."""
    import matplotlib
    from figure_canvas import Margins, NativeCanvas
    from journal_style import COLORS
    from native_schematics import COMPOSITIONS

    soma_rgb = matplotlib.colors.to_rgb(COLORS["soma"])
    for name, draw in COMPOSITIONS.items():
        canvas = NativeCanvas(3.4, 1, margins=Margins(), lock_reserves=False,
                              letters=False)
        ax = canvas.panel(name, 0, 0, 6, schematic=True, lock=False)
        draw(ax)                       # raises if the rule is broken
        somata = [p for p in ax.patches
                  if tuple(p.get_facecolor()[:3]) == soma_rgb]
        arrows = [t for t in ax.texts if t.get_text() == "δ"]
        assert somata, name
        assert len(arrows) == len(somata), (name, len(arrows), len(somata))
        matplotlib.pyplot.close(canvas.fig)


def test_delta0_exemption_needs_a_reason_and_is_recorded() -> None:
    import matplotlib
    from native_schematics import DELTA0_EXEMPTIONS, SchematicRuleError

    canvas, f = _frame()
    f.balanced_tree((0.0, 0.0, 1.0, 1.0), mode="plain")
    with pytest.raises(SchematicRuleError):
        f.require_delta0()                       # a soma with no δ0
    with pytest.raises(SchematicRuleError):
        f.require_delta0(allow_no_delta0=True)   # no reason given
    before = len(DELTA0_EXEMPTIONS)
    f.require_delta0(allow_no_delta0=True, reason="stimulus-only card")
    assert len(DELTA0_EXEMPTIONS) == before + 1
    assert any(n["kind"] == "delta0-exemption"
               for n in f.ax._journal_schematic_notes)
    matplotlib.pyplot.close(canvas.fig)


def test_gate_and_shunt_are_the_inhibitory_contact_family() -> None:
    """GLYPH RULE (3): filled red contact closed, open red inactive, badge."""
    import matplotlib
    from journal_style import COLORS
    from native_schematics import SchematicRuleError

    inh = matplotlib.colors.to_rgb(COLORS["inh"])
    canvas, f = _frame()
    nodes = f.balanced_tree((0.0, 0.0, 1.0, 1.0), mode="plain")
    n_before = len(f.ax.patches)
    f.gate(nodes["JL"], closed=False, nodes=nodes, node="JL")
    open_inner = f.ax.patches[n_before]
    n_before = len(f.ax.patches)
    f.gate(nodes["JR"], closed=True, nodes=nodes, node="JR",
           badge=("a", "p", " = 10"))
    closed_inner = f.ax.patches[n_before]
    assert tuple(open_inner.get_facecolor()[:3]) == (1.0, 1.0, 1.0)
    assert tuple(open_inner.get_edgecolor()[:3]) == inh
    assert tuple(closed_inner.get_facecolor()[:3]) == inh
    # closing a gate fades the descendant subtree; it never adds a new mark
    faded = [ln for ln in f.ax.lines
             if abs(ln.get_linewidth() - 0.55) < 1e-6]
    assert faded, "a closed gate must fade its subtree"
    # every badge is one 7 pt token
    assert all(t.get_fontsize() == 7.0 for t in f.ax.texts)
    # a red bar / free red segment is refused outright
    with pytest.raises(SchematicRuleError):
        f.dendrite((0.1, 0.1), (0.3, 0.1), color=COLORS["inh"])
    with pytest.raises(SchematicRuleError):
        f.rule(0.2, 0.1, 0.3, color=COLORS["gate"])
    matplotlib.pyplot.close(canvas.fig)


def test_exactly_four_delivery_glyphs_with_aliases() -> None:
    """GLYPH RULE (4): four modes, older names aliased, nothing else."""
    import matplotlib
    from native_schematics import DELIVERY_MODES, resolve_delivery_mode

    assert DELIVERY_MODES == ("scalar", "neuron", "subtree", "exact")
    assert resolve_delivery_mode("broadcast") == "scalar"
    assert resolve_delivery_mode("per_neuron") == "neuron"
    assert resolve_delivery_mode("ancestry") == "subtree"
    assert resolve_delivery_mode("path") == "exact"
    with pytest.raises(ValueError):
        resolve_delivery_mode("teleport")
    canvas, f = _frame()
    nodes = f.balanced_tree((0.05, 0.05, 0.72, 0.70), mode="plain")
    f.credit_delivery(nodes, mode="broadcast")           # alias draws
    assert any(t.get_text() == "s" for t in f.ax.texts), \
        "the layer scalar's source dot is tagged s"
    matplotlib.pyplot.close(canvas.fig)


def test_no_glyph_strokes_above_the_area_cap() -> None:
    """GLYPH RULE (5): every capsule / band / bar is a fill, not a fat stroke."""
    import matplotlib
    from figure_canvas import DECORATIVE_LW_PT
    from native_schematics import SCHEMATICS, Frame
    from figure_canvas import Margins, NativeCanvas

    for name in ("anatomy_pipeline", "route_resolution"):
        canvas = NativeCanvas(3.0, 1, margins=Margins(), lock_reserves=False,
                              letters=False)
        ax = canvas.panel(name, 0, 0, 12, schematic=True, lock=False)
        SCHEMATICS[name](ax)
        wide = [ln.get_linewidth() for ln in ax.lines
                if ln.get_linewidth() >= DECORATIVE_LW_PT]
        assert not wide, (name, wide)
        matplotlib.pyplot.close(canvas.fig)


def test_matrix_rule_asserts_cells_headers_and_offers_collapsed_bands() -> None:
    """GLYPH RULE (6): >= 6 pt per row and column, header <= 1.5 x column."""
    import matplotlib
    import numpy as np
    from native_schematics import MIN_CELL_PT, MatrixTooDense

    assert MIN_CELL_PT == 6.0
    canvas, f = _frame(height=4.0)
    A = np.eye(24)[:, :4]
    tall = (0.0, 0.0, f.fx(40.0), f.fy(60.0))     # 24 rows in 60 pt
    with pytest.raises(MatrixTooDense):
        f.dictionary_matrix(tall, A, label=None)  # no bands: no remedy
    f.dictionary_matrix(tall, A, label=None, row_groups=[6, 6, 6, 6])
    printed = [t.get_text() for t in f.ax.texts]
    assert "rows collapsed: 6 per band" in printed
    with pytest.raises(MatrixTooDense):           # 12 columns in 40 pt
        f.dictionary_matrix((0.4, 0.0, f.fx(40.0), f.fy(60.0)),
                            np.zeros((4, 12)), label=None)
    with pytest.raises(MatrixTooDense):           # header over 1.5 x column
        f.dictionary_matrix((0.4, 0.0, f.fx(30.0), f.fy(40.0)),
                            np.zeros((4, 3)), label=None,
                            col_labels=["a very long header", "b", "c"])
    matplotlib.pyplot.close(canvas.fig)


@pytest.mark.parametrize("span", [6, 12])
def test_compositions_survive_six_and_twelve_module_widths(tmp_path, span):
    """The spec's regression sheet: every composition at both slot widths.

    A glyph that only works at its design width is not a glyph.  The sheets
    are two rows of 12/span panels on a full-size page, so the canvas rules
    (aspect, fill) apply exactly as they do to a real figure, and the strict
    audit's text-over-data and text-collision checks are what "no label
    overlaps any mark" means here -- rendered and audited, never eyeballed.
    """
    import matplotlib
    from figure_canvas import Margins, NativeCanvas, audit_native_pdf
    from native_schematics import COMPOSITIONS

    per_row = 12 // span
    names = list(COMPOSITIONS)
    pages = [names[i:i + 2 * per_row] for i in range(0, len(names), 2 * per_row)]
    for p, page in enumerate(pages):
        canvas = NativeCanvas(6.8, 2, hgutter_pt=26.0, vgutter_pt=26.0,
                              margins=Margins(left=34.0, right=12.0, top=18.0,
                                              bottom=12.0),
                              lock_reserves=False)
        for i, name in enumerate(page):
            row, col = divmod(i, per_row)
            canvas.panel(name, row, col * span, span, schematic=True,
                         lock=False, letter=chr(ord("A") + i))
            COMPOSITIONS[name](canvas.axes[name])
        out = tmp_path / f"glyph_regression_{span}_{p}.pdf"
        problems = canvas.save(out, png=False, quiet=True)
        assert not problems, (page, problems)
        report = audit_native_pdf(out, strict=True, report=True)
        assert not report.violations, (page,
                                       [str(v) for v in report.violations])
        matplotlib.pyplot.close(canvas.fig)
