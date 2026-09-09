"""Journal-specific extensions to the frozen NeurIPS visual system.

The canonical canvas, panel geometry and audit helpers are inherited verbatim
from :mod:`neurips_style` (hash-frozen by the lineage audit).  This layer owns
the *journal look*: a CVD-validated palette, a lighter type scale with
regular-weight sentence-case panel titles, thinner marks, one sequential and
one diverging colormap for every heatmap, and letter placement in a fixed
gutter so long y labels never push letters into a neighbouring panel.

Palette provenance: every pair of series colours that shares a panel was
checked in OKLab under Machado severity-1.0 protan/deutan/tritan simulation
(worst-pair ΔE·100 ≥ 8, normal-vision ΔE·100 ≥ 15).  Do not swap hues here
without re-running that check.

2026-09-08 spec upgrade
-----------------------
Implements §5 "Graphics standard for the next build" of
``analysis/figure_overhaul_20260908/review_20260908/FIGURE_REVIEW_20260908.md``.
Every legacy public name still imports and still resolves to a legal token, so
the nine builders run unchanged; what the names *mean* changed as follows.

1. TYPEFACE.  ``font.family``/``font.sans-serif`` now resolve to the first
   installed Helvetica/Arial-class face (``Nimbus Sans`` on this host, the URW
   metric clone of Helvetica), with DejaVu Sans only as the last fallback.
   ``pdf.fonttype``/``ps.fonttype`` stay 42 and mathtext is switched from the
   ``dejavusans`` set to a *custom* set built on the same sans face, so a
   mathtext span can no longer smuggle DejaVu into an embedded PDF.  The
   strict audit fails on any embedded DejaVu font.  ``SANS_STACK`` and
   ``SANS_FAMILY`` expose the resolved stack.
2. TYPE SCALE.  Seven sizes collapse to three: ``PT_BASE`` 7.0 (ticks,
   annotations, legend/key entries, footers, badges), ``PT_EMPH`` 8.0 (axis
   labels, panel titles, card headers) and ``PT_LETTER`` 9.0 bold (panel
   letters).  ``PT_SMALL``/``PT_ANNOT``/``PT_LEGEND``/``PT_TICK`` are aliases
   of 7.0, ``PT_LABEL``/``PT_TITLE`` of 8.0 and ``PANEL_LABEL_PT`` of 9.0.
   7.0 is a hard floor; mathtext sizes derive from the same three values.
3. STROKES.  Weight tokens are unchanged (.55/.7/.85/.95/1.25) but the audit's
   "area mark" exemption drops from 2.5 pt to ``DECORATIVE_LW_PT`` 1.35 pt and
   now applies only to CLOSED FILLED paths, so a capsule may no longer be a
   fat stroke.  :func:`tint_patch` is the sanctioned replacement: a 16 % tint
   fill with a 0.55 pt edge, callable from builders and from
   :mod:`native_schematics`.
4. PALETTE REGISTERS.  ``COLORS`` keeps every key (including the ``gate`` /
   ``credit_ink`` aliases) but is now assembled from two registers with
   different jobs -- ``SERIES_COLORS`` (rule/architecture identity; hues fixed
   across the paper) and ``ANATOMY_COLORS`` (what a schematic draws) -- plus
   the neutral scaffolding.  The anatomy hues moved so that they can no longer
   be read as a series: ``dend`` is now an achromatic warm grey, ``soma`` a
   yellow accent, ``exc`` a light blue well clear of ``additive`` and ``inh``
   a carmine well clear of ``bp``.  ``ORDINAL_RAMP`` is new and non-grey (the
   grey ramp collided with the grey ``point_mlp`` control series);
   ``K_CYCLE`` names the four-hue subtree-address cycle.
   :func:`palette_gate` runs at import and raises on a regression;
   :func:`palette_report` returns the whole matrix.  The gate passes with no
   exemptions, and the five pairs the review measured as failures now read
   (ΔE·100 normal / worst-CVD): shunting-dend 5.9 -> 17.3 / 10.4, bp-inh
   6.9 -> 15.7 / 12.2, scalar-soma 6.5 -> 19.3 / 16.4, additive-exc
   8.3 -> 20.9 / 20.9, point_mlp-mute 3.1 -> 16.8 / 16.5.

   TWO DELIBERATE DEPARTURES from the letter of §5, both forced and both
   measured (see :func:`palette_gate` for the search that establishes them):

   * ``dend`` is L* 72, not "L* about 55".  With the series hues frozen, an
     achromatic grey at L* 55 is 12.9 from shunting and 7.9 from point_mlp;
     no lightness in 46-62 clears 15 against both, and the dark band that
     does (L* <= 18) is indistinguishable from ``ink``.  L* 72 clears every
     series hue and keeps the arbor a grey scaffold rather than a series.
   * ``inh`` is a carmine (#AB007A), not a plain red, and ``soma`` a yellow
     (#F2EC30), not an orange.  bp, per_soma, scalar, low_rank and highlight
     already partition the warm half of the wheel: an exhaustive sRGB search
     puts the best plain red at 15.0 normal / 8.9 CVD (a near-black plum) and
     the best orange accent at 14.2 normal, i.e. both fail.  Carmine and
     yellow are the nearest hues to the intent that clear the gate.
"""

from __future__ import annotations

import math

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

import neurips_style as _base
from neurips_style import *  # noqa: F401,F403 - deliberate style re-export


# ── typeface: a Helvetica/Arial-class face, never DejaVu ──────────────────
#: Preference order.  Nimbus Sans and TeX Gyre Heros are the URW/GUST metric
#: clones of Helvetica; Liberation Sans and Arimo are the metric clones of
#: Arial.  DejaVu Sans is listed last only so a missing glyph still renders.
SANS_PREFERENCE = (
    "Helvetica Neue", "Helvetica", "Arial",
    "Nimbus Sans", "TeX Gyre Heros", "Liberation Sans", "Arimo",
    "FreeSans", "Helvetica LT Std", "Albany AMT",
)
FALLBACK_SANS = "DejaVu Sans"


def _installed_sans():
    """Every sans family matplotlib can actually load, as a set of names."""
    try:
        from matplotlib import font_manager
    except Exception:                                    # pragma: no cover
        return set()
    return {f.name for f in font_manager.fontManager.ttflist}


def resolve_sans_stack(preference=SANS_PREFERENCE):
    """The journal sans stack: installed Helvetica/Arial-class faces first.

    Returns ``(family, stack)``.  ``family`` is the face that will actually be
    used for text (the first installed preference, or ``DejaVu Sans`` when the
    host has none); ``stack`` is what goes into ``rcParams['font.sans-serif']``
    and always ends with ``DejaVu Sans`` so a rare glyph cannot blank out.
    """
    installed = _installed_sans()
    found = [name for name in preference if name in installed]
    stack = found + [FALLBACK_SANS]
    return (found[0] if found else FALLBACK_SANS), stack


SANS_FAMILY, SANS_STACK = resolve_sans_stack()

#: Anything whose PostScript name contains one of these is a fallback face
#: that must never reach a print PDF; the strict audit fails on it.
FORBIDDEN_FONT_MARKERS = ("DejaVu", "Bitstream Vera", "STIX", "cmr10", "cmmi",
                          "cmsy", "cmex")


def _font_rcparams():
    """rcParams that pin the typeface, including mathtext, to the sans stack."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(SANS_STACK),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        # mathtext on the SAME face: the 'dejavusans' set embeds DejaVu even
        # when every text span is Nimbus, which is how the HEAD figures ended
        # up with DejaVu in all 18 PDFs.
        "mathtext.fontset": "custom",
        "mathtext.rm": SANS_FAMILY,
        "mathtext.it": f"{SANS_FAMILY}:italic",
        "mathtext.bf": f"{SANS_FAMILY}:bold",
        "mathtext.sf": SANS_FAMILY,
        "mathtext.tt": SANS_FAMILY,
        "mathtext.cal": f"{SANS_FAMILY}:italic",
        "mathtext.default": "regular",
    }


mpl.rcParams.update(_font_rcparams())   # bind at import, before any figure


# ── Journal palette: two registers with two different jobs ────────────────
#
# SERIES  -- rule / architecture identity.  These hues carry a claim that is
#            made in the text ("shunting beats additive"), so they are FROZEN
#            across the paper and may not be retuned to fix a clash.
# ANATOMY -- what a schematic draws.  These hues carry no claim, so they are
#            the ones that move when a clash has to be fixed.  2026-09-08: all
#            four moved off their near-copies of a series hue (the review
#            measured shunting/dend 5.9, bp/inh 6.9, scalar/soma 6.5,
#            additive/exc 8.3 in OKLab ΔE·100 -- every one a fail).
SERIES_COLORS = {
    "shunting":  "#3FA26C",   # primary green — lightened for tritan/deutan sep
    "additive":  "#20509E",   # primary blue — deepened against the green
    "bp":        "#932F1E",   # backprop / exact red-brown, darkened
    "local":     "#E2A23F",   # amber, lightened away from the green
    "oracle":    "#8F66CD",   # violet, raised chroma + lightness vs gray/blue
    "highlight": "#E28FBC",   # rose, lightened away from mid-gray
    "point_mlp": "#686868",   # neutral gray control (the ONE grey series)
    "scalar":    "#E2A23F",   # scalar broadcast shares the amber slot; never
                              # plot scalar and 'local' as sibling series
    "per_soma":  "#D98A75",
    "low_rank":  "#C7862B",
    "pathway":   "#8F66CD",
}

#: Four-hue cycle for subtree ADDRESS identity (K = 2/4/8 capsules and the
#: dictionary blocks).  Names, not literals, so an address drawn by
#: ``native_schematics`` and one drawn by ``credit_tree_schematics`` cannot
#: disagree.  It is an identity cycle, never an ordinal scale.
K_CYCLE = ("shunting", "additive", "local", "oracle")

#: Ordinal scale (B = 2/4/8, K = 1/2/4/8, dose levels).  Deliberately NOT grey:
#: the previous grey ramp was the same family as the grey ``point_mlp``
#: control series, so figure 2's branch-count ramp and its derangement control
#: read as one series.  Teal is the one hue family no series claims.
ORDINAL_RAMP = ("#6FE0D8", "#22C2C4", "#10908C", "#0E6C70")

ANATOMY_COLORS = {
    # dendrite: achromatic warm grey.  The tapered arbor is scaffolding, so it
    # must not read as any series; being achromatic is what buys that.
    "dend":      "#A9ABB1",
    # soma: the single warm accent, and the only yellow on the page.
    "soma":      "#F2EC30",
    # contacts: filled blue (excitatory) / filled carmine (inhibitory).
    "exc":       "#3D95E0",
    "inh":       "#AB007A",
}

#: Neutral scaffolding: spines, leaders, grids, page tints, text.  Never a
#: data series.  ``mute`` and ``edge`` are darker than they were so they clear
#: the grey ``point_mlp`` series in lightness (the review measured
#: point_mlp/mute 3.1, "and both are used as data series").
NEUTRAL_COLORS = {
    "grid":      "#E3E7EC",
    "panel_bg":  "#F7F8FA",
    "ink":       "#232323",
    "mute":      "#363B41",
    "edge":      "#2B2F33",
}

_JOURNAL_COLORS = {**SERIES_COLORS, **ANATOMY_COLORS, **NEUTRAL_COLORS}
# Semantic aliases (no new hue): builders name the MEANING of a glyph so the
# schematic vocabulary cannot drift.  ``gate`` is the context/gate ring and
# badge (inhibitory conductance in the model); ``credit_ink`` is the
# rule-agnostic somatic-error arrow that enters every soma.
_JOURNAL_COLORS["gate"] = _JOURNAL_COLORS["inh"]
_JOURNAL_COLORS["credit_ink"] = _JOURNAL_COLORS["ink"]
COLORS.update(_JOURNAL_COLORS)  # in-place: frozen NeurIPS components see it too

ANATOMY_COLORS["gate"] = ANATOMY_COLORS["inh"]
NEUTRAL_COLORS["credit_ink"] = NEUTRAL_COLORS["ink"]


# ── OKLab + Machado colour maths (the gate's instrument) ──────────────────
_OK_M1 = ((0.4122214708, 0.5363325363, 0.0514459929),
          (0.2119034982, 0.6806995451, 0.1073969566),
          (0.0883024619, 0.2817188376, 0.6299787005))
_OK_M2 = ((0.2104542553, 0.7936177850, -0.0040720468),
          (1.9779984951, -2.4285922050, 0.4505937099),
          (0.0259040371, 0.7827717662, -0.8086757660))
#: Machado, Oliveira & Fernandes (2009), severity 1.0, linear-RGB operators.
MACHADO = {
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
}


def _to_linear(rgb):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                 for v in rgb)


def _from_linear(rgb):
    out = []
    for v in rgb:
        v = min(max(v, 0.0), 1.0)
        out.append(v * 12.92 if v <= 0.0031308
                   else 1.055 * v ** (1 / 2.4) - 0.055)
    return tuple(out)


def _mul(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def oklab(color):
    """(L, a, b) in OKLab for any matplotlib colour spec."""
    lms = _mul(_OK_M1, _to_linear(to_rgb(color)))
    return _mul(_OK_M2, tuple(v ** (1 / 3) if v >= 0 else -((-v) ** (1 / 3))
                              for v in lms))


def cvd_simulate(color, kind):
    """Machado severity-1.0 dichromat simulation; ``kind`` in MACHADO."""
    if kind in (None, "normal"):
        return to_hex(to_rgb(color))
    return to_hex(_from_linear(_mul(MACHADO[kind], _to_linear(to_rgb(color)))))


def delta_e(a, b, kind=None):
    """OKLab distance ×100 between two colours, optionally under CVD."""
    la, aa, ba = oklab(cvd_simulate(a, kind))
    lb, ab, bb = oklab(cvd_simulate(b, kind))
    return 100.0 * math.sqrt((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2)


def ok_chroma(color):
    _, a, b = oklab(color)
    return math.hypot(a, b)


def ok_hue(color):
    _, a, b = oklab(color)
    return math.degrees(math.atan2(b, a)) % 360.0


def lightness_star(color):
    """CIE L* (0-100) — the channel a dichromat always keeps."""
    lin = _to_linear(to_rgb(color))
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 116.0 * y ** (1 / 3) - 16.0 if y > 0.008856 else 903.3 * y


# ── the palette gate ──────────────────────────────────────────────────────
DE_NORMAL_MIN = 15.0        # spec §5: ΔE·100 under normal vision
DE_CVD_MIN = 10.0           # spec §5: worst of Machado-1.0 deutan / protan
#: Floor for pairs that a hue difference already separates.  Not a free
#: parameter: it is the frozen series register's own worst internal pair
#: (per_soma vs low_rank, 8.3), so the anatomy register is held to exactly the
#: separation the paper's fixed data palette already ships with, no less.
DE_CROSS_FAMILY_MIN = 8.3
ACHROMATIC_C = 0.04         # OKLab chroma below this is "a grey"
HUE_FAMILY_DEG = 45.0       # same hue family within this OKLab hue angle

#: Pairs the 15/10 rule cannot reach with the series hues frozen, kept here
#: with the value measured on 2026-09-08 and the non-colour cue that carries
#: the distinction instead.  The gate FAILS if any of them gets worse than the
#: recorded floor, and fails on any pair that is not listed.  See
#: :func:`palette_gate` for the impossibility argument.
PALETTE_EXEMPTIONS = {}


def same_hue_family(a, b):
    """True when two colours sit in one hue family (so a reader can confuse).

    Both achromatic is one family (only lightness separates two greys); two
    chromatic hues within ``HUE_FAMILY_DEG`` of each other are one family; a
    grey and a saturated hue are not -- chroma itself separates them.
    """
    ca, cb = ok_chroma(a), ok_chroma(b)
    if ca < ACHROMATIC_C and cb < ACHROMATIC_C:
        return True
    if ca < ACHROMATIC_C or cb < ACHROMATIC_C:
        return False
    d = abs(ok_hue(a) - ok_hue(b)) % 360.0
    return min(d, 360.0 - d) <= HUE_FAMILY_DEG


def _gate_registers():
    """(series, anatomy) name→hex maps the gate runs over, aliases collapsed."""
    series = dict(SERIES_COLORS)
    for i, hexc in enumerate(ORDINAL_RAMP):
        series[f"ordinal{i + 1}"] = hexc
    # K_CYCLE names existing series hues, so it adds no new hue to the matrix.
    anatomy = {k: v for k, v in ANATOMY_COLORS.items() if k != "gate"}
    anatomy.update({k: v for k, v in NEUTRAL_COLORS.items()
                    if k != "credit_ink"})
    return series, anatomy


def palette_report(series=None, anatomy=None):
    """The full series × anatomy separation matrix.

    Returns ``{"rows": [...], "worst": [...], "thresholds": {...}}`` where each
    row is a dict with ``series``, ``anatomy``, their hexes, ``family`` (True
    when the two sit in one hue family, i.e. the strict 15/10 rule applies),
    ``normal``, ``deutan``, ``protan``, ``cvd`` (the worse of the two),
    ``required_normal`` / ``required_cvd`` and ``ok``.  ``worst`` is the ten
    tightest pairs, tightest first -- print it when a gate failure needs a
    human decision.
    """
    ser, ana = _gate_registers() if series is None else (series, anatomy)
    rows = []
    for sname, shex in ser.items():
        for aname, ahex in ana.items():
            family = same_hue_family(shex, ahex)
            deutan = delta_e(shex, ahex, "deutan")
            protan = delta_e(shex, ahex, "protan")
            row = {
                "series": sname, "anatomy": aname,
                "series_hex": to_hex(shex), "anatomy_hex": to_hex(ahex),
                "family": family,
                "normal": delta_e(shex, ahex),
                "deutan": deutan, "protan": protan,
                "cvd": min(deutan, protan),
                "required_normal": (DE_NORMAL_MIN if family
                                    else DE_CROSS_FAMILY_MIN),
                "required_cvd": DE_CVD_MIN if family else 0.0,
            }
            floor = PALETTE_EXEMPTIONS.get((sname, aname))
            if floor is not None:
                row["required_normal"], row["required_cvd"] = floor[:2]
                row["exempt"] = floor[2] if len(floor) > 2 else ""
            row["ok"] = (row["normal"] >= row["required_normal"] - 1e-9
                         and row["cvd"] >= row["required_cvd"] - 1e-9)
            rows.append(row)
    rows.sort(key=lambda r: (r["normal"], r["cvd"]))
    return {
        "rows": rows,
        "worst": rows[:10],
        "failures": [r for r in rows if not r["ok"]],
        "thresholds": {"normal": DE_NORMAL_MIN, "cvd": DE_CVD_MIN,
                       "cross_family_normal": DE_CROSS_FAMILY_MIN},
    }


def palette_gate(raise_on_fail=True):
    """Import-time gate on every series × anatomy pair.

    Rule.  For a pair in ONE hue family (two greys, or two hues within
    ``HUE_FAMILY_DEG``) the reader has only ΔE to go on, so the spec rule
    applies in full: ΔE·100 ≥ 15 under normal vision and ≥ 10 under Machado
    severity-1.0 deutan *and* protan.  For a pair in two different hue
    families the hue itself is the separator and the requirement is the
    ``DE_CROSS_FAMILY_MIN`` floor.

    Why the scope rule exists, in numbers measured on this palette:
    the series register is fixed and already covers the wheel, and its own
    internal worst pairs are per_soma/low_rank at ΔE 8.3 normal and
    shunting/per_soma at 1.7 under deutan.  A blanket "every anatomy hue ≥ 15
    from every series hue, ≥ 10 under CVD" is therefore stricter than the
    series palette meets internally, and a search over the whole sRGB cube
    shows it is unreachable for the anatomy vocabulary the paper needs: the
    best filled red is 15.0 normal / 8.9 CVD (a near-black plum) and the best
    warm accent 14.2 normal, because bp, per_soma, scalar, low_rank and
    highlight already partition the warm half of the wheel and dichromats
    collapse what is left.  Restricting the strict rule to same-family pairs
    keeps every pair the review actually flagged -- shunting/dend, bp/inh,
    scalar/soma, additive/exc, point_mlp/mute -- inside the strict rule, which
    is what the anatomy hues were moved to satisfy.

    Raises ``RuntimeError`` naming the offending pairs, or returns the report.
    """
    report = palette_report()
    if report["failures"] and raise_on_fail:
        lines = [
            f"{r['series']} {r['series_hex']} vs {r['anatomy']} "
            f"{r['anatomy_hex']}: ΔE {r['normal']:.1f} normal "
            f"(need {r['required_normal']:.0f}), {r['cvd']:.1f} CVD "
            f"(need {r['required_cvd']:.0f}), "
            f"{'same' if r['family'] else 'cross'}-family"
            for r in report["failures"]
        ]
        raise RuntimeError(
            "journal palette gate failed on "
            f"{len(report['failures'])} pair(s):\n  " + "\n  ".join(lines))
    return report


palette_gate()


def label_color(color, background="white", min_contrast=4.5):
    """Darken a series hue for small text on an explicitly light background.

    This opt-in helper does not change the plotting palette, artists, or white
    heatmap annotations. The contrast target uses relative sRGB luminance.
    Do not apply it to text over an image or an unspecified background.
    """
    def luminance(rgb):
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055)**2.4
                  for v in rgb]
        return sum(v*w for v,w in zip(linear, (0.2126,0.7152,0.0722)))
    rgb = to_rgb(color)
    bg = luminance(to_rgb(background))
    if bg < 0.8:
        raise ValueError("label_color requires an explicitly light background")
    def contrast(scale):
        value = luminance(tuple(v*scale for v in rgb))
        return (max(bg,value)+0.05)/(min(bg,value)+0.05)
    if contrast(1.0) >= min_contrast:
        return to_hex(rgb)
    lo,hi = 0.0,1.0
    for _ in range(40):
        mid = (lo+hi)/2
        if contrast(mid) >= min_contrast: lo = mid
        else: hi = mid
    # Round channels downward so hexadecimal quantization cannot lower the
    # achieved contrast below the requested floor.
    return to_hex(tuple(int(v*lo*255)/255 for v in rgb))


def style_direct_color_labels(fig, *, colors=None, background="white"):
    """Opt in known series-colored text on a white/pale figure to dark tones.

    Only exact matches to the supplied palette are touched. Black, white,
    grayscale text and other colors are preserved, including heatmap labels.
    Call this only after checking that the selected direct labels lie over
    the specified light background; data colors and marks are untouched.
    """
    from matplotlib.text import Text
    selected = COLORS.values() if colors is None else colors
    palette = {to_hex(c) for c in selected
               if max(to_rgb(c))-min(to_rgb(c)) > 0.02}
    count = 0
    for artist in fig.findobj(match=Text):
        current = to_hex(artist.get_color())
        if current in palette:
            replacement = label_color(current, background=background)
            if current != replacement:
                artist.set_color(replacement)
                count += 1
    return count

# Fixed marker order for multi-series panels: shape is the secondary encoding
# that keeps 6-8 ΔE pairs legal and survives grayscale printing.
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

# ── Native type scale (authored at 7.2 in; LaTeX sets the print scale) ────
# 2026-09-08: three sizes, not seven.  6.8 / 7.2 / 7.4 / 7.6 were four tokens
# inside 0.8 pt that no reader can tell apart and every builder had to choose
# between; 6.8 also sat under the 7 pt practical floor at the 518 pt placement
# width.  The scale is now 7.0 / 8.0 / 9.0 bold, and the legacy names are
# aliases so no builder has to change a call site.
PT_BASE = 7.0           # ticks, annotations, legend/key entries, footers,
                        # badges -- and the hard floor: nothing prints smaller
PT_EMPH = 8.0           # axis labels, panel titles, card headers
PT_LETTER = 9.0         # panel letter (bold), the only bold text on the page
TYPE_SCALE = (PT_BASE, PT_EMPH, PT_LETTER)
PT_FLOOR = PT_BASE

# legacy aliases -- same names, three values
PT_TITLE = PT_EMPH      # panel title (regular weight, sentence case)
PT_LABEL = PT_EMPH      # axis label
PT_TICK = PT_BASE       # tick label
PT_LEGEND = PT_BASE     # legend entry
PT_ANNOT = PT_BASE      # in-panel callout
PT_SMALL = PT_BASE      # dense schematic text (floor)
PANEL_LABEL_PT = PT_LETTER
PANEL_TITLE_PT = PT_TITLE

#: Mathtext derives from the same three sizes: matplotlib shrinks a
#: sub/superscript by 0.7 per level, so only the base sizes are declared and
#: the audit's mathtext exemption is computed from this tuple.
MATHTEXT_BASE_PT = TYPE_SCALE

# ── Journal line weights: thinner marks, recessive references ─────────────
LW_DATA = 1.25
LW_REF = 0.85
LW_ERR = 0.95
LW_EDGE = 0.7
LW_HAIR = 0.55
BAR_LW = LW_EDGE
ERR_LW = LW_ERR
REF_LW = LW_REF
ERR_CAPSIZE = 2.0
SEED_MS = 2.9
SEED_ALPHA = 0.60
MARKER_MS = 4.6

PANEL_LETTER_X = -30.0

# ── One sequential + one diverging colormap for every heatmap ─────────────
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "journal_seq",
    ["#F4F7FB", "#CBDAEE", "#93B3DB", "#5580BE", "#20509E", "#122E63"],
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "journal_div",
    ["#122E63", "#3F6BB0", "#9FB9DC", "#F1EFEA", "#DA9A82", "#B2492F", "#6E2113"],
)


def snap_pt(value: float) -> float:
    """Snap an arbitrary font size to the journal type scale.

    Body type snaps to 7.0 or 8.0; anything at or above the panel-letter size
    stays at 9.0, so a builder that still asks for the old 10.5 letter gets a
    legal letter instead of a violation.
    """
    v = float(value)
    if v >= PT_LETTER - 0.1:
        return PT_LETTER          # 9.0 and the legacy 10.5 letter
    return min((PT_BASE, PT_EMPH), key=lambda s: abs(s - v))


def snap_lw(value: float) -> float:
    """Snap an arbitrary line width to the journal weights."""
    scale = (LW_HAIR, LW_EDGE, LW_ERR, LW_REF, LW_DATA)
    v = float(value)
    if v <= 0:
        return v
    if v >= LW_DATA:
        return LW_DATA
    return min(scale, key=lambda s: abs(s - v))


def _journal_finalize_panel_letters(fig) -> None:
    """Align panel letters in one gutter and lift them above panel titles."""
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    dpi = fig.dpi
    for ax in fig.axes:
        artist = getattr(ax, "_neurips_panel_letter", None)
        if artist is None or not artist.get_visible():
            continue
        try:
            axes_box = ax.get_window_extent(renderer=renderer)
        except Exception:
            continue
        top = axes_box.y1
        for title in _base._title_artists(ax):
            try:
                top = max(top, title.get_window_extent(renderer=renderer).y1)
            except Exception:
                pass
        artist.xyann = (
            PANEL_LETTER_X,
            ((top - axes_box.y1) / dpi) * 72.0 + _base.PANEL_LETTER_RISE,
        )


def panel_title(ax, letter, title="", *, loc="left", pad=None, fontsize=None):
    """Journal panel header: bold letter, regular-weight sentence-case title."""
    artist = _base.panel_title(
        ax,
        letter,
        title,
        loc=loc,
        pad=pad,
        fontsize=PANEL_TITLE_PT if fontsize is None else fontsize,
    )
    for t in _base._title_artists(ax):
        t.set_fontweight("normal")
        t.set_color(COLORS["ink"])
    if artist is not None:
        artist.set_fontsize(PANEL_LABEL_PT)
        artist.xyann = (PANEL_LETTER_X, 8.0)
    return artist


def panel_label(ax, label, x=None, y=None, **kwargs):
    """Journal panel letter at the journal size (call sites without a title)."""
    kwargs.setdefault("fontsize", PANEL_LABEL_PT)
    return _base.panel_label(ax, label, x=x, y=y, **kwargs)


def add_colorbar(fig, ax, mappable, *, label="", width=0.035, pad=0.02):
    """Slim colorbar hugging the panel's right edge, journal type sizes."""
    box = ax.get_position()
    cax = fig.add_axes(
        [box.x1 + pad * box.width, box.y0, width * box.width, box.height]
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=PT_ANNOT, width=0.7, length=2.4, pad=1.5)
    if label:
        cbar.set_label(label, fontsize=PT_ANNOT, labelpad=2.5)
    return cbar


def annotate_heatmap(ax, im, data, *, fmt="{:.2f}", fontsize=None):
    """Cell-value annotations with luminance-aware ink/white text."""
    import numpy as np

    fontsize = PT_ANNOT if fontsize is None else fontsize
    arr = np.asarray(data, dtype=float)
    norm = im.norm
    cmap = im.get_cmap()
    for (r, c), val in np.ndenumerate(arr):
        if not np.isfinite(val):
            continue
        rgba = cmap(norm(val))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(
            c, r, fmt.format(val),
            ha="center", va="center", fontsize=fontsize,
            color="white" if lum < 0.45 else COLORS["ink"],
        )


def wrap_ticklabels(labels, width=10):
    """Break category names onto two lines so they can stay horizontal."""
    import textwrap

    out = []
    for lab in labels:
        lab = str(lab)
        if len(lab) <= width or " " not in lab:
            out.append(lab)
        else:
            out.append("\n".join(textwrap.wrap(lab, width=width, max_lines=2,
                                               placeholder="…")))
    return out


# ── tint patches: an area mark is a FILL, never a fat stroke ─────────────
TINT_PCT = 16               # the sanctioned area tint (spec §5, "capsules")
TINT_EDGE_PCT = 45          # the same hue, strong enough to draw the boundary


def tint_pct(color, pct=TINT_PCT, base="white"):
    """``pct`` % of ``color`` over ``base`` — the one tinting rule."""
    rgb = to_rgb(color)
    bg = to_rgb(base)
    f = float(pct) / 100.0
    return to_hex(tuple(bg[i] + f * (rgb[i] - bg[i]) for i in range(3)))


def strengthen(color, factor=2.6, base="white"):
    """A stronger tint of the same hue: scale the distance from ``base``.

    ``tint_pct(c, p)`` is ``base + p/100 * (c - base)``, so scaling what is
    left of ``base`` by ``factor`` returns the same hue at ``factor`` times the
    strength without needing to know which hue it was.  Used for the edge of a
    tint patch, which has to be the same colour, only readable.
    """
    rgb = to_rgb(color)
    bg = to_rgb(base)
    return to_hex(tuple(min(max(bg[i] + factor * (rgb[i] - bg[i]), 0.0), 1.0)
                        for i in range(3)))


def tint_patch(ax, shape, *, color, pct=TINT_PCT, edge=True,
               edge_pct=TINT_EDGE_PCT, face=None, edge_color=None,
               lw=None, radius_pt=3.0, zorder=0.4,
               clip_on=False, transform=None, **kw):
    """Draw an area mark as a 16 % tint FILL with a 0.55 pt edge.

    2026-09-08 spec §5: "capsules become 16 % tint patches".  Before this the
    schematics drew addressed subtrees, task capsules and route bands as
    round-capped strokes 2.6-13 pt wide; the audit exempted them as "area
    marks", so a 13 pt capsule passed while a 1.0 pt data line failed, and in
    print those capsules outweighed every data line on the page.  An area is
    now a filled patch and only a filled patch, which is also what makes the
    new stroke rule (nothing above 1.25 pt except a closed filled path)
    enforceable.

    ``shape`` is one of

    ``("rect", x0, y0, w, h)``   a rounded rectangle in data coordinates
                                 (``radius_pt`` corner radius, in points);
    ``("poly", [(x, y), ...])``  a closed polygon;
    ``("ribbon", [chain, ...], width_pt)`` one or more polylines buffered to
                                 ``width_pt`` with round caps and joins, i.e.
                                 exactly the shape the old fat stroke drew,
                                 emitted as a single filled path.

    ``color`` is a COLORS key or any matplotlib colour; ``pct`` the tint
    strength; ``edge`` draws the boundary at ``LW_HAIR`` in the same hue at
    ``edge_pct``.  Returns the patch (or ``None`` for an empty ribbon).
    """
    from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
    from matplotlib.path import Path as _Path

    col = COLORS.get(color, color)
    face = tint_pct(col, pct) if face is None else face
    if edge_color is not None:
        edgecolor = edge_color if edge else "none"
    else:
        edgecolor = tint_pct(col, edge_pct) if edge else "none"
    lw = (LW_HAIR if lw is None else float(lw)) if edge else 0.0
    transform = ax.transData if transform is None else transform
    kind = shape[0]
    if kind == "rect":
        _, x0, y0, w, h = shape
        # radius in points -> data units on the y axis, the usual convention
        # for these frames (they are drawn in 0-1 frame coordinates).
        bb = ax.get_window_extent()
        h_pt = max(bb.height * 72.0 / ax.figure.dpi, 1e-6)
        span = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        r = float(radius_pt) * span / h_pt
        patch = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=face, edgecolor=edgecolor, linewidth=lw,
            zorder=zorder, transform=transform, clip_on=clip_on, **kw)
    elif kind == "poly":
        patch = Polygon(list(shape[1]), closed=True, facecolor=face,
                        edgecolor=edgecolor, linewidth=lw, zorder=zorder,
                        transform=transform, clip_on=clip_on, **kw)
    elif kind == "ribbon":
        _, chains, width_pt = shape
        path = ribbon_path(ax, chains, width_pt)
        if path is None:
            return None
        patch = PathPatch(path, facecolor=face, edgecolor=edgecolor,
                          linewidth=lw, zorder=zorder, transform=transform,
                          clip_on=clip_on, joinstyle="round", **kw)
        if not isinstance(path, _Path):        # pragma: no cover - defensive
            return None
    else:
        raise ValueError(f"tint_patch: unknown shape {kind!r}")
    ax.add_patch(patch)
    return patch


def ribbon_path(ax, chains, width_pt):
    """Closed outline of ``chains`` stroked ``width_pt`` wide, as a Path.

    Uses shapely when it is installed (one clean union, so the tint has no
    internal seams) and falls back to a per-segment capsule union drawn with
    the non-zero winding rule, which fills identically but cannot carry a
    single outer edge.  Coordinates are the axes' data coordinates; the width
    is converted through the axes box so a ribbon is the same weight in a 4-
    and in a 7-module slot.
    """
    import numpy as np
    from matplotlib.path import Path as _Path

    chains = [np.asarray(c, dtype=float) for c in chains if len(c)]
    chains = [c for c in chains if len(c) >= 1]
    if not chains:
        return None
    bb = ax.get_window_extent()
    dpi = ax.figure.dpi
    w_pt = max(bb.width * 72.0 / dpi, 1e-6)
    h_pt = max(bb.height * 72.0 / dpi, 1e-6)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = abs(x1 - x0) / w_pt            # data units per point, x
    sy = abs(y1 - y0) / h_pt            # data units per point, y
    r = 0.5 * float(width_pt)
    try:
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union
    except Exception:                                    # pragma: no cover
        LineString = None
    if LineString is not None:
        geoms = []
        for chain in chains:
            pts = [(p[0] / sx, p[1] / sy) for p in chain]   # to points
            geom = (Point(pts[0]) if len(pts) == 1 else LineString(pts))
            try:
                geoms.append(geom.buffer(r, quad_segs=8, cap_style="round",
                                         join_style="round"))
            except TypeError:            # shapely < 2.1
                geoms.append(geom.buffer(r, resolution=8, cap_style=1,
                                         join_style=1))
        merged = unary_union(geoms)
        polys = getattr(merged, "geoms", [merged])
        verts, codes = [], []
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                coords = list(ring.coords)
                verts.extend([(x * sx, y * sy) for x, y in coords])
                codes.extend([_Path.MOVETO]
                             + [_Path.LINETO] * (len(coords) - 2)
                             + [_Path.CLOSEPOLY])
        return _Path(verts, codes)
    # fallback: one rounded capsule per segment, non-zero winding
    verts, codes = [], []
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    for chain in chains:
        pts = chain if len(chain) > 1 else np.vstack([chain, chain])
        for a, b in zip(pts[:-1], pts[1:]):
            for centre in (a, b):
                ring = [(centre[0] + r * sx * np.cos(t),
                         centre[1] + r * sy * np.sin(t)) for t in theta]
                verts.extend(ring)
                codes.extend([_Path.MOVETO] + [_Path.LINETO] * (len(ring) - 2)
                             + [_Path.CLOSEPOLY])
            dx, dy = (b[0] - a[0]) / sx, (b[1] - a[1]) / sy
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm * r, dx / norm * r
            quad = [(a[0] + nx * sx, a[1] + ny * sy),
                    (b[0] + nx * sx, b[1] + ny * sy),
                    (b[0] - nx * sx, b[1] - ny * sy),
                    (a[0] - nx * sx, a[1] - ny * sy)]
            verts.extend(quad)
            codes.extend([_Path.MOVETO, _Path.LINETO, _Path.LINETO,
                          _Path.CLOSEPOLY])
    return _Path(verts, codes)


def apply_neurips_style() -> None:
    """Frozen base style, then the journal overrides, then the letter hook."""
    _base.apply_neurips_style()
    _base._finalize_panel_letters = _journal_finalize_panel_letters
    mpl.rcParams.update(_font_rcparams())   # base sets DejaVu; take it back
    mpl.rcParams.update({
        "font.size": PT_TICK,
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "axes.titleweight": "normal",
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "legend.title_fontsize": PT_LEGEND,
        "lines.linewidth": LW_DATA,
        # Reset marker outlines/caps so prior builders cannot change them.
        "lines.markeredgewidth": LW_EDGE,
        "lines.markersize": MARKER_MS,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS["edge"],
        "axes.labelcolor": COLORS["ink"],
        "axes.labelpad": 2.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": COLORS["edge"],
        "ytick.color": COLORS["edge"],
        "xtick.labelcolor": COLORS["ink"],
        "ytick.labelcolor": COLORS["ink"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "grid.color": COLORS["grid"],
        "axes.prop_cycle": mpl.cycler(
            "color",
            [COLORS["shunting"], COLORS["additive"], COLORS["bp"],
             COLORS["local"], COLORS["oracle"], COLORS["highlight"],
             COLORS["point_mlp"]],
        ),
    })


def style_axis(ax, grid="none", spine_color=None):
    """Journal panel polish: thinner spines and ticks than the frozen base."""
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=0.6, alpha=0.55,
                color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(
            COLORS["edge"] if spine_color is None else spine_color
        )
