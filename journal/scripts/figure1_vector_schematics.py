"""Editable vector schematics for the journal's conceptual Figure 1.

The drawings use one geometry, palette and stroke hierarchy across the paper
and talk.  Every element is a matplotlib vector artist; SVG exports retain
live text (``svg.fonttype = 'none'``) and the PDF export uses TrueType fonts.
No raster image or generated illustration is embedded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    PT_ANNOT,
    PT_SMALL,
    apply_neurips_style,
    panel_title,
)


INK = COLORS["ink"]
MUTE = COLORS["mute"]
GRID = COLORS["grid"]
PAPER = "#FFFFFF"
PALE_BLUE = "#E8F0F9"
PALE_GREEN = "#E8F4EC"
PALE_AMBER = "#FBF1DF"
PALE_ROSE = "#F8E8E5"
PALE_VIOLET = "#F0EBF8"


TREE_POINTS = {
    "S": (0.50, 0.02),
    "J1": (0.50, 0.27),
    "JL": (0.31, 0.47),
    "JR": (0.69, 0.45),
    "JLL": (0.17, 0.64),
    "JLR": (0.40, 0.73),
    "JRL": (0.60, 0.71),
    "JRR": (0.82, 0.61),
    "T1": (0.05, 0.78),
    "T2": (0.20, 0.91),
    "T3": (0.34, 0.97),
    "T4": (0.46, 0.98),
    "T5": (0.54, 0.97),
    "T6": (0.67, 0.94),
    "T7": (0.80, 0.87),
    "T8": (0.95, 0.73),
}

TREE_EDGES = [
    ("S", "J1", 1.00),
    ("J1", "JL", 0.82), ("J1", "JR", 0.82),
    ("JL", "JLL", 0.64), ("JL", "JLR", 0.64),
    ("JR", "JRL", 0.64), ("JR", "JRR", 0.64),
    ("JLL", "T1", 0.48), ("JLL", "T2", 0.48),
    ("JLR", "T3", 0.48), ("JLR", "T4", 0.48),
    ("JRL", "T5", 0.48), ("JRL", "T6", 0.48),
    ("JRR", "T7", 0.48), ("JRR", "T8", 0.48),
]

SUBTREE_FIELDS = [
    (("JLL", "T1"), ("JLL", "T2"), "#DCEFE3"),
    (("JLR", "T3"), ("JLR", "T4"), "#E2ECF8"),
    (("JRL", "T5"), ("JRL", "T6"), "#F8EACF"),
    (("JRR", "T7"), ("JRR", "T8"), "#EDE6F6"),
]


def _setup(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _xy(box, name):
    x, y, w, h = box
    px, py = TREE_POINTS[name]
    return x + px * w, y + py * h


def _mix(hex_color: str, amount: float, base: str = "#FFFFFF") -> str:
    def rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    a, b = rgb(hex_color), rgb(base)
    out = tuple(round(amount * x + (1.0 - amount) * y) for x, y in zip(a, b))
    return "#" + "".join(f"{v:02X}" for v in out)


def _arrow(ax, start, end, *, color=MUTE, lw=LW_EDGE, rad=0.0,
           head=5.2, zorder=8, linestyle="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=f"-|>,head_length={head},head_width={0.62 * head}",
        mutation_scale=1.0,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        lw=lw,
        linestyle=linestyle,
        capstyle="round",
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _card(ax, x, y, w, h, *, face=PAPER, edge=GRID, radius=0.025,
          lw=LW_HAIR, zorder=0.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        lw=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _tree(ax, box, *, muted=False, fields=False, synapses=False,
          route=(), selected=None, gain=False):
    """Draw a compact tree in axes coordinates and return mapped points."""
    base = _mix(MUTE, 0.34) if muted else COLORS["dend"]
    route_set = {tuple(edge) for edge in route}

    if fields:
        for edge_a, edge_b, color in SUBTREE_FIELDS:
            for edge in (edge_a, edge_b):
                p0, p1 = _xy(box, edge[0]), _xy(box, edge[1])
                ax.plot(
                    [p0[0], p1[0]], [p0[1], p1[1]],
                    color=color, lw=8.5, solid_capstyle="round",
                    transform=ax.transAxes, zorder=1,
                )

    for a, b, taper in TREE_EDGES:
        p0, p1 = _xy(box, a), _xy(box, b)
        active = (a, b) in route_set
        color = COLORS["additive"] if active else base
        lw = (LW_DATA if active else 1.18) * taper
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color, lw=lw, solid_capstyle="round",
            transform=ax.transAxes, zorder=3 if active else 2,
        )

    junctions = ("J1", "JL", "JR", "JLL", "JLR", "JRL", "JRR")
    for name in junctions:
        px, py = _xy(box, name)
        active = any(name in edge for edge in route_set)
        ax.add_patch(Circle(
            (px, py), min(box[2], box[3]) * 0.018,
            facecolor=PAPER,
            edgecolor=COLORS["additive"] if active else base,
            lw=LW_EDGE,
            transform=ax.transAxes,
            zorder=4,
        ))

    sx, sy = _xy(box, "S")
    ax.add_patch(Circle(
        (sx, sy), min(box[2], box[3]) * 0.055,
        facecolor=COLORS["soma"], edgecolor=COLORS["edge"],
        lw=LW_EDGE, transform=ax.transAxes, zorder=5,
    ))

    if synapses:
        for index, name in enumerate(("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")):
            tx, ty = _xy(box, name)
            ax.add_patch(Circle(
                (tx, ty), min(box[2], box[3]) * 0.014,
                facecolor=COLORS["exc"] if index not in (1, 6) else COLORS["inh"],
                edgecolor=PAPER, lw=0.35,
                transform=ax.transAxes, zorder=5,
            ))

    if selected is not None:
        tx, ty = _xy(box, selected)
        ax.add_patch(Circle(
            (tx, ty), min(box[2], box[3]) * 0.032,
            facecolor="none", edgecolor=INK, lw=LW_EDGE,
            transform=ax.transAxes, zorder=6,
        ))

    if gain:
        gx, gy = _xy(box, "JRL")
        ax.add_patch(Circle(
            (gx, gy), min(box[2], box[3]) * 0.040,
            facecolor="none", edgecolor=INK, lw=LW_EDGE,
            transform=ax.transAxes, zorder=6,
        ))

    return {name: _xy(box, name) for name in TREE_POINTS}


def draw_credit_assignment_gap(ax, *, letter="A") -> None:
    _setup(ax)
    panel_title(ax, letter, "One error, many updates")

    tree_box = (0.14, 0.13, 0.67, 0.68)
    points = _tree(ax, tree_box, synapses=True)

    # A clean bracket turns the arbor into the visual statement; no paragraph
    # is needed inside the panel.
    ax.plot([0.18, 0.77], [0.88, 0.88], color=GRID, lw=LW_EDGE,
            transform=ax.transAxes, zorder=1)
    ax.plot([0.18, 0.18], [0.85, 0.88], color=GRID, lw=LW_EDGE,
            transform=ax.transAxes, zorder=1)
    ax.plot([0.77, 0.77], [0.85, 0.88], color=GRID, lw=LW_EDGE,
            transform=ax.transAxes, zorder=1)
    ax.text(0.475, 0.90, "many synaptic parameters", ha="center", va="bottom",
            fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    # Task loss arrives once, at the neuronal output.
    loss_xy = (0.89, 0.20)
    ax.add_patch(Circle(loss_xy, 0.055, facecolor=PALE_ROSE,
                        edgecolor=COLORS["bp"], lw=LW_EDGE,
                        transform=ax.transAxes, zorder=5))
    ax.text(*loss_xy, r"$\mathcal{L}$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["bp"], transform=ax.transAxes,
            zorder=6)
    _arrow(ax, (0.835, 0.20), (points["S"][0] + 0.025, points["S"][1]),
           color=COLORS["additive"], lw=LW_DATA, rad=-0.10)
    ax.text(0.88, 0.105, "one task error", ha="center", va="top",
            fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    # Three possible local decisions establish the problem without claiming a
    # particular rule.
    for name, symbol, color, dx in (
        ("T2", "+", COLORS["shunting"], -0.025),
        ("T5", "−", COLORS["bp"], 0.015),
        ("T8", "?", COLORS["oracle"], 0.020),
    ):
        tx, ty = points[name]
        ax.text(tx + dx, ty + 0.052, symbol, ha="center", va="center",
                fontsize=PT_ANNOT, color=color, transform=ax.transAxes,
                zorder=7)


def draw_point_vs_dendrite(ax, *, letter="B") -> None:
    _setup(ax)
    panel_title(ax, letter, "Point neuron → dendritic tree")

    # Point-neuron side: every synapse shares the same destination coordinate.
    soma = (0.23, 0.46)
    input_y = (0.25, 0.36, 0.48, 0.60, 0.71)
    for y in input_y:
        ax.plot([0.07, soma[0] - 0.052], [y, soma[1]], color=MUTE,
                lw=LW_EDGE, solid_capstyle="round", transform=ax.transAxes)
        ax.add_patch(Circle((0.07, y), 0.012, facecolor=COLORS["exc"],
                            edgecolor="none", transform=ax.transAxes, zorder=4))
    ax.add_patch(Circle(soma, 0.064, facecolor=COLORS["soma"],
                        edgecolor=COLORS["edge"], lw=LW_EDGE,
                        transform=ax.transAxes, zorder=5))
    _arrow(ax, (0.23, 0.16), (0.23, 0.375), color=COLORS["additive"],
           lw=LW_DATA)
    ax.text(0.23, 0.12, r"$\delta_u$", ha="center", va="top",
            fontsize=PT_ANNOT, color=COLORS["additive"], transform=ax.transAxes)
    ax.text(0.23, 0.82, "point", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK, transform=ax.transAxes)
    ax.text(0.23, 0.04, r"shared $\delta_u$", ha="center", va="bottom",
            fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    _arrow(ax, (0.40, 0.48), (0.51, 0.48), color=MUTE, lw=LW_EDGE)
    ax.text(0.455, 0.54, "structure", ha="center", va="bottom",
            fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    tree_box = (0.56, 0.16, 0.38, 0.63)
    points = _tree(ax, tree_box, fields=True)
    ax.text(0.75, 0.82, "dendritic", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK, transform=ax.transAxes)
    for name, label, color, offset in (
        ("JLL", r"$\delta_{u,1}$", COLORS["shunting"], (-0.02, 0.02)),
        ("JLR", r"$\delta_{u,2}$", COLORS["additive"], (-0.02, 0.02)),
        ("JRL", r"$\delta_{u,3}$", COLORS["local"], (0.02, 0.02)),
        ("JRR", r"$\delta_{u,4}$", COLORS["oracle"], (0.02, 0.02)),
    ):
        x, y = points[name]
        ax.text(x + offset[0], y + offset[1], label,
                ha="right" if offset[0] < 0 else "left", va="bottom",
                fontsize=PT_SMALL, color=color, transform=ax.transAxes)
    ax.text(0.75, 0.04, r"subtree $\delta_{u,k}$", ha="center", va="bottom",
            fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)


def draw_information_ladder(ax, *, letter="C") -> None:
    _setup(ax)
    panel_title(ax, letter, "Coordinate → address → gain")

    rows = [
        (0.68, "coordinate", "which neuron", r"$\delta_u$", "coordinate", PALE_BLUE),
        (0.38, "address", "which subtree", r"$\delta_{u,k}$", "address", PALE_GREEN),
        (0.08, "gain", "how strongly", r"$\widetilde{\alpha}_n$", "gain", PALE_AMBER),
    ]
    for y, name, question, symbol, mode, face in rows:
        _card(ax, 0.015, y, 0.97, 0.235, face=face, edge="none", radius=0.025)
        box = (0.045, y + 0.025, 0.29, 0.18)
        if mode == "coordinate":
            points = _tree(ax, box, muted=True)
            sx, sy = points["S"]
            ax.add_patch(Circle((sx, sy), 0.028, facecolor=PAPER,
                                edgecolor=COLORS["additive"], lw=LW_DATA,
                                transform=ax.transAxes, zorder=7))
            _arrow(ax, (sx + 0.09, sy - 0.01), (sx + 0.035, sy),
                   color=COLORS["additive"], lw=LW_EDGE, rad=0.12, head=4.2)
        elif mode == "address":
            _tree(ax, box, fields=True)
        else:
            _tree(ax, box, fields=True, gain=True)
        ax.text(0.39, y + 0.151, name, ha="left", va="center",
                fontsize=PT_ANNOT, color=INK, transform=ax.transAxes)
        ax.text(0.39, y + 0.077, question, ha="left", va="center",
                fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)
        ax.text(0.95, y + 0.151, symbol, ha="right", va="center",
                fontsize=PT_SMALL, color=INK if mode != "coordinate" else COLORS["additive"],
                transform=ax.transAxes)


def draw_eligibility_transport(ax, *, letter="D") -> None:
    _setup(ax)
    panel_title(ax, letter, "Eligibility × transported error")

    route = (("S", "J1"), ("J1", "JR"), ("JR", "JRL"), ("JRL", "T6"))
    tree_box = (0.035, 0.17, 0.50, 0.67)
    points = _tree(ax, tree_box, muted=True, route=route, selected="T6")

    # Arrowheads show the adjoint direction from soma toward the synapse.
    for a, b in route[:3]:
        p0, p1 = points[a], points[b]
        start = (p0[0] + 0.34 * (p1[0] - p0[0]), p0[1] + 0.34 * (p1[1] - p0[1]))
        end = (p0[0] + 0.63 * (p1[0] - p0[0]), p0[1] + 0.63 * (p1[1] - p0[1]))
        _arrow(ax, start, end, color=COLORS["additive"], lw=LW_EDGE, head=4.2)
    ax.text(points["S"][0] - 0.055, points["S"][1] - 0.005, r"$\delta_u$",
            ha="right", va="center", fontsize=PT_ANNOT,
            color=COLORS["additive"], transform=ax.transAxes)
    for name, label, dx in (("J1", r"$\alpha_1$", 0.035),
                            ("JR", r"$\alpha_2$", 0.035),
                            ("JRL", r"$\alpha_3$", -0.040)):
        x, y = points[name]
        ax.text(x + dx, y, label, ha="left" if dx > 0 else "right",
                va="center", fontsize=PT_SMALL, color=COLORS["additive"],
                transform=ax.transAxes)
    ax.text(0.27, 0.88, "transport through the ancestor path", ha="center",
            va="center", fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    _card(ax, 0.63, 0.25, 0.33, 0.54, face="#FAFBFC", edge=GRID, radius=0.028)
    ax.text(0.795, 0.72, "local eligibility", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK, transform=ax.transAxes)
    eligibility_rows = [
        (0.61, COLORS["exc"], r"$x_i$", "presynaptic activity"),
        (0.49, COLORS["shunting"], r"$E_i-V_n$", "driving force"),
        (0.37, COLORS["oracle"], r"$R_n^{\mathrm{tot}}$", "local resistance"),
    ]
    for y, color, symbol, description in eligibility_rows:
        ax.add_patch(Circle((0.69, y), 0.014, facecolor=color, edgecolor="none",
                            transform=ax.transAxes, zorder=5))
        ax.text(0.73, y + 0.017, symbol, ha="left", va="center",
                fontsize=PT_SMALL, color=INK, transform=ax.transAxes)
        ax.text(0.73, y - 0.025, description, ha="left", va="center",
                fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)
    _arrow(ax, (points["T6"][0] + 0.015, points["T6"][1] - 0.015),
           (0.625, 0.59), color=GRID, lw=LW_EDGE, head=4.0)

    ax.text(0.50, 0.075,
            r"$\dfrac{\partial\mathcal{L}}{\partial g_i}"
            r"\;=\;e_i\;\times\;\delta_u\widetilde{\alpha}_n$",
            ha="center", va="center", fontsize=PT_ANNOT, color=INK,
            transform=ax.transAxes)


def _mini_phase_icon(ax, x, y, w, h):
    ax.plot([x, x], [y, y + h], color=MUTE, lw=LW_HAIR, transform=ax.transAxes)
    ax.plot([x, x + w], [y, y], color=MUTE, lw=LW_HAIR, transform=ax.transAxes)
    xs = [x + 0.05 * w, x + 0.36 * w, x + 0.67 * w, x + 0.95 * w]
    ys = [y + 0.18 * h, y + 0.70 * h, y + 0.84 * h, y + 0.52 * h]
    ax.plot(xs, ys, color=COLORS["oracle"], lw=LW_DATA,
            solid_capstyle="round", transform=ax.transAxes, zorder=4)
    ax.plot([xs[1], xs[1]], [y, y + h], color=GRID, lw=LW_HAIR,
            linestyle=(0, (2, 2)), transform=ax.transAxes)


def draw_evidence_path(ax, *, letter="E") -> None:
    _setup(ax)
    panel_title(ax, letter, "Theory → biological boundary")

    cards = [
        (0.03, 0.56, PALE_ROSE, "1", "factorize", "exact gradient"),
        (0.54, 0.56, PALE_VIOLET, "2", "predict", "signal–noise regimes"),
        (0.54, 0.13, PALE_AMBER, "3", "test", "trained route controls"),
        (0.03, 0.13, PALE_GREEN, "4", "bound", "anatomy + physiology"),
    ]
    for x, y, face, number, verb, object_ in cards:
        _card(ax, x, y, 0.42, 0.29, face=face, edge="none", radius=0.030)
        ax.add_patch(Circle((x + 0.055, y + 0.235), 0.025, facecolor=PAPER,
                            edgecolor=GRID, lw=LW_HAIR,
                            transform=ax.transAxes, zorder=4))
        ax.text(x + 0.055, y + 0.235, number, ha="center", va="center",
                fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.10, y + 0.235, verb, ha="left", va="center",
                fontsize=PT_ANNOT, color=INK, transform=ax.transAxes)
        ax.text(x + 0.055, y + 0.018, object_, ha="left", va="bottom",
                fontsize=PT_SMALL, color=MUTE, transform=ax.transAxes)

    ax.text(0.24, 0.685, r"$e_i\,q_n$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["bp"], transform=ax.transAxes)
    _mini_phase_icon(ax, 0.64, 0.665, 0.23, 0.090)
    _tree(ax, (0.65, 0.215, 0.20, 0.125), fields=True, gain=True)
    _tree(ax, (0.14, 0.215, 0.20, 0.125), fields=False, synapses=True)

    _arrow(ax, (0.46, 0.705), (0.525, 0.705), color=MUTE, lw=LW_EDGE)
    _arrow(ax, (0.75, 0.535), (0.75, 0.435), color=MUTE, lw=LW_EDGE)
    _arrow(ax, (0.525, 0.275), (0.46, 0.275), color=MUTE, lw=LW_EDGE)


SCHEMATICS = {
    "credit_assignment_gap": draw_credit_assignment_gap,
    "point_vs_dendritic": draw_point_vs_dendrite,
    "credit_information_ladder": draw_information_ladder,
    "eligibility_transport": draw_eligibility_transport,
    "evidence_boundary_path": draw_evidence_path,
}


def export_schematic_set(output_dir: Path) -> None:
    """Export reusable, Illustrator-editable SVG versions of all panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, draw in SCHEMATICS.items():
        fig, ax = plt.subplots(figsize=(FIG_W, 4.90))
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.06, top=0.86)
        draw(ax, letter="")
        fig.savefig(
            output_dir / f"{stem}.svg",
            metadata={"Date": None, "Creator": "Dendritic local learning vector schematic system"},
        )
        plt.close(fig)


if __name__ == "__main__":
    apply_neurips_style()
    mpl.rcParams["svg.fonttype"] = "none"
    root = Path(__file__).resolve().parents[1]
    export_schematic_set(root / "figures" / "schematics")
