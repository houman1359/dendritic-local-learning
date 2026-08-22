#!/usr/bin/env python3
"""Build the standalone vector schematics used across main Figures 2--9.

The drawings themselves live in :mod:`native_schematics`, where each one is a
``draw_*(ax)`` function that fills a caller-supplied Axes.  That is what the
native full-width figure builders call, so a schematic is drawn once, at the
final scale, in the same type and stroke tokens as the data panels beside it.

This script keeps the legacy *component* assets alive: it renders each
schematic onto its own canvas in ``figures/generated`` for review and for the
compositor path that still consumes pre-rendered sub-blocks.  Nothing here
rescales a drawing; the standalone canvas is simply a different-sized Axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figure_canvas import enforce_tokens
from journal_style import FIG_W, apply_neurips_style
from native_schematics import (
    draw_alignment_boundary,
    draw_anatomy_pipeline,
    draw_credit_operator,
    draw_focal_shunt,
    draw_ownership_address,
    draw_physical_depth,
    draw_physical_generalization,
    draw_route_resolution,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# stem -> (drawing function, canvas height in inches)
COMPONENTS = {
    "schematic_fig2_ownership_address": (draw_ownership_address, 1.9),
    "schematic_fig3_route_resolution": (draw_route_resolution, 2.05),
    "schematic_fig4_credit_operator": (draw_credit_operator, 1.70),
    "schematic_fig5_physical_depth": (draw_physical_depth, 2.05),
    "schematic_fig6_generalization": (draw_physical_generalization, 1.8),
    "schematic_fig7_anatomy_pipeline": (draw_anatomy_pipeline, 1.75),
    "schematic_fig8_focal_shunt": (draw_focal_shunt, 2.05),
    "schematic_fig9_alignment_boundary": (draw_alignment_boundary, 1.8),
}


def _blank(width: float, height: float):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, stem: str) -> None:
    enforce_tokens(fig)
    fig.savefig(
        OUT / f"{stem}.pdf",
        format="pdf",
        facecolor="white",
        metadata={"Creator": "dendritic-local-learning", "CreationDate": None},
    )
    plt.close(fig)


def build(stem: str) -> Path:
    """Render one schematic component onto its own canvas."""
    draw, height = COMPONENTS[stem]
    fig, ax = _blank(FIG_W, height)
    draw(ax)
    _save(fig, stem)
    return OUT / f"{stem}.pdf"


def ownership_address() -> None:
    build("schematic_fig2_ownership_address")


def route_resolution() -> None:
    build("schematic_fig3_route_resolution")


def credit_operator() -> None:
    build("schematic_fig4_credit_operator")


def physical_depth() -> None:
    build("schematic_fig5_physical_depth")


def physical_generalization() -> None:
    build("schematic_fig6_generalization")


def anatomy_pipeline() -> None:
    build("schematic_fig7_anatomy_pipeline")


def focal_shunt() -> None:
    build("schematic_fig8_focal_shunt")


def alignment_boundary() -> None:
    build("schematic_fig9_alignment_boundary")


def main() -> None:
    apply_neurips_style()
    for stem in COMPONENTS:
        build(stem)
    print(f"Built {len(COMPONENTS)} main-figure schematic assets.")


if __name__ == "__main__":
    main()
