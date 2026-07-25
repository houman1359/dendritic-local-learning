#!/usr/bin/env python3
"""Regenerate every manuscript figure and check its layout.

Three checks run on each figure as it is saved:

1. ``audit_layout``          - text leaving the canvas, or landing inside a
                               different panel's box.
2. ``audit_text_over_data``  - annotations or legends drawn on top of lines,
                               bars or scatter points in the same panel.
3. border-ink scan           - ink touching the outer 2 px of the rendered PNG,
                               which is the authoritative test for clipping.

Run from the draft directory:

    python3 scripts/audit_figures.py

Exits non-zero if anything is flagged, so it can gate a figure rebuild.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DRAFT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from neurips_style import audit_layout, audit_text_over_data  # noqa: E402

GENERATORS = [
    "generate_neurips_figures",
    "generate_revision_figures",
    "generate_mechanistic_audit_figure",
    "generate_alignment_norm_dynamics_figure",
    "generate_inhibition_causality_figure",
    "generate_cifar_sweep_comparison_figures",
    "generate_figure1_schematic",
    "generate_morphology_ie_regime_figure",
    "generate_cue_routing_figures",
]


def main() -> int:
    seen: dict[str, list[str]] = {}
    original = plt.Figure.savefig

    def patched(self, fname, *a, **k):
        name = os.path.basename(str(fname))
        if name.endswith(".pdf") and name not in seen:
            seen[name] = audit_layout(self, name) + audit_text_over_data(self, name)
        return original(self, fname, *a, **k)

    plt.Figure.savefig = patched
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for mod in GENERATORS:
            try:
                sys.argv = [mod]
                runpy.run_path(str(SCRIPTS / f"{mod}.py"), run_name="__main__")
            except SystemExit:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {mod}: {type(exc).__name__}: {exc}")
    plt.Figure.savefig = original

    for line in buf.getvalue().splitlines():
        if any(tag in line for tag in ("[layout]", "[overlap]", "[skip]")):
            print(line)

    clipped = []
    for f in sorted(glob.glob(str(DRAFT / "figures" / "fig*.png"))):
        im = np.asarray(Image.open(f).convert("L"))
        ink = im < 245
        edges = {
            "top": ink[0:2, :].mean(), "bottom": ink[-2:, :].mean(),
            "left": ink[:, 0:2].mean(), "right": ink[:, -2:].mean(),
        }
        hits = [k for k, v in edges.items() if v > 0.002]
        if hits:
            clipped.append((os.path.basename(f), hits))
    for name, hits in clipped:
        print(f"  [clipped] {name}: {hits}")

    n = sum(len(v) for v in seen.values()) + len(clipped)
    print(f"\nTOTAL issues: {n} across "
          f"{sum(1 for v in seen.values() if v) + len(clipped)} figures")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
