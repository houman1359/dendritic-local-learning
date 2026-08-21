#!/usr/bin/env python3
"""Regenerate the vector PDF crops embedded by the workshop deck.

Self-contained asset builder for this folder: it holds only the 18 crops the
workshop deck references (the multi-deck package under
drafts/presentation/credit_assignment keeps its own superset list for the
legacy decks).  Crop boxes are specified in pixels of a 600-dpi render of the
journal source PDF and converted to source-PDF coordinates, so the outputs
retain vector text and line art.  An entry may carry an optional third
element (horizontal white padding in source pixels added on each side) and an
optional fourth element (a tuple of white cover boxes, absolute source
pixels, painted over the output to strip journal panel letters; covers sit in
the white axes margin only).

If a journal figure sheet is regenerated with a new layout, re-measure the
affected boxes against a fresh 600-dpi render before rebuilding.
"""

from __future__ import annotations

from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parent / "journal"
FIGURES = JOURNAL / "figures"
OUT = HERE / "pdf_assets"

CROPS: dict[str, tuple] = {
    # Shared paper/talk vector schematics.  These are full-page copies rather
    # than raster crops, so the workshop and journal use one graphical
    # vocabulary while retaining independently composed layouts.
    "ownership_schematic": (
        "components/schematic_fig2_ownership_address", (0, 0, 2160, 1140)
    ),
    "physical_stage_schematic": (
        "components/schematic_fig5_physical_depth", (0, 0, 1440, 1230)
    ),
    "focal_shunt_schematic": (
        "components/schematic_fig8_focal_shunt", (0, 0, 3120, 1590)
    ),
    # phase_plane: talk variant — legend column cropped off (the slide carries
    # the single legend) and the panel letter covered.  Retuned 2026-08-20
    # against the regenerated full-width band sheet (4320x1380 @600dpi): plot
    # ink ends at x=3092, legend starts at x=3160, letter "O" at (78,76)-(143,142).
    "phase_plane": (
        "main/figure_09_panel_O", (0, 0, 3120, 1380), 0,
        ((40, 20, 220, 200),),
    ),
    "capture_per_wire": (
        "main/figure_07_panels_K-L", (2200, 0, 4320, 1530), 0,
        ((2310, 0, 2510, 135),),
    ),
    "subtree_k_sweep": (
        "main/figure_03_panels_A-F", (1560, 0, 2990, 1590), 0,
        ((1560, 0, 1710, 185),),
    ),
    "physical_depth_headline": (
        "main/figure_05_panels_A-F", (3040, 0, 4320, 1430), 0,
        ((3070, 10, 3290, 165),),
    ),
    # panel F (electrotonic limit, the cable-calibration boundary panel)
    "focal_boundary": (
        "main/figure_08_panels_A-I", (2880, 1440, 4320, 2880), 0,
        ((3040, 1460, 3300, 1680),),
    ),
    "animal_signed": (
        "supplementary/figure_S17_panels_A-D", (2270, 0, 4320, 1572), 0,
        ((2280, 0, 2450, 125), (3340, 0, 3520, 125)),
    ),
    "identity_panel_b": (
        "main/figure_02_panels_A-F", (1470, 0, 2990, 1560), 0,
        ((1540, 0, 1720, 145),),
    ),
    "ownership_margin": (
        "main/figure_02_panels_G-O", (0, 1245, 1440, 2460), 0,
        ((95, 1255, 300, 1435),),
    ),
    # crop keeps the full panel-letter row (a glyph sliced by the crop edge
    # survives PDF inclusion whole) and covers the "K" inside the page.
    "clean_agreement": (
        "main/figure_02_panels_G-O", (1440, 1300, 2880, 2460), 0,
        ((1590, 1300, 1720, 1430),),
    ),
    "factorial_controls_legend": (
        "main/figure_03_panels_A-F", (0, 1670, 4320, 3450), 0,
        ((90, 1800, 270, 1945), (1540, 1800, 1720, 1945),
         (3010, 1800, 3190, 1945)),
    ),
    "depth_benefit_n": (
        "main/figure_05_panels_M-O", (990, 0, 1975, 1113), 0,
        ((1050, 0, 1240, 120),),
    ),
    # panel E (adjoint transport, the exact-transport-contrast successor)
    "transport_contrast_h": (
        "main/figure_08_panels_A-I", (1440, 1440, 2880, 2880), 0,
        ((1560, 1470, 1810, 1680),),
    ),
    "fulltree_no_signed": (
        "main/figure_09_panels_I-J", (0, 0, 4320, 1572)
    ),
    # panel N alone (capture–progress, rho_s = 0.99) — the single anchor for
    # the alignment-rescue slide.  The crop keeps the full letter row and
    # covers the letter inside the page.
    "alignment_rescue_n": (
        "main/figure_09_panels_K-N", (3330, 0, 4320, 1470), 0,
        ((3340, 0, 3640, 150),),
    ),
    "reliability_step_c": (
        "supplementary/figure_S13_panels_A-F", (3000, 0, 4320, 1420), 0,
        ((3100, 0, 3280, 215),),
    ),
    "reliability_final_f": (
        "supplementary/figure_S13_panels_A-F", (3000, 1560, 4320, 2970), 0,
        ((3100, 1565, 3280, 1690),),
    ),
    "same_span_crossover_e": (
        "supplementary/figure_S14_panels_A-F", (1560, 1595, 3000, 2800)
    ),
    "same_span_risk_f": (
        "supplementary/figure_S14_panels_A-F", (3000, 1595, 4320, 2800)
    ),
}


def crop_vector(
    name: str,
    source_stem: str,
    box: tuple[int, int, int, int],
    pad_x: int = 0,
    covers: tuple[tuple[int, int, int, int], ...] = (),
) -> Path:
    source_pdf = FIGURES / f"{source_stem}.pdf"
    if not source_pdf.is_file():
        raise FileNotFoundError(source_pdf)

    source = fitz.open(source_pdf)
    page = source[0]
    page_box = page.rect
    # 600-dpi pixel geometry of the source page, derived directly from the PDF
    # page rectangle (exact: sheets are 7.2-pt multiples = whole 600-dpi px).
    pixel_w = round(page_box.width * 600 / 72)
    pixel_h = round(page_box.height * 600 / 72)
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= pixel_w and 0 <= y0 < y1 <= pixel_h):
        source.close()
        raise ValueError(f"Invalid crop {box} for {source_pdf.name} ({pixel_w}x{pixel_h})")
    clip = fitz.Rect(
        page_box.x0 + page_box.width * x0 / pixel_w,
        page_box.y0 + page_box.height * y0 / pixel_h,
        page_box.x0 + page_box.width * x1 / pixel_w,
        page_box.y0 + page_box.height * y1 / pixel_h,
    )
    pad = page_box.width * pad_x / pixel_w
    output = fitz.open()
    target = output.new_page(width=clip.width + 2 * pad, height=clip.height)
    content = fitz.Rect(pad, 0, pad + clip.width, clip.height)
    target.show_pdf_page(content, source, 0, clip=clip, keep_proportion=False)
    scale_x = page_box.width / pixel_w
    scale_y = page_box.height / pixel_h
    for cover in covers:
        cx0, cy0, cx1, cy1 = cover
        if not (x0 <= cx0 < cx1 <= x1 and y0 <= cy0 < cy1 <= y1):
            raise ValueError(f"Cover {cover} outside crop {box} for {name}")
        rect = fitz.Rect(
            pad + (cx0 - x0) * scale_x,
            (cy0 - y0) * scale_y,
            pad + (cx1 - x0) * scale_x,
            (cy1 - y0) * scale_y,
        )
        target.draw_rect(rect, color=None, fill=(1, 1, 1))
    destination = OUT / f"{name}.pdf"
    output.save(destination, garbage=4, deflate=True)
    output.close()
    source.close()
    return destination


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    for name, entry in CROPS.items():
        source, box = entry[0], entry[1]
        pad_x = entry[2] if len(entry) > 2 else 0
        covers = entry[3] if len(entry) > 3 else ()
        try:
            destination = crop_vector(name, source, box, pad_x, covers)
        except FileNotFoundError as error:
            print(f"WARNING: skipped {name}: missing canonical source {error}")
            skipped.append(name)
            continue
        print(f"Wrote {destination}")
    if skipped:
        raise SystemExit(
            f"{len(skipped)} crop(s) skipped: {', '.join(skipped)} — "
            "re-measure against the current journal sheets before rebuilding"
        )


if __name__ == "__main__":
    main()
