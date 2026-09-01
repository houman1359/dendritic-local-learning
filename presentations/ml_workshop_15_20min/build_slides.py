#!/usr/bin/env python3
"""Build the 15--20 minute ML-audience workshop deck.

The editable master is HTML with native SVG schematics. Frozen numerical
panels are rendered at high resolution from the canonical vector PDF assets.
The public deliverables are 2560 x 1440 PNG files, a combined PDF, and a
contact sheet for visual review.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
PRESENTATIONS = HERE.parent
SOURCE_ASSETS = PRESENTATIONS / "pdf_assets"
ASSETS = HERE / "assets"
PNG_DIR = HERE / "png"
MASTER = HERE / "workshop_deck.html"
PDF_OUT = HERE / "dendritic_credit_ml_workshop.pdf"
CONTACT = HERE / "contact_sheet.png"

CSS_WIDTH = 1920
CSS_HEIGHT = 1080
PNG_WIDTH = 2560
PNG_HEIGHT = 1440
# Chromium's headless window includes an 87-CSS-pixel frame in this build.
# Give it extra height, then crop the authored 1920 x 1080 canvas exactly.
CHROMIUM_FRAME_HEIGHT = 87

PALETTE = {
    "ink": "#12233F",
    "muted": "#617187",
    "paper": "#F7F7F4",
    "white": "#FFFFFF",
    "grid": "#DDE5EA",
    "teal": "#168F83",
    "teal_pale": "#E6F3F0",
    "blue": "#2D67B1",
    "blue_pale": "#EAF1FA",
    "purple": "#7654B5",
    "purple_pale": "#F0ECF8",
    "orange": "#E48743",
    "orange_pale": "#FBEDE2",
    "red": "#BF4E5A",
    "red_pale": "#F9E9EB",
    "green": "#3D9667",
    "gray": "#8B98A7",
}


def trim_white(image: Image.Image, pad: int = 24) -> Image.Image:
    """Trim a white PDF margin without touching plot annotations."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    difference = difference.point(lambda p: 255 if p > 9 else 0)
    bbox = difference.getbbox()
    if bbox is None:
        return rgb
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(rgb.width, bbox[2] + pad)
    bottom = min(rgb.height, bbox[3] + pad)
    return rgb.crop((left, top, right, bottom))


def render_pdf_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    names = (
        "identity_panel_b",
        "cifar_confirmatory_ladder",
        "phase_plane",
        "transport_contrast_h",
        "path_necessity_task",
        "path_necessity_results",
        "subtree_k_sweep",
        "physical_stage_schematic",
        "physical_depth_headline",
        "capture_per_wire",
        "focal_shunt_schematic",
        "fulltree_no_signed",
        "alignment_rescue_n",
        "animal_signed",
    )
    for name in names:
        source = SOURCE_ASSETS / f"{name}.pdf"
        if not source.exists():
            raise FileNotFoundError(source)
        with fitz.open(source) as document:
            page = document[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0), alpha=False)
            temporary = ASSETS / f".{name}.raw.png"
            pixmap.save(temporary)
        with Image.open(temporary) as image:
            rendered = trim_white(image)
            if name == "cifar_confirmatory_ladder":
                # Imported panel letters are useful in the manuscript but
                # distracting when only one panel is shown in the talk.
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((0, 0, 128, 88), fill="white")
                font = ImageFont.truetype(
                    "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", 38
                )
                draw.text((50, 43), "60%", fill="#222222", font=font)
            if name == "phase_plane":
                # The slide uses the implementation-neutral name because the
                # plane includes both anatomical and non-anatomical routes.
                # Replace only the presentation raster's x-axis wording; the
                # numerical panel and its geometry remain unchanged.
                draw = ImageDraw.Draw(rendered)
                # Cover the complete manuscript-axis label before adding the
                # presentation wording.  The wider box also removes the final
                # letters of "task-anatomy alignment" at high raster scale.
                draw.rectangle((500, 548, 1400, 615), fill="white")
                font = ImageFont.truetype(
                    "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", 39
                )
                label = "task–route alignment"
                bbox = draw.textbbox((0, 0), label, font=font)
                draw.text(
                    ((rendered.width - (bbox[2] - bbox[0])) / 2, 552),
                    label,
                    fill="#222222",
                    font=font,
                )
            rendered.save(ASSETS / f"{name}.png", optimize=True)
        temporary.unlink()

    # The older talk crop says "scalar fallback".  The current primary MNIST
    # cohort is the strict-scalar control, so take panel B directly from the
    # canonical final Figure 2 instead of propagating the stale label.
    component_dir = PRESENTATIONS.parent / "journal" / "figures" / "components"

    def render_clip(
        source_name: str,
        output_name: str,
        fractions: tuple[float, float, float, float],
        *,
        post_crop_left: float = 0.0,
    ) -> None:
        source = component_dir / source_name
        with fitz.open(source) as document:
            page = document[0]
            rect = page.rect
            x0, y0, x1, y1 = fractions
            clip = fitz.Rect(rect.width * x0, rect.height * y0, rect.width * x1, rect.height * y1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(5.0, 5.0), clip=clip, alpha=False)
            temporary = ASSETS / f".{output_name}.raw.png"
            pixmap.save(temporary)
        with Image.open(temporary) as image:
            rendered = trim_white(image, pad=18)
            if post_crop_left:
                rendered = rendered.crop(
                    (int(rendered.width * post_crop_left), 0, rendered.width, rendered.height)
                )
            if output_name == "boundary_topology":
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((0, 0, 105, 90), fill="white")
            if output_name == "mnist_strict_scalar":
                draw = ImageDraw.Draw(rendered)
                # Remove the neighboring panel-C letter without touching the
                # exact-path data at the right edge of panel B.
                draw.rectangle((rendered.width - 125, 0, rendered.width, 95), fill="white")
            if output_name == "operator_utility_validation":
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((0, 0, 105, 90), fill="white")
            if output_name == "focal_boundary_current":
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((0, 0, 105, 82), fill="white")
                draw.rectangle((rendered.width - 105, 0, rendered.width, 82), fill="white")
            if output_name == "alignment_rotation":
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((90, 0, 225, 90), fill="white")
                draw.rectangle((0, rendered.height - 70, 110, rendered.height), fill="white")
            if output_name == "alignment_gain":
                # Keep the presentation symbol consistent with the equation
                # on slide 19 without changing the canonical paper panel.
                draw = ImageDraw.Draw(rendered)
                # Remove clipped E/F panel-letter fragments introduced by the
                # wider vertical crop used to retain the top data point.
                draw.rectangle((0, 0, 105, 55), fill="white")
                draw.rectangle((rendered.width - 105, 0, rendered.width, 55), fill="white")
                label_top = rendered.height - 72
                draw.rectangle((0, label_top, rendered.width, rendered.height), fill="white")
                font = ImageFont.truetype(
                    "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", 35
                )
                sub_font = ImageFont.truetype(
                    "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", 23
                )
                label = "imposed task–route alignment  a"
                sub_label = ""
                bbox = draw.textbbox((0, 0), label, font=font)
                sub_bbox = draw.textbbox((0, 0), sub_label, font=sub_font)
                total_width = (bbox[2] - bbox[0]) + (sub_bbox[2] - sub_bbox[0])
                x = (rendered.width - total_width) / 2
                y = rendered.height - 55
                draw.text(
                    (x, y),
                    label,
                    fill="#222222",
                    font=font,
                )
                if sub_label:
                    draw.text(
                        (x + (bbox[2] - bbox[0]), y + 20),
                        sub_label,
                        fill="#222222",
                        font=sub_font,
                    )
            rendered.save(ASSETS / f"{output_name}.png", optimize=True)
        temporary.unlink()

    render_clip(
        "main_figure_02_native.pdf",
        "mnist_strict_scalar",
        (0.035, 0.255, 0.278, 0.58),
    )
    render_clip(
        "main_figure_07_native.pdf",
        "capture_per_wire_current",
        (0.525, 0.345, 0.995, 0.995),
    )
    render_clip(
        "main_figure_08_native.pdf",
        "focal_boundary_current",
        (0.0, 0.67, 0.505, 0.995),
    )
    render_clip(
        "main_figure_09_native.pdf",
        "boundary_learning",
        (0.565, 0.0, 0.995, 0.285),
    )
    render_clip(
        "main_figure_09_native.pdf",
        "boundary_topology",
        (0.0, 0.285, 0.605, 0.575),
    )
    render_clip(
        "main_figure_03_native.pdf",
        "operator_utility_validation",
        (0.015, 0.675, 0.49, 0.995),
    )
    render_clip(
        "main_figure_09_native.pdf",
        "alignment_rotation",
        (0.575, 0.285, 0.995, 0.585),
    )
    render_clip(
        "main_figure_09_native.pdf",
        "alignment_gain",
        (0.005, 0.605, 0.37, 0.995),
    )
    render_clip(
        "main_figure_05_native.pdf",
        "hierarchy_task_current",
        (0.0, 0.0, 0.62, 0.34),
    )
    render_clip(
        "main_figure_05_native.pdf",
        "hierarchy_bandwidth_current",
        (0.61, 0.0, 0.995, 0.34),
    )
    render_clip(
        "main_figure_05_native.pdf",
        "hierarchy_learning_current",
        (0.0, 0.64, 0.325, 0.995),
    )
    render_clip(
        "main_mapped_reconstruction.pdf",
        "mapped_reconstruction",
        (0.0, 0.0, 1.0, 1.0),
    )
    render_clip(
        "main_ancestry_addresses.pdf",
        "ancestry_addresses",
        (0.0, 0.0, 1.0, 1.0),
    )


def dendrite_svg(*, compact: bool = False, labels: bool = True) -> str:
    width = 760 if compact else 900
    height = 520 if compact else 650
    label_block = ""
    if labels:
        label_block = """
        <g font-family="Arial, Nimbus Sans, sans-serif" font-size="22" font-weight="700">
          <rect x="42" y="78" width="210" height="54" rx="18" fill="#EAF1FA"/>
          <text x="147" y="112" text-anchor="middle" fill="#2D67B1">neuron identity</text>
          <path d="M250 106 C315 125 350 178 405 272" fill="none" stroke="#2D67B1" stroke-width="3" stroke-dasharray="7 7"/>
          <rect x="626" y="52" width="214" height="54" rx="18" fill="#F0ECF8"/>
          <text x="733" y="86" text-anchor="middle" fill="#7654B5">subtree address</text>
          <path d="M650 108 C612 142 596 186 588 226" fill="none" stroke="#7654B5" stroke-width="3" stroke-dasharray="7 7"/>
          <rect x="630" y="442" width="188" height="54" rx="18" fill="#E6F3F0"/>
          <text x="724" y="476" text-anchor="middle" fill="#168F83">route gain</text>
          <path d="M630 456 C584 422 560 389 544 350" fill="none" stroke="#168F83" stroke-width="3" stroke-dasharray="7 7"/>
        </g>"""
    return f"""
    <svg class="dendrite" viewBox="0 0 900 650" role="img" aria-label="Dendritic tree">
      <defs>
        <filter id="somaShadow" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="9" stdDeviation="10" flood-color="#12233F" flood-opacity=".16"/>
        </filter>
        <linearGradient id="somaFill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#F4AD6F"/><stop offset="1" stop-color="#DF7135"/>
        </linearGradient>
      </defs>
      <g fill="none" stroke-linecap="round" stroke-linejoin="round">
        <path d="M450 540 C448 482 451 430 448 367 C445 330 424 302 390 274" stroke="#168F83" stroke-width="13"/>
        <path d="M448 370 C500 336 540 299 575 245" stroke="#7654B5" stroke-width="11"/>
        <path d="M390 275 C333 249 281 222 228 176" stroke="#2D67B1" stroke-width="10"/>
        <path d="M390 275 C355 219 342 170 315 111" stroke="#2D67B1" stroke-width="8"/>
        <path d="M575 245 C625 208 676 165 731 103" stroke="#7654B5" stroke-width="9"/>
        <path d="M575 245 C623 267 674 279 738 286" stroke="#7654B5" stroke-width="8"/>
        <path d="M448 367 C505 392 555 417 607 461" stroke="#168F83" stroke-width="9"/>
        <path d="M448 367 C388 390 338 420 286 468" stroke="#168F83" stroke-width="9"/>
        <path d="M228 176 C180 150 138 122 95 82 M228 176 C183 193 143 221 102 256" stroke="#2D67B1" stroke-width="6"/>
        <path d="M315 111 C282 78 259 52 240 25 M315 111 C333 72 347 44 359 17" stroke="#2D67B1" stroke-width="5"/>
        <path d="M731 103 C770 76 802 50 832 20 M731 103 C760 125 800 141 850 150" stroke="#7654B5" stroke-width="5"/>
        <path d="M738 286 C782 266 820 245 862 213 M738 286 C783 305 818 329 851 362" stroke="#7654B5" stroke-width="5"/>
        <path d="M607 461 C648 489 685 518 721 562 M607 461 C646 444 686 431 732 425" stroke="#168F83" stroke-width="5"/>
        <path d="M286 468 C246 492 205 520 171 559 M286 468 C245 450 204 436 158 427" stroke="#168F83" stroke-width="5"/>
        <path d="M102 256 C65 271 39 292 18 319 M95 82 C65 64 39 48 18 27" stroke="#2D67B1" stroke-width="3.5"/>
      </g>
      <g fill="#FFFFFF" stroke-width="3">
        <circle cx="390" cy="275" r="8" stroke="#2D67B1"/><circle cx="575" cy="245" r="8" stroke="#7654B5"/>
        <circle cx="448" cy="367" r="8" stroke="#168F83"/><circle cx="228" cy="176" r="7" stroke="#2D67B1"/>
        <circle cx="731" cy="103" r="7" stroke="#7654B5"/><circle cx="607" cy="461" r="7" stroke="#168F83"/>
      </g>
      <circle cx="450" cy="551" r="46" fill="#EF8B43" stroke="#FFFFFF" stroke-width="7"/>
      <path d="M450 598 L450 638" stroke="#BF4E5A" stroke-width="9" stroke-linecap="round"/>
      {label_block}
    </svg>"""


def credit_assignment_svg() -> str:
    """Forward computation and backward credit with one highlighted weight."""
    layers = [(120, 4, "input"), (410, 5, "hidden"), (700, 4, "hidden"), (990, 2, "output")]
    coords: list[list[tuple[int, int]]] = []
    edges: list[str] = []
    nodes: list[str] = []
    for x, count, label in layers:
        ys = [135 + j * (330 // max(1, count - 1)) for j in range(count)]
        coords.append([(x, y) for y in ys])
        nodes.extend(
            f'<circle cx="{x}" cy="{y}" r="18" fill="#FFFFFF" stroke="#2D67B1" stroke-width="4"/>'
            for y in ys
        )
        nodes.append(f'<text x="{x}" y="525" text-anchor="middle" class="layer">{label}</text>')
    for left, right in zip(coords[:-1], coords[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                edges.append(f'<path d="M{x1+20} {y1} L{x2-20} {y2}" stroke="#D4DCE4" stroke-width="2"/>')

    # One weight is the local recipient of a gradient assembled through many
    # downstream paths.  Keep this edge and its reverse path visually dominant.
    edges.append('<path d="M138 355 L392 300" stroke="#168F83" stroke-width="9"/>')
    return f"""
    <svg viewBox="0 0 1540 570" class="wide-svg" role="img" aria-label="Network computation and backward credit assignment">
      <defs>
        <marker id="forwardArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0 0 L12 6 L0 12 z" fill="#2D67B1"/></marker>
        <marker id="creditArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0 0 L12 6 L0 12 z" fill="#BF4E5A"/></marker>
      </defs>
      <style>.layer{{font:22px Arial,sans-serif;fill:#617187}}.label{{font:700 23px Arial,sans-serif;fill:#12233F}}.small{{font:20px Arial,sans-serif;fill:#617187}}</style>
      <g>{''.join(edges)}</g><g>{''.join(nodes)}</g>
      <rect x="1110" y="220" width="160" height="92" rx="25" fill="#FBEDE2" stroke="#E48743" stroke-width="4"/>
      <text x="1190" y="275" text-anchor="middle" font-family="Georgia,serif" font-size="34" font-weight="700" fill="#12233F">loss ℒ</text>
      <path d="M1008 255 H1090" stroke="#2D67B1" stroke-width="7" marker-end="url(#forwardArrow)"/>
      <text x="690" y="45" text-anchor="middle" class="label" fill="#2D67B1">forward computation: weights → prediction → loss</text>
      <path d="M1170 340 C970 480 590 550 276 382" fill="none" stroke="#BF4E5A" stroke-width="7" stroke-dasharray="13 10" marker-end="url(#creditArrow)"/>
      <circle cx="267" cy="365" r="18" fill="#168F83" stroke="#FFFFFF" stroke-width="5"/>
      <rect x="40" y="330" width="190" height="74" rx="20" fill="#E6F3F0" stroke="#168F83" stroke-width="3"/>
      <text x="135" y="362" text-anchor="middle" class="label" fill="#168F83">one weight wᵢ</text>
      <text x="135" y="390" text-anchor="middle" class="small">one local update</text>
      <path d="M230 367 H246" stroke="#168F83" stroke-width="4"/>
      <text x="770" y="548" text-anchor="middle" class="label" fill="#BF4E5A">backward credit: how much did changing wᵢ affect ℒ?</text>
      <g transform="translate(1295,90)">
        <rect width="220" height="325" rx="26" fill="#FFFFFF" stroke="#DDE5EA" stroke-width="3"/>
        <text x="110" y="52" text-anchor="middle" class="label">credit must specify</text>
        <circle cx="38" cy="108" r="15" fill="#2D67B1"/><text x="69" y="116" class="small">which neuron?</text>
        <circle cx="38" cy="178" r="15" fill="#7654B5"/><text x="69" y="186" class="small">which location?</text>
        <circle cx="38" cy="248" r="15" fill="#BF4E5A"/><text x="69" y="256" class="small">sign and magnitude?</text>
      </g>
    </svg>"""


def network_svg() -> str:
    layers = [(140, 5, "input"), (460, 6, "hidden 1"), (800, 6, "hidden 2"), (1140, 3, "output")]
    circles = []
    edges = []
    coords: list[list[tuple[int, int]]] = []
    for x, count, label in layers:
        ys = [155 + j * (490 // max(1, count - 1)) for j in range(count)]
        coords.append([(x, y) for y in ys])
        circles.extend(f'<circle cx="{x}" cy="{y}" r="20" fill="#FFFFFF" stroke="#2D67B1" stroke-width="4"/>' for y in ys)
        circles.append(f'<text x="{x}" y="705" text-anchor="middle" class="layer-label">{label}</text>')
    for left, right in zip(coords[:-1], coords[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                edges.append(f'<path d="M{x1+22} {y1} L{x2-22} {y2}" stroke="#CCD6DF" stroke-width="2"/>')
    return f"""
    <svg viewBox="0 0 1360 780" class="network-svg">
      <defs><marker id="back" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#BF4E5A"/></marker></defs>
      <style>.layer-label{{font:24px Arial,sans-serif;fill:#617187}}</style>
      <g>{''.join(edges)}</g><g>{''.join(circles)}</g>
      <rect x="1242" y="322" width="104" height="86" rx="24" fill="#FBEDE2" stroke="#E48743" stroke-width="4"/>
      <text x="1294" y="375" text-anchor="middle" font-family="Georgia,serif" font-size="29" font-weight="700" fill="#12233F">ℒ</text>
      <path d="M1270 80 C1040 18 344 18 125 80" fill="none" stroke="#BF4E5A" stroke-width="7" marker-end="url(#back)"/>
      <text x="700" y="62" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#BF4E5A">exact reverse chain rule</text>
    </svg>"""


def bandwidth_svg() -> str:
    return """
    <svg viewBox="0 0 1540 560" class="wide-svg">
      <defs><marker id="arrB" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0 0 L11 5.5 L0 11 z" fill="#64748B"/></marker></defs>
      <g font-family="Arial,sans-serif">
        <rect x="35" y="50" width="450" height="430" rx="32" fill="#FFF8F0" stroke="#E48743" stroke-width="3"/>
        <text x="260" y="104" text-anchor="middle" font-size="30" font-weight="700" fill="#12233F">one scalar</text>
        <circle cx="260" cy="192" r="38" fill="#FBEDE2" stroke="#E48743" stroke-width="4"/>
        <text x="260" y="204" text-anchor="middle" font-size="34" font-weight="700" fill="#E48743">m</text>
        <path d="M260 235 L155 330 M260 235 L260 330 M260 235 L365 330" stroke="#E48743" stroke-width="6" marker-end="url(#arrB)"/>
        <text x="260" y="422" text-anchor="middle" font-size="23" fill="#617187">same teaching coordinate</text>

        <rect x="545" y="50" width="450" height="430" rx="32" fill="#F2F7FC" stroke="#2D67B1" stroke-width="3"/>
        <text x="770" y="104" text-anchor="middle" font-size="30" font-weight="700" fill="#12233F">one signal per neuron</text>
        <g fill="#EAF1FA" stroke="#2D67B1" stroke-width="4"><circle cx="665" cy="192" r="34"/><circle cx="770" cy="192" r="34"/><circle cx="875" cy="192" r="34"/></g>
        <g fill="#2D67B1" font-size="29" font-weight="700" text-anchor="middle"><text x="665" y="202">δ₁</text><text x="770" y="202">δ₂</text><text x="875" y="202">δ₃</text></g>
        <path d="M665 230 L665 330 M770 230 L770 330 M875 230 L875 330" stroke="#2D67B1" stroke-width="6" marker-end="url(#arrB)"/>
        <text x="770" y="422" text-anchor="middle" font-size="23" fill="#617187">identifies the postsynaptic cell</text>

        <rect x="1055" y="50" width="450" height="430" rx="32" fill="#F6F2FB" stroke="#7654B5" stroke-width="3"/>
        <text x="1280" y="104" text-anchor="middle" font-size="30" font-weight="700" fill="#12233F">signals within one tree</text>
        <circle cx="1280" cy="190" r="34" fill="#F0ECF8" stroke="#7654B5" stroke-width="4"/>
        <g stroke="#7654B5" stroke-width="6" fill="none"><path d="M1280 225 L1280 300 L1175 365"/><path d="M1280 300 L1385 365"/></g>
        <g fill="#F0ECF8" stroke="#7654B5" stroke-width="4"><circle cx="1175" cy="365" r="28"/><circle cx="1385" cy="365" r="28"/></g>
        <text x="1280" y="422" text-anchor="middle" font-size="23" fill="#617187">addresses different subtrees</text>
      </g>
    </svg>"""


def conductance_svg() -> str:
    return """
    <svg viewBox="0 0 820 540" class="wide-svg">
      <g font-family="Arial,sans-serif">
        <path d="M405 455 L405 310 M405 350 L260 230 M405 350 L555 225 M260 230 L175 115 M260 230 L315 95 M555 225 L500 95 M555 225 L660 120" fill="none" stroke="#168F83" stroke-width="12" stroke-linecap="round"/>
        <circle cx="405" cy="475" r="48" fill="#E48743" stroke="#FFFFFF" stroke-width="6"/>
        <g fill="#3D9667"><circle cx="175" cy="115" r="11"/><circle cx="315" cy="95" r="11"/><circle cx="500" cy="95" r="11"/></g>
        <circle cx="555" cy="225" r="14" fill="#BF4E5A"/>
        <g font-size="20" font-weight="800"><text x="160" y="82" fill="#3D9667">E</text><text x="300" y="62" fill="#3D9667">E</text><text x="485" y="62" fill="#3D9667">E</text><text x="570" y="205" fill="#BF4E5A">I</text></g>
        <path d="M600 224 H744" stroke="#BF4E5A" stroke-width="4" stroke-dasharray="8 7"/>
        <rect x="646" y="168" width="150" height="104" rx="20" fill="#F9E9EB"/>
        <text x="721" y="207" text-anchor="middle" font-size="23" font-weight="700" fill="#BF4E5A">shunt</text>
        <text x="721" y="240" text-anchor="middle" font-size="20" fill="#617187">raises gᵗᵒᵗ</text>
      </g>
    </svg>"""


def focal_comparison_svg() -> str:
    return """
    <svg viewBox="0 0 1120 650" class="wide-svg">
      <defs><marker id="focalArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0 0 L12 6 L0 12 z" fill="#8795A5"/></marker></defs>
      <g font-family="Arial,sans-serif">
        <rect x="30" y="55" width="465" height="510" rx="28" fill="#F3F7FB" stroke="#C9D8EA" stroke-width="3"/>
        <rect x="625" y="55" width="465" height="510" rx="28" fill="#EEF8F3" stroke="#B9DDCA" stroke-width="3"/>
        <text x="262" y="108" text-anchor="middle" font-size="29" font-weight="700" fill="#2D67B1">matched additive current</text>
        <text x="857" y="108" text-anchor="middle" font-size="29" font-weight="700" fill="#3D9667">focal shunting conductance</text>
        <g fill="none" stroke="#3D9667" stroke-linecap="round" stroke-linejoin="round">
          <path d="M260 478 L260 385 M260 395 L150 300 M260 395 L370 300 M150 300 L90 195 M150 300 L205 180 M370 300 L325 180 M370 300 L430 205" stroke-width="8"/>
          <path d="M855 478 L855 385 M855 395 L745 300 M855 395 L965 300 M745 300 L685 195 M745 300 L800 180 M965 300 L920 180 M965 300 L1025 205" stroke-width="8"/>
        </g>
        <g fill="#EF8B43" stroke="#FFFFFF" stroke-width="5"><circle cx="260" cy="495" r="32"/><circle cx="855" cy="495" r="32"/></g>
        <g fill="#FFFFFF" stroke="#3D9667" stroke-width="3"><circle cx="150" cy="300" r="9"/><circle cx="370" cy="300" r="9"/><circle cx="745" cy="300" r="9"/><circle cx="965" cy="300" r="9"/></g>
        <circle cx="370" cy="300" r="14" fill="#2D67B1" stroke="#FFFFFF" stroke-width="4"/>
        <circle cx="965" cy="300" r="14" fill="#BF4E5A" stroke="#FFFFFF" stroke-width="4"/>
        <path d="M506 310 H606" stroke="#8795A5" stroke-width="6" marker-end="url(#focalArrow)"/>
        <text x="558" y="270" text-anchor="middle" font-size="19" font-weight="700" fill="#617187">same baseline</text>
        <text x="558" y="296" text-anchor="middle" font-size="19" font-weight="700" fill="#617187">first-order current</text>
        <text x="262" y="540" text-anchor="middle" font-size="20" fill="#617187">somatic voltage restored</text>
        <text x="262" y="568" text-anchor="middle" font-size="17" fill="#617187">local dendritic voltage is not matched</text>
        <text x="857" y="548" text-anchor="middle" font-size="20" fill="#3D9667">only the shunt changes G</text>
        <path d="M965 282 C930 230 900 210 855 198" fill="none" stroke="#BF4E5A" stroke-width="5" stroke-dasharray="8 7"/>
        <text x="850" y="170" text-anchor="middle" font-size="20" font-weight="700" fill="#BF4E5A">descendant route</text>
      </g>
    </svg>"""


def transport_svg() -> str:
    return """
    <svg viewBox="0 0 850 560" class="wide-svg">
      <defs><marker id="down" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#7654B5"/></marker></defs>
      <path d="M420 475 L420 345 M420 362 L245 220 M420 362 L605 215 M245 220 L145 92 M245 220 L315 70 M605 215 L545 72 M605 215 L720 95" fill="none" stroke="#B7C2CC" stroke-width="13" stroke-linecap="round"/>
      <path d="M420 475 L420 362 L605 215 L720 95" fill="none" stroke="#7654B5" stroke-width="12" stroke-linecap="round" marker-end="url(#down)"/>
      <circle cx="420" cy="492" r="48" fill="#E48743" stroke="#FFFFFF" stroke-width="6"/>
      <g font-family="Arial,sans-serif" font-size="23" font-weight="700"><text x="724" y="62" text-anchor="middle" fill="#7654B5">compartment n</text><text x="505" y="522" fill="#E48743">somatic learning signal δ₀</text></g>
      <g fill="#FFFFFF" stroke="#7654B5" stroke-width="4"><circle cx="720" cy="95" r="11"/><circle cx="605" cy="215" r="11"/><circle cx="420" cy="362" r="11"/></g>
    </svg>"""


def credit_operator_svg() -> str:
    """Show how an available feedback pathway transforms a noisy gradient."""
    return """
    <svg viewBox="0 0 1450 600" class="wide-svg" role="img" aria-label="Credit operator transforms an exact stochastic gradient into an available update">
      <defs><marker id="opArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0 0 L12 6 L0 12 z" fill="#64748B"/></marker></defs>
      <g font-family="Arial,sans-serif">
        <rect x="35" y="92" width="390" height="410" rx="30" fill="#FFFFFF" stroke="#C9D8EA" stroke-width="3"/>
        <text x="230" y="142" text-anchor="middle" font-size="28" font-weight="700" fill="#12233F">stochastic BP gradient</text>
        <text x="230" y="188" text-anchor="middle" font-size="33" font-weight="700" fill="#2D67B1">μ<tspan baseline-shift="sub" font-size="22">BP</tspan> = μ + ξ</text>
        <path d="M95 410 C150 250 205 390 260 235 C310 130 350 318 382 210" fill="none" stroke="#9EB9DD" stroke-width="6"/>
        <path d="M95 395 C160 330 220 300 382 205" fill="none" stroke="#2D67B1" stroke-width="8"/>
        <text x="230" y="463" text-anchor="middle" font-size="20" fill="#617187">signal μ plus mini-batch noise ξ</text>

        <path d="M445 300 H560" stroke="#64748B" stroke-width="7" marker-end="url(#opArrow)"/>
        <rect x="585" y="155" width="280" height="290" rx="34" fill="#F0ECF8" stroke="#7654B5" stroke-width="4"/>
        <text x="725" y="222" text-anchor="middle" font-size="30" font-weight="700" fill="#7654B5">credit operator M</text>
        <g fill="#7654B5" opacity=".85"><rect x="650" y="270" width="28" height="105" rx="6"/><rect x="700" y="242" width="28" height="133" rx="6"/><rect x="750" y="304" width="28" height="71" rx="6"/><rect x="800" y="215" width="28" height="160" rx="6"/></g>
        <text x="725" y="414" text-anchor="middle" font-size="20" fill="#617187">select · assign · scale</text>

        <path d="M885 300 H1000" stroke="#64748B" stroke-width="7" marker-end="url(#opArrow)"/>
        <rect x="1025" y="92" width="390" height="410" rx="30" fill="#FFFFFF" stroke="#B9DED7" stroke-width="3"/>
        <text x="1220" y="142" text-anchor="middle" font-size="28" font-weight="700" fill="#12233F">available local update</text>
        <text x="1220" y="188" text-anchor="middle" font-size="33" font-weight="700" fill="#168F83">μ<tspan baseline-shift="sub" font-size="22">route</tspan> = M μ<tspan baseline-shift="sub" font-size="22">BP</tspan></text>
        <path d="M1085 395 C1150 330 1210 300 1372 205" fill="none" stroke="#168F83" stroke-width="8"/>
        <path d="M1085 432 C1160 370 1240 345 1372 296" fill="none" stroke="#B9DED7" stroke-width="6" stroke-dasharray="9 8"/>
        <text x="1220" y="463" text-anchor="middle" font-size="20" fill="#617187">retains some signal and some noise</text>
      </g>
    </svg>"""


def hierarchy_svg() -> str:
    return """
    <svg viewBox="0 0 1100 610" class="wide-svg">
      <g fill="none" stroke-linecap="round">
        <path d="M550 515 L550 420 M550 420 L300 305 M550 420 L800 305 M300 305 L175 190 M300 305 L425 190 M800 305 L675 190 M800 305 L925 190" stroke="#AEBAC5" stroke-width="9"/>
        <path d="M175 190 L115 88 M175 190 L235 88 M425 190 L365 88 M425 190 L485 88" stroke="#2D67B1" stroke-width="7"/>
        <path d="M675 190 L615 88 M675 190 L735 88 M925 190 L865 88 M925 190 L985 88" stroke="#7654B5" stroke-width="7"/>
      </g>
      <g fill="#FFFFFF" stroke-width="4"><circle cx="300" cy="305" r="13" stroke="#2D67B1"/><circle cx="800" cy="305" r="13" stroke="#7654B5"/><circle cx="550" cy="420" r="13" stroke="#168F83"/></g>
      <circle cx="550" cy="535" r="48" fill="#E48743" stroke="#FFFFFF" stroke-width="6"/>
      <g font-family="Arial,sans-serif" font-size="22" font-weight="700" text-anchor="middle">
        <text x="550" y="592" fill="#617187">one neuron · eight terminal streams</text>
        <text x="175" y="50" fill="#2D67B1">subtree channel 1</text><text x="925" y="50" fill="#7654B5">subtree channel 2</text>
      </g>
    </svg>"""


def evidence_svg() -> str:
    rows = [
        ("neuron identity", "SUPPORTED IN MODELS", "dominant in the standard-task model benchmarks", "#168F83", "#E6F3F0"),
        ("subtree addresses", "CONDITIONAL", "large when local updates conflict; topology adds a narrow gain", "#C77A31", "#FBEDE2"),
        ("route gain by shunting", "CONDITIONAL", "emerges only in permissive high-conductance regimes", "#C77A31", "#FBEDE2"),
        ("anatomical route capacity", "STRUCTURALLY SUPPORTED", "sparse candidate routes on reconstructed arbors", "#7654B5", "#F0ECF8"),
        ("endogenous morphology alignment", "NOT ESTABLISHED", "measured visual-response analyses are null", "#6F7C8A", "#EEF2F5"),
    ]
    parts = []
    for j, (name, status, note, color, pale) in enumerate(rows):
        y = 28 + j * 108
        parts.append(f"""
        <rect x="18" y="{y}" width="1160" height="86" rx="20" fill="#FFFFFF" stroke="#DDE5EA" stroke-width="2"/>
        <circle cx="60" cy="{y+43}" r="15" fill="{color}"/>
        <text x="92" y="{y+35}" font-size="25" font-weight="700" fill="#12233F">{name}</text>
        <text x="92" y="{y+64}" font-size="18" fill="#617187">{note}</text>
        <rect x="845" y="{y+20}" width="305" height="46" rx="23" fill="{pale}" stroke="{color}" stroke-width="2"/>
        <text x="997" y="{y+50}" text-anchor="middle" font-size="18" font-weight="800" fill="{color}">{status}</text>
        """)
    return f'<svg viewBox="0 0 1200 585" class="wide-svg"><g font-family="Arial,sans-serif">{"".join(parts)}</g></svg>'


def img(name: str, alt: str, cls: str = "plot") -> str:
    asset = ASSETS / f"{name}.png"
    # A content hash is unnecessary here; the nanosecond mtime is enough to
    # prevent headless Chromium from reusing a stale crop during rebuilds.
    version = asset.stat().st_mtime_ns
    return f'<img class="{cls}" src="assets/{name}.png?v={version}" alt="{html.escape(alt)}">'


def card(title: str, body: str, *, tone: str = "teal", cls: str = "") -> str:
    return f'<div class="card tone-{tone} {cls}"><div class="card-title">{title}</div>{body}</div>'


def stat(value: str, label: str, *, tone: str = "teal") -> str:
    return f'<div class="stat tone-{tone}"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'


def slide(kicker: str, title: str, body: str, takeaway: str, source: str = "") -> dict[str, str]:
    return {"kicker": kicker, "title": title, "body": body, "takeaway": takeaway, "source": source}


def build_slides() -> list[dict[str, str]]:
    slides: list[dict[str, str]] = []

    slides.append({
        "class": "title-slide",
        "kicker": "Kempner Learning Dynamics Workshop · 20-minute talk",
        "title": "When can dendritic structure help local credit assignment?",
        "body": f"""
          <div class="title-grid">
            <div class="title-copy">
              <div class="title-rule"></div>
              <div class="title-sub">From backpropagation to neuron coordinates, dendritic addresses, conductance-dependent route gain, and an alignment boundary</div>
              <div class="authors">Houman Safaai · Maceo Richards · Bernardo L. Sabatini</div>
              <div class="affiliation">Kempner Institute, Harvard University · Harvard Medical School</div>
            </div>
            <div class="hero-dendrite">{dendrite_svg(labels=True)}</div>
          </div>""",
        "takeaway": "",
        "source": "",
    })

    slides.append(slide(
        "THE PROBLEM",
        "Learning changes network weights; credit assigns each change",
        f'<div class="full-visual credit-visual">{credit_assignment_svg()}</div><div class="equation compact credit-eq">ℒ = ℒ(f(x; w), y) &nbsp;&nbsp;·&nbsp;&nbsp; Δwᵢ = −η ∂ℒ/∂wᵢ</div>',
        "A global objective must be converted into a destination, sign, and magnitude for every trainable weight.",
    ))

    slides.append(slide(
        "THE MATHEMATICAL REFERENCE",
        "Backpropagation defines the exact neuron-specific learning signal",
        f"""
        <div class="two-col col-56-44 bp-layout">
          <div>{network_svg()}</div>
          <div class="stack compact-stack">
            <div class="equation compact">zᵤ = Σ<sub>i∈in(u)</sub>wᵢxᵢ+bᵤ, &nbsp; yᵤ=fᵤ(zᵤ)</div>
            <div class="equation compact">δᵤ ≡ ∂ℒ/∂yᵤ = Σ<sub>v:u→v</sub>w<sub>uv</sub>f′<sub>v</sub>(z<sub>v</sub>)δ<sub>v</sub></div>
            <div class="signal-definition"><b>δᵤ</b> is exact task information assigned to neuron <b>u</b>—not yet to one weight.</div>
            <div class="taxonomy">
              <div><b>fixed/random feedback</b><span>feedback alignment; direct feedback</span></div>
              <div><b>inferred targets or states</b><span>equilibrium propagation; predictive coding</span></div>
              <div><b>eligibility + learning signal</b><span>three-factor rules; e-prop</span></div>
              <div><b>dendritic teaching signals</b><span>somato-dendritic and apical-error models</span></div>
            </div>
          </div>
        </div>""",
        "Backpropagation is the information reference; biological theories differ in how the returned signal is produced.",
        "Rumelhart et al. (1986); Lillicrap et al. (2016); Scellier & Bengio (2017); Bellec et al. (2020)",
    ))

    slides.append(slide(
        "POINT-NEURON LOCAL LEARNING",
        "Local learning preserves eligibility and approximates the returned signal",
        f"""
        <div class="hero-equation point-factorization">
          <span class="eq-left">∂ℒ/∂wᵢ =</span>
          <span class="term teal"><b>xᵢ f′ᵤ(zᵤ)</b><small>local eligibility eᵢ</small></span>
          <span class="times">×</span>
          <span class="term blue"><b>δᵤ</b><small>exact neuron signal</small></span>
        </div>
        <div class="two-col col-52-48 local-rule-row">
          <div class="equation">Δwᵢ = −η eᵢ δ<sup>avail</sup><sub>u</sub></div>
          <div class="stack compact-stack">
            {card('one layer-wide scalar m', '<p>Every neuron receives the same task-dependent coordinate.</p>', tone='orange')}
            {card('one signal δ<sup>avail</sup><sub>u</sub> per neuron', '<p>Feedback identifies the postsynaptic neuron while the eligibility remains weight-specific.</p>', tone='blue')}
          </div>
        </div>
        <div class="note-band"><b>A shared scalar does not equalize the weights:</b> each update still contains a different eligibility eᵢ.</div>""",
        "Locality determines where an update is computed; feedback bandwidth determines what task information it can use.",
        "Werfel, Xie & Seung (2005); Frémaux & Gerstner (2016)",
    ))

    slides.append(slide(
        "FROM A POINT TO A TREE",
        "Dendrites turn one neuronal signal into four testable routing questions",
        f"""
        <div class="two-col col-50-50 dendrite-resource-layout">
          <div class="tree-panel labeled-tree">{dendrite_svg(compact=True, labels=True)}</div>
          <div class="stack resource-stack compact-stack">
            <div class="mapping-line">network weight wᵢ &nbsp;→&nbsp; synaptic conductance gᵢ on compartment n</div>
            <div class="equation compact"><span class="hat-symbol">δ</span><sup>V</sup><sub>u</sub> = A<sub>u</sub>c<sub>u</sub>, &nbsp; A<sub>u</sub>∈ℝ<sup>Nᵤ×K</sup>, &nbsp; c<sub>u</sub>∈ℝ<sup>K</sup></div>
            <div class="route-definitions"><b>Aᵤ</b> says where each feedback channel is delivered; <b>cᵤ</b> contains the K signed values available on the current example.</div>
            <div class="question-grid">
              {card('1 · coordinate + ownership', '<p>Which neuron—and which arbor—receives each signal?</p>', tone='blue')}
              {card('2 · dendritic address', '<p>Which compartment or subtree receives it?</p>', tone='purple')}
              {card('3 · route gain', '<p>How strongly does it reach that destination?</p>', tone='orange')}
              {card('4 · task–route alignment', '<p>Does the available route span match the credit demanded by the task?</p>', tone='teal')}
            </div>
          </div>
        </div>""",
        "The experiments test coordinate, address, gain, and alignment separately instead of treating “dendrites” as one intervention.",
    ))

    slides.append(slide(
        "DIRECTED-TREE STEADY STATE",
        "Conductance makes dendritic voltage a normalized quotient",
        f"""
        <div class="two-col col-44-56 conductance-layout">
          <div>{conductance_svg()}</div>
          <div class="stack compact-stack">
            <div class="ei-definition"><b>G<sup>E</sup><sub>n</sub>=Σg<sup>E</sup><sub>i</sub>x<sup>E</sup><sub>i</sub>≥0</b><b>G<sup>I</sup><sub>n</sub>=Σg<sup>I</sup><sub>j</sub>x<sup>I</sup><sub>j</sub>≥0</b><span>E and I are distinguished by reversal potentials E<sub>E</sub> and E<sub>I</sub>, not by negative conductance.</span></div>
            <div class="equation compact">g<sup>tot</sup><sub>n</sub> = g<sup>L</sup><sub>n</sub> + G<sup>E</sup><sub>n</sub> + G<sup>I</sup><sub>n</sub> + G<sup>child</sup><sub>n</sub>, &nbsp; R<sup>tot</sup><sub>n</sub>=1/g<sup>tot</sup><sub>n</sub></div>
            <div class="equation multiline">Vₙ = R<sup>tot</sup><sub>n</sub>(g<sup>L</sup><sub>n</sub>E<sub>L</sub> + G<sup>E</sup><sub>n</sub>E<sub>E</sub> + G<sup>I</sup><sub>n</sub>E<sub>I</sub> + J<sup>child</sup><sub>n</sub>)</div>
            <div class="compare-row">
              {card('additive current', '<p>Changes the numerator without changing total conductance.</p>', tone='blue')}
              {card('shunting conductance', '<p>Raises g<sup>tot</sup><sub>n</sub> and lowers R<sup>tot</sup><sub>n</sub>, even when net shunt current is small.</p>', tone='red')}
            </div>
            <div class="shunt-callout">Vₙ≈E<sub>I</sub> ⇒ g<sub>I</sub>x<sub>I</sub>(E<sub>I</sub>−Vₙ)≈0, but g<sub>I</sub>x<sub>I</sub> still changes the denominator.</div>
          </div>
        </div>""",
        "Shunting changes local sensitivity because conductance appears in the voltage denominator.",
        "Holt & Koch (1997); Chance et al. (2002); Gidon & Segev (2012)",
    ))

    slides.append(slide(
        "EXACT LOCAL FACTOR · TRANSPORTED ERROR",
        "The exact dendritic gradient is local eligibility × transported compartment error",
        f"""
        <div class="two-col col-48-52 exact-transport-layout">
          <div class="transport-composite">
            <div>{transport_svg()}</div>
            <div class="transport-caption"><b>one somatic error δ<sup>V</sup><sub>0,u</sub></b><span>is transported into a field over compartments</span></div>
          </div>
          <div class="stack compact-stack exact-transport-equations">
            <div class="definition-box">δ<sup>V</sup><sub>n,u</sub> ≡ ∂ℒ/∂Vₙ &nbsp;: exact error assigned to compartment n of neuron u</div>
            <div class="hero-equation small gradient-factorization">
              <span class="eq-left">∂ℒ/∂gᵢ =</span>
              <span class="term teal"><b>xᵢR<sup>tot</sup><sub>n</sub>(E<sup>rev</sup><sub>i</sub>−Vₙ)</b><small>local dendritic eligibility e<sup>den</sup><sub>i</sub></small></span>
              <span class="times">×</span>
              <span class="term purple"><b>δ<sup>V</sup><sub>n,u</sub></b><small>transported compartment error</small></span>
            </div>
            <div class="equation compact">δ<sup>V</sup><sub>n,u</sub> = δ<sup>V</sup><sub>0,u</sub><span class="tilde-symbol">α</span><sub>n</sub></div>
            <div class="path-product"><span class="tilde-symbol">α</span><sub>n</sub> = ∏<sub>(i→k)∈path(n→0)</sub> f′<sub>i</sub>(V<sub>i</sub>) R<sup>tot</sup><sub>k</sub> g<sup>den</sup><sub>i→k</sub></div>
            <div class="transport-scope"><b>Directed tree:</b> one exact path product. &nbsp; <b>Reciprocal cable:</b> the same field is obtained from the steady-state adjoint.</div>
          </div>
        </div>""",
        "Feedback must deliver a compartment-specific field; the synaptic eligibility itself is local and exact.",
        "Almeida (1987); Pineda (1987); Schiess, Urbanczik & Senn (2016)",
    ))

    slides.append(slide(
        "THE CREDIT OPERATOR",
        "A feedback pathway selects, assigns, and scales stochastic credit",
        f"""
        <div class="operator-layout">
          <div class="operator-visual">{credit_operator_svg()}</div>
          <div class="operator-equations">
            <div class="equation compact">μ<sub>BP</sub>=μ+ξ, &nbsp; E[ξ]=0, &nbsp; Cov(ξ)=Σ</div>
            <div class="equation compact">μ<sub>route</sub>=Mμ<sub>BP</sub>, &nbsp; w<sup>+</sup>=w−ημ<sub>route</sub></div>
            <div class="operator-note"><span><b>M = I:</b> unrestricted backpropagation of the stochastic gradient. Restricted routes change the span, assignment, or gain of the available update.</span></div>
          </div>
        </div>""",
        "The operator M lets one theory describe scalar, neuron-specific, subtree-targeted, and exact compartment feedback.",
    ))

    slides.append(slide(
        "SIGNAL–NOISE PHASE THEORY",
        "Operator utility balances retained signal, update cost, and noise",
        f"""
        <div class="two-col col-54-46 utility-layout">
          <div class="stack compact-stack">
            <div class="utility main-utility">
              <div class="utility-name">operator utility · positive task alignment required</div>
              <div class="equation multiline">U(M)=<span class="frac"><span>[μ<sup>T</sup>Mμ]²</span><span>2L<sub>sm</sub>[‖Mμ‖²+tr(MΣM<sup>T</sup>)]</span></span></div>
            </div>
            <div class="three-term-row">
              <span class="pill teal">task-aligned signal</span>
              <span class="pill orange">finite-step update cost</span>
              <span class="pill red">admitted stochastic noise</span>
            </div>
            {stat('ρ<sub>s</sub> = 0.937', 'utility versus observed norm-matched one-step progress across 540 trained conditions', tone='teal')}
            {stat('ρ<sub>s</sub> = 0.916', 'utility versus final trained accuracy', tone='purple')}
          </div>
          <div class="plot-card utility-plot">{img('operator_utility_validation', 'Operator utility versus observed progress')}</div>
        </div>
        <div class="scope-strip">Exact for an isotropic quadratic with Hessian L<sub>sm</sub>I; otherwise a curvature bound and one-step smoothness guarantee. It predicts large contrasts and within-family progress—not every trajectory-accrued difference.</div>""",
        "Restricted routing helps only when its retained, aligned signal compensates for the signal it discards and the noise it admits.",
    ))

    slides.append(slide(
        "STANDARD IMAGE TASKS",
        "Neuron-specific feedback dominates standard image tasks",
        f"""
        <div class="two-col col-50-50 plot-pair standard-plots">
          <div class="plot-card"><div class="plot-label">MNIST · <span class="green-text">green = shunting</span> · <span class="blue-text">blue = additive</span></div>{img('mnist_strict_scalar', 'MNIST strict-scalar feedback ladder')}</div>
          <div class="plot-card"><div class="plot-label">Flattened CIFAR-10 · additive tree · “exact path” = exact compartment field</div>{img('cifar_confirmatory_ladder', 'CIFAR-10 feedback ladder')}</div>
        </div>
        <div class="standard-summary-row">
          {stat('+7.8 to +16.4 pp', 'strict scalar → one distinct feedback coordinate per neuron', tone='blue')}
          {stat('−0.86 to +0.19 pp', 'neuron-specific coordinate → exact compartment field', tone='purple')}
          {stat('≈ backprop', 'exact field on flattened CIFAR-10, within the predefined ±1-point margin', tone='teal')}
        </div>""",
        "Most of the gain comes from preserving distinct feedback coordinates across neurons; exact within-tree resolution adds little.",
        "Paired-seed means; CIFAR exact field and BP are equivalent within the predefined ±1-point margin.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 1 · DEFINITION",
        "Credit conflict creates a demand for branch-specific signals",
        f"""
        <div class="two-col col-60-40 task-layout">
          <div class="plot-card task-card">{img('path_necessity_task', 'Context-gated branch-conflict task')}</div>
          <div class="stack">
            {card('input', '<p>All B branches receive a Fashion-MNIST image and form nonzero eligibility.</p>', tone='blue')}
            {card('forward selector', '<p>Context c selects the branch whose image determines the target.</p>', tone='teal')}
            {card('conflict dose χ', '<p>Nonselected images vary from class-compatible (χ=0) to opposite-class (χ=1).</p>', tone='red')}
            <div class="equation compact">d<sup>branch</sup><sub>b</sub> = N<sup>−1</sup>Σ<sub>t</sub>δ<sub>t</sub><span class="indicator-one">1</span>[c<sub>t</sub>=b]x<sub>t,b</sub></div>
            <div class="equation compact">d<sup>shared</sup><sub>b</sub> = N<sup>−1</sup>Σ<sub>t</sub>δ<sub>t</sub>B<sup>−1</sup>x<sub>t,b</sub></div>
            <div class="branch-symbols"><b>N</b>: trials &nbsp;·&nbsp; <b>δ<sub>t</sub></b>: downstream logit gradient &nbsp;·&nbsp; <b>1[·]</b>: indicator<br><b>d<sub>b</sub></b>: mean update direction for branch b</div>
          </div>
        </div>""",
        "At zero conflict, one shared signal is sufficient; at high conflict, different branches require opposite updates.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 1 · RESULT",
        "Shared credit fails at the predicted conflict boundary",
        f"""
        <div class="two-col col-67-33 result-layout">
          <div class="plot-card large-plot">{img('path_necessity_results', 'Branch-conflict theory and trained results')}</div>
          <div class="stack">
            <div class="equation boundary-eq">λ<sub>shared</sub>=B−2χ(B−1) &nbsp;→&nbsp; χ<sub>c</sub>=B/[2(B−1)]</div>
            {stat('35–58 pp', 'branch-specific gain over shared feedback at full conflict', tone='purple')}
            {card('causal control', '<p>Cyclically deranged routes fail despite identical rank and sparsity.</p>', tone='red')}
            {card('implementation control', '<p>Analytic BP, the correct route, and a gated-point calculation are equivalent forms of the same routing matrix.</p>', tone='gray')}
          </div>
        </div>""",
        "The task establishes a need for an address—not a unique need for dendritic material.",
        "B = 2, 4, 8 branches; all empirical collapse points track the analytic boundary.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 2 · DEFINITION",
        "Nested tasks ask whether a few subtree addresses are efficient",
        f"""
        <div class="hierarchy-definition-layout">
          <div class="plot-card hierarchy-task-plot">{img('hierarchy_task_current', 'Eight-context hierarchical task with selected stream and distance-dependent distractors')}</div>
          <div class="stack compact-stack hierarchy-side">
            <div class="equation compact">δ<sup>V, avail</sup> = A<sub>K</sub>β, &nbsp; rank(A<sub>K</sub>)=K</div>
            <div class="plot-card hierarchy-bandwidth-plot">{img('hierarchy_bandwidth_current', 'Within-neuron feedback bandwidth K equals 1, 2, 4, or 8')}</div>
            {card('matched controls', '<p>Rank, sparsity, parameter count, and forward resources are held fixed; only route assignment, basis, or topology changes.</p>', tone='gray')}
          </div>
        </div>""",
        "Intermediate K tests whether tree-structured feedback is a useful low-dimensional basis for task credit.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 2 · RESULT",
        "Subtree addresses help—but fine topology adds only a narrow gain",
        f"""
        <div class="two-col col-63-37 result-layout">
          <div class="plot-card large-plot">{img('hierarchy_learning_current', 'Accuracy across feedback bandwidth with directly labeled correct, deranged, and best non-anatomical routes')}</div>
          <div class="stack">
            {stat('+61.1 pp', 'correct ancestry assignment over cyclic derangement at the same K=4 bandwidth', tone='purple')}
            {stat('+1.27 pp', 'correct ancestry over the strongest matched non-anatomical low-rank control at K=4', tone='teal')}
            {card('against the matched low-rank basis', '<p>Ancestry loses at K=1,2; wins only at K=4; and ties at full rank K=8. Rewiring removes the intermediate advantage.</p>', tone='orange')}
            {card('equivalence', '<p>Dendritic, grouped-point, and gated-point implementations coincide for the same routed field.</p>', tone='gray')}
          </div>
        </div>""",
        "The large effect is the value of within-neuron addresses; the anatomy-specific effect is conditional and modest.",
        "Twenty paired seeds; K=4 matched-topology contrast survives BH correction.",
    ))

    slides.append(slide(
        "BIOLOGICAL CAPACITY",
        "Real arbors provide sparse candidate routes, mostly through coarse geometry",
        f"""
        <div class="two-col col-48-52 anatomy-layout">
          <div class="stack anatomy-visuals compact-stack">
            <div class="plot-card mapped-arbor"><div class="plot-label">measured MICrONS morphology with mapped E/I contacts</div>{img('mapped_reconstruction', 'Reconstructed MICrONS arbor with mapped excitatory and inhibitory contacts')}</div>
            <div class="plot-card address-inset"><div class="plot-label">branch points define nested route supports</div>{img('ancestry_addresses', 'Nested ancestry addresses on a dendritic tree')}</div>
          </div>
          <div class="stack anatomy-data">
            <div class="equation compact capture-equation"><span class="capture-formula">C<sub>A</sub>(q)=‖P<sub>A</sub>q‖²/‖q‖²</span><span class="capture-definition">fraction of field energy in the route span</span></div>
            <div class="anatomy-evidence-grid">
              <div class="plot-card capture-plot">{img('capture_per_wire_current', 'Wiring-normalized field capture')}</div>
              <div class="anatomy-metrics">
                {stat('85%', 'of dense field capture', tone='teal')}
                {stat('≈7%', 'of dense route-matrix connections', tone='purple')}
                {stat('14.2×', 'dense capture per connection', tone='blue')}
                {stat('≈2.7×', 'over a density-matched shuffled dictionary', tone='orange')}
              </div>
            </div>
            <div class="precision-note">Route-matrix connections are not cable length or energy. The ordering replicated in 47 held-out cells, and the subtree advantage was positive in 10/10 quality-controlled cells from a second MICrONS mouse. These are modeled fields, not observed task gradients.</div>
          </div>
        </div>""",
        "Measured morphology supplies a sparse route dictionary; most capacity comes from branch depth and coarse topology.",
    ))

    slides.append(slide(
        "ROUTE GAIN",
        "Focal shunting changes descendant credit only in permissive regimes",
        f"""
        <div class="two-col col-58-42 shunt-layout">
          <div class="plot-card shunt-schematic">{focal_comparison_svg()}</div>
          <div class="stack">
            <div class="equation compact route-gain-sensitivity">∂ log α<sup>cond</sup><sub>n</sub> / ∂G<sup>I</sup><sub>k</sub> = −R<sup>tot</sup><sub>k</sub> <span class="indicator-one">1</span>[k∈𝒜(n)]</div>
            <div class="route-gain-reading">A shunt directly attenuates only routes descending through compartment k; the effect scales with local input resistance.</div>
            <div class="plot-card boundary-plot">{img('focal_boundary_current', 'Current Figure 8F electrotonic boundary for focal shunting')}</div>
            {card('standard passive calibration', '<p>At R<sub>m</sub>=15,000 Ω cm², the shunt-minus-current localization contrast is effectively zero.</p>', tone='gray')}
            {card('high-conductance regime', '<p>Descendant-localized changes emerge and survive active-channel extensions.</p>', tone='teal')}
          </div>
        </div>""",
        "Shunting is a state-dependent route-gain mechanism, not a generic learning advantage.",
        "Baseline first-order focal current is matched and somatic voltage restored; local dendritic voltage is not matched.",
    ))

    slides.append(slide(
        "MEASURED FUNCTIONAL EVIDENCE",
        "Measured visual responses show no morphology-specific alignment",
        f"""
        <div class="two-col col-44-56 measured-null-layout">
          <div class="stack">
            <div class="measured-pipeline">
              <div>measured presynaptic responses</div><span>→</span>
              <div>mapped conductances on each reconstructed target arbor</div><span>→</span>
              <div>held-out postsynaptic-response prediction</div>
            </div>
            {card('question', '<p>Does the measured input–output relation align more strongly with nested subtrees than with matched alternative route dictionaries?</p>', tone='teal')}
            {card('inferential scope', '<p>Seven target cells from one MICrONS mouse; exact compartment errors are the reference.</p>', tone='gray')}
          </div>
          <div class="plot-card measured-null-plot">
            <div class="plot-label">Held-out learning and topology-minus-control effects</div>
            <div class="measured-grid">{img('boundary_learning', 'Held-out complete-tree learning')}{img('boundary_topology', 'Topology effects')}</div>
          </div>
        </div>
        <div class="null-result-strip">Nested subtrees do not outperform random or site-shuffled routes for held-out prediction, field capture, or within-arbor structure–function similarity.</div>""",
        "Measured anatomy supplies candidate routes, but these visual responses provide no evidence that the task uses them preferentially.",
    ))

    slides.append(slide(
        "CONTROLLED ALIGNMENT RESCUE",
        "The same anatomical routes capture task credit aligned to their span",
        f"""
        <div class="two-col col-48-52 alignment-layout">
          <div class="stack compact-stack">
            <div class="equation compact">φ(a) = √a u<sub>∥</sub> + √(1−a) u<sub>⊥</sub>, &nbsp; a∈[0,1]</div>
            <div class="alignment-definitions">u<sub>∥</sub>∈col(A), &nbsp; u<sub>⊥</sub>⊥col(A), &nbsp; ‖u<sub>∥</sub>‖=‖u<sub>⊥</sub>‖=1</div>
            <div class="plot-card rotation-plot">{img('alignment_rotation', 'Fixed-energy task field rotated into the subtree route span')}</div>
            {card('controlled quantity', '<p>Anatomy, field energy, curvature, and route count are fixed; only alignment with the subtree span changes.</p>', tone='gray')}
          </div>
          <div class="stack">
            <div class="plot-card alignment-gain-plot">{img('alignment_gain', 'Field capture across imposed alignment')}</div>
            <div class="equation compact">C<sub>A</sub>[φ(a)]=a &nbsp; by construction</div>
            {card('n = 8 reconstructed cells', '<p>Controls test whether the gain is specific to the true subtree span. The manipulation proves conditional representational sufficiency—not trained learning or endogenous biological use.</p>', tone='teal')}
          </div>
        </div>""",
        "Alignment is sufficient for representational capture in the model; endogenous morphology-specific alignment remains unestablished.",
    ))

    slides.append(slide(
        "SYNTHESIS",
        "One alignment–bandwidth plane organizes the wins and nulls",
        f"""
        <div class="two-col col-67-33 phase-summary-layout">
          <div class="plot-card phase-dominant">{img('phase_plane', 'Task-route alignment and relative feedback bandwidth phase plane')}</div>
          <div class="stack phase-reading">
            {card('low relative bandwidth', '<p>Too few independent coordinates: neuron identity or branch address is the bottleneck.</p>', tone='orange')}
            {card('aligned intermediate bandwidth', '<p>Restricted routes can retain task credit while rejecting irrelevant dimensions.</p>', tone='teal')}
            {card('full rank or weak alignment', '<p>Route capacity saturates, or available structure does not match the task.</p>', tone='purple')}
            <div class="phase-axis-definition"><b>vertical:</b> relative bandwidth K/r<sub>eff</sub><br><b>horizontal:</b> task–route alignment</div>
            <div class="precision-note">Coordinates are estimated separately within each experiment. Regime tint and the K/r<sub>eff</sub>=1 boundary are theoretical, not fitted.</div>
          </div>
        </div>""",
        "Useful route resolution requires both sufficient feedback bandwidth and alignment between the task-credit field and the available routes.",
    ))

    slides.append(slide(
        "TAKE-HOME",
        "Coordinate → address → gain, all conditional on task–route alignment",
        f"""
        <div class="takehome-layout">
          <div class="takehome-flow">
            <div class="flow-node blue"><b>coordinate</b><span>which neuron?</span></div><div class="flow-arrow">→</div>
            <div class="flow-node purple"><b>address</b><span>which subtree?</span></div><div class="flow-arrow">→</div>
            <div class="flow-node orange"><b>gain</b><span>how strongly?</span></div><div class="flow-gate">enabled by <b>alignment + bandwidth</b></div>
          </div>
          <div class="takehome-grid">
            {card('1 · neuron-specific coordinates come first', '<p>Across MNIST and flattened CIFAR-10, preserving distinct feedback across neurons closes most of the scalar-to-exact gap.</p>', tone='blue')}
            {card('2 · addresses matter when local updates conflict', '<p>Branch-specific signals become necessary when simultaneously active subtrees require different or opposite changes.</p>', tone='purple')}
            {card('3 · topology helps only in a matched regime', '<p>Nested routes add a modest advantage at intermediate aligned bandwidth; reconstructed arbors supply sparse candidate routes.</p>', tone='teal')}
            {card('4 · conductance regulates gain conditionally', '<p>Focal shunting localizes modeled credit only in permissive states, and endogenous morphology-specific use remains unestablished.</p>', tone='orange')}
          </div>
          <div class="closing-statement">Dendrites are a conditional substrate for routing local credit—not a general replacement for backpropagation.</div>
        </div>""",
        "The central contribution is a predictive boundary map: when restricted dendritic routes help, when they do not, and why.",
    ))

    return slides


STYLE = r"""
:root {
  --ink:#12233F; --muted:#617187; --paper:#F7F7F4; --white:#FFFFFF;
  --grid:#DDE5EA; --teal:#168F83; --teal-pale:#E6F3F0;
  --blue:#2D67B1; --blue-pale:#EAF1FA; --purple:#7654B5;
  --purple-pale:#F0ECF8; --orange:#E48743; --orange-pale:#FBEDE2;
  --red:#BF4E5A; --red-pale:#F9E9EB; --green:#3D9667; --gray:#8B98A7;
}
* { box-sizing:border-box; }
html, body { margin:0; width:1920px; height:1080px; overflow:hidden; background:var(--paper); }
body { font-family:Arial,"Nimbus Sans",sans-serif; color:var(--ink); }
.slide { display:none; position:absolute; inset:0; width:1920px; height:1080px; padding:54px 76px 82px; background:
  radial-gradient(circle at 96% 2%, rgba(22,143,131,.055), transparent 27%),
  linear-gradient(180deg,#FBFBF8 0%,#F6F7F4 100%); }
.slide.active { display:block; }
.slide::before { content:""; position:absolute; left:76px; right:76px; top:141px; height:2px; background:linear-gradient(90deg,var(--teal),var(--grid) 22%,var(--grid)); }
.kicker { font-size:18px; line-height:1; font-weight:800; letter-spacing:2.4px; color:var(--teal); text-transform:uppercase; margin-bottom:11px; }
h1 { margin:0; font-family:Georgia,"Nimbus Roman",serif; font-size:52px; line-height:1.08; letter-spacing:-1.1px; color:var(--ink); max-width:1740px; }
.content { position:absolute; left:76px; right:76px; top:171px; bottom:132px; }
.takeaway { position:absolute; left:76px; right:76px; bottom:47px; min-height:61px; display:flex; align-items:center; padding:12px 26px 12px 64px; border:2px solid #C9E3DE; border-radius:16px; background:#EFF8F6; color:var(--ink); font-size:25px; line-height:1.24; font-weight:700; }
.takeaway::before { content:"→"; position:absolute; left:22px; width:29px; height:29px; display:grid; place-items:center; color:white; border-radius:50%; background:var(--teal); font-size:20px; }
.source { position:absolute; left:82px; bottom:20px; color:#7A8794; font-size:14px; letter-spacing:.05px; }
.page { position:absolute; right:80px; bottom:20px; color:#7A8794; font-size:15px; font-weight:700; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:34px; height:100%; align-items:center; }
.col-56-44 { grid-template-columns:56% 44%; }.col-52-48 { grid-template-columns:52% 48%; }
.col-50-50 { grid-template-columns:1fr 1fr; }.col-48-52 { grid-template-columns:48% 52%; }
.col-44-56 { grid-template-columns:44% 56%; }.col-60-40 { grid-template-columns:60% 40%; }
.col-67-33 { grid-template-columns:67% 33%; }.col-63-37 { grid-template-columns:63% 37%; }
.col-58-42 { grid-template-columns:58% 42%; }
.three-col { display:grid; grid-template-columns:repeat(3,1fr); gap:24px; }
.stack { display:flex; flex-direction:column; gap:18px; justify-content:center; min-width:0; }
.card { position:relative; padding:22px 26px; border-radius:22px; border:2px solid var(--grid); background:rgba(255,255,255,.94); box-shadow:0 9px 28px rgba(18,35,63,.06); }
.card-title { font-size:24px; font-weight:800; color:var(--ink); margin-bottom:8px; }
.card p { margin:0; font-size:22px; line-height:1.32; color:#405167; }
.tone-teal { border-color:#B9DED7; }.tone-blue { border-color:#CADAF0; }.tone-purple { border-color:#D6C9EA; }
.tone-orange { border-color:#F1D3B8; }.tone-red { border-color:#EBC7CB; }.tone-gray { border-color:#D8DFE5; }
.analogy { display:block; color:var(--muted); margin-top:10px; font-size:18px; font-weight:700; }
.equation { padding:22px 28px; border-radius:20px; background:white; border:2px solid var(--grid); box-shadow:0 8px 24px rgba(18,35,63,.05); font-family:Georgia,"Nimbus Roman",serif; font-size:40px; line-height:1.2; text-align:center; }
.equation.compact { font-size:32px; padding:16px 22px; }.equation.multiline { font-size:35px; }
.equation-caption { color:var(--muted); font-size:22px; text-align:center; margin-top:-7px; }
.frac { display:inline-flex; flex-direction:column; vertical-align:middle; align-items:stretch; }
.frac > span:first-child { border-bottom:2px solid currentColor; padding:0 10px 7px; }
.frac > span:last-child { padding:7px 10px 0; }
.hat-symbol,.tilde-symbol { position:relative; display:inline-block; padding-top:.08em; }.hat-symbol::after { content:"ˆ"; position:absolute; left:50%; top:-.44em; transform:translateX(-50%); font-size:.58em; font-family:Georgia,"Nimbus Roman",serif; }.tilde-symbol::after { content:"~"; position:absolute; left:50%; top:-.48em; transform:translateX(-50%); font-size:.64em; font-family:Georgia,"Nimbus Roman",serif; }
.hero-equation { display:flex; align-items:stretch; justify-content:center; gap:18px; margin:56px auto 45px; font-family:Georgia,"Nimbus Roman",serif; }
.hero-equation.small { margin:0; gap:12px; flex-wrap:wrap; }
.eq-left,.times { display:flex; align-items:center; font-size:47px; }.hero-equation.small .eq-left,.hero-equation.small .times { font-size:34px; }
.term { min-width:360px; padding:24px 28px 20px; border-radius:22px; text-align:center; border:3px solid; background:white; }
.hero-equation.small .term { min-width:300px; padding:18px 20px; }
.term b { display:block; font-size:47px; line-height:1; }.hero-equation.small .term b { font-size:31px; }
.term small { display:block; margin-top:14px; font:700 20px Arial,sans-serif; color:var(--muted); }.hero-equation.small .term small { font-size:17px; }
.term.teal { color:var(--teal); border-color:#A8D5CD; }.term.blue { color:var(--blue); border-color:#BDD1EC; }.term.purple { color:var(--purple); border-color:#CBBDE3; }
.lower-cards { margin-top:20px; }.lower-cards .card { min-height:180px; }
.precision-note,.note-band,.definition-box,.final-question { padding:16px 21px; border-radius:16px; color:#526176; background:#EEF2F5; font-size:20px; line-height:1.34; }
.note-band { margin:8px auto 0; max-width:1420px; text-align:center; }
.full-visual { height:590px; display:flex; align-items:center; justify-content:center; }
.wide-svg,.network-svg,.dendrite { width:100%; height:100%; }
.network-svg { max-height:620px; }.bandwidth { height:570px; }.roadmap { height:555px; }
.tree-panel { height:650px; padding:8px; }.tree-panel .dendrite { max-height:650px; }
.resource-stack .card { padding:21px 25px; }
.compare-row,.contrast-row { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.equation-stack { padding-right:12px; }.path-product { padding:17px 20px; text-align:center; font:28px Georgia,serif; color:var(--purple); background:var(--purple-pale); border-radius:16px; }
.axis-ribbon { display:flex; align-items:center; justify-content:center; gap:20px; margin:4px auto; color:var(--muted); font-size:19px; font-weight:700; }
.axis-line { width:720px; height:4px; background:linear-gradient(90deg,var(--orange),var(--purple)); position:relative; }
.axis-line::after { content:""; position:absolute; right:-3px; top:-7px; border-left:14px solid var(--purple); border-top:9px solid transparent; border-bottom:9px solid transparent; }
.plot-pair { height:515px; align-items:stretch; }.plot-card { display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:0; min-height:0; padding:16px; border:2px solid var(--grid); border-radius:22px; background:white; box-shadow:0 9px 28px rgba(18,35,63,.06); overflow:hidden; }
.plot { max-width:100%; max-height:100%; object-fit:contain; }.plot-label { width:100%; color:var(--muted); font-size:20px; font-weight:800; text-align:left; padding:0 6px 9px; }
.plot-label + .plot { max-height:calc(100% - 42px); }
.measured-grid { display:grid; grid-template-columns:44% 56%; gap:10px; width:100%; height:100%; align-items:center; }
.plot-label + .measured-grid { height:calc(100% - 42px); }
.measured-grid .plot { width:100%; max-height:96%; }
.stat-row { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-top:18px; }
.stat { padding:13px 18px; border-radius:17px; background:white; border:2px solid var(--grid); min-height:96px; }
.stat-value { font:700 32px Georgia,serif; color:var(--ink); }.stat-label { margin-top:4px; color:var(--muted); font-size:17px; line-height:1.22; }
.stat.tone-teal .stat-value { color:var(--teal); }.stat.tone-purple .stat-value { color:var(--purple); }.stat.tone-red .stat-value { color:var(--red); }.stat.tone-blue .stat-value { color:var(--blue); }.stat.tone-orange .stat-value { color:var(--orange); }
.theory-grid { align-items:stretch; }.phase-card { padding:12px; }.utility { padding:18px; background:white; border:2px solid #CFE2DE; border-radius:20px; }.utility-name { color:var(--teal); font-size:19px; font-weight:800; text-transform:uppercase; letter-spacing:1.3px; }.utility .equation { border:0; box-shadow:none; padding:10px 4px 0; font-size:31px; }
.three-term-row,.task-chips { display:flex; flex-wrap:wrap; gap:12px; justify-content:center; }.pill { padding:9px 17px; border-radius:999px; font-size:18px; font-weight:800; background:#EEF2F5; color:var(--ink); }.pill.teal { background:var(--teal-pale); color:var(--teal); }.pill.orange { background:var(--orange-pale); color:#B9692E; }.pill.red { background:var(--red-pale); color:var(--red); }.pill.blue { background:var(--blue-pale); color:var(--blue); }
.task-layout,.result-layout { align-items:stretch; }.task-card,.large-plot { height:100%; }.boundary-eq { font-size:34px; color:var(--red); border-color:#EBC7CB; }
.hierarchy { height:650px; }.bandwidth-scale { display:grid; gap:10px; }.bandwidth-scale > div { display:grid; grid-template-columns:105px 1fr; gap:12px; align-items:center; padding:13px 18px; background:white; border:2px solid #D8CDE9; border-radius:15px; }.bandwidth-scale b { color:var(--purple); font-size:25px; }.bandwidth-scale span { color:var(--muted); font-size:20px; }
.depth-layout { height:535px; align-items:stretch; }.schematic-card { height:285px; }.depth-result { height:100%; }.definition-box { text-align:center; background:var(--purple-pale); color:#554070; }.depth-stats { margin-top:14px; }.depth-stats .stat { min-height:84px; }
.anatomy-layout { align-items:stretch; }.anatomy-tree { position:relative; height:660px; }.route-overlay { position:absolute; left:45px; right:45px; bottom:35px; display:flex; justify-content:space-between; color:var(--purple); font-size:19px; font-weight:800; }.capture-plot { height:400px; }.compact-stats .stat { min-height:88px; padding:10px 14px; }.compact-stats .stat-value { font-size:29px; }
.shunt-layout { align-items:stretch; }.shunt-schematic { height:100%; }.boundary-plot { height:325px; }.shunt-layout .card { padding:15px 20px; }.shunt-layout .card-title { font-size:21px; }.shunt-layout .card p { font-size:18px; }
.biological-pair { height:510px; }.contrast-row { margin-top:18px; }.contrast-row .card { min-height:148px; }
.synthesis-layout { align-items:stretch; }.evidence-panel { height:650px; display:flex; align-items:center; }.final-stack .card { padding:18px 22px; }.final-question { background:linear-gradient(90deg,var(--teal-pale),var(--purple-pale)); color:var(--ink); font-size:23px; }

/* Revised 20-slide workshop sequence. */
.credit-visual { height:565px; }.credit-eq { position:absolute; left:300px; right:300px; bottom:4px; }
.bp-layout { align-items:stretch; }.bp-layout .network-svg { max-height:555px; margin-top:30px; }
.compact-stack { gap:12px; }.signal-definition,.mapping-line,.route-definitions,.operator-note,.alignment-definitions,.phase-definition { padding:13px 17px; border-radius:14px; background:#EEF2F5; color:#526176; font-size:19px; line-height:1.3; }
.signal-definition { background:var(--blue-pale); color:#315881; }
.taxonomy { display:grid; gap:7px; }.taxonomy > div { display:grid; grid-template-columns:44% 56%; align-items:center; padding:9px 13px; border-left:5px solid var(--grid); background:#FFFFFF; border-radius:9px; box-shadow:0 3px 10px rgba(18,35,63,.035); }
.taxonomy b { font-size:16px; color:var(--ink); }.taxonomy span { font-size:15px; color:var(--muted); }
.point-factorization { margin:30px auto 20px; }.point-factorization .term { min-width:385px; }.local-rule-row { height:245px; align-items:stretch; }.local-rule-row > .equation { align-self:center; }.local-rule-row .card { padding:15px 20px; }.local-rule-row .card-title { font-size:21px; }.local-rule-row .card p { font-size:18px; }
.dendrite-resource-layout { align-items:stretch; }.labeled-tree { height:660px; }.mapping-line { background:var(--teal-pale); color:#2F6C65; font-weight:700; text-align:center; }.route-definitions { background:var(--purple-pale); color:#5E4A82; font-size:17px; }.resource-stack .card { padding:13px 19px; }.resource-stack .card-title { font-size:20px; }.resource-stack .card p { font-size:18px; }
.question-grid { display:grid; grid-template-columns:1fr 1fr; gap:11px; }.question-grid .card { min-height:116px; padding:13px 17px; }.question-grid .card-title { font-size:18px; margin-bottom:5px; }.question-grid .card p { font-size:17px; line-height:1.23; }
.conductance-layout,.gradient-layout,.transport-layout { align-items:stretch; }.conductance-layout > div:first-child,.gradient-layout > div:first-child,.transport-layout > div:first-child { display:flex; align-items:center; }.conductance-layout .equation.multiline { font-size:30px; }.shunt-callout { padding:13px 18px; border:2px solid #EBC7CB; border-radius:14px; background:var(--red-pale); color:#8C3D48; font:700 19px/1.3 Georgia,serif; text-align:center; }
.ei-definition { display:grid; grid-template-columns:auto auto; gap:6px 18px; align-items:center; padding:12px 17px; border-radius:14px; background:linear-gradient(90deg,var(--teal-pale),var(--red-pale)); color:#38576A; font-size:18px; }.ei-definition b:first-child { color:var(--green); }.ei-definition b:nth-child(2) { color:var(--red); }.ei-definition span { grid-column:1 / -1; color:#536276; font-size:16px; text-align:center; }
.gradient-factorization { margin:0; }.gradient-factorization .term { min-width:250px; }.gradient-factorization .term:first-of-type { min-width:475px; }.gradient-factorization .term b { font-size:27px; }.transport-layout .card { padding:17px 21px; }.transport-layout .card-title { font-size:21px; }.transport-layout .card p { font-size:19px; }
.exact-transport-layout { align-items:stretch; }.transport-composite { display:grid; grid-template-rows:1fr auto; height:100%; min-height:0; }.transport-composite > div:first-child { min-height:0; display:flex; align-items:center; }.transport-caption { display:flex; flex-direction:column; gap:5px; margin:0 30px 12px; padding:12px 18px; border-radius:14px; background:var(--purple-pale); text-align:center; color:#5E4A82; }.transport-caption b { font-size:20px; }.transport-caption span { font-size:17px; }.exact-transport-equations { gap:11px; }.exact-transport-equations .definition-box { font-size:18px; padding:12px 17px; }.exact-transport-equations .equation.compact { font-size:29px; padding:13px 18px; }.exact-transport-equations .path-product { font-size:24px; padding:13px 17px; }.transport-scope { padding:12px 16px; border-radius:14px; background:#EEF2F5; color:#536276; font-size:17px; line-height:1.28; text-align:center; }
.operator-layout { display:grid; grid-template-rows:455px 170px; gap:10px; height:100%; }.operator-visual { min-height:0; }.operator-equations { display:grid; grid-template-columns:1fr 1fr 1.25fr; gap:15px; align-items:stretch; }.operator-equations .equation { display:flex; align-items:center; justify-content:center; font-size:27px; }.operator-note { display:flex; align-items:center; font-size:18px; }.operator-note b { display:inline-block; margin-right:.28em; color:var(--ink); }
.utility-layout { height:575px; align-items:stretch; }.main-utility { padding:14px; }.main-utility .equation { padding:7px 2px 0; font-size:29px; }.utility-plot { padding:10px; }.scope-strip { margin-top:12px; padding:10px 18px; border-radius:14px; background:#EEF2F5; color:#56667A; font-size:17px; text-align:center; }.utility-layout .stat { min-height:73px; padding:8px 15px; }.utility-layout .stat-value { font-size:27px; }.utility-layout .stat-label { font-size:15px; }
.standard-plots { height:470px; }.comparison-table { margin-top:12px; display:grid; border:2px solid var(--grid); border-radius:15px; overflow:hidden; background:#FFFFFF; }.comparison-table > div { display:grid; grid-template-columns:1.5fr 1fr 1fr; gap:12px; padding:7px 16px; border-top:1px solid var(--grid); font-size:17px; align-items:center; }.comparison-table > div:first-child { border-top:0; }.comparison-head { background:#EEF2F5; color:var(--muted); font-weight:800; }.blue-text { color:var(--blue); }.green-text { color:var(--green); }.red-text { color:var(--red); }
.standard-summary-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-top:12px; }.standard-summary-row .stat { min-height:101px; padding:11px 16px; }.standard-summary-row .stat-value { font-size:27px; }.standard-summary-row .stat-label { font-size:16px; }
.hierarchy-definition-layout { display:grid; grid-template-columns:55% 45%; gap:28px; height:100%; align-items:stretch; }.hierarchy-task-plot { height:100%; }.hierarchy-bandwidth-plot { height:385px; }.hierarchy-side .card { padding:15px 20px; }.hierarchy-side .card-title { font-size:20px; }.hierarchy-side .card p { font-size:18px; }
.branch-symbols { padding:8px 12px; border-radius:12px; background:#EEF2F5; color:var(--muted); font-size:18px; line-height:1.24; text-align:center; }.branch-symbols b { color:var(--ink); }.indicator-one { font-family:Arial,Nimbus Sans,sans-serif; font-weight:800; }
.four-stat-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:8px; }.four-stat-row .stat { min-width:0; }.capture-equation { padding:9px 16px; display:grid; grid-template-columns:auto 1fr; align-items:baseline; justify-content:center; gap:25px; }.capture-formula { color:var(--ink); font:26px/1.2 Georgia,serif; white-space:nowrap; }.capture-definition { color:var(--muted); font:17px/1.2 Arial,sans-serif; }.anatomy-evidence-grid { display:grid; grid-template-columns:46% 54%; gap:12px; height:388px; min-height:0; }.anatomy-layout .capture-plot { height:100%; padding:10px; }.anatomy-layout .capture-plot .plot { width:100%; height:100%; }.anatomy-metrics { display:grid; grid-template-rows:repeat(4,1fr); gap:9px; min-height:0; }.anatomy-metrics .stat { min-height:0; padding:9px 13px; display:flex; flex-direction:column; justify-content:center; }.anatomy-metrics .stat-value { font-size:27px; }.anatomy-metrics .stat-label { font-size:15px; }.anatomy-data { height:100%; justify-content:flex-start; }.anatomy-data .precision-note { font-size:15px; line-height:1.28; padding:10px 14px; }
.anatomy-visuals { height:100%; justify-content:flex-start; overflow:hidden; }.anatomy-visuals .plot-card { min-height:0; flex:none; }.mapped-arbor { height:68%; }.address-inset { height:29%; }.anatomy-visuals .plot-label { font-size:16px; }.anatomy-visuals .plot { width:100%; height:calc(100% - 36px); min-height:0; object-fit:contain; }
.measured-null-layout { height:570px; align-items:stretch; }.measured-pipeline { display:grid; gap:7px; text-align:center; }.measured-pipeline div { padding:14px; border-radius:14px; background:#FFFFFF; border:2px solid var(--grid); font-size:19px; font-weight:700; }.measured-pipeline span { color:var(--teal); font-size:25px; line-height:1; }.measured-null-plot { height:100%; }.null-result-strip { margin-top:12px; padding:13px 20px; border-radius:14px; background:#EEF2F5; color:#56667A; font-size:18px; font-weight:700; text-align:center; }
.alignment-layout { align-items:stretch; }.rotation-plot { height:310px; }.alignment-gain-plot { height:455px; }.alignment-definitions { background:var(--purple-pale); color:#5E4A82; text-align:center; }.alignment-layout .card { padding:14px 19px; }.alignment-layout .card-title { font-size:20px; }.alignment-layout .card p { font-size:18px; }
.final-layout { align-items:stretch; }.phase-stack { gap:10px; }.final-phase { height:455px; padding:9px; }.phase-definition { background:var(--purple-pale); color:#5E4A82; text-align:center; font-family:Georgia,serif; }.evidence-panel.categorical { height:535px; }.final-stack { gap:11px; }.final-stack .final-question { font-size:18px; padding:13px 17px; }
.route-gain-sensitivity { font-size:27px !important; }.route-gain-reading { padding:11px 15px; border-radius:13px; background:var(--orange-pale); color:#80542E; font-size:17px; line-height:1.27; text-align:center; }
.phase-summary-layout { align-items:stretch; }.phase-dominant { height:100%; padding:10px; }.phase-dominant .plot { width:100%; height:100%; }.phase-reading { gap:12px; }.phase-reading .card { padding:16px 19px; }.phase-reading .card-title { font-size:20px; }.phase-reading .card p { font-size:18px; }.phase-axis-definition { padding:13px 17px; border-radius:14px; background:var(--purple-pale); color:#5E4A82; font-size:19px; line-height:1.35; }.phase-reading .precision-note { padding:12px 16px; font-size:16px; }
.takehome-layout { height:100%; display:grid; grid-template-rows:128px 1fr 83px; gap:18px; }.takehome-flow { display:flex; align-items:center; justify-content:center; gap:18px; }.flow-node { width:260px; padding:18px 22px; border-radius:20px; background:white; border:3px solid; text-align:center; box-shadow:0 8px 24px rgba(18,35,63,.05); }.flow-node b { display:block; font:700 29px Georgia,serif; }.flow-node span { display:block; margin-top:5px; color:var(--muted); font-size:18px; }.flow-node.blue { border-color:#BDD1EC; color:var(--blue); }.flow-node.purple { border-color:#CBBDE3; color:var(--purple); }.flow-node.orange { border-color:#F1D3B8; color:#B9692E; }.flow-arrow { font-size:43px; color:#7B8998; }.flow-gate { margin-left:22px; padding:17px 22px; border-radius:18px; background:linear-gradient(90deg,var(--teal-pale),var(--purple-pale)); color:#42546A; font-size:20px; }.takehome-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }.takehome-grid .card { display:flex; flex-direction:column; justify-content:center; padding:22px 27px; }.takehome-grid .card-title { font-size:23px; }.takehome-grid .card p { font-size:20px; line-height:1.3; }.closing-statement { display:flex; align-items:center; justify-content:center; padding:16px 25px; border-radius:18px; background:#12233F; color:white; font:700 26px/1.25 Georgia,serif; text-align:center; }

.title-slide { padding:0; background:linear-gradient(135deg,#F9FAF7 0%,#F4F7F5 64%,#EEF5F3 100%); }
.title-slide::before,.title-slide .takeaway,.title-slide .source { display:none; }
.title-slide .kicker { position:absolute; top:76px; left:92px; z-index:2; }
.title-slide h1 { position:absolute; left:92px; top:170px; width:940px; font-size:76px; line-height:1.03; letter-spacing:-2.1px; z-index:2; }
.title-slide .content { inset:0; }.title-grid { height:100%; display:grid; grid-template-columns:54% 46%; }
.title-copy { position:relative; padding-left:92px; }.title-rule { position:absolute; top:137px; width:116px; height:7px; border-radius:4px; background:var(--teal); }
.title-sub { position:absolute; top:455px; width:815px; color:#496078; font:31px/1.35 Arial,sans-serif; }
.authors { position:absolute; top:675px; font-size:27px; font-weight:800; }.affiliation { position:absolute; top:721px; color:var(--muted); font-size:21px; }
.hero-dendrite { padding:75px 30px 40px 0; display:flex; align-items:center; }
.title-slide .page { bottom:35px; }
"""


def render_master(slides: list[dict[str, str]]) -> None:
    sections = []
    total = len(slides)
    for index, item in enumerate(slides, start=1):
        slide_class = item.get("class", "")
        takeaway = f'<div class="takeaway">{item["takeaway"]}</div>' if item["takeaway"] else ""
        source = f'<div class="source">{item["source"]}</div>' if item["source"] else ""
        sections.append(f"""
        <section class="slide {slide_class}" data-slide="{index}">
          <div class="kicker">{item['kicker']}</div>
          <h1>{item['title']}</h1>
          <div class="content">{item['body']}</div>
          {takeaway}{source}<div class="page">{index:02d} / {total:02d}</div>
        </section>""")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=1920,initial-scale=1"><style>{STYLE}</style></head>
<body>{''.join(sections)}
<script>
const requested = Number(new URLSearchParams(window.location.search).get('slide') || 1);
const slides = [...document.querySelectorAll('.slide')];
slides.forEach((element, index) => element.classList.toggle('active', index + 1 === requested));
</script></body></html>"""
    MASTER.write_text(document, encoding="utf-8")


def render_pngs(slides: list[dict[str, str]]) -> None:
    chromium = shutil.which("chromium-browser") or shutil.which("chromium")
    if chromium is None:
        raise SystemExit("Chromium is required to render the HTML/SVG master")
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = HERE / ".render_tmp"
    raw_dir.mkdir(exist_ok=True)
    for index in range(1, len(slides) + 1):
        raw = raw_dir / f"slide_{index:02d}.png"
        command = [
            chromium,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=900",
            "--force-device-scale-factor=1.333333",
            f"--window-size={CSS_WIDTH},{CSS_HEIGHT + CHROMIUM_FRAME_HEIGHT}",
            f"--screenshot={raw}",
            f"{MASTER.as_uri()}?slide={index}",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)
        with Image.open(raw) as image:
            rendered = image.convert("RGB")
            if rendered.width >= PNG_WIDTH and rendered.height >= PNG_HEIGHT:
                rendered = rendered.crop((0, 0, PNG_WIDTH, PNG_HEIGHT))
            else:
                rendered = rendered.resize((PNG_WIDTH, PNG_HEIGHT), Image.Resampling.LANCZOS)
            rendered.save(PNG_DIR / f"slide_{index:02d}.png", optimize=True)
    shutil.rmtree(raw_dir)


def build_contact_sheet(slide_count: int) -> None:
    thumb_size = (512, 288)
    columns = 4
    rows = (slide_count + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 532 + 20, rows * 308 + 20), "#E7ECEF")
    for index in range(1, slide_count + 1):
        with Image.open(PNG_DIR / f"slide_{index:02d}.png") as image:
            thumb = ImageOps.fit(image.convert("RGB"), thumb_size, method=Image.Resampling.LANCZOS)
        x = 20 + ((index - 1) % columns) * 532
        y = 20 + ((index - 1) // columns) * 308
        canvas.paste(thumb, (x, y))
    canvas.save(CONTACT, optimize=True)


def build_pdf(slide_count: int) -> None:
    document = fitz.open()
    page_rect = fitz.Rect(0, 0, 960, 540)
    for index in range(1, slide_count + 1):
        page = document.new_page(width=page_rect.width, height=page_rect.height)
        page.insert_image(page.rect, filename=str(PNG_DIR / f"slide_{index:02d}.png"))
    document.set_metadata({
        "title": "When dendritic structure helps local credit assignment",
        "author": "Houman Safaai, Maceo Richards, Bernardo L. Sabatini",
        "subject": "15--20 minute ML-audience workshop presentation",
    })
    document.save(PDF_OUT, deflate=True)
    document.close()


def validate(slide_count: int) -> None:
    expected = (PNG_WIDTH, PNG_HEIGHT)
    files = sorted(PNG_DIR.glob("slide_*.png"))
    if len(files) != slide_count:
        raise SystemExit(f"Expected {slide_count} slide PNGs, found {len(files)}")
    for path in files:
        with Image.open(path) as image:
            if image.size != expected:
                raise SystemExit(f"{path.name}: expected {expected}, got {image.size}")
            extrema = image.convert("L").getextrema()
            if extrema[1] - extrema[0] < 40:
                raise SystemExit(f"{path.name}: likely blank render")
    with fitz.open(PDF_OUT) as document:
        if document.page_count != slide_count:
            raise SystemExit(f"PDF has {document.page_count} pages; expected {slide_count}")
        for page in document:
            if abs(page.rect.width / page.rect.height - 16 / 9) > 0.001:
                raise SystemExit("PDF contains a non-16:9 page")
    print(f"Validated {slide_count} slides at {PNG_WIDTH}x{PNG_HEIGHT}; PDF is 16:9")


def main() -> None:
    render_pdf_assets()
    slides = build_slides()
    render_master(slides)
    render_pngs(slides)
    build_contact_sheet(len(slides))
    build_pdf(len(slides))
    validate(len(slides))


if __name__ == "__main__":
    main()
