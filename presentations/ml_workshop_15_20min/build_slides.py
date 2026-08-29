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
from PIL import Image, ImageChops, ImageOps


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
        "focal_boundary",
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
            trim_white(image).save(ASSETS / f"{name}.png", optimize=True)
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
            rendered.save(ASSETS / f"{output_name}.png", optimize=True)
        temporary.unlink()

    render_clip(
        "main_figure_02_native.pdf",
        "mnist_strict_scalar",
        (0.025, 0.238, 0.337, 0.58),
    )
    render_clip(
        "main_figure_07_native.pdf",
        "capture_per_wire_current",
        (0.49, 0.345, 0.995, 0.995),
    )
    render_clip(
        "main_figure_09_native.pdf",
        "boundary_learning",
        (0.29, 0.005, 0.57, 0.315),
        post_crop_left=0.012,
    )
    render_clip(
        "main_figure_09_native.pdf",
        "boundary_topology",
        (0.545, 0.005, 0.985, 0.315),
        post_crop_left=0.055,
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


def credit_pipeline_svg() -> str:
    nodes = [
        (115, "task loss", "global outcome", "#E48743"),
        (430, "network", "many layers", "#2D67B1"),
        (745, "neuron", "many branches", "#168F83"),
        (1060, "subtree", "many synapses", "#7654B5"),
        (1375, "parameter", "one update", "#BF4E5A"),
    ]
    blocks = []
    for x, title, sub, color in nodes:
        blocks.append(
            f'<rect x="{x}" y="185" width="245" height="175" rx="28" fill="#FFFFFF" stroke="{color}" stroke-width="4"/>'
            f'<circle cx="{x+44}" cy="225" r="16" fill="{color}" opacity=".18"/>'
            f'<circle cx="{x+44}" cy="225" r="7" fill="{color}"/>'
            f'<text x="{x+122}" y="273" text-anchor="middle" class="svg-title">{title}</text>'
            f'<text x="{x+122}" y="313" text-anchor="middle" class="svg-sub">{sub}</text>'
        )
    arrows = []
    for x in (360, 675, 990, 1305):
        arrows.append(f'<path d="M{x} 272 H{x+58}" stroke="#98A6B6" stroke-width="5" marker-end="url(#arrow)"/>')
    return f"""
    <svg viewBox="0 0 1740 600" class="wide-svg">
      <defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#98A6B6"/></marker></defs>
      <style>.svg-title{{font:700 29px Arial,sans-serif;fill:#12233F}}.svg-sub{{font:23px Arial,sans-serif;fill:#617187}}</style>
      {''.join(blocks)}{''.join(arrows)}
      <path d="M1498 408 C1280 520 520 520 238 408" fill="none" stroke="#BF4E5A" stroke-width="6" stroke-dasharray="13 11" marker-end="url(#arrow)"/>
      <text x="875" y="548" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#BF4E5A">credit must travel backward; plasticity remains local</text>
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
        <g fill="#2D67B1"><circle cx="175" cy="115" r="11"/><circle cx="315" cy="95" r="11"/><circle cx="500" cy="95" r="11"/></g>
        <circle cx="555" cy="225" r="14" fill="#BF4E5A"/>
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
        <text x="558" y="276" text-anchor="middle" font-size="20" font-weight="700" fill="#617187">same local ΔV</text>
        <text x="262" y="548" text-anchor="middle" font-size="21" fill="#617187">forward voltage matched</text>
        <text x="857" y="548" text-anchor="middle" font-size="21" fill="#3D9667">G changes → transported q changes</text>
        <path d="M965 282 C930 230 900 210 855 198" fill="none" stroke="#BF4E5A" stroke-width="5" stroke-dasharray="8 7"/>
        <text x="850" y="170" text-anchor="middle" font-size="20" font-weight="700" fill="#BF4E5A">descendant route</text>
      </g>
    </svg>"""


def transport_svg() -> str:
    return """
    <svg viewBox="0 0 850 560" class="wide-svg">
      <defs><marker id="down" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0 0 L11 5.5 L0 11 z" fill="#7654B5"/></marker></defs>
      <path d="M420 475 L420 345 M420 362 L245 220 M420 362 L605 215 M245 220 L145 92 M245 220 L315 70 M605 215 L545 72 M605 215 L720 95" fill="none" stroke="#B7C2CC" stroke-width="13" stroke-linecap="round"/>
      <path d="M720 95 L605 215 L420 362 L420 475" fill="none" stroke="#7654B5" stroke-width="12" stroke-linecap="round" marker-end="url(#down)"/>
      <circle cx="420" cy="492" r="48" fill="#E48743" stroke="#FFFFFF" stroke-width="6"/>
      <g font-family="Arial,sans-serif" font-size="23" font-weight="700"><text x="724" y="62" text-anchor="middle" fill="#7654B5">compartment n</text><text x="505" y="522" fill="#E48743">somatic error δ₀</text></g>
      <g fill="#FFFFFF" stroke="#7654B5" stroke-width="4"><circle cx="720" cy="95" r="11"/><circle cx="605" cy="215" r="11"/><circle cx="420" cy="362" r="11"/></g>
    </svg>"""


def experiment_roadmap_svg() -> str:
    cards = [
        (65, "1", "ordinary tasks", "Is neuron identity enough?", "#2D67B1"),
        (450, "2", "controlled conflict", "When must branches differ?", "#BF4E5A"),
        (835, "3", "hierarchical routes", "Which bandwidth and topology?", "#7654B5"),
        (1220, "4", "reconstructed arbors", "Are routes available and used?", "#168F83"),
    ]
    out = []
    for x, num, title, q, color in cards:
        out.append(f"""
        <rect x="{x}" y="115" width="330" height="310" rx="30" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>
        <circle cx="{x+50}" cy="166" r="28" fill="{color}"/><text x="{x+50}" y="176" text-anchor="middle" fill="#FFFFFF" font-size="27" font-weight="700">{num}</text>
        <text x="{x+165}" y="236" text-anchor="middle" fill="#12233F" font-size="29" font-weight="700">{title}</text>
        <text x="{x+165}" y="290" text-anchor="middle" fill="#617187" font-size="22"><tspan x="{x+165}" dy="0">{q.split(' ')[0]} {q.split(' ')[1]}</tspan><tspan x="{x+165}" dy="31">{' '.join(q.split(' ')[2:])}</tspan></text>
        <path d="M{x+90} 362 H{x+240}" stroke="{color}" stroke-width="8" stroke-linecap="round" opacity=".72"/>
        """)
    return f'<svg viewBox="0 0 1620 540" class="wide-svg"><g font-family="Arial,sans-serif">{"".join(out)}</g></svg>'


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
        ("neuron identity", "supported", "ordinary tasks + animal coordinate", "#168F83", 1.0),
        ("subtree addresses", "conditional", "large only under task-aligned conflict", "#E48743", 0.72),
        ("route gain by shunting", "conditional", "requires a high-conductance regime", "#E48743", 0.55),
        ("endogenous morphology alignment", "not established", "measured-response analyses are null", "#8B98A7", 0.28),
    ]
    parts = []
    for j, (name, status, note, color, width) in enumerate(rows):
        y = 74 + j * 125
        parts.append(f"""
        <text x="35" y="{y+30}" font-size="27" font-weight="700" fill="#12233F">{name}</text>
        <text x="35" y="{y+65}" font-size="20" fill="#617187">{note}</text>
        <rect x="650" y="{y}" width="720" height="54" rx="27" fill="#E8EDF1"/>
        <rect x="650" y="{y}" width="{720*width:.0f}" height="54" rx="27" fill="{color}"/>
        <text x="1395" y="{y+35}" text-anchor="end" font-size="22" font-weight="700" fill="{color}">{status}</text>
        """)
    return f'<svg viewBox="0 0 1450 585" class="wide-svg"><g font-family="Arial,sans-serif">{"".join(parts)}</g></svg>'


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
        "kicker": "Kempner Learning Dynamics Workshop · 15–20 minutes",
        "title": "When dendritic structure helps local credit assignment",
        "body": f"""
          <div class="title-grid">
            <div class="title-copy">
              <div class="title-rule"></div>
              <div class="title-sub">From neuron-level errors to subtree addresses, route gain, and an alignment boundary</div>
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
        "Credit assignment is a routing problem",
        f'<div class="full-visual">{credit_pipeline_svg()}</div><div class="equation compact">parameter credit &nbsp;=&nbsp; ∂ℒ / ∂θᵢ</div>',
        "A circuit computes one global outcome, but learning must reach the local parameters that caused it.",
    ))

    slides.append(slide(
        "THE MATHEMATICAL REFERENCE",
        "Backpropagation provides exact parameter credit",
        f"""
        <div class="two-col col-56-44">
          <div>{network_svg()}</div>
          <div class="stack">
            <div class="equation">ΔW<sup>ℓ</sup><sub>ji</sub> = −η · h<sup>ℓ−1</sup><sub>i</sub> · δ<sup>ℓ</sup><sub>j</sub></div>
            {card('local factor', '<p>Presynaptic activity and the derivative at the receiving unit.</p>', tone='teal')}
            {card('nonlocal factor', '<p>A parameter-specific error assembled by the reverse chain rule.</p>', tone='red')}
            <div class="precision-note">We use backpropagation to define the target information—not as a claim about literal neural implementation.</div>
          </div>
        </div>""",
        "Backpropagation answers three questions for every parameter: destination, sign, and magnitude.",
        "Rumelhart et al. (1986); Lillicrap et al. (2020)",
    ))

    slides.append(slide(
        "LOCAL LEARNING",
        "A local rule preserves eligibility and approximates the error",
        f"""
        <div class="hero-equation">
          <span class="eq-left">Δwᵢ = −η</span>
          <span class="term teal"><b>eᵢ</b><small>synapse-local eligibility</small></span>
          <span class="times">×</span>
          <span class="term blue"><b>δ̂ᵤ</b><small>communicated learning signal</small></span>
        </div>
        <div class="three-col lower-cards">
          {card('local', '<p><b>eᵢ</b> can use presynaptic activity, voltage, driving force, and gates.</p>', tone='teal')}
          {card('task-dependent', '<p><b>δ̂ᵤ</b> may depend on a global objective even when the synaptic update is local.</p>', tone='blue')}
          {card('information-limited', '<p>The feedback pathway determines which distinctions the rule can express.</p>', tone='purple')}
        </div>""",
        "Locality specifies where an update is computed; feedback bandwidth specifies what the update can know.",
        "Hebb (1949); Frémaux & Gerstner (2016); Bellec et al. (2020)",
    ))

    slides.append(slide(
        "FEEDBACK BANDWIDTH",
        "The first bottleneck is which neuron should learn",
        f'<div class="full-visual bandwidth">{bandwidth_svg()}</div><div class="note-band"><b>Important:</b> a shared scalar does not make neurons identical. Each synapse still has its own eligibility and initialization; the scalar only removes task-specific feedback coordinates.</div>',
        "Neuron identity and within-neuron address are different credit-assignment problems.",
        "Werfel, Xie & Seung (2005); Lillicrap et al. (2016)",
    ))

    slides.append(slide(
        "FROM A POINT TO A TREE",
        "A dendritic arbor adds state, addresses, and route gain",
        f"""
        <div class="two-col col-52-48">
          <div class="tree-panel">{dendrite_svg(compact=True, labels=False)}</div>
          <div class="stack resource-stack">
            {card('1 · local state', '<p>Voltages and conductances change the eligibility computed at each synapse.</p><span class="analogy">ML analogy: feature-dependent local Jacobian</span>', tone='teal')}
            {card('2 · subtree address', '<p>A small number of signals can target nested groups of synapses.</p><span class="analogy">ML analogy: structured low-rank routing</span>', tone='purple')}
            {card('3 · route gain', '<p>Conductance can scale how strongly error propagates along one path.</p><span class="analogy">ML analogy: state-dependent feedback preconditioner</span>', tone='orange')}
          </div>
        </div>""",
        "The question is not whether dendrites add parameters—it is whether their spatial structure matches the credit the task requires.",
    ))

    slides.append(slide(
        "FORWARD DYNAMICS",
        "Conductance makes dendritic voltage a normalized quotient",
        f"""
        <div class="two-col col-44-56">
          <div>{conductance_svg()}</div>
          <div class="stack">
            <div class="equation multiline">Vₙ = <span class="frac"><span>gᴸEᴸ + Σᵢ gᵢxᵢEᵢ + Σ<sub>c</sub> g<sup>den</sup><sub>c→n</sub>a<sub>c</sub></span><span>gᴸ + Σᵢ gᵢxᵢ + Σ<sub>c</sub> g<sup>den</sup><sub>c→n</sub></span></span></div>
            <div class="equation-caption">R<sup>tot</sup><sub>n</sub> = 1 / g<sup>tot</sup><sub>n</sub> is the local input resistance.</div>
            <div class="compare-row">
              {card('additive current', '<p>Can match the local voltage change without changing <b>gᵗᵒᵗ</b>.</p>', tone='blue')}
              {card('shunting conductance', '<p>Changes both voltage and the denominator that controls sensitivity.</p>', tone='red')}
            </div>
          </div>
        </div>""",
        "A shunt can change the gain of other synapses even when its own net current is small.",
        "Holt & Koch (1997); Chance et al. (2002); Gidon & Segev (2012)",
    ))

    slides.append(slide(
        "THE EXACT DENDRITIC GRADIENT",
        "Eligibility stays local; the error becomes a field over the arbor",
        f"""
        <div class="two-col col-48-52">
          <div>{transport_svg()}</div>
          <div class="stack equation-stack">
            <div class="hero-equation small">
              <span class="eq-left">∂ℒ/∂gᵢ =</span>
              <span class="term teal"><b>xᵢR<sup>tot</sup><sub>n</sub>(Eᵢ−Vₙ)</b><small>local dendritic eligibility eᵢ</small></span>
              <span class="times">×</span>
              <span class="term purple"><b>∂ℒ/∂Vₙ</b><small>compartment error</small></span>
            </div>
            <div class="equation">∂ℒ/∂Vₙ = δ<sub>0,u</sub> · α̃ₙ</div>
            <div class="path-product">α̃ₙ = ∏<sub>(i→k) on path(n→0)</sub> f′ᵢ(Vᵢ) R<sup>tot</sup><sub>k</sub> g<sup>den</sup><sub>i→k</sub></div>
            <div class="precision-note">In a general compartmental model, the same transported field is obtained from the steady-state adjoint J<sub>V</sub><sup>T</sup>q = ∇<sub>V</sub>ℒ.</div>
          </div>
        </div>""",
        "Dendrites do not create the circuit-level error; they can distribute it through spatially structured gains and addresses.",
        "Almeida (1987); Pineda (1987)",
    ))

    slides.append(slide(
        "EXPERIMENTAL LOGIC",
        "We test the need for progressively richer credit",
        f'<div class="full-visual roadmap">{experiment_roadmap_svg()}</div><div class="axis-ribbon"><span>low feedback resolution</span><div class="axis-line"></div><span>high feedback resolution</span></div>',
        "Every positive result is paired with a matched point implementation to separate the information resource from dendritic material.",
    ))

    slides.append(slide(
        "STANDARD IMAGE TASKS",
        "Neuron-specific feedback supplies nearly all useful resolution",
        f"""
        <div class="two-col col-50-50 plot-pair">
          <div class="plot-card"><div class="plot-label">MNIST · strict-scalar confirmation</div>{img('mnist_strict_scalar', 'MNIST strict-scalar feedback ladder')}</div>
          <div class="plot-card"><div class="plot-label">CIFAR-10 · fresh additive confirmation</div>{img('cifar_confirmatory_ladder', 'CIFAR-10 feedback ladder')}</div>
        </div>
        <div class="stat-row">
          {stat('+0.19 pp', 'exact path over neuron-specific feedback on additive MNIST', tone='purple')}
          {stat('−0.86 pp', 'exact path versus neuron-specific feedback on CIFAR-10', tone='red')}
          {stat('no generic gain', 'harder data do not create a need for within-tree routing', tone='blue')}
        </div>""",
        "On ordinary classification, selecting the correct neuron is the dominant feedback bottleneck.",
        "Paired-seed means; exact intervals and tests are reported in Fig. 2 and Source Data.",
    ))

    slides.append(slide(
        "A STOCHASTIC CREDIT OPERATOR",
        "Restricted feedback helps only when it rejects more noise than signal",
        f"""
        <div class="two-col col-50-50 theory-grid">
          <div class="stack">
            <div class="equation">μ<sub>M</sub> = M(μ + ξ)</div>
            <div class="utility">
              <div class="utility-name">operator utility</div>
              <div class="equation multiline">U(M) = <span class="frac"><span>[ μ<sup>T</sup>Mμ ]²</span><span>2L [ ‖Mμ‖² + tr(MΣM<sup>T</sup>) ]</span></span></div>
            </div>
            <div class="three-term-row">
              <span class="pill teal">retained signal</span>
              <span class="pill orange">finite-step cost</span>
              <span class="pill red">admitted noise</span>
            </div>
            {stat('ρ = 0.937', 'utility versus observed one-step progress across trained conditions', tone='teal')}
          </div>
          <div class="plot-card phase-card">{img('phase_plane', 'Alignment and bandwidth phase plane')}</div>
        </div>""",
        "Useful routing requires both task alignment and the right bandwidth; more detailed feedback is not always better.",
        "Theory is exact for quadratic losses and a one-step smoothness bound otherwise.",
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
            <div class="equation compact">d<sub>b</sub><sup>branch</sup> ∝ Σᵢ δᵢ 𝟙[cᵢ=b] xᵢ,b</div>
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
            <div class="equation boundary-eq">χ<sub>c</sub> = B / [2(B−1)]</div>
            {stat('35–58 pp', 'branch-specific gain over shared feedback at full conflict', tone='purple')}
            {card('causal control', '<p>Cyclically deranged routes fail despite identical rank and sparsity.</p>', tone='red')}
            {card('implementation control', '<p>A context-gated point model matches correct routing.</p>', tone='gray')}
          </div>
        </div>""",
        "The task establishes a need for an address—not a unique need for dendritic material.",
        "B = 2, 4, 8 branches; all empirical collapse points track the analytic boundary.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 2 · DEFINITION",
        "Nested tasks ask whether a few subtree addresses are efficient",
        f"""
        <div class="two-col col-48-52">
          <div class="tree-panel hierarchy">{hierarchy_svg()}</div>
          <div class="stack">
            <div class="bandwidth-scale">
              <div><b>K = 1</b><span>one signal for the whole neuron</span></div>
              <div><b>K = 2</b><span>two coarse subtrees</span></div>
              <div><b>K = 4</b><span>four intermediate subtrees</span></div>
              <div><b>K = 8</b><span>one signal per terminal branch</span></div>
            </div>
            {card('task hierarchy', '<p>Eight active streams occupy the leaves; opposite-sign distractors strengthen with tree distance.</p>', tone='purple')}
            {card('matched controls', '<p>Same rank, sparsity, parameter count, and forward resources; only route assignment or basis changes.</p>', tone='gray')}
          </div>
        </div>""",
        "Intermediate K tests whether tree-structured feedback is a useful low-dimensional basis for task credit.",
    ))

    slides.append(slide(
        "CONTROLLED TASK 2 · RESULT",
        "Subtree addresses help—but fine topology adds only a narrow gain",
        f"""
        <div class="two-col col-63-37 result-layout">
          <div class="plot-card large-plot">{img('subtree_k_sweep', 'Accuracy across feedback bandwidth')}</div>
          <div class="stack">
            {stat('+61 pp', 'matched subtrees over one neuron-shared signal at K=4', tone='purple')}
            {stat('+1.3 pp', 'matched subtrees over the strongest non-anatomical low-rank control at K=4', tone='teal')}
            {card('boundary', '<p>Matched subtrees lose at low K and tie at full rank. Rewiring removes the intermediate advantage.</p>', tone='orange')}
            {card('equivalence', '<p>Dendritic, grouped-point, and gated-point implementations coincide for the same routed field.</p>', tone='gray')}
          </div>
        </div>""",
        "The large effect is the value of within-neuron addresses; the anatomy-specific effect is conditional and modest.",
        "Twenty paired seeds; K=4 matched-topology contrast survives BH correction.",
    ))

    slides.append(slide(
        "A SEPARATE FORWARD QUESTION",
        "Physical depth helps only when serial computation matches the task",
        f"""
        <div class="two-col col-48-52 depth-layout">
          <div class="stack">
            <div class="plot-card schematic-card">{img('physical_stage_schematic', 'Matched serial and grouped-point architectures')}</div>
            <div class="task-chips"><span class="pill teal">nested factors</span><span class="pill blue">flat factors</span><span class="pill orange">local ratios</span></div>
            <div class="definition-box"><b>D<sub>p</sub></b> = serial physical stages &nbsp;·&nbsp; <b>H</b> = task hierarchy &nbsp;·&nbsp; <b>α</b> = task–sensor alignment</div>
          </div>
          <div class="plot-card depth-result">{img('physical_depth_headline', 'Physical-depth boundary')}</div>
        </div>
        <div class="stat-row depth-stats">
          {stat('+30.9 pp', 'aligned serial BP: D3 over D1 on the calibrated H=3 task', tone='teal')}
          {stat('+11.0 pp', 'exact compartment feedback over one shared somatic signal at D3', tone='purple')}
          {stat('not universal', 'useful depth saturates and can hurt on local-ratio tasks', tone='orange')}
        </div>""",
        "Depth is a task-matched inductive bias under constraints, not a general expressivity advantage over point networks.",
        "Grouped-point and flexible point controls separate composition, parameter count, and implementation.",
    ))

    slides.append(slide(
        "BIOLOGICAL CAPACITY",
        "Real arbors provide sparse candidate routes, mostly through coarse geometry",
        f"""
        <div class="two-col col-48-52 anatomy-layout">
          <div class="tree-panel anatomy-tree">{dendrite_svg(compact=True, labels=False)}<div class="route-overlay"><span>branch points define nested supports</span><span>q → P<sub>A</sub>q</span></div></div>
          <div class="stack">
            <div class="plot-card capture-plot">{img('capture_per_wire_current', 'Wiring-normalized capture')}</div>
            <div class="stat-row compact-stats">
              {stat('85%', 'of dense field capture', tone='teal')}
              {stat('7%', 'of dense feedback connections', tone='purple')}
              {stat('≈2.8×', 'cellwise capture per connection over density-matched shuffled routes', tone='orange')}
            </div>
            <div class="precision-note">Replicated in a disjoint 47-cell cohort and ten cells from a second MICrONS mouse. These are modeled fields on measured anatomy—not observed task gradients.</div>
          </div>
        </div>""",
        "Morphology supplies a sparse route dictionary; most capacity comes from branch depth and coarse topology.",
    ))

    slides.append(slide(
        "ROUTE GAIN",
        "Focal shunting changes descendant credit only in permissive regimes",
        f"""
        <div class="two-col col-58-42 shunt-layout">
          <div class="plot-card shunt-schematic">{focal_comparison_svg()}</div>
          <div class="stack">
            <div class="equation compact">q′ = q − [κ q<sub>k</sub> / (1 + κ(G⁻¹)<sub>kk</sub>)] G⁻¹e<sub>k</sub></div>
            <div class="plot-card boundary-plot">{img('focal_boundary', 'Electrotonic boundary for focal shunting')}</div>
            {card('standard passive calibration', '<p>The shunt-minus-current localization contrast is effectively zero.</p>', tone='gray')}
            {card('high-conductance regime', '<p>Descendant-localized changes emerge and survive active-channel extensions.</p>', tone='teal')}
          </div>
        </div>""",
        "Shunting is a state-dependent route-gain mechanism, not a generic learning advantage.",
        "Matched-current controls isolate the conductance operator from the local voltage change.",
    ))

    slides.append(slide(
        "FUNCTIONAL EVIDENCE",
        "Measured responses are null; imposed task alignment rescues the same routes",
        f"""
        <div class="two-col col-50-50 plot-pair biological-pair">
          <div class="plot-card"><div class="plot-label">Measured MICrONS visual responses</div><div class="measured-grid">{img('boundary_learning', 'Held-out complete-tree learning')}{img('boundary_topology', 'Topology effects')}</div></div>
          <div class="plot-card"><div class="plot-label">Controlled rotation into the subtree span</div>{img('alignment_rescue_n', 'Alignment rescue')}</div>
        </div>
        <div class="contrast-row">
          {card('what the data show', '<p>Nested subtrees do not beat random or site-shuffled routes for held-out response prediction or field capture.</p>', tone='gray')}
          {card('what the rescue shows', '<p>Holding anatomy and field energy fixed, capture rises continuously as the task field rotates into the subtree span.</p>', tone='teal')}
        </div>""",
        "The routes are sufficient when aligned, but their endogenous use is not established by the measured visual-response task.",
    ))

    slides.append(slide(
        "SYNTHESIS",
        "Dendrites provide conditional resources for local credit",
        f"""
        <div class="two-col col-58-42 synthesis-layout">
          <div class="evidence-panel">{evidence_svg()}</div>
          <div class="stack final-stack">
            {card('1 · identify the neuron', '<p>One learning coordinate per neuron carries most of the practical benefit on ordinary tasks.</p>', tone='blue')}
            {card('2 · address the subtree', '<p>Useful when branches require different task-dependent updates and route bandwidth is matched.</p>', tone='purple')}
            {card('3 · regulate route gain', '<p>Conductance can shape transport, but only in an appropriate electrotonic state.</p>', tone='teal')}
            <div class="final-question">The decisive variable is <b>alignment between task credit and the available route span.</b></div>
          </div>
        </div>""",
        "Dendritic structure is a conditional substrate for routing local credit—not a general replacement for backpropagation.",
    ))

    return slides


STYLE = r"""
:root {
  --ink:#12233F; --muted:#617187; --paper:#F7F7F4; --white:#FFFFFF;
  --grid:#DDE5EA; --teal:#168F83; --teal-pale:#E6F3F0;
  --blue:#2D67B1; --blue-pale:#EAF1FA; --purple:#7654B5;
  --purple-pale:#F0ECF8; --orange:#E48743; --orange-pale:#FBEDE2;
  --red:#BF4E5A; --red-pale:#F9E9EB; --gray:#8B98A7;
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
.shunt-layout { align-items:stretch; }.shunt-schematic { height:100%; }.boundary-plot { height:285px; }
.biological-pair { height:510px; }.contrast-row { margin-top:18px; }.contrast-row .card { min-height:148px; }
.synthesis-layout { align-items:stretch; }.evidence-panel { height:650px; display:flex; align-items:center; }.final-stack .card { padding:18px 22px; }.final-question { background:linear-gradient(90deg,var(--teal-pale),var(--purple-pale)); color:var(--ink); font-size:23px; }

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
