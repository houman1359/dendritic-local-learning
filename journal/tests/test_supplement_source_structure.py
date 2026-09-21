"""Guard the active SI source tree and its documented retained fragments."""

import json
from pathlib import Path
import re
import sys


JOURNAL = Path(__file__).resolve().parents[1]
SUPPLEMENT = JOURNAL / "supplementary"
MASTER = SUPPLEMENT / "supplementary.tex"
sys.path.insert(0, str(JOURNAL / "scripts"))
from tex_sources import expanded_tex, tex_sources


INACTIVE_FRAGMENTS = {
    "anatomy_commonmode_methods.tex",
    "boolean_morphology_methods.tex",
    "conductance_credit_controls_figures.tex",
    "conductance_credit_demand_methods.tex",
    "conductance_expanded_rates_figure.tex",
    "conductance_expanded_rates_methods.tex",
    "credit_first_bridge_methods.tex",
    "image_ladder_controls_figure.tex",
    "image_ladder_controls_methods.tex",
    "morphology_calibration_methods.tex",
    "morphology_conductance_methods.tex",
    "morphology_credit_methods.tex",
    "morphology_end_to_end_methods.tex",
    "morphology_followup_methods.tex",
    "noise_task_identity.tex",
    "physical_depth_budget_methods.tex",
    "review_followups.tex",
    "shunt_normalized_dose_figure.tex",
    "shunt_weak_channel_figure.tex",
}


def test_notation_guide_precedes_section_and_figure_s1():
    sources = list(tex_sources(MASTER))
    notation = SUPPLEMENT / "curated/si_notation.tex"
    assert sources.count(notation) == 1
    for fragment in ("si_01_exact.tex", "si_01_exact_figures.tex"):
        assert sources.index(notation) < sources.index(SUPPLEMENT / "curated" / fragment)

    text = expanded_tex(MASTER)
    notation_label = text.index(r"\label{tab:notation}")
    assert notation_label < text.index(r"\label{note:exact_rules}")
    assert notation_label < text.index(r"\label{fig:si_mechanistic_chain}")


def test_notation_label_is_unique_in_active_closure():
    labels = re.findall(r"\\label\s*\{\s*tab:notation\s*\}", expanded_tex(MASTER))
    assert len(labels) == 1


def test_retained_top_level_fragments_are_excluded_and_documented():
    active = set(tex_sources(MASTER))
    top_level = set(SUPPLEMENT.glob("*.tex"))
    assert len(INACTIVE_FRAGMENTS) == 19
    assert {path.name for path in top_level - active} == INACTIVE_FRAGMENTS
    assert top_level & active == {MASTER}

    readme = (SUPPLEMENT / "README.md").read_text(encoding="utf-8")
    inactive_section = readme.split("## Inactive top-level TeX fragments", 1)[1]
    documented = set(re.findall(r"(?m)^- `([^`]+\.tex)`$", inactive_section))
    assert documented == INACTIVE_FRAGMENTS
    for name in INACTIVE_FRAGMENTS:
        assert (SUPPLEMENT / name).is_file()
        assert (SUPPLEMENT / name) not in active


def test_documented_composition_inventory_matches_manifest():
    manifest = json.loads(
        (JOURNAL / "configs/supplement_consolidation/manifest.json").read_text()
    )
    assets = manifest["assets"]
    whole = [asset for asset in assets if asset.get("whole_source_sheet", False)]
    composed = [asset for asset in assets if not asset.get("whole_source_sheet", False)]
    assert len(assets) == 36
    assert len(whole) == 26
    assert len(composed) == 10
    single_source_reflows = {
        asset["figure"]
        for asset in composed
        if len({panel["source_asset"] for panel in asset["panels"]}) == 1
    }
    assert single_source_reflows == {"S5", "S26"}
    assert all(asset["paste_scale"] == 1.0 for asset in assets)
    assert all(panel["scale"] == 1.0 for asset in assets for panel in asset["panels"])
