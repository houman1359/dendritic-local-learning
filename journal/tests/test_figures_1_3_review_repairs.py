"""Publication figures retain scientific qualifiers and every displayed seed."""

import csv
import importlib.util
import io
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import pandas as pd
import pytest


JOURNAL = Path(__file__).resolve().parents[1]


def load_builder(relative_path):
    spec = importlib.util.spec_from_file_location(
        "review_repairs_" + Path(relative_path).stem,
        JOURNAL / "scripts" / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure1_delivery_footer_subscripts_clear_card_borders():
    framework = load_builder("credit_first_figures/build_framework.py")
    canvas = framework.NativeCanvas(
        472 / 72, 3, row_weights=[126, 126, 100],
        hgutter_pt=40, vgutter_pt=30,
        margins=framework.Margins(left=36, right=12, top=22, bottom=38))
    ax = canvas.panel("B", 0, 4, 8, schematic=True,
                      title="Three ways to spread one somatic error")
    framework.deliveries(ax)
    stream = io.BytesIO()
    canvas.fig.savefig(stream, format="pdf")
    plt.close(canvas.fig)
    with fitz.open(stream=stream.getvalue(), filetype="pdf") as document:
        page = document[0]
        cards = [drawing["rect"] for drawing in page.get_drawings()
                 if 120 < drawing["rect"].height < 130
                 and 80 < drawing["rect"].width < 110
                 and drawing["fill"] is not None]
        assert len(cards) == 3
        # review pass 2026-09-23: the card footers moved to the legend; each
        # card keeps one incoming somatic error, inside its own border
        footers = [block for block in page.get_text("blocks")
                   if "one s across neurons" in block[4]
                   or "spread evenly" in block[4]]
        assert not footers
        incoming_errors = [block for block in page.get_text("blocks")
                           if block[4].strip() == "δ0"]
        assert len(incoming_errors) == 3
        for x0, top, x1, bottom, *_ in incoming_errors:
            card = next(card for card in cards if card.x0 < (x0 + x1) / 2 < card.x1)
            assert bottom <= card.y1 - 1.0
    caption = (JOURNAL / "main.tex").read_text().split(
        r"\label{fig:framework}")[0].rsplit(r"\caption{", 1)[1]
    assert r"Strict scalar $s$ shared across neurons" in caption
    assert r"$\alpha_j$ are successive edge derivatives" in caption


def test_figure2_failed_acquisition_is_qualified_at_the_deranged_row():
    builder = load_builder("build_main_figure_04.py")
    canvas = builder.NativeCanvas(
        builder.HEIGHT_IN, 3, row_weights=builder.ROW_PT,
        hgutter_pt=builder.HGUTTER_PT, vgutter_pt=builder.VGUTTER_PT,
        margins=builder.MARGINS)
    ax = canvas.panel("E", 1, 8, 4, lock=False)
    folder = JOURNAL / "source_data/trained_subtree_address"
    builder.forgetting_forest(
        canvas, ax, pd.read_csv(folder / "condition_summary.csv"),
        pd.read_csv(folder / "seed_outcomes.csv"),
        pd.read_csv(folder / "paired_contrasts.csv"))
    qualifiers = [text for text in ax.texts if text.get_text() == "not learned"]
    assert len(qualifiers) == 1
    assert qualifiers[0].xy[1] == 2
    plt.close(canvas.fig)


def frozen_encoder_gaps():
    """Independently pair frozen rows by explicit IDs, without builder helpers."""
    values = {}
    for readout, folder in (("soft", "review_coefficient_encoder"),
                            ("hard", "review_coefficient_hard_readout")):
        with (JOURNAL / "source_data" / folder / "trajectories.csv").open() as stream:
            for row in csv.DictReader(stream):
                if int(row["epoch"]) != 80 or int(row["cue_delay_trials"]) != 0:
                    continue
                if row["method"] not in {"learned_local_cue", "oracle_context"}:
                    continue
                key = (readout, int(row["calibration_samples"]),
                       float(row["cue_noise_sd"]), int(row["seed"]), row["method"])
                assert key not in values
                values[key] = float(row["heldout_accuracy"])
    return {key[:-1]: 100 * (value - values[key[:-1] + ("oracle_context",)])
            for key, value in values.items() if key[-1] == "learned_local_cue"}


def check_forest_seed_records(rows, builder):
    assert len(rows) == 100
    expected = frozen_encoder_gaps()
    expected_keys = {(readout, size, noise, seed)
                     for readout, size, noise, _ in builder.GAP_ROWS
                     for seed in range(52000, 52020)}
    actual_keys = set()
    for row in rows:
        key = (row["readout"], int(row["calibration_samples"]),
               float(row["cue_noise_sd"]), int(row["seed"]))
        assert key not in actual_keys
        actual_keys.add(key)
        assert row["panel"] == "F" and row["record"] == "paired_seed"
        assert int(row["epoch"]) == 80 and int(row["cue_delay_trials"]) == 0
        assert row["control"] == "oracle_context"
        assert int(row["n_seeds"]) == 20
        assert float(row["accuracy_difference_pp"]) == pytest.approx(
            expected[key], abs=2e-11, rel=0)
    assert actual_keys == expected_keys
    # The one displayed hard/noise-free row is not a pooled n=60 sample.
    for size in (16, 64, 256):
        assert all(expected[("hard", size, 0.0, seed)] == 0.0
                   for seed in range(52000, 52020))


def test_figure3_forest_export_matches_frozen_seed_trajectories():
    builder = load_builder("credit_first_figures/build_ancestry.py")
    records = builder.gap_seed_records(builder.gap_seeds())
    check_forest_seed_records(records, builder)


def test_figure3_published_table_contains_every_drawn_forest_seed():
    builder = load_builder("credit_first_figures/build_ancestry.py")
    table = pd.read_csv(JOURNAL / "source_data/curated_publication/figure_03_plotted.csv")
    selected = table[table.panel.eq("F") & table.record.eq("paired_seed")]
    check_forest_seed_records(selected.to_dict("records"), builder)
    assert table[table.panel.eq("F")].record.value_counts().to_dict() == {
        "paired_seed": 100, "paired_contrast": 54, "grid_mean": 18}


def test_figure3_pairing_rejects_missing_oracle_seed(monkeypatch):
    builder = load_builder("credit_first_figures/build_ancestry.py")
    original_read = builder.pd.read_csv

    def missing_partner(path, *args, **kwargs):
        table = original_read(path, *args, **kwargs)
        drop = (table.epoch.eq(80) & table.cue_delay_trials.eq(0)
                & table.calibration_samples.eq(16) & table.cue_noise_sd.eq(0)
                & table.method.eq("oracle_context") & table.seed.eq(52000))
        return table.loc[~drop]

    monkeypatch.setattr(builder.pd, "read_csv", missing_partner)
    with pytest.raises(AssertionError):
        builder.gap_seeds()
