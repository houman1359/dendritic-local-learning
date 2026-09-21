"""Current anatomy/measured figures preserve coordinates and biological IDs."""

import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


JOURNAL = Path(__file__).resolve().parents[1]
SOURCES = JOURNAL / "source_data"
BUILDERS = JOURNAL / "scripts/credit_first_figures"
sys.path.insert(0, str(BUILDERS))
from source_data_export import concat_exact_id_tables, exact_id_table


def load_builder(name):
    spec = importlib.util.spec_from_file_location(
        "review_" + name, BUILDERS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_figure(number):
    return pd.read_csv(SOURCES / f"curated_publication/figure_{number:02d}_plotted.csv",
                       dtype={"root_id": "string", "target_root_id": "string",
                              "pre_pt_root_id": "string"})


def test_exact_ids_survive_mixed_rows_and_csv_roundtrip():
    roots = [648518346349538440, 648518346349538466, 864691135409937097]
    cells = exact_id_table([{"root_id": root, "value": i}
                            for i, root in enumerate(roots)])
    combined = concat_exact_id_tables([pd.DataFrame([{"mean": 1.0}]), cells])
    result = pd.read_csv(io.StringIO(combined.to_csv(index=False)),
                         dtype={"root_id": "string"})
    assert result.root_id.dropna().tolist() == list(map(str, roots))


def test_floating_point_identifiers_are_rejected_before_export():
    with pytest.raises(ValueError, match="exact integer"):
        exact_id_table([{"root_id": float(864691135409937097)}])


def test_figure8_field_is_unweighted_before_weighted_capture():
    anatomy = load_builder("build_anatomy")
    raw = np.array([[-2.0], [-1.0]])
    arb = {"i_sites": [anatomy.ROUTE_SITE],
           "npz": {"response": raw, "weighted_response": raw * [[1.0], [8.0]]},
           "blocks": {"first": [0], "second": [1]},
           "block_keys": ["first", "second"]}
    field, scale = anatomy.block_field(arb)
    np.testing.assert_array_equal(field, [-1.0, -0.5])
    assert scale == 2.0


def test_figure8_all_eight_pinky_cells_join_without_identifier_collisions():
    published = read_figure(8)
    cells = published[published.panel.eq("H") & published.cohort.eq("pinky")
                      & published.cell_residual_capture.notna()]
    reference = pd.read_csv(SOURCES / "anatomy_commonmode/pinky/cell_method_summary.csv",
                            dtype={"root_id": "string"})
    reference = reference[reference.channels.eq(8)]
    assert cells.root_id.nunique() == 8
    assert cells.groupby("method").root_id.nunique().eq(8).all()
    joined = cells.merge(reference, on=["root_id", "method"], how="left",
                         validate="one_to_one", suffixes=("_drawn", "_source"))
    assert len(joined) == 40
    np.testing.assert_allclose(joined.cell_residual_capture,
                               joined.residual_capture_source, rtol=0, atol=1e-14)


def test_figure8_export_distinguishes_dictionary_and_raw_field_coordinates():
    table = read_figure(8)
    b = table[table.panel.eq("B")]
    assert b.record.value_counts().to_dict() == {
        "dictionary_block": 8, "selected_route": 7}
    c = table[table.panel.eq("C")].sort_values("block")
    assert len(c) == 8 and c.record.eq("modeled_field_block").all()
    anatomy = load_builder("build_anatomy")
    expected, scale = anatomy.block_field(anatomy.arbor_routes())
    np.testing.assert_allclose(c.field_normalized, expected, atol=1e-14)
    np.testing.assert_allclose(c.field_t, expected * scale, atol=1e-14)
    np.testing.assert_allclose(c.field_scale, scale, atol=1e-14)


def test_figure8_microns_ids_join_and_preserve_paired_cell_differences():
    table = read_figure(8)
    reference = pd.read_csv(SOURCES / "anatomy_commonmode/v661/cell_method_summary.csv",
                            dtype={"root_id": "string"})
    reference = reference[reference.channels.eq(8)].pivot(
        index="root_id", columns="method", values="residual_capture")
    fan = table[table.panel.eq("G") & table.cell_difference_pp.notna()]
    assert len(fan) == 188 and fan.root_id.nunique() == 47
    for control, group in fan.groupby("control"):
        expected = 100 * (reference.loc[group.root_id, "common + ancestry"]
                          - reference.loc[group.root_id, control])
        np.testing.assert_allclose(group.cell_difference_pp, expected, atol=1e-12)


def test_figure9_exact_microns_ids_identify_every_per_cell_mark():
    table = read_figure(9)
    representative = table[table.panel.eq("B")].root_id.unique().tolist()
    assert representative == ["864691135409937097"]
    cells = table[table.record.eq("cell_value")]
    assert cells.root_id.notna().all()
    assert cells.root_id.str.fullmatch(r"\d+").all()
    expected = {"C": 8, "D": 8, "E": 8, "F": 53, "G": 53, "H": 8}
    assert cells.groupby("panel").root_id.nunique().to_dict() == expected
    original = pd.read_csv(SOURCES / "figure3/segment_metrics.csv",
                           dtype={"root_id": "string"}).root_id.unique()
    assert set(cells[cells.panel.eq("C")].root_id) == set(original)
    signed = pd.read_csv(SOURCES / "shunt_ancestry_gain/signed_calibration/signed_cell_effects.csv",
                         dtype={"root_id": "string"})
    keys = ["cohort", "regime", "perturbation", "category", "root_id"]
    joined = cells[cells.panel.eq("F")].merge(signed, on=keys, validate="one_to_one")
    assert len(joined) == 424
    np.testing.assert_allclose(joined.value, joined.signed_log_change, atol=1e-14)


def test_figure10_source_data_uses_current_panels_and_keeps_nested_units():
    table = read_figure(10)
    assert set(table.panel) == {"B", "C", "D", "D-inset", "E", "F"}
    b = table[table.panel.eq("B")]
    assert b.record.value_counts().to_dict() == {
        "scan_value": 13, "target_value": 7, "target_mean": 1}
    c = table[table.panel.eq("C")]
    assert c.record.value_counts().to_dict() == {"target_value": 28, "target_mean": 4}
    assert c[c.record.eq("target_value")].groupby("endpoint").target_root_id.nunique().eq(7).all()
    d = table[table.panel.eq("D")]
    assert d.record.value_counts().to_dict() == {
        "detection_probability": 42, "observed_estimate": 1}
    inset = table[table.panel.eq("D-inset")]
    assert len(inset) == 125 and inset.pre_pt_root_id.nunique() == 102
    assert inset.drawn.eq(True).sum() == 123
    e = table[table.panel.eq("E")]
    matrix = e.pivot(index="input_row", columns="route_column", values="value")
    actual = np.load(SOURCES / "credit_first_figures/figure_08_actual_support.npz")["matrix"]
    np.testing.assert_array_equal(matrix, actual)
    f = table[table.panel.eq("F")]
    assert len(f) == 13 and f.target_root_id.nunique() == 7
    assert f.one_site_routes.eq(True).sum() == 6
    assert f.coverage.mean() == pytest.approx(0.47972467452105455)
