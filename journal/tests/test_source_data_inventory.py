from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPTS = JOURNAL / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_nature_source_data", SCRIPTS / "build_nature_source_data.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_source_data_inventory_matches_final_display_numbering() -> None:
    figures = {item.figure for item in builder.FILES}
    main_numbers = {
        int(match.group(1))
        for figure in figures
        if (match := re.fullmatch(r"Figure (\d+)", figure))
    }
    supplementary_numbers = {
        int(match.group(1))
        for figure in figures
        if (match := re.fullmatch(r"Supplementary Figure (\d+)", figure))
    }
    assert main_numbers == set(range(2, 10))  # Figure 1 is conceptual.
    assert supplementary_numbers == set(range(1, 30))


def test_source_data_destinations_are_unique_and_sources_exist() -> None:
    destinations = [item.destination for item in builder.FILES]
    assert len(destinations) == len(set(destinations))
    assert not [item.source for item in builder.FILES if not (JOURNAL / item.source).is_file()]


def test_supplementary_figure_4_uses_only_current_panel_mapping() -> None:
    items = [
        item for item in builder.FILES if item.figure == "Supplementary Figure 4"
    ]
    assert {item.panels for item in items} == {"a", "b", "c", "d", "e"}
    assert all(
        "SuppFig4a_depth_scaling" in item.destination
        for item in items
        if item.panels == "a"
    )
    assert all(
        "SuppFig4b_broadcast_noise" in item.destination
        for item in items
        if item.panels == "b"
    )
    assert all(
        "SuppFig4c_cifar10_control" in item.destination
        for item in items
        if item.panels == "c"
    )
    assert all(
        "raw_additive" in item.destination for item in items if item.panels == "d"
    )
    assert all(
        item.source.startswith("source_data/fashion_feedback_ladder/")
        and "SuppFig4e_Fashion_MNIST" in item.destination
        for item in items
        if item.panels == "e"
    )


def test_figure_2_redesign_uses_current_panel_sources() -> None:
    by_source = {item.source: item for item in builder.FILES}

    gradient = by_source["source_data/figure2/feedback_gradient_runs.csv"]
    assert (gradient.figure, gradient.panels, gradient.destination) == (
        "Figure 2",
        "c",
        "Figure_2/Fig2c_feedback_gradient_runs.csv",
    )

    transport = by_source[
        "source_data/figure2/path_gain_dispersion_ladder_runs.csv"
    ]
    assert (transport.figure, transport.panels, transport.destination) == (
        "Figure 2",
        "d-e",
        "Figure_2/Fig2d-e_path_gain_dispersion_ladder_runs.csv",
    )

    conductance_only = by_source["source_data/figure2/path_gain_cv_runs.csv"]
    assert (conductance_only.figure, conductance_only.panels) == (
        "Supplementary Figure 1",
        "a",
    )

    mnist_contrasts = by_source[
        "source_data/mnist_feedback_ladder/paired_contrasts.csv"
    ]
    assert (mnist_contrasts.figure, mnist_contrasts.panels) == ("Figure 2", "g")

    for source in (
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "source_data/prospective_input_validity/routing_valid_paired_contrasts.csv",
    ):
        assert any(
            item.source == source
            and (item.figure, item.panels) == ("Figure 2", "g")
            for item in builder.FILES
        )

    central = [
        item
        for item in builder.FILES
        if item.source.startswith(
            "source_data/prospective_input_validity/central_valid_"
        )
    ]
    assert len(central) == 3
    assert all(
        (item.figure, item.panels) == ("Supplementary Figure 19", "a-c")
        and item.destination.startswith(
            "Supplementary_Figure_19/SuppFig19a-c_"
        )
        for item in central
    )

    fashion = [
        item
        for item in builder.FILES
        if item.source.startswith("source_data/fashion_feedback_ladder/")
    ]
    assert len(fashion) == 4
    assert all(
        (item.figure, item.panels) == ("Supplementary Figure 4", "e")
        for item in fashion
    )
    assert not any(
        item.destination == "Figure_2/Fig2g_seed_outcomes.csv"
        for item in builder.FILES
    )


def test_figure_2_and_s4_provenance_matches_redesign() -> None:
    manifest = JOURNAL / "source_data" / "provenance_manifest.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["entry_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }

    transport = rows["fig2.path_gain_dispersion"]
    assert (transport["figure"], transport["panel"]) == ("fig2", "d-e")
    assert transport["source_path"].endswith(
        "/source_data/figure2/path_gain_dispersion_ladder_runs.csv"
    )
    assert transport["sha256"] == (
        "fecb52eb239fd9584899ea4c280eac1f6ec07716d09cadfe1eed88120b135d14"
    )

    old_path_gain = rows["fig2.path_gain"]
    assert (old_path_gain["figure"], old_path_gain["panel"]) == ("figS1", "a")
    assert (rows["fig2.c"]["figure"], rows["fig2.c"]["panel"]) == (
        "fig2",
        "c",
    )
    assert (
        rows["mnist.ladder.contrasts"]["figure"],
        rows["mnist.ladder.contrasts"]["panel"],
    ) == ("fig2", "g")
    for entry_id in ("prospective.routing.runs", "prospective.routing.contrasts"):
        assert (rows[entry_id]["figure"], rows[entry_id]["panel"]) == (
            "fig2/figS19",
            "g/d",
        )

    for entry_id in (
        "fashion.asset",
        "fashion.outcomes",
        "fashion.conditions",
        "fashion.contrasts",
        "fashion.audit",
    ):
        assert (rows[entry_id]["figure"], rows[entry_id]["panel"]) == (
            "figS4",
            "e",
        )
    assert rows["fashion.asset"]["source_path"].endswith(
        "/figures/supplementary/figure_S04_panels_A-E.pdf"
    )


def test_figure_2_ownership_release_is_the_mnist_d2_d4_subset(
    tmp_path: Path,
) -> None:
    expected = (
        (
            "Figure_2/Fig2g_ownership_run_outcomes.csv",
            80,
            {"family": {"routing"}, "task": {"mnist"}, "depth": {"2", "4"}},
        ),
        (
            "Figure_2/Fig2g_ownership_contrasts.csv",
            4,
            {"task": {"mnist"}, "depth": {"2", "4"}},
        ),
        (
            "Supplementary_Figure_19/SuppFig19d_ownership_run_outcomes.csv",
            120,
            {"family": {"routing"}},
        ),
        (
            "Supplementary_Figure_19/SuppFig19d_ownership_contrasts.csv",
            6,
            {},
        ),
        (
            "Supplementary_Figure_8/SuppFig8a-i_seed_outcomes.csv",
            160,
            {"family": {"fixed_budget"}},
        ),
    )
    for destination, expected_rows, expected_values in expected:
        item = next(item for item in builder.FILES if item.destination == destination)
        output = tmp_path / Path(destination).name
        builder.copy_source_file(item, JOURNAL / item.source, output)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_rows
        for field, values in expected_values.items():
            assert {row[field] for row in rows} == values
        if destination.startswith("Figure_2/"):
            assert {row["core"] for row in rows} == {
                "dendritic_additive",
                "dendritic_shunting",
            }


def test_figure_2_source_data_readmes_describe_current_a_to_g() -> None:
    transport = next(
        item
        for item in builder.FILES
        if item.destination
        == "Figure_2/Fig2d-e_path_gain_dispersion_ladder_runs.csv"
    )
    text = "\n".join((builder.README, builder.FIGURE_2_README, transport.role, transport.notes))
    assert "pre-reactivation voltage-error" in text
    assert "Panels A and F are schematics" in builder.FIGURE_2_README
    assert "Supplementary Figure 19A--C" in builder.README
    assert "Figure 2 contains the 480" not in builder.README
    for stale in ("post-activation", "0.3%", "18--26", "18-26"):
        assert stale not in text
