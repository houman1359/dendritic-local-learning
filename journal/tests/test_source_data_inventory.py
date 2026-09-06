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
    assert main_numbers == set(range(1, 9))  # Figure 1 includes trained dictionary statistics.
    assert supplementary_numbers == set(range(1, 48))
    prospective = [item for item in builder.FILES if item.figure == "Supplementary Figure 35"]
    assert prospective
    assert all(item.source.startswith("source_data/prospective_morphology_selection/") for item in prospective)
    assert any(item.source.endswith("/sealed_confirmatory_selections.csv") for item in prospective)
    assert any(item.source.endswith("/candidate_outcomes.csv") for item in prospective)
    interaction = [item for item in builder.FILES if item.figure == "Figure 4"]
    assert any("credit_rule_bridge/" in item.source for item in interaction)
    assert any("credit_resolution_bridge/" in item.source for item in interaction)
    anatomy = [item for item in builder.FILES if item.figure == "Figure 6"]
    assert any("anatomy_commonmode/" in item.source for item in anatomy)
    assert not any("morphology_calibration" in item.source for item in anatomy)
    assert any("release_task_identity/task_identity.json" in item.source for item in builder.FILES)



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


def test_s45_retains_original_image_panel_sources() -> None:
    by_source = {item.source: item for item in builder.FILES if item.figure.startswith("Supplementary")}

    gradient = by_source["source_data/figure2/feedback_gradient_runs.csv"]
    assert (gradient.figure, gradient.panels, gradient.destination) == (
        "Supplementary Figure 45",
        "c",
        "Supplementary_Figure_45/SuppFig45c_feedback_gradient_runs.csv",
    )

    transport = by_source[
        "source_data/figure2/path_gain_dispersion_ladder_runs.csv"
    ]
    assert (transport.figure, transport.panels, transport.destination) == (
        "Supplementary Figure 45",
        "d-e",
        "Supplementary_Figure_45/SuppFig45d-e_path_gain_dispersion_ladder_runs.csv",
    )

    conductance_only = by_source["source_data/figure2/path_gain_cv_runs.csv"]
    assert (conductance_only.figure, conductance_only.panels) == (
        "Supplementary Figure 1",
        "a",
    )

    mnist_contrasts = by_source[
        "source_data/mnist_feedback_ladder/paired_contrasts.csv"
    ]
    assert (mnist_contrasts.figure, mnist_contrasts.panels) == ("Supplementary Figure 45", "g")

    for source in (
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "source_data/prospective_input_validity/routing_valid_paired_contrasts.csv",
    ):
        assert any(
            item.source == source
            and (item.figure, item.panels) == ("Supplementary Figure 45", "g")
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
        (item.figure, item.panels) == ("Supplementary Figure 19", "text")
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
        item.destination == "Supplementary_Figure_45/SuppFig45g_seed_outcomes.csv"
        for item in builder.FILES
    )


def test_s45_image_and_s4_provenance_preserve_source_lineage() -> None:
    manifest = JOURNAL / "source_data" / "provenance_manifest.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["entry_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }

    transport = rows["fig2.path_gain_dispersion"]
    assert (transport["figure"], transport["panel"]) == ("figS45", "d-e")
    assert transport["source_path"].endswith(
        "/source_data/figure2/path_gain_dispersion_ladder_runs.csv"
    )
    assert transport["sha256"] == (
        "fecb52eb239fd9584899ea4c280eac1f6ec07716d09cadfe1eed88120b135d14"
    )

    old_path_gain = rows["fig2.path_gain"]
    assert (old_path_gain["figure"], old_path_gain["panel"]) == ("figS1", "a")
    assert (rows["fig2.c"]["figure"], rows["fig2.c"]["panel"]) == (
        "figS45",
        "c",
    )
    assert (
        rows["mnist.ladder.contrasts"]["figure"],
        rows["mnist.ladder.contrasts"]["panel"],
    ) == ("figS45", "g")
    for entry_id in ("prospective.routing.runs", "prospective.routing.contrasts"):
        assert (rows[entry_id]["figure"], rows[entry_id]["panel"]) == (
            "figS45/figS19",
            "g/text",
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
            "Supplementary_Figure_45/SuppFig45g_ownership_run_outcomes.csv",
            80,
            {"family": {"routing"}, "task": {"mnist"}, "depth": {"2", "4"}},
        ),
        (
            "Supplementary_Figure_45/SuppFig45g_ownership_contrasts.csv",
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
        if destination.startswith("Supplementary_Figure_45/"):
            assert {row["core"] for row in rows} == {
                "dendritic_additive",
                "dendritic_shunting",
            }


def test_current_readmes_follow_manifest_and_archive_keeps_original_hashes(tmp_path: Path) -> None:
    current = next(item for item in builder.FILES if item.figure == "Figure 6")
    directory = Path(current.destination).parts[0]
    (tmp_path / directory).mkdir()
    builder.write_display_readmes(tmp_path, [{
        "file": current.destination, "figure": current.figure,
        "panels": current.panels, "independent_unit": current.independent_unit,
        "status": current.status,
    }])
    text = (tmp_path / directory / "README.md").read_text()
    assert "Figure 6 source data" in text
    assert current.panels in text
    assert "eight main and 47 supplementary figures" in builder.README
    assert "initialization selector" not in text


def test_path_portability_preserves_numeric_values(tmp_path: Path) -> None:
    import json
    path = tmp_path / "lineage.json"
    source = {"loss": 0.123456789012345, "seed": 17,
              "source": "/n/home13/example/original/run.json"}
    path.write_text(json.dumps(source))
    changes = builder.portable_text_copy(path)
    released = json.loads(path.read_text())
    assert released["loss"] == source["loss"] and released["seed"] == source["seed"]
    assert released["source"].endswith("example/original/run.json")
    assert changes and "/n/home13/" not in path.read_text()
