from __future__ import annotations

import csv
import importlib.util
import json
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


def panel_letters(value: str) -> set[str]:
    """Normalize combined current associations such as ``b-d, f``."""
    result = set()
    for first, last in re.findall(r"\b([a-h])(?:\s*[-–]+\s*([a-h]))?\b", value.lower()):
        result.update(chr(k) for k in range(ord(first), ord(last or first) + 1))
    return result


def curated_assets() -> list[dict]:
    return json.loads(
        (JOURNAL / "configs/supplement_consolidation/manifest.json").read_text()
    )["assets"]


def test_source_data_inventory_matches_final_display_numbering() -> None:
    files = list(builder.FILES)
    figures = {item.figure for item in files}
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
    assert main_numbers == set(builder.MAIN_FIGURE_NUMBERS)
    assert supplementary_numbers == set(range(1, len(builder.SUPPLEMENTARY_FIGURES) + 1))
    selector = next(a for a in curated_assets() if a["id"] == "original_selector")
    assert selector["figure"] == "S34"
    prospective = [item for item in files if item.figure == "Supplementary Figure 34"]
    assert prospective
    assert all(item.source.startswith("source_data/prospective_morphology_selection/") for item in prospective)
    assert any(item.source.endswith("/sealed_confirmatory_selections.csv") for item in prospective)
    assert any(item.source.endswith("/candidate_outcomes.csv") for item in prospective)
    interaction = [item for item in files if item.figure == "Figure 4"]
    assert any("credit_rule_bridge/" in item.source for item in interaction)
    assert any(
        "g" in panel_letters(item.panels)
        and item.source.endswith("credit_rule_extension/summaries/all_diagnostics.csv")
        for item in interaction
    )
    assert any("credit_resolution_bridge/" in item.source for item in files if item.figure == "Methods")
    anatomy = [item for item in files if item.figure == "Figure 8"]
    assert any("anatomy_commonmode/" in item.source for item in anatomy)
    assert not any("morphology_calibration" in item.source for item in anatomy)
    assert any("release_task_identity/task_identity.json" in item.source for item in files)


def test_source_data_destinations_are_unique_and_sources_exist() -> None:
    files = list(builder.FILES)
    destinations = [item.destination for item in files]
    assert len(destinations) == len(set(destinations))
    assert not [item.source for item in files if not (JOURNAL / item.source).is_file()]


def test_every_curated_panel_releases_its_declared_numerical_sources() -> None:
    files = list(builder.FILES)
    assets = curated_assets()
    assert len(assets) == len(builder.SUPPLEMENTARY_FIGURES) == 36
    seen = set()
    for asset in assets:
        figure = "Supplementary Figure " + asset["figure"].removeprefix("S")
        for panel in asset["panels"]:
            for declared in panel["numerical_source_paths"]:
                assert "source_data/" in declared, (asset["id"], declared)
                source = "source_data/" + declared.split("source_data/", 1)[1]
                matches = [item for item in files if item.figure == figure and item.source == source]
                assert matches, (asset["id"], panel["panel"], source)
                assert any(panel["panel"].lower() in panel_letters(item.panels) for item in matches), (
                    asset["id"], panel["panel"], source, [item.panels for item in matches]
                )
                seen.add((figure, panel["panel"], source))
    assert seen


def test_curated_table_destinations_release_complete_hash_pinned_sources() -> None:
    """Removing a printed endpoint table must not remove its promised records."""
    curation = json.loads(
        (JOURNAL / "configs/supplement_consolidation/publication_curation_map.json").read_text()
    )
    pinned = json.loads(
        (JOURNAL / "configs/figure_structure/table_source_hashes.json").read_text()
    )["source_sha256"]
    declared = {}
    for table in curation["tables"]:
        for destination in table["destinations"]:
            source, expected = destination["path"], destination["sha256"]
            assert declared.setdefault(source, expected) == expected, source
    added = {table["source"] for table in curation["new_tables"] if "source" in table}
    assert declared and set(pinned) == set(declared) | added
    assert all(pinned[source] == expected for source, expected in declared.items())

    complete = {
        item.source for item in builder.FILES
        if item.destination not in builder.DESTINATION_ROW_FILTERS
    }
    sys.path.insert(0, str(JOURNAL / "code/release_noise"))
    from release_hashes import verify_released_file

    for source, expected in pinned.items():
        assert source in complete, source
        # A portable archive may translate a historical path in a protocol.
        # Authenticate that declared transformation against the original pin;
        # a display-filtered copy can never replace the complete source.
        verdict = verify_released_file(JOURNAL / source, expected)
        assert verdict["verified"], (source, verdict)


def test_original_image_evidence_is_retained_without_obsolete_display_claims() -> None:
    files = list(builder.FILES)
    retained = [item for item in files if item.destination.startswith("Methods/retained_evidence/")]
    for source in (
        "source_data/figure2/feedback_gradient_runs.csv",
        "source_data/figure2/path_gain_dispersion_ladder_runs.csv",
        "source_data/figure2/path_gain_cv_runs.csv",
        "source_data/mnist_feedback_ladder/paired_contrasts.csv",
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "source_data/prospective_input_validity/routing_valid_paired_contrasts.csv",
    ):
        matches = [item for item in retained if item.source == source]
        assert matches, source
        assert all((item.figure, item.panels) == ("Methods", "supporting evidence") for item in matches)
    central = [item for item in retained if item.source.startswith("source_data/prospective_input_validity/central_valid_")]
    assert len(central) == 3
    fashion = [item for item in retained if item.source.startswith("source_data/fashion_feedback_ladder/")]
    assert len(fashion) == 4
    # Displayed image panels now use the semantic consolidated manifest.
    image_geometry = next(a for a in curated_assets() if a["id"] == "mnist_dictionary_geometry")
    assert image_geometry["figure"] == "S7"
    # The feedback-gradient panels left the image sheet for their own
    # error-field sheet; the release follows the panel, not the old number.
    error_field = next(a for a in curated_assets() if a["id"] == "error_field_geometry")
    assert error_field["figure"] == "S8"
    for source in (
        "source_data/figure2/feedback_gradient_runs.csv",
        "source_data/figure2/path_gain_dispersion_ladder_runs.csv",
    ):
        assert any(item.figure == "Supplementary Figure 8" and item.source == source for item in files)


def test_curated_image_provenance_preserves_original_source_lineage() -> None:
    with (JOURNAL / "source_data/provenance_manifest.tsv").open(newline="", encoding="utf-8") as handle:
        rows = {row["entry_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    transport = rows["fig2.path_gain_dispersion"]
    assert transport["source_path"].endswith("/source_data/figure2/path_gain_dispersion_ladder_runs.csv")
    assert transport["sha256"] == "fecb52eb239fd9584899ea4c280eac1f6ec07716d09cadfe1eed88120b135d14"
    for entry_id in (
        "fig2.path_gain_dispersion", "fig2.path_gain", "fig2.c", "mnist.ladder.contrasts",
        "prospective.routing.runs", "prospective.routing.contrasts", "fashion.asset",
        "fashion.outcomes", "fashion.conditions", "fashion.contrasts", "fashion.audit",
    ):
        assert (rows[entry_id]["figure"], rows[entry_id]["panel"]) == (
            "methods", "retained scientific evidence",
        )
    assert rows["fashion.asset"]["source_path"].endswith("/figures/supplementary/figure_S04_panels_A-E.pdf")
    # Retaining historical IDs does not substitute for explicit current associations.
    for source_suffix in (
        "/source_data/figure2/path_gain_dispersion_ladder_runs.csv",
        "/source_data/figure2/feedback_gradient_runs.csv",
    ):
        assert any(row["figure"] == "figS8" and row["source_path"].endswith(source_suffix)
                   for row in rows.values())


def test_figure_2_ownership_release_is_the_mnist_d2_d4_subset(
    tmp_path: Path,
) -> None:
    expected = (
        (
            "Methods/retained_evidence/Figure_2/Fig2g_ownership_run_outcomes.csv",
            80,
            {"family": {"routing"}, "task": {"mnist"}, "depth": {"2", "4"}},
        ),
        (
            "Methods/retained_evidence/Figure_2/Fig2g_ownership_contrasts.csv",
            4,
            {"task": {"mnist"}, "depth": {"2", "4"}},
        ),
        (
            "Methods/retained_evidence/Supplementary_Figure_19/SuppFig19d_ownership_run_outcomes.csv",
            120,
            {"family": {"routing"}},
        ),
        (
            "Methods/retained_evidence/Supplementary_Figure_19/SuppFig19d_ownership_contrasts.csv",
            6,
            {},
        ),
        (
            "Methods/retained_evidence/Supplementary_Figure_8/SuppFig8a-i_seed_outcomes.csv",
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
        if destination.startswith("Methods/retained_evidence/Figure_2/"):
            assert {row["core"] for row in rows} == {
                "dendritic_additive",
                "dendritic_shunting",
            }



def test_filtered_subsets_also_release_complete_original_source_tables(tmp_path: Path) -> None:
    files = list(builder.FILES)
    filtered_sources = {item.source for item in files if item.destination in builder.DESTINATION_ROW_FILTERS}
    assert filtered_sources == {
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "source_data/prospective_input_validity/routing_valid_paired_contrasts.csv",
    }
    for index, source in enumerate(sorted(filtered_sources)):
        complete = [item for item in files if item.source == source
                    and item.destination not in builder.DESTINATION_ROW_FILTERS]
        assert complete, source
        output = tmp_path / f"complete_{index}.csv"
        builder.copy_source_file(complete[0], JOURNAL / source, output)
        with (JOURNAL / source).open(newline="") as handle:
            original = list(csv.DictReader(handle))
        with output.open(newline="") as handle:
            released = list(csv.DictReader(handle))
        assert len(released) == len(original)
        drop = builder.SANITIZED_DROP_COLUMNS.get(source, set())
        assert released == [{key: value for key, value in row.items() if key not in drop}
                            for row in original]
        if not drop:
            assert output.read_bytes() == (JOURNAL / source).read_bytes()


def test_current_readmes_follow_manifest_and_archive_keeps_original_hashes(tmp_path: Path) -> None:
    current = next(item for item in builder.FILES if item.figure == "Figure 8")
    directory = Path(current.destination).parts[0]
    (tmp_path / directory).mkdir()
    builder.write_display_readmes(tmp_path, [{
        "file": current.destination, "figure": current.figure,
        "panels": current.panels, "independent_unit": current.independent_unit,
        "status": current.status,
    }])
    text = (tmp_path / directory / "README.md").read_text()
    assert "Figure 8 source data" in text
    assert current.panels in text
    assert f"{len(builder.MAIN_FIGURE_NUMBERS)} main and {len(builder.SUPPLEMENTARY_FIGURES)} supplementary figures" in builder.README
    assert "initialization selector" not in text


def test_path_portability_preserves_numeric_values(tmp_path: Path) -> None:
    import json
    path = tmp_path / "lineage.json"
    source = {"loss": 0.123456789012345, "seed": 17,
              "source": "/".join(("", "n", "home13", "example", "original", "run.json"))}
    path.write_text(json.dumps(source))
    changes = builder.portable_text_copy(path)
    released = json.loads(path.read_text())
    assert released["loss"] == source["loss"] and released["seed"] == source["seed"]
    assert released["source"].endswith("example/original/run.json")
    assert changes and ("/n/" + "home13/") not in path.read_text()


def test_complete_anatomy_dictionary_evaluations_are_included() -> None:
    import gzip
    complete = sorted((JOURNAL / "source_data/anatomy_commonmode").rglob("rows_*.csv.gz"))
    assert len(complete) == 65
    released = {item.source for item in builder.FILES}
    assert all(str(path.relative_to(JOURNAL)) in released for path in complete)
    rows = 0
    for path in complete:
        with gzip.open(path, "rt") as handle:
            rows += sum(1 for _ in csv.DictReader(handle))
    assert rows == 149331
