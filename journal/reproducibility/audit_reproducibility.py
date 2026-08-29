#!/usr/bin/env python3
"""Audit the journal reproducibility subpackage without accessing raw data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_origins() -> int:
    path = ROOT / "reproducibility" / "origin_manifest.tsv"
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise AssertionError("origin manifest is empty")
    exact = 0
    for row in rows:
        destination = ROOT / row["journal_copy"]
        if not destination.is_file():
            raise AssertionError(f"missing journal copy: {destination}")
        if row["journal_modification"].startswith("byte-identical"):
            observed = sha256(destination)
            if observed != row["origin_sha256"]:
                raise AssertionError(
                    f"byte-identical copy changed: {destination}: {observed}"
                )
            exact += 1
    return exact


def audit_feedback_rows() -> None:
    path = ROOT / "configs" / "regular_tree" / "reported_feedback_definition_15seed.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["network_type"], row["feedback"])].append(int(row["seed"]))
    expected_seeds = list(range(42, 57))
    if len(rows) != 60 or len(groups) != 4:
        raise AssertionError(f"unexpected feedback table shape: {len(rows)}, {len(groups)}")
    for group, seeds in groups.items():
        if sorted(seeds) != expected_seeds:
            raise AssertionError(f"unexpected seeds for {group}: {sorted(seeds)}")


def audit_targets() -> None:
    path = ROOT / "reproducibility" / "task_derived_primary_targets.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 7:
        raise AssertionError(f"expected seven targets, found {len(rows)}")
    if sum(int(row["n_connected_sites"]) for row in rows) != 69:
        raise AssertionError("connected-site total is not 69")
    if sum(int(row["n_manual_matches"]) for row in rows) != 27:
        raise AssertionError("manual-match total is not 27")
    if sum(int(row["n_partner_pairs"]) for row in rows) != 356:
        raise AssertionError("partner-pair total is not 356")
    for row in rows:
        observed = (
            int(row["n_trials"]),
            int(row["n_unique_stimuli"]),
            int(row["n_repeated_stimuli"]),
            float(row["response_finite_fraction"]),
        )
        if observed != (464, 280, 136, 1.0):
            raise AssertionError(f"unexpected target record: {row['target_root_id']}")
        reliability = float(row["median_repeat_reliability"])
        if not -1.0 <= reliability <= 1.0:
            raise AssertionError(f"invalid reliability: {row['target_root_id']}")


def audit_original_cohort_accounting() -> None:
    path = ROOT / "reproducibility" / "original_cohort_functional_accounting.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 8:
        raise AssertionError(f"expected eight original-cohort rows, found {len(rows)}")
    roots = [row["target_root_id"] for row in rows]
    if len(set(roots)) != 8:
        raise AssertionError("original-cohort roots are not unique")
    included = [row for row in rows if row["primary_functional_status"] == "included"]
    excluded = [row for row in rows if row["primary_functional_status"] == "excluded"]
    if len(included) != 7 or len(excluded) != 1:
        raise AssertionError("original-cohort accounting must contain seven included and one excluded target")
    designs = {row["structural_selection_design"] for row in rows}
    if designs != {
        "hand-authored layer-diverse pilot from an existing anatomy/functional-coregistration cohort"
    }:
        raise AssertionError(f"unexpected structural selection descriptions: {designs}")
    omitted = excluded[0]
    if omitted["target_root_id"] != "864691136043380566":
        raise AssertionError(f"unexpected excluded root: {omitted['target_root_id']}")
    if omitted["target_nucleus_id"] != "292648":
        raise AssertionError(f"unexpected excluded nucleus: {omitted['target_nucleus_id']}")
    if (omitted["trial_manifest_rows"], omitted["trial_manifest_rows_with_dandi_asset"]) != ("1", "0"):
        raise AssertionError("excluded target's archived DANDI accounting changed")
    primary_rows = list(
        csv.DictReader(
            (ROOT / "reproducibility" / "task_derived_primary_targets.csv").open(
                encoding="utf-8"
            )
        )
    )
    primary_roots = {row["target_root_id"] for row in primary_rows}
    if primary_roots != {row["target_root_id"] for row in included}:
        raise AssertionError("seven-target manifest does not match original-cohort accounting")

    # When the source draft is still adjacent, independently check the two
    # tables from which the accounting was derived. The journal package remains
    # auditable after detachment because this source-side check is optional.
    source = ROOT.parent.parent / "dendritic-credit-routing"
    cells_path = source / "data" / "microns_morphology" / "cell_manifest.csv"
    trials_path = (
        source
        / "imported"
        / "population"
        / "results"
        / "microns_dandi_trial_manifest"
        / "microns_dandi_trial_manifest.csv"
    )
    if cells_path.is_file() and trials_path.is_file():
        cells = list(csv.DictReader(cells_path.open(encoding="utf-8")))
        trials = list(csv.DictReader(trials_path.open(encoding="utf-8")))
        if {row["root_id"] for row in cells} != set(roots):
            raise AssertionError("accounting roots differ from the source morphology manifest")
        for row in rows:
            cell = next(item for item in cells if item["root_id"] == row["target_root_id"])
            if cell["source_root_id"] != row["source_root_id"]:
                raise AssertionError(
                    f"source root changed for {row['target_root_id']}: "
                    f"{cell['source_root_id']} != {row['source_root_id']}"
                )
            if int(cell["pilot_order"]) != int(row["pilot_order"]):
                raise AssertionError(f"pilot order changed for {row['target_root_id']}")
            source_trials = [
                item
                for item in trials
                if item["nucleus_id"] == row["target_nucleus_id"]
            ]
            with_asset = [
                item
                for item in source_trials
                if item["has_dandi_asset"].strip().lower() == "true"
            ]
            observed = (len(source_trials), len(with_asset))
            expected = (
                int(row["trial_manifest_rows"]),
                int(row["trial_manifest_rows_with_dandi_asset"]),
            )
            if observed != expected:
                raise AssertionError(
                    f"source trial accounting changed for {row['target_root_id']}: "
                    f"{observed} != {expected}"
                )


def audit_rerun_specs() -> None:
    expected = {
        "feedback_definition_shunting_15seed.yaml": (
            "dendritic_shunting",
            "analytical",
        ),
        "feedback_definition_additive_15seed.yaml": (
            "dendritic_additive",
            "occupancy_quantile",
        ),
    }
    for filename, (core, policy) in expected.items():
        path = ROOT / "configs" / "reruns" / filename
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = payload["base_config"]
        if payload["experiment_settings"] != {
            "seeds_per_condition": 15,
            "base_seed": 42,
        }:
            raise AssertionError(f"unexpected seeds in {filename}")
        modes = payload["sweep_config"][
            "training.main.learning_strategy_config.error_broadcast_mode"
        ]
        if modes != ["per_soma", "per_soma_shared"]:
            raise AssertionError(f"unexpected feedback modes in {filename}")
        observed = (
            base["model"]["core"]["type"],
            base["model"]["core"]["reactivation"]["init_policy"],
        )
        if observed != (core, policy):
            raise AssertionError(f"unexpected core/policy in {filename}: {observed}")
        if payload["sweep_contract"]["expected_config_count"] != 30:
            raise AssertionError(f"unexpected sweep count in {filename}")


# Frozen execution records of completed cluster runs.  These CIFAR-10 launch
# specifications document exactly what ran (the confirmatory one is byte-pinned
# by analyze_cifar10_additive_feedback_ladder_confirmatory.py's
# EXPECTED_INPUT_YAML_SHA256); rewriting their site-specific paths would break
# the freeze chain or falsify the execution record, so they are excluded from
# the private-path release rule and are not shipped as re-runnable recipes.
FROZEN_EXECUTION_RECORDS = {
    "configs/cifar10_additive_feedback_ladder_confirmatory.yaml",
    "configs/cifar10_additive_operator_compatibility.yaml",
    "configs/cifar10_bp_recipe_init_screen.yaml",
    "configs/cifar10_credit_ladder_pilot.yaml",
    "configs/cifar10_historical_bp_reproduction.yaml",
}


def audit_private_paths() -> None:
    forbidden = (
        re.compile(r"/n/(?:home[^/]*|holylabs)/"),
        re.compile(r"/home/[A-Za-z0-9._-]+/"),
        re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    )
    suffixes = {".py", ".json", ".yaml", ".yml", ".md", ".tsv", ".csv"}
    hits: list[str] = []
    for directory in ("code", "configs", "reproducibility"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if str(path.relative_to(ROOT)) in FROZEN_EXECUTION_RECORDS:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in forbidden):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise AssertionError(f"private absolute paths remain: {hits}")


def main() -> None:
    exact_copies = audit_origins()
    audit_feedback_rows()
    audit_targets()
    audit_original_cohort_accounting()
    audit_rerun_specs()
    audit_private_paths()
    config = json.loads(
        (ROOT / "configs" / "task_derived" / "primary_four_channel.json").read_text(
            encoding="utf-8"
        )
    )
    if config["feedback"]["requested_channels"] != 4:
        raise AssertionError("primary task-derived budget is not four channels")
    print(
        "reproducibility audit passed; "
        f"{exact_copies} byte-identical source copies verified, "
        "60 feedback rows, 7 task targets and the 8-cell cohort accounting validated"
    )


if __name__ == "__main__":
    main()
