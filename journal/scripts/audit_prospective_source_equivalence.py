#!/usr/bin/env python3
"""Verify behavior-neutral source changes after prospective sweep freezing.

The prospective sweeps record hashes for every source file that can affect the
model or update rule.  This audit is deliberately narrow: it accepts only the
four exact post-freeze edits listed below, and only when the frozen experiment
configurations cannot exercise their new behavior.  Every other source change
remains a hard failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = JOURNAL_ROOT.parents[2]
RUN_ROOT = JOURNAL_ROOT / "prospective_runs"

FACTORY_PATH = "repo:src/dendritic_modeling/networks/architectures/factory.py"
INDEXED_PATH = (
    "repo:src/dendritic_modeling/networks/architectures/"
    "excitation_inhibition/synapse/indexed_sparse.py"
)
SPATIAL_PATH = (
    "repo:src/dendritic_modeling/networks/architectures/"
    "excitation_inhibition/synapse/spatial_morphology.py"
)
CONTROL_PLANE_PATH = "repo:src/dendritic_modeling/scripts/sweeps/control_plane.py"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_digest(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return text.replace(old, new, 1)


def _normalize_factory(text: str) -> str:
    replacements = (
        ("    _LEGENDRE_MEMORY_TYPES,\n", "", "Legendre type import"),
        (
            "    _build_legendre_memory_architecture,\n",
            "",
            "Legendre builder import",
        ),
        (
            "    | set(_LEGENDRE_MEMORY_TYPES)\n",
            "",
            "Legendre built-in type registration",
        ),
        (
            "    builders.update(\n"
            "        dict.fromkeys(_LEGENDRE_MEMORY_TYPES, "
            "_build_legendre_memory_architecture)\n"
            "    )\n",
            "",
            "Legendre registry entry",
        ),
    )
    for old, new, label in replacements:
        text = _replace_once(text, old, new, label)
    return text


def _normalize_indexed_sparse(text: str) -> str:
    replacements = (
        ("from numbers import Integral\n", "", "Integral import"),
        (
            "        if isinstance(K, bool) or not isinstance(K, Integral):\n"
            "            raise TypeError(\"K must be an integer\")\n"
            "        K = int(K)\n",
            "        if not isinstance(K, int):\n"
            "            K = int(K)\n",
            "K validation",
        ),
        (
            "        if isinstance(output_chunk_size, bool) or not isinstance(\n"
            "            output_chunk_size, Integral\n"
            "        ):\n"
            "            raise TypeError(\"output_chunk_size must be an integer\")\n"
            "        output_chunk_size = int(output_chunk_size)\n",
            "",
            "output chunk validation",
        ),
    )
    for old, new, label in replacements:
        text = _replace_once(text, old, new, label)
    return text


def _normalize_spatial_morphology(text: str) -> str:
    text = _replace_once(
        text,
        "\n\ndef uses_spatial_morphology(config, pathway: str) -> bool:\n"
        "    \"\"\"Return whether one configured pathway uses spatial image routing.\"\"\"\n"
        "\n"
        "    pathway_config = _resolve_pathway_config(config, pathway)\n"
        "    method = str(pathway_config.get(\"method\", \"\")).strip().lower()\n"
        "    return method in SPATIAL_MORPHOLOGY_METHODS\n",
        "",
        "recurrent spatial-routing helper",
    )
    return _replace_once(
        text,
        '    "uses_spatial_morphology",\n',
        "",
        "recurrent spatial-routing export",
    )


def _normalize_control_plane(text: str) -> str:
    return _replace_once(
        text,
        "    SchedulerProfile(\n"
        '        profile_id="kempner_dev_rtx",\n'
        '        account="kempner_dev",\n'
        '        partition="kempner_rtx",\n'
        "    ),\n",
        "",
        "RTX scheduler profile",
    )


NORMALIZERS = {
    FACTORY_PATH: _normalize_factory,
    INDEXED_PATH: _normalize_indexed_sparse,
    SPATIAL_PATH: _normalize_spatial_morphology,
    CONTROL_PLANE_PATH: _normalize_control_plane,
}


def _latest_run_dirs() -> list[Path]:
    patterns = (
        "journal_confirmatory_*_depth_feedback_*",
        "journal_canary_inhibition_dose_*",
        "journal_canary_spatial_topology_*",
        "journal_canary_ancestry_routing_*",
        "journal_canary_fixed_budget_depth_*",
        "journal_confirmatory_inhibition_dose_*",
        "journal_confirmatory_spatial_topology_*",
        "journal_confirmatory_ancestry_routing_*",
        "journal_confirmatory_fixed_budget_depth_*",
    )
    by_prefix: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(RUN_ROOT.glob(pattern)):
            if not path.is_dir():
                continue
            prefix = re.sub(r"_\d{14}$", "", path.name)
            previous = by_prefix.get(prefix)
            if previous is None or path.name > previous.name:
                by_prefix[prefix] = path
    return sorted(by_prefix.values())


def _manifest_identity(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "frozen_sweep_manifest.json"
    return json.loads(path.read_text())["source_identity"]


def _workspace_path(recorded_path: str) -> Path:
    if not recorded_path.startswith("repo:"):
        raise RuntimeError(f"Unsupported source path: {recorded_path}")
    return WORKSPACE_ROOT / recorded_path.removeprefix("repo:")


def _config_scope(run_dirs: list[Path]) -> dict[str, Any]:
    architecture_types: set[str] = set()
    scheduler_profiles: set[str] = set()
    k_types: set[str] = set()
    chunk_types: set[str] = set()
    config_count = 0
    indexed_config_count = 0
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "frozen_sweep_manifest.json").read_text())
        profile = manifest.get("scheduler_profile") or {}
        scheduler_profiles.add(str(profile.get("profile_id", "")))
        for record in manifest["generated_configs"]:
            config = OmegaConf.load(run_dir / record["path"])
            config_count += 1
            architecture_types.add(str(config.model.core.type))
            sparsity_type = str(
                OmegaConf.select(config, "model.core.sparsity.type", default="")
            ).lower()
            if sparsity_type != "indexed":
                continue
            indexed_config_count += 1
            connectivity = OmegaConf.select(config, "model.core.connectivity")
            k_values: list[Any] = []
            for key in (
                "ee_synapses_per_branch_per_layer",
                "ei_synapses_per_branch_per_layer",
                "ie_synapses_per_branch_per_layer",
                "ii_synapses_per_branch_per_layer",
            ):
                values = OmegaConf.select(config, f"model.core.connectivity.{key}")
                if values is not None:
                    k_values.extend(list(values))
            if not k_values and connectivity is not None:
                scalar = OmegaConf.select(config, "model.core.connectivity.synapses_per_branch")
                if scalar is not None:
                    k_values.append(scalar)
            chunk_value = OmegaConf.select(
                config, "model.core.sparsity.indexed.output_chunk_size"
            )
            if not k_values:
                raise RuntimeError(
                    f"No indexed-sparse K values found in {run_dir / record['path']}"
                )
            k_types.update(type(value).__name__ for value in k_values)
            chunk_types.add(type(chunk_value).__name__)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in k_values
            ):
                raise RuntimeError(f"Non-integer indexed K in {run_dir / record['path']}")
            if isinstance(chunk_value, bool) or not isinstance(chunk_value, int):
                raise RuntimeError(
                    "Non-integer indexed output_chunk_size in "
                    f"{run_dir / record['path']}"
                )
    forbidden = {"legendre_memory", "lmu", "legendre_memory_unit"}
    overlap = forbidden & architecture_types
    if overlap:
        raise RuntimeError(f"New factory registrations are exercised: {sorted(overlap)}")
    if "kempner_dev_rtx" in scheduler_profiles:
        raise RuntimeError("New RTX scheduler profile is exercised by a frozen run")
    return {
        "config_count": config_count,
        "architecture_types": sorted(architecture_types),
        "scheduler_profiles": sorted(scheduler_profiles),
        "indexed_config_count": indexed_config_count,
        "indexed_K_python_types": sorted(k_types),
        "indexed_output_chunk_size_python_types": sorted(chunk_types),
    }


def audit() -> dict[str, Any]:
    run_dirs = _latest_run_dirs()
    if not run_dirs:
        raise RuntimeError("No prospective run directories found")
    identities = {_identity_digest(_manifest_identity(path)): _manifest_identity(path) for path in run_dirs}

    accepted_identities: list[str] = []
    file_records: dict[str, dict[str, Any]] = {}
    for identity_sha, identity in identities.items():
        mismatches: list[str] = []
        for record in identity.get("files", []):
            recorded_path = str(record["path"])
            frozen_sha = str(record["sha256"])
            current_path = _workspace_path(recorded_path)
            current_sha = _sha256_file(current_path)
            if current_sha == frozen_sha:
                continue
            normalizer = NORMALIZERS.get(recorded_path)
            if normalizer is None:
                mismatches.append(recorded_path)
                continue
            normalized_sha = _sha256_bytes(
                normalizer(current_path.read_text()).encode()
            )
            if normalized_sha != frozen_sha:
                mismatches.append(recorded_path)
                continue
            prior = file_records.get(recorded_path)
            candidate = {
                "frozen_sha256": frozen_sha,
                "current_sha256": current_sha,
                "normalized_sha256": normalized_sha,
            }
            if prior is not None and prior != candidate:
                mismatches.append(recorded_path)
                continue
            file_records[recorded_path] = candidate
        if mismatches:
            raise RuntimeError(
                f"Unverified source changes for identity {identity_sha}: "
                + ", ".join(sorted(set(mismatches)))
            )
        accepted_identities.append(identity_sha)

    scope = _config_scope(run_dirs)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "accepted_frozen_identity_sha256": sorted(accepted_identities),
        "run_directories": [str(path.relative_to(JOURNAL_ROOT)) for path in run_dirs],
        "current_source_exceptions": file_records,
        "configuration_scope": scope,
        "interpretation": (
            "The factory change only registers unused Legendre-memory architectures. "
            "The indexed-sparse change only validates integer-valued K and chunk-size "
            "arguments that are already Python integers in every audited configuration. "
            "The spatial-morphology change only exports a helper used by recurrent "
            "architectures absent from the audited configurations. The control-plane "
            "change only registers an RTX scheduler profile that no frozen run selected. "
            "After removing those exact inactive changes, each file matches its frozen "
            "SHA-256 byte for byte."
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    scope = report["configuration_scope"]
    lines = [
        "# Prospective source-equivalence audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        "Scope: this is a collection-time attestation tied to the recorded "
        "source hashes. Frozen manifests and archived source copies define "
        "the historical executions after a development checkout advances.",
        "",
        report["interpretation"],
        "",
        f"Audited run families: {len(report['run_directories'])}.",
        f"Audited frozen configurations: {scope['config_count']}.",
        "Architecture types: " + ", ".join(scope["architecture_types"]) + ".",
        "Scheduler profiles: " + ", ".join(scope["scheduler_profiles"]) + ".",
        f"Indexed-sparse configurations: {scope['indexed_config_count']}.",
        "",
        "| Source file | Frozen SHA-256 | Current SHA-256 | Normalized SHA-256 |",
        "|---|---|---|---|",
    ]
    for source, values in sorted(report["current_source_exceptions"].items()):
        lines.append(
            f"| `{source}` | `{values['frozen_sha256']}` | "
            f"`{values['current_sha256']}` | `{values['normalized_sha256']}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=JOURNAL_ROOT / "analysis/prospective_source_equivalence.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=JOURNAL_ROOT / "analysis/prospective_source_equivalence.md",
    )
    args = parser.parse_args()
    report = audit()
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.markdown_output)
    print(
        "verified source equivalence for "
        f"{len(report['accepted_frozen_identity_sha256'])} frozen identities"
    )


if __name__ == "__main__":
    main()
