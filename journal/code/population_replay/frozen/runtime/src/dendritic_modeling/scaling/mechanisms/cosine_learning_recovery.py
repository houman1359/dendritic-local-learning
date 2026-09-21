"""Separate evaluation of qualified original models after an NFS alias failure.

The original campaign, its failed coordinator verification, models, receipts and
source snapshots remain untouched. A copied independent audit may qualify only
exact named missing NFS aliases whose recorded bytes equal a present, durably
bound sibling. Every other original binding stays mandatory. Numerical replay
and evaluation use byte-identical copies of the original five source modules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ORIGINAL_SOURCE_NAMES = (
    "cosine_learning_campaign.py",
    "cosine_learning_tasks.py",
    "radial_width_learning.py",
    "rank_learning.py",
    "rank_orthogonal_v2.py",
)
QUALIFIED_STATUS = "qualified_complete_duplicate_nfs_aliases"


def _original_source_names(initialized):
    """Carry a declared optional dependency without changing old five-file releases."""
    if "source/cosine_family_tasks.py" in initialized["bindings"]:
        return (*ORIGINAL_SOURCE_NAMES, "cosine_family_tasks.py")
    return ORIGINAL_SOURCE_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _progress(phase, **details):
    """Bounded operational progress; no numerical results or receipt changes."""
    print(json.dumps({"utc": _now(), "phase": phase, **details}), flush=True)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Artifact bindings must be nonempty relative paths")
    destination = root / path
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact binding escapes its declared root")
    return destination


def _merge(bindings, name, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid artifact digest")
    if name in bindings and bindings[name] != digest:
        raise ValueError(f"Conflicting original artifact bindings: {name}")
    bindings[name] = digest


def _check(root, bindings):
    for name, expected in bindings.items():
        if sha(_relative(root, name)) != expected:
            raise ValueError(f"Bound artifact changed: {name}")


def _load_original(release):
    source = release / "original_source"
    name = (
        "_cosine_recovery_original_"
        + hashlib.sha256(str(source).encode()).hexdigest()[:20]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, source / "__init__.py", submodule_search_locations=[str(source)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(name + ".cosine_learning_campaign")


def _marker_snapshot(root):
    path = root / "study_complete.json"
    return {"present": path.exists(), "sha256": sha(path) if path.exists() else None}


def initialize(release, original_root, audit_path):
    """Freeze a distinct recovery release; never initialize or edit the original."""
    release = Path(release).resolve()
    original_root = Path(original_root).resolve()
    audit_path = Path(audit_path).resolve()
    if release == original_root or release.is_relative_to(original_root):
        raise ValueError("Recovery release must be separate from the original campaign")
    if (release / "release.json").exists():
        raise FileExistsError("Recovery release already exists")
    audit = read(audit_path)
    if (
        audit.get("status") != QUALIFIED_STATUS
        or audit.get("errors") != []
        or Path(audit["original_root"]).resolve() != original_root
        or audit["initialized_sha256"] != sha(original_root / "initialized.json")
    ):
        raise ValueError(
            "Independent qualification audit is not valid for this original"
        )
    original_initialized = read(original_root / "initialized.json")
    _check(original_root, original_initialized["bindings"])
    if read(original_root / "config.json").get(
        "response_family", "fixed_three_cosine"
    ) != "fixed_three_cosine":
        raise ValueError("This qualified recovery adapter supports the fixed response only")
    release.mkdir(parents=True, exist_ok=True)
    (release / "source").mkdir(exist_ok=False)
    (release / "original_source").mkdir(exist_ok=False)
    (release / "source" / "__init__.py").write_text('"""Frozen recovery adapter."""\n')
    shutil.copyfile(__file__, release / "source" / "cosine_learning_recovery.py")
    for name in (*_original_source_names(original_initialized), "__init__.py"):
        original = original_root / "source" / name
        if sha(original) != original_initialized["bindings"]["source/" + name]:
            raise ValueError(
                "Original source snapshot differs from initialized binding"
            )
        shutil.copyfile(original, release / "original_source" / name)
    shutil.copyfile(audit_path, release / "qualification_audit.json")
    bound = [
        release / "qualification_audit.json",
        *sorted((release / "source").iterdir()),
        *sorted((release / "original_source").iterdir()),
    ]
    bound += [
        release / name
        for name in ("protocol.md", "preflight_checks.json", "environment.json")
        if (release / name).exists()
    ]
    _write(
        release / "release.json",
        {
            "schema": "cosine_learning_recovery_release_v1",
            "utc": _now(),
            "original_root": str(original_root),
            "original_initialized_sha256": audit["initialized_sha256"],
            "original_completion_marker": _marker_snapshot(original_root),
            "qualification_audit_sha256": sha(release / "qualification_audit.json"),
            "bindings": {str(path.relative_to(release)): sha(path) for path in bound},
            "completion_basis": "qualified_original_models",
            "original_coordinator_status": "failed",
            "scope": (
                "Separate qualified recovery, preserving original failed coordinator "
                "verification and all original files. Only exact independently audited "
                "duplicate NFS aliases receive a qualification; no scientific model or "
                "durable artifact is waived, retrained, selected or rewritten."
            ),
        },
    )
    verify_inputs(release)
    return release


def _verify_release(release):
    setup = read(release / "release.json")
    if setup.get("schema") != "cosine_learning_recovery_release_v1":
        raise ValueError("Unknown recovery release schema")
    _check(release, setup["bindings"])
    if sha(__file__) != setup["bindings"]["source/cosine_learning_recovery.py"]:
        raise ValueError("Executing recovery adapter differs from frozen source")
    if sha(release / "qualification_audit.json") != setup["qualification_audit_sha256"]:
        raise ValueError("Independent qualification audit changed")
    original = Path(setup["original_root"]).resolve()
    if _marker_snapshot(original) != setup["original_completion_marker"]:
        raise ValueError(
            "Original completion-marker state changed after recovery setup"
        )
    return setup


def _original_expectations(module, original, audit):
    module.verify(original)
    initialized = read(original / "initialized.json")
    if sha(original / "initialized.json") != audit["initialized_sha256"]:
        raise ValueError("Original initialized receipt changed")
    expected = dict(initialized["bindings"])
    _merge(expected, "initialized.json", audit["initialized_sha256"])
    config = read(original / "config.json")
    cases = read(original / "manifest.json")["study"]
    if set(audit["case_receipt_hashes"]) != {case["id"] for case in cases}:
        raise ValueError(
            "Independent audit does not bind exactly every original study case"
        )
    receipts = []
    for case in cases:
        directory = Path("cases") / "study" / case["id"]
        case_path = directory / "complete.json"
        receipt = read(original / case_path)
        case_hash = audit["case_receipt_hashes"][case["id"]]
        if sha(original / case_path) != case_hash:
            raise ValueError("Original study case receipt changed")
        _merge(expected, str(case_path), case_hash)
        if (
            receipt.get("status") != "complete"
            or receipt.get("case") != case
            or receipt["initialized_sha256"] != audit["initialized_sha256"]
        ):
            raise ValueError("Original case is not a completed exact manifest member")
        for name, digest in receipt["artifacts"].items():
            _merge(expected, name, digest)
        key = module.dataset_key(case)
        data_path = "data/" + key + ".npz"
        sidecar_path = "data/" + key + ".json"
        _merge(expected, data_path, receipt["dataset_sha256"])
        _merge(expected, sidecar_path, receipt["dataset_receipt_sha256"])
        sidecar = read(original / sidecar_path)
        if (
            sidecar["initialized_sha256"] != audit["initialized_sha256"]
            or sidecar["sha256"] != receipt["dataset_sha256"]
            or sidecar["receipt"]["normalization"] != receipt["normalization"]
            or sidecar["receipt"]["test_generated"]
        ):
            raise ValueError("Original public dataset receipt differs")
        _merge(
            expected,
            "private_teachers/" + key + ".json",
            sidecar["private_teacher_sha256"],
        )
        _merge(expected, "oracle_banks/" + key + ".npz", sidecar["oracle_bank_sha256"])
        if case["method"] == "oracle_private_plane":
            if receipt["oracle_bank_sha256"] != sidecar["oracle_bank_sha256"]:
                raise ValueError("Original private-plane diagnostic binding differs")
        elif receipt["oracle_bank_sha256"] is not None:
            raise ValueError("An ordinary original fit claims private oracle input")
        checkpoint_budgets = module._checkpoints(config, case)
        if set(receipt["checkpoints"]) != {str(value) for value in checkpoint_budgets}:
            raise ValueError(
                "Original milestone inventory differs from frozen schedule"
            )
        predecessor, iterations, closures, budget = None, 0, 0, 0
        final_state = None
        for index in range(math.ceil(case["steps"] / config["restart_every"])):
            stage_path = directory / f"stage_{index:03d}" / "complete.json"
            stage_key = str(stage_path)
            if stage_key not in receipt["artifacts"]:
                raise ValueError("Original case omitted a restart-block receipt")
            stage = read(original / stage_path)
            steps = min(config["restart_every"], case["steps"] - budget)
            if (
                stage["case"] != case
                or stage["stage"] != index
                or stage["steps"] != steps
                or stage["predecessor_sha256"] != predecessor
                or stage["initialized_sha256"] != audit["initialized_sha256"]
                or stage["dataset_sha256"] != receipt["dataset_sha256"]
                or stage["dataset_receipt_sha256"] != receipt["dataset_receipt_sha256"]
            ):
                raise ValueError("Original restart chain or TRAIN binding differs")
            for name, digest in stage["artifacts"].items():
                if receipt["artifacts"].get(name) != digest:
                    raise ValueError("Original case and restart artifact maps disagree")
                _merge(expected, name, digest)
            for name in ("state_path", "fit_path"):
                if stage[name] not in stage["artifacts"]:
                    raise ValueError(
                        "Original restart omitted its state or fit binding"
                    )
            fit = read(original / stage["fit_path"])
            if fit["status"] != "complete" or fit["config"]["steps"] != steps:
                raise ValueError(
                    "Original numerical fit does not match its restart budget"
                )
            iterations += fit["iterations"]
            closures += fit["closure_calls"]
            budget += steps
            checkpoint = stage["checkpoint"]
            if budget in checkpoint_budgets:
                if (
                    checkpoint != receipt["checkpoints"][str(budget)]
                    or checkpoint["budget_steps"] != budget
                    or checkpoint["iterations"] != iterations
                    or checkpoint["closure_calls"] != closures
                    or checkpoint["state_path"] != stage["state_path"]
                    or checkpoint["predictions_path"] not in stage["artifacts"]
                ):
                    raise ValueError(
                        "Original checkpoint binding or iteration accounting differs"
                    )
            elif checkpoint is not None:
                raise ValueError("Original restart has an undeclared milestone")
            predecessor = receipt["artifacts"][stage_key]
            final_state = stage["state_path"]
        if (
            receipt["state_path"] != final_state
            or receipt["iterations"] != iterations
            or receipt["closure_calls"] != closures
            or receipt["metrics"]
            != receipt["checkpoints"][str(case["steps"])]["metrics"]
        ):
            raise ValueError("Original final-model accounting differs")
        receipts.append(receipt)
    return expected, receipts


def verify_inputs(release):
    """Verify every historical expectation, waiving only exact audited aliases."""
    release = Path(release).resolve()
    _progress("verify_recovery_inputs_start", release=str(release))
    setup = _verify_release(release)
    original = Path(setup["original_root"]).resolve()
    audit = read(release / "qualification_audit.json")
    if (
        audit.get("status") != QUALIFIED_STATUS
        or audit.get("errors") != []
        or Path(audit["original_root"]).resolve() != original
        or audit["initialized_sha256"] != setup["original_initialized_sha256"]
    ):
        raise ValueError(
            "Independent qualification audit does not authorize this recovery"
        )
    module = _load_original(release)
    initialized = read(original / "initialized.json")
    for name in _original_source_names(initialized):
        original_digest = initialized["bindings"]["source/" + name]
        if sha(release / "original_source" / name) != original_digest:
            raise ValueError("Copied original numerical source changed")
    _progress("verify_original_structure_start")
    expected, receipts = _original_expectations(module, original, audit)
    _progress(
        "verify_original_structure_complete",
        models=len(receipts),
        expected_bindings=len(expected),
    )
    aliases = audit["qualified_missing_aliases"]
    if not aliases or len({entry["path"] for entry in aliases}) != len(aliases):
        raise ValueError("A nonempty unique inventory of qualified aliases is required")
    qualified = {}
    for entry in aliases:
        name, canonical = entry["path"], entry["canonical_path"]
        path, canonical_path = _relative(original, name), _relative(original, canonical)
        if (
            not re.fullmatch(r"\.nfs[0-9a-f]{24}", path.name)
            or path.parent != canonical_path.parent
            or path.parent.name[:8] != "attempt_"
            or canonical_path.name.startswith(".")
            or expected.get(name) != entry["sha256"]
            or expected.get(canonical) != entry["canonical_sha256"]
            or entry["sha256"] != entry["canonical_sha256"]
            or audit["durable_bindings"].get(canonical) != entry["canonical_sha256"]
        ):
            raise ValueError(
                "Qualification is not an exact duplicate NFS alias of a bound same-attempt sibling"
            )
        if sha(canonical_path) != entry["canonical_sha256"]:
            raise ValueError("Qualified alias canonical durable artifact changed")
        if path.exists() and sha(path) != entry["sha256"]:
            raise ValueError("A currently present qualified alias has different bytes")
        qualified[name] = entry
    mandatory = {
        name: digest for name, digest in expected.items() if name not in qualified
    }
    durable = audit["durable_bindings"]
    if any(name in durable for name in qualified):
        raise ValueError(
            "Qualified ephemeral aliases must not be called durable bindings"
        )
    for name, digest in mandatory.items():
        if durable.get(name) != digest:
            raise ValueError(
                f"Independent audit omitted or changed a mandatory original artifact: {name}"
            )
    # The audit may additionally bind profile receipts and the failed Slurm log.
    # Those additional durable inputs remain mandatory and are also rehashed.
    _progress("verify_original_durable_bindings_start", bindings=len(durable))
    _check(original, durable)
    _progress(
        "verify_recovery_inputs_complete",
        bindings=len(durable),
        qualified_aliases=len(qualified),
    )
    return module, receipts, dict(durable)


def _release_bindings(release):
    setup = read(release / "release.json")
    return {"release.json": sha(release / "release.json"), **setup["bindings"]}


def verify_complete(release):
    """Read-only verification for downstream analysis, including every output."""
    release = Path(release).resolve()
    _, receipts, bindings = verify_inputs(release)
    result = read(release / "evaluation_complete.json")
    if (
        result.get("status") != "complete"
        or result.get("completion_basis") != "qualified_original_models"
        or result.get("original_coordinator_status") != "failed"
        or result["models"] != len(receipts)
        or result["endpoint_rows"] != sum(len(row["checkpoints"]) for row in receipts)
        or result["test_rows"] != len(receipts)
        or result["input_bindings"] != bindings
        or result["release_bindings"] != _release_bindings(release)
        or result["qualification_audit_sha256"]
        != sha(release / "qualification_audit.json")
        or result["original_root"] != read(release / "release.json")["original_root"]
    ):
        raise ValueError("Recovery completion contract changed")
    _check(release, result["release_bindings"])
    _check(release, result["artifacts"])
    return result


def _public_observations(module, original, case):
    path = original / "data" / (module.dataset_key(case) + ".npz")
    sidecar = read(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if any("test" in name for name in arrays):
        raise ValueError("Original fitting observations unexpectedly contain TEST")
    for name, value in arrays.items():
        if module.tasks.array_sha256(value) != sidecar["receipt"]["array_hashes"][name]:
            raise ValueError("Original public observation array hash changed")
    return arrays


def _replay_milestones(module, original, receipts, config):
    observations, rows, checks = {}, [], []
    total = sum(len(receipt["checkpoints"]) for receipt in receipts)
    _progress("milestone_replay_start", milestones=total)
    for receipt in receipts:
        case = receipt["case"]
        key = module.dataset_key(case)
        if key not in observations:
            observations[key] = _public_observations(module, original, case)
        arrays = observations[key]
        normalization = receipt["normalization"]
        base_row = module._row_base(case, normalization)
        for budget, checkpoint in sorted(
            receipt["checkpoints"].items(), key=lambda item: int(item[0])
        ):
            state = original / checkpoint["state_path"]
            model = module.width.load_state(state, base_module=module.base)
            module._count_model(model, case)
            with np.load(
                original / checkpoint["predictions_path"], allow_pickle=False
            ) as archive:
                saved = {name: archive[name].copy() for name in archive.files}
            checkpoint_checks = {}
            for split in ("train", "validation"):
                prediction, replay_error = module._replay_prediction(
                    model, state, arrays["x_" + split], config
                )
                np.testing.assert_allclose(
                    prediction,
                    saved[split],
                    rtol=config["audit"]["forward_rtol"],
                    atol=config["audit"]["forward_atol"],
                )
                saved_mse = float(np.mean((saved[split] - arrays["y_" + split]) ** 2))
                if (
                    saved_mse != checkpoint["metrics"][split + "_mse"]
                    or saved_mse * normalization["scale"] ** 2
                    != checkpoint["metrics"][split + "_raw_mse"]
                ):
                    raise ValueError(
                        "Saved checkpoint scalar metric differs from its saved predictions"
                    )
                checkpoint_checks[split] = {
                    "native_vs_independent_max_error": replay_error,
                    "native_vs_saved_max_error": float(
                        np.max(np.abs(prediction - saved[split]))
                    ),
                    "native_equals_saved_exactly": bool(
                        np.array_equal(prediction, saved[split])
                    ),
                    "saved_metric_recomputed_exactly": True,
                }
            checks.append(
                {
                    "case_id": case["id"],
                    "budget_steps": int(budget),
                    "state_sha256": sha(state),
                    "saved_predictions_sha256": sha(
                        original / checkpoint["predictions_path"]
                    ),
                    "checks": checkpoint_checks,
                }
            )
            metrics = checkpoint["metrics"]
            rows.append(
                {
                    **base_row,
                    "kind": "train_validation_checkpoint",
                    "budget_steps": int(budget),
                    "actual_iterations": checkpoint["iterations"],
                    **{
                        name: metrics[name]
                        for name in (
                            "train_mse",
                            "validation_mse",
                            "train_raw_mse",
                            "validation_raw_mse",
                        )
                    },
                }
            )
            if len(checks) % 28 == 0 or len(checks) == total:
                _progress(
                    "milestone_replay_progress", completed=len(checks), total=total
                )
    return rows, checks


def evaluate(release, *, require_slurm=True):
    """Replay all original milestones, then evaluate every unchanged final model."""
    release = Path(release).resolve()
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production recovery evaluation requires a Slurm allocation")
    module, receipts, bindings = verify_inputs(release)
    marker = release / "evaluation_complete.json"
    if marker.exists():
        return verify_complete(release)
    original = Path(read(release / "release.json")["original_root"])
    config = read(original / "config.json")
    endpoints = sum(len(row["checkpoints"]) for row in receipts)
    if require_slurm and (
        len(receipts) != 84
        or endpoints != 252
        or config["evaluation_n"] != 65536
        or any(row["case"]["steps"] != 600 for row in receipts)
    ):
        raise ValueError(
            "Production recovery must evaluate the exact original 84-model, 252-milestone study"
        )
    torch.set_num_threads(1)
    with module.exclusive_lease(release / "recovery.lease"):
        if marker.exists():
            return verify_complete(release)
        module, receipts, bindings = verify_inputs(release)
        directory = release / "evaluation"
        if directory.exists() and list(directory.rglob("failure.json")):
            raise RuntimeError(
                "A previous recovery failure is retained; no automatic retry"
            )
        directory.mkdir(exist_ok=True)
        attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
        attempt.mkdir()
        started = time.monotonic()
        created = []
        try:
            release_binding_snapshot = _release_bindings(release)
            model_bindings = attempt / "model_bindings.json"
            module.write(
                model_bindings,
                {
                    "original_root": str(original),
                    "input_bindings": bindings,
                    "release_bindings": release_binding_snapshot,
                    "case_receipt_hashes": read(release / "qualification_audit.json")[
                        "case_receipt_hashes"
                    ],
                    "qualification_audit_sha256": sha(
                        release / "qualification_audit.json"
                    ),
                    "fixture_mode": not require_slurm,
                    "completion_basis": "qualified_original_models",
                    "original_coordinator_status": "failed",
                },
            )
            created.append(model_bindings)
            # This replay completes before any fresh evaluation observations exist.
            endpoint_rows, milestone_checks = _replay_milestones(
                module, original, receipts, config
            )
            replay_path = attempt / "milestone_replay.json"
            module.write(
                replay_path,
                {
                    "status": "passed",
                    "milestones": len(milestone_checks),
                    "checks": milestone_checks,
                    "audit": config["audit"],
                    "scope": "Unchanged native/NumPy replay tolerances on all original TRAIN/validation milestones; saved metrics recomputed exactly. No fitting or selection.",
                },
            )
            created.append(replay_path)
            # Recheck original models, observations and copied sources immediately
            # before the first fresh TEST draw.
            _progress("pre_test_integrity_verification_start")
            _, checked_receipts, checked_bindings = verify_inputs(release)
            if checked_bindings != bindings or checked_receipts != receipts:
                raise ValueError(
                    "Original inputs changed before evaluation observation release"
                )
            _check(release, release_binding_snapshot)
            datasets = {}
            _progress(
                "independent_observation_generation_start",
                datasets=len(
                    {module.dataset_key(receipt["case"]) for receipt in receipts}
                ),
            )
            for receipt in receipts:
                case = receipt["case"]
                key = module.dataset_key(case)
                if key not in datasets:
                    datasets[key] = module._evaluation_dataset(
                        original, case, config, attempt, sha(model_bindings)
                    )
                    created += [
                        attempt / ("test_" + key + suffix)
                        for suffix in (".npz", ".json")
                    ]
            _progress(
                "independent_observation_generation_complete", datasets=len(datasets)
            )
            test_rows = []
            _progress("final_model_evaluation_start", models=len(receipts))
            for receipt in receipts:
                case = receipt["case"]
                arrays, normalization = datasets[module.dataset_key(case)]
                base_row = module._row_base(case, normalization)
                state = original / receipt["state_path"]
                model = module.width.load_state(state, base_module=module.base)
                module._count_model(model, case)
                prediction, error = module._replay_prediction(
                    model, state, arrays["x_test"], config
                )
                losses = (prediction - arrays["y_test"]) ** 2
                if len(losses) < 2:
                    raise ValueError("At least two independent observations required")
                mse = float(np.mean(losses))
                mcse = float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
                row = {
                    **base_row,
                    "kind": "independent_observation_evaluation",
                    "budget_steps": case["steps"],
                    "actual_iterations": receipt["iterations"],
                    "test_mse": mse,
                    "test_raw_mse": mse * normalization["scale"] ** 2,
                    "test_mcse": mcse,
                    "test_raw_mcse": mcse * normalization["scale"] ** 2,
                    "test_rows": len(losses),
                    "empirical_test_below_two_direction_floor": mse
                    < base_row["two_direction_normalized_floor"],
                }
                test_rows.append(row)
                prediction_path = attempt / ("prediction_" + case["id"] + ".npz")
                module._save_arrays(prediction_path, {"prediction": prediction})
                module.write(
                    prediction_path.with_suffix(".json"),
                    {
                        "case": case,
                        "metrics": row,
                        "prediction_sha256": sha(prediction_path),
                        "state_sha256": sha(state),
                        "replay_max_error": error,
                        "qualification_audit_sha256": sha(
                            release / "qualification_audit.json"
                        ),
                        "model_bindings_sha256": sha(model_bindings),
                        "scope": "Every unchanged final model; independent-observation MCSE conditional on fixed development teacher/model. No rigorous interval or model selection.",
                    },
                )
                created += [prediction_path, prediction_path.with_suffix(".json")]
                if len(test_rows) % 7 == 0 or len(test_rows) == len(receipts):
                    _progress(
                        "final_model_evaluation_progress",
                        completed=len(test_rows),
                        total=len(receipts),
                    )
            if len(test_rows) != len(receipts) or len(endpoint_rows) != endpoints:
                raise ValueError("Recovery omitted a declared model or milestone")
            module._export_evaluation(attempt, endpoint_rows, test_rows)
            created += [attempt / "all_endpoints.csv", attempt / "summary.md"]
            summary = attempt / "summary.md"
            disclosure = (
                "This is a SEPARATE QUALIFIED RECOVERY. The original coordinator "
                "verification remains failed because its artifact inventory included "
                "an incidental missing NFS alias. An independent audit qualified only "
                "exact duplicate aliases of present bound sibling files. All original "
                "models, source snapshots, numerical tolerances and receipts are "
                "unchanged; no original completion marker was created. All original "
                "TRAIN/validation milestones were replayed before fresh evaluation "
                "observations were generated.\n\n"
            )
            summary.write_text(disclosure + summary.read_text())
            _progress("pre_publication_integrity_verification_start")
            _, checked_receipts, checked_bindings = verify_inputs(release)
            if checked_bindings != bindings or checked_receipts != receipts:
                raise ValueError("Original inputs changed before recovery publication")
            _check(release, release_binding_snapshot)
            # Explicit output paths, never recursive filesystem discovery: NFS
            # aliases and temporary names cannot enter this durable inventory.
            artifacts = {str(path.relative_to(release)): sha(path) for path in created}
            for path in created:
                target = directory / path.name
                os.link(path, target)
                artifacts[str(target.relative_to(release))] = sha(target)
            _progress("final_original_binding_recheck_start", bindings=len(bindings))
            _check(original, bindings)
            _check(release, release_binding_snapshot)
            _check(release, artifacts)
            result = {
                "utc": _now(),
                "status": "complete",
                "completion_basis": "qualified_original_models",
                "original_coordinator_status": "failed",
                "models": len(test_rows),
                "endpoint_rows": len(endpoint_rows),
                "test_rows": len(test_rows),
                "evaluation_n": config["evaluation_n"],
                "qualification_audit_sha256": sha(release / "qualification_audit.json"),
                "qualified_missing_aliases": read(release / "qualification_audit.json")[
                    "qualified_missing_aliases"
                ],
                "original_root": str(original),
                "input_bindings": bindings,
                "release_bindings": release_binding_snapshot,
                "artifacts": artifacts,
                "csv_path": "evaluation/all_endpoints.csv",
                "summary_path": "evaluation/summary.md",
                "milestone_replay_path": "evaluation/milestone_replay.json",
                "seconds": time.monotonic() - started,
                "fixture_mode": not require_slurm,
                "artifact_inventory_policy": "Only explicitly declared new outputs are bound; no recursive inventory of incidental filesystem names.",
                "scope": (
                    "Qualified recovery preserving the original failed coordinator status. "
                    "Every original final model evaluated, unchanged sources/tolerances, "
                    "all original milestones replayed, no fitting/selection. Fixed "
                    "DEVELOPMENT teachers, conditional independent-observation MCSE only; "
                    "no rigorous population-risk certificate, exponent or fresh-response confirmation."
                ),
            }
            module.write(marker, result)
            _progress(
                "recovery_complete",
                models=len(test_rows),
                milestones=len(endpoint_rows),
            )
            return result
        except Exception as error:
            module.write(
                attempt / "failure.json",
                {
                    "utc": _now(),
                    "error": repr(error),
                    "seconds": time.monotonic() - started,
                    "original_coordinator_status": "failed",
                    "scope": "Recovery failure preserved separately; original files untouched.",
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage", choices=("initialize", "evaluate"), required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    if args.stage == "initialize":
        if args.original_root is None or args.audit_path is None:
            parser.error("initialize requires --original-root and --audit-path")
        initialize(args.release, args.original_root, args.audit_path)
    else:
        print(json.dumps(evaluate(args.release), indent=2), flush=True)


if __name__ == "__main__":
    main()
