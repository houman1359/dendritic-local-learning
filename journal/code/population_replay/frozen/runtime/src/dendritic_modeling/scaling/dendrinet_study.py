"""Generate and structurally qualify real production DendriNet scaling specs.

This module does not prepare, submit, or run a campaign. The 48-task profile is
an engineering qualification, not evidence of an error/parameter exponent.
Calibration and sweep specifications are plans requiring profile/convergence
review before preparation with :mod:`dendritic_modeling.scaling.campaign`.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any


def architecture_catalog(*, extended_controls: bool = False) -> list[dict]:
    """Independent recipes; a budget resolver selects each recipe's soma width."""
    rows: list[dict] = []

    def add(name: str, family: str, factors: list[int], e: int = 16,
            i: int = 0, depth: int = 1) -> None:
        rows.append(dict(name=name, family=family, branch_factors=factors,
                         contacts_e=e, contacts_i=i, network_depth=depth))

    for depth in (1, 2, 3):
        add(f"dense_d{depth}", "dense", [], depth=depth)
    for depth in (1, 2, 3):
        add(f"sparse_k16_d{depth}", "sparse", [], depth=depth)
    for name, factors in (("flat", []), ("b2", [2]), ("b6", [6]),
                          ("b22", [2, 2]), ("b222", [2, 2, 2])):
        add(f"signed_{name}_k16", "dendritic_signed", factors)
    for contacts in (4, 32):
        add(f"signed_b22_k{contacts}", "dendritic_signed", [2, 2], e=contacts)
    for depth in (2, 3):
        add(f"signed_b22_k16_d{depth}", "dendritic_signed", [2, 2], depth=depth)
    for name, factors in (("b6", [6]), ("b22", [2, 2])):
        for integration in ("additive", "shunting"):
            add(f"{integration}_{name}_e12_i4", f"dendritic_{integration}",
                factors, e=12, i=4)
    for e, i in ((3, 1), (24, 8)):
        add(f"shunting_b22_e{e}_i{i}", "dendritic_shunting", [2, 2], e=e, i=i)
    add("shunting_b222_e12_i4", "dendritic_shunting", [2, 2, 2], e=12, i=4)
    add("shunting_b22_e12_i4_d3", "dendritic_shunting", [2, 2], e=12, i=4, depth=3)
    add("grouped_b22_e12_i4", "grouped_gated", [2, 2], e=12, i=4)
    if extended_controls:
        # A DendriNet with two internal branch levels has three gates on its
        # longest path. Stacking two/three populations gives six/nine gates.
        for depth in (6, 9):
            add(f"dense_d{depth}", "dense", [], depth=depth)
            add(f"sparse_k16_d{depth}", "sparse", [], depth=depth)
    return rows


def morphology_axes(architecture: dict) -> dict:
    factors = architecture["branch_factors"]
    level, sites = 1, 1
    for factor in factors:
        level *= factor
        sites += level
    dendritic = architecture["family"].startswith("dendritic_")
    return {
        "family": architecture["family"],
        "branch_factors": list(factors),
        "internal_branch_depth": len(factors) if dendritic else None,
        "compartments_per_soma": sites if dendritic else None,
        "network_depth": architecture["network_depth"],
        "longest_branch_soma_gate_path": (
            (len(factors) + 1) * architecture["network_depth"] if dendritic else None
        ),
        "contacts_e_per_compartment": architecture["contacts_e"] if dendritic else None,
        "contacts_i_per_compartment": architecture["contacts_i"] if dendritic else None,
        "contact_placement": (
            ("all_nonsomatic_depths_no_soma_contacts" if factors else "soma_only")
            if dendritic else "ordinary_control"
        ),
        "width_policy": "resolve_integer_soma_or_hidden_width_at_whole_model_budget",
    }


def make_spec(stage: str = "profile", *, dataset: str = "synthetic_composition",
              target_seed: int = 3101, data_seed: int = 20260915,
              parameter_budgets: list[int] | None = None) -> dict:
    """Return a standard scaling.campaign spec with a separate design receipt.

    Calibration uses equal LR search coverage, not validation-selected rates.
    The candidate sweep's default LR/horizon must be replaced after calibration;
    it cannot establish a scaling law merely by completing its fixed horizon.
    Model seeds also vary sparse topology in the existing campaign interface.
    Target functions and sample seeds require separately generated campaigns.
    """
    if stage not in {"profile", "calibration", "sweep", "depth_controls"}:
        raise ValueError("stage must be profile, calibration, sweep, or depth_controls")
    if dataset not in {"synthetic_composition", "synthetic_global", "synthetic_gain"}:
        raise ValueError("Unsupported qualification dataset")
    rows = architecture_catalog(extended_controls=stage == "depth_controls")
    if stage == "depth_controls":
        rows = rows[-4:]
    axes = {row["name"]: morphology_axes(row) for row in rows}
    if stage == "calibration":
        # Calibrate every profiled architecture; do not transfer a favorable LR
        # from a shallow signed arm to deep conductance arms without evidence.
        expanded = []
        expanded_axes = {}
        for row in rows:
            for lr in (0.0003, 0.001, 0.003):
                name = f"{row['name']}_lr{str(lr).replace('.', 'p')}"
                expanded.append({**deepcopy(row), "name": name,
                                 "training_overrides": {"lr": lr}})
                expanded_axes[name] = {**axes[row["name"]],
                                       "base_architecture_id": row["name"]}
        rows, axes = expanded, expanded_axes
    budgets = parameter_budgets if parameter_budgets is not None else {
        "profile": [8192, 131072], "calibration": [8192, 131072],
        "sweep": [8192, 16384, 32768, 65536, 131072],
        "depth_controls": [16384, 32768, 65536, 131072],
    }[stage]
    steps = {"profile": 40, "calibration": 1000, "sweep": 4000, "depth_controls": 4000}[stage]
    spec = {
        "name": f"real_dendrinet_{stage}_{dataset}_t{target_seed}_v1",
        "phase": stage if stage in {"profile", "calibration"} else "pilot",
        "evaluate_test": False,
        "budget_tolerance": 0.02,
        "parameter_budgets": list(budgets),
        "data_budgets": [4096] if stage in {"profile", "calibration"} else [4096, 16384],
        "seeds": [101] if stage == "profile" else ([101, 103] if stage == "calibration" else [211, 223, 227]),
        "model_defaults": {
            "input_dim": 64, "output_dim": 4, "activation": "relu",
            "input_encoding": "signed_split", "initialization": "fan_in",
            "clip_contacts": False, "projection_backend": "eager",
        },
        "architectures": rows,
        "data": {
            "dataset": dataset, "data_seed": int(data_seed), "target_seed": int(target_seed),
            "validation_size": 1024 if stage == "profile" else 4096,
            "test_size": 8192, "task_complexity": {"terms": 16, "interaction_strength": 0.6},
        },
        "training": {
            "steps": steps, "batch_size": 128, "lr": 0.001,
            "weight_decay": 0.0001, "eval_every": 20 if stage == "profile" else 250,
            "device": "cuda", "num_threads": 2, "deterministic": True,
            "diagnostics_every": 20 if stage == "profile" else 250,
        },
        "study_design": {
            "schema_version": "real_dendrinet_study_v1",
            "stage": stage,
            "qualification_only": stage == "profile",
            "requires_prior_profile_and_convergence_review": stage != "profile",
            "architecture_axes": axes,
            "outcome": "classification cross-entropy and accuracy; no regression risk equivalence claimed",
            "budget_policy": "all registered whole-model parameters including readout; within 2%; no dormant padding",
            "allocation_note": "At fixed P, increasing branches/contacts/depth reduces feasible width. Width and internal complexity are allocation alternatives, not independent fixed-P axes.",
            "equal_compartment_depth_pairs": [
                ["signed_b6_k16", "signed_b22_k16"],
                ["additive_b6_e12_i4", "additive_b22_e12_i4"],
                ["shunting_b6_e12_i4", "shunting_b22_e12_i4"],
            ],
            "interpretation_limits": [
                "A 40-step profile measures execution, memory and early conditioning, not scaling exponents.",
                "Contacts exist at every nonsomatic depth; depth changes input injection paths even in equal-compartment comparisons.",
                "ReLU gates are enabled at branches and somas; signed_split encoding is shared by every arm.",
                "Signed ReLU hidden trees have no learned gate thresholds and are positively homogeneous; only the affine readout adds a bias.",
                "Positive shunting produces nonnegative voltages, so its ReLU gates act as identity; divisive integration supplies the internal nonlinearity.",
                "Random sparse supports are unstructured; no learned routing or local input grouping is implied.",
                "Fixed parameter counts do not match FLOPs, memory, optimizer conditioning, or training convergence.",
                "The existing seed axis couples initialization and routing; independent target/data replicates are separate campaigns.",
                "The sweep is a candidate plan. Rates, horizon, wider budgets and dataset/function replication require qualification.",
            ],
            "larger_budget_candidates_pending_memory_profile": [262144, 524288],
            "extended_depth_control_policy": "Separate depth_controls spec starts at P16384: dense_d9 has no width within 2% of P8192 (nearest7996). Reuse sweep's matching dataset, samples, seeds, and signed/shunting DendriNet d2/d3 references; do not rerun duplicate references. Gates along a path are a comparator axis, not equivalence of circuit dynamics.",
            "extended_depth_reference_architectures": [
                "signed_b22_k16_d2", "signed_b22_k16_d3", "shunting_b22_e12_i4_d3"
            ],
        },
    }
    validate_design(spec)
    return spec


def validate_design(spec: dict) -> None:
    """Reject silent confounds before costly model construction or preparation."""
    if spec.get("evaluate_test") is not False:
        raise ValueError("Development study must not open test data")
    tolerance = spec.get("budget_tolerance", 1)
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 <= tolerance <= 0.02:
        raise ValueError("Whole-model budget tolerance must be <=2%")
    for key in ("parameter_budgets", "data_budgets", "seeds"):
        values = spec[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "seeds" else 1) for v in values):
            raise ValueError(f"{key} contains an invalid integer")
    names = [row["name"] for row in spec["architectures"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Architecture names must be nonempty and unique")
    for row in spec["architectures"]:
        model = {**spec["model_defaults"], **row}
        if model.get("clip_contacts") is not False:
            raise ValueError("Contact clipping would change the declared architecture")
        if model.get("input_encoding") != "signed_split":
            raise ValueError("Every qualification arm must use the shared input encoding")
        expected = morphology_axes(row)
        observed = spec["study_design"]["architecture_axes"][row["name"]]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale architecture-axis receipt: {row['name']}")


def qualify_spec(spec: dict) -> dict[str, Any]:
    """Instantiate each architecture/budget, verify production paths and count.

    CPU structural qualification only; CUDA forward/backward and conditioning
    qualification still require the short profile campaign. No data are opened.
    """
    import torch
    from .models import build_model, match_parameter_budget, model_report
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import DendriNet

    validate_design(spec)
    torch.set_num_threads(1)
    rows = []
    for architecture in spec["architectures"]:
        model_spec = deepcopy(spec["model_defaults"])
        model_spec.update({k: v for k, v in architecture.items() if k not in {"name", "training_overrides"}})
        for budget in spec["parameter_budgets"]:
            resolved, planned = match_parameter_budget(model_spec, budget, tolerance=spec["budget_tolerance"])
            model = build_model(resolved)
            actual = model_report(model)
            population_names = [name for name, module in model.named_modules() if isinstance(module, DendriNet)]
            dendritic = resolved["family"].startswith("dendritic_")
            expected_count = resolved["network_depth"] if dendritic else 0
            if len(population_names) != expected_count:
                raise ValueError("Model did not instantiate the declared production DendriNet populations")
            if actual["total_parameters"] != planned["total_parameters"]:
                raise ValueError("Repeated model construction changed parameter count")
            if actual["trainable_parameters"] != actual["total_parameters"]:
                raise ValueError("Study model contains nontrainable parameter padding")
            for layer in actual["layer_receipts"]:
                if dendritic and (layer["contacts_e"] != resolved["contacts_e"] or layer["contacts_i"] != resolved["contacts_i"]):
                    raise ValueError("Observed contacts differ from declared contacts")
            inventory = actual.get("dendrinet_inventory")
            if dendritic:
                if inventory is None:
                    raise ValueError("Production DendriNet inventory is required for qualification")
                if inventory["population_count"] != expected_count:
                    raise ValueError("Production inventory disagrees with DendriNet module tree")
                axes = morphology_axes(architecture)
                if inventory["total_compartments"] != expected_count * resolved["width"] * axes["compartments_per_soma"]:
                    raise ValueError("Observed compartments differ from declared morphology")
            rows.append({
                "architecture_id": architecture["name"], "target_parameters": budget,
                "actual_parameters": actual["total_parameters"], "width": resolved["width"],
                "budget_relative_error": planned["budget_relative_error"],
                "production_dendrinet_module_names": population_names,
                "resolved_spec": resolved, "dendrinet_inventory": inventory,
            })
    # With all-depth contacts, [6] and [2,2] have exactly six nonsomatic
    # compartments/couplings per soma and identical parameter cost. This is a
    # depth/topology comparison at identical width, not only approximately P.
    for left, right in spec["study_design"]["equal_compartment_depth_pairs"]:
        for budget in spec["parameter_budgets"]:
            matches = [r for r in rows if r["target_parameters"] == budget
                       and r["architecture_id"].split("_lr")[0] in {left, right}]
            bases = {r["architecture_id"].split("_lr")[0] for r in matches}
            if bases and bases != {left, right}:
                raise ValueError("An equal-compartment comparison is missing its paired arm")
            if len({(r["actual_parameters"], r["width"]) for r in matches}) > 1:
                raise ValueError("Equal-compartment morphology pair changed whole P or width")
    return {
        "schema_version": "real_dendrinet_structural_qualification_v1",
        "status": "passed", "cuda_profile_completed": False, "test_data_opened": False,
        "task_count": len(spec["architectures"]) * math.prod(len(spec[k]) for k in ("parameter_budgets", "data_budgets", "seeds")),
        "resolved_model_count": len(rows), "models": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-spec")
    generate.add_argument("--stage", choices=("profile", "calibration", "sweep", "depth_controls"), default="profile")
    generate.add_argument("--dataset", default="synthetic_composition")
    generate.add_argument("--target-seed", type=int, default=3101)
    generate.add_argument("--data-seed", type=int, default=20260915)
    generate.add_argument("--parameter-budgets", type=int, nargs="+")
    generate.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser("qualify-spec")
    qualify.add_argument("--spec", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.command == "generate-spec":
        result = make_spec(args.stage, dataset=args.dataset, target_seed=args.target_seed,
                           data_seed=args.data_seed, parameter_budgets=args.parameter_budgets)
    else:
        result = qualify_spec(json.loads(args.spec.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "status": result.get("status", "specification_written")}))


if __name__ == "__main__":
    main()
