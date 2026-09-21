"""Bounded production DendriNet conditioning and gate qualification designs.

Generating or structurally qualifying a spec does not prepare or launch jobs.
The readout intervention changes only initialization. The gate profile changes
the model's function class and counts its learned gates before matching width.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .dendrinet_study import make_spec, morphology_axes, qualify_spec, validate_design


BASE_RECIPES = (
    "dense_d3", "sparse_k16_d3", "signed_b22_k16", "signed_b22_k16_d3",
    "additive_b22_e12_i4", "shunting_b22_e12_i4", "shunting_b22_e12_i4_d3",
)
READOUT_POLICIES = {
    "preserve": {"mode": "preserve"},
    "train_batch_rms": {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 128},
}
GATES = ("relu", "param_relu", "param_tanh")
ANCHOR_RECIPES = ("shunting_b22_e12_i4", "shunting_b22_e12_i4_d3")


def make_intervention_spec(kind: str) -> dict:
    if kind not in {"readout_pair", "gate_profile"}:
        raise ValueError("kind must be readout_pair or gate_profile")
    spec = make_spec("profile")
    catalog = {row["name"]: row for row in spec["architectures"]}
    selected = BASE_RECIPES if kind == "readout_pair" else tuple(
        name for name in BASE_RECIPES if name != "additive_b22_e12_i4"
    )
    rows, axes = [], {}
    for base_name in selected:
        protocols = ["main"]
        if kind == "readout_pair" and base_name in ANCHOR_RECIPES:
            protocols.append("anchor_lr0p003_steps1000")
        for protocol in protocols:
            for treatment in (READOUT_POLICIES if kind == "readout_pair" else GATES):
                row = deepcopy(catalog[base_name])
                row["name"] = f"{base_name}_{'init' if kind == 'readout_pair' else 'gate'}_{treatment}"
                if protocol != "main":
                    row["name"] += f"_{protocol}"
                if kind == "readout_pair":
                    row["training_overrides"] = {
                        "readout_initialization": deepcopy(READOUT_POLICIES[treatment])
                    }
                    if protocol != "main":
                        row["training_overrides"].update(lr=0.003, steps=1000)
                else:
                    row["activation"] = treatment
                    if treatment != "relu":
                        row["activation_init"] = {"gain": 1.0, "threshold": 0.0}
                rows.append(row)
                axes[row["name"]] = {
                    **morphology_axes(row), "base_architecture_id": base_name,
                    "intervention": treatment, "protocol": protocol,
                    "activation": row.get("activation", spec["model_defaults"]["activation"]),
                }
    spec.update({
        "name": f"real_dendrinet_{kind}_synthetic_composition_t3101_v1",
        "phase": "development" if kind == "readout_pair" else "profile",
        "seeds": [401, 409] if kind == "readout_pair" else [401],
        "architectures": rows,
    })
    spec["data"]["validation_size"] = 4096
    spec["training"].update({
        "steps": 500 if kind == "readout_pair" else 40,
        "eval_every": 100 if kind == "readout_pair" else 20,
        "diagnostics_every": 100 if kind == "readout_pair" else 20,
        "readout_initialization": {"mode": "preserve"},
    })
    spec["study_design"] = {
        "schema_version": "real_dendrinet_interventions_v1",
        "stage": kind,
        "qualification_only": kind == "gate_profile",
        "architecture_axes": axes,
        "equal_compartment_depth_pairs": [],
        "expected_tasks": 72 if kind == "readout_pair" else 36,
        "outcome": "development classification cross-entropy, accuracy, and conditioning diagnostics",
        "budget_policy": "registered whole-model parameters including readout and learned gates; <=2%; no padding or contact clipping",
        "data_policy": f"TRAIN 4096 and validation 4096 from existing target 3101/data seed 20260915; no test materialization; model/support seeds {spec['seeds']} remain coupled.",
        "selection_policy": (
            "Main readout contrasts: fixed LR 0.001/500 steps. Shunting d1/d3 anchors: LR 0.003/1000 steps, with paired initialization arms within this protocol. No automatic continuation to 4000 steps."
            if kind == "readout_pair" else
            "Fixed LR 0.001/40 steps with explicit gain 1/threshold 0 for learned gates; engineering qualification only. No automatic longer-training release."
        ),
        "protocol_counts": {"main": 56, "anchor_lr0p003_steps1000": 16} if kind == "readout_pair" else {"main": 36},
        "paired_policy": (
            "Within each base recipe/P/seed/protocol, preserve and train_batch_rms share exact architecture, initial hidden state, sparse support, training examples/order and optimizer settings; only final-readout initialization differs. RMS uses the first 128 TRAIN examples without labels and adds zero parameters."
            if kind == "readout_pair" else
            "Each base/P compares relu,param_relu,param_tanh at matched whole-model budget. Every gate arm preserves readout initialization; no simultaneous RMS treatment. Learned gain/threshold are counted and width is rematched."
        ),
        "interpretation_limits": [
            "Completed calibration already showed validation cross-entropy deterioration in some trajectories; more optimization is not automatically better.",
            "The initial calibration review found budget-specific preferred learning-rate disagreement in 16/24 recipes and validation-CE deterioration in 108/288 fits. These observations motivate bounded interventions, not a uniform longer horizon.",
            "The 500-step readout contrasts address learning speed. The 1000-step/LR 0.003 shunting anchors allow a more successful optimization regime; comparing those protocols changes both LR and horizon and does not isolate either effect.",
            "Readout rescaling changes the initial logits and their upstream gradients; it does not diagnose every hidden conditioning failure.",
            "At fixed width, param_relu with gain1/threshold0 has initial ReLU behavior. Whole-P matching can change width and support, so the gate-profile arms are not an exact initialization-equivalence comparison.",
            "Param_tanh is the production nonnegative bounded gate; its gain 1/threshold 0 setting is an explicit starting point, not a data-calibrated operating point.",
            "Plain ReLU acts as identity on valid positive shunting voltages. Gate presence alone does not demonstrate active nonlinear computation.",
            "These development interventions and 40-step gate profiles cannot establish a new learned scaling exponent or a main accuracy/compute frontier.",
        ],
    }
    validate_intervention_spec(spec)
    return spec


def validate_intervention_spec(spec: dict) -> None:
    validate_design(spec)
    kind = spec["study_design"].get("stage")
    if kind not in {"readout_pair", "gate_profile"}:
        raise ValueError("Unknown intervention kind")
    expected_training = {"lr": 0.001, "batch_size": 128,
                         "steps": 500 if kind == "readout_pair" else 40,
                         "eval_every": 100 if kind == "readout_pair" else 20,
                         "diagnostics_every": 100 if kind == "readout_pair" else 20}
    if any(spec["training"].get(key) != value for key, value in expected_training.items()):
        raise ValueError("Base training protocol differs from the declared intervention")
    if spec["model_defaults"].get("activation") != "relu":
        raise ValueError("Base model defaults must preserve the profiled ReLU recipe")
    expected_bases = set(BASE_RECIPES)
    if kind == "gate_profile":
        expected_bases.remove("additive_b22_e12_i4")
    seen: dict[tuple[str, str], dict] = {}
    if spec["study_design"]["equal_compartment_depth_pairs"]:
        raise ValueError("This selected intervention grid has no equal-compartment tree pairs")
    if spec["training"].get("readout_initialization") != READOUT_POLICIES["preserve"]:
        raise ValueError("Base readout initialization must preserve the existing weights")
    for row in spec["architectures"]:
        receipt = spec["study_design"]["architecture_axes"][row["name"]]
        base, treatment = receipt["base_architecture_id"], receipt["intervention"]
        protocol = receipt["protocol"]
        key = (base, protocol)
        if base not in expected_bases or treatment in seen.setdefault(key, {}):
            raise ValueError("Duplicate or unknown intervention arm")
        seen[key][treatment] = row
        if kind == "readout_pair":
            if treatment not in READOUT_POLICIES:
                raise ValueError("Unknown readout treatment")
            expected_overrides = {"readout_initialization": READOUT_POLICIES[treatment]}
            if protocol == "anchor_lr0p003_steps1000" and base in ANCHOR_RECIPES:
                expected_overrides.update(lr=0.003, steps=1000)
            elif protocol != "main":
                raise ValueError("Unknown readout protocol")
            if row.get("training_overrides") != expected_overrides:
                raise ValueError("Readout pair must differ only in readout initialization")
        else:
            if protocol != "main":
                raise ValueError("Gate profile has no optimization anchor protocol")
            if treatment not in GATES or row.get("activation") != treatment:
                raise ValueError("Gate treatment and declared activation disagree")
            if row.get("training_overrides"):
                raise ValueError("Gate profile must not bundle a readout/training intervention")
            expected = None if treatment == "relu" else {"gain": 1.0, "threshold": 0.0}
            if row.get("activation_init") != expected:
                raise ValueError("Learned gates require the declared gain1/threshold0 initialization")
    expected_keys = {(base, "main") for base in expected_bases}
    if kind == "readout_pair":
        expected_keys |= {(base, "anchor_lr0p003_steps1000") for base in ANCHOR_RECIPES}
    if set(seen) != expected_keys:
        raise ValueError("Intervention grid is missing a base recipe")
    expected_treatments = set(READOUT_POLICIES if kind == "readout_pair" else GATES)
    for arms in seen.values():
        if set(arms) != expected_treatments:
            raise ValueError("Intervention grid is missing a paired arm")
        stripped = [{k: v for k, v in row.items() if k not in
                     {"name", "training_overrides", "activation", "activation_init"}}
                    for row in arms.values()]
        if any(value != stripped[0] for value in stripped[1:]):
            raise ValueError("Intervention arms changed another architectural axis")
        if kind == "readout_pair" and any(row.get("activation", "relu") != "relu" or "activation_init" in row for row in arms.values()):
            raise ValueError("Readout pair must retain the original ReLU gates")
    count = len(spec["architectures"])
    for axis in ("parameter_budgets", "data_budgets", "seeds"):
        count *= len(spec[axis])
    if count != spec["study_design"]["expected_tasks"]:
        raise ValueError("Intervention task count differs from its declared grid")


def qualify_intervention_spec(spec: dict) -> dict:
    validate_intervention_spec(spec)
    result = qualify_spec(spec)
    result["intervention_kind"] = spec["study_design"]["stage"]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-spec")
    generate.add_argument("--kind", choices=("readout_pair", "gate_profile"), required=True)
    generate.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser("qualify-spec")
    qualify.add_argument("--spec", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = (make_intervention_spec(args.kind) if args.command == "generate-spec"
              else qualify_intervention_spec(json.loads(args.spec.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "status": result.get("status", "specification_written")}))


if __name__ == "__main__":
    main()
