"""Verify and replay one rescue condition with the executed population runtime."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "release_noise"))
from release_hashes import digest, verify_released_file


def verified_sources():
    frozen = HERE / "frozen"
    identity = json.loads((frozen / "identity.json").read_text())
    for relative, expected in identity["files"].items():
        result = verify_released_file(frozen / relative, expected)
        if not result["verified"]:
            raise ValueError(f"Unverified source {relative}: {result['reason']}")
    protocol = json.loads((frozen / "fresh_protocol.json").read_text())
    for relative, expected in protocol["source_sha256"].items():
        if relative in identity["files"]:
            if identity["files"][relative] != expected:
                raise ValueError(f"Source identity disagrees with original protocol: {relative}")
    return frozen, identity, protocol


def main(args):
    frozen, identity, protocol = verified_sources()
    if args.verify_only:
        print(json.dumps({"verified_sources": len(identity["files"]),
                          "original_protocol_sha256": identity["original_fresh_protocol_sha256"]}))
        return
    if args.output is None:
        raise ValueError("A new --output directory is required")
    if args.seed not in protocol["fresh_seeds"] and args.sgd_rate is None:
        raise ValueError("Use one of the twenty frozen fresh seeds for a rescue replay")
    if args.sgd_rate is not None:
        if args.sgd_rate not in [.003, .01, .03, .1, .3, 1.]:
            raise ValueError("Rate is outside the bounded SGD development grid")
        if args.seed not in protocol["development_seeds"] or args.rule != "exact":
            raise ValueError("SGD development used only exact credit and three development seeds")
        if args.separable or args.common_rate:
            raise ValueError("SGD development cannot be combined with Adam task sensitivities")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(frozen / name) for name in
                   ["study", "selection", "base", "runtime/src"]]
    import torch
    import numpy as np
    import rescue
    import dendritic_modeling
    if not Path(dendritic_modeling.__file__).resolve().is_relative_to(frozen / "runtime"):
        raise ValueError("Population replay did not import the verified frozen runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.sgd_rate is not None:
        optimizer, rate, budget, phase = "sgd", args.sgd_rate, 32768, "development"
    else:
        optimizer, budget, phase = "adam", 4096, "fresh"
        rate = .03 if args.common_rate or args.separable else protocol["fixed_rates"][f"adam_b9_{args.rule}"]
    steps = args.smoke_steps or budget
    if steps <= 0 or steps > budget:
        raise ValueError("Smoke steps must be positive and no larger than the original budget")
    if args.separable:
        import experiment
        def data(seed, split, size, variant="interaction", severity=1.):
            return experiment.dataset(seed, split, size, "separable", severity)
        context = patch.multiple(rescue, dataset=data, diagnostics=lambda *a: [])
    else:
        context = nullcontext()
    with context:
        result, states = rescue.train(args.seed, args.rule, optimizer, 9, rate, steps, phase)
    torch.save(states, args.output / "states.pt")
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = dict(mode="excluded_smoke" if args.smoke_steps else "existing_seed_replay",
                  original_protocol_sha256=identity["original_fresh_protocol_sha256"],
                  source_identity_sha256=digest(frozen / "identity.json"),
                  seed=args.seed, rule=args.rule, optimizer=optimizer, rate=rate,
                  steps=steps, phase=phase, nonlinear_parents=True,
                  target="separable" if args.separable else "interaction",
                  state_sha256=digest(args.output / "states.pt"),
                  result_sha256=digest(args.output / "result.json"),
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  scope="Same frozen equations, data and selection; library differences can change trajectories")
    (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026100200)
    parser.add_argument("--rule", choices=["exact", "broadcast", "resistance", "derivative",
                                         "shuffled_derivative", "full_chain"], default="derivative")
    parser.add_argument("--common-rate", action="store_true")
    parser.add_argument("--separable", action="store_true")
    parser.add_argument("--sgd-rate", type=float)
    parser.add_argument("--smoke-steps", type=int)
    main(parser.parse_args())
