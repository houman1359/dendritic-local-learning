"""TRAIN-only logit-scale intervention on an existing affine readout."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


def _hidden_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("readout."):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialize_readout(model: nn.Module, training_inputs: torch.Tensor, spec: dict) -> dict:
    """Rescale only Linear readout weights/bias; never inspect labels or VAL.

    This intervention changes the optimization initialization, not the model
    class or parameter inventory. The declared TRAIN prefix is already inside
    the unique-data budget, and is recorded separately from optimized examples.
    """
    if not isinstance(spec, dict) or spec.get("mode") not in {"preserve", "train_batch_rms"}:
        raise ValueError("readout_initialization.mode must be preserve or train_batch_rms")
    if spec["mode"] == "preserve":
        if set(spec) != {"mode"}:
            raise ValueError("preserve mode has no calibration settings")
        return {"mode": "preserve", "training_examples_exposed": 0, "parameters_added": 0}
    if set(spec) != {"mode", "target_rms", "examples"}:
        raise ValueError("train_batch_rms requires exactly target_rms and examples")
    target, examples = spec["target_rms"], spec["examples"]
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise ValueError("readout target_rms must be finite and positive")
    if isinstance(examples, bool) or not isinstance(examples, int) or not 1 <= examples <= len(training_inputs):
        raise ValueError("readout examples must select an existing nonempty TRAIN prefix")
    if not isinstance(getattr(model, "readout", None), nn.Linear):
        raise ValueError("This intervention requires the existing nn.Linear readout")
    readout = model.readout
    parameters = sum(p.numel() for p in model.parameters())
    hidden_before = _hidden_hash(model)
    device = readout.weight.device
    x = training_inputs[:examples].to(device)
    training_modes = [(module, module.training) for module in model.modules()]
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    try:
        with torch.no_grad():
            before = model(x)
            if not torch.isfinite(before).all():
                raise FloatingPointError("Nonfinite logits before readout initialization")
            initial_rms = float(before.double().square().mean().sqrt().item())
            if initial_rms == 0:
                raise ValueError("Zero logits cannot be rescaled to a positive RMS")
            scale = float(target) / initial_rms
            readout.weight.mul_(scale)
            if readout.bias is not None:
                readout.bias.mul_(scale)
            after = model(x)
            if not torch.isfinite(after).all():
                raise FloatingPointError("Nonfinite logits after readout initialization")
            final_rms = float(after.double().square().mean().sqrt().item())
            if not math.isclose(final_rms, float(target), rel_tol=2e-5, abs_tol=1e-8):
                raise FloatingPointError("Readout RMS intervention failed its target check")
        hidden_after = _hidden_hash(model)
        if hidden_after != hidden_before or sum(p.numel() for p in model.parameters()) != parameters:
            raise RuntimeError("Readout initialization changed hidden state or parameter inventory")
    except BaseException:
        model.load_state_dict(original_state)
        raise
    finally:
        for module, was_training in training_modes:
            module.training = was_training
    return {
        "mode": "train_batch_rms", "target_rms": float(target),
        "initial_rms": initial_rms, "achieved_rms": final_rms, "scale": scale,
        "training_examples_exposed": examples,
        "initialization_forward_passes": 2,
        "initialization_examples_processed": 2 * examples,
        "prefix_within_existing_unique_training_budget": True,
        "labels_used": False, "validation_used": False, "parameters_added": 0,
        "hidden_state_sha256_before": hidden_before,
        "hidden_state_sha256_after": hidden_after,
        "parameters_scaled": readout.weight.numel() + (0 if readout.bias is None else readout.bias.numel()),
    }
