"""Auditable fixed-support model families for parameter-budget experiments.

The dendritic arms instantiate the production ``ConfigurableEINetwork``.  All
arms include a counted affine readout and the same information-preserving input
encoding.  No dense candidate bank, dynamic rewiring, or dormant budget-filling
parameter is introduced.  ``width`` always means exposed hidden outputs/somas.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn

from dendritic_modeling.networks.activations import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_backend,
)

FAMILIES = (
    "dense",
    "sparse",
    "dendritic_signed",
    "dendritic_additive",
    "dendritic_shunting",
    "grouped_gated",
)
_DEFAULTS = {
    "width": 32,
    "branch_factors": [2],
    "contacts_e": 4,
    "contacts_i": 0,
    "network_depth": 1,
    "seed": 0,
    "activation": "relu",
    "input_encoding": "signed_split",
    "clip_contacts": False,
    "initialization": "fan_in",
    "projection_backend": "eager",
}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _normalize_readout(readout: Any, output_dim: int) -> dict:
    """Validate opt-in signed allocation; omitted seeds retain inheritance."""
    if not isinstance(readout, dict):
        raise ValueError("readout must be a dict")
    kind = readout.get("kind")
    if not isinstance(kind, str) or kind not in {"dense", "indexed", "low_rank"}:
        raise ValueError("readout.kind must be dense, indexed, or low_rank")
    allowed = {"kind", "bias", "init_std", "parameter_seed"}
    if kind == "indexed":
        allowed |= {"topk", "topology_seed", "projection_backend"}
    elif kind == "low_rank":
        allowed.add("rank")
    unknown = set(readout) - allowed
    if unknown:
        raise ValueError(f"Unknown readout keys for {kind}: {sorted(unknown)}")
    result = {"bias": True, "init_std": 0.02, **deepcopy(readout)}
    if not isinstance(result["bias"], bool):
        raise ValueError("readout.bias must be a bool")
    std = result["init_std"]
    if (
        isinstance(std, bool)
        or not isinstance(std, Real)
        or not math.isfinite(std)
        or std <= 0
    ):
        raise ValueError("readout.init_std must be finite and > 0")
    result["init_std"] = float(std)
    for name in ("parameter_seed", "topology_seed"):
        if name in result:
            result[name] = _integer(result[name], f"readout.{name}", 0)
    if kind == "indexed":
        result["topk"] = _integer(result.get("topk"), "readout.topk")
        backend = result.get("projection_backend", "eager")
        if not isinstance(backend, str):
            raise ValueError("readout.projection_backend must be a string")
        result["projection_backend"] = normalize_indexed_projection_backend(backend)
    elif kind == "low_rank":
        result["rank"] = _integer(result.get("rank"), "readout.rank")
        if result["rank"] > output_dim:
            raise ValueError("readout.rank must be <= output_dim")
    return result


def _normalize_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise TypeError("model spec must be a dict")
    allowed = set(_DEFAULTS) | {
        "family",
        "input_dim",
        "output_dim",
        "topology_seed",
        "readout",
        "activation_init",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown model spec keys: {sorted(unknown)}")
    result = {**deepcopy(_DEFAULTS), **deepcopy(spec)}
    if result.get("family") not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    for name in ("input_dim", "output_dim", "width", "network_depth", "contacts_e"):
        result[name] = _integer(result.get(name), name)
    result["contacts_i"] = _integer(result["contacts_i"], "contacts_i", 0)
    result["seed"] = _integer(result["seed"], "seed", 0)
    result["topology_seed"] = _integer(
        result.get("topology_seed", result["seed"]), "topology_seed", 0
    )
    if not isinstance(result["branch_factors"], (list, tuple)):
        raise ValueError("branch_factors must be a sequence")
    result["branch_factors"] = [
        _integer(value, "branch factor") for value in result["branch_factors"]
    ]
    if not isinstance(result["activation"], str) or result["activation"] not in {
        "relu",
        "softplus",
        "sigmoid",
        "param_relu",
        "param_tanh",
    }:
        raise ValueError(
            "activation must be relu, softplus, sigmoid, param_relu, or param_tanh"
        )
    learned_gate = result["activation"] in {"param_relu", "param_tanh"}
    if learned_gate:
        result["activation_init"] = _normalize_activation_init(
            result.get("activation_init"), result["activation"]
        )
    elif "activation_init" in result:
        raise ValueError("activation_init requires param_relu or param_tanh")
    if result["initialization"] not in {"fan_in", "legacy_xavier"}:
        raise ValueError("initialization must be fan_in or legacy_xavier")
    if not isinstance(result["projection_backend"], str):
        raise ValueError("projection_backend must be a string")
    result["projection_backend"] = normalize_indexed_projection_backend(
        result["projection_backend"]
    )
    if result["family"] == "dense" and result["projection_backend"] != "eager":
        raise ValueError(
            "dense has no indexed projections; projection_backend must be eager"
        )
    if result["input_encoding"] not in {"signed_split", "nonnegative", "raw"}:
        raise ValueError("input_encoding must be signed_split, nonnegative, or raw")
    if result["input_encoding"] == "raw" and result["family"] in {
        "dendritic_additive",
        "dendritic_shunting",
    }:
        raise ValueError(
            "Positive E/I families require signed_split or nonnegative inputs; "
            "input_encoding='raw' is not a valid conductance-input contract"
        )
    if not isinstance(result["clip_contacts"], bool):
        raise ValueError("clip_contacts must be a bool")
    if result["family"] == "dendritic_signed" and result["contacts_i"]:
        raise ValueError("dendritic_signed has no I bank; set contacts_i=0")
    if "readout" in result:
        result["readout"] = _normalize_readout(result["readout"], result["output_dim"])
    return result


def _normalize_activation_init(settings: Any, activation: str) -> dict:
    """Require declared, fixed production gate initialization without calibration."""
    if not isinstance(settings, dict) or set(settings) != {"gain", "threshold"}:
        raise ValueError("activation_init must declare exactly gain and threshold")
    result = {}
    for name, value in settings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"activation_init.{name} must be a finite real number")
        result[name] = float(value)
    if result["gain"] <= 0:
        raise ValueError("activation_init.gain must be > 0")
    # ParametricReLU.initialize clamps smaller gains. Reject them rather than
    # let production initialization silently differ from conventional controls.
    if activation == "param_relu" and result["gain"] < 1e-8:
        raise ValueError("param_relu activation_init.gain must be >= 1e-8")
    return result


def _activation(
    name: str, output_dim: int, initialization: dict | None = None
) -> nn.Module:
    gain, threshold = 1.5, 0.5
    if initialization is not None:
        gain, threshold = initialization["gain"], initialization["threshold"]
    return ActivationFactory.create(
        name, output_dim=output_dim, init_m=gain, init_b=threshold
    )


def _compartments(branch_factors: list[int]) -> int:
    sites, level = 1, 1
    for factor in branch_factors:
        level *= factor
        sites += level
    return sites


def _contacts(requested: int, available: int, spec: dict) -> int:
    if requested > available and not spec["clip_contacts"]:
        raise ValueError(
            f"Requested {requested} contacts per output but only {available} "
            "sources are available; choose a feasible width/contact count or "
            "explicitly set clip_contacts=true."
        )
    return min(requested, available)


class _InputEncoding(nn.Module):
    def __init__(self, input_dim: int, kind: str):
        super().__init__()
        self.input_dim, self.kind = input_dim, kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, {self.input_dim}] inputs")
        if self.kind == "signed_split":
            return torch.cat((x.relu(), (-x).relu()), dim=-1)
        if self.kind == "raw":
            return x
        if bool((x < 0).any()):
            raise ValueError("input_encoding='nonnegative' received negative input")
        return x


class _SparseAffine(nn.Module):
    """Signed, fixed-index production projection plus an ordinary affine bias."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        contacts: int,
        seed: int,
        initialization: str,
        *,
        gate_projection: bool = False,
        projection_backend: str = "eager",
    ):
        super().__init__()
        fan_in = initialization == "fan_in"
        self.projection = IndexedSparseLinear(
            input_dim,
            output_dim,
            contacts,
            param_space="linear",
            weight_transform="identity",
            init_method="kaiming_normal" if fan_in else "xavier_normal",
            # Value projections use ReLU gain sqrt(2); sigmoid-gate logits
            # use gain 1. Both depend on contact fan-in, never output width.
            init_gain=1.0 / math.sqrt(2.0) if fan_in and gate_projection else 1.0,
            seed=seed,
            projection_backend=projection_backend,
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x) + self.bias


class _IndexedAffineReadout(nn.Module):
    """Signed production output projection with a counted optional affine bias."""

    def __init__(self, projection: IndexedSparseLinear, *, bias: bool):
        super().__init__()
        if (
            not isinstance(projection, IndexedSparseLinear)
            or projection.weight_transform != "identity"
        ):
            raise ValueError(
                "Indexed affine readout requires a signed indexed projection"
            )
        self.projection = projection
        self.bias = (
            nn.Parameter(projection.pre_w.new_zeros(projection.out_features))
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.projection(x)
        return output if self.bias is None else output + self.bias


def _build_readout(spec: dict) -> nn.Module:
    # Keep the historical call and RNG sequence unchanged when not opted in.
    if "readout" not in spec:
        return nn.Linear(spec["width"], spec["output_dim"])

    from dendritic_modeling.networks.architectures.replacement.projections import (
        make_output_projection,
    )

    readout = spec["readout"]
    indexed = readout["kind"] == "indexed"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            readout.get("parameter_seed", spec["seed"])
        )
        projection = make_output_projection(
            spec["width"],
            spec["output_dim"],
            rank=readout.get("rank"),
            topk=readout.get("topk"),
            topology_mode="indexed",
            bias=False if indexed else readout["bias"],
            init_std=readout["init_std"],
            seed=readout.get("topology_seed", spec["topology_seed"]),
            projection_backend=readout.get("projection_backend", "eager"),
            weight_transform="identity",
        )
        if indexed:
            # The factory's sparse initializer otherwise ignores init_std.
            # Match the signed normal per-stored-weight policy of its dense
            # and low-rank paths, while retaining its fixed support and kernel.
            nn.init.normal_(projection.pre_w, mean=0.0, std=readout["init_std"])
            return _IndexedAffineReadout(projection, bias=readout["bias"])
        return projection


class _GroupedGated(nn.Module):
    """Generic sparse gated subunits, group RMS normalization, and learned sum.

    This conventional comparator has no dendritic circuit dynamics. Each soma
    group contains as many parallel sites as the dendritic nonsomatic-site
    inventory (one for a flat morphology). Both sparse projections, biases,
    branch-combination weights, and output bias are counted. It is a bundled
    grouping/gating/normalization control, not an exact shunting equivalent.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        groups: int,
        contacts_e: int,
        contacts_i: int,
        activation: str,
        seed: int,
        initialization: str,
        *,
        projection_backend: str = "eager",
        activation_init: dict | None = None,
    ):
        super().__init__()
        sites = width * groups
        self.width, self.groups = width, groups
        self.value = _SparseAffine(
            input_dim,
            sites,
            contacts_e,
            seed,
            initialization,
            projection_backend=projection_backend,
        )
        self.gate = _SparseAffine(
            input_dim,
            sites,
            contacts_i,
            seed + 1,
            initialization,
            gate_projection=True,
            projection_backend=projection_backend,
        )
        self.activation = _activation(activation, sites, activation_init)
        self.combine = nn.Parameter(torch.empty(width, groups))
        nn.init.normal_(self.combine, std=groups**-0.5)
        self.bias = nn.Parameter(torch.zeros(width))
        self.output_activation = _activation(activation, width, activation_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = self.activation(self.value(x)) * torch.sigmoid(self.gate(x))
        branches = branches.reshape(x.shape[0], self.width, self.groups)
        # Include a unit reference so singleton groups remain amplitude-sensitive.
        branches = branches * torch.rsqrt(
            1.0 + branches.square().mean(-1, keepdim=True)
        )
        return self.output_activation((branches * self.combine).sum(-1) + self.bias)


def _dendritic_config(spec: dict, contacts_e: int, contacts_i: int, seed: int) -> dict:
    signed = spec["family"] == "dendritic_signed"
    activation_init = spec.get("activation_init", {"gain": 1.5, "threshold": 0.5})
    return {
        "architecture": {
            "excitatory_layer_sizes": [spec["width"]],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": spec["branch_factors"],
            "inhibitory_branch_factors": spec["branch_factors"],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [contacts_e],
            "ei_synapses_per_branch_per_layer": [0],
            "ie_synapses_per_branch_per_layer": [contacts_i],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "input_mode": 1,
            "independent_pathways": False,
            "output_activation": None,
            "allow_direct_inhibitory_stream": True,
        },
        "morphology": {
            "somatic_synapses": not bool(spec["branch_factors"]),
            "use_shunting": spec["family"] == "dendritic_shunting",
            "weight_transform": "identity" if signed else "softplus",
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "type": "indexed",
            "init_method": (
                "kaiming_normal"
                if spec["initialization"] == "fan_in"
                else "xavier_normal"
            ),
            "indexed": {"seed": seed, "projection_backend": spec["projection_backend"]},
        },
        "reactivation": {
            "enabled": True,
            "type": spec["activation"],
            # Fixed applies explicit m/b to learned gates at every branch and
            # soma. Historical parameter-free gates ignore these same values.
            "init_m": activation_init["gain"],
            "init_b": activation_init["threshold"],
            "init_policy": "fixed",
        },
        "blocklinear": {"efficient": True},
        "implementation": {"adaptive_initialization": False, "print_hooks": False},
        "biological_neuron": not signed,
    }


def _initialize_signed_dendritic_fan_in(module: nn.Module) -> dict[str, int]:
    """Reset production signed-final-init using actual contacts/children fan-in.

    The production identity-weight initializer reapplies Xavier after the
    projection constructors. Only existing signed parameters are reset here;
    circuit dynamics, support, and positive-conductance initialization remain
    those of the production modules.
    """
    fan_ins = {}
    for name, child in module.named_modules():
        if (
            isinstance(child, IndexedSparseLinear)
            and child.weight_transform == "identity"
        ):
            nn.init.kaiming_normal_(child.pre_w)
            fan_ins[name + ".pre_w"] = child.K
        elif isinstance(child, BlockLinear) and child.weight_transform == "identity":
            nn.init.kaiming_normal_(child.log_weight)
            fan_ins[name + ".log_weight"] = child.block_size
    return fan_ins


class _ScalingModel(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.scaling_spec = deepcopy(spec)
        self.encoding = _InputEncoding(spec["input_dim"], spec["input_encoding"])
        encoded_dim = spec["input_dim"] * (
            2 if spec["input_encoding"] == "signed_split" else 1
        )
        self.encoded_input_dim = encoded_dim
        self.layer_receipts: list[dict] = []
        layers: list[nn.Module] = []
        input_dim = encoded_dim
        family, width = spec["family"], spec["width"]
        for index in range(spec["network_depth"]):
            support_seed = spec["topology_seed"] + index * 104729
            receipt = {"layer": index, "input_dim": input_dim, "output_dim": width}
            if family == "dense":
                affine = nn.Linear(input_dim, width)
                if spec["initialization"] == "fan_in":
                    nn.init.kaiming_normal_(affine.weight)
                    nn.init.zeros_(affine.bias)
                layer = nn.Sequential(
                    affine,
                    _activation(spec["activation"], width, spec.get("activation_init")),
                )
                receipt.update({"contacts_e": input_dim, "contacts_i": 0})
            elif family == "sparse":
                contacts = _contacts(
                    spec["contacts_e"] + spec["contacts_i"], input_dim, spec
                )
                layer = nn.Sequential(
                    _SparseAffine(
                        input_dim,
                        width,
                        contacts,
                        support_seed,
                        spec["initialization"],
                        projection_backend=spec["projection_backend"],
                    ),
                    _activation(spec["activation"], width, spec.get("activation_init")),
                )
                receipt.update({"contacts_e": contacts, "contacts_i": 0})
            else:
                contacts_e = _contacts(spec["contacts_e"], input_dim, spec)
                contacts_i = _contacts(spec["contacts_i"], input_dim, spec)
                if family == "grouped_gated":
                    # A zero I count does not remove the generic gate; it reuses
                    # the declared value fan-in, with its full cost recorded.
                    contacts_i = contacts_i or contacts_e
                    groups = max(1, _compartments(spec["branch_factors"]) - 1)
                    layer = _GroupedGated(
                        input_dim,
                        width,
                        groups,
                        contacts_e,
                        contacts_i,
                        spec["activation"],
                        support_seed,
                        spec["initialization"],
                        projection_backend=spec["projection_backend"],
                        activation_init=spec.get("activation_init"),
                    )
                    receipt["parallel_groups"] = groups
                else:
                    config = _dendritic_config(
                        spec, contacts_e, contacts_i, support_seed
                    )
                    layer = ConfigurableEINetwork(config=config, input_dim=input_dim)
                    receipt["production_config"] = config
                    if (
                        family == "dendritic_signed"
                        and spec["initialization"] == "fan_in"
                    ):
                        receipt["final_signed_parameter_fan_ins"] = (
                            _initialize_signed_dendritic_fan_in(layer)
                        )
                    elif family in {"dendritic_additive", "dendritic_shunting"}:
                        receipt["final_positive_initialization"] = (
                            "production_analytical_expectation"
                        )
                receipt.update({"contacts_e": contacts_e, "contacts_i": contacts_i})
            layers.append(layer)
            self.layer_receipts.append(receipt)
            input_dim = width
        self.hidden = nn.Sequential(*layers)
        self.readout = _build_readout(spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden(self.encoding(x)))


def build_model(spec: dict) -> nn.Module:
    """Instantiate a seeded whole model without perturbing the caller's RNG.

    ``seed`` controls initial parameters; ``topology_seed`` defaults to it and
    controls static support. Defaults use a common signed-split input encoding
    and ReLU at every hidden point/branch. Native-input point controls may opt
    into ``input_encoding='raw'``; raw input is also supported by signed trees
    and generic grouped gates. Supported activations are nonnegative
    so successive positive-conductance layers receive valid inputs. Models are
    created on CPU; callers select device/dtype explicitly afterward.

    Learned ``param_relu`` and ``param_tanh`` gates require an explicit
    ``activation_init={"gain": m, "threshold": b}``. The positive gain is m,
    not its stored logarithm: gates evaluate m*relu(V-b) or
    (tanh(m*(V-b))+1)/2. Both log(m) and b are learned and counted at every
    hidden gate, including dendritic somas. Conventional controls use the same
    production activation classes and settings. Initialization is fixed with
    no implicit calibration; the caller must choose a threshold appropriate
    to its voltage regime. For param_relu, gain must be >=1e-8 to avoid the
    production initializer's floor. Omitted options preserve historical specs,
    state, and reports. Gate choices do not promise matched initial body weights.

    ``initialization='fan_in'`` uses the registered production Kaiming-normal
    sparse initializer, whose preweight variance is 2/K rather than 2/(K+width).
    Signed dendritic projections and child couplings are reset after production
    initialization using contacts/children as their row fan-in.
    Dense hidden values use the same fan-in rule; grouped gate logits use gain
    one. With no ``readout`` field, the affine readout retains ordinary PyTorch
    fan-in initialization and the exact historical state/RNG sequence.
    Positive weights retain the production analytical initializer, which
    overwrites their constructor weights, and the softplus transform.
    This removes a width confound, not all differences
    in conditioning across morphologies or integration rules. The archival
    ``legacy_xavier`` option reproduces this harness's original initialization.

    ``projection_backend`` selects the production indexed-projection backend,
    defaulting to ``eager``. It applies to every sparse value/gate/E/I bank;
    dense models accept only the default. Explicit accelerator backends are
    constructed on CPU but require a supported device at forward time. Backend
    changes preserve parameters and support, not floating-point reduction order.

    An optional ``readout`` dict selects signed production output allocation:
    ``kind='dense'``, ``kind='indexed', topk=K``, or ``kind='low_rank', rank=r``.
    All default to a counted affine bias and normal initialization with
    ``init_std=0.02`` per stored weight (per factor for low rank). Thus explicit
    dense initialization differs from the legacy omitted-field path. No
    activation is inserted between low-rank factors. Optional ``parameter_seed``
    isolates readout initialization from hidden-layer RNG consumption; if
    omitted it inherits ``seed``. Indexed readouts additionally accept
    ``topology_seed`` (inherits the core topology seed) and ``projection_backend``
    (defaults to eager independently of core backend). Their fixed indices are
    stored buffers, never learned candidate weights. Readout topk/rank are
    never clipped by ``clip_contacts``.
    """
    resolved = _normalize_spec(spec)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(resolved["seed"])
        return _ScalingModel(resolved)


def _dendrinet_inventory(model: nn.Module) -> dict | None:
    """Inspect actual production populations; never infer counts from a label.

    Depth zero is the soma, whereas ``forward_index`` runs from leaves to the
    soma. A contact means a stored indexed synaptic weight, not a dense masked
    candidate. Parameter roles form an exclusive partition of the whole model;
    a tied parameter with multiple roles is counted once under ``shared``.
    Registered buffers are reported separately from learned parameters and may
    include mutable runtime state, so they are not all called fixed topology.
    """
    dendrinets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DendriNet)
    ]
    if not dendrinets:
        return None

    role_names = (
        "excitatory_synapses",
        "inhibitory_synapses",
        "recurrent_synapses",
        "child_couplings",
        "reactivation",
        "readout",
        "shared",
        "other",
    )
    roles: dict[int, set[str]] = {}

    def assign_role(module: nn.Module | None, role: str) -> None:
        if module is not None:
            for parameter in module.parameters():
                roles.setdefault(id(parameter), set()).add(role)

    def parameter_count(module: nn.Module | None) -> int:
        return 0 if module is None else sum(p.numel() for p in module.parameters())

    def bank_receipt(bank: nn.Module | None) -> dict | None:
        if bank is None:
            return None
        result = {
            "module_type": type(bank).__name__,
            "total_parameters": parameter_count(bank),
            "input_dim": int(bank.in_features),
            "output_dim": int(bank.out_features),
            "weight_transform": getattr(bank, "weight_transform", None),
        }
        if isinstance(bank, IndexedSparseLinear):
            indices = bank.connection_indices.detach().cpu().contiguous()
            result.update(
                {
                    "contacts_per_compartment": int(bank.K),
                    "stored_contacts": int(bank.pre_w.numel()),
                    "connection_indices_shape": list(indices.shape),
                    "connection_indices_bytes": indices.numel()
                    * indices.element_size(),
                    "connection_indices_sha256": hashlib.sha256(
                        indices.numpy().tobytes()
                    ).hexdigest(),
                    "support_policy": "static_indexed",
                }
            )
        return result

    populations = []
    for name, dendrinet in dendrinets:
        depths = []
        for forward_index, branch in enumerate(dendrinet.branch_layers):
            excitation = getattr(branch, "branch_excitation", None)
            inhibition = getattr(branch, "branch_inhibition", None)
            coupling = getattr(branch, "branches_to_output", None)
            gate = getattr(branch, "reactivation", None)
            assign_role(excitation, "excitatory_synapses")
            assign_role(inhibition, "inhibitory_synapses")
            assign_role(coupling, "child_couplings")
            assign_role(gate, "reactivation")
            for recurrent_name in ("branch_recurrent", "branch_rec_inhibition"):
                assign_role(getattr(branch, recurrent_name, None), "recurrent_synapses")
            excitation_receipt = bank_receipt(excitation)
            inhibition_receipt = bank_receipt(inhibition)
            depths.append(
                {
                    "module_name": f"{name}.branch_layers.{forward_index}",
                    "forward_index": forward_index,
                    "depth_from_soma": int(branch.layer_idx),
                    "is_soma": branch.layer_idx == 0,
                    "compartment_count": int(branch.n_branches),
                    "compartments_per_soma": branch.n_branches // dendrinet.n_soma,
                    "contacts_e_per_compartment": (
                        excitation_receipt.get("contacts_per_compartment")
                        if excitation_receipt is not None
                        else 0
                    ),
                    "contacts_i_per_compartment": (
                        inhibition_receipt.get("contacts_per_compartment")
                        if inhibition_receipt is not None
                        else 0
                    ),
                    "input_banks": {
                        "excitatory": excitation_receipt,
                        "inhibitory": inhibition_receipt,
                    },
                    "input_synapse_parameters": parameter_count(excitation)
                    + parameter_count(inhibition),
                    "children_per_compartment": (
                        int(coupling.block_size) if coupling is not None else 0
                    ),
                    "child_coupling_parameters": parameter_count(coupling),
                    "gate_enabled": bool(branch.reactivate),
                    "gate_type": type(gate).__name__ if gate is not None else None,
                    "gate_parameters": parameter_count(gate),
                    "integration": "shunting" if branch.use_shunting else "additive",
                    "additive_mode": branch.additive_mode,
                    "weight_transform": branch.weight_transform,
                    "total_parameters": parameter_count(branch),
                }
            )
        contacted_depths = [
            d["depth_from_soma"] for d in depths if d["input_synapse_parameters"] > 0
        ]
        expected_nonsomatic = list(range(dendrinet.n_branch_layers, 0, -1))
        if contacted_depths == [0]:
            placement = "soma_only"
        elif contacted_depths == expected_nonsomatic and expected_nonsomatic:
            placement = "all_nonsomatic_depths"
        elif contacted_depths == [*expected_nonsomatic, 0]:
            placement = "all_depths_including_soma"
        else:
            placement = "explicit_contacted_depths"
        populations.append(
            {
                "module_name": name,
                "module_type": f"{type(dendrinet).__module__}.{type(dendrinet).__name__}",
                "n_soma": int(dendrinet.n_soma),
                "branch_factors": list(dendrinet.branch_factors),
                "internal_branch_depth": int(dendrinet.n_branch_layers),
                "compartment_count": sum(d["compartment_count"] for d in depths),
                "contact_placement": placement,
                "contacted_depths_from_soma": contacted_depths,
                "total_parameters": parameter_count(dendrinet),
                "depths": depths,
            }
        )

    if isinstance(model, _ScalingModel):
        assign_role(model.readout, "readout")
    partition = {role: 0 for role in role_names}
    trainable_partition = {role: 0 for role in role_names}
    parameter_roles = {}
    for name, parameter in model.named_parameters():
        candidates = roles.get(id(parameter), set())
        role = next(iter(candidates)) if len(candidates) == 1 else (
            "shared" if candidates else "other"
        )
        partition[role] += parameter.numel()
        if parameter.requires_grad:
            trainable_partition[role] += parameter.numel()
        parameter_roles[name] = role

    buffers = []
    for name, buffer in model.named_buffers():
        owner_name, _, local_name = name.rpartition(".")
        owner = model.get_submodule(owner_name) if owner_name else model
        buffers.append(
            {
                "name": name,
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
                "numel": buffer.numel(),
                "bytes": buffer.numel() * buffer.element_size(),
                "persistent": local_name not in owner._non_persistent_buffers_set,
            }
        )
    somas = sum(p["n_soma"] for p in populations)
    compartments = sum(p["compartment_count"] for p in populations)
    return {
        "schema_version": 1,
        "population_count": len(populations),
        "total_somas": somas,
        "total_compartments": compartments,
        "total_nonsomatic_compartments": compartments - somas,
        "parameter_partition": partition,
        "trainable_parameter_partition": trainable_partition,
        "parameter_roles": parameter_roles,
        "registered_buffer_inventory": buffers,
        "populations": populations,
        "interpretation": (
            "Observed registered architecture and storage, not effective function "
            "dimension or active nonlinear-region count. Configured ReLU gates "
            "can act linearly on nonnegative branch voltages."
        ),
    }


def model_report(model: nn.Module) -> dict:
    """Count unique registered scalars and storage, including the affine head."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    topology = hashlib.sha256()
    topology_bytes = 0
    for name, buffer in model.named_buffers():
        if not buffer.is_floating_point() and not buffer.is_complex():
            topology.update(name.encode())
            topology.update(str(tuple(buffer.shape)).encode())
            topology.update(buffer.detach().cpu().contiguous().numpy().tobytes())
            topology_bytes += buffer.numel() * buffer.element_size()
    report = {
        "total_parameters": sum(p.numel() for p in parameters),
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
        "topology_bytes": topology_bytes,
        "topology_sha256": topology.hexdigest(),
        "parameter_shapes": {
            name: list(p.shape) for name, p in model.named_parameters()
        },
        "indexed_projection_backends": {
            name: {
                "requested": module.projection_backend,
                "resolved": module._last_resolved_projection_backend,
                "device": (
                    str(module._last_projection_device)
                    if module._last_projection_device is not None
                    else None
                ),
            }
            for name, module in model.named_modules()
            if isinstance(module, IndexedSparseLinear)
        },
    }
    dendrinet_inventory = _dendrinet_inventory(model)
    if dendrinet_inventory is not None:
        report["dendrinet_inventory"] = dendrinet_inventory
    if isinstance(model, _ScalingModel):
        spec = deepcopy(model.scaling_spec)
        report.update(
            {
                "resolved_spec": spec,
                "family": spec["family"],
                "input_encoding": spec["input_encoding"],
                "initialization": spec["initialization"],
                "projection_backend": spec["projection_backend"],
                "encoded_input_dim": model.encoded_input_dim,
                "hidden_parameters": sum(p.numel() for p in model.hidden.parameters()),
                "readout_parameters": sum(
                    p.numel() for p in model.readout.parameters()
                ),
                "layer_receipts": deepcopy(model.layer_receipts),
                "support_policy": (
                    "static_indexed" if spec["family"] != "dense" else "dense"
                ),
                "compartments_per_soma": (
                    _compartments(spec["branch_factors"])
                    if spec["family"].startswith("dendritic_")
                    else None
                ),
            }
        )
        if "readout" in spec:
            readout = deepcopy(spec["readout"])
            readout_inventory = model_report(model.readout)
            readout["parameter_seed"] = readout.get("parameter_seed", spec["seed"])
            referenced_columns = spec["width"]
            if readout["kind"] == "indexed":
                readout["topology_seed"] = readout.get(
                    "topology_seed", spec["topology_seed"]
                )
                referenced_columns = int(
                    model.readout.projection.connection_indices.unique().numel()
                )
            readout.update(
                {
                    "factory": "replacement.projections.make_output_projection",
                    "weight_transform": "identity",
                    "initialization": "normal_per_stored_weight",
                    "input_dim": spec["width"],
                    "output_dim": spec["output_dim"],
                    "total_parameters": report["readout_parameters"],
                    "parameter_bytes": readout_inventory["parameter_bytes"],
                    "buffer_bytes": readout_inventory["buffer_bytes"],
                    "topology_bytes": readout_inventory["topology_bytes"],
                    "topology_sha256": readout_inventory["topology_sha256"],
                    "referenced_input_columns": referenced_columns,
                    "unreferenced_input_columns": spec["width"] - referenced_columns,
                    "support_policy": (
                        "static_indexed" if readout["kind"] == "indexed" else "dense"
                    ),
                    "learned_factors": 2 if readout["kind"] == "low_rank" else 1,
                }
            )
            report["readout"] = readout
            if readout["kind"] == "indexed":
                report["support_policy"] = "static_indexed"
    return report


def match_parameter_budget(
    spec: dict, target_parameters: int, tolerance: float = 0.02
) -> tuple[dict, dict]:
    """Find the nearest feasible integer width by counted monotone search.

    All fields other than width remain fixed. A match is accepted only when the
    instantiated *whole-model* count differs from target by at most tolerance.
    Failure is explicit; neither extra inert weights nor an over-budget-only
    convention is used. Matching does not equalize compute or activation memory.
    """
    target = _integer(target_parameters, "target_parameters")
    if (
        isinstance(tolerance, bool)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 1
    ):
        raise ValueError("tolerance must be finite and in [0, 1)")
    resolved = _normalize_spec(spec)
    minimum_width = 1
    if resolved["network_depth"] > 1 and not resolved["clip_contacts"]:
        if resolved["family"] == "sparse":
            minimum_width = resolved["contacts_e"] + resolved["contacts_i"]
        elif resolved["family"] != "dense":
            minimum_width = max(resolved["contacts_e"], resolved["contacts_i"])
    readout = resolved.get("readout", {})
    minimum_width = max(minimum_width, readout.get("topk", 1), readout.get("rank", 1))
    cache: dict[int, dict] = {}

    def count(width: int) -> int:
        if width not in cache:
            candidate = {**resolved, "width": width}
            cache[width] = model_report(build_model(candidate))
        return cache[width]["total_parameters"]

    low = minimum_width
    low_count = count(low)
    high = low
    if low_count < target:
        high = max(low + 1, resolved["width"])
        while count(high) < target:
            low = high
            high *= 2
        while high - low > 1:
            middle = (low + high) // 2
            if count(middle) < target:
                low = middle
            else:
                high = middle
    best = min(
        {low, high}, key=lambda width: (abs(count(width) - target), count(width))
    )
    report = cache[best]
    error = abs(report["total_parameters"] - target) / target
    if error > tolerance:
        raise ValueError(
            f"No width matches target_parameters={target} within tolerance={tolerance:g}; "
            f"nearest feasible width={best} has {report['total_parameters']} parameters "
            f"(relative error={error:.6g})."
        )
    report.update(
        {
            "target_parameters": target,
            "budget_tolerance": float(tolerance),
            "budget_relative_error": error,
            "budget_matched": True,
        }
    )
    return deepcopy(report["resolved_spec"]), report


__all__ = ["FAMILIES", "build_model", "match_parameter_budget", "model_report"]
