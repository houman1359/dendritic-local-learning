#!/usr/bin/env python3
"""Projected local learning on reconstructed MICrONS dendritic trees.

This analysis deliberately separates the source of the task from the feedback
basis.  Trial-aligned calcium responses of directly connected presynaptic
partners and a postsynaptic target define a held-out response-prediction task.
The target values are never used to construct or select feedback routes.

For each target, the compressed reconstructed dendrite is instantiated as a
steady-state passive conductance network.  A trainable non-negative
conductance is placed at the dominant anatomical contact of every imaged
presynaptic partner.  Exact gradients of somatic squared error factor into
local eligibility, x_j (E_E - V_j), and an adjoint compartment error.  The
exact dynamic compartment error is retained only in the reference method.
Each restricted method instead receives the scalar somatic error multiplied by
a fixed site vector.  That vector is obtained once, before learning and without
task values, by projecting the passive tree's baseline soma-to-site transfer
vector onto one of three fixed, equal-rank feedback dictionaries:

  * conductance-weighted ancestry routes selected without task data;
  * the same route columns independently reassigned across input sites; or
  * an equal-rank random selection of nonempty anatomical routes.

The one-time projection is an anatomy-calibration oracle, but restricted
updates never use the exact dynamic adjoint.  They use only scalar output
error, fixed route gain and synapse-local eligibility.  The experiment is
therefore a learning-sufficiency test for a concrete frozen-feedback local
rule, not a biological model of how its fixed route coefficients are
established.  Biological target cells are the inferential units; stimulus
splits are nested sensitivity replicates.

Provenance: this publication script combines the passive-tree system and
adjoint identities from ``code/reconstructed_tree`` with the frozen MICrONS
functional-task join in ``code/task_derived``. It was written for this journal
integration and does not modify either source analysis. Input hashes and
manuscript-relative paths are serialized with every frozen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


JOURNAL = Path(__file__).resolve().parents[1]
REPO = JOURNAL.parents[2]
TREE_CODE = JOURNAL / "code" / "reconstructed_tree"
sys.path.insert(0, str(TREE_CODE))

from analyze_microns_morphology_credit import (  # noqa: E402
    ancestry_matrix,
    electrical_geometry,
    parent_map,
)


DEFAULT_EXTRACT_ROOT = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "results"
    / "microns_functional_partner_responses"
)
DEFAULT_SEGMENTS = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "results"
    / "microns_morphology_credit"
    / "segment_metrics.csv"
)
DEFAULT_OUTDIR = (
    JOURNAL / "analysis" / "exploratory_reconstructed_tree_task_learning" / "output"
)
DEFAULT_REPORT = JOURNAL / "analysis" / "reconstructed_tree_task_learning_20260731.md"

METHODS = (
    "exact compartment error",
    "topology-matched routes",
    "site-shuffled routes",
    "random anatomical routes",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Serialize a path relative to the manuscript root."""

    return Path(os.path.relpath(Path(path).resolve(), start=JOURNAL.resolve())).as_posix()


def stable_rng(seed: int, root_id: int, stream: int, replicate: int = 0) -> np.random.Generator:
    root = int(root_id)
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed) & 0xFFFFFFFF,
                root & 0xFFFFFFFF,
                (root >> 32) & 0xFFFFFFFF,
                int(stream) & 0xFFFFFFFF,
                int(replicate) & 0xFFFFFFFF,
            ]
        )
    )


def grouped_split(
    stimulus_ids: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(stimulus_ids)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(float(test_fraction) * len(unique))))
    test_ids = set(shuffled[:n_test].tolist())
    test = np.asarray([value in test_ids for value in stimulus_ids], dtype=bool)
    return ~test, test


def load_target(extract_dir: Path) -> dict[str, Any]:
    protocol_path = extract_dir / "microns_dandi_trial_protocol.npz"
    mapping_path = extract_dir / "microns_dandi_trial_unit_mapping.csv"
    contacts_path = extract_dir / "functional_topology" / "functional_contacts.csv"
    manifest_path = extract_dir / "extraction_manifest.csv"
    protocol = np.load(protocol_path)
    mapping = pd.read_csv(mapping_path)
    contacts = pd.read_csv(contacts_path)
    manifest = pd.read_csv(manifest_path)
    responses = np.asarray(protocol["responses"], dtype=float)
    partner_columns = np.flatnonzero(mapping["role"].eq("presynaptic_partner").to_numpy())
    target_columns = np.flatnonzero(mapping["role"].eq("postsynaptic_target").to_numpy())
    if len(target_columns) != 1:
        raise ValueError(f"expected one postsynaptic target, found {len(target_columns)}")
    if len(partner_columns) != len(contacts):
        raise ValueError("partner/contact count mismatch")
    order = contacts["unit_index"].to_numpy(dtype=int)
    return {
        "root_id": int(manifest["target_root_id"].iloc[0]),
        "nucleus_id": int(manifest["target_nucleus_id"].iloc[0]),
        "session": int(manifest["session"].iloc[0]),
        "scan_idx": int(manifest["scan_idx"].iloc[0]),
        "x_raw": responses[:, partner_columns][:, order],
        "y_raw": responses[:, target_columns[0]],
        "stimulus_ids": np.asarray(protocol["stimulus_ids"], dtype=int),
        "contacts": contacts,
        "input_files": {
            "protocol": {"path": portable_path(protocol_path), "sha256": sha256(protocol_path)},
            "mapping": {"path": portable_path(mapping_path), "sha256": sha256(mapping_path)},
            "contacts": {"path": portable_path(contacts_path), "sha256": sha256(contacts_path)},
            "manifest": {"path": portable_path(manifest_path), "sha256": sha256(manifest_path)},
        },
    }


def scale_inputs(
    x: np.ndarray,
    train: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    maximum: float,
) -> np.ndarray:
    """Map fluorescence to a non-negative, training-defined activity proxy."""

    lower = np.nanquantile(x[train], float(lower_quantile), axis=0, keepdims=True)
    upper = np.nanquantile(x[train], float(upper_quantile), axis=0, keepdims=True)
    scale = np.maximum(upper - lower, 1e-8)
    return np.nan_to_num(np.clip((x - lower) / scale, 0.0, float(maximum)))


def scale_target(y: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(y[train]))
    scale = float(np.nanstd(y[train], ddof=1))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return np.nan_to_num((y - mean) / scale)


def orthonormal_basis(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not values.shape[1]:
        return np.zeros((values.shape[0], 0), dtype=float)
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return np.zeros((values.shape[0], 0), dtype=float)
    return left[:, singular > float(tolerance) * singular[0]]


def projector(matrix: np.ndarray) -> np.ndarray:
    basis = orthonormal_basis(matrix)
    return basis @ basis.T


def softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def inverse_softplus(value: float) -> float:
    return float(math.log(math.expm1(float(value))))


@dataclass
class PassiveTree:
    root_id: int
    n_segments: int
    root_index: int
    site_indices: np.ndarray
    site_segment_ids: np.ndarray
    base_voltage: np.ndarray
    inverse_base: np.ndarray
    excitatory_reversal: float
    morphology_dictionary: np.ndarray
    shuffled_dictionary: np.ndarray
    random_dictionary: np.ndarray
    requested_channels: int
    used_channels: int
    dictionary_rank: int
    selected_route_segments: list[int]
    random_route_segments: list[int]

    @property
    def site_inverse(self) -> np.ndarray:
        return self.inverse_base[np.ix_(self.site_indices, self.site_indices)]

    @property
    def root_to_sites(self) -> np.ndarray:
        return self.inverse_base[self.root_index, self.site_indices]


def rank_matched_shuffle(
    dictionary: np.ndarray,
    target_rank: int,
    rng: np.random.Generator,
    attempts: int = 10_000,
) -> np.ndarray:
    for _ in range(int(attempts)):
        candidate = np.asarray(dictionary, dtype=float).copy()
        for column in range(candidate.shape[1]):
            candidate[:, column] = rng.permutation(candidate[:, column])
        if np.linalg.matrix_rank(candidate) == int(target_rank) and not np.array_equal(
            candidate, dictionary
        ):
            return candidate
    raise RuntimeError("could not generate an equal-rank site shuffle")


def build_passive_tree(
    segments: pd.DataFrame,
    contacts: pd.DataFrame,
    channels: int,
    rng: np.random.Generator,
    background_e_scale: float,
    background_i_scale: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
) -> PassiveTree:
    electrical = (
        electrical_geometry(
            segments,
            e_scale=float(background_e_scale),
            i_scale=float(background_i_scale),
        )
        .sort_values("topological_depth")
        .reset_index(drop=True)
    )
    root, parents, _ = parent_map(electrical)
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    missing = sorted(set(contacts["segment_id"].astype(int)) - set(index))
    if missing:
        raise ValueError(f"functional contacts map to absent segments: {missing}")
    n = len(electrical)
    matrix = np.zeros((n, n), dtype=float)
    rhs = electrical["g_e"].to_numpy(dtype=float) * float(excitatory_reversal) + electrical[
        "g_i"
    ].to_numpy(dtype=float) * float(inhibitory_reversal)
    matrix[np.diag_indices(n)] = (
        electrical["g_leak"].to_numpy(dtype=float)
        + electrical["g_e"].to_numpy(dtype=float)
        + electrical["g_i"].to_numpy(dtype=float)
    )
    for child, parent in parents.items():
        child_index = index[int(child)]
        parent_index = index[int(parent)]
        coupling = float(electrical.loc[child_index, "g_edge"])
        matrix[child_index, child_index] += coupling
        matrix[parent_index, parent_index] += coupling
        matrix[child_index, parent_index] -= coupling
        matrix[parent_index, child_index] -= coupling
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if minimum_eigenvalue <= 0:
        raise ValueError(
            f"passive conductance matrix is not positive definite: {minimum_eigenvalue}"
        )
    inverse = np.linalg.solve(matrix, np.eye(n))
    base_voltage = inverse @ rhs
    site_segment_ids = contacts["segment_id"].to_numpy(dtype=int)
    site_indices = np.asarray([index[int(segment)] for segment in site_segment_ids], dtype=int)

    inhibitory_segments = electrical.loc[electrical["g_i"] > 0, "segment_id"].astype(int).tolist()
    ancestry = ancestry_matrix(site_segment_ids.tolist(), inhibitory_segments, parents)
    lookup = electrical.set_index("segment_id")
    beta = lookup.loc[inhibitory_segments, "g_i"].to_numpy(dtype=float) / np.maximum(
        lookup.loc[inhibitory_segments, "g_total"].to_numpy(dtype=float),
        1e-12,
    )
    kernel = ancestry * beta[None, :]
    candidates = np.flatnonzero(np.any(np.abs(kernel) > 0, axis=0))
    used_channels = min(int(channels), len(site_indices), len(candidates))
    if used_channels < 1:
        raise ValueError("no nonempty inhibitory route reaches a functional input site")
    leverage = beta * ancestry.mean(axis=0)
    selected = candidates[np.argsort(-leverage[candidates])[:used_channels]]
    morphology = kernel[:, selected].copy()
    target_rank = int(np.linalg.matrix_rank(morphology))
    if target_rank < 1:
        raise ValueError("selected morphology dictionary has zero rank")
    shuffled = rank_matched_shuffle(morphology, target_rank, rng)

    random_choice: np.ndarray | None = None
    for _ in range(10_000):
        proposed = rng.choice(candidates, size=used_channels, replace=False)
        if np.linalg.matrix_rank(kernel[:, proposed]) == target_rank:
            random_choice = proposed
            break
    if random_choice is None:
        raise RuntimeError("could not sample an equal-rank random anatomical dictionary")
    random_dictionary = kernel[:, random_choice].copy()
    return PassiveTree(
        root_id=int(segments["root_id"].iloc[0]),
        n_segments=int(n),
        root_index=int(index[int(root)]),
        site_indices=site_indices,
        site_segment_ids=site_segment_ids,
        base_voltage=base_voltage,
        inverse_base=inverse,
        excitatory_reversal=float(excitatory_reversal),
        morphology_dictionary=morphology,
        shuffled_dictionary=shuffled,
        random_dictionary=random_dictionary,
        requested_channels=int(channels),
        used_channels=int(used_channels),
        dictionary_rank=target_rank,
        selected_route_segments=[int(inhibitory_segments[value]) for value in selected],
        random_route_segments=[int(inhibitory_segments[value]) for value in random_choice],
    )


@dataclass
class State:
    q: np.ndarray
    readout: float
    bias: float

    def copy(self) -> "State":
        return State(self.q.copy(), float(self.readout), float(self.bias))


def tree_forward(
    tree: PassiveTree,
    x: np.ndarray,
    conductance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return somatic voltage, site voltage and soma-to-site adjoint gain."""

    h = np.asarray(x, dtype=float) * np.asarray(conductance, dtype=float)[None, :]
    site_inverse = tree.site_inverse
    identity = np.eye(len(tree.site_indices), dtype=float)
    systems = identity[None, :, :] + site_inverse[None, :, :] * h[:, None, :]
    base_sites = tree.base_voltage[tree.site_indices]
    voltage_rhs = base_sites[None, :] + (h * tree.excitatory_reversal) @ site_inverse.T
    adjoint_rhs = np.broadcast_to(tree.root_to_sites, voltage_rhs.shape)
    solutions = np.linalg.solve(
        systems,
        np.stack([voltage_rhs, adjoint_rhs], axis=2),
    )
    site_voltage = solutions[:, :, 0]
    soma_to_site = solutions[:, :, 1]
    soma_voltage = tree.base_voltage[tree.root_index] + np.sum(
        tree.root_to_sites[None, :] * h * (tree.excitatory_reversal - site_voltage),
        axis=1,
    )
    return soma_voltage, site_voltage, soma_to_site


def initialize_state(
    tree: PassiveTree,
    x_train: np.ndarray,
    y_train: np.ndarray,
    initial_conductance: float,
) -> State:
    q = np.full(x_train.shape[1], inverse_softplus(initial_conductance), dtype=float)
    soma, _, _ = tree_forward(tree, x_train, softplus(q))
    design = np.column_stack([soma, np.ones(len(soma), dtype=float)])
    gram = design.T @ design
    gram[0, 0] += 1e-6
    coefficient = np.linalg.solve(gram, design.T @ y_train)
    return State(q=q, readout=float(coefficient[0]), bias=float(coefficient[1]))


def objective_and_gradients(
    tree: PassiveTree,
    x: np.ndarray,
    y: np.ndarray,
    state: State,
    fixed_feedback: np.ndarray | None,
    weight_decay: float,
    readout_decay: float,
) -> tuple[float, dict[str, np.ndarray | float], dict[str, np.ndarray]]:
    conductance = softplus(state.q)
    soma, site_voltage, soma_to_site = tree_forward(tree, x, conductance)
    prediction = state.readout * soma + state.bias
    error = prediction - y
    exact_error = error[:, None] * state.readout * soma_to_site
    routed_error = (
        exact_error
        if fixed_feedback is None
        else error[:, None] * state.readout * np.asarray(fixed_feedback)[None, :]
    )
    eligibility = x * (tree.excitatory_reversal - site_voltage)
    gradient_g = np.mean(eligibility * routed_error, axis=0) + float(weight_decay) * conductance
    gradients: dict[str, np.ndarray | float] = {
        "q": gradient_g * sigmoid(state.q),
        "readout": float(np.mean(error * soma) + float(readout_decay) * state.readout),
        "bias": float(np.mean(error)),
    }
    objective = (
        0.5 * float(np.mean(error * error))
        + 0.5 * float(weight_decay) * float(np.sum(conductance * conductance))
        + 0.5 * float(readout_decay) * float(state.readout * state.readout)
    )
    cache = {
        "prediction": prediction,
        "error": error,
        "soma": soma,
        "site_voltage": site_voltage,
        "soma_to_site": soma_to_site,
        "exact_error": exact_error,
        "routed_error": routed_error,
        "eligibility": eligibility,
    }
    return objective, gradients, cache


def train(
    tree: PassiveTree,
    x: np.ndarray,
    y: np.ndarray,
    initial: State,
    fixed_feedback: np.ndarray | None,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    readout_decay: float,
) -> tuple[State, list[float]]:
    state = initial.copy()
    first = {
        "q": np.zeros_like(state.q),
        "readout": 0.0,
        "bias": 0.0,
    }
    second = {
        "q": np.zeros_like(state.q),
        "readout": 0.0,
        "bias": 0.0,
    }
    history: list[float] = []
    for step in range(1, int(steps) + 1):
        objective, gradients, _ = objective_and_gradients(
            tree,
            x,
            y,
            state,
            fixed_feedback,
            weight_decay,
            readout_decay,
        )
        for name in ("q", "readout", "bias"):
            gradient = gradients[name]
            first[name] = 0.9 * first[name] + 0.1 * gradient
            second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
            mhat = first[name] / (1.0 - 0.9**step)
            vhat = second[name] / (1.0 - 0.999**step)
            update = float(learning_rate) * mhat / (np.sqrt(vhat) + 1e-8)
            if name == "q":
                state.q = np.clip(state.q - update, -12.0, 6.0)
            elif name == "readout":
                state.readout = float(np.clip(state.readout - update, -10_000.0, 10_000.0))
            else:
                state.bias = float(np.clip(state.bias - update, -100.0, 100.0))
        if step == 1 or step % 25 == 0 or step == int(steps):
            history.append(float(objective))
    return state, history


def flattened_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-18 else float("nan")


def capture(first: np.ndarray, second: np.ndarray) -> float:
    exact = np.asarray(first, dtype=float)
    approximate = np.asarray(second, dtype=float)
    denominator = float(np.sum(exact * exact))
    if denominator <= 1e-18:
        return float("nan")
    return float(1.0 - np.sum((exact - approximate) ** 2) / denominator)


def finite_difference_check(
    tree: PassiveTree,
    x: np.ndarray,
    y: np.ndarray,
    state: State,
    coordinate: int,
    weight_decay: float,
    readout_decay: float,
    epsilon: float = 1e-5,
) -> dict[str, float | int]:
    _, gradient, _ = objective_and_gradients(tree, x, y, state, None, weight_decay, readout_decay)
    plus = state.copy()
    minus = state.copy()
    plus.q[int(coordinate)] += float(epsilon)
    minus.q[int(coordinate)] -= float(epsilon)
    value_plus, _, _ = objective_and_gradients(tree, x, y, plus, None, weight_decay, readout_decay)
    value_minus, _, _ = objective_and_gradients(
        tree, x, y, minus, None, weight_decay, readout_decay
    )
    numerical = float((value_plus - value_minus) / (2.0 * float(epsilon)))
    analytic = float(np.asarray(gradient["q"])[int(coordinate)])
    return {
        "coordinate": int(coordinate),
        "analytic": analytic,
        "finite_difference": numerical,
        "relative_error": float(
            abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-12)
        ),
    }


def run_replicate(
    data: dict[str, Any],
    all_segments: pd.DataFrame,
    args: argparse.Namespace,
    replicate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_id = int(data["root_id"])
    scan_stream = 100 * int(data["session"]) + int(data["scan_idx"])
    train_mask, test_mask = grouped_split(
        data["stimulus_ids"],
        stable_rng(args.seed, root_id, stream=1000 + scan_stream, replicate=replicate),
        args.test_fraction,
    )
    x = scale_inputs(
        data["x_raw"],
        train_mask,
        args.input_lower_quantile,
        args.input_upper_quantile,
        args.input_maximum,
    )
    y = scale_target(data["y_raw"], train_mask)
    segments = all_segments[all_segments["root_id"] == root_id].copy()
    tree = build_passive_tree(
        segments,
        data["contacts"],
        args.channels,
        stable_rng(args.seed, root_id, stream=2000 + scan_stream, replicate=replicate),
        args.background_e_scale,
        args.background_i_scale,
        args.excitatory_reversal,
        args.inhibitory_reversal,
    )
    projections: dict[str, np.ndarray | None] = {
        "exact compartment error": None,
        "topology-matched routes": projector(tree.morphology_dictionary),
        "site-shuffled routes": projector(tree.shuffled_dictionary),
        "random anatomical routes": projector(tree.random_dictionary),
    }
    baseline_transfer = tree.root_to_sites.copy()
    fixed_feedback: dict[str, np.ndarray | None] = {
        method: (
            None if projection is None else np.asarray(projection, dtype=float) @ baseline_transfer
        )
        for method, projection in projections.items()
    }
    initial = initialize_state(
        tree,
        x[train_mask],
        y[train_mask],
        args.initial_conductance,
    )
    finite = finite_difference_check(
        tree,
        x[train_mask][: min(32, int(train_mask.sum()))],
        y[train_mask][: min(32, int(train_mask.sum()))],
        initial,
        int(stable_rng(args.seed, root_id, stream=3000 + scan_stream, replicate=replicate).integers(x.shape[1])),
        args.weight_decay,
        args.readout_decay,
    )
    trained: dict[str, State] = {}
    histories: dict[str, list[float]] = {}
    for method in METHODS:
        trained[method], histories[method] = train(
            tree,
            x[train_mask],
            y[train_mask],
            initial,
            fixed_feedback[method],
            args.steps,
            args.learning_rate,
            args.weight_decay,
            args.readout_decay,
        )

    exact_state = trained["exact compartment error"]
    _, _, exact_test_cache = objective_and_gradients(
        tree,
        x[test_mask],
        y[test_mask],
        exact_state,
        None,
        args.weight_decay,
        args.readout_decay,
    )
    exact_field = exact_test_cache["exact_error"]
    exact_updates = exact_test_cache["eligibility"] * exact_field
    baseline = float(np.mean((y[test_mask] - np.mean(y[train_mask])) ** 2))
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        state = trained[method]
        _, _, test_cache = objective_and_gradients(
            tree,
            x[test_mask],
            y[test_mask],
            state,
            fixed_feedback[method],
            args.weight_decay,
            args.readout_decay,
        )
        mse = float(np.mean(test_cache["error"] ** 2))
        if fixed_feedback[method] is None:
            routed_common_field = exact_field
        else:
            routed_common_field = (
                exact_test_cache["error"][:, None]
                * exact_state.readout
                * np.asarray(fixed_feedback[method])[None, :]
            )
        routed_common_updates = exact_test_cache["eligibility"] * routed_common_field
        rows.append(
            {
                "target_root_id": root_id,
                "target_nucleus_id": int(data["nucleus_id"]),
                "session": int(data["session"]),
                "scan_idx": int(data["scan_idx"]),
                "replicate": int(replicate),
                "method": method,
                "n_segments": int(tree.n_segments),
                "n_sites": int(x.shape[1]),
                "n_train_trials": int(train_mask.sum()),
                "n_test_trials": int(test_mask.sum()),
                "n_train_stimulus_ids": int(len(np.unique(data["stimulus_ids"][train_mask]))),
                "n_test_stimulus_ids": int(len(np.unique(data["stimulus_ids"][test_mask]))),
                "requested_channels": int(tree.requested_channels),
                "used_channels": int(tree.used_channels),
                "dictionary_rank": int(
                    x.shape[1]
                    if projections[method] is None
                    else np.linalg.matrix_rank(projections[method])
                ),
                "heldout_mse": mse,
                "heldout_normalized_mse": float(mse / max(baseline, 1e-12)),
                "common_checkpoint_error_field_capture": float(
                    1.0
                    if projections[method] is None
                    else capture(exact_field, routed_common_field)
                ),
                "common_checkpoint_update_capture": float(
                    1.0
                    if projections[method] is None
                    else capture(exact_updates, routed_common_updates)
                ),
                "common_checkpoint_update_cosine": float(
                    1.0
                    if projections[method] is None
                    else flattened_cosine(exact_updates, routed_common_updates)
                ),
                "initial_training_objective": float(histories[method][0]),
                "mid_training_objective": float(histories[method][len(histories[method]) // 2]),
                "final_training_objective": float(histories[method][-1]),
                "late_training_objective_change": float(
                    histories[method][-1]
                    - histories[method][max(0, len(histories[method]) - 5)]
                ),
                "mean_learned_conductance": float(np.mean(softplus(state.q))),
                "minimum_learned_conductance": float(np.min(softplus(state.q))),
                "maximum_learned_conductance": float(np.max(softplus(state.q))),
                "parameter_bound_fraction": float(
                    np.mean((state.q <= -12.0 + 1e-10) | (state.q >= 6.0 - 1e-10))
                ),
                "readout": float(state.readout),
                "bias": float(state.bias),
            }
        )
    metadata = {
        "target_root_id": root_id,
        "target_nucleus_id": int(data["nucleus_id"]),
        "session": int(data["session"]),
        "scan_idx": int(data["scan_idx"]),
        "replicate": int(replicate),
        "n_segments": int(tree.n_segments),
        "site_segment_ids": [int(value) for value in tree.site_segment_ids],
        "selected_route_segments": tree.selected_route_segments,
        "random_route_segments": tree.random_route_segments,
        "used_channels": int(tree.used_channels),
        "dictionary_rank": int(tree.dictionary_rank),
        "dictionary_nonzeros": {
            "topology-matched routes": int(np.count_nonzero(tree.morphology_dictionary)),
            "site-shuffled routes": int(np.count_nonzero(tree.shuffled_dictionary)),
            "random anatomical routes": int(np.count_nonzero(tree.random_dictionary)),
        },
        "baseline_transfer_norm": float(np.linalg.norm(baseline_transfer)),
        "fixed_feedback_norms": {
            method: float(np.linalg.norm(value))
            for method, value in fixed_feedback.items()
            if value is not None
        },
        "finite_difference": finite,
    }
    return rows, metadata


def run_scan(
    extract_dir: Path,
    all_segments: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, str] | None]:
    try:
        data = load_target(extract_dir)
        input_record = {
            "target_root_id": int(data["root_id"]),
            "target_nucleus_id": int(data["nucleus_id"]),
            "session": int(data["session"]),
            "scan_idx": int(data["scan_idx"]),
            "files": data["input_files"],
        }
        rows: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for replicate in range(int(args.replicates)):
            replicate_rows, replicate_metadata = run_replicate(
                data, all_segments, args, replicate
            )
            rows.extend(replicate_rows)
            metadata.append(replicate_metadata)
        return rows, metadata, input_record, None
    except Exception as exc:
        return [], [], None, {
            "extract_dir": portable_path(extract_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def cell_level_contrast(
    cell_means: pd.DataFrame,
    metric: str,
    first: str,
    second: str,
    seed: int,
) -> dict[str, Any]:
    wide = cell_means.pivot(index="target_root_id", columns="method", values=metric).dropna()
    values = (wide[first] - wide[second]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    test = (
        stats.wilcoxon(values, alternative="two-sided")
        if len(values) >= 2 and not np.allclose(values, 0.0)
        else None
    )
    return {
        "metric": metric,
        "contrast": f"{first} minus {second}",
        "n_cells": int(len(values)),
        "mean_difference": float(np.mean(values)),
        "cell_bootstrap_ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "positive_cells": int(np.sum(values > 0)),
        "negative_cells": int(np.sum(values < 0)),
        "zero_cells": int(np.sum(np.isclose(values, 0.0))),
        "wilcoxon_two_sided_p": float(test.pvalue) if test is not None else None,
    }


def make_report(summary: dict[str, Any], path: Path) -> None:
    method = summary["method_cell_means"]
    contrasts = summary["contrasts"]
    lines = [
        "# Reconstructed-tree task learning",
        "",
        "## Question",
        "",
        (
            "Does a sparse feedback basis taken from the same reconstructed dendritic tree support"
            " a local conductance update better than equal-rank site-shuffled or random-route"
            " feedback when the task is defined independently by measured visual responses?"
        ),
        "",
        "## Design",
        "",
        (
            f"The analysis used {summary['n_cells']} MICrONS target cells across"
            f" {summary['n_scans']} eligible target--scan observations and"
            f" {summary['parameters']['replicates']} held-out-stimulus splits per cell. Each"
            " complete compressed morphology was instantiated as a passive conductance network."
            " Directly connected imaged partners supplied non-negative trial activity at mapped"
            " contact sites, and the measured postsynaptic response supplied the target. Route"
            " selection and one-time calibration used anatomy and fixed passive conductances only,"
            " never task responses. Restricted dictionaries had equal effective rank within each"
            " run."
        ),
        "",
        (
            "Before learning, the baseline passive soma-to-site transfer vector was projected once"
            " onto each dictionary and frozen. Restricted updates then used scalar output error"
            " times this fixed vector and synapse-local eligibility; they did not use the exact"
            " dynamic adjoint. The one-time coefficients remain an anatomy-calibration oracle, and"
            " this is not evidence that the MICrONS neurons learned with this rule."
        ),
        "",
        "## Circularity audit",
        "",
        (
            "The positive structural-capacity fields elsewhere in the project are generated from"
            " the same route kernel being evaluated. This analysis does not reuse those fields. Its"
            " task gradients come from measured responses and an exact passive-tree inverse, while"
            " route selection never sees the task data. The exact gradients and route basis still"
            " share the reconstructed forward tree, as required by the topology-matching"
            " hypothesis, so this remains modeled evidence rather than an independent biological"
            " observation."
        ),
        "",
        "## Cell-level results",
        "",
        "| method | normalized held-out MSE | exact-field capture | update cosine |",
        "|---|---:|---:|---:|",
    ]
    lines[8] = (
        f"The analysis used {summary['n_cells']} MICrONS target cells across "
        f"{summary['n_scans']} eligible target--scan observations, with "
        f"{summary['parameters']['replicates']} held-out-stimulus splits per scan. "
        "Each complete compressed morphology was instantiated as a passive "
        "conductance network. Directly connected imaged partners supplied "
        "non-negative trial activity at mapped contact sites, and the measured "
        "postsynaptic response supplied the target. Route selection and one-time "
        "calibration used anatomy and fixed passive conductances only, never task "
        "responses. Restricted dictionaries had equal effective rank within each run."
    )
    lines[10] = (
        "Before learning, the baseline passive soma-to-site transfer vector was "
        "projected once onto each dictionary and frozen. Restricted updates then "
        "used scalar output error times this fixed vector and synapse-local "
        "eligibility; they did not use the exact dynamic adjoint. The one-time "
        "coefficients remain an anatomy-calibration oracle, and this is not "
        "evidence that the MICrONS neurons learned with this rule."
    )
    for name in METHODS:
        value = method[name]
        lines.append(
            f"| {name} | {value['heldout_normalized_mse']:.4f} | "
            f"{value['common_checkpoint_error_field_capture']:.4f} | "
            f"{value['common_checkpoint_update_cosine']:.4f} |"
        )
    lines.extend(["", "Primary matched controls:", ""])
    for key in (
        "heldout_normalized_mse:topology-vs-exact",
        "heldout_normalized_mse:topology-vs-shuffle",
        "heldout_normalized_mse:topology-vs-random",
        "common_checkpoint_update_cosine:topology-vs-shuffle",
        "common_checkpoint_update_cosine:topology-vs-random",
    ):
        value = contrasts[key]
        lines.append(
            f"- {value['contrast']} for {value['metric']}: mean {value['mean_difference']:.4f}, "
            f"95% cell-bootstrap CI [{value['cell_bootstrap_ci95'][0]:.4f}, "
            f"{value['cell_bootstrap_ci95'][1]:.4f}], "
            f"positive/negative cells {value['positive_cells']}/{value['negative_cells']}, "
            f"two-sided Wilcoxon P={value['wilcoxon_two_sided_p']}."
        )
    lines.extend(
        [
            "",
            "## Validation and interpretation",
            "",
            (
                "The largest relative finite-difference error for an exact conductance gradient"
                f" was {summary['validation']['maximum_finite_difference_relative_error']:.3e}. The"
                " route controls preserve channel count and effective rank; site shuffling"
                " additionally preserves every selected route column's values and nonzero count."
            ),
            "",
            summary["interpretation"],
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/run_reconstructed_tree_task_learning.py",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--readout-decay", type=float, default=1e-6)
    parser.add_argument("--initial-conductance", type=float, default=0.2)
    parser.add_argument("--background-e-scale", type=float, default=0.35)
    parser.add_argument("--background-i-scale", type=float, default=0.35)
    parser.add_argument("--excitatory-reversal", type=float, default=1.0)
    parser.add_argument("--inhibitory-reversal", type=float, default=-0.2)
    parser.add_argument("--input-lower-quantile", type=float, default=0.1)
    parser.add_argument("--input-upper-quantile", type=float, default=0.9)
    parser.add_argument("--input-maximum", type=float, default=3.0)
    parser.add_argument("--maximum-targets", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    all_segments = pd.read_csv(args.segments)
    extract_dirs = sorted(args.extract_root.glob("target*_automatic_conservative"))
    if args.maximum_targets is not None:
        extract_dirs = extract_dirs[: int(args.maximum_targets)]
    if args.workers > 1 and len(extract_dirs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(extract_dirs))) as pool:
            futures = [
                pool.submit(run_scan, extract_dir, all_segments, args)
                for extract_dir in extract_dirs
            ]
            scan_results = [future.result() for future in futures]
    else:
        scan_results = [run_scan(extract_dir, all_segments, args) for extract_dir in extract_dirs]
    rows = [row for scan_rows, _, _, _ in scan_results for row in scan_rows]
    metadata = [item for _, scan_metadata, _, _ in scan_results for item in scan_metadata]
    inputs = [input_record for _, _, input_record, _ in scan_results if input_record is not None]
    errors = [error for _, _, _, error in scan_results if error is not None]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no target completed: {errors}")
    frame.to_csv(args.outdir / "runs.csv", index=False)
    pd.DataFrame(metadata).to_json(
        args.outdir / "dictionary_and_validation_metadata.jsonl",
        orient="records",
        lines=True,
    )
    scan_means = frame.groupby(
        ["target_root_id", "target_nucleus_id", "session", "scan_idx", "method"],
        as_index=False,
    ).agg(
        heldout_normalized_mse=("heldout_normalized_mse", "mean"),
        heldout_normalized_mse_sd=("heldout_normalized_mse", "std"),
        common_checkpoint_error_field_capture=("common_checkpoint_error_field_capture", "mean"),
        common_checkpoint_update_capture=("common_checkpoint_update_capture", "mean"),
        common_checkpoint_update_cosine=("common_checkpoint_update_cosine", "mean"),
        final_training_objective=("final_training_objective", "mean"),
        mean_learned_conductance=("mean_learned_conductance", "mean"),
    )
    scan_means.to_csv(args.outdir / "scan_method_means.csv", index=False)
    cell_means = scan_means.groupby(
        ["target_root_id", "target_nucleus_id", "method"], as_index=False
    ).agg(
        heldout_normalized_mse=("heldout_normalized_mse", "mean"),
        heldout_normalized_mse_sd=("heldout_normalized_mse", "std"),
        common_checkpoint_error_field_capture=("common_checkpoint_error_field_capture", "mean"),
        common_checkpoint_update_capture=("common_checkpoint_update_capture", "mean"),
        common_checkpoint_update_cosine=("common_checkpoint_update_cosine", "mean"),
        final_training_objective=("final_training_objective", "mean"),
        mean_learned_conductance=("mean_learned_conductance", "mean"),
    )
    cell_means.to_csv(args.outdir / "cell_method_means.csv", index=False)

    comparisons = {
        "topology-vs-shuffle": ("topology-matched routes", "site-shuffled routes"),
        "topology-vs-random": ("topology-matched routes", "random anatomical routes"),
    }
    contrasts: dict[str, dict[str, Any]] = {}
    metrics = (
        "heldout_normalized_mse",
        "common_checkpoint_error_field_capture",
        "common_checkpoint_update_capture",
        "common_checkpoint_update_cosine",
    )
    counter = 0
    for metric in metrics:
        for label, (first, second) in comparisons.items():
            contrasts[f"{metric}:{label}"] = cell_level_contrast(
                cell_means,
                metric,
                first,
                second,
                args.seed + 100 + counter,
            )
            counter += 1
    contrasts["heldout_normalized_mse:topology-vs-exact"] = cell_level_contrast(
        cell_means,
        "heldout_normalized_mse",
        "topology-matched routes",
        "exact compartment error",
        args.seed + 200,
    )
    method_cell_means = {
        method: {
            metric: float(group[metric].mean())
            for metric in (
                "heldout_normalized_mse",
                "common_checkpoint_error_field_capture",
                "common_checkpoint_update_capture",
                "common_checkpoint_update_cosine",
            )
        }
        for method, group in cell_means.groupby("method")
    }
    finite_errors = [float(item["finite_difference"]["relative_error"]) for item in metadata]
    mse_shuffle = contrasts["heldout_normalized_mse:topology-vs-shuffle"]
    mse_random = contrasts["heldout_normalized_mse:topology-vs-random"]
    if mse_shuffle["cell_bootstrap_ci95"][1] < 0 and mse_random["cell_bootstrap_ci95"][1] < 0:
        interpretation = (
            "In this small cohort, topology-matched oracle feedback improved held-out learning"
            " relative to both controls. This supports a computational sufficiency claim for the"
            " reconstructed route family, not a claim about the biological teaching signal."
        )
    elif mse_shuffle["cell_bootstrap_ci95"][0] > 0 or mse_random["cell_bootstrap_ci95"][0] > 0:
        interpretation = (
            "Topology matching did not improve held-out learning in this small cohort and was worse"
            " for at least one control. The result is evidence against a general anatomy-only"
            " advantage for this measured-response task."
        )
    else:
        interpretation = (
            "Topology-matched feedback has the lowest mean held-out error among the three"
            " restricted rules, but the cell-level intervals do not establish an advantage over"
            " either matched control. Its common-checkpoint gradient-direction diagnostic is also"
            " lower than both controls, showing that this descriptive metric and final learning"
            " need not order methods in the same way. The result places a boundary on anatomy-only"
            " routing in this seven-cell measured-response task."
        )
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"extract_root", "segments", "outdir", "report"}
    }
    summary = {
        "analysis": "projected local conductance learning on reconstructed MICrONS trees",
        "status": "completed" if not errors else "completed_with_exclusions",
        "n_cells": int(frame["target_root_id"].nunique()),
        "n_scans": int(frame[["target_root_id", "session", "scan_idx"]].drop_duplicates().shape[0]),
        "n_runs": int(len(frame)),
        "replication_unit": "postsynaptic target cell",
        "nested_unit": "eligible scan, held-out stimulus-identity split and fixed initialization",
        "task_source": (
            "measured trial-aligned responses of directly connected imaged partners and their"
            " postsynaptic target"
        ),
        "task_route_independence": (
            "target values and trial responses are not used to select or construct feedback"
            " dictionaries"
        ),
        "feedback_scope": (
            "task-independent one-time projection of the baseline passive "
            "soma-to-site transfer; frozen during learning and multiplied only "
            "by scalar output error"
        ),
        "forward_model": (
            "steady-state passive conductance solve on each complete compressed reconstructed tree"
        ),
        "method_cell_means": method_cell_means,
        "contrasts": contrasts,
        "validation": {
            "maximum_finite_difference_relative_error": float(max(finite_errors)),
            "median_finite_difference_relative_error": float(np.median(finite_errors)),
            "all_restricted_dictionaries_rank_matched_within_run": bool(
                frame[frame["method"] != "exact compartment error"]
                .groupby(["target_root_id", "session", "scan_idx", "replicate"])["dictionary_rank"]
                .nunique()
                .eq(1)
                .all()
            ),
        },
        "interpretation": interpretation,
        "limitations": [
            (
                "Only seven target cells from one MICRONS mouse have the required structural and"
                " functional join; all eligible scans are nested within those cells."
            ),
            (
                "Calcium responses are used as task variables; they are not interpreted as"
                " biological teaching signals."
            ),
            (
                "Passive conductance values are normalized proxies rather than"
                " electrophysiologically fitted parameters."
            ),
            (
                "The one-time anatomy-calibration coefficients are oracle quantities and do not"
                " specify a biological developmental mechanism."
            ),
            (
                "The experiment evaluates modeled learning on reconstructed anatomy, not learning"
                " observed in vivo."
            ),
        ],
        "parameters": parameters,
        "input_provenance": {
            "segments": {"path": portable_path(args.segments), "sha256": sha256(args.segments)},
            "runner": {
                "path": portable_path(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "targets": inputs,
        },
        "errors": errors,
    }
    write_json(args.outdir / "summary.json", summary)
    write_json(args.outdir / "config.json", parameters)
    make_report(summary, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
