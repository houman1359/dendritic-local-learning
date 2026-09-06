#!/usr/bin/env python3
"""Independent exact finite-domain audit of four-input Boolean tree capacity.

No imports from the learning experiment or earlier morphology implementations.
Truth tables, rational Walsh coefficients and ranks are exact. Singular-value
tails and normalized coefficients use floating-point arithmetic and are labeled
accordingly. Exact reconstructions are certified in rational arithmetic before
the root is centered and variance-normalized.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "source_data/boolean_theory"
FAMILIES = ("and4", "or4", "parity4", "or_of_ands", "xor_of_ands",
            "and_of_xors", "nested")
FORMULAS = {
    "and4": "a AND b AND c AND d",
    "or4": "a OR b OR c OR d",
    "parity4": "a XOR b XOR c XOR d",
    "or_of_ands": "(a AND b) OR (c AND d)",
    "xor_of_ands": "(a AND b) XOR (c AND d)",
    "and_of_xors": "(a XOR b) AND (c XOR d)",
    "nested": "a AND (b OR (c AND d))",
}
# Pattern bit 0 is a, then b, c, d; physical inputs x=2*bit-1.
BITS = tuple(tuple((p >> i) & 1 for i in range(4)) for p in range(16))
X = tuple(tuple(2 * b - 1 for b in row) for row in BITS)


def truth(family):
    values = []
    for a, b, c, d in BITS:
        values.append({
            "and4": a & b & c & d,
            "or4": a | b | c | d,
            "parity4": a ^ b ^ c ^ d,
            "or_of_ands": (a & b) | (c & d),
            "xor_of_ands": (a & b) ^ (c & d),
            "and_of_xors": (a ^ b) & (c ^ d),
            "nested": a & (b | (c & d)),
        }[family])
    return tuple(values)


def character(pattern, mask):
    value = 1
    for j in range(4):
        if mask & (1 << j):
            value *= X[pattern][j]
    return value


def walsh(values):
    return tuple(sum(Q(values[p]) * character(p, mask) for p in range(16)) / 16
                 for mask in range(16))


def exact_rank(matrix):
    """Rational row reduction, independently of the numerical SVD."""
    a = [[Q(v) for v in row] for row in matrix]
    if not a:
        return 0
    rank = 0
    for column in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        pivot_value = a[rank][column]
        a[rank] = [v / pivot_value for v in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][column]:
                scale = a[i][column]
                a[i] = [v - scale * w for v, w in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def support(tree):
    if isinstance(tree, int):
        return (tree,)
    return tuple(sorted(support(tree[0]) + support(tree[1])))


def label(tree):
    if isinstance(tree, int):
        return "abcd"[tree]
    return "(" + label(tree[0]) + "," + label(tree[1]) + ")"


@lru_cache(None)
def all_trees(leaves=(0, 1, 2, 3)):
    """Unordered rooted full binary trees; minimum leaf fixes child ordering."""
    if len(leaves) == 1:
        return (leaves[0],)
    answer = []
    # Each unordered partition is represented once by putting its minimum left.
    for flags in itertools.product((0, 1), repeat=len(leaves) - 1):
        left = (leaves[0],) + tuple(v for v, f in zip(leaves[1:], flags) if f)
        right = tuple(v for v, f in zip(leaves[1:], flags) if not f)
        if not right:
            continue
        answer.extend((a, b) for a in all_trees(left) for b in all_trees(right))
    return tuple(sorted(answer, key=label))


def depth(tree):
    return 0 if isinstance(tree, int) else 1 + max(map(depth, tree))


def nodes(tree):
    if isinstance(tree, int):
        return ()
    return nodes(tree[0]) + nodes(tree[1]) + (tree,)


def centered_matrix(coefficients, subset):
    outside = tuple(i for i in range(4) if i not in subset)
    matrix = []
    for inner_mask in range(1, 1 << len(subset)):
        row = []
        for outer_mask in range(1 << len(outside)):
            mask = sum(1 << k for i, k in enumerate(subset) if inner_mask & (1 << i))
            mask += sum(1 << k for i, k in enumerate(outside) if outer_mask & (1 << i))
            row.append(coefficients[mask])
        matrix.append(row)
    return matrix


def cut_audit(coefficients, subset, variance):
    matrix = centered_matrix(coefficients, subset)
    values = np.linalg.svd(np.array(matrix, dtype=float), compute_uv=False)
    # The raw matrix is exactly rational; its spectral tail is evaluated in float64.
    rank = exact_rank(matrix)
    tail = float(values[1:] @ values[1:]) if rank > 1 else 0.0
    return dict(subset="".join("abcd"[i] for i in subset),
                outside="".join("abcd"[i] for i in range(4) if i not in subset),
                centered_rank=rank, raw_mse_lower_bound=tail,
                normalized_mse_lower_bound=tail / float(variance),
                squared_singular_values_raw=json.dumps((values ** 2).tolist()),
                centered_matrix_raw_rational=json.dumps([[str(x) for x in row] for row in matrix]))


def signed_slice(values, subset):
    """Choose the lexicographically first nonconstant Boolean outside-context slice."""
    outside = tuple(i for i in range(4) if i not in subset)
    for context in itertools.product((0, 1), repeat=len(outside)):
        desired = []
        for bits in BITS:
            fixed = list(bits)
            for i, bit in zip(outside, context):
                fixed[i] = bit
            index = sum(b << i for i, b in enumerate(fixed))
            desired.append(Q(2 * values[index] - 1))
        if len(set(desired)) > 1:
            return tuple(desired), dict(zip(map(str, outside), context))
    raise ValueError("Subtree is irrelevant to the target; not present in these seven families")


def evaluate_rational(tree, parameters):
    if isinstance(tree, int):
        return tuple(Q(x[tree]) for x in X)
    left = evaluate_rational(tree[0], parameters)
    right = evaluate_rational(tree[1], parameters)
    a, b, c, d = parameters[label(tree)]
    return tuple(a + b * l + c * r + d * l * r for l, r in zip(left, right))


def exact_construction(values, tree):
    """Build a rational certificate from Boolean slices, without regression or SVD.

    Internal nonroot features are signed Boolean slices. A local truth table on
    their four attainable child-output pairs gives the four coefficients by
    direct interpolation. A failure is reported, never replaced by a fit.
    """
    parameters, contexts = {}, {}
    for node in nodes(tree):
        desired, context = ((tuple(map(Q, values)), {}) if node == tree
                            else signed_slice(values, support(node)))
        left = evaluate_rational(node[0], parameters)
        right = evaluate_rational(node[1], parameters)
        local_table = {}
        for l, r, target in zip(left, right, desired):
            pair = (l, r)
            if pair in local_table and local_table[pair] != target:
                return None
            local_table[pair] = target
        pairs = tuple(itertools.product((Q(-1), Q(1)), repeat=2))
        if set(local_table) != set(pairs):
            return None
        parameters[label(node)] = tuple(
            sum(local_table[l, r] * basis(l, r) for l, r in pairs) / 4
            for basis in (lambda l, r: 1, lambda l, r: l,
                          lambda l, r: r, lambda l, r: l * r))
        contexts[label(node)] = context
        assert evaluate_rational(node, parameters) == desired
    exact_output = evaluate_rational(tree, parameters)
    assert exact_output == tuple(map(Q, values))
    mean = sum(map(Q, values)) / 16
    variance = sum((Q(v) - mean) ** 2 for v in values) / 16
    root_coeff = parameters[label(tree)]
    adjusted_root = (root_coeff[0] - mean,) + root_coeff[1:]
    normalized = {name: list(map(float, vals)) for name, vals in parameters.items()}
    normalized[label(tree)] = [float(v) / np.sqrt(float(variance)) for v in adjusted_root]
    # An exact squared check handles the irrational normalizing scale.
    squared_coefficients = [v * v for name, vals in parameters.items()
                            if name != label(tree) for v in vals]
    squared_coefficients += [v * v / variance for v in adjusted_root]
    max_squared = max(squared_coefficients)
    return dict(raw_node_coefficients_rational={k: list(map(str, v)) for k, v in parameters.items()},
                normalized_node_coefficients=normalized,
                normalized_root_numerator_rational=list(map(str, adjusted_root)),
                normalized_root_denominator="sqrt(" + str(variance) + ")",
                signed_slice_outside_contexts=contexts,
                raw_truth_table_verified_exactly=True,
                max_normalized_coefficient_squared_rational=str(max_squared),
                max_abs_normalized_coefficient=float(np.sqrt(float(max_squared))),
                all_normalized_coefficients_within_box_2_exact=bool(max_squared <= 4))


def gradient_audit(values, variance):
    # Derivative of the unique multi-affine extension with respect to x_i.
    # Direct paired truth-value differences, independent of the Fourier routine.
    grad = [[Q(values[p | (1 << j)] - values[p & ~(1 << j)], 2)
             for j in range(4)] for p in range(16)]
    moment = [[sum(row[i] * row[j] for row in grad) / 16
               for j in range(4)] for i in range(4)]
    spectrum = np.linalg.eigvalsh(np.array(moment, dtype=float))
    normalized_spectrum = spectrum / float(variance)
    return dict(input_gradient_rank_exact=exact_rank(moment),
                input_gradient_second_moment_raw_rational=json.dumps([[str(v) for v in row] for row in moment]),
                input_gradient_second_moment_eigenvalues_raw=json.dumps(spectrum.tolist()),
                input_gradient_second_moment_eigenvalues_normalized=json.dumps(normalized_spectrum.tolist()),
                input_gradient_participation_rank=float(spectrum.sum() ** 2 / (spectrum @ spectrum)))


def gate_values(gate, u, v):
    if gate == "AND":
        return u * v, v, u
    if gate == "OR":
        return u + v - u * v, 1 - v, 1 - u
    if gate == "XOR":
        return u + v - 2 * u * v, 1 - 2 * v, 1 - 2 * u
    raise ValueError(gate)


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(out=DEFAULT_OUT, make_plot=True):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    trees = all_trees()
    assert len(trees) == 15 and sum(depth(t) == 2 for t in trees) == 3
    target_rows, truth_rows, coefficient_rows, cut_rows = [], [], [], []
    tree_rows, tree_cut_rows, certificates = [], [], []
    for family in FAMILIES:
        values = truth(family)
        mean = sum(map(Q, values)) / 16
        variance = sum((Q(v) - mean) ** 2 for v in values) / 16
        coefficients = walsh(values)
        assert coefficients[0] == mean
        assert sum(v * v for v in coefficients[1:]) == variance
        for p, (bits, value) in enumerate(zip(BITS, values)):
            truth_rows.append(dict(family=family, pattern_index=p,
                                   a=bits[0], b=bits[1], c=bits[2], d=bits[3],
                                   x_a=X[p][0], x_b=X[p][1], x_c=X[p][2], x_d=X[p][3],
                                   target_raw=value,
                                   target_normalized=float(Q(value) - mean) / np.sqrt(float(variance))))
        for mask, coeff in enumerate(coefficients):
            coefficient_rows.append(dict(family=family, mask=mask,
                support="".join("abcd"[j] for j in range(4) if mask & (1 << j)) or "constant",
                order=mask.bit_count(), coefficient_raw_rational=str(coeff),
                coefficient_raw=float(coeff), coefficient_normalized=(0.0 if mask == 0 else float(coeff) / np.sqrt(float(variance)))))
        cuts = {}
        for mask in range(1, 15):
            subset = tuple(j for j in range(4) if mask & (1 << j))
            cuts[subset] = cut_audit(coefficients, subset, variance)
            cut_rows.append(dict(family=family, **cuts[subset]))
        family_trees = []
        for index, tree in enumerate(trees):
            tree_id = "T" + str(index + 1).zfill(2)
            relevant = [cuts[support(node)] for node in nodes(tree)[:-1]]
            max_rank = max(r["centered_rank"] for r in relevant)
            bound_raw = max(r["raw_mse_lower_bound"] for r in relevant)
            cert = exact_construction(values, tree)
            compatible = max_rank <= 1
            assert (cert is not None) == compatible
            if cert is not None:
                assert cert["all_normalized_coefficients_within_box_2_exact"]
                certificates.append(dict(family=family, tree_id=tree_id, tree=label(tree),
                    mean_rational=str(mean), variance_rational=str(variance), **cert))
            row = dict(family=family, tree_id=tree_id, tree=label(tree), depth=depth(tree),
                internal_cuts=";".join(r["subset"] for r in relevant),
                internal_nodes=3, coefficients=12, edges=6,
                maximum_centered_cut_rank=max_rank,
                raw_mse_lower_bound=bound_raw,
                normalized_mse_lower_bound=bound_raw / float(variance),
                zero_bound_exact=compatible, exact_construction_verified=cert is not None,
                exact_construction_max_abs_coefficient=(cert["max_abs_normalized_coefficient"] if cert else ""),
                exact_construction_box_2_verified=(cert["all_normalized_coefficients_within_box_2_exact"] if cert else ""))
            family_trees.append(row)
            tree_rows.append(row)
            tree_cut_rows.extend(dict(family=family, tree_id=tree_id, tree=label(tree), **r) for r in relevant)
        min_depth = min(row["depth"] for row in family_trees if row["exact_construction_verified"])
        target_rows.append(dict(family=family, formula=FORMULAS[family],
            raw_mean_rational=str(mean), raw_variance_rational=str(variance),
            raw_mean=float(mean), raw_variance=float(variance),
            exact_compatible_trees=sum(row["exact_construction_verified"] for row in family_trees),
            exact_compatible_balanced_trees=sum(row["exact_construction_verified"] and row["depth"] == 2 for row in family_trees),
            minimum_exact_depth=min_depth,
            minimum_balanced_normalized_mse_lower_bound=min(row["normalized_mse_lower_bound"] for row in family_trees if row["depth"] == 2),
            max_coefficient_across_exact_constructions=max(c["max_abs_normalized_coefficient"] for c in certificates if c["family"] == family),
            **gradient_audit(values, variance)))
    gate_rows, corner_rows = [], []
    for gate in ("AND", "OR", "XOR"):
        for u, v in itertools.product(np.linspace(0, 1, 101), repeat=2):
            output, left, right = gate_values(gate, float(u), float(v))
            row = dict(gate=gate, left=float(u), right=float(v), output=output,
                       d_output_d_left=left, d_output_d_right=right)
            gate_rows.append(row)
            if u in (0, 1) and v in (0, 1):
                corner_rows.append(row)
    depth_rows = [dict(family=row["family"], formula=row["formula"],
        and_gates={"and4":3,"or4":0,"parity4":0,"or_of_ands":2,"xor_of_ands":2,"and_of_xors":1,"nested":2}[row["family"]],
        or_gates={"and4":0,"or4":3,"parity4":0,"or_of_ands":1,"xor_of_ands":0,"and_of_xors":0,"nested":1}[row["family"]],
        xor_gates={"and4":0,"or4":0,"parity4":3,"or_of_ands":0,"xor_of_ands":1,"and_of_xors":2,"nested":0}[row["family"]],
        minimum_exact_depth=row["minimum_exact_depth"],
        exact_compatible_trees=row["exact_compatible_trees"],
        exact_compatible_balanced_trees=row["exact_compatible_balanced_trees"],
        minimum_balanced_normalized_mse_lower_bound=row["minimum_balanced_normalized_mse_lower_bound"])
        for row in target_rows]
    main_rows = [dict(**row, grouping=row["internal_cuts"].replace(";", "|"),
        compatible=row["exact_construction_verified"],
        normalized_mse_lower_bound_exact="0" if row["exact_construction_verified"] else "8/15",
        display_label="exact possible" if row["exact_construction_verified"] else "NMSE ≥ 8/15")
        for row in tree_rows if row["family"] == "xor_of_ands" and row["depth"] == 2]
    root_credit_rows = []
    for family, gate in (("or_of_ands", "OR"), ("xor_of_ands", "XOR"), ("and_of_xors", "AND")):
        variance = next(row["raw_variance"] for row in target_rows if row["family"] == family)
        for p, (a, b, c, d) in enumerate(BITS):
            u, v = ((a ^ b, c ^ d) if family == "and_of_xors" else (a & b, c & d))
            output, du, dv = gate_values(gate, u, v)
            root_credit_rows.append(dict(family=family, pattern_index=p, root_gate=gate,
                left_branch_raw=u, right_branch_raw=v, output_raw=output,
                d_raw_output_d_raw_left=du, d_raw_output_d_raw_right=dv,
                d_normalized_output_d_raw_left=du/np.sqrt(variance),
                d_normalized_output_d_raw_right=dv/np.sqrt(variance)))
    outputs = {"truth_tables.csv": truth_rows, "walsh_coefficients.csv": coefficient_rows,
               "target_summary.csv": target_rows, "all_cuts.csv": cut_rows,
               "tree_capacity.csv": tree_rows, "tree_cuts.csv": tree_cut_rows,
               "depth_summary.csv": depth_rows, "main6a_grouping.csv": main_rows,
               "gate_credit_fields.csv": gate_rows, "gate_derivative_corners.csv": corner_rows,
               "canonical_root_credit_by_pattern.csv": root_credit_rows}
    for name, rows in outputs.items():
        csv_write(out / name, rows)
    (out / "exact_constructions.json").write_text(json.dumps(certificates, indent=2, sort_keys=True) + "\n")
    if make_plot:
        plot_gate_grid(out)
    report = dict(status="exhaustive_finite_domain_analytic_audit_no_training",
        families=len(FAMILIES), patterns_per_family=16, labeled_unordered_binary_trees=15,
        balanced_trees=3, depth_3_trees=12, family_tree_pairs=len(tree_rows),
        family_cut_pairs=len(cut_rows), exact_constructions=len(certificates),
        all_compatible_constructions_inside_box_2=True,
        largest_certified_absolute_coefficient=max(c["max_abs_normalized_coefficient"] for c in certificates),
        largest_certified_coefficient_squared_rational=str(max(Q(c["max_normalized_coefficient_squared_rational"]) for c in certificates)),
        centered_rank_method="exact rational row reduction",
        singular_value_tail_method="float64 SVD of exact rational raw Walsh matrix",
        exact_construction_method="nonconstant signed Boolean slices; exact rational four-corner interpolation",
        normalization="raw y in {0,1}; z=(y-E[y])/sqrt(Var(y)); x=2*bit-1",
        gradient_definition="uncentered second moment of gradient of unique multi-affine extension with respect to x",
        bound_scope="uniform full-domain truth-value regression; maximum cut tail is a lower bound, not sum and not classification error",
        coefficient_gauge="signed Boolean internal states for certificates; separate canonical 0/1 gates for derivative illustrations",
        runtime_seconds=time.perf_counter()-start, python_version=platform.python_version(), numpy_version=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tables={name: dict(rows=len(rows), sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name, rows in outputs.items()})
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # Validation facts are generated from exact assertions above. pytest results
    # are separately recorded by validate.py rather than inferred here.
    validation = dict(exact_rank_and_construction_agreement_family_tree_pairs=len(tree_rows),
        exact_raw_truth_table_constructions_verified=len(certificates),
        exact_parameter_box_certificates_verified=len(certificates),
        exact_parseval_and_mean_checks=len(FAMILIES),
        exhaustive_tree_count=15, maximum_coefficient_squared_rational=report["largest_certified_coefficient_squared_rational"],
        source_sha256=report["source_sha256"],
        exact_constructions_sha256=hashlib.sha256((out/"exact_constructions.json").read_bytes()).hexdigest())
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def plot_gate_grid(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    axis = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.9), constrained_layout=True)
    for ax, gate in zip(axes, ("AND", "OR", "XOR")):
        values = np.array([[gate_values(gate, u, v)[1] for u in axis] for v in axis])
        im = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1),
                       norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap="RdBu_r")
        ax.set(xlabel="Left branch output u", ylabel="Right branch output v",
               title=gate + ": ∂output/∂u")
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    fig.colorbar(im, ax=axes, label="Canonical gate derivative", shrink=.8)
    fig.suptitle("Illustrative 0–1 gate gauge; learned internal states need not share this gauge", fontsize=9)
    fig.savefig(out / "canonical_gate_derivatives.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "canonical_gate_derivatives.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    generate(args.out, make_plot=not args.no_plot)
