"""Matched analysis of direct and population-mediated inhibitory routing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval, stable_seed
from dendritic_modeling.scripts.sweeps.paired_comparison import paired_arm_comparison


@dataclass(frozen=True)
class InhibitoryRoutingComparison:
    """Seed-level records and paired summaries for one fixed contact budget."""

    per_seed: pd.DataFrame
    accuracy_summary: pd.DataFrame
    mechanism_paired_units: pd.DataFrame
    mechanism_paired_summary: pd.DataFrame
    routing_paired_units: pd.DataFrame
    routing_paired_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Routing comparison is missing required columns: {missing}")


def fixed_budget_inhibitory_routing_comparison(
    data: pd.DataFrame,
    *,
    budget: int,
    expected_allocations: list[tuple[int, int]],
    expected_routes: list[str],
    route_labels: dict[str, str],
    expected_seeds_per_cell: int,
    draws: int,
    bootstrap_seed: int,
    dataset: str | None = None,
    route_column: str = "inhibitory_routing",
    mechanism_column: str = "use_shunting",
    excitatory_column: str = "ee_value",
    inhibitory_column: str = "ie_value",
    metric: str = "test_accuracy",
    seed_column: str = "seed",
) -> InhibitoryRoutingComparison:
    """Validate and summarize a paired fixed-budget inhibitory-routing sweep.

    The trained-model seed is the experimental unit. Mechanism effects are
    shunting minus additive within route; routing effects are the second route
    minus the first route within mechanism.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if len(expected_routes) != 2 or len(set(expected_routes)) != 2:
        raise ValueError("Exactly two distinct expected routes are required")
    if set(route_labels) != set(expected_routes):
        raise ValueError("route_labels must name exactly the expected routes")
    if expected_seeds_per_cell <= 0 or draws <= 0:
        raise ValueError("expected_seeds_per_cell and draws must be positive")

    required = [
        seed_column,
        metric,
        mechanism_column,
        excitatory_column,
        inhibitory_column,
        route_column,
    ]
    if dataset is not None:
        required.append("dataset")
    _require_columns(data, required)
    work = data.copy()
    work[excitatory_column] = pd.to_numeric(work[excitatory_column], errors="raise")
    work[inhibitory_column] = pd.to_numeric(work[inhibitory_column], errors="raise")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.loc[
        (work[excitatory_column] + work[inhibitory_column]).eq(int(budget))
    ].copy()
    if work[metric].isna().any():
        raise ValueError("Routing comparison contains incomplete metric records")
    if dataset is not None and set(work["dataset"].astype(str)) != {str(dataset)}:
        raise ValueError(f"Routing comparison must contain dataset={dataset!r} only")
    observed_routes = set(work[route_column].astype(str))
    if observed_routes != set(expected_routes):
        raise ValueError(f"Unexpected inhibitory routes: {sorted(observed_routes)}")

    work[mechanism_column] = work[mechanism_column].astype(bool)
    work["mechanism"] = work[mechanism_column].map(
        {False: "additive", True: "shunting"}
    )
    work["routing_label"] = work[route_column].map(route_labels)
    work["n_e"] = work[excitatory_column].astype(int)
    work["n_i"] = work[inhibitory_column].astype(int)
    work["e_fraction"] = work["n_e"] / float(budget)
    work["e_to_i_ratio_label"] = work.apply(
        lambda row: f"{int(row.n_e)}:{int(row.n_i)}", axis=1
    )

    allocations = set(zip(work["n_e"], work["n_i"]))
    if allocations != set(expected_allocations):
        raise ValueError(f"Unexpected fixed-budget allocations: {sorted(allocations)}")
    counts = work.groupby([route_column, mechanism_column, "n_e", "n_i"]).size()
    expected_cells = 2 * 2 * len(expected_allocations)
    expected_rows = expected_cells * int(expected_seeds_per_cell)
    if (
        len(work) != expected_rows
        or len(counts) != expected_cells
        or not counts.eq(int(expected_seeds_per_cell)).all()
    ):
        raise ValueError(
            "Incomplete routing factorial: "
            f"rows={len(work)}/{expected_rows}, cells={len(counts)}/{expected_cells}, "
            f"counts={sorted(counts.unique())}"
        )

    accuracy_rows = []
    group_columns = [route_column, mechanism_column, "n_e", "n_i"]
    for key, group in work.groupby(group_columns, sort=True, dropna=False):
        route, use_shunting, n_e, n_i = key
        interval = bootstrap_mean_interval(
            group[metric].to_numpy(float),
            draws=draws,
            seed=stable_seed(bootstrap_seed, "inhibitory-routing-accuracy", *key),
            require_n=expected_seeds_per_cell,
        )
        accuracy_rows.append(
            {
                route_column: route,
                "routing_label": route_labels[str(route)],
                mechanism_column: bool(use_shunting),
                "mechanism": "shunting" if bool(use_shunting) else "additive",
                "n_e": int(n_e),
                "n_i": int(n_i),
                "e_fraction": float(n_e) / float(budget),
                "e_to_i_ratio_label": f"{int(n_e)}:{int(n_i)}",
                **{
                    f"{metric}_{name}": value
                    for name, value in interval.to_dict().items()
                },
            }
        )
    accuracy = pd.DataFrame(accuracy_rows).sort_values(
        [route_column, mechanism_column, "n_e"]
    )

    mechanism_units, mechanism_summaries = paired_arm_comparison(
        work,
        arm_column=mechanism_column,
        reference="False",
        comparison="True",
        metric=metric,
        pair_columns=[seed_column],
        stratum_columns=[route_column, "n_e", "n_i"],
        require_pairs=expected_seeds_per_cell,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
    routing_units, routing_summaries = paired_arm_comparison(
        work,
        arm_column=route_column,
        reference=expected_routes[0],
        comparison=expected_routes[1],
        metric=metric,
        pair_columns=[seed_column],
        stratum_columns=[mechanism_column, "n_e", "n_i"],
        require_pairs=expected_seeds_per_cell,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
    return InhibitoryRoutingComparison(
        per_seed=work,
        accuracy_summary=accuracy,
        mechanism_paired_units=mechanism_units,
        mechanism_paired_summary=pd.DataFrame(mechanism_summaries),
        routing_paired_units=routing_units,
        routing_paired_summary=pd.DataFrame(routing_summaries),
    )


__all__ = [
    "InhibitoryRoutingComparison",
    "fixed_budget_inhibitory_routing_comparison",
]
