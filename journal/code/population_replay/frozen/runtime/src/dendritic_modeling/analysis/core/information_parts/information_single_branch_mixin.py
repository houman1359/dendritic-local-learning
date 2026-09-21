"""Single-branch information-analysis helpers."""

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm


@dataclass(frozen=True)
class _SingleBranchLayerData:
    """Prepared branch-wise arrays and availability flags for one layer."""

    excitation: np.ndarray
    inhibition: np.ndarray
    output: np.ndarray
    branch_input: np.ndarray | None
    parent_soma_output: np.ndarray | None
    branch_to_parent_soma_index: np.ndarray | None
    has_exc_synapses: bool
    has_inh_synapses: bool
    has_branch_input: bool


@dataclass(frozen=True)
class _SingleBranchLdaSignals:
    """Prepared LDA aggregate signals for one layer."""

    excitation: np.ndarray | None
    inhibition: np.ndarray | None
    branch_input: np.ndarray | None
    output: np.ndarray | None
    shuffled: tuple[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray], ...]


class InformationSingleBranchMixin:
    """Compute and aggregate single-branch information metrics."""

    def _select_single_branch_indices(
        self,
        *,
        layer_name: str,
        n_branches: int,
        branch_to_parent_soma_index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select a deterministic, label-independent branch subset for one layer."""
        requested = getattr(self, "branch_sample_count_per_layer", None)
        if requested is None or requested >= n_branches:
            return np.arange(n_branches, dtype=np.int64)

        base_seed = int(getattr(self, "branch_sample_seed", 0))
        digest = hashlib.sha256(f"{base_seed}:{layer_name}".encode()).digest()
        layer_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = np.random.default_rng(layer_seed)
        strategy = getattr(self, "branch_sample_strategy", "uniform")
        if strategy == "parent_soma_balanced":
            if branch_to_parent_soma_index is None:
                raise ValueError(
                    "parent_soma_balanced branch sampling requires a "
                    "branch-to-parent-soma mapping"
                )
            parent_indices = np.asarray(branch_to_parent_soma_index, dtype=np.int64)
            if parent_indices.shape != (n_branches,):
                raise ValueError(
                    "branch-to-parent-soma mapping must contain one index per branch"
                )
            return self._select_parent_soma_balanced_indices(
                parent_indices=parent_indices,
                requested=int(requested),
                generator=generator,
            )
        return np.sort(
            generator.choice(n_branches, size=int(requested), replace=False)
        ).astype(np.int64)

    @staticmethod
    def _select_parent_soma_balanced_indices(
        *,
        parent_indices: np.ndarray,
        requested: int,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Select branches round-robin across parent somas."""
        parents = np.unique(parent_indices)
        parent_order = generator.permutation(parents)
        queues = {
            int(parent): generator.permutation(
                np.flatnonzero(parent_indices == parent)
            ).tolist()
            for parent in parents
        }
        selected: list[int] = []
        cursor = dict.fromkeys(queues, 0)
        while len(selected) < requested:
            progress = False
            for parent_raw in parent_order:
                parent = int(parent_raw)
                position = cursor[parent]
                if position >= len(queues[parent]):
                    continue
                selected.append(int(queues[parent][position]))
                cursor[parent] = position + 1
                progress = True
                if len(selected) == requested:
                    break
            if not progress:
                raise ValueError(
                    "requested more balanced branches than the mapping provides"
                )
        return np.sort(np.asarray(selected, dtype=np.int64))

    def _prepare_single_branch_layer_data(
        self,
        layer_data: dict[str, Any],
    ) -> _SingleBranchLayerData:
        """Convert one hook-captured layer into branch-wise arrays and flags."""
        E_all_branches_raw = layer_data["excitation"]
        I_all_branches_raw = layer_data["inhibition"]
        Vb_all_branches_raw = layer_data.get("branch_input")

        has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
        has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
        has_branch_input = bool(
            layer_data.get("has_branch_input", Vb_all_branches_raw is not None)
        )

        E_all_branches_filtered, I_all_branches_filtered = self.ensure_inhibitory_data(
            E_all_branches_raw, I_all_branches_raw
        )

        branch_input = (
            Vb_all_branches_raw.detach().cpu().numpy()
            if Vb_all_branches_raw is not None
            else None
        )
        parent_soma_output_raw = layer_data.get("parent_soma_output")
        parent_soma_output = (
            parent_soma_output_raw.detach().cpu().numpy()
            if parent_soma_output_raw is not None
            else None
        )
        branch_to_parent_soma_raw = layer_data.get("branch_to_parent_soma_index")
        branch_to_parent_soma_index = (
            np.asarray(branch_to_parent_soma_raw, dtype=np.int64)
            if branch_to_parent_soma_raw is not None
            else None
        )

        return _SingleBranchLayerData(
            excitation=E_all_branches_filtered.detach().cpu().numpy(),
            inhibition=I_all_branches_filtered.detach().cpu().numpy(),
            output=layer_data["output"].detach().cpu().numpy(),
            branch_input=branch_input,
            parent_soma_output=parent_soma_output,
            branch_to_parent_soma_index=branch_to_parent_soma_index,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    def _prepare_single_branch_lda_signals(
        self,
        layer_data: dict[str, Any],
        C: np.ndarray,
    ) -> _SingleBranchLdaSignals:
        """Compute deterministic and shuffled LDA aggregate signals for one layer."""
        if not self.compute_lda_weights:
            return _SingleBranchLdaSignals(
                excitation=None,
                inhibition=None,
                branch_input=None,
                output=None,
                shuffled=(),
            )

        E_lin, I_lin, Vb_lin, Vout_lin = self.compute_lda_aggregated_signals(
            layer_data, C
        )

        shuffled_signals = []
        if self.lda_n_shuffles > 0:
            for shuffle_idx in range(self.lda_n_shuffles):
                E_sh, I_sh, Vb_sh, Vout_sh = self.compute_shuffled_aggregated_signals(
                    layer_data, C, shuffle_idx
                )
                shuffled_signals.append((E_sh, I_sh, Vb_sh, Vout_sh))

        return _SingleBranchLdaSignals(
            excitation=E_lin,
            inhibition=I_lin,
            branch_input=Vb_lin,
            output=Vout_lin,
            shuffled=tuple(shuffled_signals),
        )

    def _compute_single_branch_result(
        self,
        *,
        prepared_layer: _SingleBranchLayerData,
        lda_signals: _SingleBranchLdaSignals,
        C: np.ndarray,
        branch_idx: int,
        layer_count: int,
    ) -> dict[str, Any]:
        """Compute and filter all metrics for one branch in a prepared layer."""
        E_branch = prepared_layer.excitation[:, branch_idx : branch_idx + 1]
        I_branch = prepared_layer.inhibition[:, branch_idx : branch_idx + 1]
        Vout_branch = prepared_layer.output[:, branch_idx : branch_idx + 1]
        Vb_branch = (
            prepared_layer.branch_input[:, branch_idx : branch_idx + 1]
            if prepared_layer.branch_input is not None
            else None
        )
        parent_soma_branch = None
        if (
            getattr(self, "compute_soma_coupling_mi", False)
            and prepared_layer.parent_soma_output is not None
            and prepared_layer.branch_to_parent_soma_index is not None
        ):
            parent_soma_index = int(
                prepared_layer.branch_to_parent_soma_index[branch_idx]
            )
            parent_soma_branch = prepared_layer.parent_soma_output[
                :, parent_soma_index : parent_soma_index + 1
            ]

        branch_results = self._compute_branch_metrics_with_null_baselines(
            E=E_branch,
            I_var=I_branch,
            Vout=Vout_branch,
            C=C,
            Vb=Vb_branch,
            S=parent_soma_branch,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
            seed_offset=int(layer_count * 10_000 + branch_idx),
            failure_context=f"branch {branch_idx}",
        )

        self._merge_lda_branch_metrics_in_place(
            target=branch_results,
            E_lin=lda_signals.excitation,
            I_lin=lda_signals.inhibition,
            Vb_lin=lda_signals.branch_input,
            Vout_lin=lda_signals.output,
            C=C,
            branch_idx=branch_idx,
            layer_count=layer_count,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
        )

        self._merge_shuffled_branch_metrics_in_place(
            target=branch_results,
            shuffled_signals=lda_signals.shuffled,
            C=C,
            branch_idx=branch_idx,
            has_branch_input=prepared_layer.has_branch_input,
        )

        self._filter_result_metrics_in_place(
            branch_results,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
        )
        return branch_results

    def _finalize_single_branch_results(
        self,
        *,
        all_results: list[dict[str, Any]],
        branch_names: list[str],
        layer_results: dict[str, Any],
        branch_sampling: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Average branch metrics and attach single-branch analysis metadata."""
        if not all_results:
            self.logger.warning("No branch results to average")
            return None

        results = self._average_mi_results(all_results)
        results["branch_names"] = branch_names
        results["num_branches"] = len(all_results)
        results["layer_statistics"] = layer_results
        if branch_sampling is not None:
            results["branch_sampling"] = branch_sampling
        layer_depths = self.collected_soma_relative_depths()
        if layer_depths:
            results["layer_soma_relative_depths"] = layer_depths
        results["computation_level"] = "single_branch"
        return results

    def _compute_single_branch_results(
        self,
        C: np.ndarray,
        entropy_C: Optional[float],
    ) -> dict[str, Any] | None:
        """Compute information metrics for individual branches and average them."""
        all_results = []
        branch_names = []
        layer_results = {}
        branch_sampling_layers = {}

        total_layers = len(
            [name for name, data in self.data_dict.items() if "excitation" in data]
        )
        total_branches = sum(
            data.get("excitation", torch.empty(0, 0)).shape[1]
            for data in self.data_dict.values()
            if "excitation" in data
        )
        layer_count = 0

        self.logger.info(
            f"Single-branch MI analysis: {total_layers} layers, {total_branches} branches, method={self.method}"
        )

        layers_to_process = [
            (name, data)
            for name, data in self.data_dict.items()
            if "excitation" in data
        ]

        for layer_name, layer_data in tqdm(
            layers_to_process,
            desc="Processing layers",
            leave=False,
            ncols=100,
        ):
            layer_count += 1

            prepared_layer = self._prepare_single_branch_layer_data(layer_data)
            lda_signals = self._prepare_single_branch_lda_signals(layer_data, C)

            n_branches = prepared_layer.excitation.shape[1]
            selected_branch_indices = self._select_single_branch_indices(
                layer_name=layer_name,
                n_branches=n_branches,
                branch_to_parent_soma_index=(
                    prepared_layer.branch_to_parent_soma_index
                    if getattr(self, "branch_sample_strategy", "uniform")
                    == "parent_soma_balanced"
                    else None
                ),
            )
            selected_indices_list = selected_branch_indices.tolist()
            selected_indices_sha256 = hashlib.sha256(
                np.asarray(selected_branch_indices, dtype=np.int64).tobytes()
            ).hexdigest()
            branch_sampling_layers[layer_name] = {
                "total_branches": int(n_branches),
                "selected_count": len(selected_indices_list),
                "selected_indices": selected_indices_list,
                "selected_indices_sha256": selected_indices_sha256,
                "parent_soma_population": layer_data.get("parent_soma_population"),
                "branches_per_parent_soma": layer_data.get("branches_per_parent_soma"),
            }
            if prepared_layer.branch_to_parent_soma_index is not None:
                selected_parents = prepared_layer.branch_to_parent_soma_index[
                    selected_branch_indices
                ]
                _, selected_parent_counts = np.unique(
                    selected_parents, return_counts=True
                )
                branch_sampling_layers[layer_name].update(
                    {
                        "selected_parent_soma_count": int(selected_parent_counts.size),
                        "selected_branches_per_parent_soma_min": int(
                            selected_parent_counts.min()
                        ),
                        "selected_branches_per_parent_soma_max": int(
                            selected_parent_counts.max()
                        ),
                    }
                )
            layer_branch_results = []
            layer_start_time = time.time()

            short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
            for branch_idx in tqdm(
                selected_indices_list,
                desc=f"  {short_name}",
                leave=False,
                ncols=100,
                mininterval=0.5,
            ):
                branch_start_time = time.time()

                branch_results = self._compute_single_branch_result(
                    prepared_layer=prepared_layer,
                    lda_signals=lda_signals,
                    C=C,
                    branch_idx=branch_idx,
                    layer_count=layer_count,
                )
                all_results.append(branch_results)
                layer_branch_results.append(branch_results)
                branch_names.append(f"{layer_name}_branch_{branch_idx}")

                self.logger.debug(
                    f"  Branch {branch_idx + 1}/{n_branches} completed (took {time.time() - branch_start_time:.1f}s)"
                )

            if layer_branch_results:
                layer_stats = self._compute_layer_statistics(
                    layer_branch_results,
                    entropy_C=(entropy_C if self.compute_layer_total_proxies else None),
                    topk=self.layer_total_topk,
                )
                self._apply_layer_synapse_availability_in_place(layer_stats, layer_name)
                layer_results[layer_name] = layer_stats

            self.logger.info(
                f"Layer {layer_name} completed: {len(selected_indices_list)}/{n_branches} branches processed in {time.time() - layer_start_time:.1f}s"
            )

        branch_sampling = {
            "mode": (
                "all"
                if getattr(self, "branch_sample_count_per_layer", None) is None
                else "deterministic_subset_per_layer"
            ),
            "requested_count_per_layer": getattr(
                self, "branch_sample_count_per_layer", None
            ),
            "strategy": getattr(self, "branch_sample_strategy", "uniform"),
            "seed": int(getattr(self, "branch_sample_seed", 0)),
            "selection_depends_on_data_or_labels": False,
            "layers": branch_sampling_layers,
        }
        return self._finalize_single_branch_results(
            all_results=all_results,
            branch_names=branch_names,
            layer_results=layer_results,
            branch_sampling=branch_sampling,
        )

    def _merge_lda_branch_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        E_lin: np.ndarray | None,
        I_lin: np.ndarray | None,
        Vb_lin: np.ndarray | None,
        Vout_lin: np.ndarray | None,
        C: np.ndarray,
        branch_idx: int,
        layer_count: int,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> None:
        """Compute one branch's LDA-aggregated metrics and merge renamed keys."""
        if not self.compute_lda_weights or E_lin is None:
            return

        E_branch_lin = E_lin[:, branch_idx : branch_idx + 1]
        I_branch_lin = I_lin[:, branch_idx : branch_idx + 1]
        Vb_branch_lin = (
            Vb_lin[:, branch_idx : branch_idx + 1]
            if has_branch_input and Vb_lin is not None
            else None
        )
        Vout_branch_lin = Vout_lin[:, branch_idx : branch_idx + 1]

        branch_results_lin = self._compute_branch_metrics_with_null_baselines(
            E=E_branch_lin,
            I_var=I_branch_lin,
            Vout=Vout_branch_lin,
            C=C,
            Vb=Vb_branch_lin,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
            seed_offset=int(1_000_000 + layer_count * 10_000 + branch_idx),
            failure_context=f"LDA branch {branch_idx}",
        )
        self._merge_renamed_metric_dicts_in_place(
            target=target,
            source=branch_results_lin,
            token_map={
                "E": "E_lin",
                "I": "I_lin",
                "Vb": "Vb_lin",
                "Vout": "Vout_lin",
                "Vinf": "Vinf_lin",
            },
        )

    def _merge_shuffled_branch_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        shuffled_signals: Sequence[
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]
        ],
        C: np.ndarray,
        branch_idx: int,
        has_branch_input: bool,
    ) -> None:
        """Compute one branch's shuffled LDA baselines and merge averaged keys."""
        if not shuffled_signals:
            return

        shuffled_results = []
        for E_sh, I_sh, Vb_sh, Vout_sh in shuffled_signals:
            E_branch_sh = E_sh[:, branch_idx : branch_idx + 1]
            I_branch_sh = I_sh[:, branch_idx : branch_idx + 1]
            Vb_branch_sh = (
                Vb_sh[:, branch_idx : branch_idx + 1]
                if has_branch_input and Vb_sh is not None
                else None
            )
            Vout_branch_sh = Vout_sh[:, branch_idx : branch_idx + 1]

            branch_results_sh = self.compute_metrics(
                E_branch_sh,
                I_branch_sh,
                Vout_branch_sh,
                C,
                None,
                Vb_branch_sh,
            )
            shuffled_results.append(branch_results_sh)

        self._merge_average_renamed_numeric_metrics_in_place(
            target=target,
            sources=shuffled_results,
            token_map={
                "E": "E_sh",
                "I": "I_sh",
                "Vb": "Vb_sh",
                "Vout": "Vout_sh",
                "Vinf": "Vinf_sh",
            },
        )


__all__ = [
    "InformationSingleBranchMixin",
    "_SingleBranchLayerData",
    "_SingleBranchLdaSignals",
]
