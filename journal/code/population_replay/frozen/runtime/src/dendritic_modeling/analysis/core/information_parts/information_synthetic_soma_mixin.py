"""Synthetic soma result helpers for information analysis."""

from typing import Any


class InformationSyntheticSomaMixin:
    """Add and format synthetic soma statistics for saved analysis results."""

    def _add_synthetic_soma_entry_if_needed(
        self, results: dict[str, Any], model: Any
    ) -> bool:
        # This is only needed when somatic_synapses=False and the soma layer
        # was not captured by the forward hooks.
        somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
        self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

        soma_already_captured = self._soma_layer_already_captured(results)
        if self._should_add_synthetic_soma_entry(
            somatic_synapses=somatic_synapses,
            results=results,
            soma_already_captured=soma_already_captured,
        ):
            self.logger.info(
                f"Adding synthetic soma entry to {len(results['layer_statistics'])} existing layers"
            )
            layer_statistics = results["layer_statistics"]
            real_soma_layer = self._find_real_soma_layer(layer_statistics)
            soma_stats = self._extract_soma_statistics(
                real_soma_layer, layer_statistics
            )
            soma_entry = self._build_synthetic_soma_entry(soma_stats)
            results["layer_statistics"]["synthetic_soma"] = soma_entry
            self.logger.info(
                "Added synthetic soma with real "
                f"I(Vout;C)={soma_stats['soma_vout_mean']:.4f}. "
                f"Total layers now: {len(results['layer_statistics'])}"
            )
        elif not somatic_synapses and "layer_statistics" in results:
            self.logger.info("Soma layer already captured; skipping synthetic soma.")
        elif not somatic_synapses and "layer_statistics" not in results:
            self.logger.warning(
                "somatic_synapses=False but no layer_statistics found in results"
            )
        return somatic_synapses

    @staticmethod
    def _should_add_synthetic_soma_entry(
        *,
        somatic_synapses: bool,
        results: dict[str, Any],
        soma_already_captured: bool,
    ) -> bool:
        """Return whether a synthetic soma layer should be added to results."""
        return (
            not somatic_synapses
            and "layer_statistics" in results
            and not soma_already_captured
        )

    def _soma_layer_already_captured(self, results: dict[str, Any]) -> bool:
        if "layer_statistics" not in results:
            return False

        for layer_name in results["layer_statistics"].keys():
            if self._parse_branch_layer_index(layer_name) is None:
                continue
            layer_data = results["layer_statistics"][layer_name]
            if layer_data.get("has_synapses") is False:
                self.logger.info(f"Soma layer already captured: {layer_name}")
                return True
        return False

    @staticmethod
    def _parse_branch_layer_index(layer_name: str) -> int | None:
        """Extract the branch-layer index from a module path, if present."""
        if "branch_layers." not in layer_name:
            return None
        try:
            return int(layer_name.split("branch_layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _find_real_soma_layer(layer_statistics: dict[str, Any]) -> str | None:
        max_branch_idx = -1
        real_soma_layer = None

        for layer_name in layer_statistics.keys():
            branch_idx = InformationSyntheticSomaMixin._parse_branch_layer_index(
                layer_name
            )
            if branch_idx is None:
                continue
            if branch_idx > max_branch_idx:
                max_branch_idx = branch_idx
                real_soma_layer = layer_name

        return real_soma_layer

    def _extract_soma_statistics(
        self, real_soma_layer: str | None, layer_statistics: dict[str, Any]
    ) -> dict[str, float]:
        soma_stats = self._empty_soma_statistics()

        if real_soma_layer and real_soma_layer in layer_statistics:
            real_soma_stats = layer_statistics[real_soma_layer]
            soma_stats.update(self._copy_soma_statistics_from_layer(real_soma_stats))
            self._log_soma_statistics(real_soma_layer, soma_stats)
        else:
            self.logger.warning(
                "Could not find real soma layer for I(Vout;C) computation"
            )

        return soma_stats

    @classmethod
    def _empty_soma_statistics(cls) -> dict[str, float]:
        """Return zero-valued soma statistics for every supported metric."""
        return dict.fromkeys(cls._soma_statistics_metric_map(), 0.0)

    @classmethod
    def _copy_soma_statistics_from_layer(
        cls, real_soma_stats: dict[str, Any]
    ) -> dict[str, float]:
        """Copy soma metrics from a captured real soma layer into internal keys."""
        return {
            soma_key: real_soma_stats.get(metric_key, 0.0)
            for soma_key, metric_key in cls._soma_statistics_metric_map().items()
        }

    def _log_soma_statistics(
        self, real_soma_layer: str, soma_stats: dict[str, float]
    ) -> None:
        """Log the real soma values reused for the synthetic soma entry."""
        self.logger.info(f"Found real soma layer: {real_soma_layer}")
        self.logger.info(
            "Using real soma "
            f"I(Vout;C): {soma_stats['soma_vout_mean']:.4f} "
            f"\u00b1 {soma_stats['soma_vout_std']:.4f}"
        )
        self.logger.info(
            "Using real soma "
            f"I(Vb;C): {soma_stats['soma_vb_mean']:.4f} "
            f"\u00b1 {soma_stats['soma_vb_std']:.4f}"
        )

    @staticmethod
    def _soma_statistics_metric_map() -> dict[str, str]:
        """Map internal soma-stat keys to source layer-stat metric keys."""
        return {
            "soma_vout_mean": "I(Vout;C)_mean",
            "soma_vout_std": "I(Vout;C)_std",
            "soma_vb_mean": "I(Vb;C)_mean",
            "soma_vb_std": "I(Vb;C)_std",
            "soma_vb_lin_mean": "I(Vb_lin;C)_mean",
            "soma_vb_lin_std": "I(Vb_lin;C)_std",
            "soma_vb_vout_mean": "I(Vb;Vout)_mean",
            "soma_vb_vout_std": "I(Vb;Vout)_std",
            "soma_vb_vout_c_mean": "I(Vb;Vout|C)_mean",
            "soma_vb_vout_c_std": "I(Vb;Vout|C)_std",
            "soma_vb_lin_vout_mean": "I(Vb_lin;Vout_lin)_mean",
            "soma_vb_lin_vout_std": "I(Vb_lin;Vout_lin)_std",
            "soma_vb_lin_vout_c_mean": "I(Vb_lin;Vout_lin|C)_mean",
            "soma_vb_lin_vout_c_std": "I(Vb_lin;Vout_lin|C)_std",
            "soma_vout_lin_mean": "I(Vout_lin;C)_mean",
            "soma_vout_lin_std": "I(Vout_lin;C)_std",
        }

    def _build_synthetic_soma_entry(
        self, soma_stats: dict[str, float]
    ) -> dict[str, Any]:
        soma_entry = {
            metric_key: (0.0 if soma_stat_key is None else soma_stats[soma_stat_key])
            for metric_key, soma_stat_key in self._synthetic_soma_metric_sources()
        }
        return self._filter_synthetic_soma_entry_metrics(soma_entry)

    @staticmethod
    def _synthetic_soma_metric_sources() -> tuple[tuple[str, str | None], ...]:
        """Map synthetic soma output metrics to copied soma-stat keys."""
        return (
            ("I(E;C)_mean", None),
            ("I(E;C)_std", None),
            ("I(I;C)_mean", None),
            ("I(I;C)_std", None),
            ("I(Vb;C)_mean", "soma_vb_mean"),
            ("I(Vb;C)_std", "soma_vb_std"),
            ("I(Vout;C)_mean", "soma_vout_mean"),
            ("I(Vout;C)_std", "soma_vout_std"),
            ("I(E,I;C)_mean", None),
            ("I(E,I;C)_std", None),
            ("I(E_lin;C)_mean", None),
            ("I(E_lin;C)_std", None),
            ("I(I_lin;C)_mean", None),
            ("I(I_lin;C)_std", None),
            ("I(Vb_lin;C)_mean", "soma_vb_lin_mean"),
            ("I(Vb_lin;C)_std", "soma_vb_lin_std"),
            ("I(Vout_lin;C)_mean", "soma_vout_lin_mean"),
            ("I(Vout_lin;C)_std", "soma_vout_lin_std"),
            ("I(E_lin,I_lin;C)_mean", None),
            ("I(E_lin,I_lin;C)_std", None),
            ("I(E;I)_mean", None),
            ("I(E;I)_std", None),
            ("I(E;Vout)_mean", None),
            ("I(E;Vout)_std", None),
            ("I(I;Vout)_mean", None),
            ("I(I;Vout)_std", None),
            ("I(E;Vb)_mean", None),
            ("I(E;Vb)_std", None),
            ("I(I;Vb)_mean", None),
            ("I(I;Vb)_std", None),
            ("I(Vb;Vout)_mean", "soma_vb_vout_mean"),
            ("I(Vb;Vout)_std", "soma_vb_vout_std"),
            ("I(E_lin;I_lin)_mean", None),
            ("I(E_lin;I_lin)_std", None),
            ("I(E_lin;Vout_lin)_mean", None),
            ("I(E_lin;Vout_lin)_std", None),
            ("I(I_lin;Vout_lin)_mean", None),
            ("I(I_lin;Vout_lin)_std", None),
            ("I(E_lin;Vb_lin)_mean", None),
            ("I(E_lin;Vb_lin)_std", None),
            ("I(I_lin;Vb_lin)_mean", None),
            ("I(I_lin;Vb_lin)_std", None),
            ("I(Vb_lin;Vout_lin)_mean", "soma_vb_lin_vout_mean"),
            ("I(Vb_lin;Vout_lin)_std", "soma_vb_lin_vout_std"),
            ("I(E;I|C)_mean", None),
            ("I(E;I|C)_std", None),
            ("I(E;Vout|C)_mean", None),
            ("I(E;Vout|C)_std", None),
            ("I(I;Vout|C)_mean", None),
            ("I(I;Vout|C)_std", None),
            ("I(E;Vb|C)_mean", None),
            ("I(E;Vb|C)_std", None),
            ("I(I;Vb|C)_mean", None),
            ("I(I;Vb|C)_std", None),
            ("I(Vb;Vout|C)_mean", "soma_vb_vout_c_mean"),
            ("I(Vb;Vout|C)_std", "soma_vb_vout_c_std"),
            ("I(E_lin;I_lin|C)_mean", None),
            ("I(E_lin;I_lin|C)_std", None),
            ("I(E_lin;Vout_lin|C)_mean", None),
            ("I(E_lin;Vout_lin|C)_std", None),
            ("I(I_lin;Vout_lin|C)_mean", None),
            ("I(I_lin;Vout_lin|C)_std", None),
            ("I(E_lin;Vb_lin|C)_mean", None),
            ("I(E_lin;Vb_lin|C)_std", None),
            ("I(I_lin;Vb_lin|C)_mean", None),
            ("I(I_lin;Vb_lin|C)_std", None),
            ("I(Vb_lin;Vout_lin|C)_mean", "soma_vb_lin_vout_c_mean"),
            ("I(Vb_lin;Vout_lin|C)_std", "soma_vb_lin_vout_c_std"),
        )

    def _filter_synthetic_soma_entry_metrics(
        self, soma_entry: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply soma availability rules to a synthetic soma result entry."""
        filtered_entry = {
            key: value
            for key, value in soma_entry.items()
            if self._keep_metric_key_given_availability(
                key,
                has_exc_synapses=False,
                has_inh_synapses=False,
                has_branch_input=True,
            )
        }
        filtered_entry["has_synapses"] = False
        return filtered_entry


__all__ = ["InformationSyntheticSomaMixin"]
