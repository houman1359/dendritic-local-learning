"""Plot generation helpers for information analysis."""

from dendritic_modeling.plotting.visualizations.information_plots import (
    plot_fully_aggregated_analysis,
    plot_info_vs_ei_depth_per_dendritic_layer,
    plot_layer_wise_aggregated_analysis,
    plot_network_wise_aggregated_analysis,
    plot_structured_ei_network_analysis,
)


class InformationPlottingMixin:
    """Generate optional information-analysis visualizations."""

    def _generate_information_plots(
        self, results: dict, save_path: str, somatic_synapses: bool
    ):
        """Generate plots for information analysis results."""
        try:
            # Generate dendritic depth plots based on enabled analysis modes
            if results and "layer_statistics" in results:
                per_layer_enabled = results.get("per_layer_analysis_enabled", False)
                per_einet_enabled = results.get("per_einet_analysis_enabled", False)

                if per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "Both per_layer and per_einet analysis enabled - generating detailed comparison plots..."
                    )
                    # Generate structured plots grouped by major layer and network type
                    plot_structured_ei_network_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif per_layer_enabled and not per_einet_enabled:
                    self.logger.info(
                        "per_layer analysis enabled, per_einet disabled - generating layer-wise plots..."
                    )
                    # Aggregate E/I networks, but keep layers separate
                    plot_layer_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif not per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "per_einet analysis enabled, per_layer disabled - generating network-wise plots..."
                    )
                    # Aggregate layers, but keep E/I networks separate
                    plot_network_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                else:
                    self.logger.info(
                        "Both per_layer and per_einet analysis disabled - generating overall aggregated plots..."
                    )
                    # Aggregate both layers and E/I networks
                    plot_fully_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )

            # Generate info vs EI depth per dendritic layer plots if we have layer statistics
            self.logger.debug(
                f"Checking for layer_statistics: results exists={results is not None}"
            )
            if results:
                self.logger.debug(f"Results keys: {list(results.keys())}")
                self.logger.debug(
                    f"layer_statistics in results: {'layer_statistics' in results}"
                )
                if "layer_statistics" in results:
                    self.logger.debug(
                        f"layer_statistics length: {len(results['layer_statistics'])}"
                    )

            if (
                results
                and "layer_statistics" in results
                and len(results["layer_statistics"]) > 0
            ):
                self.logger.debug(
                    f"Calling plot_info_vs_ei_depth_per_dendritic_layer with {len(results['layer_statistics'])} layer statistics"
                )
                plot_info_vs_ei_depth_per_dendritic_layer(
                    results["layer_statistics"],
                    save_path=save_path,
                    logger=self.logger,
                    somatic_synapses=somatic_synapses,
                    layer_soma_relative_depths=results.get(
                        "layer_soma_relative_depths"
                    ),
                )
            else:
                self.logger.warning(
                    "Skipping plot_info_vs_ei_depth_per_dendritic_layer - no layer statistics found"
                )

            # Generate variance analysis plots if variance analysis was computed
            if results and "variance_analysis" in results:
                try:
                    self.logger.info(
                        "Generating information variance analysis plots..."
                    )
                    self.logger.debug(
                        f"Variance analysis keys: {list(results['variance_analysis'].keys())}"
                    )
                    from dendritic_modeling.plotting.visualizations.information_plots import (
                        plot_information_variance_analysis,
                    )

                    plot_information_variance_analysis(
                        results["variance_analysis"],
                        save_path=save_path,
                        somatic_synapses=somatic_synapses,
                        logger=self.logger,
                    )
                    self.logger.info("Variance plots completed successfully!")
                except Exception as e:
                    self._log_exception_with_traceback(
                        "Error generating variance plots",
                        e,
                    )
            else:
                self.logger.warning(
                    "No variance_analysis found in results - skipping variance plots"
                )

        except Exception as e:
            self.logger.error(f"Error generating information plots: {e}")


__all__ = ["InformationPlottingMixin"]
