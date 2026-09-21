"""Data collection from sweep result directories."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config_extractor import ConfigExtractor

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects data from sweep result directories."""

    # Canonical directory structure:
    # sweep_results/
    #   ├── configs/config_*.yaml
    #   └── results/config_*/
    #       ├── performance/final.json
    #       ├── information_analysis/final (no .json)
    #       ├── weight_analysis/final
    #       └── ...

    def __init__(self, results_dir: Path):
        """
        Initialize data collector.

        Args:
            results_dir: Root directory containing sweep results
        """
        self.results_dir = Path(results_dir)
        self.configs_dir = self.results_dir / "configs"
        self.results_subdir = self.results_dir / "results"
        self.config_extractor = ConfigExtractor()

    def collect_all(self) -> pd.DataFrame:
        """
        Collect data from all configurations in the sweep.

        Returns:
            DataFrame with one row per configuration
        """
        logger.info(f"Collecting data from {self.results_dir}")

        # Find all config directories
        config_dirs = self._get_config_dirs()
        logger.info(f"Found {len(config_dirs)} configuration directories")

        if not config_dirs:
            logger.warning("No configuration directories found")
            return pd.DataFrame()

        # Collect data from each config
        all_data = []
        for i, config_dir in enumerate(config_dirs):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i + 1}/{len(config_dirs)}...")

            data = self._collect_single_config(config_dir)
            if data:
                all_data.append(data)

        logger.info(
            f"Successfully collected data from {len(all_data)}/{len(config_dirs)} configs"
        )

        return pd.DataFrame(all_data)

    def _get_config_dirs(self) -> list[Path]:
        """Get all configuration directories."""
        if not self.results_subdir.exists():
            logger.error(f"Results directory not found: {self.results_subdir}")
            return []

        # Find all config_* directories
        config_dirs = sorted(self.results_subdir.glob("config_*"))
        return [d for d in config_dirs if d.is_dir()]

    def _collect_single_config(self, config_dir: Path) -> Optional[dict]:
        """
        Collect all data for a single configuration.

        Args:
            config_dir: Path to config_X directory

        Returns:
            Dictionary with all collected data, or None if failed
        """
        # Extract config ID
        match = re.search(r"config_(\d+)", config_dir.name)
        if not match:
            logger.warning(f"Could not extract ID from {config_dir.name}")
            return None

        config_id = int(match.group(1))

        # Load config parameters
        config_yaml = self.configs_dir / f"config_{config_id}.yaml"
        if not config_yaml.exists():
            # Try alternate naming patterns
            possible_patterns = [
                f"unified_config_{config_id}.yaml",
                f"ei_sweep_config_{config_id}.yaml",
            ]
            for pattern in possible_patterns:
                alt_path = self.configs_dir / pattern
                if alt_path.exists():
                    config_yaml = alt_path
                    break

        if not config_yaml.exists():
            logger.warning(f"Config file not found for config_{config_id}")
            return None

        # Extract parameters from config
        params = self.config_extractor.extract_from_file(config_yaml)
        params["config_id"] = config_id
        params["config_file"] = str(config_yaml)

        # Collect analysis results
        params.update(self._load_performance(config_dir))
        params.update(self._load_information(config_dir))
        params.update(self._load_weights(config_dir))
        params.update(self._load_ablation(config_dir))
        params.update(self._load_snr(config_dir))
        params.update(self._load_gain(config_dir))
        params.update(self._load_noise(config_dir))
        params.update(self._load_nparams(config_dir))
        params.update(self._load_model_resources(config_dir))
        params.update(self._load_training_summary(config_dir))

        return params

    def _load_json_file(self, filepath: Path) -> Optional[dict]:
        """Load a JSON file, handling both .json and no extension."""
        # Try exact path first
        if filepath.exists():
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

        # Try without .json extension (if path has .json, try removing it)
        if filepath.suffix == ".json":
            no_ext_path = filepath.with_suffix("")
            if no_ext_path.exists():
                try:
                    with open(no_ext_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {no_ext_path}: {e}")

        # Try adding .json extension (if path doesn't have it)
        if filepath.suffix != ".json":
            json_path = filepath.with_suffix(".json")
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {json_path}: {e}")

        return None

    def _load_performance(self, config_dir: Path) -> dict:
        """Load performance metrics."""
        # perf_file = config_dir / "performance" / "final.json"
        # Search recursively for performance/final.json
        perf_files = list(config_dir.rglob("performance/final.json"))

        if not perf_files:
            # Try without .json extension
            perf_files = list(config_dir.rglob("performance/final"))

        if perf_files:
            perf_file = perf_files[0]  # Use first match found
        else:
            # Fallback to original path
            perf_file = config_dir / "performance" / "final.json"

        data = self._load_json_file(perf_file)

        if not data:
            return {}

        # Flatten performance data
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested metrics (e.g., {'accuracy': ..., 'auc': ...})
                for metric, val in value.items():
                    result[f"{metric}_{key}"] = val  # e.g., test_accuracy
            else:
                result[key] = value

        return result

    def _load_information(self, config_dir: Path) -> dict:
        """Load information analysis metrics."""
        # Search recursively for information_analysis/final.json
        info_files = list(config_dir.rglob("information_analysis/final.json"))

        if not info_files:
            # Try without .json extension
            info_files = list(config_dir.rglob("information_analysis/final"))

        if info_files:
            info_file = info_files[0]  # Use first match found
        else:
            # Fallback to original path
            info_file = config_dir / "information_analysis" / "final"

        data = self._load_json_file(info_file)

        if not data:
            return {}

        result = {}
        ablation_depth_map = self._load_ablation_depth_map(config_dir)

        # Basic MI
        if "basic_mi" in data and isinstance(data["basic_mi"], dict):
            for mi_key, mi_value in data["basic_mi"].items():
                if isinstance(mi_value, (int, float)):
                    # Convert I(E;C) format to mi_E_C
                    # Also handles: I(Vb;C) -> mi_Vb_C, I(E_lin;C) -> mi_E_lin_C
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("Vout", "V")
                        .replace(",", "_")
                    )
                    result[normalized_key] = mi_value

        # Pairwise MI
        if "pairwise_mi" in data and isinstance(data["pairwise_mi"], dict):
            for mi_key, mi_value in data["pairwise_mi"].items():
                if isinstance(mi_value, (int, float)):
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("Vout", "V")
                    )
                    result[normalized_key] = mi_value

        # Conditional MI
        if "conditional_mi" in data and isinstance(data["conditional_mi"], dict):
            for mi_key, mi_value in data["conditional_mi"].items():
                if isinstance(mi_value, (int, float)):
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("|", "_given_")
                        .replace("Vout", "V")
                    )
                    result[normalized_key] = mi_value

        # Layer statistics
        if "layer_statistics" in data and isinstance(data["layer_statistics"], dict):
            layer_names: list[str] = [
                name
                for name in data["layer_statistics"].keys()
                if "inhibitory_cells" not in name
            ]
            depths: list[int] = list(range(len(layer_names)))
            depths.reverse()
            for i, layer_name in enumerate(layer_names):
                layer_data: dict[str, float] = data["layer_statistics"][layer_name]
                if isinstance(layer_data, dict):
                    depth = ablation_depth_map.get(layer_name, depths[i])
                    for mi_key, mi_value in layer_data.items():
                        if isinstance(mi_value, (int, float)):
                            base_key = (
                                mi_key.replace("I(", "layer_mi_")
                                .replace(";", "_")
                                .replace(")", "")
                                .replace(",", "_")
                                .replace("|", "_given_")
                                .replace("Vout", "V")
                                .replace("_mean", "")
                            )
                            result[f"{base_key}_depth{depth}"] = mi_value

        return result

    def _load_ablation_depth_map(self, config_dir: Path) -> dict[str, int]:
        """Map module names to analyzer-provided ablation depths when available."""
        ablation_files = list(config_dir.rglob("ablation_analysis/final.json"))
        if not ablation_files:
            ablation_files = list(config_dir.rglob("ablation_analysis/final"))
        if ablation_files:
            ablation_file = ablation_files[0]
        else:
            ablation_file = config_dir / "ablation_analysis" / "final"

        data = self._load_json_file(ablation_file)
        if not data:
            return {}

        depth_map: dict[str, int] = {}
        layer_data = data.get("layer")
        if not isinstance(layer_data, dict):
            return depth_map

        for targets in layer_data.values():
            if not isinstance(targets, dict):
                continue
            for modules in targets.values():
                if not isinstance(modules, dict):
                    continue
                for module_name, metrics in modules.items():
                    if (
                        isinstance(metrics, dict)
                        and "depth" in metrics
                        and module_name not in depth_map
                    ):
                        try:
                            depth_map[module_name] = int(metrics["depth"])
                        except (TypeError, ValueError):
                            continue
        return depth_map

    def _load_weights(self, config_dir: Path) -> dict:
        """Load weight analysis metrics."""
        # weight_file = config_dir / "weight_analysis" / "final"
        # Search recursively for performance/final.json
        weight_files = list(config_dir.rglob("weight_analysis/final.json"))

        if not weight_files:
            # Try without .json extension
            weight_files = list(config_dir.rglob("weight_analysis/final"))

        if weight_files:
            weight_file = weight_files[0]  # Use first match found
        else:
            # Fallback to original path
            weight_file = config_dir / "weight_analysis" / "final"

        data = self._load_json_file(weight_file)

        if not data:
            return {}

        result = {}

        # Global statistics
        if "global_statistics" in data and isinstance(data["global_statistics"], dict):
            for key, value in data["global_statistics"].items():
                if isinstance(value, (int, float)):
                    result[f"weight_global_{key}"] = value

        # Layer statistics
        if "layer_statistics" in data and isinstance(data["layer_statistics"], dict):
            for layer_name, layer_data in data["layer_statistics"].items():
                if isinstance(layer_data, dict):
                    layer_key = self._format_layer_key(layer_name)
                    for metric, value in layer_data.items():
                        if isinstance(value, (int, float)):
                            safe_metric = metric.replace(".", "_").replace(" ", "_")
                            result[f"weight_{safe_metric}_{layer_key}"] = value

        return result

    def _load_ablation(self, config_dir: Path) -> dict:
        """Load ablation analysis metrics.

        Loads the canonical 3-level ablation schema
        (method -> target -> module -> metrics) and flattens to sweep-ready
        columns via :func:`ablation_columns.flatten_ablation_results`.
        """
        from dendritic_modeling.scripts.sweeps.utils.ablation_columns import (
            flatten_ablation_module_results,
            flatten_ablation_results,
        )

        ablation_files = list(config_dir.rglob("ablation_analysis/final.json"))
        if not ablation_files:
            ablation_files = list(config_dir.rglob("ablation_analysis/final"))
        if ablation_files:
            ablation_file = ablation_files[0]
        else:
            ablation_file = config_dir / "ablation_analysis" / "final"

        data = self._load_json_file(ablation_file)
        if not data:
            return {}

        result: dict[str, float] = {}

        if "layer" in data and isinstance(data["layer"], dict):
            result.update(flatten_ablation_results(data["layer"]))
            result.update(flatten_ablation_module_results(data["layer"]))
        return result

    def _load_snr(self, config_dir: Path) -> dict:
        """Load compartment SNR analysis metrics."""
        # compartment_snr_file = config_dir / "compartment_snr_analysis" / "final"
        # Search recursively for performance/final.json
        compartment_snr_files = list(
            config_dir.rglob("compartment_snr_analysis/final.json")
        )

        if not compartment_snr_files:
            # Try without .json extension
            compartment_snr_files = list(
                config_dir.rglob("compartment_snr_analysis/final")
            )

        if compartment_snr_files:
            compartment_snr_file = compartment_snr_files[0]  # Use first match found
        else:
            # Fallback to original path
            compartment_snr_file = config_dir / "compartment_snr_analysis" / "final"

        data: dict[str, dict[str, dict[str, float]]] = self._load_json_file(
            compartment_snr_file
        )

        if not data:
            return {}

        result = {}

        prefixes = ["exc", "inh", "upstream", "vout"]
        metrics = ["d_max", "d_total", "n_sig_dirs", "ratio_sig_dirs"]

        if "input_analysis" in data.keys():
            input_data = data["input_analysis"]
            for prefix in [
                "exc_inputs",
                "inh_inputs",
                "upstream_inputs",
                "network_exc_inputs",
                "network_inh_outputs",
            ]:
                for metric in metrics:
                    field_name = f"{prefix}_{metric}"
                    if field_name in input_data:
                        result[f"snr_{field_name}"] = input_data[field_name]

        # Handle layer analysis format
        if "layer_analysis" in data.keys():
            if "layer_statistics" in data["layer_analysis"].keys():
                for layer_data in data["layer_analysis"]["layer_statistics"].values():
                    depth = layer_data["depth"]
                    for prefix in prefixes:
                        for metric in metrics:
                            field_name = f"{prefix}_{metric}"
                            if field_name in layer_data:
                                result[f"snr_{field_name}_depth{depth}"] = layer_data[
                                    field_name
                                ]

        return result

    def _load_gain(self, config_dir: Path) -> dict:
        """Load multiplicative gain analysis metrics."""
        multiplicative_gain_files = list(
            config_dir.rglob("multiplicative_gain_analysis/final.json")
        )

        if not multiplicative_gain_files:
            multiplicative_gain_files = list(
                config_dir.rglob("multiplicative_gain_analysis/final")
            )

        if multiplicative_gain_files:
            multiplicative_gain_file = multiplicative_gain_files[0]
        else:
            multiplicative_gain_file = (
                config_dir / "multiplicative_gain_analysis" / "final"
            )

        data = self._load_json_file(multiplicative_gain_file)

        if not data:
            return {}

        result = {}

        gain_factors: list[float] = list(data.get("gain_factors", []))

        raw_scores: dict[str, list[float]] = data.get("raw_scores", {})
        if not raw_scores:
            raw_scores = {
                key: value
                for key, value in data.items()
                if isinstance(value, list)
                and key != "gain_factors"
                and len(value) == len(gain_factors)
            }

        id_scores: dict[str, float] = data.get("ID", {})
        ood_lower_scores: dict[str, float] = data.get("OOD_lower", {})
        ood_upper_scores: dict[str, float] = data.get("OOD_upper", {})

        metrics = list(raw_scores.keys())
        for metric in metrics:
            # collect raw scores
            for i, gain_factor in enumerate(gain_factors[: len(raw_scores[metric])]):
                result[f"mult_gain_{metric}_gf{gain_factor}"] = raw_scores[metric][i]

            # collect id scores
            if metric in id_scores:
                result[f"mult_gain_{metric}_id"] = id_scores[metric]

            # collect ood lower scores
            if metric in ood_lower_scores:
                result[f"mult_gain_{metric}_ood_lower"] = ood_lower_scores[metric]

            # collect ood upper scores
            if metric in ood_upper_scores:
                result[f"mult_gain_{metric}_ood_upper"] = ood_upper_scores[metric]

        return result

    def _load_noise(self, config_dir: Path) -> dict:
        """Load noise perturbation metrics."""
        # noise_file = config_dir / "noise_perturbation_analysis" / "final"
        # Search recursively for performance/final.json
        noise_files = list(config_dir.rglob("noise_perturbation_analysis/final.json"))

        if not noise_files:
            # Try without .json extension
            noise_files = list(config_dir.rglob("noise_perturbation_analysis/final"))

        if noise_files:
            noise_file = noise_files[0]  # Use first match found
        else:
            # Fallback to original path
            noise_file = config_dir / "noise_perturbation_analysis" / "final"

        data = self._load_json_file(noise_file)

        if not data:
            return {}

        result = {}

        # Handle noise perturbation format
        for noise_type, noise_dict in data.items():
            if isinstance(noise_dict, dict):
                for noise_level, noise_data in noise_dict.items():
                    if isinstance(noise_data, dict):
                        for metric, value in noise_data.items():
                            metric_clean = metric.replace("_", "")
                            if isinstance(value, list):
                                for i, v in enumerate(value):
                                    result[
                                        f"noise_{noise_type}_{metric_clean}_{noise_level}_sample{i}"
                                    ] = v
                            else:
                                result[
                                    f"noise_{noise_type}_{metric_clean}_{noise_level}"
                                ] = value

        return result

    def _load_training_summary(self, config_dir: Path) -> dict:
        """Load compact convergence fields without exporting epoch histories."""

        files = list(config_dir.rglob("training_summary.json"))
        if not files:
            files = list(config_dir.rglob("training_summary"))
        if not files:
            return {}
        data = self._load_json_file(files[0])
        if not isinstance(data, dict):
            return {}

        result = {}
        best_epoch = data.get("best_epoch")
        best_loss = data.get("best_loss")
        if isinstance(best_epoch, int) and not isinstance(best_epoch, bool):
            result["training_best_epoch"] = best_epoch
        if isinstance(best_loss, (int, float)) and not isinstance(best_loss, bool):
            result["training_best_valid_loss"] = best_loss

        history_fields = {
            "train_losses": "training_final_train_loss",
            "valid_losses": "training_final_valid_loss",
        }
        history_lengths = []
        for source, destination in history_fields.items():
            history = data.get(source)
            if not isinstance(history, list) or not history:
                continue
            history_lengths.append(len(history))
            final_value = history[-1]
            if isinstance(final_value, (int, float)) and not isinstance(
                final_value, bool
            ):
                result[destination] = final_value
        if history_lengths:
            result["training_epochs_completed"] = max(history_lengths)
        return result

    def _load_nparams(self, config_dir: Path) -> dict:
        """Load parameter count."""
        # nparams_file = config_dir / "nparams"
        # Search recursively for performance/final.json
        nparams_files = list(config_dir.rglob("nparams.json"))

        if not nparams_files:
            # Try without .json extension
            nparams_files = list(config_dir.rglob("nparams"))

        if nparams_files:
            nparams_file = nparams_files[0]  # Use first match found
        else:
            # Fallback to original path
            nparams_file = config_dir / "nparams"

        data = self._load_json_file(nparams_file)

        if data and "nparams" in data:
            return {"nparams": data["nparams"]}

        return {}

    def _load_model_resources(self, config_dir: Path) -> dict:
        """Load exact model-resource accounting saved by the trainer."""
        resource_files = list(config_dir.rglob("model_resources.json"))
        if not resource_files:
            resource_files = list(config_dir.rglob("model_resources"))
        if not resource_files:
            return {}

        data = self._load_json_file(resource_files[0])
        if not isinstance(data, dict):
            return {}

        result = {}
        for key in (
            "total_parameters",
            "trainable_parameters",
            "core_total_parameters",
            "core_trainable_parameters",
            "synapse_candidate_parameters",
            "non_synaptic_parameters",
            "candidate_synapse_slots",
            "active_synapses",
            "persistent_state_scalars_per_sample",
        ):
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"resource_{key}"] = value

        pathway_totals: dict[str, dict[str, float]] = {}
        synapses = data.get("synapses", [])
        if isinstance(synapses, list):
            for record in synapses:
                if not isinstance(record, dict):
                    continue
                pathway = str(record.get("pathway", "unknown"))
                totals = pathway_totals.setdefault(
                    pathway,
                    {
                        "active_synapses": 0.0,
                        "candidate_slots": 0.0,
                        "candidate_parameters": 0.0,
                    },
                )
                for key in tuple(totals):
                    value = record.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[key] += float(value)
        for pathway, totals in pathway_totals.items():
            safe_pathway = re.sub(r"[^a-zA-Z0-9]+", "_", pathway).strip("_")
            for key, value in totals.items():
                result[f"resource_{safe_pathway}_{key}"] = int(value)

        return result

    def _extract_depth(self, layer_name: str) -> int:
        """Extract depth/layer index from layer name."""
        try:
            if "branch_layers." in layer_name:
                return int(layer_name.split("branch_layers.")[-1].split(".")[0])
            elif "layer_" in layer_name:
                return int(layer_name.split("layer_")[-1])
            return 0
        except (ValueError, IndexError):
            return 0

    def _format_layer_key(self, layer_name: str) -> str:
        """Format layer name to a consistent key."""
        if "branch_layers." in layer_name:
            branch_idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
            return f"layer_{branch_idx}"
        elif layer_name == "synthetic_soma":
            return "soma"
        else:
            return layer_name.replace(".", "_")
