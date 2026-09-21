#!/usr/bin/env python3
"""
Model Architecture Validator for Dendritic Networks
==================================================

This module validates the structural/architectural parameters of dendritic models
BEFORE model creation. It ensures:
- Layer dimensions are compatible
- Synapse counts match layer configurations
- TopK constraints are satisfied
- Branch factors are consistent
- Transfer learning dimensions align
- Biological plausibility constraints

This is separate from training stability validation and runs during model setup.
"""

import logging
from typing import Any, Union

from dendritic_modeling.datasets.dimensions import flat_dataset_input_dim

logger = logging.getLogger(__name__)

DEFAULT_MULTIXOR_BITS = 4
DEFAULT_UNKNOWN_INPUT_DIM = 64
DATASET_INPUT_DIMS = {
    dataset_name: flat_dataset_input_dim(
        dataset_name,
        default_input_dim=DEFAULT_UNKNOWN_INPUT_DIM,
    )
    for dataset_name in ("cifar10", "imagenet", "mnist")
}


def _get_config_value(
    config: Union[dict, object], key: str, default: Any = None
) -> Any:
    """Safely get a value from a config object, handling both dict and attribute access."""
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


def _as_plain_dict(config_section: Union[dict, object]) -> dict[str, Any]:
    """Convert a config section to a plain dict while preserving existing behavior."""
    if hasattr(config_section, "__dict__"):
        return dict(config_section.__dict__)
    return dict(config_section)


def _build_network_config(config) -> dict[str, Any]:
    """Build the merged network config used by architecture validation."""
    if not (hasattr(config, "model") and hasattr(config.model, "core")):
        return {}

    core_config = config.model.core
    network_config: dict[str, Any] = {}

    for section_name in ("architecture", "connectivity"):
        if hasattr(core_config, section_name):
            network_config.update(_as_plain_dict(getattr(core_config, section_name)))

    if hasattr(core_config, "transfer"):
        network_config["transfer_params"] = _as_plain_dict(core_config.transfer)

    if hasattr(core_config, "morphology"):
        network_config.update(_as_plain_dict(core_config.morphology))

    return network_config


def _resolve_input_dim(
    *,
    dataset_name: str,
    task_params: dict[str, Any],
    network_config: dict[str, Any],
    transfer_params: dict[str, Any],
) -> int:
    """Resolve the validator input dimension from dataset and config metadata."""
    if dataset_name in DATASET_INPUT_DIMS:
        return DATASET_INPUT_DIMS[dataset_name]

    if dataset_name == "multixor":
        n_bits = task_params.get("n_bits", DEFAULT_MULTIXOR_BITS)
        return n_bits * 2

    return (
        network_config.get("input_dim")
        or task_params.get("input_dim")
        or transfer_params.get("excitatory_dim")
        or DEFAULT_UNKNOWN_INPUT_DIM
    )


def _layer_value(values: list[int], layer_idx: int, *, default: int = 1) -> int:
    """Return a layer-specific value, reusing the final value as the legacy fallback."""
    if layer_idx < len(values):
        return values[layer_idx]
    if values:
        return values[-1]
    return default


def _has_inhibitory_neurons(inhibitory_layer_sizes: list[int]) -> bool:
    """Return whether any inhibitory layer has a positive size."""
    return any(size for size in inhibitory_layer_sizes if size and size > 0)


def _same_layer_inhibitory_size(
    inhibitory_layer_sizes: list[int], layer_idx: int
) -> int:
    """Return the positive inhibitory size for a layer, or zero when absent."""
    if layer_idx >= len(inhibitory_layer_sizes):
        return 0

    size = inhibitory_layer_sizes[layer_idx]
    return size if size and size > 0 else 0


def _previous_layer_inhibitory_size(
    inhibitory_layer_sizes: list[int], layer_idx: int
) -> int:
    """Return the previous layer's inhibitory size with the legacy absent-layer fallback."""
    previous_idx = layer_idx - 1
    if 0 <= previous_idx < len(inhibitory_layer_sizes):
        return inhibitory_layer_sizes[previous_idx]
    return 0


def _initial_layer_input_dims(
    *,
    input_dim: int,
    independent_pathways: bool,
    excitatory_dim: int | None,
    inhibitory_dim: int | None,
) -> tuple[int, int]:
    """Resolve first-layer excitatory and inhibitory input dimensions."""
    if independent_pathways:
        return (
            excitatory_dim if excitatory_dim else input_dim // 2,
            inhibitory_dim if inhibitory_dim else input_dim // 2,
        )

    return input_dim, input_dim


def _layer_input_dims(
    *,
    layer_idx: int,
    input_dim: int,
    independent_pathways: bool,
    excitatory_dim: int | None,
    inhibitory_dim: int | None,
    excitatory_layer_sizes: list[int],
    inhibitory_layer_sizes: list[int],
) -> tuple[int, int]:
    """Resolve excitatory and inhibitory input dimensions for a validator layer."""
    if layer_idx == 0:
        return _initial_layer_input_dims(
            input_dim=input_dim,
            independent_pathways=independent_pathways,
            excitatory_dim=excitatory_dim,
            inhibitory_dim=inhibitory_dim,
        )

    return (
        excitatory_layer_sizes[layer_idx - 1],
        _previous_layer_inhibitory_size(inhibitory_layer_sizes, layer_idx),
    )


def _ie_inhibitory_source_dim(
    *,
    layer_idx: int,
    input_mode: int,
    independent_pathways: bool,
    input_dim: int,
    excitatory_dim: int | None,
    inhibitory_dim: int | None,
    inhibitory_layer_sizes: list[int],
) -> int:
    """Resolve the source dimension used to validate IE synapse counts."""
    if input_mode == 1:
        if independent_pathways:
            return inhibitory_dim if inhibitory_dim else input_dim // 2

        return excitatory_dim if excitatory_dim else input_dim

    if layer_idx == 0:
        return _same_layer_inhibitory_size(inhibitory_layer_sizes, layer_idx)

    return _previous_layer_inhibitory_size(inhibitory_layer_sizes, layer_idx)


class ModelArchitectureValidator:
    """Validates structural/architectural parameters for dendritic networks"""

    def __init__(self, config):
        self.config = config
        self.network_config = _build_network_config(config)

        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _get_config_value(self, key: str, default=None):
        """Safely get configuration value from Config object or dict."""
        return _get_config_value(self.config, key, default)

    def validate(self) -> bool:
        """Validate all constraints and return True if config is valid"""
        logger.info("Starting configuration validation")

        # Extract key parameters
        self.extract_parameters()

        # Run all checks
        for check in self._validation_checks():
            check()

        # Report results
        self.report_results()

        return len(self.errors) == 0

    def _validation_checks(self):
        """Return validation checks in their legacy execution order."""
        return (
            self.check_input_mode_consistency,
            self.check_layer_dimensions,
            self.check_synapses_per_branch,
            self.check_topk_constraints,
            self.check_branch_factors,
            self.check_empty_lists,
            self.check_transfer_dimensions,
            self.check_biological_plausibility,
        )

    def extract_parameters(self):
        """Extract key parameters from config"""
        # Get transfer params first to determine input_dim
        self.transfer_params = self.network_config.get("transfer_params", {})
        self.input_mode = self.transfer_params.get("input_mode", 0)
        self.independent_pathways = self.transfer_params.get(
            "independent_pathways", False
        )

        # Get dataset name from data config
        data_config = self._get_config_value("data", {})
        dataset_name = _get_config_value(data_config, "dataset_name", "")

        task_config = self._get_config_value("task", {})
        task_params = _get_config_value(task_config, "parameters", {})

        self.input_dim = _resolve_input_dim(
            dataset_name=dataset_name,
            task_params=task_params,
            network_config=self.network_config,
            transfer_params=self.transfer_params,
        )

        self.excitatory_layer_sizes = self.network_config.get(
            "excitatory_layer_sizes", []
        )
        self.inhibitory_layer_sizes = self.network_config.get(
            "inhibitory_layer_sizes", []
        )

        self.excitatory_branch_factors = self.network_config.get(
            "excitatory_branch_factors", []
        )
        self.inhibitory_branch_factors = self.network_config.get(
            "inhibitory_branch_factors", []
        )

        self.ee_synapses = self.network_config.get(
            "ee_synapses_per_branch_per_layer", []
        )
        self.ei_synapses = self.network_config.get(
            "ei_synapses_per_branch_per_layer", []
        )
        self.ie_synapses = self.network_config.get(
            "ie_synapses_per_branch_per_layer", []
        )
        self.ii_synapses = self.network_config.get(
            "ii_synapses_per_branch_per_layer", []
        )

        self.num_layers = len(self.excitatory_layer_sizes)

        # Transfer dimensions
        self.excitatory_dim = self.transfer_params.get("excitatory_dim", None)
        self.inhibitory_dim = self.transfer_params.get("inhibitory_dim", None)

    def check_input_mode_consistency(self):
        """Check input-mode values that are invalid independent of routing split."""
        if self.input_mode not in {0, 1}:
            self.errors.append(f"input_mode must be 0 or 1, got {self.input_mode}")

    def check_layer_dimensions(self):
        """Check layer size consistency"""
        exc_layers = len(self.excitatory_layer_sizes)
        inh_layers = len(self.inhibitory_layer_sizes)

        if self.input_mode == 1:
            # For input_mode=1, inhibitory layers can be:
            # - Empty (no inhibitory layers)
            # - Equal to excitatory layers (allowing 0 values for some layers)
            # - One less than excitatory layers
            if inh_layers not in [0, exc_layers - 1, exc_layers]:
                self.errors.append(
                    f"For input_mode=1, inhibitory layer count ({inh_layers}) must be "
                    f"0, {exc_layers-1}, or {exc_layers} (excitatory count: {exc_layers})"
                )
        else:
            # For other input modes, they should match
            if exc_layers != inh_layers:
                self.errors.append(
                    f"Mismatch in layer counts - "
                    f"excitatory: {exc_layers}, "
                    f"inhibitory: {inh_layers}"
                )

        # Check for zero-sized layers
        for i, exc in enumerate(self.excitatory_layer_sizes):
            if exc <= 0:
                self.errors.append(f"Layer {i} has non-positive excitatory size: {exc}")

        # Only check inhibitory if they exist
        for i, inh in enumerate(self.inhibitory_layer_sizes):
            if inh < 0:
                self.errors.append(f"Layer {i} has negative inhibitory size: {inh}")

    def check_synapses_per_branch(self):
        """Check synapses per branch constraints"""
        # Determine expected lengths (empty lists will be extended with default=1)
        expected_len = self.num_layers

        # Check each synapse type
        for layer_idx in range(expected_len):
            exc_input, inh_input = _layer_input_dims(
                layer_idx=layer_idx,
                input_dim=self.input_dim,
                independent_pathways=self.independent_pathways,
                excitatory_dim=self.excitatory_dim,
                inhibitory_dim=self.inhibitory_dim,
                excitatory_layer_sizes=self.excitatory_layer_sizes,
                inhibitory_layer_sizes=self.inhibitory_layer_sizes,
            )

            # Inhibitory input to excitatory cells comes from same layer (if exists)
            inh_to_exc = _same_layer_inhibitory_size(
                self.inhibitory_layer_sizes, layer_idx
            )

            # Check EE synapses
            ee_val = _layer_value(self.ee_synapses, layer_idx)
            if ee_val > exc_input:
                self.errors.append(
                    f"Layer {layer_idx} - ee_synapses ({ee_val}) > "
                    f"excitatory input dim ({exc_input})"
                )

            # Check EI synapses (only if ei_synapses list is not empty)
            if self.ei_synapses:
                ei_val = _layer_value(self.ei_synapses, layer_idx)
                if inh_to_exc > 0 and ei_val > inh_to_exc:
                    self.errors.append(
                        f"Layer {layer_idx} - ei_synapses ({ei_val}) > "
                        f"inhibitory neurons in layer ({inh_to_exc})"
                    )
                elif inh_to_exc == 0 and ei_val > 0:
                    self.errors.append(
                        f"Layer {layer_idx} has ei_synapses={ei_val} but no inhibitory neurons exist"
                    )

            # Check IE synapses (inhibitory TO excitatory connections) - only if list is not empty
            if self.ie_synapses:
                ie_val = _layer_value(self.ie_synapses, layer_idx)
                inhibitory_source_dim = _ie_inhibitory_source_dim(
                    layer_idx=layer_idx,
                    input_mode=self.input_mode,
                    independent_pathways=self.independent_pathways,
                    input_dim=self.input_dim,
                    excitatory_dim=self.excitatory_dim,
                    inhibitory_dim=self.inhibitory_dim,
                    inhibitory_layer_sizes=self.inhibitory_layer_sizes,
                )

                # Check IE constraint
                if inhibitory_source_dim > 0 and ie_val > inhibitory_source_dim:
                    self.errors.append(
                        f"Layer {layer_idx} - ie_synapses ({ie_val}) > "
                        f"inhibitory input dim ({inhibitory_source_dim})"
                    )
                elif inhibitory_source_dim == 0 and ie_val > 0:
                    self.errors.append(
                        f"Layer {layer_idx} has ie_synapses={ie_val} but no inhibitory input available"
                    )

            # Check II synapses (only if inhibitory neurons exist)
            if (
                layer_idx < len(self.inhibitory_layer_sizes)
                and self.inhibitory_layer_sizes[layer_idx] > 0
            ):
                ii_val = _layer_value(self.ii_synapses, layer_idx)
                if ii_val > inh_input:
                    self.errors.append(
                        f"Layer {layer_idx} - ii_synapses ({ii_val}) > "
                        f"inhibitory input dim ({inh_input})"
                    )

    def check_topk_constraints(self):
        """Check TopK constraints"""
        # EE synapses must always be positive (excitatory neurons always exist)
        for i, val in enumerate(self.ee_synapses):
            if val <= 0:
                self.errors.append(f"ee_synapses[{i}] must be positive, got {val}")

        # Other synapse types can be 0 when target neurons don't exist
        for syn_type, syn_list in [
            ("ei", self.ei_synapses),
            ("ie", self.ie_synapses),
            ("ii", self.ii_synapses),
        ]:
            for i, val in enumerate(syn_list):
                if val < 0:  # Allow 0, but not negative
                    self.errors.append(
                        f"{syn_type}_synapses[{i}] must be non-negative, got {val}"
                    )
                elif val == 0:
                    # Check if 0 makes sense given the layer configuration
                    if syn_type == "ei" and i < len(self.inhibitory_layer_sizes):
                        # EI: excitatory TO inhibitory connections
                        if self.inhibitory_layer_sizes[i] > 0:
                            self.warnings.append(
                                f"ei_synapses[{i}] is 0 but layer {i} has "
                                f"{self.inhibitory_layer_sizes[i]} inhibitory neurons"
                            )
                    elif syn_type == "ie" and i < len(self.inhibitory_layer_sizes):
                        # IE: inhibitory TO excitatory connections
                        # In input_mode=1, first layer gets inhibitory input from transfer function
                        # Subsequent layers get inhibitory input from previous layer's inhibitory neurons
                        if i == 0:
                            # First layer can have IE synapses from transfer function
                            if self.input_mode == 1:
                                self.warnings.append(
                                    "ie_synapses[0] is 0 - layer 0 excitatory neurons won't receive "
                                    "inhibitory input from transfer function"
                                )
                        else:
                            # Other layers need previous layer to have inhibitory neurons
                            if (
                                i - 1 < len(self.inhibitory_layer_sizes)
                                and self.inhibitory_layer_sizes[i - 1] > 0
                            ):
                                self.warnings.append(
                                    f"ie_synapses[{i}] is 0 but previous layer {i-1} has "
                                    f"{self.inhibitory_layer_sizes[i-1]} inhibitory neurons"
                                )
                    elif syn_type == "ii" and i < len(self.inhibitory_layer_sizes):
                        # For II synapses, check if previous layer had inhibitory neurons
                        if i > 0 and i - 1 < len(self.inhibitory_layer_sizes):
                            if (
                                self.inhibitory_layer_sizes[i - 1] > 0
                                and self.inhibitory_layer_sizes[i] > 0
                            ):
                                self.warnings.append(
                                    f"ii_synapses[{i}] is 0 but both layer {i-1} and {i} have inhibitory neurons"
                                )

    def check_branch_factors(self):
        """Check branch factor constraints"""
        # Branch factors represent hierarchical branching structure
        # e.g., [2, 2] means 2 branches that each split into 2 more (total 4 branches)
        # They don't need to match the number of layers

        # Just check that they are valid positive integers
        for i, bf in enumerate(self.excitatory_branch_factors):
            if bf <= 0:
                self.errors.append(
                    f"excitatory_branch_factors[{i}] must be positive, got {bf}"
                )

        for i, bf in enumerate(self.inhibitory_branch_factors):
            if bf <= 0:
                self.errors.append(
                    f"inhibitory_branch_factors[{i}] must be positive, got {bf}"
                )

    def check_empty_lists(self):
        """Check for empty synapse lists and validate they're appropriate for the configuration"""
        has_inhibitory_neurons = _has_inhibitory_neurons(self.inhibitory_layer_sizes)

        # EE synapses are always required
        if not self.ee_synapses:
            self.errors.append(
                "ee_synapses_per_branch_per_layer cannot be empty - excitatory connections are required"
            )

        # EI synapses: only allow empty if no inhibitory neurons exist
        if not self.ei_synapses:
            if has_inhibitory_neurons:
                self.errors.append(
                    "ei_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory neurons, empty ei_synapses is valid (no E-to-I connections needed)

        # IE synapses: check based on input_mode and layer configuration
        if not self.ie_synapses:
            if self.input_mode == 1:
                # input_mode=1 provides inhibitory input, so IE connections are expected
                self.errors.append(
                    "ie_synapses_per_branch_per_layer cannot be empty for input_mode=1 (inhibitory input available)"
                )
            elif has_inhibitory_neurons:
                # Other modes with inhibitory neurons should have IE connections
                self.errors.append(
                    "ie_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory input available, empty ie_synapses is valid

        # II synapses: only allow empty if no inhibitory neurons exist
        if not self.ii_synapses:
            if has_inhibitory_neurons:
                self.errors.append(
                    "ii_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory neurons, empty ii_synapses is valid (no I-to-I connections needed)

    def check_transfer_dimensions(self):
        """Check transfer function dimension settings"""
        if self.independent_pathways:
            # With independent pathways, check if dimensions are auto-set
            if not self.excitatory_dim or not self.inhibitory_dim:
                self.warnings.append(
                    "Transfer dimensions not set, will auto-set to input_dim/2 each"
                )
        else:
            # With shared pathway, dimensions should match input
            if self.excitatory_dim and self.excitatory_dim != self.input_dim:
                self.warnings.append(
                    f"With shared pathway, excitatory_dim ({self.excitatory_dim}) "
                    f"should match input_dim ({self.input_dim})"
                )

    def check_biological_plausibility(self):
        """Check biological plausibility constraints"""
        # Warn if too many synapses per branch
        for syn_type, syn_list in [
            ("ee", self.ee_synapses),
            ("ei", self.ei_synapses),
            ("ie", self.ie_synapses),
            ("ii", self.ii_synapses),
        ]:
            for i, val in enumerate(syn_list):
                if val > 100:
                    self.warnings.append(
                        f"{syn_type}_synapses[{i}] = {val} is very high. "
                        "Biological neurons typically have 10-50 synapses per dendrite"
                    )

    def report_results(self):
        """Report all errors and warnings"""
        logger.info("Configuration validation summary:")
        logger.info(f"  Input mode: {self.input_mode}")
        logger.info(f"  Independent pathways: {self.independent_pathways}")
        logger.info(f"  Number of layers: {self.num_layers}")

        if self.errors:
            logger.error(f"Configuration validation found {len(self.errors)} error(s):")
            for error in self.errors:
                logger.error(f"  {error}")

        if self.warnings:
            logger.warning(
                f"Configuration validation found {len(self.warnings)} warning(s):"
            )
            for warning in self.warnings:
                logger.warning(f"  {warning}")

        if not self.errors and not self.warnings:
            logger.info("Configuration validation passed - all constraints satisfied")
        elif not self.errors:
            logger.info("Configuration validation passed with warnings")


def validate_model_architecture(config: dict) -> bool:
    """Validate model architecture configuration.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if architecture is valid
    """
    validator = ModelArchitectureValidator(config)
    return validator.validate()
