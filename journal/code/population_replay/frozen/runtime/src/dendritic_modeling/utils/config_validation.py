"""
Unified Configuration Validation for Dendritic Networks
======================================================

This module provides a unified entry point for all configuration validation,
combining both model architecture and training stability validation.

Usage:
    from dendritic_modeling.utils.config_validation import validate_full_config

    # Validate everything
    is_valid = validate_full_config(config)

    # Or validate specific aspects
    arch_valid = validate_model_architecture(config)
    training_valid = validate_training_stability(config)
"""

import logging

from .model_architecture_validator import validate_model_architecture
from .training_stability_validator import validate_training_stability

logger = logging.getLogger(__name__)


def _run_architecture_validation(config: dict) -> bool:
    """Run required model-architecture validation and log the existing outcome."""
    logger.info("Validating model architecture...")
    arch_valid = validate_model_architecture(config)

    if not arch_valid:
        logger.error("Model architecture validation failed - cannot continue")
        return False

    logger.info("Model architecture validation passed")
    return True


def _run_training_validation(config: dict, *, strict: bool) -> bool:
    """Run training-stability validation and apply strict/non-strict policy."""
    logger.info("Validating training stability...")
    training_valid = validate_training_stability(config)

    if training_valid:
        logger.info("Training stability validation passed")
        return True

    if strict:
        logger.error("Training stability validation failed in strict mode")
        return False

    logger.warning("Training stability validation failed - proceeding with warnings")
    return True


def validate_full_config(config: dict, strict: bool = True) -> bool:
    """
    Perform complete configuration validation including both architecture and training stability.

    Args:
        config: Configuration dictionary
        strict: If True, both validations must pass. If False, only architecture validation is required.

    Returns:
        bool: True if configuration is valid according to the strict setting
    """
    logger.info("Starting full configuration validation")

    # 1. Model Architecture Validation (required)
    if not _run_architecture_validation(config):
        return False

    # 2. Training Stability Validation (warnings only unless strict=True)
    if not _run_training_validation(config, strict=strict):
        return False

    logger.info("Full configuration validation completed successfully")
    return True


def validate_architecture_only(config: dict) -> bool:
    """Validate only model architecture (faster, for sweeps/tests)."""
    return validate_model_architecture(config)


def validate_training_only(config: dict) -> bool:
    """Validate only training stability (for training scripts)."""
    return validate_training_stability(config)


# Summary function for logging
def get_validation_summary(config: dict) -> dict[str, bool]:
    """
    Get validation results for both architecture and training stability.

    Returns:
        dict: Summary of validation results
    """
    return {
        "architecture_valid": validate_model_architecture(config),
        "training_stable": validate_training_stability(config),
    }
