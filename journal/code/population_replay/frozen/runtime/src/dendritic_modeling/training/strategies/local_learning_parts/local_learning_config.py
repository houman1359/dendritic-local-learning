"""Compatibility wrapper for local-learning config normalization."""

from dendritic_modeling.config.local_learning import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)

__all__ = [
    "build_local_rule_config",
    "coerce_legacy_local_rule_config",
    "section_to_dict",
]
