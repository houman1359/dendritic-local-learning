"""Public API for conservative, risk-aware FMI-v3 architecture selection.

FMI-v3 leaves the v1/v2 compiler and their frozen artifacts unchanged.  It
uses those teacher estimators as an advisory capacity prior, executes a flat
dendritic incumbent, and exposes a separately preregistered matched local
calibration gate for structural promotion.
"""

from dendritic_modeling.analysis.fmi.prospective_protocol_v3 import (
    EXPOSURE_INDEX_SCHEMA,
    FROZEN_PROTOCOL_SCHEMA,
    PROTOCOL_REGISTRATION_SCHEMA,
    freeze_prospective_protocol_v3,
    scan_site_exposures,
    write_frozen_protocol,
)
from dendritic_modeling.analysis.fmi.selection_policy_v3 import (
    AXIS_SOURCES,
    CAPACITY_AXES,
    LOCAL_CALIBRATION_AXES,
    V3_ADVISORY_SCHEMA,
    V3_PROMOTION_DECISION_SCHEMA,
    V3_PROMOTION_POLICY_SCHEMA,
    MorphologyPromotionPolicyV3,
    apply_morphology_promotion_v3,
    build_fmi_v3_advisory,
    build_morphology_promotion_policy_v3,
)

__all__ = [
    "AXIS_SOURCES",
    "CAPACITY_AXES",
    "EXPOSURE_INDEX_SCHEMA",
    "FROZEN_PROTOCOL_SCHEMA",
    "LOCAL_CALIBRATION_AXES",
    "PROTOCOL_REGISTRATION_SCHEMA",
    "V3_ADVISORY_SCHEMA",
    "V3_PROMOTION_DECISION_SCHEMA",
    "V3_PROMOTION_POLICY_SCHEMA",
    "MorphologyPromotionPolicyV3",
    "apply_morphology_promotion_v3",
    "build_fmi_v3_advisory",
    "build_morphology_promotion_policy_v3",
    "freeze_prospective_protocol_v3",
    "scan_site_exposures",
    "write_frozen_protocol",
]
