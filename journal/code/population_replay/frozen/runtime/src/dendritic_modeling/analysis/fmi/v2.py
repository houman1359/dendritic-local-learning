"""Versioned public API for the PopulationNetwork FMI-v2 compiler.

This module is intentionally separate from :mod:`dendritic_modeling.analysis.fmi`:
the latter is byte-frozen by the prospective-v1 protocol and must retain its
recorded SHA-256. New experiments should import the v2 API from here.
"""

from dendritic_modeling.analysis.fmi.capture import (
    CapturedTeacherBoundary,
    TeacherBoundaryTarget,
    capture_teacher_boundaries,
    profile_captured_boundaries,
)
from dendritic_modeling.analysis.fmi.profiler import (
    FMIProfilerConfig,
    profile_teacher_component,
    profiler_config_from_preset,
)
from dendritic_modeling.analysis.fmi.reference_grid_v2 import (
    DEFAULT_RESOURCE_METRICS,
    score_reference_grid,
)
from dendritic_modeling.analysis.fmi.synthetic_teachers import (
    PLANTED_TEACHER_KINDS,
    PlantedPositiveEITeacher,
    PlantedTeacherSpec,
    build_planted_teacher,
)
from dendritic_modeling.networks.architectures.replacement.compiler import (
    CompiledReplacementPlan,
    compile_replacement_candidate,
)
from dendritic_modeling.networks.architectures.replacement.selection import (
    ResolvedReplacementSelection,
    load_fmi_fingerprint,
    resolve_replacement_selection,
)

__all__ = [
    "DEFAULT_RESOURCE_METRICS",
    "PLANTED_TEACHER_KINDS",
    "CapturedTeacherBoundary",
    "CompiledReplacementPlan",
    "FMIProfilerConfig",
    "PlantedPositiveEITeacher",
    "PlantedTeacherSpec",
    "ResolvedReplacementSelection",
    "TeacherBoundaryTarget",
    "build_planted_teacher",
    "capture_teacher_boundaries",
    "compile_replacement_candidate",
    "load_fmi_fingerprint",
    "profile_captured_boundaries",
    "profile_teacher_component",
    "profiler_config_from_preset",
    "resolve_replacement_selection",
    "score_reference_grid",
]
