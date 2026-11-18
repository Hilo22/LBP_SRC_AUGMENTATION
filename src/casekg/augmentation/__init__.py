"""CASEKG Clinical Data Augmentation Package."""
from __future__ import annotations

from casekg.augmentation.hydrator import (
    BLOCKED_ETIOLOGY_TOKENS,
    SpanHydrator,
    normalize_surface,
)
from casekg.augmentation.pipeline import (
    AugmentationPipeline,
    reorder_adjacent_clauses,
    select_fold_variants,
)
from casekg.augmentation.synonym_swapper import (
    ALLOWED_ONTOLOGY_PREFIXES,
    SynonymSwapper,
    apply_replacements,
    is_valid_replacement_candidate,
    preserve_initial_case,
)
from casekg.augmentation.types import (
    AugmentedRecord,
    AugmentationError,
    ClauseSpan,
    EntitySpan,
    HydratedDocument,
    ReplacementEdit,
    SafetyValidationError,
    SpanAlignmentError,
)

__all__ = [
    "ALLOWED_ONTOLOGY_PREFIXES",
    "AugmentedRecord",
    "AugmentationError",
    "AugmentationPipeline",
    "BLOCKED_ETIOLOGY_TOKENS",
    "ClauseSpan",
    "EntitySpan",
    "HydratedDocument",
    "ReplacementEdit",
    "SafetyValidationError",
    "SpanAlignmentError",
    "SpanHydrator",
    "SynonymSwapper",
    "apply_replacements",
    "is_valid_replacement_candidate",
    "normalize_surface",
    "preserve_initial_case",
    "reorder_adjacent_clauses",
    "select_fold_variants",
]
