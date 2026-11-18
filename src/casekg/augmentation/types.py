"""Data structures and exceptions for the CASEKG Data Augmentation package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class AugmentationError(Exception):
    """Base exception for augmentation errors."""


class SpanAlignmentError(AugmentationError):
    """Raised when an entity or clause span fails exact substring alignment."""


class SafetyValidationError(AugmentationError):
    """Raised when an augmented record violates clinical or offset invariants."""


@dataclass(frozen=True)
class ClauseSpan:
    """Document-level coordinates and section assignment for a clinical clause."""

    clause_uid: str
    unit_index: int
    primary_section: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class EntitySpan:
    """Document-level coordinates and ontology metadata for a clinical mention."""

    span_id: str
    start: int
    end: int
    text: str
    clause_uid: str
    curie: str = ""
    semantic_type: str = ""
    role: str = ""
    assertion: str = "PRESENT"
    source: str = "semantic_graph"
    protected: bool = False


@dataclass(frozen=True)
class HydratedDocument:
    """A clinical case record hydrated with verified document-level clause and entity spans."""

    document_id: str
    dataset: str
    disease_family_label: str
    lineage_group_id: str
    text_field: str
    text: str
    clauses: tuple[ClauseSpan, ...]
    spans: tuple[EntitySpan, ...]
    row: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ReplacementEdit:
    """A single span replacement operation in original document coordinates."""

    start: int
    end: int
    original_text: str
    replacement_text: str
    curie: str
    semantic_type: str
    clause_uid: str
    reason: str


@dataclass(frozen=True)
class AugmentedRecord:
    """A verified augmented clinical case variant with updated spans and audit trail."""

    variant_id: str
    parent_document_id: str
    dataset: str
    disease_family_label: str
    lineage_group_id: str
    method: str
    seed: int
    original_text: str
    augmented_text: str
    clauses: tuple[ClauseSpan, ...]
    spans: tuple[EntitySpan, ...]
    edits: tuple[ReplacementEdit, ...]
    row: dict[str, Any] = field(repr=False)
    is_augmented: bool = True
