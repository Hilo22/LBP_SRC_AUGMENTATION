"""AugmentationPipeline: combines SpanHydrator, SynonymSwapper, clause reordering & invariant checks."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

from casekg.augmentation.hydrator import (
    BLOCKED_ETIOLOGY_TOKENS,
    SpanHydrator,
    normalize_surface,
)
from casekg.augmentation.synonym_swapper import (
    ALLOWED_ONTOLOGY_PREFIXES,
    SynonymSwapper,
    apply_replacements,
    is_valid_replacement_candidate,
)
from casekg.augmentation.types import (
    AugmentedRecord,
    ClauseSpan,
    EntitySpan,
    HydratedDocument,
    ReplacementEdit,
    SafetyValidationError,
    SpanAlignmentError,
)

ANAPHORIC_OR_TEMPORAL_START_RE = re.compile(
    r"^(?:he|she|they|his|her|their|this|these|that|those|it|its|"
    r"however|subsequently|later|thereafter|moreover|furthermore|meanwhile|"
    r"two|three|four|five|six|seven|one|following|after|before|during|"
    r"repeat|further|in addition|consequently|therefore|thus)\b",
    re.I,
)


def reorder_adjacent_clauses(
    text: str,
    clauses: Sequence[ClauseSpan],
    spans: Sequence[EntitySpan],
    rng: random.Random,
) -> tuple[str, tuple[ClauseSpan, ...], tuple[EntitySpan, ...], bool]:
    """Safely swap one pair of independent adjacent clauses within the same clinical section.

    Preserves 100% of entity and clause span offsets. Never moves unit_index 0
    (patient presentation opener) or clauses starting with anaphoric/temporal cues.
    """
    if len(clauses) < 3:
        return text, tuple(clauses), tuple(spans), False

    candidates: list[int] = []
    for idx in range(1, len(clauses) - 1):
        c1 = clauses[idx]
        c2 = clauses[idx + 1]
        if c1.primary_section != c2.primary_section:
            continue
        if c1.primary_section not in ("EXAMINATION_INVESTIGATION", "PAST_MEDICAL_HISTORY"):
            continue
        gap = text[c1.end:c2.start]
        if not gap or gap.strip() != "":
            continue
        if ANAPHORIC_OR_TEMPORAL_START_RE.search(c1.text.strip()):
            continue
        if ANAPHORIC_OR_TEMPORAL_START_RE.search(c2.text.strip()):
            continue
        # Ensure no span crosses outside c1 or c2
        spans_in_window = [
            sp for sp in spans if sp.start < c2.end and sp.end > c1.start
        ]
        if any(
            not ((c1.start <= sp.start and sp.end <= c1.end) or (c2.start <= sp.start and sp.end <= c2.end))
            for sp in spans_in_window
        ):
            continue
        candidates.append(idx)

    if not candidates:
        return text, tuple(clauses), tuple(spans), False

    chosen_idx = rng.choice(candidates)
    c1 = clauses[chosen_idx]
    c2 = clauses[chosen_idx + 1]
    gap = text[c1.end:c2.start]

    # New text replaces [c1.start : c2.end] with c2.text + gap + c1.text
    new_text = text[:c1.start] + c2.text + gap + c1.text + text[c2.end:]
    c2_shift = c1.start - c2.start  # negative shift (moves left to c1.start)
    c1_shift = (c1.start + len(c2.text) + len(gap)) - c1.start  # positive shift (moves right)

    new_clauses: list[ClauseSpan] = []
    for i, cl in enumerate(clauses):
        if i == chosen_idx:
            ns = cl.start + c1_shift
            ne = cl.end + c1_shift
            new_clauses.append(
                ClauseSpan(cl.clause_uid, cl.unit_index, cl.primary_section, ns, ne, new_text[ns:ne])
            )
        elif i == chosen_idx + 1:
            ns = cl.start + c2_shift
            ne = cl.end + c2_shift
            new_clauses.append(
                ClauseSpan(cl.clause_uid, cl.unit_index, cl.primary_section, ns, ne, new_text[ns:ne])
            )
        else:
            new_clauses.append(cl)
    new_clauses.sort(key=lambda c: (c.start, c.end))

    new_spans: list[EntitySpan] = []
    for sp in spans:
        if c1.start <= sp.start and sp.end <= c1.end:
            ns = sp.start + c1_shift
            ne = sp.end + c1_shift
        elif c2.start <= sp.start and sp.end <= c2.end:
            ns = sp.start + c2_shift
            ne = sp.end + c2_shift
        else:
            ns, ne = sp.start, sp.end
        updated = EntitySpan(
            span_id=sp.span_id,
            start=ns,
            end=ne,
            text=sp.text,
            clause_uid=sp.clause_uid,
            curie=sp.curie,
            semantic_type=sp.semantic_type,
            role=sp.role,
            assertion=sp.assertion,
            source=sp.source,
            protected=sp.protected,
        )
        if new_text[updated.start:updated.end] != updated.text:
            raise SpanAlignmentError(f"Clause reorder span mismatch for {updated.span_id}")
        new_spans.append(updated)
    new_spans.sort(key=lambda s: (s.start, s.end))

    return new_text, tuple(new_clauses), tuple(new_spans), True


class AugmentationPipeline:
    """End-to-end deterministic clinical Data Augmentation pipeline with invariant checks."""

    def __init__(
        self,
        hydrator: SpanHydrator | None = None,
        swapper: SynonymSwapper | None = None,
        *,
        enable_clause_reorder: bool = False,
    ) -> None:
        self.hydrator = hydrator or SpanHydrator()
        self.swapper = swapper or SynonymSwapper()
        self.enable_clause_reorder = enable_clause_reorder

    def validate_invariants(
        self,
        parent_doc: HydratedDocument,
        record: AugmentedRecord,
    ) -> None:
        """Verify offset alignment, clinical safety, label leakage, and JSONL validity.

        Raises `SafetyValidationError` or `SpanAlignmentError` on any violation.
        """
        aug_text = record.augmented_text
        orig_text = parent_doc.text

        # 1. Verify 100% entity span offset alignment and span count conservation
        if len(record.spans) != len(parent_doc.spans):
            raise SafetyValidationError(
                f"Span count changed in {record.variant_id}: {len(parent_doc.spans)} -> {len(record.spans)}"
            )
        parent_by_id = {sp.span_id: sp for sp in parent_doc.spans}
        for sp in record.spans:
            if sp.start < 0 or sp.end > len(aug_text) or sp.start >= sp.end:
                raise SpanAlignmentError(f"Invalid span bounds [{sp.start}:{sp.end}] in {record.variant_id}")
            actual = aug_text[sp.start:sp.end]
            if actual != sp.text:
                raise SpanAlignmentError(
                    f"Entity offset drift in {record.variant_id} ({sp.span_id}): "
                    f"expected {sp.text!r}, got {actual!r}"
                )
            p_sp = parent_by_id.get(sp.span_id)
            if p_sp is None:
                raise SafetyValidationError(f"Unknown span_id {sp.span_id} in {record.variant_id}")
            if p_sp.protected and sp.text != p_sp.text:
                raise SafetyValidationError(
                    f"Protected span {sp.span_id} ({p_sp.text!r}) was modified to {sp.text!r}"
                )
            if p_sp.assertion != sp.assertion or p_sp.curie != sp.curie:
                raise SafetyValidationError(f"Assertion or CURIE mutated on span {sp.span_id}")

        # 2. Verify 100% clause span alignment
        for cl in record.clauses:
            if cl.start < 0 or cl.end > len(aug_text) or cl.start > cl.end:
                raise SpanAlignmentError(f"Invalid clause bounds [{cl.start}:{cl.end}] in {record.variant_id}")
            if aug_text[cl.start:cl.end] != cl.text:
                raise SpanAlignmentError(f"Clause offset drift in {record.variant_id} ({cl.clause_uid})")

        # 3. Verify edits use only allowed ontologies (never MONDO) and pass candidate filters
        for ed in record.edits:
            if ed.curie.startswith("MONDO:") or not ed.curie.startswith(ALLOWED_ONTOLOGY_PREFIXES):
                raise SafetyValidationError(
                    f"Disallowed ontology CURIE {ed.curie!r} in edit for {record.variant_id}"
                )
            if not is_valid_replacement_candidate(ed.replacement_text, ed.original_text, self.swapper.blacklist):
                raise SafetyValidationError(
                    f"Invalid replacement candidate {ed.replacement_text!r} for {ed.original_text!r}"
                )

        # 4. Verify zero newly introduced etiology/diagnosis blacklist terms
        norm_orig = normalize_surface(orig_text)
        norm_aug = normalize_surface(aug_text)
        for token in BLOCKED_ETIOLOGY_TOKENS:
            if token in norm_aug and token not in norm_orig:
                raise SafetyValidationError(
                    f"Lexical label leakage detected in {record.variant_id}: introduced {token!r}"
                )
        diag_str = str(parent_doc.row.get("specific_final_diagnosis_text") or "")
        norm_diag = normalize_surface(diag_str)
        if len(norm_diag) >= 5 and norm_diag in norm_aug and norm_diag not in norm_orig:
            raise SafetyValidationError(
                f"Target diagnosis string {diag_str!r} leaked into {record.variant_id}"
            )

        # 5. Verify lineage_group_id and disease_family_label inheritance
        if record.lineage_group_id != parent_doc.lineage_group_id:
            raise SafetyValidationError("Augmented record failed to inherit parent lineage_group_id")
        if record.disease_family_label != parent_doc.disease_family_label:
            raise SafetyValidationError("Augmented record failed to inherit parent disease_family_label")
        if not record.is_augmented or not record.row.get("is_augmented"):
            raise SafetyValidationError("Augmented record missing is_augmented=True flag")

        # 6. Verify JSONL serialization integrity
        serialized = json.dumps(record.row, ensure_ascii=False)
        if "\n" in serialized or "\r" in serialized:
            raise SafetyValidationError("JSONL line contains embedded newline")
        roundtrip = json.loads(serialized)
        if roundtrip.get("document_id") != record.variant_id:
            raise SafetyValidationError("JSONL roundtrip document_id mismatch")

    def augment_document(
        self,
        doc: HydratedDocument,
        *,
        seed: int = 20261117,
        view_index: int = 0,
    ) -> AugmentedRecord:
        """Generate and validate a deterministic augmented variant for `doc`."""
        doc_hash = int(hashlib.sha256(f"{doc.document_id}:{seed}:{view_index}".encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(doc_hash)

        edits = self.swapper.propose_edits(doc, rng, ensure_at_least_one=True)
        aug_text, aug_clauses, aug_spans = apply_replacements(doc, edits)
        method = "curie_synonym_swap"

        if self.enable_clause_reorder:
            reordered_text, reordered_clauses, reordered_spans, did_reorder = reorder_adjacent_clauses(
                aug_text, aug_clauses, aug_spans, rng
            )
            if did_reorder:
                aug_text, aug_clauses, aug_spans = reordered_text, reordered_clauses, reordered_spans
                method = "curie_synonym_swap+clause_reorder" if edits else "clause_reorder"

        variant_hash = hashlib.sha256(
            f"{doc.document_id}:{method}:{seed}:{view_index}:{aug_text}".encode("utf-8")
        ).hexdigest()[:12]
        variant_id = f"{doc.document_id}__AUG_{view_index}_{variant_hash}"

        new_row = copy.deepcopy(dict(doc.row))
        new_row["document_id"] = variant_id
        new_row["parent_document_id"] = doc.document_id
        new_row["lineage_group_id"] = doc.lineage_group_id
        new_row["disease_family_label"] = doc.disease_family_label
        new_row["input_text"] = aug_text
        new_row["prediagnosis_text"] = aug_text
        new_row["is_augmented"] = True
        new_row["augmentation_method"] = method
        new_row["augmentation_seed"] = seed
        new_row["augmentation_view_index"] = view_index
        new_row["augmentation_edits"] = [
            {
                "start": e.start,
                "end": e.end,
                "original_text": e.original_text,
                "replacement_text": e.replacement_text,
                "curie": e.curie,
                "semantic_type": e.semantic_type,
                "clause_uid": e.clause_uid,
            }
            for e in edits
        ]

        record = AugmentedRecord(
            variant_id=variant_id,
            parent_document_id=doc.document_id,
            dataset=doc.dataset,
            disease_family_label=doc.disease_family_label,
            lineage_group_id=doc.lineage_group_id,
            method=method,
            seed=seed,
            original_text=doc.text,
            augmented_text=aug_text,
            clauses=aug_clauses,
            spans=aug_spans,
            edits=edits,
            row=new_row,
            is_augmented=True,
        )
        self.validate_invariants(doc, record)
        return record

    def augment_row(
        self,
        row: Mapping[str, Any],
        *,
        seed: int = 20261117,
        view_index: int = 0,
    ) -> AugmentedRecord:
        """Hydrate and augment a single clinical case dictionary."""
        doc = self.hydrator.hydrate_row(row)
        return self.augment_document(doc, seed=seed, view_index=view_index)


def select_fold_variants(
    variants: Sequence[AugmentedRecord],
    fit_document_ids: set[str],
    validation_document_ids: set[str] | None = None,
) -> list[AugmentedRecord]:
    """Select only augmented variants whose parent belongs to `fit_document_ids`.

    Enforces a strict fold isolation guard against validation leakage.
    """
    if validation_document_ids and (fit_document_ids & validation_document_ids):
        raise SafetyValidationError("Train/validation document_id sets overlap!")
    selected = [v for v in variants if v.parent_document_id in fit_document_ids]
    if validation_document_ids:
        for v in selected:
            if v.parent_document_id in validation_document_ids:
                raise SafetyValidationError(
                    f"Data leakage detected: variant {v.variant_id} derived from validation case {v.parent_document_id}"
                )
    return selected
