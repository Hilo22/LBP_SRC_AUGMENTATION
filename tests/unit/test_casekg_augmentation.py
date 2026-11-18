"""Unit tests for the `casekg.augmentation` package."""
from __future__ import annotations

# Import C-extensions first on Windows to avoid FLS slot exhaustion if torch is loaded later.
import duckdb  # noqa: F401
import numpy as np  # noqa: F401
import scipy.sparse  # noqa: F401

import json
import random
import unittest
from pathlib import Path

from casekg.paths import PROJECT_ROOT
from casekg.augmentation import (
    AugmentedRecord,
    AugmentationPipeline,
    ClauseSpan,
    EntitySpan,
    HydratedDocument,
    ReplacementEdit,
    SafetyValidationError,
    SpanAlignmentError,
    SpanHydrator,
    SynonymSwapper,
    apply_replacements,
    is_valid_replacement_candidate,
    preserve_initial_case,
    reorder_adjacent_clauses,
    select_fold_variants,
)


class AugmentationUnitTests(unittest.TestCase):
    """Test suite for SpanHydrator, SynonymSwapper, and AugmentationPipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hydrator = SpanHydrator()
        cls.swapper = SynonymSwapper(swap_prob=0.5, max_swaps_per_doc=8)
        cls.pipeline = AugmentationPipeline(
            hydrator=cls.hydrator,
            swapper=cls.swapper,
            enable_clause_reorder=True,
        )
        cohort_path = PROJECT_ROOT / "results/causal_mapping_audit/CURRENT_SOURCE_ISOLATED_PRIMARY_ROWS.jsonl"
        cls.cohort_rows = [
            json.loads(line)
            for line in cohort_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_runtime_candidate_filters_reject_codes_and_latin(self) -> None:
        """Verify Filter (a), Filter (b), and Filter (c) reject unnatural candidates."""
        # Filter (a): technical/system codes, digits, underscores, all-caps
        self.assertFalse(is_valid_replacement_candidate("vstest", "vital signs measurement"))
        self.assertFalse(is_valid_replacement_candidate("labtest", "laboratory test"))
        self.assertFalse(is_valid_replacement_candidate("VSTEST", "vital signs measurement"))
        self.assertFalse(is_valid_replacement_candidate("type_2_pain", "back pain"))
        self.assertFalse(is_valid_replacement_candidate("stage 4 pain", "back pain"))

        # Filter (b): Latin single-letter prefixes and Latin anatomical nomina
        self.assertFalse(is_valid_replacement_candidate("m iliotibilias", "iliotibialis muscle"))
        self.assertFalse(is_valid_replacement_candidate("iliotibilias", "iliotibialis muscle"))
        self.assertFalse(is_valid_replacement_candidate("a lumbalis", "lumbar artery"))
        self.assertFalse(is_valid_replacement_candidate("n ischiadicus", "sciatic nerve"))
        self.assertFalse(is_valid_replacement_candidate("v cava", "vena cava"))
        self.assertFalse(is_valid_replacement_candidate("lig flavum", "ligamentum flavum"))
        self.assertFalse(is_valid_replacement_candidate("musculus psoas", "psoas muscle"))

        # Filter (c): length [4, 60] and ratio [1/3, 3.0]
        self.assertFalse(is_valid_replacement_candidate("dry", "dry skin"))  # len < 4
        self.assertFalse(is_valid_replacement_candidate("a" * 61, "short phrase requires long text"))
        self.assertFalse(is_valid_replacement_candidate("very long replacement phrase exceeding ratio", "pain"))

        # Valid natural English clinical synonyms pass
        self.assertTrue(is_valid_replacement_candidate("low backache", "low back pain"))
        self.assertTrue(is_valid_replacement_candidate("vital signs assessment", "vital signs measurement"))
        self.assertTrue(is_valid_replacement_candidate("xerosis", "dry skin"))

    def test_initial_case_preservation(self) -> None:
        """Verify capitalized spans preserve capitalization after replacement."""
        self.assertEqual(preserve_initial_case("Low back pain", "low backache"), "Low backache")
        self.assertEqual(preserve_initial_case("low back pain", "low backache"), "low backache")
        self.assertEqual(preserve_initial_case("Dry skin", "xerosis"), "Xerosis")

    def test_multi_edit_zero_offset_drift(self) -> None:
        """Verify 100% span & clause alignment across shrinking and expanding multi-edits."""
        c0_text = "Dry skin and low back pain were noted on vital signs measurement."
        c1_text = "Blood chemistry measurement was normal."
        full_text = c0_text + " " + c1_text

        clauses = (
            ClauseSpan("c0", 0, "PRESENTING_SYMPTOMS", 0, len(c0_text), c0_text),
            ClauseSpan("c1", 1, "EXAMINATION_INVESTIGATION", len(c0_text) + 1, len(full_text), c1_text),
        )
        s0_start = full_text.index("Dry skin")
        s1_start = full_text.index("low back pain")
        s2_start = full_text.index("vital signs measurement")
        s3_start = full_text.index("Blood chemistry measurement")

        spans = (
            EntitySpan("s0", s0_start, s0_start + len("Dry skin"), "Dry skin", "c0", "HP:0000958", "PHENOTYPE"),
            EntitySpan("s1", s1_start, s1_start + len("low back pain"), "low back pain", "c0", "SYMP:0000635", "SYMPTOM"),
            EntitySpan("s2", s2_start, s2_start + len("vital signs measurement"), "vital signs measurement", "c0", "MAXO:0000489", "MEDICAL_ACTION"),
            EntitySpan("s3", s3_start, s3_start + len("Blood chemistry measurement"), "Blood chemistry measurement", "c1", "MAXO:0000787", "MEDICAL_ACTION"),
        )
        doc = HydratedDocument(
            document_id="SYNTH001",
            dataset="primary_train",
            disease_family_label="MECHANICAL",
            lineage_group_id="GRP_SYNTH001",
            text_field="prediagnosis_text",
            text=full_text,
            clauses=clauses,
            spans=spans,
            row={"document_id": "SYNTH001", "lineage_group_id": "GRP_SYNTH001", "disease_family_label": "MECHANICAL"},
        )

        edits = (
            ReplacementEdit(spans[0].start, spans[0].end, "Dry skin", "Xerosis", "HP:0000958", "PHENOTYPE", "c0", "test"),
            ReplacementEdit(spans[1].start, spans[1].end, "low back pain", "low backache", "SYMP:0000635", "SYMPTOM", "c0", "test"),
            ReplacementEdit(spans[2].start, spans[2].end, "vital signs measurement", "vital signs assessment", "MAXO:0000489", "MEDICAL_ACTION", "c0", "test"),
            ReplacementEdit(spans[3].start, spans[3].end, "Blood chemistry measurement", "Blood chemistry test", "MAXO:0000787", "MEDICAL_ACTION", "c1", "test"),
        )
        aug_text, aug_clauses, aug_spans = apply_replacements(doc, edits)

        for sp in aug_spans:
            self.assertEqual(aug_text[sp.start:sp.end], sp.text)
        for cl in aug_clauses:
            self.assertEqual(aug_text[cl.start:cl.end], cl.text)
        self.assertEqual([sp.text for sp in aug_spans], ["Xerosis", "low backache", "vital signs assessment", "Blood chemistry test"])

    def test_label_leakage_and_mondo_protection(self) -> None:
        """Verify MONDO concepts and target diagnosis terms are blocked at every layer."""
        text = "The patient had low back pain and suspected multiple myeloma."
        s0 = text.index("low back pain")
        s1 = text.index("multiple myeloma")
        doc = HydratedDocument(
            document_id="LEAK001",
            dataset="primary_train",
            disease_family_label="NEOPLASTIC",
            lineage_group_id="GRP_LEAK001",
            text_field="prediagnosis_text",
            text=text,
            clauses=(ClauseSpan("c0", 0, "PRESENTING_SYMPTOMS", 0, len(text), text),),
            spans=(
                EntitySpan("s0", s0, s0 + len("low back pain"), "low back pain", "c0", "SYMP:0000635", "SYMPTOM", protected=False),
                EntitySpan("s1", s1, s1 + len("multiple myeloma"), "multiple myeloma", "c0", "MONDO:0009693", "DISEASE", protected=True),
            ),
            row={
                "document_id": "LEAK001",
                "lineage_group_id": "GRP_LEAK001",
                "disease_family_label": "NEOPLASTIC",
                "specific_final_diagnosis_text": "Multiple myeloma",
            },
        )
        # Swapper must return 0 candidates for protected or MONDO span
        self.assertEqual(self.swapper.candidate_synonyms_for_span(doc.spans[1]), [])

        # Pipeline invariant check must reject any record that introduces a blacklisted diagnosis or MONDO edit
        valid_rec = self.pipeline.augment_document(doc, seed=42)
        self.assertIn("multiple myeloma", valid_rec.augmented_text)

        bad_edit = ReplacementEdit(
            start=s0,
            end=s0 + len("low back pain"),
            original_text="low back pain",
            replacement_text="spondylodiscitis",
            curie="MONDO:0001234",
            semantic_type="DISEASE",
            clause_uid="c0",
            reason="malicious_test",
        )
        bad_text, bad_clauses, bad_spans = apply_replacements(doc, (bad_edit,))
        bad_rec = AugmentedRecord(
            variant_id="BAD001",
            parent_document_id="LEAK001",
            dataset="primary_train",
            disease_family_label="NEOPLASTIC",
            lineage_group_id="GRP_LEAK001",
            method="malicious",
            seed=42,
            original_text=text,
            augmented_text=bad_text,
            clauses=bad_clauses,
            spans=bad_spans,
            edits=(bad_edit,),
            row={"document_id": "BAD001", "is_augmented": True},
        )
        with self.assertRaises(SafetyValidationError):
            self.pipeline.validate_invariants(doc, bad_rec)

    def test_determinism_and_full_57_cohort_invariants(self) -> None:
        """Verify byte-identical determinism and 100% invariant compliance across all 57 cases."""
        self.assertEqual(len(self.cohort_rows), 57)
        variants_run1 = []
        variants_run2 = []

        for row in self.cohort_rows:
            r1 = self.pipeline.augment_row(row, seed=20261117, view_index=0)
            r2 = self.pipeline.augment_row(row, seed=20261117, view_index=0)
            self.assertEqual(r1.variant_id, r2.variant_id)
            self.assertEqual(r1.augmented_text, r2.augmented_text)
            self.assertEqual(r1.edits, r2.edits)
            self.assertEqual(
                json.dumps(r1.row, sort_keys=True, ensure_ascii=False),
                json.dumps(r2.row, sort_keys=True, ensure_ascii=False),
            )
            for sp in r1.spans:
                self.assertEqual(r1.augmented_text[sp.start:sp.end], sp.text)
            for cl in r1.clauses:
                self.assertEqual(r1.augmented_text[cl.start:cl.end], cl.text)
            variants_run1.append(r1)
            variants_run2.append(r2)

        # Verify fold-isolation selector prevents validation leakage
        train_ids = {r["document_id"] for r in self.cohort_rows[:40]}
        val_ids = {r["document_id"] for r in self.cohort_rows[40:]}
        selected = select_fold_variants(variants_run1, train_ids, val_ids)
        self.assertEqual(len(selected), 40)
        self.assertTrue(all(v.parent_document_id in train_ids for v in selected))

        with self.assertRaises(SafetyValidationError):
            select_fold_variants(variants_run1, train_ids, train_ids)

    def test_clause_reorder_preserves_offsets(self) -> None:
        """Verify adjacent independent clause reordering preserves 100% of entity offsets."""
        c0 = "A 55-year-old female presented with low back pain."
        c1 = "Blood chemistry measurement was within normal limits."
        c2 = "Vital signs measurement showed normal blood pressure."
        text = f"{c0} {c1} {c2}"
        clauses = (
            ClauseSpan("u0", 0, "PRESENTING_SYMPTOMS", 0, len(c0), c0),
            ClauseSpan("u1", 1, "EXAMINATION_INVESTIGATION", len(c0) + 1, len(c0) + 1 + len(c1), c1),
            ClauseSpan("u2", 2, "EXAMINATION_INVESTIGATION", len(c0) + 1 + len(c1) + 1, len(text), c2),
        )
        s0 = text.index("low back pain")
        s1 = text.index("Blood chemistry measurement")
        s2 = text.index("Vital signs measurement")
        spans = (
            EntitySpan("e0", s0, s0 + len("low back pain"), "low back pain", "u0", "SYMP:0000635", "SYMPTOM"),
            EntitySpan("e1", s1, s1 + len("Blood chemistry measurement"), "Blood chemistry measurement", "u1", "MAXO:0000787", "MEDICAL_ACTION"),
            EntitySpan("e2", s2, s2 + len("Vital signs measurement"), "Vital signs measurement", "u2", "MAXO:0000489", "MEDICAL_ACTION"),
        )
        new_text, new_clauses, new_spans, did_swap = reorder_adjacent_clauses(
            text, clauses, spans, random.Random(7)
        )
        self.assertTrue(did_swap)
        self.assertEqual(new_text, f"{c0} {c2} {c1}")
        for sp in new_spans:
            self.assertEqual(new_text[sp.start:sp.end], sp.text)
        for cl in new_clauses:
            self.assertEqual(new_text[cl.start:cl.end], cl.text)


if __name__ == "__main__":
    unittest.main()
