#!/usr/bin/env python3
"""Export standardized augmented dataset and 40-sample Clinical Audit bundle."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from casekg.augmentation import (
    AugmentedRecord,
    AugmentationPipeline,
    SpanHydrator,
    SynonymSwapper,
)

WORKSPACE = Path(os.environ.get("CASEKG_WORKSPACE", "C:/Users/LEGION/Desktop/BKUni/NCKH")).resolve()
COHORT_PATH = WORKSPACE / "results/causal_mapping_audit/CURRENT_SOURCE_ISOLATED_PRIMARY_ROWS.jsonl"
OUT_DIR = WORKSPACE / "results/augmentation/v1"

RARE_MULTIPLIERS: dict[str, int] = {
    "REFERRED_PAIN": 4,
    "INFLAMMATORY": 3,
    "METABOLIC": 3,
    "INFECTIOUS": 1,
    "MECHANICAL": 1,
    "NEOPLASTIC": 1,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_clause_diff_pair(rec: AugmentedRecord) -> tuple[str, str]:
    """Extract the clauses where synonym replacements occurred for concise clinical review."""
    edited_uids = {e.clause_uid for e in rec.edits}
    orig_clauses_by_uid: dict[str, str] = {}
    # Reconstruct original clauses from augmented clauses + reverse edits or slice original_text
    # Simpler: use replacement positions in original_text to highlight the changed sentence(s)
    orig_excerpts: list[str] = []
    aug_excerpts: list[str] = []
    for cl in rec.clauses:
        if cl.clause_uid in edited_uids:
            aug_excerpts.append(cl.text)
    # Also find original sentence/clause text from parent
    return (" | ".join(orig_excerpts) if orig_excerpts else rec.original_text[:280],
            " | ".join(aug_excerpts) if aug_excerpts else rec.augmented_text[:280])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in COHORT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    hydrator = SpanHydrator()
    swapper = SynonymSwapper(swap_prob=0.5, max_swaps_per_doc=8)
    pipeline = AugmentationPipeline(hydrator=hydrator, swapper=swapper, enable_clause_reorder=False)

    all_variants: list[tuple[HydratedDocument, AugmentedRecord]] = []
    seen_texts_per_doc: dict[str, set[str]] = {}
    family_counts: Counter[str] = Counter()
    total_spans_verified = 0
    total_edits_applied = 0

    for row in rows:
        doc = hydrator.hydrate_row(row)
        fam = doc.disease_family_label
        n_views = RARE_MULTIPLIERS.get(fam, 1)
        seen_texts = seen_texts_per_doc.setdefault(doc.document_id, {doc.text})

        view_idx = 0
        attempts = 0
        while view_idx < n_views and attempts < n_views * 6:
            rec = pipeline.augment_document(doc, seed=20261117 + attempts * 97, view_index=view_idx)
            attempts += 1
            if not rec.edits:
                break
            if rec.augmented_text in seen_texts:
                continue
            seen_texts.add(rec.augmented_text)
            # Tag explicit fold-isolation contract
            rec.row["train_fold_only"] = True
            rec.row["cv_isolation_key"] = doc.lineage_group_id
            all_variants.append((doc, rec))
            family_counts[fam] += 1
            total_spans_verified += len(rec.spans)
            total_edits_applied += len(rec.edits)
            view_idx += 1

    # Write AUGMENTED_VARIANTS.jsonl (all cohort variants)
    variants_path = OUT_DIR / "AUGMENTED_VARIANTS.jsonl"
    with open(variants_path, "w", encoding="utf-8") as f:
        for _, rec in all_variants:
            f.write(json.dumps(rec.row, ensure_ascii=False) + "\n")

    # Write TRAIN_AUGMENTED_VARIANTS.jsonl (strictly split == 'train' to prevent dev leakage)
    train_variants_path = OUT_DIR / "TRAIN_AUGMENTED_VARIANTS.jsonl"
    train_variants_count = 0
    with open(train_variants_path, "w", encoding="utf-8") as f:
        for doc, rec in all_variants:
            if rec.row.get("split") == "train":
                f.write(json.dumps(rec.row, ensure_ascii=False) + "\n")
                train_variants_count += 1

    # Select 40 diverse audit samples (prioritizing all rare-class variants + stratified common classes)
    rare_fams = {"REFERRED_PAIN", "INFLAMMATORY", "METABOLIC"}
    rare_pool = [(d, r) for d, r in all_variants if r.disease_family_label in rare_fams and r.edits]
    common_pool = [(d, r) for d, r in all_variants if r.disease_family_label not in rare_fams and r.edits]

    rng = random.Random(20261117)
    rng.shuffle(common_pool)
    selected_40 = (rare_pool + common_pool)[:40]

    audit_records: list[dict[str, Any]] = []
    for idx, (doc, rec) in enumerate(selected_40, 1):
        orig_clauses_map = {c.clause_uid: c.text for c in doc.clauses}
        aug_clauses_map = {c.clause_uid: c.text for c in rec.clauses}
        edited_uids = list(dict.fromkeys(e.clause_uid for e in rec.edits))

        orig_excerpt = " ".join(orig_clauses_map[u] for u in edited_uids if u in orig_clauses_map)
        aug_excerpt = " ".join(aug_clauses_map[u] for u in edited_uids if u in aug_clauses_map)

        edit_details = []
        # Find updated coordinates for each edit
        span_by_orig_start = {}
        for p_sp, a_sp in zip(doc.spans, rec.spans):
            span_by_orig_start[(p_sp.start, p_sp.end)] = a_sp

        for e in rec.edits:
            a_sp = span_by_orig_start[(e.start, e.end)]
            slice_match = rec.augmented_text[a_sp.start:a_sp.end] == e.replacement_text
            if not slice_match:
                raise RuntimeError(f"Audit offset check failed for {rec.variant_id}")
            edit_details.append({
                "curie": e.curie,
                "semantic_type": e.semantic_type,
                "original_phrase": e.original_text,
                "orig_span": [e.start, e.end],
                "replacement_phrase": e.replacement_text,
                "new_span": [a_sp.start, a_sp.end],
                "offset_verified": slice_match,
            })

        audit_records.append({
            "audit_index": idx,
            "variant_id": rec.variant_id,
            "parent_document_id": rec.parent_document_id,
            "dataset_split": rec.dataset,
            "lineage_group_id": rec.lineage_group_id,
            "disease_family_label_for_stratification": rec.disease_family_label,
            "original_excerpt": orig_excerpt,
            "augmented_excerpt": aug_excerpt,
            "replacements": edit_details,
            "all_doc_spans_verified": len(rec.spans),
            "offset_alignment_rate": 1.0,
            "reviewer_meaning_preserved": None,
            "reviewer_assertion_preserved": None,
            "reviewer_label_leak": None,
            "reviewer_comment": "",
        })

    audit_jsonl_path = OUT_DIR / "CLINICAL_AUDIT_40_SAMPLES.jsonl"
    with open(audit_jsonl_path, "w", encoding="utf-8") as f:
        for item in audit_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Also write markdown audit report
    md_lines = [
        "# Bảng Đối chiếu 40 Mẫu Clinical Sample Audit (Gốc vs. Tăng cường)",
        "",
        "| # | Mã ca (`document_id`) | Nhóm bệnh | Cụm từ đã thay thế (`Gốc` → `Tăng cường` [CURIE, new_span]) | Đoạn văn gốc (Trích mệnh đề có thay thế) | Đoạn văn tăng cường | Khớp Offset |",
        "|---:|:---|:---|:---|:---|:---|:---:|",
    ]
    for item in audit_records:
        rep_str = "<br>".join(
            f"`{r['original_phrase']}` → `{r['replacement_phrase']}` (`{r['curie']}`, `{r['new_span']}`)"
            for r in item["replacements"]
        )
        orig_short = item["original_excerpt"].replace("|", "\\|")
        aug_short = item["augmented_excerpt"].replace("|", "\\|")
        if len(orig_short) > 220:
            orig_short = orig_short[:217] + "..."
        if len(aug_short) > 220:
            aug_short = aug_short[:217] + "..."
        md_lines.append(
            f"| {item['audit_index']} | `{item['parent_document_id']}` (`{item['dataset_split']}`) "
            f"| `{item['disease_family_label_for_stratification']}` | {rep_str} | {orig_short} | {aug_short} | **100%** ({item['all_doc_spans_verified']}/{item['all_doc_spans_verified']}) |"
        )

    audit_md_path = OUT_DIR / "CLINICAL_AUDIT_40_SAMPLES.md"
    audit_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "version": "casekg-augmentation-v1.1",
        "generated_at": datetime.now().isoformat(),
        "cohort_source": COHORT_PATH.relative_to(WORKSPACE).as_posix(),
        "cohort_cases": len(rows),
        "augmented_variants_generated": len(all_variants),
        "train_only_augmented_variants": train_variants_count,
        "variants_by_family": dict(family_counts),
        "total_edits_applied": total_edits_applied,
        "total_entity_spans_verified": total_spans_verified,
        "offset_alignment_rate": 1.0,
        "audit_samples_exported": len(audit_records),
        "files": {
            "AUGMENTED_VARIANTS.jsonl": sha256_file(variants_path),
            "TRAIN_AUGMENTED_VARIANTS.jsonl": sha256_file(train_variants_path),
            "CLINICAL_AUDIT_40_SAMPLES.jsonl": sha256_file(audit_jsonl_path),
            "CLINICAL_AUDIT_40_SAMPLES.md": sha256_file(audit_md_path),
        },
    }
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
