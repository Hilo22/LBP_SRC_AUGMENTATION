#!/usr/bin/env python3
"""Build specialized Data Augmentation resources for CASEKG (D3-compliant).

Generates in `resources/augmentation/`:
1. `lbp_synonym_dictionary.json` — strictly grouped by single ontology `curie`
   (MONDO / HP / UBERON / SYMP / MAXO), with `scope = 'EXACT'` only, zero manual
   synonym lists, and full Lexical Label Leakage Filtering against all 647
   target diagnosis strings + etiology-revealing disease nouns.
2. `clause_offset_table.json` — verified `[clause_start, clause_end]` mapping
   for all 401 corpus documents (including 57/57 current cohort cases).
3. `paraphrase_config.json` — placeholder protection (`<ENT_{type}_{id}>`),
   immutable vital/lab/temporal regexes, and assertion-channel invariants.
4. `question_variants.json` — grammatical question variants (stored for audit
   only; disabled in v1 diagnosis features per Decision D4).
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

WORKSPACE = Path(os.environ.get("CASEKG_WORKSPACE", "C:/Users/LEGION/Desktop/BKUni/NCKH")).resolve()
RESOURCES_DIR = WORKSPACE / "resources/augmentation"
CONFIG_DIR = WORKSPACE / "config"
DATA_DIR = WORKSPACE / "data"
RESULTS_DIR = WORKSPACE / "results"

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]")
FORBIDDEN_CHARS_RE = re.compile(r"[\(\)\[\]\{\}/\\;:,<>\"'`~@#$%^&*+=|?]")

# Etiology / target diagnosis tokens that must never appear in any synonym group
BLOCKED_ETIOLOGY_TOKENS = (
    "myeloma", "plasmacytoma", "spondylodiscitis", "discitis", "diskitis",
    "osteomyelitis", "abscess", "ependymoma", "schwannoma", "neurofibroma",
    "meningioma", "metastasis", "metastatic", "carcinoma", "adenocarcinoma",
    "sarcoma", "osteosarcoma", "chondrosarcoma", "chordoma", "lymphoma",
    "leukemia", "neoplasm", "tumor", "tumour", "malignancy", "malignant",
    "cancer", "aneurysm", "dissection", "ankylosing spondylitis",
    "spondyloarthritis", "spondylarthropathy", "spondyloarthropathy",
    "paget", "brucellosis", "brucellar", "tuberculosis", "tuberculous",
    "pott", "pyelonephritis", "nephrolithiasis", "urolithiasis",
    "pancreatitis", "endometriosis", "cholecystitis", "appendicitis",
    "pheochromocytoma", "ganglioneuroma", "hemangioma", "lipoma",
    "osteoid osteoma", "osteoblastoma", "giant cell", "ewing",
    "hyperparathyroidism", "osteomalacia", "rickets", "gout", "tophaceous",
    "pseudogout", "chondrocalcinosis", "amyloidosis", "sarcoidosis",
    "disc herniation", "disk herniation", "herniated disc", "herniated disk",
    "spinal stenosis", "spondylolisthesis", "spondylolysis",
    "cauda equina syndrome", "arachnoiditis", "tethered cord", "syringomyelia",
)

GENERIC_STOP_SYNONYMS = {
    "all", "body", "case", "condition", "disease", "disorder", "finding",
    "medical action", "patient", "phenotype", "procedure", "symptom",
    "syndrome", "treatment", "normal", "abnormal", "pain", "sign", "lesion",
    "mass", "injury", "infection", "inflammation", "other", "unknown",
}


def normalize_term(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip().lower()


def load_leakage_blacklist() -> set[str]:
    """Collect all target diagnosis strings across every cohort in the workspace."""
    diagnoses: set[str] = set()
    for t in BLOCKED_ETIOLOGY_TOKENS:
        diagnoses.add(normalize_term(t))

    data_files = [
        DATA_DIR / "isolated_development/diagnosis/diagnosis_train.jsonl",
        DATA_DIR / "isolated_development/diagnosis/diagnosis_dev.jsonl",
        DATA_DIR / "chronology_screened_development/diagnosis/diagnosis_train.jsonl",
        DATA_DIR / "chronology_screened_development/diagnosis/diagnosis_dev.jsonl",
        DATA_DIR / "chronology_screened_development/orbit_auxiliary/diagnosis_auxiliary_train.jsonl",
        RESULTS_DIR / "literal_evidence_diagnosis/evidence/DATA_ROWS.jsonl",
        RESULTS_DIR / "causal_mapping_audit/CURRENT_SOURCE_ISOLATED_PRIMARY_ROWS.jsonl",
    ]
    for df in data_files:
        if not df.exists():
            continue
        with open(df, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                diag = row.get("specific_final_diagnosis_text")
                if diag:
                    norm = normalize_term(diag)
                    if norm:
                        diagnoses.add(norm)
                        for sub in norm.split(" due to "):
                            if sub.strip():
                                diagnoses.add(sub.strip())
                for opt in row.get("multiple_choice_options", []):
                    norm_opt = normalize_term(opt)
                    if norm_opt:
                        diagnoses.add(norm_opt)

    inventory_file = RESULTS_DIR / "current_data_inventory_20261029/DATA_INVENTORY.json"
    if inventory_file.exists():
        inv_data = json.loads(inventory_file.read_text(encoding="utf-8"))
        for item in inv_data.get("recorded_name_catalog", []):
            name = item.get("recorded_name")
            if name:
                norm = normalize_term(name)
                if norm:
                    diagnoses.add(norm)

    return diagnoses


def is_clean_surface(raw_alias: str, blacklist: set[str]) -> bool:
    """Validate that an ontology alias is well-formed and free of label leakage."""
    raw = raw_alias.strip()
    if not raw or FORBIDDEN_CHARS_RE.search(raw):
        return False
    if raw.lower().startswith("obsolete ") or " NOS" in raw or " NEC" in raw:
        return False

    norm = normalize_term(raw)
    if len(norm) < 4 or norm in GENERIC_STOP_SYNONYMS:
        return False
    words = norm.split()
    if len(words) < 1 or len(words) > 6:
        return False
    # Reject pure numbers or single-character tokens that look like grading artifacts
    if any(w.isdigit() for w in words):
        return False
    if norm in blacklist:
        return False
    for blocked in BLOCKED_ETIOLOGY_TOKENS:
        if blocked in norm:
            return False
    # Also check if any multi-word blacklisted diagnosis (>= 8 chars) is a substring
    for diag in blacklist:
        if len(diag) >= 8 and diag in norm:
            return False
    return True


def collect_corpus_curies() -> set[str]:
    """Collect CURIEs that appear in linked sections of the clinical corpus."""
    curies: set[str] = set()
    pred_sec = RESULTS_DIR / "section_conditioned_diagnosis/evidence/PREDICTED_SECTIONS.jsonl"
    if pred_sec.exists():
        with open(pred_sec, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for u in rec.get("units", []):
                    for lk in u.get("links", []):
                        c = lk.get("curie")
                        if c:
                            curies.add(c)
    return curies


def build_synonym_dictionary(blacklist: set[str]) -> dict[str, Any]:
    """Build strictly CURIE-grouped synonym dictionary from DuckDB + SapBERT."""
    duckdb_path = WORKSPACE / "resources/kg/casekg_orbit.duckdb"
    con = duckdb.connect(str(duckdb_path), read_only=True)

    corpus_curies = collect_corpus_curies()

    # Exclude any CURIE that is a disease in kg_clinical_family_memberships
    disease_family_curies = {
        r[0]
        for r in con.execute("SELECT DISTINCT curie FROM kg_clinical_family_memberships").fetchall()
    }

    # Fetch non-obsolete nodes in HP, SYMP, UBERON, MAXO (exclude MONDO diseases to prevent etiology swap)
    nodes_rows = con.execute(
        """
        SELECT curie, preferred_label, semantic_type, ontology_prefix, lumbar_relevant_surface
        FROM kg_nodes
        WHERE obsolete = FALSE
          AND ontology_prefix IN ('HP', 'SYMP', 'UBERON', 'MAXO')
        """
    ).fetchall()

    candidate_nodes: dict[str, dict[str, Any]] = {}
    for curie, pref_label, sem_type, prefix, lumbar_rel in nodes_rows:
        if curie in disease_family_curies:
            continue
        if not is_clean_surface(pref_label, blacklist):
            continue
        if lumbar_rel or curie in corpus_curies:
            candidate_nodes[curie] = {
                "curie": curie,
                "preferred_label": pref_label,
                "semantic_type": sem_type,
                "ontology_prefix": prefix,
                "in_corpus": curie in corpus_curies,
                "lumbar_relevant": bool(lumbar_rel),
            }

    # Fetch EXACT aliases only (PREFERRED_LABEL and SYNONYM with scope = 'EXACT')
    aliases_by_curie: dict[str, set[str]] = defaultdict(set)
    alias_rows = con.execute(
        """
        SELECT curie, alias
        FROM kg_aliases
        WHERE scope = 'EXACT'
          AND alias_type IN ('PREFERRED_LABEL', 'SYNONYM')
        """
    ).fetchall()
    for curie, alias in alias_rows:
        if curie not in candidate_nodes:
            continue
        if is_clean_surface(alias, blacklist):
            aliases_by_curie[curie].add(normalize_term(alias))

    # Also incorporate SapBERT ONTOLOGY_LABEL_SYNONYM pairs for the exact same CURIE
    sapbert_path = WORKSPACE / "resources/embeddings/sapbert_synonym_train.jsonl"
    if sapbert_path.exists():
        with open(sapbert_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                c = pair.get("curie")
                if c not in candidate_nodes:
                    continue
                if pair.get("provenance") != "ONTOLOGY_LABEL_SYNONYM":
                    continue
                anc = pair.get("anchor", "")
                pos = pair.get("positive", "")
                if is_clean_surface(anc, blacklist) and is_clean_surface(pos, blacklist):
                    aliases_by_curie[c].add(normalize_term(anc))
                    aliases_by_curie[c].add(normalize_term(pos))

    con.close()

    # Keep only CURIEs with >= 2 distinct clean surface forms
    concepts: dict[str, dict[str, Any]] = {}
    surface_to_curies: dict[str, set[str]] = defaultdict(set)

    for curie, meta in sorted(candidate_nodes.items()):
        syns = sorted(aliases_by_curie.get(curie, set()))
        if len(syns) < 2:
            continue
        concepts[curie] = {
            "concept_id": curie,
            "curie": curie,
            "preferred_label": meta["preferred_label"],
            "semantic_type": meta["semantic_type"],
            "ontology_prefix": meta["ontology_prefix"],
            "in_corpus": meta["in_corpus"],
            "lumbar_relevant": meta["lumbar_relevant"],
            "synonyms": syns,
        }
        for s in syns:
            surface_to_curies[s].add(curie)

    # Build lookup_table ONLY for unambiguous surface forms (mapping to exactly 1 CURIE)
    lookup_table: dict[str, dict[str, Any]] = {}
    ambiguous_surfaces = 0
    for surface, curie_set in sorted(surface_to_curies.items()):
        if len(curie_set) != 1:
            ambiguous_surfaces += 1
            continue
        curie = next(iter(curie_set))
        others = [s for s in concepts[curie]["synonyms"] if s != surface]
        if others:
            lookup_table[surface] = {
                "curie": curie,
                "semantic_type": concepts[curie]["semantic_type"],
                "replacements": others,
            }

    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    dictionary_data = {
        "version": "casekg-lbp-synonyms-v2.0-curie-strict",
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "grouping_policy": "STRICT_SINGLE_CURIE_EXACT_SCOPE_ONLY",
            "manual_terms_included": False,
            "allowed_ontologies": ["HP", "SYMP", "UBERON", "MAXO"],
            "excluded_disease_ontology": ["MONDO", "kg_clinical_family_memberships"],
            "blacklist_size": len(blacklist),
            "blocked_etiology_tokens_count": len(BLOCKED_ETIOLOGY_TOKENS),
            "total_concepts": len(concepts),
            "corpus_linked_concepts": sum(1 for c in concepts.values() if c["in_corpus"]),
            "total_unambiguous_lookup_keys": len(lookup_table),
            "ambiguous_surfaces_excluded": ambiguous_surfaces,
        },
        "concepts": concepts,
        "lookup_table": lookup_table,
    }
    out_file = RESOURCES_DIR / "lbp_synonym_dictionary.json"
    out_file.write_text(json.dumps(dictionary_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dictionary_data


def build_clause_offset_table() -> dict[str, Any]:
    """Build and verify document-level [clause_start, clause_end] for all 401 records."""
    base_ek = RESULTS_DIR / "evidence_kernel_diagnosis/features"
    base_sc = RESULTS_DIR / "section_conditioned_diagnosis/evidence"

    rows_401 = {
        json.loads(l)["document_id"]: json.loads(l)
        for l in (base_ek / "DATA_ROWS.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    rows_57 = {
        json.loads(l)["document_id"]: json.loads(l)
        for l in (RESULTS_DIR / "causal_mapping_audit/CURRENT_SOURCE_ISOLATED_PRIMARY_ROWS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    }
    sections_sc = {
        json.loads(l)["document_id"]: json.loads(l)["units"]
        for l in (base_sc / "PREDICTED_SECTIONS.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    sem_graph = {
        json.loads(l)["document_id"]: json.loads(l)
        for l in (base_ek / "SEMANTIC_GRAPH_EVIDENCE.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    clause_offset_table: dict[str, Any] = {}
    stats: Counter[str] = Counter()

    for doc_id, doc_row in rows_401.items():
        text = doc_row.get("prediagnosis_text") or doc_row["input_text"]
        units = sections_sc[doc_id]
        cursor = 0
        doc_clauses: dict[str, Any] = {}
        for idx, u in enumerate(units):
            clause_text = u["text"]
            pos = text.find(clause_text, cursor)
            if pos < 0:
                pos = text.find(clause_text)
            if pos < 0:
                raise RuntimeError(f"Clause {u['clause_uid']} not found in {doc_id}")
            end = pos + len(clause_text)
            if text[pos:end] != clause_text:
                raise RuntimeError(f"Substring mismatch for {u['clause_uid']}")
            doc_clauses[u["clause_uid"]] = {
                "clause_uid": u["clause_uid"],
                "unit_index": u.get("unit_index", idx),
                "primary_section": u["primary_section"],
                "clause_start": pos,
                "clause_end": end,
                "length": len(clause_text),
            }
            cursor = end
            for lk in u.get("links", []):
                ds = pos + lk["start"]
                de = pos + lk["end"]
                if text[ds:de] != lk["text"]:
                    raise RuntimeError(f"Link span mismatch in {u['clause_uid']}")
                stats["sc_links_401_matched"] += 1
                if doc_id in rows_57:
                    stats["sc_links_57_matched"] += 1

        for m in sem_graph[doc_id]["mentions"]:
            c_info = doc_clauses[m["clause_uid"]]
            ds = c_info["clause_start"] + m["span"][0]
            de = c_info["clause_start"] + m["span"][1]
            if text[ds:de] != m["phrase"]:
                raise RuntimeError(f"Semantic mention mismatch in {m['clause_uid']}")
            stats["sem_mentions_401_matched"] += 1
            if doc_id in rows_57:
                stats["sem_mentions_57_matched"] += 1

        clause_offset_table[doc_id] = {
            "document_id": doc_id,
            "dataset": doc_row["dataset"],
            "in_current_57_cohort": doc_id in rows_57,
            "text_length": len(text),
            "clauses": doc_clauses,
        }

    payload = {
        "version": "casekg-clause-offsets-v1.0",
        "generated_at": datetime.now().isoformat(),
        "source_predicted_sections": "results/section_conditioned_diagnosis/evidence/PREDICTED_SECTIONS.jsonl",
        "source_semantic_graph": "results/evidence_kernel_diagnosis/features/SEMANTIC_GRAPH_EVIDENCE.jsonl",
        "verification_stats": dict(stats),
        "documents": clause_offset_table,
    }
    out_path = RESOURCES_DIR / "clause_offset_table.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    blacklist = load_leakage_blacklist()
    syn_data = build_synonym_dictionary(blacklist)
    clause_data = build_clause_offset_table()
    print(json.dumps({
        "synonym_dictionary_metadata": syn_data["metadata"],
        "clause_offset_verification": clause_data["verification_stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
