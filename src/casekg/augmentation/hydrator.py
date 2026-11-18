"""SpanHydrator: merges and verifies document-level entity and clause spans."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from casekg.paths import PROJECT_ROOT
from casekg.augmentation.types import (
    ClauseSpan,
    EntitySpan,
    HydratedDocument,
    SpanAlignmentError,
)

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]")
NEGATION_CUE_RE = re.compile(
    r"\b(?:no|not|never|without|den(?:y|ies|ied)|negative|absence|absent|nil|free)\b",
    re.I,
)

BLOCKED_ETIOLOGY_TOKENS: tuple[str, ...] = (
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


def normalize_surface(text: str) -> str:
    """Normalize text for case-insensitive dictionary and blacklist matching."""
    text = unicodedata.normalize("NFKD", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip().lower()


class SpanHydrator:
    """Hydrates clinical documents with verified document-level clause and entity spans."""

    def __init__(
        self,
        clause_offset_path: Path | None = None,
        predicted_sections_path: Path | None = None,
        semantic_graph_path: Path | None = None,
        synonym_dict_path: Path | None = None,
        paraphrase_config_path: Path | None = None,
        text_field: str = "prediagnosis_text",
    ) -> None:
        self.text_field = text_field
        self.clause_offset_path = (
            clause_offset_path
            or PROJECT_ROOT / "resources/augmentation/clause_offset_table.json"
        )
        self.predicted_sections_path = (
            predicted_sections_path
            or PROJECT_ROOT / "results/section_conditioned_diagnosis/evidence/PREDICTED_SECTIONS.jsonl"
        )
        self.semantic_graph_path = (
            semantic_graph_path
            or PROJECT_ROOT / "results/evidence_kernel_diagnosis/features/SEMANTIC_GRAPH_EVIDENCE.jsonl"
        )
        self.synonym_dict_path = (
            synonym_dict_path
            or PROJECT_ROOT / "resources/augmentation/lbp_synonym_dictionary.json"
        )
        self.paraphrase_config_path = (
            paraphrase_config_path
            or PROJECT_ROOT / "resources/augmentation/paraphrase_config.json"
        )

        self._clause_table: dict[str, dict[str, Any]] = {}
        if self.clause_offset_path.exists():
            payload = json.loads(self.clause_offset_path.read_text(encoding="utf-8"))
            self._clause_table = payload.get("documents", {})

        self._section_units: dict[str, list[dict[str, Any]]] = {}
        if self.predicted_sections_path.exists():
            with open(self.predicted_sections_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._section_units[rec["document_id"]] = rec.get("units", [])

        self._semantic_mentions: dict[str, list[dict[str, Any]]] = {}
        if self.semantic_graph_path.exists():
            with open(self.semantic_graph_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._semantic_mentions[rec["document_id"]] = rec.get("mentions", [])

        self._swappable_curies: set[str] = set()
        self._unambiguous_lookup: dict[str, dict[str, Any]] = {}
        if self.synonym_dict_path.exists():
            syn_data = json.loads(self.synonym_dict_path.read_text(encoding="utf-8"))
            self._swappable_curies = set(syn_data.get("concepts", {}).keys())
            self._unambiguous_lookup = syn_data.get("lookup_table", {})

        self._immutable_regexes: list[re.Pattern[str]] = []
        if self.paraphrase_config_path.exists():
            p_cfg = json.loads(self.paraphrase_config_path.read_text(encoding="utf-8"))
            for item in p_cfg.get("immutable_patterns", []):
                self._immutable_regexes.append(re.compile(item["regex"], re.I))

        self.blacklist: set[str] = {normalize_surface(t) for t in BLOCKED_ETIOLOGY_TOKENS}

    def is_blacklisted_text(self, text: str, row_blacklist: set[str] | None = None) -> bool:
        """Return True if text matches or touches any diagnosis/etiology blacklist term."""
        norm = normalize_surface(text)
        if not norm:
            return False
        if norm in self.blacklist or (row_blacklist and norm in row_blacklist):
            return True
        for token in BLOCKED_ETIOLOGY_TOKENS:
            if token in norm:
                return True
        if row_blacklist:
            for diag in row_blacklist:
                if len(diag) >= 6 and (diag in norm or norm in diag):
                    return True
        return False

    def _immutable_intervals(self, text: str) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        for rx in self._immutable_regexes:
            for m in rx.finditer(text):
                intervals.append((m.start(), m.end()))
        return intervals

    @staticmethod
    def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start < b_end and b_start < a_end

    def _build_clauses(self, doc_id: str, text: str, units: Sequence[ Mapping[str, Any]] | None = None) -> tuple[ClauseSpan, ...]:
        raw_units = list(units) if units is not None else self._section_units.get(doc_id, [])
        if not raw_units:
            return (
                ClauseSpan(
                    clause_uid=f"primary:{doc_id}:0",
                    unit_index=0,
                    primary_section="PRESENTING_SYMPTOMS",
                    start=0,
                    end=len(text),
                    text=text,
                ),
            )

        cached_clauses = self._clause_table.get(doc_id, {}).get("clauses", {})
        clauses: list[ClauseSpan] = []
        cursor = 0
        for idx, u in enumerate(raw_units):
            uid = str(u["clause_uid"])
            u_text = str(u["text"])
            c_meta = cached_clauses.get(uid)
            if c_meta is not None:
                c_start = int(c_meta["clause_start"])
                c_end = int(c_meta["clause_end"])
            else:
                pos = text.find(u_text, cursor)
                if pos < 0:
                    pos = text.find(u_text)
                if pos < 0:
                    raise SpanAlignmentError(f"Clause {uid!r} not found in document {doc_id!r}")
                c_start = pos
                c_end = pos + len(u_text)

            if text[c_start:c_end] != u_text:
                raise SpanAlignmentError(
                    f"Clause offset mismatch for {uid!r}: expected {u_text!r}, got {text[c_start:c_end]!r}"
                )
            cursor = c_end
            clauses.append(
                ClauseSpan(
                    clause_uid=uid,
                    unit_index=int(u.get("unit_index", idx)),
                    primary_section=str(u.get("primary_section", "PRESENTING_SYMPTOMS")),
                    start=c_start,
                    end=c_end,
                    text=u_text,
                )
            )
        return tuple(clauses)

    def hydrate_row(
        self,
        row: Mapping[str, Any],
        *,
        custom_units: Sequence[Mapping[str, Any]] | None = None,
        custom_mentions: Sequence[Mapping[str, Any]] | None = None,
    ) -> HydratedDocument:
        """Hydrate a single clinical case row with verified document-level spans."""
        doc_id = str(row["document_id"])
        dataset = str(row.get("dataset") or row.get("split") or "primary_train")
        family = str(row.get("disease_family_label") or "UNKNOWN")
        group_id = str(row.get("lineage_group_id") or row.get("group_id") or doc_id)
        text = str(row.get(self.text_field) or row.get("input_text") or "")
        if not text:
            raise SpanAlignmentError(f"Document {doc_id!r} has empty text")

        row_blacklist: set[str] = set()
        diag_str = row.get("specific_final_diagnosis_text")
        if diag_str:
            norm_d = normalize_surface(str(diag_str))
            if norm_d:
                row_blacklist.add(norm_d)
                for part in norm_d.split(" due to "):
                    if part.strip():
                        row_blacklist.add(part.strip())
        for opt in row.get("multiple_choice_options", []):
            norm_o = normalize_surface(str(opt))
            if norm_o:
                row_blacklist.add(norm_o)

        clauses = self._build_clauses(doc_id, text, custom_units)
        clause_by_uid = {c.clause_uid: c for c in clauses}

        # Collect candidate spans in document coordinates
        by_coord: dict[tuple[int, int], dict[str, Any]] = {}

        units_src = list(custom_units) if custom_units is not None else self._section_units.get(doc_id, [])
        for u in units_src:
            uid = str(u["clause_uid"])
            c_span = clause_by_uid.get(uid)
            if c_span is None:
                continue
            for lk in u.get("links", []):
                rel_s = int(lk["start"])
                rel_e = int(lk["end"])
                doc_s = c_span.start + rel_s
                doc_e = c_span.start + rel_e
                lk_text = str(lk["text"])
                if text[doc_s:doc_e] != lk_text:
                    raise SpanAlignmentError(
                        f"Link span mismatch in {uid}: text[{doc_s}:{doc_e}]={text[doc_s:doc_e]!r} != {lk_text!r}"
                    )
                curie = str(lk.get("curie") or "")
                sem_type = str(lk.get("semantic_type") or "")
                assertion = str(lk.get("assertion") or "AFFIRMED")
                entry = by_coord.get((doc_s, doc_e))
                if entry is None or (curie in self._swappable_curies and entry["curie"] not in self._swappable_curies):
                    by_coord[(doc_s, doc_e)] = {
                        "start": doc_s,
                        "end": doc_e,
                        "text": lk_text,
                        "clause_uid": uid,
                        "curie": curie,
                        "semantic_type": sem_type,
                        "role": c_span.primary_section,
                        "assertion": assertion,
                        "source": "predicted_sections",
                    }

        mentions_src = list(custom_mentions) if custom_mentions is not None else self._semantic_mentions.get(doc_id, [])
        for m in mentions_src:
            uid = str(m["clause_uid"])
            c_span = clause_by_uid.get(uid)
            if c_span is None:
                continue
            rel_s, rel_e = int(m["span"][0]), int(m["span"][1])
            doc_s = c_span.start + rel_s
            doc_e = c_span.start + rel_e
            m_text = str(m["phrase"])
            if text[doc_s:doc_e] != m_text:
                raise SpanAlignmentError(
                    f"Mention span mismatch in {uid}: text[{doc_s}:{doc_e}]={text[doc_s:doc_e]!r} != {m_text!r}"
                )
            existing = by_coord.get((doc_s, doc_e))
            if existing is not None:
                existing["role"] = str(m.get("role") or existing["role"])
                existing["assertion"] = str(m.get("assertion_channel") or existing["assertion"])
            else:
                norm_m = normalize_surface(m_text)
                inferred = self._unambiguous_lookup.get(norm_m, {})
                by_coord[(doc_s, doc_e)] = {
                    "start": doc_s,
                    "end": doc_e,
                    "text": m_text,
                    "clause_uid": uid,
                    "curie": str(inferred.get("curie") or ""),
                    "semantic_type": str(inferred.get("semantic_type") or ""),
                    "role": str(m.get("role") or c_span.primary_section),
                    "assertion": str(m.get("assertion_channel") or "PRESENT"),
                    "source": "semantic_graph",
                }

        # Resolve overlapping spans: prefer CURIE-linked swappable spans, then longer spans,
        # while marking any span that had a partial boundary collision as protected if needed.
        raw_list = list(by_coord.values())

        def _priority(item: dict[str, Any]) -> tuple[int, int, int]:
            c = item["curie"]
            is_swappable = int(bool(c) and c in self._swappable_curies and not c.startswith("MONDO:"))
            has_curie = int(bool(c) and not c.startswith("MONDO:"))
            length = item["end"] - item["start"]
            return (is_swappable, has_curie, length)

        ranked = sorted(raw_list, key=lambda x: (-_priority(x)[0], -_priority(x)[1], -_priority(x)[2], x["start"]))
        selected: list[dict[str, Any]] = []
        for cand in ranked:
            if any(self._overlaps(cand["start"], cand["end"], s["start"], s["end"]) for s in selected):
                continue
            selected.append(cand)

        selected.sort(key=lambda x: (x["start"], x["end"]))
        immutable_spans = self._immutable_intervals(text)

        final_spans: list[EntitySpan] = []
        for idx, item in enumerate(selected):
            s = item["start"]
            e = item["end"]
            sp_text = item["text"]
            curie = item["curie"]
            if text[s:e] != sp_text:
                raise SpanAlignmentError(f"Internal alignment failure at [{s}:{e}] in {doc_id}")

            # Check word boundaries
            left_ok = s == 0 or not text[s - 1].isalnum()
            right_ok = e == len(text) or not text[e].isalnum()

            # Check immutable overlap, negation cue, MONDO disease curie, embedded punctuation, or blacklist touch
            touches_immutable = any(self._overlaps(s, e, im_s, im_e) for im_s, im_e in immutable_spans)
            has_negation_word = bool(NEGATION_CUE_RE.search(sp_text))
            has_embedded_punct = any(ch in sp_text for ch in "()[]{};:,/\\")
            is_mondo = curie.startswith("MONDO:")
            touches_blacklist = self.is_blacklisted_text(sp_text, row_blacklist)

            protected = (
                not left_ok
                or not right_ok
                or touches_immutable
                or has_negation_word
                or has_embedded_punct
                or is_mondo
                or touches_blacklist
            )
            final_spans.append(
                EntitySpan(
                    span_id=f"{doc_id}:span:{idx}",
                    start=s,
                    end=e,
                    text=sp_text,
                    clause_uid=item["clause_uid"],
                    curie=curie,
                    semantic_type=item["semantic_type"],
                    role=item["role"],
                    assertion=item["assertion"],
                    source=item["source"],
                    protected=protected,
                )
            )

        return HydratedDocument(
            document_id=doc_id,
            dataset=dataset,
            disease_family_label=family,
            lineage_group_id=group_id,
            text_field=self.text_field,
            text=text,
            clauses=clauses,
            spans=tuple(final_spans),
            row=dict(row),
        )
