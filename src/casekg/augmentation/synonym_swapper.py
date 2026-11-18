"""SynonymSwapper: deterministic CURIE-locked synonym replacement with zero offset drift."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from casekg.paths import PROJECT_ROOT
from casekg.augmentation.hydrator import (
    BLOCKED_ETIOLOGY_TOKENS,
    NEGATION_CUE_RE,
    normalize_surface,
)
from casekg.augmentation.types import (
    ClauseSpan,
    EntitySpan,
    HydratedDocument,
    ReplacementEdit,
    SpanAlignmentError,
)

ALLOWED_ONTOLOGY_PREFIXES: tuple[str, ...] = ("HP:", "SYMP:", "UBERON:", "MAXO:")

# Filter (a): Technical/CDISC system codes (e.g. 'vstest'), digits, underscores
SYSTEM_CODE_RE = re.compile(
    r"(?:^[a-z]{1,6}test$|^[a-z]{1,4}exam$|^[a-z]{1,4}meas$|[0-9_])",
    re.I,
)

# Filter (b): Latin single-letter / abbreviated anatomical prefixes, Latin nomina & comparative zoology terms
LATIN_PREFIX_RE = re.compile(
    r"^(?:m|n|a|v|r|rr|aa|vv|nn|mm|lig|art|musc|nerv|fasc)\.?\s+",
    re.I,
)
NON_CLINICAL_OR_LATIN_RE = re.compile(
    r"\b(?:musculus|musculi|arteria|arteriae|nervus|nervi|vena|venae|iliotibilias|"
    r"articulatio|foramina|processus|ligamentum|tuberculum|canalis|holocentrum|"
    r"autocentrum|arcocentrum|chordacentrum|pleurocentrum|intercentrum|centra|centrum|"
    r"propodium|epipodium|mesopodium|metapodium|autopodium|stylopodium|zeugopodium|"
    r"hind|forelimb|hindlimb|flipper|flippers|fissura|cerebrocerebellaris|cerebro|"
    r"regio|lumbalis|glutealis|clunis|jecur|vesica|urinaria|pulmo|os|ossa|"
    r"lombus|lumbos|abdominal back|medulla|spinalis|foramen|vertebrale|spatium|epidurale|back of abdomen|"
    r"corpus|ostia|ostium|costiveness|neurapophysis|hemapophysis|"
    r"dorsum|crus|pes|manus|caput|collum|brachium|antebrachium|visage|venter|pectus|"
    r"yolk|shelf|shelves|scale|scales|fin|fins|snout|claw|hoof|gill|larva|"
    r"foetal|fetal|embryonic|teleost|avian|reptilian|amphibian|endochondral|"
    r"pre-cartilage|condensation|mesenchyme|primordium|anlage|somite|"
    r"set|series|cluster|entity|element|organism|character|quality|attribute|"
    r"subdivision|subdomain|zone|proper|sensu|obsolete|"
    r"axial skeleton|initial segment|zeugopod|termination of|wall of|portion of|"
    r"region of|part of back|lumbar skeleton|posterior abdominal wall)\b",
    re.I,
)

# Ambiguous single-word everyday English polysemes that should not be swapped in isolation
AMBIGUOUS_SINGLE_WORDS: frozenset[str] = frozenset({
    "back", "left", "right", "process", "processes", "extension", "flexion",
    "scale", "scales", "face", "head", "neck", "arm", "arms", "leg", "legs",
    "foot", "feet", "hand", "hands", "body", "bodies", "part", "parts",
    "region", "area", "side", "point", "line", "form", "state", "rate",
    "time", "year", "month", "week", "day", "work", "home", "male", "female",
    "man", "woman", "child", "adult", "blood", "bone", "bones", "skin",
    "joint", "joints", "muscle", "muscles", "nerve", "nerves", "tissue",
    "cell", "cells", "lamina", "laminae", "arch", "canal", "space", "disc",
    "disk", "root", "roots", "cord", "tract", "wall", "floor", "roof",
    "base", "apex", "margin", "border", "surface", "layer", "zone", "level",
    "levels", "history", "normal", "acute", "chronic", "mild", "moderate",
    "severe", "focal", "diffuse", "general", "local", "distal", "proximal",
    "anterior", "posterior", "lateral", "medial", "superior", "inferior",
    "internal", "external", "central", "peripheral", "bilateral", "unilateral",
    "pain", "mass", "lesion", "sign", "signs", "symptom", "symptoms",
    "finding", "findings", "test", "tests", "study", "studies", "scan",
    "image", "images", "view", "views", "series", "course", "episode",
    "admission", "discharge", "visit", "clinic", "hospital", "department",
    "patient", "case", "report", "examination", "assessment", "evaluation",
    "therapy", "treatment", "management", "care", "follow", "review",
    "foramen", "skeleton", "toes", "toe", "finger", "fingers", "heel",
    "knee", "hip", "shoulder", "elbow", "wrist", "ankle", "chest", "pelvis",
    "abdomen", "thorax", "spine", "vertebra", "vertebrae", "progressive",
    "disorder", "disease", "syndrome", "condition", "radiation", "thigh",
    "limb", "limbs", "system", "systems", "fluid", "fluids", "liver",
    "bladder", "lung", "lungs", "sacrum", "sclera", "bowel", "onset", "sharp",
})


def preserve_initial_case(original_text: str, replacement_text: str) -> str:
    """Preserve initial capitalization when the original surface starts uppercase."""
    if not original_text or not replacement_text:
        return replacement_text
    if original_text[0].isupper():
        return replacement_text[0].upper() + replacement_text[1:]
    return replacement_text


def is_valid_replacement_candidate(
    candidate: str,
    original: str,
    blacklist: set[str] | None = None,
) -> bool:
    """Apply the 3 mandatory runtime candidate filters before synonym selection.

    Filter (a): Reject technical/system codes (e.g. 'vstest', all-caps codes, digits/underscores).
    Filter (b): Reject Latin single-letter/abbreviated anatomical prefixes ('m ', 'n ', 'a ', 'v ', 'lig ')
                and comparative zoology / non-clinical archaic terms.
    Filter (c): Keep only candidates with length in [4, 60] and length ratio in [1/3, 3.0].
    """
    cand = candidate.strip()
    orig = original.strip()
    if not cand or not orig:
        return False

    # Filter (c): Length [4, 60] and ratio [1/3, 3.0]
    if len(cand) < 4 or len(cand) > 60:
        return False
    ratio = len(cand) / float(len(orig))
    if ratio < (1.0 / 3.0) or ratio > 3.0:
        return False

    norm_cand = normalize_surface(cand)
    norm_orig = normalize_surface(orig)
    if not norm_cand or norm_cand == norm_orig:
        return False

    # Reject swaps where the original span is an ambiguous single common word
    orig_tokens = norm_orig.split()
    if len(orig_tokens) == 1 and orig_tokens[0] in AMBIGUOUS_SINGLE_WORDS:
        return False
    cand_tokens = norm_cand.split()
    if len(cand_tokens) == 1 and cand_tokens[0] in AMBIGUOUS_SINGLE_WORDS:
        return False

    # Reject trivial singular/plural inflections of the exact same word
    if norm_cand + "s" == norm_orig or norm_orig + "s" == norm_cand:
        return False

    # Filter (a): Technical/CDISC system codes, digits, underscores, all-uppercase codes
    if "_" in cand or any(ch.isdigit() for ch in cand):
        return False
    if cand.isupper() and len(cand) <= 10:
        return False
    if any(SYSTEM_CODE_RE.search(tok) for tok in cand_tokens):
        return False

    # Filter (b): Latin single-letter prefixes ('m ', 'n ', 'a ', 'v ', 'lig '), Latin nomina & zoology terms
    if LATIN_PREFIX_RE.search(cand) or LATIN_PREFIX_RE.search(norm_cand):
        return False
    if NON_CLINICAL_OR_LATIN_RE.search(norm_cand) or NON_CLINICAL_OR_LATIN_RE.search(norm_orig):
        return False

    # Do not introduce negation cues or blacklisted disease/etiology terms
    if NEGATION_CUE_RE.search(norm_cand):
        return False
    if blacklist and norm_cand in blacklist:
        return False
    for blocked in BLOCKED_ETIOLOGY_TOKENS:
        if blocked in norm_cand:
            return False
    return True


def apply_replacements(
    doc: HydratedDocument,
    edits: Sequence[ReplacementEdit],
) -> tuple[str, tuple[ClauseSpan, ...], tuple[EntitySpan, ...]]:
    """Apply non-overlapping replacement edits in one pass and recompute all offsets.

    Guarantees 100% zero-drift alignment:
    - `augmented_text[new_span.start:new_span.end] == new_span.text` for every entity span.
    - `augmented_text[new_clause.start:new_clause.end] == new_clause.text` for every clause.
    """
    if not edits:
        return doc.text, doc.clauses, doc.spans

    sorted_edits = sorted(edits, key=lambda e: (e.start, e.end))
    text = doc.text

    prev_end = -1
    edit_by_coord: dict[tuple[int, int], ReplacementEdit] = {}
    for ed in sorted_edits:
        if ed.start < prev_end:
            raise SpanAlignmentError(f"Overlapping edits at [{ed.start}:{ed.end}] in {doc.document_id}")
        if ed.start < 0 or ed.end > len(text) or ed.start >= ed.end:
            raise SpanAlignmentError(f"Out-of-bounds edit [{ed.start}:{ed.end}] in {doc.document_id}")
        if text[ed.start:ed.end] != ed.original_text:
            raise SpanAlignmentError(
                f"Edit original_text mismatch at [{ed.start}:{ed.end}]: "
                f"expected {ed.original_text!r}, found {text[ed.start:ed.end]!r}"
            )
        for sp in doc.spans:
            if (ed.start, ed.end) == (sp.start, sp.end):
                continue
            if ed.start < sp.end and sp.start < ed.end:
                raise SpanAlignmentError(
                    f"Edit [{ed.start}:{ed.end}] partially overlaps span [{sp.start}:{sp.end}]"
                )
        for cl in doc.clauses:
            if ed.start < cl.end and cl.start < ed.end:
                if not (cl.start <= ed.start and ed.end <= cl.end):
                    raise SpanAlignmentError(
                        f"Edit [{ed.start}:{ed.end}] crosses boundary of clause {cl.clause_uid}"
                    )
        prev_end = ed.end
        edit_by_coord[(ed.start, ed.end)] = ed

    pieces: list[str] = []
    cursor = 0
    new_len_so_far = 0
    edit_new_coords: dict[tuple[int, int], tuple[int, int]] = {}
    delta_checkpoints: list[tuple[int, int]] = []
    cumulative_delta = 0

    for ed in sorted_edits:
        prefix = text[cursor:ed.start]
        pieces.append(prefix)
        new_len_so_far += len(prefix)

        new_start = new_len_so_far
        pieces.append(ed.replacement_text)
        new_len_so_far += len(ed.replacement_text)
        new_end = new_len_so_far

        edit_new_coords[(ed.start, ed.end)] = (new_start, new_end)
        cumulative_delta += len(ed.replacement_text) - (ed.end - ed.start)
        delta_checkpoints.append((ed.end, cumulative_delta))
        cursor = ed.end

    pieces.append(text[cursor:])
    augmented_text = "".join(pieces)

    def shift_pos(orig_pos: int) -> int:
        delta = 0
        for orig_end, cum_d in delta_checkpoints:
            if orig_end <= orig_pos:
                delta = cum_d
            else:
                break
        return orig_pos + delta

    new_spans: list[EntitySpan] = []
    for sp in doc.spans:
        key = (sp.start, sp.end)
        if key in edit_new_coords:
            ns, ne = edit_new_coords[key]
            ed = edit_by_coord[key]
            updated = EntitySpan(
                span_id=sp.span_id,
                start=ns,
                end=ne,
                text=ed.replacement_text,
                clause_uid=sp.clause_uid,
                curie=sp.curie,
                semantic_type=sp.semantic_type,
                role=sp.role,
                assertion=sp.assertion,
                source=sp.source,
                protected=sp.protected,
            )
        else:
            ns = shift_pos(sp.start)
            ne = shift_pos(sp.end)
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
        if augmented_text[updated.start:updated.end] != updated.text:
            raise SpanAlignmentError(
                f"Post-swap span alignment failed for {updated.span_id}: "
                f"augmented_text[{updated.start}:{updated.end}]="
                f"{augmented_text[updated.start:updated.end]!r} != {updated.text!r}"
            )
        new_spans.append(updated)

    new_clauses: list[ClauseSpan] = []
    for cl in doc.clauses:
        nc_start = shift_pos(cl.start)
        nc_end = shift_pos(cl.end)
        nc_text = augmented_text[nc_start:nc_end]
        new_clauses.append(
            ClauseSpan(
                clause_uid=cl.clause_uid,
                unit_index=cl.unit_index,
                primary_section=cl.primary_section,
                start=nc_start,
                end=nc_end,
                text=nc_text,
            )
        )

    return augmented_text, tuple(new_clauses), tuple(new_spans)


class SynonymSwapper:
    """Strictly 1-to-1 same-CURIE synonym swapper with clinical candidate filtering."""

    def __init__(
        self,
        synonym_dict_path: Path | None = None,
        *,
        swap_prob: float = 0.45,
        max_swaps_per_doc: int = 8,
    ) -> None:
        if not 0.0 <= swap_prob <= 1.0:
            raise ValueError(f"swap_prob must be in [0.0, 1.0], got {swap_prob}")
        self.swap_prob = swap_prob
        self.max_swaps_per_doc = max_swaps_per_doc
        self.synonym_dict_path = (
            synonym_dict_path
            or PROJECT_ROOT / "resources/augmentation/lbp_synonym_dictionary.json"
        )
        raw = json.loads(self.synonym_dict_path.read_text(encoding="utf-8"))
        self.concepts: dict[str, dict[str, Any]] = raw.get("concepts", {})
        self.lookup_table: dict[str, dict[str, Any]] = raw.get("lookup_table", {})
        self.blacklist: set[str] = {normalize_surface(t) for t in BLOCKED_ETIOLOGY_TOKENS}

    def candidate_synonyms_for_span(self, span: EntitySpan) -> list[str]:
        """Return filtered, clinically natural synonyms sharing the exact same CURIE."""
        if span.protected:
            return []
        curie = span.curie
        norm_orig = normalize_surface(span.text)
        if not norm_orig:
            return []
        if len(norm_orig.split()) == 1 and norm_orig in AMBIGUOUS_SINGLE_WORDS:
            return []

        raw_candidates: list[str] = []
        if curie:
            if not curie.startswith(ALLOWED_ONTOLOGY_PREFIXES):
                return []
            # Exclude teleost/amphibian comparative anatomy UBERON IDs (UBERON:2*, 3*, 4*)
            if curie.startswith("UBERON:") and not curie.startswith("UBERON:0"):
                return []
            concept = self.concepts.get(curie)
            if concept is None:
                return []
            concept_syns = concept.get("synonyms", [])
            if norm_orig not in concept_syns:
                return []
            raw_candidates = list(concept_syns)
        else:
            entry = self.lookup_table.get(norm_orig)
            if entry is None:
                return []
            inferred_curie = str(entry.get("curie") or "")
            if not inferred_curie.startswith(ALLOWED_ONTOLOGY_PREFIXES):
                return []
            if inferred_curie.startswith("UBERON:") and not inferred_curie.startswith("UBERON:0"):
                return []
            raw_candidates = list(entry.get("replacements", []))

        valid: list[str] = []
        for cand in raw_candidates:
            if is_valid_replacement_candidate(cand, span.text, self.blacklist):
                formatted = preserve_initial_case(span.text, cand)
                if formatted != span.text and formatted not in valid:
                    valid.append(formatted)
        return valid

    def propose_edits(
        self,
        doc: HydratedDocument,
        rng: random.Random,
        *,
        ensure_at_least_one: bool = True,
    ) -> tuple[ReplacementEdit, ...]:
        """Select deterministic synonym replacement edits for eligible spans in `doc`."""
        eligible: list[tuple[EntitySpan, list[str]]] = []
        for sp in doc.spans:
            cands = self.candidate_synonyms_for_span(sp)
            if cands:
                eligible.append((sp, cands))

        if not eligible:
            return ()

        chosen_edits: list[ReplacementEdit] = []
        for sp, cands in eligible:
            if len(chosen_edits) >= self.max_swaps_per_doc:
                break
            if rng.random() <= self.swap_prob:
                rep = rng.choice(cands)
                resolved_curie = sp.curie or str(self.lookup_table.get(normalize_surface(sp.text), {}).get("curie") or "")
                resolved_sem = sp.semantic_type or str(self.lookup_table.get(normalize_surface(sp.text), {}).get("semantic_type") or "")
                chosen_edits.append(
                    ReplacementEdit(
                        start=sp.start,
                        end=sp.end,
                        original_text=sp.text,
                        replacement_text=rep,
                        curie=resolved_curie,
                        semantic_type=resolved_sem,
                        clause_uid=sp.clause_uid,
                        reason=f"curie_synonym:{resolved_curie}",
                    )
                )

        if not chosen_edits and ensure_at_least_one:
            sp, cands = rng.choice(eligible)
            rep = rng.choice(cands)
            resolved_curie = sp.curie or str(self.lookup_table.get(normalize_surface(sp.text), {}).get("curie") or "")
            resolved_sem = sp.semantic_type or str(self.lookup_table.get(normalize_surface(sp.text), {}).get("semantic_type") or "")
            chosen_edits.append(
                ReplacementEdit(
                    start=sp.start,
                    end=sp.end,
                    original_text=sp.text,
                    replacement_text=rep,
                    curie=resolved_curie,
                    semantic_type=resolved_sem,
                    clause_uid=sp.clause_uid,
                    reason=f"curie_synonym:{resolved_curie}",
                )
            )

        chosen_edits.sort(key=lambda e: (e.start, e.end))
        return tuple(chosen_edits)
