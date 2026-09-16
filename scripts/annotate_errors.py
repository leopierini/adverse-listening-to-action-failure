"""Categorize ASR errors per the docs/design/METRICS.md §8 / §10 schema.

For each row in the input JSONL (which must already have a non-null transcription),
align the hypothesis against the gold sentence and tag each alignment "chunk" (a
substitution / insertion / deletion span) with:

    type           — substitute | insert | delete
    error_subtype  — for substitutions only:
                       phonetic_similar  (Metaphone hashes share at least one code)
                       number_error      (ref or hyp is a number word/digit, different value)
                       named_entity_error (ref word falls inside a SLURP entity of type
                                          person/event_name/place_name/etc.)
                       semantic_drift    (anything else)
    slot_position  — critical (overlaps a gold-entity token span) | function (does not)
    ref            — the reference word(s) for this chunk
    hyp            — the hypothesis word(s) for this chunk
    entity_type    — the SLURP entity type at the ref position, or null

Reads:  --in   (JSONL from batch_evaluate.py or compute_metrics.py)
Writes: --out  (same rows, with row["error_categories"] populated)

Usage:
    python annotate_errors.py \\
        --in  results/clean_baseline_full_metrics.jsonl \\
        --out results/clean_baseline_full_annotated.jsonl
"""

import argparse

import io_guards
import json
import re
import string
from pathlib import Path

import jiwer
from metaphone import doublemetaphone

DEFAULT_DATASET = "dataset/gold_devel_416.jsonl"  # SLURP/ removed 2026-06-02

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")

# Entity types that count as "named entity" for the named-entity-error category.
NAMED_ENTITY_TYPES = {
    "person", "event_name", "place_name", "business_name", "food_type",
    "list_name", "podcast_name", "song_name", "artist_name", "playlist_name",
    "media_type", "movie_name", "music_genre", "transport_name", "weather_descriptor",
}

# Number words we treat as "numeric" for the number-error category.
NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion",
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
    "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth",
    "eighteenth","nineteenth","twentieth",
    "once","twice",
}

_DIGIT_RE = re.compile(r"^\d+([.,]\d+)?$")


def is_numeric(word: str) -> bool:
    if not word:
        return False
    w = word.lower().strip()
    if _DIGIT_RE.match(w):
        return True
    return w in NUMBER_WORDS


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


_APOS_RE = re.compile(r"['’ʼ]")  # ASCII + typographic apostrophes


def normalize_hyp_text(s: str) -> str:
    """Hypothesis normalizer for alignment against gold tokens.

    Apostrophes are DELETED (o'clock → oclock), matching gold_token_surfaces,
    which strips punctuation *within* each token. Replacing them with a space
    (as normalize_text does) would split o'clock into two hyp tokens while the
    reference keeps one → spurious substitution+insertion chunks on every
    contraction. Other punctuation still becomes a space."""
    if s is None:
        return ""
    s = s.lower()
    s = _APOS_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def phonetic_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    ma = set(c for c in doublemetaphone(a) if c)
    mb = set(c for c in doublemetaphone(b) if c)
    return bool(ma & mb)


_CONTENT_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at",
    "my", "me", "is", "it", "this", "that", "with", "from", "all", "any",
}


def _dm_code(word):
    """Primary Double-Metaphone code (fall back to the secondary)."""
    a, b = doublemetaphone(word or "")
    return a or b or ""


def _levenshtein(s, t):
    if s == t:
        return 0
    if not s:
        return len(t)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s, 1):
        cur = [i]
        for j, ct in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cs != ct)))
        prev = cur
    return prev[-1]


def double_metaphone_distance(ref, hyp):
    """Normalized Levenshtein over the Double-Metaphone codes of ref/hyp, in [0,1].
    The confirmatory H3 predictor (Philips 2000): 0 = phonetically identical,
    1 = maximally different. Multi-word spans: concatenate per-word codes."""
    rc = "".join(_dm_code(w) for w in (ref or "").split())
    hc = "".join(_dm_code(w) for w in (hyp or "").split())
    if not rc and not hc:
        return 0.0
    if not rc or not hc:
        return 1.0
    return _levenshtein(rc, hc) / max(len(rc), len(hc))


def row_phonetic_distance(categories):
    """Mean Double-Metaphone distance over critical-slot substitution chunks.
    None if the row has no such chunk (→ excluded from the H3 fit)."""
    dists = [double_metaphone_distance(c.get("ref"), c.get("hyp"))
             for c in categories
             if c.get("type") == "substitute" and c.get("slot_position") == "critical"]
    return sum(dists) / len(dists) if dists else None


def is_full_hallucination(wer, ref_tokens, hyp_tokens):
    """True when WER≥1 AND the hypothesis is NON-EMPTY AND ref/hyp share no
    content token — Whisper invented text rather than mis-transcribing (tracker
    §8). An empty hypothesis is a TOTAL DELETION (scored WER=1.0 upstream by
    compute_metrics.py) — its own category, not a hallucination. Content =
    tokens minus stopwords."""
    if wer is None or wer < 1.0:
        return False
    if not hyp_tokens:
        return False
    rc = {t for t in ref_tokens if t and t not in _CONTENT_STOPWORDS}
    hc = {t for t in hyp_tokens if t and t not in _CONTENT_STOPWORDS}
    return len(rc & hc) == 0


def load_gold_index(dataset_path: str) -> dict:
    idx = {}
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("slurp_id")
            if sid is None:
                continue
            idx[sid] = {
                "sentence": d.get("sentence", ""),
                "tokens": d.get("tokens", []),
                "entities": d.get("entities", []),
            }
    return idx


def build_token_to_entity_map(gold: dict) -> dict:
    """token_index → entity_type (string), or absent if not part of any entity."""
    mapping = {}
    for ent in gold.get("entities", []):
        etype = ent.get("type")
        for tok_idx in ent.get("span", []):
            mapping[tok_idx] = etype
    return mapping


def gold_token_surfaces(gold: dict) -> list:
    """Lowercase, punctuation-stripped surface forms aligned to the SLURP token indices.
    Note: SLURP's `tokens` field is the canonical reference ordering."""
    out = []
    for t in gold.get("tokens", []):
        s = (t.get("surface") or "").lower()
        s = _APOS_RE.sub("", s)   # typographic apostrophes are not in string.punctuation
        s = _PUNCT_RE.sub("", s).strip()
        out.append(s)
    return out


def classify_substitution(ref_word: str, hyp_word: str, entity_type) -> str:
    """Return the error_subtype for one substitution."""
    if entity_type and entity_type in NAMED_ENTITY_TYPES:
        return "named_entity_error"
    if is_numeric(ref_word) or is_numeric(hyp_word):
        if ref_word != hyp_word:
            return "number_error"
    if phonetic_similar(ref_word, hyp_word):
        return "phonetic_similar"
    return "semantic_drift"


def annotate_row(row: dict, gold: dict) -> list:
    """Build the error_categories list for one row, using the gold annotations."""
    hyp = row.get("transcription")
    if hyp is None:
        return [], [], []
    surfaces = gold_token_surfaces(gold)
    # Keep only non-empty normalized tokens, remembering each one's ORIGINAL
    # SLURP index. jiwer drops empty strings from the word list, which would
    # silently shift every later index off the gold entity spans.
    ref_tokens, ref_orig_idx = [], []
    for i, s in enumerate(surfaces):
        if s:
            ref_tokens.append(s)
            ref_orig_idx.append(i)
    if not ref_tokens:
        # Fall back to splitting the gold sentence (entity spans can't be
        # resolved in this case; positions are identity-mapped)
        ref_tokens = normalize_hyp_text(gold.get("sentence", "")).split()
        ref_orig_idx = list(range(len(ref_tokens)))
    hyp_tokens = normalize_hyp_text(hyp).split()
    if not ref_tokens or not hyp_tokens:
        return [], ref_tokens, hyp_tokens

    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    chunks = out.alignments[0]
    tok_to_ent = build_token_to_entity_map(gold)

    categories = []
    for c in chunks:
        if c.type == "equal":
            continue
        # ref indices for this chunk
        ref_idxs = list(range(c.ref_start_idx, c.ref_end_idx))
        hyp_idxs = list(range(c.hyp_start_idx, c.hyp_end_idx))
        ref_words = " ".join(ref_tokens[i] for i in ref_idxs)
        hyp_words = " ".join(hyp_tokens[i] for i in hyp_idxs)
        # Determine slot position: critical if any ref index is inside an entity
        # (entity spans are looked up via the original SLURP token indices)
        entity_types_hit = [tok_to_ent[ref_orig_idx[i]] for i in ref_idxs
                            if ref_orig_idx[i] in tok_to_ent]
        critical = bool(entity_types_hit)
        ent_type = entity_types_hit[0] if entity_types_hit else None

        if c.type == "substitute":
            # If multiple words substituted at once, classify based on the first pair
            if ref_idxs and hyp_idxs:
                rw = ref_tokens[ref_idxs[0]]
                hw = hyp_tokens[hyp_idxs[0]]
            else:
                rw = ref_words
                hw = hyp_words
            subtype = classify_substitution(rw, hw, ent_type)
        elif c.type == "delete":
            # Deletion of slot vs function word
            subtype = None  # left for now; slot_position carries the load-bearing signal
        elif c.type == "insert":
            subtype = None
        else:
            subtype = None

        categories.append({
            "type": c.type,                       # substitute | delete | insert
            "error_subtype": subtype,             # for substitutions; None otherwise
            "slot_position": "critical" if critical else "function",
            "ref": ref_words,
            "hyp": hyp_words,
            "entity_type": ent_type,
        })
    return categories, ref_tokens, hyp_tokens


def main():
    p = argparse.ArgumentParser(description="Annotate ASR errors with subtype + slot position.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    args = p.parse_args()
    # A scoring script truncates its output before reading its input;
    # --in X --out X therefore empties X and exits 0 (scripts/io_guards.py).
    io_guards.refuse_in_place_or_exit(args.in_path, args.out)

    print(f"📋 Loading SLURP gold index from {args.dataset}")
    gold = load_gold_index(args.dataset)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_annotated = 0
    n_errors = 0
    by_subtype = {}
    by_slot_position = {"critical": 0, "function": 0}

    with open(args.in_path) as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            if row.get("pathway") == "omni":
                row.setdefault("phonetic_distance", None)
                row.setdefault("full_hallucination", None)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            sid = row.get("slurp_id")
            g = gold.get(sid)
            if g is None or row.get("transcription") is None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            cats, ref_tokens, hyp_tokens = annotate_row(row, g)
            row["error_categories"] = cats
            row["phonetic_distance"] = row_phonetic_distance(cats)
            row["full_hallucination"] = is_full_hallucination(
                row.get("wer"), ref_tokens, hyp_tokens)
            for c in cats:
                n_errors += 1
                by_slot_position[c["slot_position"]] = by_slot_position.get(c["slot_position"], 0) + 1
                key = c.get("error_subtype") or c["type"]
                by_subtype[key] = by_subtype.get(key, 0) + 1
            if cats:
                n_annotated += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {n_rows} rows → {args.out}")
    print(f"   Rows with at least one error annotation: {n_annotated}")
    print(f"   Total error chunks: {n_errors}")
    if n_errors:
        print(f"\n   By type/subtype:")
        for k, v in sorted(by_subtype.items(), key=lambda kv: -kv[1]):
            print(f"     {k:25s}: {v}")
        print(f"\n   By slot position:")
        for k, v in by_slot_position.items():
            print(f"     {k:25s}: {v}")


if __name__ == "__main__":
    main()
