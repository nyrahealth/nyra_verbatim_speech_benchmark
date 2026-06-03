"""
Gold token tagging for verbatim ASR evaluation.

Given a verbatim transcript (with disfluencies) and an intended transcript
(clean), plus the edit-distance alignment between them, this module assigns
each verbatim token exactly one tag:

* **FLUENT** — aligned to an intended token with matching normalised form
* **FILLER** — filled pause (``[UH]``, ``[UM]``, …)
* **SOUND**  — non-speech sound event (``[laughter]``, ``[cough]``, …)
* **CUTOFF** — truncated word fragment (``th-``, ``w-``)
* **REP**    — repeated word preceding its FLUENT counterpart
* **OTHER**  — none of the above
  (substitution, false start, discourse marker …)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from normalize import normalize_tokens, is_cutoff, is_filler


def _is_bracket_event(norm_tok: str) -> bool:
    return norm_tok.startswith("[") and norm_tok.endswith("]")


def _is_sound_event(norm_tok: str, language: str = "en") -> bool:
    return _is_bracket_event(norm_tok) and not is_filler(
        norm_tok,
        language=language,
    )


def create_gold_labels(
    v_tokens: List[str],
    i_tokens: List[str],
    alignment: List[Tuple[Optional[int], Optional[int]]],
    language: str = "en",
) -> Dict:
    """
    Derive per-token tags for a verbatim/intended pair.

    Two-pass algorithm:

    **Pass 1** — deterministic surface-form rules (priority order):
    CUTOFF > FILLER > SOUND > FLUENT (aligned + norms match) > OTHER

    **Pass 2** — repetition detection:
    Walk backward from each FLUENT token; consecutive unmatched tokens with
    the same normalised form become REP with a shared ``group_id``.

    Returns:
        ``{"tokens", "tags", "group_ids", "normalized"}``
    """
    v_norm = normalize_tokens(v_tokens)
    i_norm = normalize_tokens(i_tokens)

    v_to_i: Dict[int, int] = {}
    i_to_v: Dict[int, int] = {}
    for v_idx, i_idx in alignment:
        if v_idx is not None and i_idx is not None:
            v_to_i[v_idx] = i_idx
            i_to_v[i_idx] = v_idx

    tags: List[str] = ["OTHER"] * len(v_tokens)
    group_ids: List[Optional[int]] = [None] * len(v_tokens)

    # Pass 1
    for v_idx, tok in enumerate(v_tokens):
        norm = v_norm[v_idx]
        if is_cutoff(tok):
            tags[v_idx] = "CUTOFF"
        elif is_filler(norm, language=language):
            tags[v_idx] = "FILLER"
        elif _is_sound_event(norm, language=language):
            tags[v_idx] = "SOUND"
        elif v_idx in v_to_i and v_norm[v_idx] == i_norm[v_to_i[v_idx]]:
            tags[v_idx] = "FLUENT"
        else:
            tags[v_idx] = "OTHER"

    # Pass 2 — repetition groups
    next_group_id = 0
    for i_idx, v_keep in sorted(i_to_v.items(), key=lambda x: x[0]):
        word = i_norm[i_idx]
        if not word:
            continue
        rep_positions: List[int] = []
        p = v_keep - 1
        while p >= 0:
            if tags[p] == "FLUENT":
                break
            if tags[p] in ("FILLER", "SOUND", "CUTOFF"):
                break
            if v_norm[p] != word:
                break
            if p in v_to_i:
                break
            rep_positions.append(p)
            p -= 1

        if rep_positions:
            gid = next_group_id
            next_group_id += 1
            for rp in rep_positions:
                tags[rp] = "REP"
                group_ids[rp] = gid

    return {
        "tokens": v_tokens,
        "tags": tags,
        "group_ids": group_ids,
        "normalized": v_norm,
    }


def extract_repetition_groups(gold_labels: Dict) -> List[Dict]:
    """Return a structured summary of repetition groups for inspection."""
    tokens: List[str] = gold_labels["tokens"]
    tags: List[str] = gold_labels["tags"]
    group_ids: List[Optional[int]] = gold_labels["group_ids"]
    normalized: List[str] = (
        gold_labels.get("normalized") or normalize_tokens(tokens)
    )

    groups: Dict[int, List[int]] = {}
    for idx, (tag, gid) in enumerate(zip(tags, group_ids)):
        if tag == "REP" and gid is not None:
            groups.setdefault(gid, []).append(idx)

    rep_groups: List[Dict] = []
    for gid, extra_positions in sorted(groups.items()):
        word = normalized[extra_positions[0]]
        kept_position: Optional[int] = None
        start = max(extra_positions) + 1
        for j in range(start, len(tokens)):
            if tags[j] == "FLUENT" and normalized[j] == word:
                kept_position = j
                break

        extra_sorted = sorted(extra_positions)
        extra_count = len(extra_sorted)
        total_count = extra_count + (1 if kept_position is not None else 0)

        rep_groups.append({
            "group_id": gid,
            "word": word,
            "extra_positions": extra_sorted,
            "kept_position": kept_position,
            "extra_count": extra_count,
            "total_count": total_count,
        })

    return rep_groups
