#!/usr/bin/env python3
"""
Verbatim ASR Evaluation — Step 2.

Loads gold transcripts from a HuggingFace dataset or cached prediction rows,
then evaluates predicted verbatim and intended transcripts.

Usage::

    python evaluate.py \\
        --predictions predictions.jsonl \\
        --datasets nyrahealth/disfluency_speech_english \\
        --splits test \\
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from normalize import (
    normalize_cutoff_markers,
    normalize_for_wer,
    normalize_token,
    normalize_tokens,
    tokenize,
    is_filler,
    is_sound_event,
    is_cutoff,
    get_event_label,
    FILLER_LABELS,
    SOUND_EVENT_LABELS,
)
from align import align_sequences
from tag import create_gold_labels, extract_repetition_groups


# ═══════════════════════════════════════════════════════════════════════════════
# Gold Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_gold(
    verbatim: str,
    intended: str,
    sample_id: str,
    language: str = "en",
    cutoff_source: str = "*",
    cutoff_target: str = "-",
) -> Dict:
    """
    Convert a (verbatim, intended) pair into a fully tagged gold sample.

    1. Normalise cutoff markers.
    2. Tokenize and normalize.
    3. Align verbatim → intended (uniform costs, ``tags=None``).
    4. Derive per-token tags via two-pass algorithm.
    """
    verbatim = normalize_cutoff_markers(verbatim, source=cutoff_source, target=cutoff_target)
    intended = normalize_cutoff_markers(intended, source=cutoff_source, target=cutoff_target)

    v_tokens = tokenize(verbatim)
    i_tokens = tokenize(intended)
    v_norm = normalize_tokens(v_tokens)
    i_norm = normalize_tokens(i_tokens)

    alignment = align_sequences(v_tokens, i_tokens, v_norm, i_norm, tags=None)
    gold_labels = create_gold_labels(v_tokens, i_tokens, alignment, language=language)
    rep_groups = extract_repetition_groups(gold_labels)

    return {
        "id": sample_id,
        "gold_verbatim": verbatim,
        "gold_intended": intended,
        "language": language,
        "gold_tokens": gold_labels["tokens"],
        "gold_tags": gold_labels["tags"],
        "gold_group_ids": gold_labels["group_ids"],
        "gold_normalized": gold_labels["normalized"],
        "gold_rep_groups": rep_groups,
    }


def _language_from_dataset(dataset_name: str) -> str:
    """Infer benchmark language from the dataset name when rows omit it."""
    return "de" if "german" in (dataset_name or "").lower() else "en"


# ═══════════════════════════════════════════════════════════════════════════════
# Edit Distance Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def _edit_distance_detailed(ref: List[str], hyp: List[str]) -> Dict[str, int]:
    """Edit distance with S/D/I breakdown."""
    n, m = len(ref), len(hyp)
    INF = float("inf")
    dp = [[(INF, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dc = dp[i - 1][j]
                d_val = (dc[0] + 1, dc[1], dc[2] + 1, dc[3])
                ic = dp[i][j - 1]
                i_val = (ic[0] + 1, ic[1], ic[2], ic[3] + 1)
                sc = dp[i - 1][j - 1]
                s_val = (sc[0] + 1, sc[1] + 1, sc[2], sc[3])
                dp[i][j] = min([d_val, i_val, s_val], key=lambda x: x[0])
    total, subs, dels, ins = dp[n][m]
    return {"total": total, "substitutions": subs, "deletions": dels, "insertions": ins}


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length (order-preserving containment)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = tmp
    return dp[m]


# ═══════════════════════════════════════════════════════════════════════════════
# Repetition Detection (for predictions)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_repetition_start(tokens: List[str], idx: int, norm_tokens: List[str]) -> bool:
    if idx >= len(tokens) - 1:
        return False
    norm = norm_tokens[idx]
    if not norm or norm.startswith("[") or norm.rstrip(".,!?;:'\"").endswith("-"):
        return False
    if all(c in ".,!?;:'\"-()" for c in norm):
        return False
    return norm == norm_tokens[idx + 1]


def _detect_pred_reps(pred_tokens: List[str], pred_norm: List[str]) -> List[int]:
    """Return indices of predicted repetition tokens (all but last in each run)."""
    reps: List[int] = []
    i = 0
    while i < len(pred_norm) - 1:
        if _is_repetition_start(pred_tokens, i, pred_norm):
            j = i
            while j < len(pred_norm) - 1 and pred_norm[j] == pred_norm[j + 1]:
                reps.append(j)
                j += 1
            i = j + 1
        else:
            i += 1
    return reps


def _count_pred_rep_groups(pred_tokens: List[str], pred_norm: List[str]) -> int:
    groups = 0
    i = 0
    while i < len(pred_norm) - 1:
        if _is_repetition_start(pred_tokens, i, pred_norm):
            groups += 1
            while i < len(pred_norm) - 1 and pred_norm[i] == pred_norm[i + 1]:
                i += 1
            i += 1
        else:
            i += 1
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Precision / Recall / F1 Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


# ═══════════════════════════════════════════════════════════════════════════════
# Sample-Level Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sample(
    gold: Dict,
    pred_verbatim: str,
    pred_intended: Optional[str] = None,
) -> Dict:
    """
    Evaluate a single sample.

    Computes:
      - String-level metrics (vWER, vCER, iWER, iCER, DIER, fluent deletion)
      - Event-level metrics (filler / sound / cutoff / rep F1 + variants)
      - Diagnostics (tag confusion, omission / hallucination rates)
    """
    gold_tokens: List[str] = gold["gold_tokens"]
    gold_tags: List[str] = gold["gold_tags"]
    gold_group_ids = gold["gold_group_ids"]
    language = gold.get("language", "en")

    if pred_intended is None:
        pred_intended = pred_verbatim

    # ── String-level metrics ──────────────────────────────────────────────
    gv_wer = normalize_for_wer(gold["gold_verbatim"]).split()
    pv_wer = normalize_for_wer(pred_verbatim).split()
    v_edits = _edit_distance(gv_wer, pv_wer)
    v_ref = len(gv_wer)
    vWER = (v_edits / v_ref) if v_ref > 0 else None

    v_ref_s = " ".join(gv_wer)
    v_hyp_s = " ".join(pv_wer)
    v_c_edits = _edit_distance(list(v_ref_s), list(v_hyp_s))
    v_c_ref = len(v_ref_s)
    vCER = (v_c_edits / v_c_ref) if v_c_ref > 0 else None

    gi_wer = normalize_for_wer(gold["gold_intended"]).split()
    pi_wer = normalize_for_wer(pred_intended).split()
    i_details = _edit_distance_detailed(gi_wer, pi_wer)
    i_edits = i_details["total"]
    i_ref = len(gi_wer)
    iWER = (i_edits / i_ref) if i_ref > 0 else None
    iSR = (i_details["substitutions"] / i_ref) if i_ref > 0 else None
    iDR = (i_details["deletions"] / i_ref) if i_ref > 0 else None
    iIR = (i_details["insertions"] / i_ref) if i_ref > 0 else None

    gi_ref_s = " ".join(gi_wer)
    pi_hyp_s = " ".join(pi_wer)
    i_c_edits = _edit_distance(list(gi_ref_s), list(pi_hyp_s))
    i_c_ref = len(gi_ref_s)
    iCER = (i_c_edits / i_c_ref) if i_c_ref > 0 else None

    # Fluent deletion rate (LCS-based)
    pv_tokens_norm = normalize_tokens(tokenize(pred_verbatim))
    gold_content_norm = [
        normalize_token(t) for t, tag in zip(gold_tokens, gold_tags) if tag == "FLUENT"
    ]
    fluent_ref = len(gold_content_norm)
    fluent_matched = _lcs_len(gold_content_norm, pv_tokens_norm)
    fluent_deletion_rate = (1.0 - fluent_matched / fluent_ref) if fluent_ref > 0 else None

    # ── Event-level metrics (alignment-based) ─────────────────────────────
    pred_tokens = tokenize(pred_verbatim)
    gold_norm = normalize_tokens(gold_tokens)
    pred_norm = normalize_tokens(pred_tokens)

    # Align gold verbatim vs predicted verbatim (tag-aware)
    eval_alignment = align_sequences(
        gold_tokens, pred_tokens, gold_norm, pred_norm, tags=gold_tags
    )

    gold_to_pred: Dict[int, int] = {}
    pred_to_gold: Dict[int, int] = {}
    for g_idx, p_idx in eval_alignment:
        if g_idx is not None and p_idx is not None:
            gold_to_pred[g_idx] = p_idx
            pred_to_gold[p_idx] = g_idx

    # Gold positions by tag
    gold_filler_pos = [i for i, t in enumerate(gold_tags) if t == "FILLER"]
    gold_sound_pos = [i for i, t in enumerate(gold_tags) if t == "SOUND"]
    gold_cutoff_pos = [i for i, t in enumerate(gold_tags) if t == "CUTOFF"]
    gold_rep_pos = [i for i, t in enumerate(gold_tags) if t == "REP"]
    gold_event_pos = sorted(set(gold_filler_pos) | set(gold_sound_pos))

    # Pred positions by surface form
    pred_filler_pos = [
        i for i, tok in enumerate(pred_tokens)
        if is_filler(tok, language=language)
    ]
    pred_sound_pos = [i for i, tok in enumerate(pred_tokens) if is_sound_event(tok)]
    pred_cutoff_pos = [i for i, tok in enumerate(pred_tokens) if is_cutoff(tok)]
    pred_event_pos = sorted(set(pred_filler_pos) | set(pred_sound_pos))
    pred_rep_pos = _detect_pred_reps(pred_tokens, pred_norm)

    def _aligned_match(gold_pos: List[int], pred_pos: List[int]) -> Tuple[int, int, int]:
        pred_in_gold: Set[int] = set()
        unaligned = 0
        for p in pred_pos:
            if p in pred_to_gold:
                pred_in_gold.add(pred_to_gold[p])
            else:
                unaligned += 1
        gold_set = set(gold_pos)
        tp = len(gold_set & pred_in_gold)
        fp = len(pred_in_gold - gold_set) + unaligned
        fn = len(gold_set - pred_in_gold)
        return tp, fp, fn

    filler_tp, filler_fp, filler_fn = _aligned_match(gold_filler_pos, pred_filler_pos)
    sound_tp, sound_fp, sound_fn = _aligned_match(gold_sound_pos, pred_sound_pos)
    cutoff_tp, cutoff_fp, cutoff_fn = _aligned_match(gold_cutoff_pos, pred_cutoff_pos)
    event_tp, event_fp, event_fn = _aligned_match(gold_event_pos, pred_event_pos)
    rep_tp, rep_fp, rep_fn = _aligned_match(gold_rep_pos, pred_rep_pos)

    # Cutoff matching variants
    def _strip_cutoff_marker(tok: str) -> str:
        return tok.rstrip("-.,!?;:'\"").lower()

    cutoff_strict_tp = cutoff_strict_fn = 0
    cutoff_lenient_tp = cutoff_lenient_fn = 0
    for g_idx in gold_cutoff_pos:
        if g_idx in gold_to_pred:
            p_idx = gold_to_pred[g_idx]
            cutoff_lenient_tp += 1
            if _strip_cutoff_marker(gold_tokens[g_idx]) == _strip_cutoff_marker(pred_tokens[p_idx]):
                cutoff_strict_tp += 1
            else:
                cutoff_strict_fn += 1
        else:
            cutoff_strict_fn += 1
            cutoff_lenient_fn += 1
    cutoff_strict_fp = cutoff_fp
    cutoff_lenient_fp = cutoff_fp

    # Exact label matching
    def _exact_label_match(
        gpos: List[int],
    ) -> Tuple[int, int, int]:
        tp = fn = 0
        matched_pred: Set[int] = set()
        for g in gpos:
            gl = get_event_label(gold_tokens[g], language=language)
            if g in gold_to_pred:
                p = gold_to_pred[g]
                pl = get_event_label(pred_tokens[p], language=language)
                if gl and pl and gl == pl:
                    tp += 1
                    matched_pred.add(p)
                else:
                    fn += 1
            else:
                fn += 1
        all_pred_events = {
            i for i, tok in enumerate(pred_tokens)
            if is_filler(tok, language=language) or is_sound_event(tok)
        }
        fp = len(all_pred_events - matched_pred)
        return tp, fp, fn

    exact_filler_tp, exact_filler_fp, exact_filler_fn = _exact_label_match(gold_filler_pos)
    exact_sound_tp, exact_sound_fp, exact_sound_fn = _exact_label_match(gold_sound_pos)

    # Per-label breakdown
    def _per_label_metrics(gpos: List[int], kind: str) -> Dict[str, Dict]:
        gold_by_label: Dict[str, List[int]] = {}
        for g in gpos:
            label = get_event_label(gold_tokens[g], language=language)
            if label:
                gold_by_label.setdefault(label, []).append(g)
        pred_by_label: Dict[str, List[int]] = {}
        for p, tok in enumerate(pred_tokens):
            label = get_event_label(tok, language=language)
            if not label:
                continue
            if kind == "filler" and is_filler(tok, language=language):
                pred_by_label.setdefault(label, []).append(p)
            elif kind == "sound" and is_sound_event(tok):
                pred_by_label.setdefault(label, []).append(p)
        all_labels = sorted(set(gold_by_label) | set(pred_by_label))
        result: Dict[str, Dict] = {}
        for label in all_labels:
            gl = set(gold_by_label.get(label, []))
            pl = pred_by_label.get(label, [])
            pig: Set[int] = set()
            unaligned = 0
            for p in pl:
                if p in pred_to_gold:
                    pig.add(pred_to_gold[p])
                else:
                    unaligned += 1
            tp = len(gl & pig)
            fp = len(pig - gl) + unaligned
            fn = len(gl - pig)
            result[label] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": _f1(tp, fp, fn),
                "support": len(gl),
            }
        return result

    filler_by_label = _per_label_metrics(gold_filler_pos, "filler")
    sound_by_label = _per_label_metrics(gold_sound_pos, "sound")

    # ── Tag confusion matrix ──────────────────────────────────────────────
    pred_rep_set = set(pred_rep_pos)
    pred_tags_inferred: List[str] = []
    for i, tok in enumerate(pred_tokens):
        if is_cutoff(tok):
            pred_tags_inferred.append("CUTOFF")
        elif is_filler(tok, language=language):
            pred_tags_inferred.append("FILLER")
        elif is_sound_event(tok):
            pred_tags_inferred.append("SOUND")
        elif i in pred_rep_set:
            pred_tags_inferred.append("REP")
        else:
            pred_tags_inferred.append("FLUENT")

    tag_confusion: Dict[str, Dict[str, int]] = {}
    tag_del: Dict[str, int] = {}
    tag_ins: Dict[str, int] = {}
    for g_idx, p_idx in eval_alignment:
        if g_idx is None and p_idx is not None:
            pt = pred_tags_inferred[p_idx]
            tag_ins[pt] = tag_ins.get(pt, 0) + 1
        elif g_idx is not None and p_idx is None:
            gt = gold_tags[g_idx]
            tag_del[gt] = tag_del.get(gt, 0) + 1
        elif g_idx is not None and p_idx is not None:
            gt = gold_tags[g_idx]
            pt = pred_tags_inferred[p_idx]
            row = tag_confusion.setdefault(gt, {})
            row[pt] = row.get(pt, 0) + 1

    # ── Disfluency omission / hallucination ───────────────────────────────
    disfluency_types = {"FILLER", "SOUND", "CUTOFF", "REP"}
    gold_is_disf = [t in disfluency_types for t in gold_tags]
    pred_is_disf = [t in disfluency_types for t in pred_tags_inferred]

    disflu_omitted = sum(
        1 for g, d in enumerate(gold_is_disf) if d and gold_to_pred.get(g) is None
    )
    disflu_gold_total = sum(gold_is_disf)
    disflu_hallucinated = sum(
        1 for p, d in enumerate(pred_is_disf) if d and pred_to_gold.get(p) is None
    )
    disflu_pred_total = sum(pred_is_disf)

    # ── Repetition group counting ─────────────────────────────────────────
    pred_rep_groups = _count_pred_rep_groups(pred_tokens, pred_norm)
    gold_rep_groups = len({
        gid for gid, t in zip(gold_group_ids, gold_tags)
        if t == "REP" and gid is not None
    })
    rep_count_mae = abs(pred_rep_groups - gold_rep_groups)

    # ── Assemble results ──────────────────────────────────────────────────
    filler_p, filler_r, filler_f = _prf(filler_tp, filler_fp, filler_fn)
    sound_p, sound_r, sound_f = _prf(sound_tp, sound_fp, sound_fn)
    event_p, event_r, event_f = _prf(event_tp, event_fp, event_fn)
    cutoff_p, cutoff_r, cutoff_f = _prf(cutoff_tp, cutoff_fp, cutoff_fn)
    rep_p, rep_r, rep_f = _prf(rep_tp, rep_fp, rep_fn)
    cs_p, cs_r, cs_f = _prf(cutoff_strict_tp, cutoff_strict_fp, cutoff_strict_fn)
    cutoff_lenient_recall = (
        cutoff_lenient_tp / len(gold_cutoff_pos) if gold_cutoff_pos else 0.0
    )
    cl_p, cl_r, cl_f = _prf(cutoff_lenient_tp, cutoff_lenient_fp, cutoff_lenient_fn)
    disfluency_tp = filler_tp + sound_tp + cutoff_lenient_tp + rep_tp
    disfluency_fp = filler_fp + sound_fp + cutoff_lenient_fp + rep_fp
    disfluency_fn = filler_fn + sound_fn + cutoff_lenient_fn + rep_fn
    disfluency_p, disfluency_r, disfluency_f = _prf(
        disfluency_tp,
        disfluency_fp,
        disfluency_fn,
    )

    return {
        # String-level
        "vWER": vWER,
        "vCER": vCER,
        "iWER": iWER,
        "iCER": iCER,
        "iSR": iSR,
        "iDR": iDR,
        "iIR": iIR,
        "fluent_deletion_rate": fluent_deletion_rate,
        # Event-level
        "disfluency_f1": disfluency_f,
        "disfluency_precision": disfluency_p,
        "disfluency_recall": disfluency_r,
        "filler_f1": filler_f, "filler_precision": filler_p, "filler_recall": filler_r,
        "sound_f1": sound_f, "sound_precision": sound_p, "sound_recall": sound_r,
        "event_f1": event_f, "event_precision": event_p, "event_recall": event_r,
        "cutoff_f1": cutoff_f, "cutoff_precision": cutoff_p, "cutoff_recall": cutoff_r,
        "cutoff_strict_f1": cs_f, "cutoff_strict_precision": cs_p, "cutoff_strict_recall": cs_r,
        "cutoff_lenient_recall": cutoff_lenient_recall,
        "cutoff_lenient_f1": cl_f,
        "cutoff_lenient_precision": cl_p,
        "rep_f1": rep_f, "rep_precision": rep_p, "rep_recall": rep_r,
        # Support counts
        "filler_support": len(gold_filler_pos),
        "sound_support": len(gold_sound_pos),
        "disfluency_support": (
            len(gold_filler_pos)
            + len(gold_sound_pos)
            + len(gold_cutoff_pos)
            + len(gold_rep_pos)
        ),
        "event_support": len(gold_event_pos),
        "cutoff_support": len(gold_cutoff_pos),
        "rep_support": len(gold_rep_pos),
        # Exact label matching
        "exact_filler_f1": _f1(exact_filler_tp, exact_filler_fp, exact_filler_fn),
        "exact_sound_f1": _f1(exact_sound_tp, exact_sound_fp, exact_sound_fn),
        # Per-label
        "filler_by_label": filler_by_label,
        "sound_by_label": sound_by_label,
        # Repetition counting
        "rep_count_mae": rep_count_mae,
        "rep_count_within1": 1.0 if rep_count_mae <= 1 else 0.0,
        "rep_count_exact": 1.0 if rep_count_mae == 0 else 0.0,
        "rep_group_count_gold": gold_rep_groups,
        "rep_group_count_pred": pred_rep_groups,
        # Diagnostics
        "tag_confusion": tag_confusion,
        "tag_del": tag_del,
        "tag_ins": tag_ins,
        # Raw counts for corpus aggregation
        "_counts": {
            "filler": {"tp": filler_tp, "fp": filler_fp, "fn": filler_fn},
            "sound": {"tp": sound_tp, "fp": sound_fp, "fn": sound_fn},
            "event": {"tp": event_tp, "fp": event_fp, "fn": event_fn},
            "cutoff": {"tp": cutoff_tp, "fp": cutoff_fp, "fn": cutoff_fn},
            "cutoff_strict": {"tp": cutoff_strict_tp, "fp": cutoff_strict_fp, "fn": cutoff_strict_fn},
            "cutoff_lenient": {"tp": cutoff_lenient_tp, "fp": cutoff_lenient_fp, "fn": cutoff_lenient_fn},
            "rep": {"tp": rep_tp, "fp": rep_fp, "fn": rep_fn},
            "exact_filler": {"tp": exact_filler_tp, "fp": exact_filler_fp, "fn": exact_filler_fn},
            "exact_sound": {"tp": exact_sound_tp, "fp": exact_sound_fp, "fn": exact_sound_fn},
            "disfluency": {
                "tp": disfluency_tp,
                "fp": disfluency_fp,
                "fn": disfluency_fn,
            },
        },
        "_counts_a": {
            "vWER_edits": v_edits, "vWER_ref": v_ref,
            "vCER_edits": v_c_edits, "vCER_ref": v_c_ref,
            "iWER_edits": i_edits, "iWER_ref": i_ref,
            "iWER_subs": i_details["substitutions"],
            "iWER_dels": i_details["deletions"],
            "iWER_ins": i_details["insertions"],
            "iCER_edits": i_c_edits, "iCER_ref": i_c_ref,
            "fluent_matched": fluent_matched, "fluent_ref": fluent_ref,
        },
        "_disflu_binary": {
            "omitted": disflu_omitted,
            "gold_total": disflu_gold_total,
            "hallucinated": disflu_hallucinated,
            "pred_total": disflu_pred_total,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus-Level Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(sample_metrics: List[Dict]) -> Dict:
    """Micro-average all sample-level metrics into corpus-level results."""
    if not sample_metrics:
        return {"num_samples": 0}

    # ── WER / CER / Fluent deletion ───────────────────────────────────────
    wer_keys = [
        "vWER_edits", "vWER_ref", "vCER_edits", "vCER_ref",
        "iWER_edits", "iWER_ref", "iWER_subs", "iWER_dels", "iWER_ins",
        "iCER_edits", "iCER_ref", "fluent_matched", "fluent_ref",
    ]
    wer_counts: Dict[str, int] = {k: 0 for k in wer_keys}
    for m in sample_metrics:
        ca = m.get("_counts_a") or {}
        for k in wer_keys:
            wer_counts[k] += int(ca.get(k, 0))

    def _rate(e: int, r: int) -> Optional[float]:
        return (e / r) if r > 0 else None

    agg: Dict[str, Any] = {}
    agg["vWER"] = _rate(wer_counts["vWER_edits"], wer_counts["vWER_ref"])
    agg["vCER"] = _rate(wer_counts["vCER_edits"], wer_counts["vCER_ref"])
    agg["iWER"] = _rate(wer_counts["iWER_edits"], wer_counts["iWER_ref"])
    agg["iCER"] = _rate(wer_counts["iCER_edits"], wer_counts["iCER_ref"])
    agg["iSR"] = _rate(wer_counts["iWER_subs"], wer_counts["iWER_ref"])
    agg["iDR"] = _rate(wer_counts["iWER_dels"], wer_counts["iWER_ref"])
    agg["iIR"] = _rate(wer_counts["iWER_ins"], wer_counts["iWER_ref"])
    agg["fluent_deletion_rate"] = (
        1.0 - wer_counts["fluent_matched"] / wer_counts["fluent_ref"]
        if wer_counts["fluent_ref"] > 0 else None
    )

    for k in ["vWER", "vCER", "iWER", "iCER", "iSR", "iDR", "iIR", "fluent_deletion_rate"]:
        if agg.get(k) is not None:
            agg[f"{k}_percent"] = agg[k] * 100

    # ── Event F1 (micro) ─────────────────────────────────────────────────
    event_keys = [
        "disfluency", "filler", "sound", "event", "cutoff", "cutoff_strict",
        "cutoff_lenient", "rep", "exact_filler", "exact_sound",
    ]
    totals: Dict[str, Dict[str, int]] = {k: {"tp": 0, "fp": 0, "fn": 0} for k in event_keys}
    support_keys = ["disfluency", "filler", "sound", "event", "cutoff", "rep"]
    total_support: Dict[str, int] = {k: 0 for k in support_keys}

    label_totals: Dict[str, Dict[str, Dict[str, int]]] = {"filler": {}, "sound": {}}

    for m in sample_metrics:
        counts = m.get("_counts") or {}
        for key in event_keys:
            if key in counts:
                for field in ("tp", "fp", "fn"):
                    totals[key][field] += counts[key].get(field, 0)
        for key in support_keys:
            total_support[key] += m.get(f"{key}_support", 0)
        for kind in ("filler", "sound"):
            by_label = m.get(f"{kind}_by_label") or {}
            for label, data in by_label.items():
                if label not in label_totals[kind]:
                    label_totals[kind][label] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
                for field in ("tp", "fp", "fn", "support"):
                    label_totals[kind][label][field] += data.get(field, 0)

    for key in ["disfluency", "filler", "sound", "event", "cutoff", "rep"]:
        t = totals[key]
        p, r, f = _prf(t["tp"], t["fp"], t["fn"])
        agg[f"{key}_f1"] = f
        agg[f"{key}_precision"] = p
        agg[f"{key}_recall"] = r
        agg[f"{key}_support"] = total_support[key]

    cs = totals["cutoff_strict"]
    csp, csr, csf = _prf(cs["tp"], cs["fp"], cs["fn"])
    agg["cutoff_strict_f1"] = csf
    agg["cutoff_strict_precision"] = csp
    agg["cutoff_strict_recall"] = csr

    cl = totals["cutoff_lenient"]
    agg["cutoff_lenient_recall"] = cl["tp"] / max(1, cl["tp"] + cl["fn"])
    clp, clr, clf = _prf(cl["tp"], cl["fp"], cl["fn"])
    agg["cutoff_lenient_precision"] = clp
    agg["cutoff_lenient_f1"] = clf

    for key in ("exact_filler", "exact_sound"):
        t = totals[key]
        agg[f"{key}_f1"] = _f1(t["tp"], t["fp"], t["fn"])

    for kind in ("filler", "sound"):
        agg[f"{kind}_by_label"] = {}
        for label, data in label_totals[kind].items():
            p, r, f = _prf(data["tp"], data["fp"], data["fn"])
            agg[f"{kind}_by_label"][label] = {
                "precision": p, "recall": r, "f1": f, "support": data["support"],
            }

    # ── Repetition counting (utterance-level averages) ────────────────────
    for key in ("rep_count_mae", "rep_count_within1", "rep_count_exact"):
        vals = [m.get(key, 0.0) for m in sample_metrics]
        agg[key] = sum(vals) / len(vals) if vals else 0.0
    agg["rep_group_count_gold_total"] = sum(int(m.get("rep_group_count_gold", 0)) for m in sample_metrics)
    agg["rep_group_count_pred_total"] = sum(int(m.get("rep_group_count_pred", 0)) for m in sample_metrics)

    # ── Tag confusion ─────────────────────────────────────────────────────
    tag_confusion: Dict[str, Dict[str, int]] = {}
    tag_del: Dict[str, int] = {}
    tag_ins: Dict[str, int] = {}
    for m in sample_metrics:
        for gt, row in (m.get("tag_confusion") or {}).items():
            dst = tag_confusion.setdefault(gt, {})
            for pt, n in (row or {}).items():
                dst[pt] = dst.get(pt, 0) + int(n)
        for gt, n in (m.get("tag_del") or {}).items():
            tag_del[gt] = tag_del.get(gt, 0) + int(n)
        for pt, n in (m.get("tag_ins") or {}).items():
            tag_ins[pt] = tag_ins.get(pt, 0) + int(n)

    agg["tag_confusion"] = tag_confusion
    agg["tag_del"] = tag_del
    agg["tag_ins"] = tag_ins

    # Tag-level PRF
    all_tags = sorted(
        set(tag_confusion) | set(tag_del) | set(tag_ins)
        | {"FLUENT", "OTHER", "FILLER", "SOUND", "CUTOFF", "REP"}
    )
    row_sums = {
        gt: sum((tag_confusion.get(gt) or {}).values()) + int(tag_del.get(gt, 0))
        for gt in all_tags
    }
    col_sums = {
        pt: sum(int((tag_confusion.get(gt) or {}).get(pt, 0)) for gt in all_tags) + int(tag_ins.get(pt, 0))
        for pt in all_tags
    }
    tag_prf: Dict[str, Dict] = {}
    for tag in all_tags:
        tp = int((tag_confusion.get(tag) or {}).get(tag, 0))
        support = int(row_sums.get(tag, 0))
        pred_n = int(col_sums.get(tag, 0))
        fn = max(support - tp, 0)
        fp = max(pred_n - tp, 0)
        p, r, f = _prf(tp, fp, fn)
        tag_prf[tag] = {
            "precision": p, "recall": r, "f1": f,
            "support": support, "predicted": pred_n, "tp": tp, "fp": fp, "fn": fn,
        }
    agg["tag_prf"] = tag_prf

    # ── Disfluency omission / hallucination ───────────────────────────────
    do_total = dg_total = dh_total = dp_total = 0
    for m in sample_metrics:
        b = m.get("_disflu_binary") or {}
        do_total += int(b.get("omitted", 0))
        dg_total += int(b.get("gold_total", 0))
        dh_total += int(b.get("hallucinated", 0))
        dp_total += int(b.get("pred_total", 0))
    agg["disfluency_omission_rate"] = (do_total / dg_total) if dg_total > 0 else None
    agg["disfluency_omission_support"] = dg_total
    agg["disfluency_hallucination_rate"] = (dh_total / dg_total) if dg_total > 0 else None
    agg["disfluency_hallucination_support"] = dg_total
    agg["num_samples"] = len(sample_metrics)

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: Dict, title: str = "Evaluation Results") -> str:
    """Format metrics as a human-readable summary and return it as a string."""
    lines: List[str] = []
    a = lines.append

    a("")
    a("=" * 80)
    a(f"  {title}")
    a("=" * 80)
    a("")

    # Transcript quality
    a("-- Transcript Quality -------------------------------------------------")
    for k, label in [
        ("vWER", "vWER  (verbatim)"),
        ("vCER", "vCER"),
        ("iWER", "iWER  (intended)"),
        ("iCER", "iCER"),
        ("fluent_deletion_rate", "Fluent deletion rate"),
    ]:
        v = metrics.get(k)
        if v is not None:
            a(f"  {label:30s} {v:.2%}")
    a("")

    # DIER
    if any(metrics.get(k) is not None for k in ("iSR", "iDR", "iIR")):
        a("-- DIER (iSR + iDR + iIR = iWER) -------------------------------------")
        for k, label in [("iSR", "Substitution rate"), ("iDR", "Deletion rate"), ("iIR", "Insertion rate")]:
            v = metrics.get(k)
            if v is not None:
                a(f"  {label:30s} {v:.2%}")
        a("")

    # Event detection
    a("-- Event Detection (alignment-based) ----------------------------------")
    for kind in ("disfluency", "filler", "sound", "event", "cutoff", "rep"):
        f1v = metrics.get(f"{kind}_f1")
        pv = metrics.get(f"{kind}_precision", 0)
        rv = metrics.get(f"{kind}_recall", 0)
        sup = metrics.get(f"{kind}_support", "?")
        if f1v is not None:
            a(f"  {kind:12s} F1={f1v:.2%}  P={pv:.2%}  R={rv:.2%}  (n={sup})")
    csf = metrics.get("cutoff_strict_f1")
    if csf is not None:
        a(f"  {'cutoff_strict':12s} F1={csf:.2%}")
    clr = metrics.get("cutoff_lenient_recall")
    clf = metrics.get("cutoff_lenient_f1")
    if clf is not None:
        a(
            f"  {'cutoff_lenient':12s} F1={clf:.2%}  "
            f"P={metrics.get('cutoff_lenient_precision', 0):.2%}  R={clr or 0:.2%}"
        )
    elif clr is not None:
        a(f"  {'cutoff_lenient':12s} R={clr:.2%}")
    a("")

    # Exact label
    a("-- Exact Label Match --------------------------------------------------")
    for k, label in [("exact_filler_f1", "Exact Filler F1"), ("exact_sound_f1", "Exact Sound F1")]:
        v = metrics.get(k)
        if v is not None:
            a(f"  {label:30s} {v:.2%}")
    a("")

    # Per-label top
    for kind in ("filler", "sound"):
        by_label = metrics.get(f"{kind}_by_label") or {}
        items = sorted(
            [(d.get("support", 0), lab, d.get("f1", 0)) for lab, d in by_label.items() if d.get("support", 0) > 0],
            reverse=True,
        )[:8]
        if items:
            a(f"-- Per-Label: {kind} (top by support) --------------------------------")
            for sup, lab, f1v in items:
                a(f"  [{lab}]  F1={f1v:.2%}  (n={sup})")
            a("")

    # Disfluency rates
    a("-- Disfluency Rates ---------------------------------------------------")
    dor = metrics.get("disfluency_omission_rate")
    if dor is not None:
        a(f"  Omission rate:      {dor:.2%}  (n={metrics.get('disfluency_omission_support', '?')})")
    dhr = metrics.get("disfluency_hallucination_rate")
    if dhr is not None:
        a(f"  Hallucination rate: {dhr:.2%}  (n={metrics.get('disfluency_hallucination_support', '?')})")
    a("")

    # Repetition counting
    a("-- Repetition Counting ------------------------------------------------")
    for k, label in [
        ("rep_count_mae", "Count MAE"),
        ("rep_count_exact", "Count Exact"),
        ("rep_count_within1", "Count Within +/-1"),
    ]:
        v = metrics.get(k)
        if v is not None:
            if "mae" in k.lower():
                a(f"  {label:30s} {v:.2f}")
            else:
                a(f"  {label:30s} {v:.2%}")
    a("")

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Loading & Evaluation Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(path: str) -> List[Dict]:
    preds: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def load_gold_from_predictions(
    predictions: List[Dict],
    cutoff_source: str,
    cutoff_target: str,
) -> List[Dict]:
    """Build gold samples from self-contained prediction JSONL rows."""
    gold_samples: List[Dict] = []
    seen: Set[str] = set()
    for row in predictions:
        sid = str(row.get("id", ""))
        if not sid or sid in seen:
            continue
        verbatim = str(row.get("gold_verbatim", "") or row.get("verbatim_ground_truth", "") or "")
        intended = str(row.get("gold_intended", "") or row.get("intended_ground_truth", "") or "")
        if not verbatim or not intended:
            continue
        language = row.get("language") or _language_from_dataset(row.get("dataset", ""))
        gold = preprocess_gold(
            verbatim,
            intended,
            sid,
            language=language,
            cutoff_source=cutoff_source,
            cutoff_target=cutoff_target,
        )
        gold["dataset"] = row.get("dataset", "")
        gold["split"] = row.get("split", "")
        gold_samples.append(gold)
        seen.add(sid)
    return gold_samples


def load_gold_from_hf(
    dataset_name: str,
    splits: List[str],
    cutoff_source: str,
    cutoff_target: str,
    verbatim_key: str = "verbatim_transcript",
    intended_key: str = "intended_transcript",
) -> List[Dict]:
    """Load a HuggingFace dataset and preprocess gold labels on-the-fly.

    Supports both HuggingFace Hub names and local ``load_from_disk`` paths.
    """
    if os.path.isdir(dataset_name):
        from datasets import load_from_disk
        ds_dict = load_from_disk(dataset_name)
    else:
        from datasets import load_dataset
        ds_dict = {s: load_dataset(dataset_name, split=s) for s in splits}

    gold_samples: List[Dict] = []
    for split in splits:
        ds = ds_dict[split] if isinstance(ds_dict, dict) else ds_dict[split]
        non_audio_cols = [c for c in ds.column_names if c != "audio"]
        for row in ds.select_columns(non_audio_cols):
            sid = str(row.get("id", row.get("ID", "")))
            verbatim = str(row.get(verbatim_key, "") or "")
            intended = str(row.get(intended_key, "") or "")
            if not verbatim or not intended:
                continue
            gold = preprocess_gold(
                verbatim,
                intended,
                sid,
                language=_language_from_dataset(dataset_name),
                cutoff_source=cutoff_source,
                cutoff_target=cutoff_target,
            )
            gold["dataset"] = dataset_name
            gold["split"] = split
            gold_samples.append(gold)
    return gold_samples


def evaluate_corpus(
    gold_samples: List[Dict],
    predictions: List[Dict],
) -> Dict:
    """Evaluate all predictions against gold and return aggregated metrics."""
    pred_lookup = {str(p["id"]): p for p in predictions}

    all_metrics: List[Dict] = []
    missing = 0
    for gold in gold_samples:
        sid = str(gold["id"])
        pred = pred_lookup.get(sid)
        if pred is None:
            missing += 1
            continue
        m = evaluate_sample(
            gold,
            pred.get("pred_verbatim", ""),
            pred.get("pred_intended"),
        )
        all_metrics.append(m)

    if missing:
        print(f"Warning: {missing} gold samples had no matching prediction.")

    return aggregate_metrics(all_metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate verbatim ASR predictions against HuggingFace gold datasets."
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to normalized prediction JSONL file.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[],
        help="HuggingFace dataset name(s), e.g. nyrahealth/disfluency_speech_english",
    )
    parser.add_argument("--splits", nargs="+", default=["test"], help="Dataset splits to evaluate.")
    parser.add_argument("--cutoff_source", default="*", help="Cutoff marker in dataset (default: *).")
    parser.add_argument("--cutoff_target", default="-", help="Cutoff marker in model output (default: -).")
    parser.add_argument("--verbatim_key", default="verbatim_transcript")
    parser.add_argument("--intended_key", default="intended_transcript")
    parser.add_argument(
        "--use_prediction_gold",
        action="store_true",
        help="Use gold_verbatim/gold_intended embedded in the predictions JSONL.",
    )
    parser.add_argument("--output_dir", default="results/", help="Directory for report output.")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    all_gold: List[Dict] = []
    if args.use_prediction_gold:
        print("Loading gold from prediction JSONL rows...")
        all_gold = load_gold_from_predictions(
            predictions,
            args.cutoff_source,
            args.cutoff_target,
        )
        print(f"  {len(all_gold)} samples loaded.")
    else:
        if not args.datasets:
            parser.error("--datasets is required unless --use_prediction_gold is set")
        for ds_name in args.datasets:
            print(f"Loading gold from {ds_name} (splits={args.splits})...")
            gold = load_gold_from_hf(
                ds_name, args.splits,
                args.cutoff_source, args.cutoff_target,
                args.verbatim_key, args.intended_key,
            )
            all_gold.extend(gold)
            print(f"  {len(gold)} samples loaded.")

    print(f"\nTotal gold samples: {len(all_gold)}")
    print("Evaluating...")

    metrics = evaluate_corpus(all_gold, predictions)

    if args.use_prediction_gold:
        title = f"Evaluation: cached predictions ({os.path.basename(args.predictions)})"
    else:
        title = f"Evaluation: {', '.join(args.datasets)} ({'+'.join(args.splits)})"
    summary_text = print_report(metrics, title)

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": args.datasets,
        "splits": args.splits,
        "predictions_file": args.predictions,
        "cutoff_source": args.cutoff_source,
        "cutoff_target": args.cutoff_target,
        "metrics": metrics,
    }
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote report to: {report_path}")

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# {title}\n\n```\n{summary_text}\n```\n")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
