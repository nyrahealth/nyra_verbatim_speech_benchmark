"""
Edit-distance alignment for verbatim ASR evaluation.

Provides a single :func:`align_sequences` function that computes a
minimum-edit alignment between two token sequences.  When optional ``tags``
are supplied the costs are adjusted with a small epsilon so that ties are
broken in favour of matching FLUENT tokens and deleting disfluent ones
(per Jamshid Lou & Johnson, 2020).

Output is a list of ``(a_idx, b_idx)`` pairs where either side may be
``None``:

* ``(a_idx, b_idx)`` — tokens align (match or substitution)
* ``(a_idx, None)``  — extra token in sequence A (deletion from B's view)
* ``(None, b_idx)``  — extra token in sequence B (insertion from A's view)
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

DISFLUENT_TAGS: Set[str] = {"REP", "FILLER", "SOUND", "CUTOFF"}


def align_sequences(
    seq_a_tokens: List[str],
    seq_b_tokens: List[str],
    seq_a_norm: List[str],
    seq_b_norm: List[str],
    tags: Optional[List[str]] = None,
    eps: float = 1e-7,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Minimum-edit alignment on normalised token sequences.

    Args:
        seq_a_tokens: Raw tokens of sequence A.
        seq_b_tokens: Raw tokens of sequence B.
        seq_a_norm:   Normalised forms of A (used for equality comparison).
        seq_b_norm:   Normalised forms of B.
        tags:         Optional per-token tags for A.  When provided, epsilon
                      biasing is applied: disfluent tokens are cheaper to
                      delete and slightly more expensive to match, so the
                      aligner prefers matching FLUENT tokens when tied.
        eps:          Epsilon magnitude for tie-breaking (default 1e-7).

    Returns:
        Ordered list of ``(a_idx | None, b_idx | None)`` pairs.
    """
    n = len(seq_a_norm)
    m = len(seq_b_norm)
    use_tags = tags is not None and len(tags) == n

    def _is_disfluent(i: int) -> bool:
        return use_tags and tags[i] in DISFLUENT_TAGS  # type: ignore[index]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        del_cost = (1.0 - eps) if _is_disfluent(i - 1) else 1.0
        dp[i][0] = dp[i - 1][0] + del_cost
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1.0
        back[0][j] = "I"

    for i in range(1, n + 1):
        ai = seq_a_norm[i - 1]
        disfluent = _is_disfluent(i - 1)

        for j in range(1, m + 1):
            bj = seq_b_norm[j - 1]
            is_match = ai == bj

            del_cost = (1.0 - eps) if disfluent else 1.0
            d = dp[i - 1][j] + del_cost

            ins_cost = (1.0 + eps) if disfluent else 1.0
            ins = dp[i][j - 1] + ins_cost

            if is_match:
                match_cost = eps if disfluent else 0.0
                sub = dp[i - 1][j - 1] + match_cost
            else:
                sub_cost = (1.0 + eps) if disfluent else 1.0
                sub = dp[i - 1][j - 1] + sub_cost

            best = sub
            op = "M" if is_match else "S"
            if d < best:
                best = d
                op = "D"
            if ins < best:
                best = ins
                op = "I"

            dp[i][j] = best
            back[i][j] = op

    alignment: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            alignment.append((i - 1, None))
            i -= 1
        elif op == "I":
            alignment.append((None, j - 1))
            j -= 1
        else:
            if i > 0:
                alignment.append((i - 1, None))
                i -= 1
            elif j > 0:
                alignment.append((None, j - 1))
                j -= 1

    alignment.reverse()
    return alignment
