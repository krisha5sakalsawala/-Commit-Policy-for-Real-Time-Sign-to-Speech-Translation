"""
======================================================
Commit Policy (Shared for Offline + Streaming)
======================================================
This module implements the exact commit policy logic
used for both offline evaluation and continuous streaming.

Parameters:
    THRESH        - confidence threshold
    WINDOW        - number of stable frames required
    LOOKAHEAD     - small delay to confirm stability
    MARGIN_DELTA  - top-2 probability difference
    ENTROPY_MAX   - max allowed uncertainty
    CLEAR_AFTER_COMMIT - whether to clear window after commit
    USE_COOLDOWN  - cooldown period between commits
======================================================
"""

import numpy as np
from scipy.stats import entropy

def commit_decisions(
    probs,
    blank_id,
    THRESH=0.45,
    WINDOW=2,
    LOOKAHEAD=1,
    MARGIN_DELTA=0.01,
    ENTROPY_MAX=2.0,
    CLEAR_AFTER_COMMIT=True,
    USE_COOLDOWN=False,
    COOLDOWN_SEC=0.0,
    frame_time=0.0  # optional (seconds per frame) for cooldown
):
    """
    Apply commit policy on logits probabilities.
    Returns list of (frame_idx, token_id, confidence, entropy)
    """
    commits = []
    window_conf = []
    prev_token = None
    last_commit_time = -9999.0

    T = probs.shape[0]

    for t in range(T):
        # Average over lookahead
        end_t = min(T, t + LOOKAHEAD + 1)
        avg_prob = np.mean(probs[t:end_t, :], axis=0)
        token_id = int(np.argmax(avg_prob))
        p_sorted = np.sort(avg_prob)
        p1, p2 = float(p_sorted[-1]), float(p_sorted[-2])
        ent = float(entropy(avg_prob))
        conf = p1

        # Update stability window
        window_conf.append((token_id, conf))
        if len(window_conf) > WINDOW:
            window_conf.pop(0)

        tokens = [tok for tok, _ in window_conf]
        confs = [c for _, c in window_conf]

        stable = (len(window_conf) == WINDOW) and (len(set(tokens)) == 1)
        margin_ok = (p1 - p2) >= MARGIN_DELTA
        conf_ok = np.mean(confs) >= THRESH
        ent_ok = ent <= ENTROPY_MAX
        blank_ok = token_id != blank_id
        dup_ok = token_id != prev_token

        # optional cooldown
        cooldown_ok = True
        if USE_COOLDOWN:
            cooldown_ok = (t * frame_time - last_commit_time) >= COOLDOWN_SEC

        if stable and conf_ok and margin_ok and ent_ok and blank_ok and dup_ok and cooldown_ok:
            commits.append((t, token_id, np.mean(confs), ent))
            prev_token = token_id
            last_commit_time = t * frame_time

            if CLEAR_AFTER_COMMIT:
                window_conf.clear()

    return commits
