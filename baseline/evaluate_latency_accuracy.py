"""
======================================================
LATENCY–ACCURACY TRADEOFF ANALYSIS (FINAL VERSION)
======================================================
Uses per-sample latencies stored in:

    tts_latency_samples.csv

Expected columns:
    • sample_index
    • text
    • latency

Computes:
    • mean latency
    • p50 latency
    • p95 latency
    • SER (Sign Error Rate)
    • WER (Word Error Rate)

Outputs:
    latency_accuracy_summary.csv
======================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
BASE_DIR = r"D:\Graduate Project\sign_to_speech_project"

# Per-sample latency file created by text_to_speech.py
LAT_SAMPLE_CSV = os.path.join(BASE_DIR, "results", "tts_latency_samples.csv")

# Prediction files for SER / WER
GLOSS_CSV = os.path.join(BASE_DIR, "results", "phoenix_test_predicted_gloss.csv")
TEXT_CSV  = os.path.join(BASE_DIR, "g2t_out", "phoenix_test_predicted_text.csv")

# Output
OUT_SUMMARY = os.path.join(BASE_DIR, "results", "latency_accuracy_summary.csv")


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def compute_percentiles(latencies):
    """Compute mean/p50/p95 from list of floats."""
    lst = [x for x in latencies if x is not None]
    if len(lst) == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(lst)),
        "p50": float(np.percentile(lst, 50)),
        "p95": float(np.percentile(lst, 95)),
    }


def compute_edit_distance(a_tokens, b_tokens):
    seq = SequenceMatcher(None, a_tokens, b_tokens)
    return 1 - seq.ratio()


def compute_SER_WER(gloss_csv, text_csv):
    """Compute SER/WER from gloss and text CSV predictions."""
    ser, wer = None, None

    # --- SER ---
    if os.path.exists(gloss_csv):
        df = pd.read_csv(gloss_csv)
        if {"predicted_gloss", "ref_gloss"}.issubset(df.columns):
            diffs = []
            for _, row in df.iterrows():
                p = str(row["predicted_gloss"]).split()
                r = str(row["ref_gloss"]).split()
                diffs.append(compute_edit_distance(p, r))
            ser = float(np.mean(diffs))

    # --- WER ---
    if os.path.exists(text_csv):
        df = pd.read_csv(text_csv)
        if {"pred_text", "ref_text"}.issubset(df.columns):
            diffs = []
            for _, row in df.iterrows():
                p = str(row["pred_text"]).split()
                r = str(row["ref_text"]).split()
                diffs.append(compute_edit_distance(p, r))
            wer = float(np.mean(diffs))

    return ser, wer


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    print("\n📊 Evaluating Latency–Accuracy (Per-Sample TTS)…\n")

    # ------------------------------------------------
    # LOAD PER-SAMPLE LATENCIES
    # ------------------------------------------------
    if not os.path.exists(LAT_SAMPLE_CSV):
        raise FileNotFoundError(
            f"Missing per-sample latency file:\n{LAT_SAMPLE_CSV}\n"
            f"Run updated text_to_speech.py to generate it."
        )

    df = pd.read_csv(LAT_SAMPLE_CSV)

    # Validate expected structure
    expected_cols = {"sample_index", "text", "latency"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain exactly these columns: {expected_cols}\n"
            f"Found: {df.columns.tolist()}"
        )

    # Convert latency column
    latencies = df["latency"].astype(float).dropna().tolist()

    print(f"Loaded {len(latencies)} latency samples.")
    stats = compute_percentiles(latencies)

    print(f"Mean latency: {stats['mean']:.4f}s")
    print(f"P50 latency : {stats['p50']:.4f}s")
    print(f"P95 latency : {stats['p95']:.4f}s\n")

    # ------------------------------------------------
    # COMPUTE SER / WER
    # ------------------------------------------------
    ser, wer = compute_SER_WER(GLOSS_CSV, TEXT_CSV)
    print(f"📈 SER: {ser:.4f}   |   WER: {wer:.4f}\n")

    # ------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------
    summary = pd.DataFrame([
        {
            "Label": "baseline",
            "Mean_Latency(s)": stats["mean"],
            "p50_Latency(s)": stats["p50"],
            "p95_Latency(s)": stats["p95"],
            "SER": ser,
            "WER": wer,
        }
    ])

    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"✅ Summary saved to:\n{OUT_SUMMARY}")
    print(summary)
