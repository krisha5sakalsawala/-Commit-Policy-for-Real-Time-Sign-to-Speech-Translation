"""
======================================================
Evaluate Advanced Commit Policy – True End-to-End Latency
======================================================
Aggregates:
 • mean / p50 / p95 end-to-end latency
 • SER (Sign Error Rate)
 • WER (if available)
for each threshold from the advanced commit policy.
======================================================

"""

import os
import glob
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
BASE_DIR = r"D:\Graduate Project\F"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "threshold")

# Files pattern
LAT_EVENT_PATTERN = os.path.join(RESULTS_DIR, "commit_events_adv_th*.csv")
GLOSS_PATTERN = os.path.join(RESULTS_DIR, "phoenix_test_predicted_gloss_commit_adv_th*.csv")
TEXT_CSV = os.path.join(RESULTS_DIR, "phoenix_test_predicted_text.csv")  
OUT_PATH = os.path.join(RESULTS_DIR, "true_e2e_latency_adv_summary.csv")

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------
def pct(vals, p):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.percentile(vals, p)) if len(vals) > 0 else None

def compute_edit_distance(a_tokens, b_tokens):
    seq = SequenceMatcher(None, a_tokens, b_tokens)
    return 1 - seq.ratio()

def compute_SER_WER(gloss_csv, text_csv):
    ser, wer = None, None
    # SER
    if os.path.exists(gloss_csv):
        df = pd.read_csv(gloss_csv)
        if {"predicted_gloss", "ref_gloss"}.issubset(df.columns):
            diffs = []
            for _, row in df.iterrows():
                p = str(row["predicted_gloss"]).split()
                r = str(row["ref_gloss"]).split()
                diffs.append(compute_edit_distance(p, r))
            ser = np.mean(diffs)
    # WER
    if os.path.exists(text_csv):
        df2 = pd.read_csv(text_csv)
        if {"pred_text", "ref_text"}.issubset(df2.columns):
            diffs = []
            for _, row in df2.iterrows():
                p = str(row["pred_text"]).split()
                r = str(row["ref_text"]).split()
                diffs.append(compute_edit_distance(p, r))
            wer = np.mean(diffs)
    return ser, wer

# ----------------------------------------------------
# MAIN EVALUATION
# ----------------------------------------------------
rows = []
for lat_path in sorted(glob.glob(LAT_EVENT_PATTERN)):
    th = float(lat_path.split("th")[-1].replace(".csv", ""))
    gloss_file = lat_path.replace("commit_events_adv", "phoenix_test_predicted_gloss_commit_adv")

    # --- Latency ---
    df_lat = pd.read_csv(lat_path)
    if len(df_lat) == 0:
        print(f"⚠️ No events found for θ={th}")
        continue
    e2e = df_lat["commit_time_sec"].tolist()
    mean_e2e, p50_e2e, p95_e2e = np.mean(e2e), pct(e2e, 50), pct(e2e, 95)

    # --- SER / WER ---
    ser, wer = compute_SER_WER(gloss_file, TEXT_CSV)
    ser_str = f"{ser:.3f}" if ser is not None else "N/A"
    wer_str = f"{wer:.3f}" if wer is not None else "N/A"
    print(f"θ={th} | mean_e2e={mean_e2e:.3f}s | SER={ser_str} | WER={wer_str}")

    rows.append({
        "Threshold": th,
        "mean_e2e": mean_e2e,
        "p50_e2e": p50_e2e,
        "p95_e2e": p95_e2e,
        "events": len(df_lat),
        "SER": ser,
        "WER": wer
    })

summary = pd.DataFrame(rows).sort_values("Threshold")
summary.to_csv(OUT_PATH, index=False)
print(f"\n✅ Summary saved to: {OUT_PATH}\n")
print(summary)

# ----------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------
if len(summary) > 0:
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    # SER/WER on left Y-axis
    ax1.plot(summary["Threshold"], summary["SER"], "o-", color="tab:red", label="SER")
    if summary["WER"].notna().any():
        ax1.plot(summary["Threshold"], summary["WER"], "s--", color="tab:orange", label="WER")

    # Latency on right Y-axis
    ax2.plot(summary["Threshold"], summary["p50_e2e"], "d-", color="tab:blue", label="Latency p50 (s)")

    ax1.set_xlabel("Commit Threshold (θ)")
    ax1.set_ylabel("Error Rate", color="tab:red")
    ax2.set_ylabel("Latency (s)", color="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Advanced Commit Policy – Latency vs Accuracy Trade-Off")
    plt.tight_layout()
    plt.show()

print("\n🎯 Evaluation complete with advanced commit policy metrics.")
