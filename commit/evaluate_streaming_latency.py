"""
======================================================
Evaluate Streaming Latency (Commit-only vs Full-Streaming)
======================================================
This script analyzes latency and accuracy for:
  • Commit-only (offline simulated streaming)
  • Full-streaming (sign→speech pipeline)

It computes:
  • Mean, p50, p95 end-to-end latency
  • Event count
  • SER / WER if available
  • Comparison with baseline

Usage:
    python evaluate_streaming_latency.py

"""

import os
import glob
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE_DIR = r"D:\Graduate Project\F"
RESULTS_STREAM = os.path.join(BASE_DIR, "results_streaming")
RESULTS_OFFLINE = os.path.join(BASE_DIR, "results")

# Auto-detect mode
if os.path.exists(os.path.join(RESULTS_STREAM, "tts_outputs_streaming")):
    MODE = "full-streaming"
    RESULTS_DIR = RESULTS_STREAM
else:
    MODE = "commit-only"
    RESULTS_DIR = RESULTS_OFFLINE

COMMIT_LOG = os.path.join(RESULTS_DIR, "commit_events_adv_th0.45.csv")
SPEECH_DIR = os.path.join(RESULTS_DIR, "tts_outputs_streaming" if MODE == "full-streaming" else "tts_outputs")
SER_PATH = os.path.join(BASE_DIR, "results", "latency_accuracy_summary.csv")
OUT_SUMMARY = os.path.join(BASE_DIR, "results", "true_e2e_latency_comparison.csv")

print(f"🔹 Evaluation Mode Detected: {MODE}")
print(f"🔹 Commit Log: {COMMIT_LOG}")
print(f"🔹 Speech Directory: {SPEECH_DIR}\n")


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def pct(values, p):
    """Safe percentile calculation."""
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, p)) if len(vals) > 0 else None


# ------------------------------------------------------------
# LOAD COMMIT EVENTS
# ------------------------------------------------------------
if not os.path.exists(COMMIT_LOG):
    raise FileNotFoundError(f"Commit events file not found: {COMMIT_LOG}")

df = pd.read_csv(COMMIT_LOG)
if "threshold" in df.columns:
    thresholds = sorted(df["threshold"].unique())
else:
    thresholds = [0.45]  # fallback

rows = []

# ------------------------------------------------------------
# COMPUTE LATENCIES PER THRESHOLD
# ------------------------------------------------------------
for th in thresholds:
    df_th = df[df["threshold"] == th] if "threshold" in df.columns else df
    commit_times = df_th["commit_time_sec"].tolist()

    # Derive speech latency from TTS filenames (e.g., *_58ms.wav.mp3)
    speech_files = glob.glob(os.path.join(SPEECH_DIR, "*.mp3"))
    if speech_files:
        speech_times = []
        for f in speech_files:
            base = os.path.basename(f)
            if "_" in base and "ms" in base:
                try:
                    t_ms = base.split("_")[-1].replace("ms.wav.mp3", "").replace("ms.mp3", "")
                    speech_times.append(float(t_ms) / 1000.0)
                except:
                    continue

        mean_e2e = np.mean(speech_times) if speech_times else np.mean(commit_times)
        p50_e2e = pct(speech_times, 50) if speech_times else pct(commit_times, 50)
        p95_e2e = pct(speech_times, 95) if speech_times else pct(commit_times, 95)
        event_count = len(speech_times)
    else:
        mean_e2e = np.mean(commit_times)
        p50_e2e = pct(commit_times, 50)
        p95_e2e = pct(commit_times, 95)
        event_count = len(df_th)

    # SER / WER if columns exist
    ser_val = df_th["SER"].iloc[0] if "SER" in df_th.columns else None
    wer_val = df_th["WER"].iloc[0] if "WER" in df_th.columns else None

    rows.append({
        "Mode": MODE,
        "Threshold": th,
        "mean_e2e": mean_e2e,
        "p50_e2e": p50_e2e,
        "p95_e2e": p95_e2e,
        "events": event_count
    })

df_out = pd.DataFrame(rows)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------
if os.path.exists(OUT_SUMMARY):
    prev = pd.read_csv(OUT_SUMMARY)
    df_out = pd.concat([prev, df_out], ignore_index=True)

df_out.to_csv(OUT_SUMMARY, index=False)
print(f"\n✅ Results appended to: {OUT_SUMMARY}")
print(df_out)

# ------------------------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------------------------
print("\n📈 Aggregate Statistics:")
print(f"  Mean E2E Latency : {df_out['mean_e2e'].mean():.3f} s")
print(f"  Median Latency   : {df_out['p50_e2e'].mean():.3f} s")
print(f"  95th Percentile  : {df_out['p95_e2e'].mean():.3f} s")
print(f"  Total Events     : {df_out['events'].iloc[-1]}")

if "SER" in df_out.columns and df_out["SER"].notna().any():
    print(f"  Average SER      : {df_out['SER'].dropna().mean():.3f}")
