"""
======================================================
TEXT-TO-SPEECH LATENCY + STREAMING SUPPORT
======================================================
Supports two modes:
  • Baseline Latency Evaluation (default)
  • Real-Time Streaming Mode (direct call with output path)

Features:
  • Commit Latency, Retract Rate, FPS metrics
  • Smart save logic (prevents overwriting baseline files)

"""

import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from gtts import gTTS

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
BASE_DIR = r"D:\Graduate Project\F"
PRED_CSV = os.path.join(BASE_DIR, "g2t_out","phoenix_test_predicted_text.csv")
OUT_AUDIO_DIR = os.path.join(BASE_DIR, "results", "tts_outputs")
RESULTS_CSV = os.path.join(BASE_DIR, "results", "tts_latency_metrics.csv")
# NEW: per-sample latency CSV (append mode)
PER_SAMPLE_CSV = os.path.join(BASE_DIR, "results", "tts_latency_samples.csv")
os.makedirs(os.path.dirname(PER_SAMPLE_CSV), exist_ok=True)
os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
# If file does not exist, create with headers
if not os.path.exists(PER_SAMPLE_CSV):
    pd.DataFrame(columns=["sample_index", "text", "latency"]).to_csv(PER_SAMPLE_CSV, index=False)


# ----------------------------------------------------
# UNIVERSAL TEXT-TO-SPEECH FUNCTION
# ----------------------------------------------------
def text_to_speech(text: str, output_name):
    
    start = time.time()
    try:
        if not text.strip():
            print("⚠️ Empty text, skipping TTS.")
            return None, None

        # Detect absolute/custom path (streaming)
        if isinstance(output_name, str) and (
            os.path.isabs(output_name) or "\\" in output_name or "/" in output_name
        ):
            out_base = output_name
        else:
            out_base = os.path.join(OUT_AUDIO_DIR, f"tts_output_{output_name}")

        os.makedirs(os.path.dirname(out_base), exist_ok=True)

        # Generate and save TTS
        tts = gTTS(text)
        out_path = f"{out_base}.mp3"
        tts.save(out_path)

        end = time.time()
        latency = end - start
        print(f"🎤 Saved speech: {out_path}  |  Latency: {latency:.3f}s")
        return latency, out_path

    except Exception as e:
        print(f"⚠️ Error generating TTS for {output_name}: {e}")
        return None, None


# ----------------------------------------------------
# METRIC COMPUTATION
# ----------------------------------------------------
def compute_metrics(latencies, predictions_over_time=None, fps_base=25):
    """
    Compute latency-related metrics.
    """
    metrics = {}
    latencies = [l for l in latencies if l is not None]

    if len(latencies) > 0:
        metrics["AvgCommitLatency(sec)"] = np.mean(latencies)
        metrics["MinLatency(sec)"] = np.min(latencies)
        metrics["MaxLatency(sec)"] = np.max(latencies)
    else:
        metrics["AvgCommitLatency(sec)"] = None

    """# Placeholder for retraction rate (for commit-policy analysis)
    if predictions_over_time:
        retractions = 0
        for i in range(1, len(predictions_over_time)):
            if not predictions_over_time[i].startswith(predictions_over_time[i - 1]):
                retractions += 1
        metrics["RetractRate"] = retractions / len(predictions_over_time)
    else:
        metrics["RetractRate"] = None"""

    total_audio = len(latencies)
    total_time = np.sum(latencies) if len(latencies) else 1
    metrics["FPS"] = total_audio / total_time
    metrics["NumSamples"] = len(latencies)
    metrics["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return metrics


# ----------------------------------------------------
# BASELINE LATENCY EVALUATION
# ----------------------------------------------------
if __name__ == "__main__":
    print("\n🎤 Starting Text-to-Speech latency evaluation...\n")

    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Prediction CSV not found: {PRED_CSV}")

    pred_df = pd.read_csv(PRED_CSV)

    # Detect text column (or fallback to gloss)
    text_col_candidates = [c for c in pred_df.columns if "text" in c.lower()]
    if not text_col_candidates:
        print("ℹ️ No 'text' column found. Using gloss column instead for TTS.")
        text_col = [c for c in pred_df.columns if "gloss" in c.lower()][0]
    else:
        text_col = text_col_candidates[0]

    sentences = pred_df[text_col].astype(str).tolist()

    # Generate TTS and measure latency (limit to 50 samples for speed)
    latencies = []
    for i, sentence in enumerate(sentences[:50]):
        print(f"[{i+1}/{len(sentences)}] Generating audio...")
        latency, path = text_to_speech(sentence, i)
        # NEW: Log per-sample latency
        if latency is not None:
            pd.DataFrame(
                [[i, sentence, latency]],
                columns=["sample_index", "text", "latency"]
            ).to_csv(PER_SAMPLE_CSV, mode='a', header=False, index=False)

        if latency is not None:
            latencies.append(latency)
        time.sleep(0.5)  # simulate streaming gap

    # Compute metrics
    results = compute_metrics(latencies)
    print("\n✅ Latency Evaluation Complete\n")
    for k, v in results.items():
        print(f"{k:25s}: {v}")

    # Save results
    df = pd.DataFrame([results])
    if os.path.exists(RESULTS_CSV):
        old = pd.read_csv(RESULTS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

    json_path = RESULTS_CSV.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {RESULTS_CSV}")
    print(f"📁 JSON summary: {json_path}")

