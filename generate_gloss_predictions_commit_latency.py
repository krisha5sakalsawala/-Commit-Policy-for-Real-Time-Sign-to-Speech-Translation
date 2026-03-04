"""
======================================================
Generate Gloss Predictions with Soft Commit Policy + Latency
======================================================
This version is tuned for models with moderate confidence:
 • Lower threshold range (0.35–0.45)
 • Shorter stability window (3)
 • Minimal lookahead (1)
 • Relaxed margin & entropy filters
 • Logs commit policy parameters + latencies
======================================================

"""

import os, sys
sys.path.append(r"D:\Graduate Project\F\baseline") 
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from cslr_train_ctc import BiLSTM_CTC
from scipy.stats import entropy

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
CHECKPOINT_PATH = r"baseline\cslr_ctc_best.pt"
LANDMARKS_DIR = r"D:\Graduate Project\F\data\landmarks\test"
OUTPUT_DIR = r"D:\Graduate Project\F\results\threshold"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Commit Policy Parameters (softened) ---
WINDOW = 2               # shorter stability requirement
LOOKAHEAD = 1            # minimal delay (~20–40 ms)
MARGIN_DELTA = 0.01      # small gap between top-2 probs
ENTROPY_MAX = 2.0        # allow more uncertainty
THRESHOLDS = [ 0.05, 0.10, 0.15, 0.20,0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]  # softer confidence levels

# ----------------------------------------------------
# LOAD MODEL + VOCAB
# ----------------------------------------------------
print("🔹 Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
vocab_dict = checkpoint["vocab"]

vocab = type("Vocab", (), {})()
vocab.stoi = vocab_dict
vocab.itos = {v: k for k, v in vocab_dict.items()}
vocab.blank = "<blank>"
vocab.pad = "<pad>"

meta_csv = os.path.join(LANDMARKS_DIR, "test_metadata.csv")
df = pd.read_csv(meta_csv)
feat_dim = np.load(df.iloc[0]["npy_path"]).shape[1]

model = BiLSTM_CTC(in_dim=feat_dim, hidden=256, vocab_size=len(vocab.stoi))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()
blank_id = vocab.stoi[vocab.blank]
print("✅ Model ready for inference.\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# INFERENCE LOOP
# ----------------------------------------------------
for THRESH in THRESHOLDS:
    print(f"🚀 Running with COMMIT_THRESHOLD={THRESH}, WINDOW={WINDOW}, "
          f"LOOKAHEAD={LOOKAHEAD}, MARGIN={MARGIN_DELTA}, ENTROPY_MAX={ENTROPY_MAX}")
    pred_rows, event_rows = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"θ={THRESH}"):
        npy_path = row["npy_path"]
        if not os.path.exists(npy_path):
            continue

        # ---- Load & Normalize ----
        x = np.load(npy_path).astype(np.float32)
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-5
        x = (x - mu) / sd

        xb = torch.tensor(x).unsqueeze(0).to(DEVICE)
        x_lens = torch.tensor([x.shape[0]], dtype=torch.long).to(DEVICE)

        # ---- Inference ----
        with torch.no_grad():
            start_time = time.time()
            logits = model(xb, x_lens)
            probs = torch.softmax(logits[0, :x.shape[0], :], dim=-1).cpu().numpy()

        # ---- Identify video_id ----
        if "video_id" in df.columns:
            video_id = row["video_id"]
        elif "file_name" in df.columns:
            video_id = os.path.splitext(os.path.basename(row["file_name"]))[0]
        else:
            video_id = f"sample_{i}"

        committed_tokens = []
        window_conf = []
        prev_token = None
        T = probs.shape[0]

        # ---- Commit Policy ----
        for t in range(T):
            end_t = min(T, t + LOOKAHEAD + 1)
            avg_prob = np.mean(probs[t:end_t, :], axis=0)
            token_id = np.argmax(avg_prob)

            # confidence stats
            p_sorted = np.sort(avg_prob)
            p1, p2 = p_sorted[-1], p_sorted[-2]
            conf = p1
            margin_ok = (p1 - p2) >= MARGIN_DELTA
            ent = entropy(avg_prob)

            window_conf.append((token_id, conf))
            if len(window_conf) > WINDOW:
                window_conf.pop(0)

            tokens = [tok for tok, _ in window_conf]
            confs = [c for _, c in window_conf]

            if (
                len(set(tokens)) == 1 and
                np.mean(confs) >= THRESH and
                margin_ok and
                ent <= ENTROPY_MAX and
                token_id != blank_id and
                token_id != prev_token
            ):
                committed_tokens.append(token_id)
                prev_token = token_id
                commit_time = time.time() - start_time
                event_rows.append({
                    "video_id": video_id,
                    "frame_idx": t,
                    "commit_time_sec": commit_time,
                    "gloss": vocab.itos[token_id],
                    "confidence": np.mean(confs),
                    "entropy": ent,
                    "threshold": THRESH,
                    "window": WINDOW,
                    "lookahead": LOOKAHEAD,
                    "margin_delta": MARGIN_DELTA,
                    "entropy_max": ENTROPY_MAX
                })
                window_conf.clear()

        # ---- Save predictions ----
        pred_glosses = [vocab.itos[i] for i in committed_tokens if i in vocab.itos]
        pred_rows.append({
            "video_id": video_id,
            "predicted_gloss": " ".join(pred_glosses),
            "ref_gloss": row["gloss"]
        })

    # ---- Save Output Files ----
    gloss_csv = os.path.join(OUTPUT_DIR, f"phoenix_test_predicted_gloss_commit_adv_th{THRESH}.csv")
    latency_csv = os.path.join(OUTPUT_DIR, f"commit_events_adv_th{THRESH}.csv")
    pd.DataFrame(pred_rows).to_csv(gloss_csv, index=False)
    pd.DataFrame(event_rows).to_csv(latency_csv, index=False)

    print(f"✅ Saved gloss predictions: {gloss_csv}")
    print(f"📁 Saved latency events: {latency_csv}\n")

print("\n🎯 All thresholds processed successfully with soft commit policy!")
