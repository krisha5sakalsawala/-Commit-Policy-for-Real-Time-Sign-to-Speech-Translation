""" 
======================================================
Baseline End-to-End Latency Evaluation
======================================================
Handles:
  - Empty gloss sequences (skip safely)
  - Decoding or TTS errors (continue)
  - Saves .mp3 files + latency CSV
======================================================
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from cslr_train_ctc import BiLSTM_CTC
from train_gloss2text_attn import Seq2Seq, build_vocab, decode, SPECIALS
from text_to_speech import text_to_speech as synthesize_speech

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
BASE_DIR = r"D:\Graduate Project\F"
CHECKPOINT_CSLR = os.path.join(BASE_DIR, "baseline", "cslr_ctc_best.pt")
CHECKPOINT_G2T  = os.path.join(BASE_DIR, "g2t_out", "g2t_attn_best.pt")
LANDMARKS_DIR   = os.path.join(BASE_DIR, "data", "landmarks", "test")

OUT_DIR = os.path.join(BASE_DIR, "results_baseline")
AUDIO_DIR = os.path.join(OUT_DIR, "tts_outputs_baseline")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "baseline_end2end_latency.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
print("🔹 Loading CSLR (Sign→Gloss) model...")
ckpt_cslr = torch.load(CHECKPOINT_CSLR, map_location=DEVICE)
vocab_dict = ckpt_cslr["vocab"]

vocab = type("Vocab", (), {})()
vocab.stoi = vocab_dict
vocab.itos = {v: k for k, v in vocab_dict.items()}
vocab.blank = "<blank>"
vocab.pad = "<pad>"

meta_csv = os.path.join(LANDMARKS_DIR, "test_metadata.csv")
df = pd.read_csv(meta_csv)
feat_dim = np.load(df.iloc[0]["npy_path"]).shape[1]

model_cslr = BiLSTM_CTC(in_dim=feat_dim, hidden=256, vocab_size=len(vocab.stoi))
model_cslr.load_state_dict(ckpt_cslr["model"])
model_cslr.to(DEVICE).eval()
print("✅ CSLR model loaded.\n")

print("🔹 Loading Gloss→Text (Seq2Seq) model...")
ckpt_g2t = torch.load(CHECKPOINT_G2T, map_location=DEVICE)
gloss_stoi, text_stoi = ckpt_g2t["gloss_stoi"], ckpt_g2t["text_stoi"]
text_itos = {v: k for k, v in text_stoi.items()}

model_g2t = Seq2Seq(src_vocab=len(gloss_stoi), tgt_vocab=len(text_stoi))
model_g2t.load_state_dict(ckpt_g2t["model"], strict=False)
model_g2t.to(DEVICE).eval()
print("✅ Gloss→Text model loaded.\n")

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
records = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Baseline End2End"):
    npy_path = row["npy_path"]
    if not os.path.exists(npy_path):
        continue

    video_id = row.get("video_id", f"sample_{i}")
    x = np.load(npy_path).astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-5
    x = (x - mu) / sd

    xb = torch.tensor(x).unsqueeze(0).to(DEVICE)
    x_lens = torch.tensor([x.shape[0]], dtype=torch.long).to(DEVICE)

    try:
        # ---- Stage 1: Sign → Gloss ----
        t0 = time.time()
        with torch.no_grad():
            logits = model_cslr(xb, x_lens)
            probs = torch.softmax(logits[0, :x.shape[0], :], dim=-1)
            pred = probs.argmax(dim=-1).tolist()
        gloss_seq = [vocab.itos[i] for i in pred if i != vocab.stoi[vocab.blank]]
        gloss_str = " ".join(gloss_seq)
        t1 = time.time()

        if len(gloss_seq) == 0:
            print(f"⚠️  Skipping {video_id}: no valid gloss tokens predicted.")
            continue

        # ---- Stage 2: Gloss → Text ----
        gloss_ids = [[gloss_stoi.get(g, SPECIALS["<unk>"]) for g in gloss_seq]]
        gloss_tensor = torch.tensor(gloss_ids, dtype=torch.long).to(DEVICE)
        src_len = torch.tensor([len(gloss_seq)], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            text_ids = model_g2t.greedy_decode(gloss_tensor, src_len, max_len=60)
            text_str = " ".join(decode(text_ids[0].tolist(), text_itos))
        t2 = time.time()

        # ---- Stage 3: Text → Speech ----
        audio_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
        tts_latency, _ = synthesize_speech(text_str, audio_path)
        t3 = time.time()

        # ---- Record all latencies ----
        records.append({
            "video_id": video_id,
            "gloss_latency(s)": round(t1 - t0, 3),
            "text_latency(s)": round(t2 - t1, 3),
            "tts_latency(s)": round(tts_latency, 3) if tts_latency else None,
            "total_latency(s)": round(t3 - t0, 3),
            "pred_text": text_str,
            "audio_path": audio_path
        })

    except Exception as e:
        print(f"⚠️  Skipped {video_id} due to error: {e}")
        continue

# ----------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------
if records:
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Baseline End-to-End Latency Evaluation Complete!")
    print(f"📁 CSV saved to: {OUT_CSV}")
    print(f"🔊 Audio saved to: {AUDIO_DIR}\n")
    print(df_out.head())
else:
    print("⚠️ No valid samples processed — check inputs or model.")
