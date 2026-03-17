"""
======================================================
SIGN → GLOSS → TEXT → SPEECH (Buffered Streaming)
======================================================
Continuous sign-to-speech translation with real-time
commit policy triggering + gloss buffer recording.

Enhancement:
- Each 2-gloss commit is recorded to CSV ("stream_predicted_gloss.csv")
- CSV is lowercase-normalized for G2T vocab consistency
- Proper model reset and evaluation per decoding
======================================================
"""

import os, sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy

BASE_DIR = r"D:\Graduate Project\F"
sys.path.append(BASE_DIR)

# --- Import local modules ---
from baseline.cslr_train_ctc import BiLSTM_CTC
from baseline.train_gloss2text_attn import Seq2Seq, decode, SPECIALS, tokenize
from baseline.text_to_speech import text_to_speech as synthesize_speech
from commit_policy import commit_decisions

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

CHECKPOINT_CSLR = os.path.join(BASE_DIR, "baseline", "cslr_ctc_best.pt")
CHECKPOINT_G2T  = os.path.join(BASE_DIR, "g2t_out", "g2t_attn_best.pt")
LANDMARKS_DIR   = os.path.join(BASE_DIR, "data", "landmarks", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "results_streaming")
SPEECH_DIR = os.path.join(OUTPUT_DIR, "tts_outputs_streaming")
EVENT_LOG  = os.path.join(OUTPUT_DIR, "commit_events_adv_th0.45.csv")
STREAM_GLOSS_CSV = os.path.join(OUTPUT_DIR, "stream_predicted_gloss.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SPEECH_DIR, exist_ok=True)

# --- Commit Policy Parameters ---
THRESH = 0.45
WINDOW = 2
LOOKAHEAD = 1
MARGIN_DELTA = 0.01
ENTROPY_MAX = 2.0
CLEAR_AFTER_COMMIT = True
USE_COOLDOWN = False
COOLDOWN_SEC = 0.0

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

# --- Gloss→Text model ---
print("🔹 Loading Gloss→Text (Seq2Seq) model...")
ckpt_g2t = torch.load(CHECKPOINT_G2T, map_location=DEVICE)
gloss_stoi, text_stoi = ckpt_g2t["gloss_stoi"], ckpt_g2t["text_stoi"]
gloss_itos = {v: k for k, v in gloss_stoi.items()}
text_itos  = {v: k for k, v in text_stoi.items()}

model_g2t = Seq2Seq(src_vocab=len(gloss_stoi), tgt_vocab=len(text_stoi))
missing, unexpected = model_g2t.load_state_dict(ckpt_g2t["model"], strict=False)
print(f"ℹ️ Missing keys: {missing}")
print(f"ℹ️ Unexpected keys: {unexpected}")
model_g2t.to(DEVICE).eval()

print("✅ All models loaded successfully.\n")

# ----------------------------------------------------
# PREPARE OUTPUT CSV FILES
# ----------------------------------------------------
if not os.path.exists(STREAM_GLOSS_CSV):
    pd.DataFrame(columns=["video_id", "timestamp_ms", "predicted_gloss"]).to_csv(STREAM_GLOSS_CSV, index=False)
    print(f"🆕 Created stream gloss CSV at {STREAM_GLOSS_CSV}")

# ----------------------------------------------------
# STREAMING LOOP
# ----------------------------------------------------
event_rows = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Streaming Sign→Speech"):
    npy_path = row["npy_path"]
    if not os.path.exists(npy_path):
        continue

    x = np.load(npy_path).astype(np.float32)
    mu, sd = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-5
    x = (x - mu) / sd

    xb = torch.tensor(x).unsqueeze(0).to(DEVICE)
    x_lens = torch.tensor([x.shape[0]], dtype=torch.long).to(DEVICE)

    video_id = row.get("video_id", f"sample_{i}")
    print(f"\n🎥 Processing video: {video_id}")

    with torch.no_grad():
        start_time = time.time()
        logits = model_cslr(xb, x_lens)
        probs = torch.softmax(logits[0, :x.shape[0], :], dim=-1).cpu().numpy()

        commits = commit_decisions(
            probs=probs,
            blank_id=vocab.stoi[vocab.blank],
            THRESH=THRESH,
            WINDOW=WINDOW,
            LOOKAHEAD=LOOKAHEAD,
            MARGIN_DELTA=MARGIN_DELTA,
            ENTROPY_MAX=ENTROPY_MAX,
            CLEAR_AFTER_COMMIT=CLEAR_AFTER_COMMIT,
            USE_COOLDOWN=USE_COOLDOWN,
            COOLDOWN_SEC=COOLDOWN_SEC,
            frame_time=1/25
        )

        committed_tokens = []
        gloss_buffer = []

        for (frame_idx, token_id, conf, ent) in commits:
            gloss_word = vocab.itos[token_id]
            commit_time = time.time() - start_time
            committed_tokens.append(token_id)
            gloss_buffer.append(gloss_word)

            print(f"⏱️ {commit_time:.3f}s → Committed gloss: {gloss_word} (conf={conf:.2f})")

            event_rows.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "commit_time_sec": commit_time,
                "gloss": gloss_word,
                "confidence": conf,
                "entropy": ent,
                "threshold": THRESH
            })

            # ---- Every 2-gloss commit triggers translation ----
            if len(gloss_buffer) >= 2:
                # Normalize (lowercase) for G2T vocab
                normalized_tokens = [t.lower() for t in gloss_buffer]
                gloss_seq = " ".join(normalized_tokens)
                timestamp_ms = int(commit_time * 1000)

                # Save to CSV
                pd.DataFrame([[video_id, timestamp_ms, gloss_seq]],
                             columns=["video_id", "timestamp_ms", "predicted_gloss"]
                             ).to_csv(STREAM_GLOSS_CSV, mode='a', index=False, header=False)
                print(f"💾 Stored gloss buffer to CSV: {gloss_seq}")

                # ---- Translate immediately ----
                gloss_ids = [[gloss_stoi.get(tok, SPECIALS["<unk>"]) for tok in normalized_tokens]]
                gloss_tensor = torch.tensor(gloss_ids, dtype=torch.long).to(DEVICE)
                src_len = torch.tensor([len(normalized_tokens)], dtype=torch.long).to(DEVICE)

                # Ensure deterministic decoding
                model_g2t.eval()
                if hasattr(model_g2t.encoder, "rnn"):
                    model_g2t.encoder.rnn.flatten_parameters()
                if hasattr(model_g2t.decoder.attn, "proj"):
                    del model_g2t.decoder.attn.proj

                with torch.no_grad():
                    text_ids = model_g2t.greedy_decode(gloss_tensor, src_len, max_len=50)
                text_str = " ".join(decode(text_ids[0].tolist(), text_itos))
                print(f"🗣️ Buffered translation: {text_str}")

                # ---- Generate speech ----
                speech_path = f"{SPEECH_DIR}/{video_id}_{timestamp_ms}ms.wav"
                synthesize_speech(text_str, speech_path)
                print(f"🎤 Saved speech: {speech_path}")

                # ---- Log translation ----
                log_path = os.path.join(OUTPUT_DIR, "streaming_text_log.csv")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{video_id},{commit_time:.3f},{gloss_seq},{text_str},{speech_path}\n")

                gloss_buffer.clear()

    print(f"✅ Finished video {video_id} ({len(committed_tokens)} glosses committed)")

# ----------------------------------------------------
# SAVE COMMIT EVENTS
# ----------------------------------------------------
if event_rows:
    pd.DataFrame(event_rows).to_csv(EVENT_LOG, index=False)
    print(f"\n📁 Commit events saved to: {EVENT_LOG}")
else:
    print("\n⚠️ No commits recorded. Check threshold or policy parameters.")

