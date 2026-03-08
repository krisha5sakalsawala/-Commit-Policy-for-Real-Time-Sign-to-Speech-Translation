import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from cslr_train_ctc import BiLSTM_CTC, greedy_ctc_decode  # reuse from your script

# -------------------------
# CONFIG
# -------------------------
CHECKPOINT_PATH = "cslr_ctc_best.pt"
LANDMARKS_DIR = r"D:\Graduate Project\F\data\landmarks\test"
OUTPUT_CSV = r"D:\Graduate Project\F\results\phoenix_test_predicted_gloss.csv"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD MODEL + VOCAB
# -------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
vocab_dict = checkpoint["vocab"]

# Rebuild vocab-like object
vocab = type("Vocab", (), {})()
vocab.stoi = vocab_dict
vocab.itos = {v: k for k, v in vocab_dict.items()}
vocab.blank = "<blank>"
vocab.pad = "<pad>"

# Load test metadata
sample_csv = os.path.join(LANDMARKS_DIR, "test_metadata.csv")
df = pd.read_csv(sample_csv)

# Infer feature dimension from first .npy
feat_dim = np.load(df.iloc[0]["npy_path"]).shape[1]

# Initialize model
model = BiLSTM_CTC(in_dim=feat_dim, hidden=256, vocab_size=len(vocab.stoi))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# -------------------------
# GENERATE PREDICTIONS
# -------------------------
pred_rows = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting glosses"):
    npy_path = row["npy_path"]
    if not os.path.exists(npy_path):
        continue

    # Load and normalize landmarks
    x = np.load(npy_path).astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-5
    x = (x - mu) / sd

    xb = torch.tensor(x).unsqueeze(0).to(DEVICE)  # (1, T, F)
    x_lens = torch.tensor([x.shape[0]], dtype=torch.long).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(xb, x_lens)
        pred_ids = greedy_ctc_decode(logits[0, :x.shape[0], :], vocab.stoi[vocab.blank])
        pred_glosses = [vocab.itos[i] for i in pred_ids if i in vocab.itos]

    # Add both predicted and reference gloss
    pred_rows.append({
        "video_id": row["video_id"] if "video_id" in row else f"sample_{i}",
        "predicted_gloss": " ".join(pred_glosses),
        "ref_gloss": row["gloss"] if "gloss" in row else ""
    })

# -------------------------
# SAVE CSV
# -------------------------
pred_df = pd.DataFrame(pred_rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved predictions with reference glosses to {OUTPUT_CSV}")
print(pred_df.head())
