import os
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils: vocab, metrics, decode
# -----------------------------
class GlossVocab:
    def __init__(self, gloss_sequences):
        # tokens are glosses split by space
        self.pad = "<pad>"
        self.blank = "<blank>"   # for CTC
        self.unk = "<unk>"
        tokens = set()
        for g in gloss_sequences:
            for t in g.strip().split():
                tokens.add(t)
        self.itos = [self.pad, self.blank, self.unk] + sorted(tokens)
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, gloss_str):
        return [self.stoi.get(t, self.stoi[self.unk]) for t in gloss_str.strip().split()]
    def decode(self, ids):
        return [self.itos[i] for i in ids]
    def __len__(self): return len(self.itos)

def levenshtein(a, b):
    # token-level edit distance
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,     # del
                           dp[i][j-1] + 1,     # ins
                           dp[i-1][j-1] + cost) # sub
    return dp[-1][-1]

def greedy_ctc_decode(logits, blank_id):
    # logits: (T, V). Return collapsed tokens without repeats/blanks.
    pred = logits.argmax(dim=-1).tolist()
    collapsed = []
    prev = None
    for p in pred:
        if p != blank_id and p != prev:
            collapsed.append(p)
        prev = p
    return collapsed

# -----------------------------
# Dataset
# -----------------------------
class LandmarkCTCDataset(Dataset):
    def __init__(self, metadata_csv, vocab):
        self.df = pd.read_csv(metadata_csv)
        # Keep only rows whose .npy actually exists & has frames
        keep = []
        for i, row in self.df.iterrows():
            if isinstance(row["npy_path"], str) and os.path.exists(row["npy_path"]):
                keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)
        self.vocab = vocab

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"])  # (T, F)
        x = x.astype(np.float32)
        # simple per-video normalization (optional)
        if x.size > 0:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-5
            x = (x - mu) / sd
        y = self.vocab.encode(row["gloss"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def collate_ctc(batch):
    # batch: list of (x: [T,F], y: [L])
    xs, ys = zip(*batch)
    T = [x.shape[0] for x in xs]
    F = xs[0].shape[1] if len(xs)>0 else 0
    L = [y.shape[0] for y in ys]

    maxT = max(T)
    maxL = max(L) if len(L)>0 else 0

    xpad = torch.zeros(len(batch), maxT, F, dtype=torch.float32)
    for i, x in enumerate(xs):
        xpad[i, :x.shape[0]] = x

    ycat = torch.cat(ys, dim=0) if len(ys)>0 else torch.tensor([], dtype=torch.long)
    x_lens = torch.tensor(T, dtype=torch.long)
    y_lens = torch.tensor(L, dtype=torch.long)

    return xpad, ycat, x_lens, y_lens

# -----------------------------
# Model: BiLSTM + Linear (CTC)
# -----------------------------
class BiLSTM_CTC(nn.Module):
    def __init__(self, in_dim, hidden, vocab_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim,
                           hidden_size=hidden,
                           num_layers=num_layers,
                           dropout=dropout if num_layers>1 else 0.0,
                           bidirectional=True,
                           batch_first=True)
        self.fc = nn.Linear(hidden*2, vocab_size)  # bi-directional
    def forward(self, x, x_lens):
        # x: (B, T, F)
        packed = nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)
        logits = self.fc(out)  # (B, T, V)
        return logits

# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb, x_lens, y_lens in tqdm(loader, desc="Train", leave=False):
        xb = xb.to(device)
        x_lens = x_lens.to(device)
        yb = yb.to(device)
        y_lens = y_lens.to(device)

        optimizer.zero_grad()
        logits = model(xb, x_lens)                # (B,T,V)
        log_probs = logits.log_softmax(dim=-1)    # CTC expects log-probs

        # For CTC loss, need shape (T,B,V) and lengths on CPU
        log_probs = log_probs.transpose(0, 1)     # (T,B,V)
        loss = criterion(log_probs, yb, x_lens.cpu(), y_lens.cpu())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate_dev(model, loader, vocab, device):
    model.eval()
    blank_id = vocab.stoi[vocab.blank]
    total_edits = 0
    total_tokens = 0
    for xb, yb, x_lens, y_lens in tqdm(loader, desc="Dev", leave=False):
        xb = xb.to(device)
        x_lens = x_lens.to(device)

        logits = model(xb, x_lens)        # (B,T,V)
        for i in range(xb.size(0)):
            T_i = x_lens[i].item()
            # greedily decode this sample
            greedy = greedy_ctc_decode(logits[i, :T_i, :], blank_id)
            # slice target for this sample
            L_i = y_lens[i].item()
            # yb is concatenated; we need offset accumulation
        # reconstruct batch targets
        # We'll iterate carefully:
        offset = 0
        for i in range(xb.size(0)):
            L_i = y_lens[i].item()
            tgt = yb[offset:offset+L_i].tolist()
            offset += L_i

        
    # Small batch loop to keep it simple
    total_edits = 0
    total_tokens = 0
    for xb, yb, x_lens, y_lens in loader:
        xb = xb.to(device)
        logits = model(xb, x_lens.to(device))  # (B,T,V)

        # rebuild per-sample targets
        offset = 0
        for i in range(xb.size(0)):
            L_i = y_lens[i].item()
            tgt = yb[offset:offset+L_i].tolist()
            offset += L_i

            T_i = x_lens[i].item()
            pred_ids = greedy_ctc_decode(logits[i, :T_i, :], blank_id)

            # compute token-level edit distance (SER proxy)
            # remove special tokens if any slipped in
            tgt_clean = [t for t in tgt if t not in (vocab.stoi[vocab.pad], vocab.stoi[vocab.blank])]
            ed = levenshtein(pred_ids, tgt_clean)
            total_edits += ed
            total_tokens += max(1, len(tgt_clean))

    ser = total_edits / max(1, total_tokens)
    return ser

# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) Load CSVs & build vocab from TRAIN gloss
    train_csv = os.path.join(args.landmarks_dir, "train", "train_metadata.csv")
    dev_csv   = os.path.join(args.landmarks_dir, "dev",   "dev_metadata.csv")
    if not os.path.exists(train_csv) or not os.path.exists(dev_csv):
        raise FileNotFoundError("Could not find train/dev metadata CSVs under landmarks_dir")

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)

    vocab = GlossVocab(train_df["gloss"].tolist())
    print(f"Vocab size (incl. specials): {len(vocab)}")

    # 2) Datasets and loaders
    train_set = LandmarkCTCDataset(train_csv, vocab)
    dev_set   = LandmarkCTCDataset(dev_csv, vocab)

    # infer feature dim from first sample
    if len(train_set) == 0:
        raise RuntimeError("Train set is empty; check your paths.")
    feat_dim = np.load(train_set.df.iloc[0]["npy_path"]).shape[1]
    print(f"Feature dim: {feat_dim}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ctc)
    dev_loader   = DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_ctc)

    # 3) Model, loss, optim
    model = BiLSTM_CTC(in_dim=feat_dim, hidden=args.hidden, vocab_size=len(vocab)).to(device)
    ctc_loss = nn.CTCLoss(blank=vocab.stoi[vocab.blank], zero_infinity=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume Checkpoint Logic
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✅ Resumed from checkpoint: {args.resume}")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"➡️  Continuing from epoch {start_epoch}")

    best_ser = math.inf

    # 4) Train
    best_ser = math.inf
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, ctc_loss, optim, device)
        ser = evaluate_dev(model, dev_loader, vocab, device)
        print(f"Epoch {epoch:02d} | Train CTC Loss: {tr_loss:.4f} | Dev SER: {ser:.3f}")

        if ser < best_ser:
            best_ser = ser
            torch.save({"model": model.state_dict(), "vocab": vocab.stoi},
                       os.path.join(args.out_dir, "cslr_ctc_best.pt"))
            print("✅ Saved best model")

    print(f"Best Dev SER: {best_ser:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_dir", type=str,
                        default=r"D:\Graduate Project\F\data\landmarks",
                        help="Folder containing train/dev/test subfolders with *_metadata.csv and .npy")
    parser.add_argument("--out_dir", type=str, default=".", help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")  # 🆕 Added

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
