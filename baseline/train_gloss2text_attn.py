import os, argparse, math, random
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===============================================
# CONFIG
# ===============================================
BASE_DIR = r"D:\Graduate Project\F"
DEFAULT_TRAIN = os.path.join(BASE_DIR, "phoenix_train_clean.csv")
DEFAULT_DEV   = os.path.join(BASE_DIR, "phoenix_dev_clean.csv")
DEFAULT_PRED  = os.path.join(BASE_DIR, "results", "phoenix_test_predicted_gloss.csv")
DEFAULT_OUT   = os.path.join(BASE_DIR, "g2t_out")

SPECIALS = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}

# ===============================================
# Tokenization / Vocabulary
# ===============================================
def tokenize(s):
    return s.lower().strip().split()

def build_vocab(sentences, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for s in sentences:
        cnt.update(tokenize(str(s)))
    itos = [None]*4
    for k,v in SPECIALS.items(): itos[v] = k
    for w,f in cnt.items():
        if f >= min_freq and w not in SPECIALS:
            itos.append(w)
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode(tokens, stoi, add_sos_eos=False, max_len=None):
    ids = [stoi.get(t, SPECIALS["<unk>"]) for t in tokens]
    if add_sos_eos:
        ids = [SPECIALS["<sos>"]] + ids + [SPECIALS["<eos>"]]
    if max_len is not None:
        ids = ids[:max_len]
    return ids

def decode(ids, itos):
    out = []
    for i in ids:
        if i == SPECIALS["<eos>"]: break
        if i in (SPECIALS["<sos>"], SPECIALS["<pad>"]): continue
        out.append(itos[i] if i < len(itos) else "<unk>")
    return out

# ===============================================
# Dataset
# ===============================================
class Gloss2TextDataset(Dataset):
    def __init__(self, df, gloss_stoi, text_stoi, max_src_len=100, max_tgt_len=50):
        self.df = df.reset_index(drop=True)
        self.gloss_stoi = gloss_stoi
        self.text_stoi = text_stoi
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        g = str(self.df.iloc[idx]["gloss_clean"])
        t = str(self.df.iloc[idx]["text_clean"])
        src = encode(tokenize(g), self.gloss_stoi, add_sos_eos=False, max_len=self.max_src_len)
        tgt = encode(tokenize(t), self.text_stoi, add_sos_eos=True,  max_len=self.max_tgt_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate(batch):
    srcs, tgts = zip(*batch)
    src_len = [len(x) for x in srcs]
    tgt_len = [len(y) for y in tgts]
    max_src, max_tgt = max(src_len), max(tgt_len)
    pad_id = SPECIALS["<pad>"]

    src_pad = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        src_pad[i,:len(s)] = s
        tgt_pad[i,:len(t)] = t
    return src_pad, torch.tensor(src_len), tgt_pad, torch.tensor(tgt_len)

# ===============================================
# Seq2Seq with Luong Attention
# ===============================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=SPECIALS["<pad>"])
        self.rnn = nn.LSTM(emb_dim, hid, num_layers=num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers>1 else 0.)
        self.fc = nn.Linear(hid*2, hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        emb = self.dropout(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h,c) = self.rnn(packed)
        out,_ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        h0 = torch.tanh(self.fc(h_cat)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        return out, (h0, c0)

class LuongAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.scale = 1.0 / math.sqrt(hid)

    def forward(self, dec_h, enc_out, src_mask):
        if not hasattr(self, "proj"):
            self.proj = nn.Linear(enc_out.size(-1), dec_h.size(-1), bias=False).to(enc_out.device)
        keys = self.proj(enc_out)
        scores = torch.bmm(keys, dec_h.transpose(1,2)).squeeze(-1) * self.scale
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid, enc_out_dim, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=SPECIALS["<pad>"])
        self.rnn = nn.LSTM(emb_dim + hid, hid, batch_first=True)
        self.attn = LuongAttention(hid)
        self.proj_ctx = nn.Linear(enc_out_dim, hid, bias=False)
        self.fc = nn.Linear(hid, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_prev, hidden, enc_out, src_mask):
        emb = self.dropout(self.emb(y_prev)).unsqueeze(1)
        dec_h = hidden[0].transpose(0,1)
        ctx, _ = self.attn(dec_h, enc_out, src_mask)
        ctx_proj = self.proj_ctx(ctx)
        rnn_in = torch.cat([emb, ctx_proj], dim=-1)
        out, hidden = self.rnn(rnn_in, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb, hid, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, emb, hid, enc_out_dim=hid*2, dropout=dropout)
        self.tgt_vocab = tgt_vocab

    def make_src_mask(self, src):
        return (src != SPECIALS["<pad>"])

    def forward(self, src, src_len, tgt, teacher_forcing=0.5):
        enc_out, hidden = self.encoder(src, src_len)
        src_mask = self.make_src_mask(src)
        B, T = tgt.size()
        logits = []
        y = tgt[:,0]
        for t in range(1, T):
            logit, hidden = self.decoder(y, hidden, enc_out, src_mask)
            logits.append(logit.unsqueeze(1))
            y = tgt[:,t] if random.random() < teacher_forcing else logit.argmax(dim=-1)
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=50):
        enc_out, hidden = self.encoder(src, src_len)
        src_mask = self.make_src_mask(src)
        B = src.size(0)
        y = torch.full((B,), SPECIALS["<sos>"], dtype=torch.long, device=src.device)
        outs = []
        for _ in range(max_len):
            logit, hidden = self.decoder(y, hidden, enc_out, src_mask)
            y = logit.argmax(dim=-1)
            outs.append(y.unsqueeze(1))
            if (y == SPECIALS["<eos>"]).all(): break
        return torch.cat(outs, dim=1) if outs else torch.full((B,1), SPECIALS["<eos>"], device=src.device)

# ===============================================
# Training and Evaluation
# ===============================================
def train_epoch(model, loader, criterion, opt, device, tf=0.5):
    model.train(); total = 0.0
    for src, src_len, tgt, _ in tqdm(loader, desc="Train", leave=False):
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        opt.zero_grad()
        logits = model(src, src_len, tgt, teacher_forcing=tf)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate_bleu(model, loader, tgt_itos, device):
    model.eval(); smoothie = SmoothingFunction().method4
    refs, hyps = [], []
    for src, src_len, tgt, _ in tqdm(loader, desc="Dev", leave=False):
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        gen = model.greedy_decode(src, src_len, max_len=60)
        for i in range(src.size(0)):
            ref = decode(tgt[i].tolist(), tgt_itos)
            hyp = decode(gen[i].tolist(), tgt_itos)
            if len(ref)==0: continue
            refs.append([ref])
            hyps.append(hyp if len(hyp)>0 else ["<unk>"])
    return corpus_bleu(refs, hyps, smoothing_function=smoothie) if refs else 0.0

# ===============================================
# Build parallel gloss-text DataFrame from CSLR predictions
# ===============================================
def make_parallel_from_pred(pred_csv, ref_df, gloss_col="predicted_gloss", ref_col="text_clean"):
    pred = pd.read_csv(pred_csv)
    n = min(len(pred), len(ref_df))
    glosses = pred[gloss_col].fillna("").astype(str).iloc[:n].tolist()
    texts   = ref_df[ref_col].fillna("").astype(str).iloc[:n].tolist()
    return pd.DataFrame({"gloss_clean": glosses, "text_clean": texts})


# ===============================================
# TRAIN LOOP
# ===============================================
def main():
    nltk.download("punkt", quiet=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📘 Loading data...")
    train_df = pd.read_csv(DEFAULT_TRAIN)
    dev_df   = pd.read_csv(DEFAULT_DEV)

    gloss_stoi, gloss_itos = build_vocab(train_df["gloss_clean"])
    text_stoi,  text_itos  = build_vocab(train_df["text_clean"])
    print(f"Vocab sizes — gloss:{len(gloss_itos)} text:{len(text_itos)}")

    train_loader = DataLoader(
        Gloss2TextDataset(train_df, gloss_stoi, text_stoi),
        batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(
        Gloss2TextDataset(dev_df, gloss_stoi, text_stoi),
        batch_size=32, shuffle=False, collate_fn=collate)

    model = Seq2Seq(len(gloss_itos), len(text_itos), emb=256, hid=256).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIALS["<pad>"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_bleu = 0.0
    for ep in range(1, 16):
        tr_loss = train_epoch(model, train_loader, criterion, opt, device)
        bleu = evaluate_bleu(model, dev_loader, text_itos, device)
        print(f"Epoch {ep:02d} | Train Loss: {tr_loss:.3f} | Dev BLEU: {bleu:.3f}")

        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(DEFAULT_OUT, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "gloss_stoi": gloss_stoi,
                        "text_stoi": text_stoi},
                        os.path.join(DEFAULT_OUT, "g2t_attn_best.pt"))
            print("✅ Saved best model")

    print(f"Best Dev BLEU: {best_bleu:.3f}")

if __name__ == "__main__":
    main()
