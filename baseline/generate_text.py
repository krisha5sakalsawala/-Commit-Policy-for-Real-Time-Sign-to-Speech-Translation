import os
import pandas as pd
from tqdm import tqdm
import torch

# Import needed components from training file
from train_gloss2text_attn import (
    Gloss2TextDataset, collate, decode, make_parallel_from_pred,
    Seq2Seq, BASE_DIR, DEFAULT_PRED, DEFAULT_DEV, DEFAULT_OUT, SPECIALS
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained checkpoint
    ckpt_path = os.path.join(DEFAULT_OUT, "g2t_attn_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"📦 Loading trained Gloss→Text model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    gloss_stoi = ckpt["gloss_stoi"]
    text_stoi  = ckpt["text_stoi"]

    # Build inverse vocab
    gloss_itos = {v:k for k,v in gloss_stoi.items()}
    text_itos  = {v:k for k,v in text_stoi.items()}

    # Load dev as reference text
    dev_df = pd.read_csv(DEFAULT_DEV)

    # Build gloss/text pairs from predicted gloss file
    print("🔄 Preparing gloss-text pairs from CSLR predictions...")
    test_df = make_parallel_from_pred(DEFAULT_PRED, dev_df)

    # Replace empty gloss rows
    test_df["gloss_clean"] = test_df["gloss_clean"].apply(
        lambda x: "<unk>" if str(x).strip()=="" else str(x)
    )

    # Create dataset and dataloader
    loader = torch.utils.data.DataLoader(
        Gloss2TextDataset(test_df, gloss_stoi, text_stoi),
        batch_size=32, shuffle=False, collate_fn=collate
    )

    # Restore trained model
    model = Seq2Seq(len(gloss_itos), len(text_itos), emb=256, hid=256).to(device)
    with torch.no_grad():
        dummy_src = torch.zeros((1, 5), dtype=torch.long).to(device)
        dummy_len = torch.tensor([5]).to(device)
        model.greedy_decode(dummy_src, dummy_len)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("🎯 Translating gloss → text ...")
    outputs = []

    with torch.no_grad():
        for src, src_len, tgt, tgt_len in tqdm(loader, desc="Translate"):
            src, src_len = src.to(device), src_len.to(device)
            gen = model.greedy_decode(src, src_len)

            for i in range(src.size(0)):
                hyp = " ".join(decode(gen[i].tolist(), text_itos))
                ref = " ".join(decode(tgt[i].tolist(), text_itos))
                outputs.append({"pred_text": hyp, "ref_text": ref})

    # Save final output
    out_path = os.path.join(DEFAULT_OUT, "phoenix_test_predicted_text.csv")
    pd.DataFrame(outputs).to_csv(out_path, index=False)

    print(f"✅ Saved translations to: {out_path}")

if __name__ == "__main__":
    main()
