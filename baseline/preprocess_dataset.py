import os
import json
import gzip
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from typing import Dict, List, Any, Tuple, Optional

"""
Preprocess sign-language dataset:
- Reads .gzip annotations (JSON list OR pipe-delimited lines: video|gloss|text)
- Matches each record to a video
- Extracts MediaPipe Holistic landmarks per-frame
- Saves per-video numpy arrays to landmarks/{split}/{video_id}.npy
- Writes metadata.csv with mapping and stats

Run:
python preprocess_dataset.py --root /path/to/dataset_root \
  --splits train dev test \
  --annos annotations.train.gzip annotations.dev.gzip annotations.test.gzip
    
Notes:
- MediaPipe outputs normalized 2D coords [0,1]; we keep them as-is.
- If you want fewer landmarks, you can toggle FACE/HANDS/POSE flags below.
"""

import argparse

# -----------------------------
# Config: which landmarks to extract
# -----------------------------
USE_FACE = True        # 468 points (x,y)
USE_POSE = True        # 33 points (x,y)
USE_HANDS = True       # 21 left + 21 right (x,y)

# Flatten order per frame: [pose, left_hand, right_hand, face] (if enabled)
def extract_frame_vector(results) -> np.ndarray:
    frame_vec = []
    if USE_POSE and results.pose_landmarks:
        frame_vec.extend([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
    if USE_HANDS and results.left_hand_landmarks:
        frame_vec.extend([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark])
    elif USE_HANDS:
        frame_vec.extend([[np.nan, np.nan]] * 21)
    if USE_HANDS and results.right_hand_landmarks:
        frame_vec.extend([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark])
    elif USE_HANDS:
        frame_vec.extend([[np.nan, np.nan]] * 21)
    if USE_FACE and results.face_landmarks:
        # Use a subset if you want fewer points; here we keep all 468
        frame_vec.extend([[lm.x, lm.y] for lm in results.face_landmarks.landmark])
    elif USE_FACE:
        frame_vec.extend([[np.nan, np.nan]] * 468)

    if len(frame_vec) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(frame_vec, dtype=np.float32).reshape(-1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root folder")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="List of splits")
    ap.add_argument("--annos", nargs="+", required=True,
                    help="List of .gzip annotation files aligned with --splits order")
    ap.add_argument("--out_csv", default="metadata.csv", help="Output metadata CSV path")
    ap.add_argument("--max_videos", type=int, default=None, help="(Optional) limit for quick tests")
    ap.add_argument("--skip_existing", action="store_true", help="Skip videos with existing .npy")
    ap.add_argument("--min_frames", type=int, default=8, help="Skip very short clips")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def read_annotations_gzip(path: str) -> List[Dict[str, Any]]:
    """
    Supports two formats inside .gzip:
    1) JSON list of dicts with keys like: video_id, gloss (list or string), translation
    2) Pipe-delimited lines: <video_id>|<gloss text>|<translation text>
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read()

    # Try JSON first
    try:
        data = json.loads(text)
        # Normalize a bit
        records = []
        for d in data:
            vid = d.get("video_id") or d.get("name") or d.get("id") or d.get("video") or None
            gloss = d.get("gloss") or d.get("gloss_sequence") or d.get("glosses")
            if isinstance(gloss, str):
                gloss_tokens = gloss.strip().split()
            elif isinstance(gloss, list):
                gloss_tokens = [str(x) for x in gloss]
            else:
                gloss_tokens = []

            sent = d.get("translation") or d.get("spoken_sentence") or d.get("text") or ""
            fps = d.get("fps") or d.get("frame_rate")
            n_frames = d.get("num_frames") or d.get("frames")

            records.append({
                "video_id": vid,
                "gloss": gloss_tokens,
                "sentence": sent,
                "fps": fps,
                "n_frames": n_frames
            })
        # filter None ids
        return [r for r in records if r["video_id"]]
    except Exception:
        pass

    # Otherwise treat as pipe-delimited lines
    lines = text.strip().splitlines()
    records = []
    for ln in lines:
        # Expected: video_id|gloss text|translation text
        parts = ln.strip().split("|")
        if len(parts) < 3:
            # try a variant with commas or tabs if needed
            parts = ln.strip().split("\t")
            if len(parts) < 3:
                continue
        vid = parts[0].strip().replace(".mp4", "")
        gloss_tokens = parts[1].strip().split()
        sent = parts[2].strip()
        records.append({
            "video_id": vid,
            "gloss": gloss_tokens,
            "sentence": sent,
            "fps": None,
            "n_frames": None
        })
    return records


def video_path_for(root: str, split: str, video_id: str) -> Optional[str]:
    # Try a few naming conventions
    cand = [
        os.path.join(root, "videos", split, f"{video_id}.mp4"),
        os.path.join(root, "videos", split, f"{video_id}.avi"),
        os.path.join(root, split, f"{video_id}.mp4"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def extract_video_landmarks(video_path: str) -> Tuple[np.ndarray, float]:
    """Return (T x F) array and fps (float). F is flattened landmark dim."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    )

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        vec = extract_frame_vector(results)
        frames.append(vec)
    cap.release()
    holistic.close()

    # Pad missing per-frame vectors to same length (if e.g., first frame had no detections)
    maxF = max(len(v) for v in frames) if frames else 0
    if maxF == 0:
        return np.zeros((0,)), fps

    for i in range(len(frames)):
        if len(frames[i]) < maxF:
            pad = np.full((maxF - len(frames[i]),), np.nan, dtype=np.float32)
            frames[i] = np.concatenate([frames[i], pad], axis=0)

    arr = np.stack(frames, axis=0).astype(np.float32)  # (T, F)
    return arr, fps


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    root = args.root
    out_csv = os.path.join(root, args.out_csv)

    if len(args.splits) != len(args.annos):
        raise ValueError("Provide one .gzip annotation path for each split in --splits")

    rows = []
    total_processed = 0

    for split, anno in zip(args.splits, args.annos):
        print(f"\n[Split: {split}] reading annotations:", anno)
        records = read_annotations_gzip(anno)
        print(f"  Found {len(records)} records in {anno}")

        out_dir = os.path.join(root, "landmarks", split)
        ensure_dir(out_dir)

        if args.max_videos:
            records = records[:args.max_videos]

        for rec in tqdm(records, desc=f"Extract {split}"):
            vid = rec["video_id"].replace(".mp4", "")
            vp = video_path_for(root, split, vid)
            if vp is None:
                # Try adding .mp4 extension from id
                alt = video_path_for(root, split, vid + ".mp4")
                if alt:
                    vp = alt
            if vp is None:
                if args.verbose:
                    print(f"  [WARN] Video not found for {vid}")
                continue

            npy_path = os.path.join(out_dir, f"{vid}.npy")
            if args.skip_existing and os.path.exists(npy_path):
                # still record metadata (will fill fps/n_frames later if unknown)
                arr = np.load(npy_path, mmap_mode="r")
                n_frames = arr.shape[0]
                fps = rec["fps"] or np.nan
                rows.append({
                    "video_id": vid,
                    "split": split,
                    "npy_path": os.path.relpath(npy_path, root),
                    "gloss": " ".join(rec["gloss"]),
                    "sentence": rec["sentence"],
                    "n_frames": n_frames,
                    "fps": fps
                })
                continue

            try:
                arr, fps = extract_video_landmarks(vp)
            except Exception as e:
                if args.verbose:
                    print(f"  [ERROR] {vid}: {e}")
                continue

            if arr.shape[0] < args.min_frames:
                if args.verbose:
                    print(f"  [SKIP] too short: {vid} ({arr.shape[0]} frames)")
                continue

            np.save(npy_path, arr)
            rows.append({
                "video_id": vid,
                "split": split,
                "npy_path": os.path.relpath(npy_path, root),
                "gloss": " ".join(rec["gloss"]),
                "sentence": rec["sentence"],
                "n_frames": arr.shape[0],
                "fps": float(fps)
            })
            total_processed += 1

    # Write metadata CSVq
    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(["split", "video_id"], inplace=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote metadata CSV: {out_csv} (rows={len(df)})")
        # Show a couple of examples
        print(df.head(3).to_string(index=False))
    else:
        print("\nNo videos processed. Check paths and formats.")


if __name__ == "__main__":
    main()
