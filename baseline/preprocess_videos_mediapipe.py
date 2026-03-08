import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import pickle, gzip

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def extract_landmarks_from_video(video_path, visualize=False):
    """Extract pose + hands landmarks from a single video, with optional visualization."""
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    with mp_holistic.Holistic(static_image_mode=False,
                              model_complexity=1,
                              enable_segmentation=False,
                              refine_face_landmarks=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR → RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Collect pose + hands landmarks
            frame_landmarks = []
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0] * (33 * 3))

            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks.extend([0] * (21 * 3))

            all_landmarks.append(frame_landmarks)

            """ # ---- Visualization ----
            if visualize:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw pose + hands
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())

                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 128, 0)))

                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(128, 0, 0)))

                cv2.imshow("MediaPipe Visualization", cv2.resize(image, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                """

    cap.release()
    cv2.destroyAllWindows()
    return np.array(all_landmarks)


def process_split(split_name, video_dir, save_dir, annotations, visualize=False):
    """Process all videos in a split and save metadata."""
    os.makedirs(save_dir, exist_ok=True)
    records = []

    for ann in tqdm(annotations, desc=f"Processing {split_name}"):
        video_name = ann["name"].split("/")[-1] + ".mp4"
        gloss = ann["gloss"]
        text = ann["text"]
        signer = ann["signer"]

        video_path = os.path.join(video_dir, video_name)
        out_path = os.path.join(save_dir, video_name.replace(".mp4", ".npy"))

        if not os.path.exists(video_path):
            print(f"⚠️ Missing video: {video_name}")
            continue

        try:
            landmarks = extract_landmarks_from_video(video_path, visualize=visualize)
            np.save(out_path, landmarks)
            records.append({
                "video": video_name,
                "signer": signer,
                "gloss": gloss,
                "text": text,
                "npy_path": out_path.replace("\\", "/"),
                "frames": landmarks.shape[0]
            })
        except Exception as e:
            print(f"⚠️ Error processing {video_name}: {e}")

    # Save metadata CSV
    df = pd.DataFrame(records)
    meta_path = os.path.join(save_dir, f"{split_name}_metadata.csv")
    df.to_csv(meta_path, index=False)
    print(f"✅ Saved metadata CSV for {split_name}: {meta_path}")
    return df


def load_annotations(gzip_path):
    """Load PHOENIX dataset annotation file."""
    with gzip.open(gzip_path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    base_video_dir = r"D:\Graduate Project\F\data\videos_phoenix\videos"
    base_save_dir = r"D:\Graduate Project\F\data\landmarks"

    # Load all splits
    splits = {
        "train": load_annotations(r"D:\Graduate Project\F\data\phoenix14t.pami0.train.annotations_only.gzip"),
        "dev":   load_annotations(r"D:\Graduate Project\F\data\phoenix14t.pami0.dev.annotations_only.gzip"),
        "test":  load_annotations(r"D:\Graduate Project\F\data\phoenix14t.pami0.test.annotations_only.gzip"),
    }

    for split, annotations in splits.items():
        process_split(split,
                      os.path.join(base_video_dir, split),
                      os.path.join(base_save_dir, split),
                      annotations,
                      visualize=True)

