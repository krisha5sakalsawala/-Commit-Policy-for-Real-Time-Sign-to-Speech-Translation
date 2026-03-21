# -Commit-Policy-for-Real-Time-Sign-to-Speech-Translation

# AI for Accessibility: Real-Time Sign-to-Speech Translation

## 📌 Overview
This project presents a **real-time Sign-to-Speech translation system** designed to improve accessibility for Deaf and Hard-of-Hearing individuals. Unlike traditional systems that wait for complete video input, our approach processes sign language **in a streaming manner**, generating speech output with extremely low latency (~0.12 seconds).

The key innovation of this project is a **Commit Policy Framework**, which determines when partial predictions are stable enough to be emitted, enabling real-time interaction.

---

## 🚀 Key Features
- 🔄 Real-time streaming Sign-to-Speech pipeline  
- ⚡ Ultra-low latency (~0.12s end-to-end)  
- 🧠 BiLSTM-CTC based Continuous Sign Language Recognition (CSLR)  
- 🎯 Commit Policy for stable gloss emission  
- 🔤 Gloss-to-Text translation using Seq2Seq with Attention  
- 🔊 Text-to-Speech (TTS) integration using gTTS  
- 📊 Latency vs Accuracy trade-off analysis  
- 🧪 Evaluation on PHOENIX-2014T dataset  

---

## 🧠 System Architecture

The pipeline consists of four main modules:

1. **Sign → Gloss Recognition (CSLR)**
   - BiLSTM encoder with CTC loss
   - Frame-wise gloss probability prediction

2. **Commit Policy (Core Contribution)**
   - Determines when to emit gloss tokens
   - Uses:
     - Confidence threshold
     - Margin stability
     - Entropy filtering
     - Lookahead consistency

3. **Gloss → Text Translation**
   - Seq2Seq model with attention
   - Converts gloss segments into German sentences

4. **Text → Speech (TTS)**
   - gTTS for real-time audio generation

---

## 🔥 Novel Contribution

This project introduces a **commit-policy-based streaming framework for CSLR**, which:
- Enables **real-time gloss emission**
- Reduces latency from ~0.52s → ~0.12s
- Improves **segmentation and interpretability**
- Stabilizes predictions without retraining CSLR models

---

## 📊 Results

| Metric | Offline | Streaming (Proposed) |
|--------|--------|---------------------|
| Avg Latency | 0.52s | **0.12s** |
| SER | 0.60 | 0.77 |
| WER | 0.91 | 0.91 |
| Real-Time Capability | ❌ | ✅ |

📌 Key Insight:
- Slight drop in accuracy
- **Huge improvement in real-time usability**

---

## 📈 Commit Policy Details

A gloss is emitted when all conditions are satisfied:

- Confidence ≥ θ (0.45)
- Margin ≥ δ (0.01)
- Entropy ≤ Hmax (2.0)
- Temporal consistency (lookahead)

This ensures:
✔ Stability  
✔ Low delay  
✔ Reduced hallucinations  

---

## 📂 Dataset

- **PHOENIX-2014T**
  - German Sign Language dataset
  - Includes:
    - Video sequences
    - Gloss annotations
    - Spoken sentences

---

## 🛠️ Tech Stack

- Python 3.11
- PyTorch
- OpenCV
- MediaPipe (Holistic landmarks)
- NumPy / Pandas
- gTTS (Text-to-Speech)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/sign-to-speech.git
cd sign-to-speech

pip install -r requirements.txt
