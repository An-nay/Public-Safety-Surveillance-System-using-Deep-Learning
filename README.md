# Public Safety Surveillance System using Deep Learning

A real-time violence detection system built with MobileNetV2 and YOLOv8. Detects violent behaviour in live webcam feeds and pre-recorded videos, sending automated Telegram alerts with a snapshot, location, and timestamp.

Published Paper: "Public Safety Surveillance System using Deep Learning", IEEE 2024  
Institution: BMS College of Engineering, Bangalore  
Authors: Ankita M, Garima Prajapati, Ananya Srinivas, Anurag Soni  

---

## Table of Contents

- [Overview](#overview)
- [What Changed from v1](#what-changed-from-v1)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Telegram Bot Setup](#telegram-bot-setup)
- [Future Scope](#future-scope)
- [Citation](#citation)

---

## Overview

This system detects violent behaviour in video streams, both pre-recorded and live webcam. It uses a fine-tuned MobileNetV2 classifier to label each frame as violent or non-violent. A human detection gate (YOLOv8 for webcam, Faster R-CNN for video) filters out frames with no humans before running the violence classifier, significantly reducing false positives.

When violence is detected across enough frames, the system sends a Telegram alert with a snapshot, location, and timestamp.

Key features:
- Binary Violence / Non-Violence classification per frame using MobileNetV2
- YOLOv8 human detection gate for real-time webcam use
- Faster R-CNN human detection gate for video file processing
- Real-time webcam support via local Python script
- Automated Telegram bot alerts with image snapshot
- Video-level train/test split to prevent data leakage
- 90% accuracy with no data leakage (improved model)

---

## What Changed from v1

| Area | v1 (Original) | v2 (Current) |
|------|--------------|--------------|
| Training split | Frame-level (caused data leakage, 96% accuracy) | Video-level (no leakage, 90% accuracy) |
| Human detection (webcam) | Not integrated | YOLOv8 nano gate |
| Human detection (video) | Standalone notebook only | Faster R-CNN integrated in prediction notebook |
| Webcam script | Jupyter notebook only | Standalone `.py` script with virtual environment |
| Alert trigger | 50 consecutive violence frames | 200 frames above 0.90 confidence (tunable) |
| Preprocessing | BGR frame fed directly | BGR→RGB conversion before resize, matching training exactly |
| Model loading | `load_model()` | `load_model(..., compile=False)` to handle version differences |
| Training dataset size | 350 videos per class | 700 videos per class (80/20 split) |
| Model architecture head | Single sigmoid on GAP | GAP → Dropout(0.5) → Dense(256) → Dropout(0.3) → sigmoid |

---

## Project Structure

```
public-safety-surveillance/
│
├── notebooks/
│   ├── full-pipeline-v-nv-detection.ipynb        ← combined training + RCNN gate + prediction
│   ├── telebot_webcam_yolo.py                     ← NEW: webcam script with YOLOv8 gate
│
├── models/
│   └── modelnew.h5                                ← trained MobileNetV2 model
│
├── paper/
│   └── InCCCS202456.pdf
│
├── requirements_kaggle.txt                         ← for kaggle/video pipeline
├── requirements_webcam.txt                        ← for local webcam script
└── README.md
```

---

## Pipeline

```
1. full-pipeline-v-nv-detection.ipynb
   ├── Data preprocessing (video-level split, augmentation)
   ├── Faster R-CNN human gate
   ├── MobileNetV2 training
   └─► modelnew.h5
              │
              ▼
2. v-nv-prediction-public-safety-surveillance-system.ipynb
   (test on video files, annotated output)
              │
              ▼
3. telebot_webcam_yolo.py  (LOCAL)
   (live webcam + YOLOv8 human gate + Telegram alerts)
```

---

## Notebooks

| # | File | Purpose | Input | Output |
|---|------|---------|-------|--------|
| 1 | `full-pipeline-v-nv-detection.ipynb` | Combined pipeline: data preprocessing with video-level split, Faster R-CNN human gate, MobileNetV2 training, evaluation | Real Life Violence Dataset (Kaggle) | `modelnew.h5` |
| 2 | `v-nv-prediction-public-safety-surveillance-system.ipynb` | Run predictions on video files, count violent vs non-violent frames, produce annotated output video | `modelnew.h5`, test `.mp4` files | Annotated `v_output.avi`, frame counts |
| 3 | `telebot_webcam_yolo.py` | Live webcam feed with YOLOv8 human detection gate + violence classification + recurring Telegram alerts | `modelnew.h5`, webcam | Telegram alerts with snapshot |

---

## Dataset

Dataset: [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), available on Kaggle.

Expected structure after download:
```
Real Life Violence Dataset/
├── Violence/       <- 1000 .mp4 video files
└── NonViolence/    <- 1000 .mp4 video files
```

- 700 videos per class used (1400 total), 80/20 video-level train/test split
- Frames extracted every 7th frame to avoid duplication
- Augmentations applied to training frames only: horizontal flip, zoom (1.3x), random brightness, rotation (±25°)
- Frames resized to 128×128 pixels, normalised to [0, 1]
- Video-level split prevents frames from the same video appearing in both train and test sets

---

## Model Performance

### v2 Model (no data leakage) — recommended

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NonViolence | 0.88 | 0.90 | 0.89 | 2415 |
| Violence | 0.91 | 0.90 | 0.91 | 2940 |
| **Accuracy** | | | **0.90** | **5355** |
| Macro avg | 0.90 | 0.90 | 0.90 | 5355 |
| Weighted avg | 0.90 | 0.90 | 0.90 | 5355 |

No data leakage 

### v1 Model (frame-level split, data leakage) — not recommended

| Metric | Value |
|--------|-------|
| Accuracy | 96% |
| F1 Score | 0.985 |
| Precision | 0.987 |
| Recall | 0.984 |
| Data leakage | Yes |

Architecture: MobileNetV2 (ImageNet weights, frozen base, last 30 layers unfrozen) + GAP → Dropout(0.5) → Dense(256, relu, L2) → Dropout(0.3) → Dense(1, sigmoid)  
Training: 25 epochs, early stopping, AdamW optimiser, label smoothing, batch size 4

---

## Setup and Installation

### Kaggle notebooks (training + video prediction)

All training and video prediction notebooks run on Kaggle with GPU runtime (T4 x2).

```bash
pip install tensorflow keras opencv-python albumentations imutils
```

Or use the provided file:
```bash
pip install -r requirements_kaggle.txt
```

### Local webcam script (Apple Silicon Mac)

Requires Python 3.11 via pyenv. Use a virtual environment to avoid numpy conflicts.

```bash
cd notebooks
python -m venv venv_violence
source venv_violence/bin/activate
pip install -r requirements_webcam.txt
```

> **Note for Apple Silicon (M1/M2/M3):** Use `tensorflow-macos` and `tensorflow-metal` instead of standard `tensorflow`. Do not install `tf_keras` — it pulls in a newer tensorflow that breaks numpy compatibility.

---

## How to Run

### Notebook 1 — Full Training Pipeline (Kaggle)

1. Upload the Real Life Violence Dataset to Kaggle as a dataset input
2. Upload the extracted Faster R-CNN model folder as a Kaggle dataset input
3. Update the `VIDEO_DIR` and `RCNN_MODEL` paths at the top of `full-pipeline-v-nv-detection.ipynb`
4. Open the notebook on Kaggle with GPU runtime (Settings → Accelerator → GPU T4 x2)
5. Run all cells — training takes ~30–60 min on GPU
6. Download `modelnew.h5` from the `/kaggle/working/` output directory

### Notebook 2 — Predict on Videos (Kaggle)

1. Add `modelnew.h5` as a Kaggle dataset input, or upload it directly
2. Upload test videos as a Kaggle dataset input
3. Update video paths in the notebook
4. Run all cells — outputs annotated `v_output.avi` and frame counts to `/kaggle/working/output/`

### Webcam Script — Live Detection (Local, Apple Silicon)

1. Place `modelnew.h5` in the same folder as `telebot_webcam_yolo.py`
2. Activate the virtual environment:
   ```bash
   cd notebooks
   source venv_violence/bin/activate
   ```
3. Update credentials in the script:
   ```python
   BOT_TOKEN = "your_bot_token_here"
   CHAT_ID   = your_chat_id_here   # integer, no quotes
   LOCATION  = "Your location string"
   ```
4. Run:
   ```bash
   python telebot_webcam_yolo.py
   ```
5. Press **Ctrl+C** to stop

**Tunable parameters:**
```python
label = bool((preds > 0.90)[0])  # confidence threshold per frame
if trueCount == 200:              # number of violent frames to trigger alert
```

**Fallback (if ultralytics causes numpy conflicts):**
```bash
python telebot_webcam.py   # uses HOG instead of YOLOv8
```

---

## Telegram Bot Setup

> **Before running the webcam script:** place `modelnew.h5` in the **same folder** as `telebot_webcam_yolo.py`. The script loads it from the current directory with `load_model('modelnew.h5')`.

1. Open Telegram, search `@BotFather`, send `/newbot`
2. Follow the prompts — copy your **Bot Token**
3. Open your bot in Telegram and press **Start** (required before the bot can message you)
4. Get your Chat ID:
   ```bash
   python -c "
   import telepot, json
   bot = telepot.Bot('YOUR_TOKEN')
   print(json.dumps(bot.getUpdates(), indent=2))
   "
   ```
   Look for `"chat": {"id": ...}` in the output
5. Set credentials in the script — `CHAT_ID` must be an integer with no quotes

---

## Future Scope

- Multi-modal analysis combining audio and video
- Real-time streaming from IP/CCTV cameras
- Face blurring for privacy-preserving detection
- Anomaly detection beyond binary classification
- Integration with law enforcement APIs

---

## Citation

```
Ankita M, Garima Prajapati, Ananya Srinivas, Anurag Soni, Dr. Manjunath P S,
"Public Safety Surveillance System using Deep Learning",
IEEE, 2024. DOI: 979-8-3503-5885-8/24
```

---

## Changes Made from the Published Paper

The published paper (IEEE 2024) describes the original v1 implementation. This repository contains an improved v2. Key differences:

**Running on Kaggle instead of Google Colab**
The original paper used Google Colab. This implementation runs on Kaggle, which provides higher free storage limits and more reliable GPU runtimes (T4 × 2), making it better suited for the full 700-video-per-class dataset.

**R-CNN Integration (previously standalone code)**
In the original paper, the Faster R-CNN human detection component was a separate, disconnected notebook. It was never wired into the violence detection pipeline. In v2, human detection is integrated directly as a gating pre-filter — every frame passes through R-CNN first, and MobileNetV2 only runs if a human is detected. This eliminates false-positive violence detections on empty scenes.

**Resolved version conflicts from R-CNN integration**
Integrating TF1-based Faster R-CNN (which uses `tensorflow.compat.v1`) alongside TF2-based MobileNetV2 in the same notebook caused session and graph conflicts. These were resolved by isolating the TF1 session inside the `DetectorAPI` class and using `tf.disable_v2_behavior()` scoped to that graph only.

---

## License

This project is for academic and research purposes. Dataset usage is subject to the original [Kaggle dataset license](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset).
