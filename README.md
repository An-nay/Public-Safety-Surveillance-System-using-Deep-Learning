# Public Safety Surveillance System using Deep Learning

A real-time violence detection system built with MobileNetV2 and Faster R-CNN. Sends automated Telegram alerts when violence is detected.

Published Paper: "Public Safety Surveillance System using Deep Learning", IEEE 2024
Institution: BMS College of Engineering, Bangalore
Authors: Ankita M, Garima Prajapati, Ananya Srinivas, Anurag Soni
Guide: Dr. Manjunath P S

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Telegram Bot Setup](#telegram-bot-setup)
- [Future Scope](#future-scope)

---

## Overview

This system detects violent behaviour in video streams, both pre-recorded and live webcam. It uses a fine-tuned MobileNetV2 classifier to label each frame as violent or non-violent. When violence is detected across 50 consecutive frames, the system sends a Telegram alert to a configured group with the timestamp, location, and a snapshot.

Key features:
- Binary Violence / Non-Violence classification per frame
- Real-time webcam support
- Automated Telegram bot alerts with image snapshot
- Human detection using Faster R-CNN (Inception V2)
- 96% accuracy, F1 Score of 0.985

---

## Project Structure

```
public-safety-surveillance/
│
├── notebooks/
│   ├── Model_training-2.ipynb
│   ├── Human_Detection.ipynb
│   ├── V_NV_Prediction.ipynb
│   ├── Alert_System.ipynb
│   └── Telebot_webcam_integration.ipynb
│
├── models/
│   └── modelnew.h5
│
├── paper/
│   └── PID56.pdf
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
1. Model_training-2.ipynb
   └─► modelnew.h5
           │
           ▼
2. V_NV_Prediction.ipynb         3. Human_Detection.ipynb
   (test on video files)            (standalone human detection)
           │
           ▼
4. Telebot_webcam_integration.ipynb
   (live webcam + Telegram alerts)
```

Human_Detection.ipynb runs independently. It is not connected to the alert pipeline. It was built as a human presence pre-filter and is planned for future integration.

---

## Notebooks

| # | Notebook | Purpose | Input | Output |
|---|----------|---------|-------|--------|
| 1 | Model_training-2.ipynb | Preprocess dataset, train MobileNetV2 classifier | Real Life Violence Dataset (Google Drive) | modelnew.h5 |
| 2 | Human_Detection.ipynb | Detect humans in video using Faster R-CNN | frozen_inference_graph.pb, test video | Annotated frames |
| 3 | V_NV_Prediction.ipynb | Run predictions on video files | modelnew.h5, test .mp4 files | Annotated v_output.avi |
| 4 | Alert_System.ipynb | v1 prototype. Violence detection on pre-recorded video + one-time Telegram alert | modelnew.h5, test .mp4 | Alert message + snapshot |
| 5 | Telebot_webcam_integration.ipynb | v2 upgrade. Replaced pre-recorded video with live webcam feed. Alert resets and re-triggers every 50 violence frames. Moved from Colab to local execution | modelnew.h5, webcam | Live annotated feed + recurring alerts |

---

## Dataset

Dataset: [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), available on Kaggle.

Expected structure after download:
```
Real Life Violence Dataset/
├── Violence/       <- 1000 .mp4 video files
└── NonViolence/    <- 1000 .mp4 video files
```

- Frames extracted every 7th frame to avoid duplication
- Augmentations: horizontal flip, zoom (1.3x), random brightness, rotation (+/-25 degrees)
- Frames resized to 128x128 pixels
- 350 videos per class used for training (700 total) due to memory constraints
- 70/30 train-test split using Stratified Shuffle Split

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 96% |
| F1 Score | 0.985 |
| Precision | 0.987 |
| Recall | 0.9839 |

Architecture: MobileNetV2 (frozen base) + Dense sigmoid head
Training: 25 epochs max, early stopping, learning rate warm-up + decay, batch size 4

---

## Setup and Installation

### Colab notebooks (Notebooks 1 to 4)

All notebooks run on Google Colab with GPU runtime. No local setup needed beyond uploading files.

### Local webcam notebook (Notebook 5)

```bash
pip install tensorflow keras opencv-python telepot geopy pytz
```

---

## How to Run

### Notebook 1 - Train the Model (Google Colab)

1. Upload the dataset to Google Drive at:
   ```
   MyDrive/6th sem mms/archive/Real Life Violence Dataset/
   ```
2. Open Model_training-2.ipynb in Colab with GPU runtime
3. Mount Google Drive when prompted
4. Run all cells
5. Save modelnew.h5 from /kaggle/working/RiskDetection/ to /content/

### Notebook 2 - Human Detection (Google Colab)

Download faster_rcnn_inception_v2_coco_2018_01_28.tar from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and extract frozen_inference_graph.pb to /content/ before running Human_Detection.ipynb.

1. Upload faster_rcnn_inception_v2_coco_2018_01_28.tar to Colab
2. Extract it:
   ```python
   import tarfile
   tarfile.open('faster_rcnn_inception_v2_coco_2018_01_28.tar').extractall('/content/')
   ```
3. Upload your test video as Testing video.mp4 to /content/
4. Run all cells

### Notebook 3 - Predict on Videos (Google Colab)

1. Confirm modelnew.h5 is at /content/modelnew.h5
2. Upload test videos V_19.mp4 and NV_2.mp4 to /content/
3. Run all cells

### Notebook 4 - Alert System (Google Colab)

1. Confirm modelnew.h5 is at /content/modelnew.h5
2. Upload test video V_1.mp4 to /content/
3. Update your Telegram credentials (see Telegram Bot Setup below)
4. Run all cells. The alert fires after 50 consecutive violence frames.

### Notebook 5 - Live Webcam + Alerts (Local)

1. Place modelnew.h5 in the same directory as the notebook
2. Update your Telegram credentials and location string
3. Run locally. This does not work in Colab as it requires webcam access:
   ```bash
   jupyter notebook Telebot_webcam_integration.ipynb
   ```
   or
   ```bash
   jupyter nbconvert --to script Telebot_webcam_integration.ipynb
   python Telebot_webcam_integration.py
   ```
4. Press q to stop the webcam feed

---

## Telegram Bot Setup

1. Open Telegram, search @BotFather, and send /newbot
2. Follow the prompts to name your bot and get your Bot Token
3. Create a group, add your bot, and send a message
4. Visit the following URL to get your Chat ID:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
5. Find "chat":{"id":...}. This is your Chat ID. It is a negative number for groups.
6. Replace the credentials in the notebook:
   ```python
   bot = telepot.Bot('YOUR_BOT_TOKEN_HERE')
   bot.sendMessage(YOUR_CHAT_ID_HERE, ...)
   ```
7. Update the location string to your actual location:
   ```python
   location = 'Your Location\n LAT° N, LON° E'
   ```

Never commit your bot token or chat ID to a public repository. Use environment variables or a .env file instead.

---

## Future Scope

- Add Human Detection as a pre-filter before violence classification
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

## License

This project is for academic and research purposes. Dataset usage is subject to the original [Kaggle dataset license](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset).
