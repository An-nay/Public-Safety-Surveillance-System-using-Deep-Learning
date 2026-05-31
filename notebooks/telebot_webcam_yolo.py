import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import telepot
from datetime import datetime
import pytz
from ultralytics import YOLO

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID   = YOUR_CHAT_ID_HERE
LOCATION  = "YOUR_LOCATION_WITH_COORDINATES_HERE"

def getTime():
    AEST = pytz.timezone('Australia/Sydney')
    return datetime.now(AEST)

def print_results(video=None, webcam=False):
    trueCount = 0
    imageSaved = 0
    filename = 'savedImage.jpg'
    print("Loading model ...")
    model = load_model('modelnew.h5', compile=False)
    Q = deque(maxlen=128)

    # YOLOv8 human detector — downloads automatically on first run
    yolo = YOLO('yolov8n.pt')  # nano version, fastest for M1

    if webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Running... Press Ctrl+C to stop.")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame.")
                break

            frame_count += 1

            # YOLOv8 human detection gate
            results = yolo(frame, classes=[0], verbose=False)  # class 0 = person
            humans_detected = len(results[0].boxes) > 0

            if not humans_detected:
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count} | No human detected | ViolentCount: {trueCount}")
                continue

            # match training preprocessing exactly
            frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_resized, (128, 128))
            frame_resized = frame_resized.astype('float32') / 255.0
            output = cv2.resize(frame, (128, 128))

            preds = model.predict(np.expand_dims(frame_resized, axis=0), verbose=0)[0]
            print(f"pred: {preds[0]:.4f}")

            Q.append(preds)
            label = bool((preds > 0.90)[0])
            if label:
                trueCount += 1

            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | Violence: {label} | ViolentCount: {trueCount}")

            if trueCount == 200:
                print("VIOLENCE DETECTED — sending Telegram alert...")
                if imageSaved == 0:
                    cv2.imwrite(filename, output)
                    imageSaved = 1
                timeMoment = getTime()
                bot = telepot.Bot(BOT_TOKEN)
                bot.sendMessage(CHAT_ID, f"VIOLENCE ALERT!!\nLOCATION: {LOCATION}\nTIME: {timeMoment}")
                bot.sendPhoto(CHAT_ID, photo=open(filename, 'rb'))
                print("Alert sent!")
                trueCount = 0
                imageSaved = 0

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("[INFO] cleaning up...")
        cap.release()

print_results(webcam=True)
