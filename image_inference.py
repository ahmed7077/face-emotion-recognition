# ====================================================
# INFERENCE SCRIPT (VS Code Compatible)
# ====================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PATHS (EDIT)
# ----------------------------------------------------
MODEL_PATH = r"C:\Users\Ahmed\FER2013\efficientnetb0_fer_best.h5"
IMG_PATH = r"C:\Users\Ahmed\FER2013\test_images\happy.jpg"

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
assert os.path.exists(IMG_PATH), f"Image not found: {IMG_PATH}"

model = load_model(MODEL_PATH)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------
def preprocess_image(bgr_img, target_size=224):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size))
    arr = resized.astype(np.float32)
    arr = efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ----------------------------------------------------
# Load image
# ----------------------------------------------------
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise ValueError("Could not read image.")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

img_display = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)

# ----------------------------------------------------
# Predict
# ----------------------------------------------------
if len(faces) == 0:
    inp = preprocess_image(img_bgr)
    preds = model.predict(inp)
    cls = int(np.argmax(preds))
    prob = float(preds[0][cls])
    label = f"{EMOTIONS[cls]} ({prob:.2f})"
    print("Prediction:", label)

    plt.imshow(img_display)
    plt.title(label)
    plt.axis("off")
    plt.show()

else:
    for (x, y, w, h) in faces:
        face = img_bgr[y:y+h, x:x+w]
        inp = preprocess_image(face)
        preds = model.predict(inp)
        cls = int(np.argmax(preds))
        prob = float(preds[0][cls])
        label = f"{EMOTIONS[cls]} ({prob:.2f})"

        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_display, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    plt.imshow(img_display)
    plt.axis("off")
    plt.show()
