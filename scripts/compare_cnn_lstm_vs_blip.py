import os
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===== CNN–LSTM imports =====
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import DenseNet201

# ===== BLIP imports =====
from transformers import BlipProcessor, BlipForConditionalGeneration

# ===============================
# CONFIG
# ===============================

FRAMES_ROOT = r"C:\ML projects\Image captioning\msvd_progress_frames\content\msvd_progress_frames"
MAX_FRAMES = 8

CNN_MODEL_PATH = "baseline_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
METADATA_PATH = "metadata.pkl"

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ===============================
# LOAD RANDOM VIDEO
# ===============================

video_ids = [
    d for d in os.listdir(FRAMES_ROOT)
    if os.path.isdir(os.path.join(FRAMES_ROOT, d))
]

VIDEO_ID = random.choice(video_ids)
video_path = os.path.join(FRAMES_ROOT, VIDEO_ID)

frames = sorted(
    [f for f in os.listdir(video_path) if f.endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])
)[:MAX_FRAMES]

frame_paths = [os.path.join(video_path, f) for f in frames]

print(f"\n🎯 Selected video: {VIDEO_ID}")
print(f"🖼️ Using {len(frame_paths)} frames")

# ===============================
# LOAD CNN–LSTM
# ===============================

cnn_model = load_model(CNN_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(METADATA_PATH, "rb") as f:
    max_length = pickle.load(f)["max_length"]

# DenseNet feature extractor (same as training)
base_cnn = DenseNet201(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return base_cnn.predict(img, verbose=0)

index_word = {v: k for k, v in tokenizer.word_index.items()}

def predict_cnn_lstm(feature):
    text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], max_length)
        yhat = cnn_model.predict([feature, seq], verbose=0)
        word_id = np.argmax(yhat)
        word = index_word.get(word_id)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text.replace("startseq", "").replace("endseq", "").strip()

# ===============================
# LOAD BLIP
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL_NAME
).to(device)
blip_model.eval()

def predict_blip(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_length=20,
            num_beams=3,
            do_sample=False
        )
    return blip_processor.decode(output[0], skip_special_tokens=True)

# ===============================
# RUN COMPARISON
# ===============================

images = []
cnn_caps = []
blip_caps = []

for path in frame_paths:
    images.append(Image.open(path))

    feat = extract_features(path)
    cnn_caps.append(predict_cnn_lstm(feat))

    blip_caps.append(predict_blip(path))

## ===============================
# CLEAN COMPARISON VISUALIZATION
# ===============================

fig, axes = plt.subplots(len(images), 3, figsize=(16, 4 * len(images)))

for i in range(len(images)):
    # Image
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f"Frame {i}", fontsize=12)
    axes[i, 0].axis("off")

    # CNN–LSTM Caption
    axes[i, 1].text(
        0.5, 0.5,
        cnn_caps[i],
        fontsize=11,
        ha="center",
        va="center",
        wrap=True
    )
    axes[i, 1].set_title("CNN–LSTM", fontsize=12)
    axes[i, 1].axis("off")

    # BLIP Caption
    axes[i, 2].text(
        0.5, 0.5,
        blip_caps[i],
        fontsize=11,
        ha="center",
        va="center",
        wrap=True
    )
    axes[i, 2].set_title("BLIP (Transformer)", fontsize=12)
    axes[i, 2].axis("off")

plt.suptitle(
    f"CNN–LSTM vs BLIP Caption Comparison\nVideo ID: {VIDEO_ID}",
    fontsize=16
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

