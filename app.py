

import streamlit as st
st.cache_resource.clear()
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Image Captioning", layout="centered")

# =========================
# LOAD SAVED FILES
# =========================

@st.cache_resource
def load_model_and_assets():
    model = tf.keras.models.load_model("baseline_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    return model, tokenizer, meta["max_length"]

caption_model, tokenizer, max_length = load_model_and_assets()

# =========================
# FEATURE EXTRACTOR (SAME AS TRAINING)
# =========================

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

@st.cache_resource
def load_feature_extractor():
    base_model = DenseNet201()
    return Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output
    )

feature_extractor = load_feature_extractor()

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    return feature_extractor.predict(img)

# =========================
# CAPTION GENERATION
# =========================

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, feature, tokenizer, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# =========================
# STREAMLIT UI
# =========================

st.title("🖼️ Image Caption Generator")
st.caption("Baseline CNN + DenseNet201 + LSTM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        feature = extract_features("temp.jpg")
        caption = predict_caption(caption_model, feature, tokenizer, max_length)

    st.success(caption)
