import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from collections import defaultdict
import tempfile
import time

from scripts.cnn_lstm_predictor import CNNLSTMPredictor
from scripts.blip_predictor import BLIPPredictor
from utils.error_analyzer import ErrorAnalyzer
from utils.temporal_analyzer import TemporalAnalyzer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Error-Centric Video Captioning Analysis",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Error-Centric Video Captioning Analysis")
st.caption("CNN-LSTM vs Attention-LSTM vs Transformer (BLIP)")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    return {
        "CNN-LSTM": CNNLSTMPredictor(
            "models/baseline_model.h5",
            "models/baseline_tokenizer.pkl",
            "models/baseline_metadata.pkl"
        ),
        "BLIP": BLIPPredictor()
    }

models = load_models()

# ------------------ SIDEBAR ------------------
st.sidebar.header("📁 Upload Frames")
uploaded_files = st.sidebar.file_uploader(
    "Upload all frame images (8 frames)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

run_btn = st.sidebar.button("🚀 Run Analysis")

# ------------------ UTILS ------------------
def sort_frames(files):
    return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x.name)) or 0))

# ------------------ MAIN PIPELINE ------------------
if uploaded_files and run_btn:

    frames = sort_frames(uploaded_files)
    analyzer = ErrorAnalyzer()
    temporal = TemporalAnalyzer()

    results = []
    error_counts = defaultdict(lambda: defaultdict(int))
    processing_times = defaultdict(list)

    progress = st.progress(0)

    for idx, file in enumerate(frames):
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        captions = {}
        errors = {}

        for model_name, model in models.items():
            start = time.time()
            caption = model.generate(image_np)
            processing_times[model_name].append(time.time() - start)

            captions[model_name] = caption
            detected_errors = analyzer.analyze(caption)
            errors[model_name] = detected_errors

            for e in detected_errors:
                error_counts[model_name][e] += 1

        temporal.add_frame(idx, captions)

        results.append({
            "frame": idx,
            "image": image,
            "captions": captions,
            "errors": errors
        })

        progress.progress((idx + 1) / len(frames))

    st.success("✅ Analysis Complete")

    # ================= DASHBOARD =================
    tab1, tab2, tab3 = st.tabs([
        "📊 Executive Summary",
        "🖼 Frame-wise Comparison",
        "⚠ Error Taxonomy"
    ])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("Executive Summary")

        metrics = []
        total_frames = len(results)

        for model in models:
            total_errors = sum(error_counts[model].values())
            metrics.append({
                "Model": model,
                "Error Rate": round(total_errors / total_frames, 3),
                "Avg Time (s)": round(np.mean(processing_times[model]), 3)
            })

        df_metrics = pd.DataFrame(metrics)
        st.dataframe(df_metrics, use_container_width=True)

        fig = px.bar(
            df_metrics,
            x="Model",
            y="Error Rate",
            color="Model",
            title="Overall Error Rate (Lower is Better)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("Frame-wise Caption Comparison")

        for r in results:
            st.markdown(f"### Frame {r['frame']}")
            st.image(r["image"], width=300)

            cols = st.columns(len(models))
            for i, model in enumerate(models):
                with cols[i]:
                    st.markdown(f"**{model}**")
                    st.write(r["captions"][model])
                    if r["errors"][model]:
                        st.error(", ".join(r["errors"][model]))
                    else:
                        st.success("No major errors")

    # ---------- TAB 3 ----------
    with tab3:
        st.subheader("Error Taxonomy Analysis")

        taxonomy = []
        for model, errs in error_counts.items():
            for etype, count in errs.items():
                taxonomy.append({
                    "Model": model,
                    "Error Type": etype.replace("_", " ").title(),
                    "Count": count
                })

        df_tax = pd.DataFrame(taxonomy)

        fig = px.bar(
            df_tax,
            x="Error Type",
            y="Count",
            color="Model",
            barmode="group",
            title="Error Distribution by Model"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Lower counts indicate better grounding, temporal consistency, "
            "and fewer hallucinations."
        )

else:
    st.info("👈 Upload 8 video frames to begin analysis")
