# An Error-Centric Comparative Analysis of CNN–LSTM, Attention-Based, and Transformer Models for Frame-Level Video Captioning

## 📌 Overview

Video captioning is a challenging problem in computer vision that requires understanding objects, actions, and context over time. This project presents a **comparative, error-centric analysis** of three captioning paradigms:

* **CNN–LSTM**: A baseline image captioning model.
* **CNN–Attention–LSTM**: An architecture with improved spatial reasoning.
* **Transformer-based Vision–Language Model (BLIP)**: A modern patch-level attention model.

The models are evaluated on **video frames extracted from the MSVD dataset**, while the baseline is trained on **Flickr8k**, enabling an analysis of **cross-dataset generalization**, temporal logic, and architectural limitations.

## 🎯 Key Contributions

* **Comparative Analysis**: Evaluates how different architectures handle "Progress-Aware" frames.
* **Dataset Bias Study**: Demonstrates how models trained on static images (Flickr8k) generalize to video frames (MSVD).
* **Spatial Grounding**: Shows how attention mechanisms improve object localization.
* **Error Taxonomy**: Defines specific temporal errors like "Premature Prediction" and "Temporal Stagnation."
* **Visual Benchmarking**: Provides side-by-side visual comparisons on real video frames.

## 🧠 Models Compared

### 1️⃣ CNN–LSTM (Baseline)

* **Encoder**: CNN (DenseNet/ResNet).
* **Bottleneck**: Single global feature vector.
* **Decoder**: LSTM-based language decoder.
* **Characteristics**: Often suffers from a representational bottleneck and lacks spatial/temporal reasoning.

### 2️⃣ CNN–Attention–LSTM

* **Mechanism**: Spatial feature maps from CNN with soft attention over image regions.
* **Improvement**: Provides better alignment between vision and language by "looking" at specific objects during word generation.

### 3️⃣ Transformer-Based Model (BLIP)

* **Architecture**: Patch-level vision tokens with cross-modal attention.
* **Strengths**: Pretrained on large-scale vision–language data, producing semantically rich and accurate descriptions.

## 📊 Evaluation Metrics

We utilize standard NLP metrics for quantitative assessment:

* **BLEU-1 / BLEU-4**
* **METEOR**
* **CIDEr**

Evaluation is conducted across two domains:

1. **In-Domain**: Flickr8k Test Set.
2. **Out-of-Distribution**: MSVD progress-aware video frames.

## 🔍 Error-Centric Analysis (The "Why")

Unlike standard benchmarks, this study categorizes *why* models fail:

* **Object Hallucination**: Describing objects not present in the frame.
* **Temporal Stagnation**: Repeating the same caption for all frames in a video sequence.
* **Premature Prediction**: Describing an action (e.g., "ice breaking") before it happens.
* **Attribute Errors**: Incorrect colors, counts, or sizes.
* **Action Ambiguity**: Using generic verbs instead of precise action verbs.
