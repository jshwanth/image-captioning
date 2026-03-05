# Image Captioning Analysis on Video Frames

A comparative study of **CNN-LSTM, Attention-based, and Transformer (BLIP)** models for generating captions on video frames.

This project evaluates how well captioning models trained on **image datasets (Flickr8k)** generalize to **video frames extracted from the MSVD dataset**, and analyzes their performance using both **metrics and qualitative error analysis**.

---

## Models Compared

1. **CNN–LSTM (Baseline)**
   - DenseNet201 feature extractor
   - LSTM decoder

2. **Attention-Based Model**
   - CNN encoder
   - Bahdanau attention
   - GRU decoder

3. **Transformer Model**
   - BLIP Vision-Language model
   - Pretrained multimodal transformer

---

## Dataset

- **Training:** Flickr8k  
- **Evaluation:** MSVD video dataset

Instead of full videos, **8 frames are sampled per video** to analyze caption quality across time.

---

## Evaluation Metrics

The generated captions are evaluated using:

- **BLEU-4**
- **METEOR**
- **ROUGE-L**

Example results:

| Model | BLEU-4 | METEOR | ROUGE-L |
|------|------|------|------|
| CNN-LSTM | 0.192 | 0.543 | 0.558 |
| Attention | 0.189 | 0.416 | 0.480 |
| BLIP | **0.396** | **0.699** | **0.669** |

Transformer-based models show significantly better caption quality.

---

## Dashboard

A **Streamlit dashboard** was built to visualize the model outputs.

Features:

- Random video frame sampling
- Caption generation using all models
- Frame-by-frame metric comparison
- Temporal performance plots
- Radar and distribution charts

Run the dashboard:

```bash
streamlit run scripts/dashboard.py
```

## Project Structure
```bash
project/
│
├── data/
│   └── msvd_progress_frames
│
├── models/
│
├── scripts/
│   ├── model_inference.py
│   └── dashboard.py
│
└── README.md
```
## Technologies Used

 - Python
 - TensorFlow / Keras
 - PyTorch
 - HuggingFace Transformers
 - Streamlit
 - Plotly

 ## Results
 ![alt text](<data/Screenshot 2026-03-05 194228.png>)
 ![alt text](<data/Screenshot 2026-03-05 194308.png>)



