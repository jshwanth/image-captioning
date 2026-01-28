import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

# ===============================
# CONFIG
# ===============================

FRAMES_ROOT = "C:\\ML projects\\Image captioning\\data\\msvd_progress_frames"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_FRAMES = 8        # only 8 frames (efficient + research-friendly)
MAX_LENGTH = 20       # shorter captions = faster + cleaner
NUM_BEAMS = 3         # deterministic decoding

# ===============================
# LOAD MODEL
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
torch.set_grad_enabled(False)

# ===============================
# SELECT RANDOM VIDEO
# ===============================

video_ids = [
    d for d in os.listdir(FRAMES_ROOT)
    if os.path.isdir(os.path.join(FRAMES_ROOT, d))
]

if not video_ids:
    raise RuntimeError("❌ No video folders found in FRAMES_ROOT")

VIDEO_ID = random.choice(video_ids)
video_path = os.path.join(FRAMES_ROOT, VIDEO_ID)

print(f"🎯 Selected video: {VIDEO_ID}")

# ===============================
# LOAD FRAMES (TEMPORAL ORDER)
# ===============================

frames = sorted(
    [f for f in os.listdir(video_path) if f.endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])
)[:MAX_FRAMES]

frame_paths = [os.path.join(video_path, f) for f in frames]

print(f"🖼️ Processing {len(frame_paths)} frames...")

# ===============================
# BATCH CAPTION GENERATION (FAST)
# ===============================

images = [Image.open(p).convert("RGB") for p in frame_paths]

inputs = processor(images=images, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        num_beams=NUM_BEAMS,
        do_sample=False
    )

captions = processor.batch_decode(output_ids, skip_special_tokens=True)
captions = [c.strip() for c in captions]

# ===============================
# PRINT RESULTS
# ===============================

for i, cap in enumerate(captions):
    print(f"Frame {i}: {cap}")

# ===============================
# VISUALIZATION
# ===============================

fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 5))

if len(images) == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.set_title(f"Frame {i}\n{captions[i]}", fontsize=10)
    ax.axis("off")

plt.suptitle(
    f"BLIP Frame-Level Captioning (MSVD)\nVideo ID: {VIDEO_ID}",
    fontsize=14
)

plt.tight_layout()
plt.show()
