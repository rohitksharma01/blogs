---
layout: default
title: Building a QR Code Detector and Decoder with TensorFlow
---

# Building a QR Code Detector and Decoder with TensorFlow

[← Back to Home](./index.md)

## Table of Contents
1. [Data Generation](#1-data-generation)
2. [Model Architecture](#2-model-architecture)
3. [Training Pipeline](#3-training-pipeline)
4. [Inference and Decoding (Static Images)](#4-inference-and-decoding-static-images)
5. [Decoding QR Codes from GIFs](#5-decoding-qr-codes-from-gifs)
6. [Results and Observations](#6-results-and-observations)
7. [Results Gallery](#7-results-gallery)
8. [Tips for Robustness](#8-tips-for-robustness)
9. [Try It Yourself](#9-try-it-yourself)

---

This post walks through how we:
- Generated a large synthetic dataset of QR and non-QR images
- Trained a TensorFlow model to detect QR presence
- Decoded QR codes (including from animated GIFs) using OpenCV and pyzbar
- Applied multiple strategies to robustly decode difficult GIF QR codes

## Technologies & Libraries Used

The project leverages the following key libraries:

**Machine Learning & Computer Vision:**
- **TensorFlow** (≥2.14): Deep learning framework for building and training the QR detection CNN
- **OpenCV** (≥4.8): Computer vision library for image processing and QR code detection/decoding
- **NumPy** (≥1.24): Numerical computing for array operations and image manipulation
- **scikit-learn** (≥1.3): Machine learning utilities for data splitting and preprocessing

**QR Code Generation & Decoding:**
- **qrcode[pil]** (≥7.4): QR code generation with PIL integration for creating diverse training datasets
- **pyzbar**: Advanced barcode/QR code decoding library with robust detection capabilities
- **Pillow** (≥10.0): Image processing library for handling various image formats including animated GIFs

**Visualization & Utilities:**
- **matplotlib** (≥3.8): Plotting and visualization for training metrics and results
- **tqdm** (≥4.66): Progress bars for long-running operations

You can install all dependencies with:
```bash
pip install -r requirements.txt
```

The full project lives in this workspace. Key scripts:
- `src/generate_qr.py`: QR generator with color, transparency, and background styles
- `src/generate_non_qr.py`: Non-QR negative sample generator
- `src/prepare_dataset.py`: Train/val split script
- `src/model.py`: CNN architecture
- `src/train.py`: Training pipeline
- `src/infer_and_decode.py`: Image-level detection + decoding
- `src/infer_gif.py`: GIF cumulative overlays + detection/decoding
- `src/decode_animated_qr.py`: Multi-strategy GIF QR decoder

## 1. Data Generation

This is the most important bit, you need a lot of QRCodes to train your models. You can try doanloading as much as you want, but I wouldnt go that route. I asked AI to generate loads of QRcodes and non qr code images. 

Being a simple guy and from a cybersecurity background I havent encountered funky QRCodes only the black ones, one I use daily for my gym entry and others for return labels from online shopping. 

The initial model I created, easily detected the normal QRCodes and decoded them but then i let my curiosity run wild. 
Lets find QRCodes in giphy, our beloved gif playground. I immediately realized this model wont work for all use cases. 

I encountered QR Codes of all shapes, size colors and what not. So I wanted to denerate QR Code accordingly, with different colors, backgrounds and noisy qrcodes. 

I generated QR and non-QR images to build a robust binary classifier (QR vs non-QR).

- QR styles:
  - Gradient backgrounds with colored QR and alpha
  - Solid color backgrounds with colored QR
  - Noisy backgrounds with classic B/W QR

Example commands:
```zsh
# Colored + gradient background QR
python3 src/generate_qr.py --out data/raw/qr_gradient --count 4000 --size 256 --colored --bg-mode gradient --alpha-min 0.4 --alpha-max 0.9

# Colored + solid background QR
python3 src/generate_qr.py --out data/raw/qr_solid --count 3000 --size 256 --colored --bg-mode solid

# Classic QR on noisy background
python3 src/generate_qr.py --out data/raw/qr_noisy --count 3000 --size 256 --bg-mode noisy

# Non-QR negatives
python3 src/generate_non_qr.py --out data/raw/non_qr --count 10000 --size 256
```

Merge mixed QR styles and split:
```zsh
mkdir -p data/raw/qr
cp data/raw/qr_gradient/*.png data/raw/qr/
cp data/raw/qr_solid/*.png data/raw/qr/
cp data/raw/qr_noisy/*.png data/raw/qr/

python3 src/prepare_dataset.py --raw-dir data/raw --out-dir data/dataset --val-ratio 0.2
```

Example outputs:

![Colored QR on gradient background](assets/qr_gradient_sample.png)

**Caption:** Colored QR with semi-transparent modules over a vertical gradient.
```
Detection: 100% confidence
Decoded: 74055288-cb33-4fe9-9bec-fdd6fc9ebd97
```

![Colored QR on solid background](assets/qr_solid_sample.png)

**Caption:** Colored QR rendered on a solid pastel background.
```
Detection: 100% confidence
Decoded: (OpenCV could not extract - high transparency)
```

![Classic QR on noisy background](assets/qr_noisy_sample.png)

**Caption:** Classic black/white QR composited onto a noisy textured background.
```
Detection: 100% confidence
Decoded: IwJurj8TZsKpketN5vICdibE5
```

## 2. Model Architecture

We used a compact CNN classifier (`src/model.py`) over 128x128 grayscale inputs:
- Rescaling + light augmentation
- 3 conv blocks with BatchNorm and MaxPool
- GlobalAveragePooling + dense layers
- Sigmoid output for QR presence

Trainable parameters: ~615K.

## 3. Training Pipeline

Training loads `data/dataset/{train,val}/{qr,non_qr}` via `image_dataset_from_directory`, converts to grayscale, and trains with early stopping and checkpointing.

```zsh
python3 src/train.py --data-dir data/dataset --img-size 128 --batch-size 128 --epochs 15 --model-dir models
```

Artifacts:
- Best checkpoint: `models/qr_classifier.h5`
- Keras SavedModel: `models/qr_classifier_savedmodel.keras`

## 4. Inference and Decoding (Static Images)

For a static image, we first run the classifier to check if it likely contains a QR. If yes, we attempt to decode using OpenCV's `QRCodeDetector`.

```zsh
python3 src/infer_and_decode.py --image path/to/image.png --model models/qr_classifier.h5 --threshold 0.5
```

If decoding succeeds, you'll see the QR payload printed.

## 5. Decoding QR Codes from GIFs

Animated GIFs can be tricky. So we have to look into each frame or atleast a few of them to find and decode a QRCode.
There are different strategies, either look frame by frame or overlay frames to check if qw have a QRCode.

### 5.1 Cumulative Overlay and Decode
`src/infer_gif.py` can:
- Extract frames
- Overlay frames cumulatively (1..n) or overlay all frames
- Run detection and decode with OpenCV
- Save overlays to `tmp_cumulative_overlays/` for inspection

```zsh
# Cumulative overlays: save images, run detection and decode
python3 src/infer_gif.py --gif assets/sunqr.gif --model models/qr_classifier.h5 --threshold 0.5 --overlay-mode cumulative --blend-mode average
```

**Example 1: Sun QR Code Animation**

![Sun QR Animation](assets/sunqr.gif)

This animated GIF features a QR code with a sun/radial background animation that cycles through different brightness levels.

```zsh
python3 src/decode_animated_qr.py --gif assets/sunqr.gif --save-debug
```

**Decoding Result:** Successfully decoded using **Strategy 1 (Individual frames)**. The decoder extracted 15 frames from the animation and successfully decoded the QR code from the first frame.

**Decoded Output:**
```
Awesome
```

**Example 2: Bird QR Code Animation**

![Bird QR Animation](assets/birdqr.gif)

This GIF shows a QR code overlaid on an animated bird background. The moving background elements present a challenge for detection.

```zsh
python3 src/decode_animated_qr.py --gif assets/birdqr.gif --save-debug
```

**Decoding Result:** Successfully decoded using **Strategy 1 (Individual frames)**. Despite only having 2 frames and an animated bird background, the decoder successfully extracted the QR code from the first frame.

**Decoded Output:**
```
https://ilovefreesoftware.com
```

### 5.2 Multi-Strategy Animated Decoder
`src/decode_animated_qr.py` implements 5 strategies and exits immediately upon first successful decode:

- Strategy 1: Individual frames
- Strategy 2: Enhanced frames (binary/otsu/adaptive thresholds, sharpening, contrast, denoise)
- Strategy 3: Cumulative overlays (average/max/median) + enhancements
- Strategy 4: Sliding window overlays
- Strategy 5: Frame differences

The decoder automatically tries all strategies in sequence and stops at the first successful decode. Debug images are saved to `qr_decode_debug/` to help understand which approach worked.

**Key Learnings from Animated GIF Decoding:**
- Animated backgrounds require different strategies depending on whether the QR itself is animated or static
- Median blending is particularly effective for static QR codes with animated backgrounds
- Individual frame enhancement works best when the QR code has consistent structure across frames
- Saving debug images helps identify which strategy worked and why

## 6. Results and Observations

- The classifier confidently detects QR presence across diverse styles (colored, transparency, backgrounds).
- Decoding reliability depends on clarity and contrast; cumulative overlays can blur modules, so enhancements (thresholding, sharpening) often help.
- For animated GIF QR codes, trying multiple strategies increases the chance of decoding:
  - Individual frames may already contain a clean QR frame.
  - Otsu/adaptive thresholding frequently recovers sufficient contrast.
  - Sliding window overlays offer a good trade-off between noise reduction and sharpness.

## 7. Results Gallery

Below are decoded outputs from our test samples:

**Gradient Background QR:**
- Image: `qr_gradient_sample.png`
- Detection probability: 1.0000
- Decoded: `74055288-cb33-4fe9-9bec-fdd6fc9ebd97`

**Solid Background QR:**
- Image: `qr_solid_sample.png`
- Detection probability: 1.0000
- Decoded: *(QR detected but OpenCV couldn't extract data — likely high transparency or color contrast issue)*

**Noisy Background QR:**
- Image: `qr_noisy_sample.png`
- Detection probability: 1.0000
- Decoded: `IwJurj8TZsKpketN5vICdibE5`

All samples were correctly identified as containing QR codes. Decoding success varies with contrast and module clarity.

## 8. Tips for Robustness

- Increase dataset diversity: scale, rotation, noise, blur, colors, backgrounds.
- Use more augmentation during training.
- Consider adding a segmentation or object-detection model to localize the QR region before decoding.
- For heavily animated GIFs, export frames and inspect overlays manually (we save them for you).

## 9. Try It Yourself

1. Create venv and install deps:
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate data (as above), prepare dataset, and train.
3. Run inference on images and GIFs using the commands provided.
