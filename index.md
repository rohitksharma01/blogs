---
layout: default
title: QR Code Detector & Decoder
---

# QR Code Detector & Decoder

Phishing via QR (“quishing”) is the buzzword these days for anyone dealing with email threats, and it's a pain for security teams.
It's always a cat-and-mouse game with threats, and this one keeps getting cheekier as we tackle the easier ones.

As I explored more, I can see why, QR codes are very robust and can be detected even if many parts aren't visible or are corrupt, that's what makes them challenging. You can find QR codes in many forms: blurry, skewed, cut off, and to make it worse, as GIF QR codes.

I was curious about detecting QR codes in GIF format and applying my ML knowledge to good use.

My approach was simple:
- Create a TFLite model for identifying QR codes
- Use the models to identify if an image has a QR code
- Use Python to decode the QR codes 
- Apply strategies to use the above approach to decode a GIF QR code
- If we see a new GIF that's not detected, we can use it to train new models

Let's put AI to work, because vibe coding is what we do these days! 
I used GitHub Copilot with multiple models auto-selected. 

## Documentation
📖 **[Read the Full Technical Blog →](./qr-model-blog.md)**

💻 **[View on GitHub →](https://github.com/rohitksharma01/QRDecoder)**

## Project Overview
- **Dataset:** 10,000 QR images (gradient, solid, noisy backgrounds) + 10,000 non-QR images
- **Model:** Compact CNN with ~615K parameters, 128×128 grayscale inputs
- **Accuracy:** 100% detection confidence on diverse QR styles
- **Decoder:** Multi-strategy approach for animated GIFs

## Quick Links
- [Dataset generation](./qr-model-blog.md#1-data-generation): `generate_qr.py`, `generate_non_qr.py`
- [Training](./qr-model-blog.md#3-training-pipeline): `train.py`
- [Inference](./qr-model-blog.md#4-inference-and-decoding-static-images): `infer_and_decode.py`
- [GIF decoding](./qr-model-blog.md#5-decoding-qr-codes-from-gifs): `infer_gif.py`
- [Tips for robustness](./qr-model-blog.md#7-tips-for-robustness)

## Sample Images
![Gradient QR](./assets/qr_gradient_sample.png)
![Solid QR](./assets/qr_solid_sample.png)
![Noisy QR](./assets/qr_noisy_sample.png)
