---
layout: default
title: QR Code Detector & Decoder
---

# QR Code Detector & Decoder

Quishing is the buzzword these days whoever is dealing with emails threats and its a pain for security teams. 
It's always a cat and mouse game with threats and this is another one getting cheekier as we tackle thhe easier ones.

As I explored more I can see why, QR Codes are very robust and it can be detected even if many parts of the QR code arent visible or corrupt, thats what makes it more challenging. You can find QR Code in many forms, blurry, skewed, cutoff and to make it worse a gif QR Code. 

I was curious about detecting QRCode in GIF format and apply my ML knowledge to good use. 

My approach was simple.
- Create a LiteRT model for identifying QRCodes
- Use the models to indetify if an image has QR Code
- Use python to decode the QR Codes 
- Apply strategies to use the above approach to decode a gif qr code
- If we see a new gif thats not detected, we can use it to train new models

Let's put AI to work, because vibe coding is what we do these days! 
I used github copilot with multiple models auto selected. 

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
