# QR Code Detector & Decoder

Welcome! This site documents how we built a QR code presence classifier with TensorFlow and robust decoders for static images and animated GIFs.

## Documentation
📖 **[Read the Full Technical Blog →](./qr-model-blog.md)**

## Project Overview
- **Dataset:** 10,000 QR images (gradient, solid, noisy backgrounds) + 10,000 non-QR images
- **Model:** Compact CNN with ~615K parameters, 128×128 grayscale inputs
- **Accuracy:** 100% detection confidence on diverse QR styles
- **Decoder:** Multi-strategy approach for animated GIFs

## Quick Links
- [Dataset generation](./qr-model-blog.md#1-data-generation): `src/generate_qr.py`, `src/generate_non_qr.py`
- [Training](./qr-model-blog.md#3-training-pipeline): `src/train.py`
- [Inference](./qr-model-blog.md#4-inference-and-decoding-static-images): `src/infer_and_decode.py`
- [GIF overlays](./qr-model-blog.md#51-cumulative-overlay-and-decode): `src/infer_gif.py`
- [Animated decoder](./qr-model-blog.md#52-multi-strategy-animated-decoder): `src/decode_animated_qr.py`
- [Results Gallery](./qr-model-blog.md#7-results-gallery)

## Sample Images
![Gradient QR](./assets/qr_gradient_sample.png)
![Solid QR](./assets/qr_solid_sample.png)
![Noisy QR](./assets/qr_noisy_sample.png)
