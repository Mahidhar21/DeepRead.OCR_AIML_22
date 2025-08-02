# DeepRead.OCR_AIML_22
Text Playground: Advanced OCR with multi-language printed text detection, full-page handwritten OCR, ClassroomOCR to erase obstructions, real-time ScribbleOCR for student notes, and a captcha solver. Built for accuracy and ease on Hugging Face.

# PolyOCR — Robust Multilingual & Multimodal OCR Toolkit

This repository contains five advanced yet practical computer vision projects developed under IITISoC 2025, aimed at solving real-world problems in Optical Character Recognition (OCR). Each model is uniquely designed to tackle distinct challenges such as multilingual text, handwritten input, real-time recognition, frame-based obstruction removal, and CAPTCHA decoding. All models are deployed using **Hugging Face Spaces** and backed by models stored in Hugging Face model repositories.

---

## Features Overview

### 1. **Multilingual OCR with Region-Specific Language Detection**
An enhanced OCR pipeline that combines **PaddleOCR** with **CLIP** to handle text in images containing multiple languages. It:
- Detects text regions using PP-OCRv5.
- Applies perspective warping for normalization.
- Uses CLIP to detect language for each region.
- Recognizes text using the appropriate language-specific OCR model.

Ideal for processing multilingual documents, signboards, or images where different scripts are mixed.

### 2. **Handwritten Text Recognition (PaddleOCR + TrOCR)**
This system improves handwritten text recognition using a modular pipeline:
- PaddleOCR first detects regions of interest.
- Perspective warping aligns the handwritten lines.
- Microsoft’s **TrOCR**, a Transformer-based model, is used for accurate recognition.

Designed for full-page handwritten inputs like scanned exam sheets, notes, or forms.

### 3. **Real-Time Handwriting OCR Canvas**
An interactive Streamlit-based drawing canvas that:
- Captures user input in real time.
- Triggers OCR every 4 seconds or when the canvas is updated.
- Uses TrOCR to convert strokes into digital text.

Useful in classrooms, live note-taking environments, or accessibility solutions for visually impaired users.

### 4. **Frame-Aware Obstruction Remover**
This system reconstructs clean board images from classroom videos where teachers might obstruct the board:
- Detects frames with unobstructed views using a custom **YOLO** model.
- Aligns selected frames with **ORB** feature matching.
- Applies **median fusion** and optional person masking to remove dynamic elements.
- Produces a sharpened image with high board content clarity.

Especially helpful for digitizing lecture content without teacher-induced occlusions.

### 5. **CAPTCHA Solver using ResNet50**
Solves distorted, rotated, and noisy CAPTCHA images by:
- Using **ResNet50** for character-wise recognition instead of end-to-end prediction.
- Segmenting characters and decoding them individually.
- Achieving ~82% accuracy on complex CAPTCHA datasets.

Deployed with a clean Gradio interface for real-time usage and testing.

---

##  Deployment Methodology

All models in this repository are:
- **Deployed via Hugging Face Spaces** using either **Streamlit** or **Gradio** for interactive interfaces.
- Backed by pre-trained or fine-tuned models hosted on **Hugging Face Model Hub**.
- Optimized for real-time or near real-time usage on the web.
  
Each directory in the repo contains:
- The corresponding Python application (`app.py`).
- Requirements file (`requirements.txt`) to replicate environments.
- Links to model cards or Hugging Face repos where applicable.
