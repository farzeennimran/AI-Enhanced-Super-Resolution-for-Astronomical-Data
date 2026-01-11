# 🌌 AI-ESRAD

## AI-Enhanced Super-Resolution for Astronomical Data

## 📌 Overview

Astronomical imaging is fundamentally constrained by low resolution, noise, and observational limitations, making it difficult to detect faint celestial objects such as distant galaxies, stellar clusters, and exoplanetary structures.

AI-ESRAD proposes an AI-driven super-resolution framework that reconstructs high-resolution (HR) astronomical images from low-resolution (LR) inputs while preserving scientific integrity.
The project introduces a learned GAN-based downsampling pipeline to generate realistic synthetic datasets and applies state-of-the-art super-resolution models to enhance astronomical imagery.

## 🚀 Key Contributions

✔️ GAN-based learned downsampling for realistic LR image generation
✔️ Synthetic HR–LR paired dataset for astronomy
✔️ Implementation of multiple SR models:

* SRGAN
* ESRGAN
* Stable Diffusion Upscaler
* Transformer-based SR
* Restormer
* ResNet-based SR

✔️ Quantitative evaluation using PSNR and SSIM
✔️ Interactive web-based GUI for image enhancement
✔️ Comparative analysis across architectures

---

## 🧠 Why Learned Downsampling (GANs)?

Traditional methods like bicubic interpolation apply a fixed mathematical function and fail to model real telescope degradations.

**GAN-based downsampling**:

* Learns realistic blur, noise, and distortions
* Mimics real telescope imaging conditions
* Improves generalization of SR models to real data
* Produces scientifically meaningful synthetic datasets

---

## 🏗️ System Architecture

### Two-Stage Pipeline

#### **Stage 1: Synthetic Dataset Generation**

```
HR Telescope Images
        ↓
GAN-based Learned Downsampling
        ↓
Realistic LR Images
        ↓
Paired (LR, HR) Dataset
```

#### **Stage 2: Super-Resolution**

```
Low-Resolution Image
        ↓
SR Model (SRGAN / ESRGAN / Transformer / Diffusion)
        ↓
Super-Resolved HR Image
```

---

## 🧪 Models Implemented

### 🔹 SRGAN

* Residual blocks + PixelShuffle
* BCE adversarial loss + perceptual loss (VGG)
* Pre-training with L1 loss

**Results:**
PSNR: **45.43 dB**
SSIM: **0.7437**

---

### 🔹 ESRGAN (Best Performer)

* RRDB blocks (Residual-in-Residual Dense Blocks)
* Relativistic discriminator
* Improved perceptual loss

**Results:**
PSNR: **52.98 dB**
SSIM: **0.8556**

---

### 🔹 Stable Diffusion (4× Upscaler)

* Used for generative enhancement
* Preserves astronomical structure
* Ideal for visualization & dataset enrichment

---

### 🔹 Transformer-Based SR

* Self-attention for long-range dependencies
* Better structural consistency

---

### 🔹 Restormer

* Multi-DConv Head Transposed Attention
* Strong SSIM performance on galaxies

---

### 🔹 ResNet-Based SR

* Strong baseline
* Faster training
* Competitive results

---

## 📊 Quantitative Results

| Model            | PSNR (dB) | SSIM       |
| ---------------- | --------- | ---------- |
| SRGAN            | 34.43     | 0.7437     |
| ESRGAN           | **52.98** | **0.8556** |
| Stable Diffusion | 36.90     | 0.7912     |
| Transformer SR   | 29.95     | 0.7241     |
| Restormer        | 31.42     | 0.7382     |

---

## 🖥️ Web Interface

A Flask-based GUI allows users to:

* Upload LR astronomical images
* View **LR → SR → HR** side-by-side
* Download enhanced results
* View PSNR & SSIM scores

---

## 📂 Repository Structure

```
AI-ESRAD/
│
├── data/
│   ├── hr_images/
│   ├── lr_images/
│   
│
├── synthetic dataset/
│   └── syntheticdataset.py
│
├── models/
│   ├── srgan/
│   ├── esrgan/
│   ├── transformer_sr/
│   ├── restormer/
│   └── diffusion/
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── 
│
├── evaluation/
│   └── visualization.py
│
├── app.py              
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-ESRAD.git
cd AI-ESRAD
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Web Application

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```

---

## 🧰 System Requirements

* **Python** ≥ 3.10
* **GPU** (NVIDIA CUDA recommended)
* **RAM** ≥ 16 GB
* OS: Windows / Linux / macOS

---

## 📐 Evaluation Metrics

* **PSNR (Peak Signal-to-Noise Ratio)**
  Measures reconstruction fidelity

* **SSIM (Structural Similarity Index)**
  Measures perceptual & structural similarity

---

## 🔬 Datasets Used

* James webb sapce telescope (JWST)
* Hubble space telescope (HST)
* Sloan Digital Sky Survey (SDSS)
* Kepler
* European Space Agency (ESA archives)

All datasets are publicly available and used for academic research only.

---

## ⚠️ Limitations

* Synthetic data may not fully capture all real telescope degradations
* High computational cost for GAN & diffusion training
* Further validation required on raw telescope observations

---

## 🎓 Academic Context

This repository accompanies the Final Year Project (FYP) titled:

> **AI-Enhanced Super-Resolution for Astronomical Data (AI-ESRAD)**

The project contributes to AI-driven astronomy, computational astrophysics, and image restoration research.

---

## 📜 License

This project is released for **academic and research use only**.
Please cite appropriately if used in publications.
