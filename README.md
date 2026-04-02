<h1 align="center"> 👁️ RetinaScan AI: Deep Learning for Retinal Diagnostics </h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

**RetinaScan AI** is an advanced medical imaging tool that leverages the **ResNet-18** architecture to identify retinal pathologies from **Optical Coherence Tomography (OCT)** scans with clinical-grade precision.

---

### 🚀 [Click Here to Try the Live Demo](https://retinascanai.streamlit.app)

---

# RetinaScan AI – Retinal Disease Detection using Deep Learning

## Overview
RetinaScan AI is a deep learning-based web application that performs automated screening of retinal diseases using OCT (Optical Coherence Tomography) images.

The model classifies retinal scans into four categories:
- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

## Problem Statement
Early detection of retinal diseases is critical to preventing irreversible vision loss. Manual diagnosis requires trained specialists and is time-intensive.

This project builds an AI-based system to assist in fast and scalable preliminary screening.

## Dataset
- **Kermany 2018 OCT Dataset**
- Total Images: 84,495
- Classes: CNV, DME, DRUSEN, NORMAL

## Model Architecture
- **ResNet-18 (Deep Residual Network)**
- Transfer Learning using pretrained ImageNet weights
- Fine-tuned for multi-class retinal image classification

## Results
- Multi-class classification across 4 retinal conditions  
- Achieves reliable performance on OCT dataset  

### (Add these images in your repo and link them here)
- Training vs Validation Accuracy Graph  
- Loss Curve  
- Confusion Matrix  

## Features
- Upload OCT retinal image  
- Real-time disease prediction  
- Multi-class classification output  
- Simple and interactive UI using Streamlit  

## Tech Stack
- Python  
- PyTorch / TensorFlow *(use the correct one)*  
- Streamlit  
- Google Drive API  

## Live Demo
https://retinascanai.streamlit.app/

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/kpriyars/RETINASCAN_AI.git

# Navigate to project folder
cd RETINASCAN_AI

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
