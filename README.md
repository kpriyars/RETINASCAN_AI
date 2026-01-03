<!-- Banner GIF -->
<p align="center">
  <img src="https://media1.tenor.com/m/uPjfbCiviRgAAAAC/eye-pupils.gif" width="100%" height="50%" />
</p>

<h1 align="center"> 👁️ RetinaScan AI: Deep Learning for Retinal Diagnostics </h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

**RetinaScan AI** is an advanced medical imaging tool that leverages the **ResNet-18** architecture to identify retinal pathologies from **Optical Coherence Tomography (OCT)** scans with clinical-grade precision.

---

### 🚀 [Click Here to Try the Live Demo](https://retinascanai.streamlit.app)

---

## 🩺 The Clinical Challenge
Early detection of retinal diseases is critical to preventing permanent vision loss. RetinaScan AI provides an automated preliminary screening for:

* **CNV (Choroidal Neovascularization):** Identifying abnormal sub-retinal vessel growth.
* **DME (Diabetic Macular Edema):** Detecting fluid accumulation in the macula.
* **DRUSEN:** Early signs of age-related macular degeneration.
* **NORMAL:** Confirming healthy retinal structural integrity.



## 🧠 Technical Workflow
1.  **Data:** Trained on 84,495 OCT images from the **Kermany 2018 Dataset**.
2.  **Architecture:** Utilizes a **Deep Residual Network (ResNet-18)** for high-feature extraction.
3.  **Optimization:** Implements **Transfer Learning** to adapt ImageNet weights to medical textures.
4.  **Deployment:** Cloud-native architecture using **Streamlit Community Cloud** and **Google Drive API** for model weight management.



## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone [https://github.com/kpriyars/RETINASCAN_AI.git](https://github.com/kpriyars/RETINASCAN_AI.git)

# Install the clinical environment
pip install -r requirements.txt

# Launch the diagnostic dashboard
streamlit run app.py
