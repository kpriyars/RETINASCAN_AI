import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RetinaScan AI", page_icon="👁️", layout="centered")

# --- CUSTOM MEDICAL THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3, p, label { color: #1E3A8A !important; }
    .report-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL DOWNLOAD & INITIALIZATION ---
MODEL_PATH = 'retina_ai_model.pth'
FILE_ID = '1kDwsJh5lviDRqHjfCIGTMHSpHYDtgr0g'

# This block ensures the file exists before any other code runs
if not os.path.exists(MODEL_PATH):
    with st.spinner("🚀 Initializing AI Brain... Downloading weights (Approx 50MB)."):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            st.rerun() # Refresh to show the UI now that file exists
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

@st.cache_resource
def load_retina_model():
    # Load architecture (ResNet-18)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 4) # 4 classes: CNV, DME, DRUSEN, NORMAL
    
    # Map to CPU for Streamlit Cloud
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize model
model = load_retina_model()

# --- MAIN UI ---
st.title("👁️ RetinaScan AI")
st.write("### Clinical OCT Diagnostic Assistant")
st.write("Upload a retinal OCT scan for instant pathology detection.")

file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

if file:
    # 1. Image Processing
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Patient OCT Scan", use_container_width=True)
    
    # 2. Transform for Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_batch = transform(img).unsqueeze(0)
    
    # 3. Prediction
    with torch.no_grad():
        output = model(input_batch)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probs, 0)
    
    # 4. Results Display
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    result = classes[predicted.item()]
    conf_score = confidence.item() * 100
    
    # Color logic
    color = "#28A745" if result == "NORMAL" else "#D32F2F"
    
    st.markdown(f"""
        <div class="report-card">
            <p style="font-size: 14px; color: #7F8C8D;">ANALYSIS RESULT</p>
            <h2 style="color: {color} !important;">{result}</h2>
            <p><b>AI Confidence:</b> {conf_score:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    descriptions = {
        'CNV': "Abnormal blood vessel growth (Neovascularization). Risk of vision loss.",
        'DME': "Fluid buildup (Edema) in the macula. Common in diabetic patients.",
        'DRUSEN': "Early signs of age-related degeneration. Monitoring recommended.",
        'NORMAL': "Retinal layers appear healthy and continuous."
    }
    st.info(f"**Medical Context:** {descriptions[result]}")
