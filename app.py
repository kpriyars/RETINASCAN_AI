import os
import gdown
import torch
import streamlit as st

# 1. Configuration
# This is the ID from the link you just sent
FILE_ID = '1kDwsJh5lviDRqHjfCIGTMHSpHYDtgr0g'
MODEL_PATH = 'retina_ai_model.pth'

@st.cache_resource
def load_trained_model():
    # 2. Check if the model already exists on the server
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Model from Google Drive... (Approx. 50MB)"):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            # gdown is the most reliable way to download from Drive in Python
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # 3. Load the model
    # Streamlit Cloud runs on Linux (CPU), so we map to 'cpu'
    device = torch.device("cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# 4. Use the model
model = load_trained_model()
