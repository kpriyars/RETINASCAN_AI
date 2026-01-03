import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- STEP 1: DOWNLOAD MUST HAPPEN FIRST ---
MODEL_PATH = 'retina_ai_model.pth'
FILE_ID = '1kDwsJh5lviDRqHjfCIGTMHSpHYDtgr0g'

# We use a function to ensure the file is ready before loading
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

download_model()

# --- STEP 2: LOAD ONLY IF FILE EXISTS ---
device = torch.device("cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
else:
    st.error("Model file is still missing after download attempt.")
    st.stop()

# --- STEP 3: UI CODE ---
st.title("👁️ RetinaScan AI")
# ... (rest of your UI code)
