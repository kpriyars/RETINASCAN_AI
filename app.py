import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- STEP 1: DOWNLOAD THE MODEL FIRST ---
MODEL_PATH = 'retina_ai_model.pth'
FILE_ID = '1kDwsJh5lviDRqHjfCIGTMHSpHYDtgr0g'

# This must happen BEFORE line 12
if not os.path.exists(MODEL_PATH):
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

# --- STEP 2: NOW LOAD THE MODEL ---
device = torch.device("cpu") # Streamlit Cloud uses CPU
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)

# Line 12 will work now because the file was downloaded above
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- STEP 3: THE REST OF YOUR APP ---
st.title("👁️ RetinaScan AI")
file = st.file_uploader("Upload OCT Scan", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    st.success(f"Diagnosis: {classes[predicted.item()]} ({confidence.item()*100:.2f}%)")
