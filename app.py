import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# 1. Load the AI Brain
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('retina_ai_model.pth', map_location=device))
model.to(device)
model.eval()

# 2. Setup the Web Page
st.set_page_config(page_title="RetinaScan AI", page_icon="👁️")
st.title("👁️ RetinaScan AI: OCT Diagnostic Assistant")
st.write("Upload a Retinal OCT scan for instant pathology detection.")

# 3. Image Upload
file = st.file_uploader("Upload OCT Image (JPG/PNG)", type=["jpeg", "jpg", "png"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Uploaded Scan", use_container_width=True)
    
    # 4. Preprocess and Predict
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    # 5. Show Results
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    result = classes[predicted.item()]
    
    st.subheader(f"Analysis Result: :red[{result}]")
    
    # Medical context
    descriptions = {
        'CNV': "Abnormal blood vessel growth (Neovascularization). Potential vision loss risk.",
        'DME': "Fluid buildup (Edema) in the macula. Common in diabetic patients.",
        'DRUSEN': "Early signs of age-related degeneration. Monitoring recommended.",
        'NORMAL': "Retinal layers appear healthy and continuous."
    }
    st.info(descriptions[result])