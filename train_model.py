import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

# 1. Settings & Device
device = torch.device("mps") # Force it to use your Mac's M-series GPU
data_dir = 'kermany2018/OCT2017 /train' # Path from our previous success

# 2. Advanced Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load 80,000+ images
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Initialize the Brain
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 4) # 4 diseases
model = model.to(device)

# 5. Loss & Optimizer (The 'Teacher')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"👁️ Starting Full Training on {len(dataset)} images...")

# 6. Training Loop (1 Epoch for a quick groundbreaking result)
model.train()
start_time = time.time()

for batch_idx, (inputs, labels) in enumerate(loader):
    inputs, labels = inputs.to(device), labels.to(device)
    
    # AI makes a guess
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # AI learns from mistakes
    loss.backward()
    optimizer.step()
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
        # Stop after 50 batches for a quick 'Proof of Concept'
        if batch_idx == 50: break 

print(f"✅ Training Step Complete! Time: {time.time() - start_time:.2f}s")

# 7. SAVE THE BRAIN
torch.save(model.state_dict(), 'retina_ai_model.pth')
print("🧠 Model saved as 'retina_ai_model.pth'")