import os
import sys

PROJECT_ROOT = 'C:/Users/gargp/OneDrive/Desktop/diffusion-generative-model'
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import torch
import torch.optim as optim

from training.dataLoader import get_loader
from model.network import SimpleCNN
from model.diffusion import Diffusion

loader = get_loader(
    "data/Dataset_Specific_Unlabelled.h5",
    batch_size=32,
    num_sample=5000
)
model=SimpleCNN(T=200)
diffusion=Diffusion()

optimizer = optim.Adam(model.parameters(), lr=3e-5)

epochs = 225

for epoch in range(epochs):

    total_loss = 0
    count = 0

    for batch in loader:

        x = batch[0]
        B, C, H, W = x.shape
        x_coords = torch.linspace(-1, 1, W).view(1, 1, 1, W).repeat(B, 1, H, 1)
        y_coords = torch.linspace(-1, 1, H).view(1, 1, H, 1).repeat(B, 1, 1, W)

        x = torch.cat([x, x_coords, y_coords], dim=1)

        loss = diffusion.loss(model, x)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count

    print("Epoch:", epoch, "Average Loss:", avg_loss)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"saved/model_epoch_{epoch}.pth")
print("Sample batch mean:", x.mean().item())
print("Sample batch std:", x.std().item())
torch.save(model.state_dict(), "model.pth")
print("Model saved")