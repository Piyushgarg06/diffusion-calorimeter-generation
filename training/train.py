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

optimizer = optim.Adam(model.parameters(), lr=5e-5)

epochs = 150

for epoch in range(epochs):

    total_loss = 0
    count = 0

    for batch in loader:

        x = batch[0]

        loss = diffusion.loss(model, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count

    print("Epoch:", epoch, "Average Loss:", avg_loss)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
print("Sample batch mean:", x.mean().item())
print("Sample batch std:", x.std().item())
torch.save(model.state_dict(), "model.pth")
print("Model saved")