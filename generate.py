import numpy as np
import torch 
from model.network import SimpleCNN
from model.diffusion import Diffusion
import h5py
import matplotlib.pyplot as plt

model = SimpleCNN(T=200)
model.load_state_dict(torch.load("Saved/model_epoch_200.pth"))
model.eval()

diffusion = Diffusion()

samples = diffusion.sample(model, (1, 8, 125, 125))
samples = samples.numpy()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ch_idx in range(8):
    ax = axes[ch_idx]
    im = ax.imshow(samples[0, ch_idx], 
                  cmap='hot', 
                  origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Channel {ch_idx}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Generated Sample - All Channels', fontsize=14)
plt.tight_layout()
plt.savefig("results/generated_sample.png", dpi=150)
plt.show()