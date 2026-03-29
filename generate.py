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

samples = diffusion.sample(model, (4, 8, 125, 125))
samples = samples.numpy()

fig, axes = plt.subplots(4, 8, figsize=(24, 12))

for sample_idx in range(4):
    for ch_idx in range(8):
        ax = axes[sample_idx, ch_idx]
        im = ax.imshow(samples[sample_idx, ch_idx], 
                      cmap='hot', 
                      origin='lower')
        plt.colorbar(im, ax=ax)
        
        if sample_idx == 0:
            ax.set_title(f'Channel {ch_idx}', fontsize=8)
        if ch_idx == 0:
            ax.set_ylabel(f'Sample {sample_idx}', fontsize=8)
            
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Generated Samples - All Channels', fontsize=14)
plt.tight_layout()
plt.savefig("results/generated_all_channels.png", dpi=150)
plt.show()
