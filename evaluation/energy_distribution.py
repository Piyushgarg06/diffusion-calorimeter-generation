import h5py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

PROJECT_ROOT = 'C:/Users/gargp/OneDrive/Desktop/diffusion-generative-model'
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from model.network import SimpleCNN
from model.diffusion import Diffusion

# load real data
f = h5py.File("data/Dataset_Specific_Unlabelled.h5","r")
real = f["jet"][:2000]

# normalize the same way as training
real = real / 255.0
real = real * 2 - 1

real_energy = real.sum(axis=(1,2,3))

# load model
model = SimpleCNN(T=200)
model.load_state_dict(torch.load("model.pth"))

diffusion = Diffusion()

# generate in batches to avoid OOM
all_samples = []
batch_size = 64
num_samples = 2000

for i in range(0, num_samples, batch_size):
    n = min(batch_size, num_samples - i)
    batch = diffusion.sample(model, (n, 8, 125, 125)).numpy()
    all_samples.append(batch)
    print(f"Generated {i + n}/{num_samples}")

samples = np.concatenate(all_samples, axis=0)

generated_energy = samples.sum(axis=(1,2,3))

plt.hist(real_energy, bins=50, alpha=0.5, label="Real")
plt.hist(generated_energy, bins=50, alpha=0.5, label="Generated")

plt.legend()
plt.title("Energy Distribution Comparison")

plt.savefig("results/energy_distribution.png", dpi=300)
plt.show()
print("Real mean energy:", real_energy.mean())
print("Generated mean energy:", generated_energy.mean())

print("Real std energy:", real_energy.std())
print("Generated std energy:", generated_energy.std())