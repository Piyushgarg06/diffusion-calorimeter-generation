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

norm = torch.load("norm.pt")
mean = norm["mean"].item()
std = norm["std"].item()

f = h5py.File("data/Dataset_Specific_Unlabelled.h5","r")
real = f["jet"][:2000]

epsilon = 1e-6
real_log = np.log(real+epsilon)
real_denorm = np.exp(real_log)
real_energy = real_denorm.sum(axis=(1,2,3))

model = SimpleCNN(T=200)
model.load_state_dict(torch.load("Saved/model_epoch_200.pth"))

diffusion = Diffusion()

all_samples = []
batch_size = 64
num_samples = 2000

for i in range(0, num_samples, batch_size):
    n = min(batch_size, num_samples - i)
    batch = diffusion.sample(model, (n, 8, 125, 125)).numpy()
    all_samples.append(batch)
    print(f"Generated {i + n}/{num_samples}")

samples = np.concatenate(all_samples, axis=0)
samples = np.clip(samples, -3, 3)
log_space = samples * std + mean
log_space = np.clip(log_space, -10, 2)
gen_denorm = np.exp(log_space)
generated_energy = gen_denorm.sum(axis=(1,2,3))


print("Generated min/max:", gen_denorm.min(), gen_denorm.max())
print("Real min/max:", real_denorm.min(), real_denorm.max())

# TOTAL ENERGY AND HISTOGRAM 

plt.hist(real_energy, bins=50, alpha=0.5, label="Real")
plt.hist(generated_energy, bins=50, alpha=0.5, label="Generated")

plt.legend()
plt.title("Energy Distribution Comparison")

plt.savefig("results/energy_distribution1.png", dpi=300)
plt.show()

print("Real mean energy:", real_energy.mean())
print("Generated mean energy:", generated_energy.mean())

print("Real std energy:", real_energy.std())
print("Generated std energy:", generated_energy.std())


# Radial Shower Analysis and sparsity

def radial_analysis(data):
    sparsities=[]
    threshold = 1e-3
    x = np.arange(125)
    y=np.arange(125)    
    xx,yy = np.meshgrid(x,y)

    profiles=[]
    for event in data:
        img = event.sum(axis=0)

        E = img.sum()
        if E==0:
            continue
        # Sparsity
        zero_pixel = (np.abs(img)<threshold).sum()
        total_pixel = img.size
        sparsity = zero_pixel/total_pixel
        sparsities.append(sparsity)

        # centroid
        xc = (xx*img).sum()/E
        yc = (yy*img).sum()/E

        dist = np.sqrt((xx-xc)**2+(yy-yc)**2)
        dist_int = dist.astype(int)
        max_r = 125
        radial_energy = np.zeros(max_r)

        for r in range(max_r):
            mask = (dist_int==r)
            if mask.sum()>0:
                radial_energy[r] = img[mask].mean()
        profiles.append(radial_energy)
    return np.mean(profiles,axis=0), np.array(sparsities)

real_radial_profile, real_sparsity = radial_analysis(np.transpose(real_denorm, (0, 3, 1, 2)))
gen_radial_profile, gen_sparsity = radial_analysis(gen_denorm)

plt.plot(real_radial_profile, label="Real")
plt.plot(gen_radial_profile, label="Generated")
plt.legend()
plt.title("Radial Profile Comparison")
plt.savefig("results/radial_profile.png", dpi=300)
plt.show()

plt.hist(real_sparsity, bins=50, alpha=0.5, label="Real")
plt.hist(gen_sparsity, bins=50, alpha=0.5, label="Generated")
plt.legend()
plt.title("Sparsity Distribution Comparison")
plt.savefig("results/sparsity_distribution.png", dpi=300)
plt.show()

        
