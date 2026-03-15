import torch
import matplotlib.pyplot as plt

from model.network import SimpleCNN
from model.diffusion import Diffusion


model = SimpleCNN(T=200)
model.load_state_dict(torch.load("model.pth"))
model.eval()

diffusion = Diffusion()

samples = diffusion.sample(model, (4, 8, 125, 125))

samples = samples.numpy()

plt.imshow(samples[0,0])
plt.colorbar()
plt.title("Generated Sample Channel 0")

plt.savefig("results/generated_sample.png", dpi=300)
plt.show()