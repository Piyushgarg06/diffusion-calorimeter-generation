import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('data/Dataset_Specific_Unlabelled.h5', 'r') as f:
    data = f['jet'][:10]
    print("shape:", data.shape)
    print("dtype:", data.dtype)
    print("min:", data.min())
    print("max:", data.max())
    
    sample = data[0]  
    
    for i in range(8):
        channel = sample[:, :, i]  
        nonzero = (channel > 0).sum()
        print(f"Channel {i}: max={channel.max():.1f}, "
              f"nonzero={nonzero}/{125*125} "
              f"({100*nonzero/(125*125):.2f}%)")


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    channel = sample[:, :, i]
    im = axes[i].imshow(channel, cmap='hot', origin='lower')
    axes[i].set_title(f'Channel {i}')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig('channels.png', dpi=150)
plt.show()