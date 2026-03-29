import h5py
import torch

file = h5py.File("data/Dataset_Specific_Unlabelled.h5", "r")
data = file["jet"][:5000] 

data = torch.tensor(data).float()

epsilon = 1e-6
data = torch.log(data + epsilon)

mean = data.mean()
std = data.std()

print("Mean:", mean.item())
print("Std:", std.item())

torch.save({
    "mean": mean,
    "std": std
}, "norm.pt")

print("Saved norm.pt")