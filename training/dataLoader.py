import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_loader(file_path, batch_size=32, num_sample=5000):
    file = h5py.File(file_path,"r")
    data=file["jet"]

    subset = data[:num_sample]
    subset=torch.tensor(subset).float()

    subset = subset / 255.0
    subset = subset * 2 - 1
    subset=subset.permute(0,3,1,2)

    dataset=TensorDataset(subset)

    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader