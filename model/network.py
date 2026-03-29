import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, T=200):
        super().__init__()
        self.T = T
        self.time_embed = nn.Embedding(T, 32)

        self.t_proj_32 = nn.Linear(32, 32)
        self.t_proj_64 = nn.Linear(32, 64)

        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)

        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x, t):
        B = x.shape[0]

        data = x[:, :8]
        coords = x[:, 8:]
        total_energy = data.sum(dim=(1, 2, 3))
        total_energy = total_energy / (total_energy.mean() + 1e-6)
        total_energy = total_energy.view(B, 1, 1, 1)
        x = torch.cat([data, coords], dim=1)

        t_emb = self.time_embed(t)                          
        t32 = self.t_proj_32(t_emb).view(B, 32, 1, 1)
        t64 = self.t_proj_64(t_emb).view(B, 64, 1, 1)
        x1 = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x1 + t32))     
        x = F.relu(self.conv3(x + t64))      
        x = F.relu(self.conv4(x + t64))      

        x = F.relu(self.conv5(x))          

        x = x + x1                          

        x = self.conv6(x)                    
        return x