import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, T=200):

        super().__init__()

        # timestep embedding
        self.time_embed = nn.Embedding(T, 32)

        self.conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 8, 3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):

        # timestep embedding
        t_emb = self.time_embed(t)

        # reshape so it can broadcast
        t_emb = t_emb[:, :, None, None]

        h = self.relu(self.conv1(x))

        # inject timestep information
        h = h + t_emb

        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))

        out = self.conv4(h)

        return out