import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, T=200):

        super().__init__()

        self.time_embed = nn.Embedding(T, 32)

        self.conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64,32,3,padding=1)
        self.conv4 = nn.Conv2d(32,8,3,padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):

        t_emb = self.time_embed(t)

        t_emb = t_emb[:, :, None, None]

        h = self.relu(self.conv1(x))

        h = h + t_emb

        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))

        out = torch.tanh(self.conv4(h))

        return out