import torch.nn as nn


class DistanceNetwork(nn.Module):
    def __init__(self, embedding_dim, normalize=True):
        super(DistanceNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 1),
        )
        self.normalize = normalize
        if normalize:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        if self.normalize:
            output = self.sigmoid(output)
        return output
