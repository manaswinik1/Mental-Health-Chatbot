"""Basic GNN model and training utilities for anomaly detection."""

from typing import Tuple
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Simple two-layer Graph Convolutional Network encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train_gnn(data: Data, epochs: int = 200, lr: float = 0.01) -> Tuple[nn.Module, torch.Tensor]:
    """Train a simple GCN and return the model and node embeddings."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNEncoder(data.num_node_features, 16, 8).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()

    embeddings = model(data.x, data.edge_index).detach().cpu()
    return model, embeddings
