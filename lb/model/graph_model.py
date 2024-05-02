import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import DenseGCNConv as GCNConv

from lb.model import BaseModel


class GNNModel(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=1, normalize=False):
        super(GNNModel, self).__init__()
        channels = [in_channels] + [hidden_channels] * n_layers + [out_channels]
        self.convs = nn.ModuleList([
            GCNConv(channels[i], channels[i+1], normalize)
            for i in range(len(channels)-1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(channels[i])
            for i in range(1, len(channels))
        ])

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = self.convs[step](x, adj, mask)
            x = F.relu(x)
        return x


class DiffPool(BaseModel):
    def __init__(self, hidden_channels, max_nodes=128, num_features=9, n_layers=1, num_classes=3):
        super(DiffPool, self).__init__()
        num_nodes = int(0.25 * max_nodes)
        self.gnn1_pool = GNNModel(num_features, hidden_channels, num_nodes, n_layers=n_layers)
        self.gnn1_embed = GNNModel(num_features, hidden_channels, hidden_channels, n_layers=n_layers)
        num_nodes = int(0.25 * num_nodes)
        self.gnn2_pool = GNNModel(hidden_channels, hidden_channels, num_nodes, n_layers=n_layers)
        self.gnn2_embed = GNNModel(hidden_channels, hidden_channels, hidden_channels, n_layers=n_layers)
        self.gnn3_embed = GNNModel(hidden_channels, hidden_channels, hidden_channels, n_layers=n_layers)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x = batch["graph"].x.float()
        adj = batch["graph"].adj.sum(axis=3).float()
        mask = batch["graph"].mask
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = torch_geometric.nn.dense_diff_pool(x, adj, s, mask)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = torch_geometric.nn.dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return {
            "logits": x,
            "l": l1 + l2,
            "e": e1 + e2,
        }
