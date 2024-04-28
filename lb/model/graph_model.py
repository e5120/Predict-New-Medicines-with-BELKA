from math import ceil

import torch
import torch_geometric
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

from lb.model import BaseModel


class GNNModel(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNNModel, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        for step in range(len(self.convs)):
            x = self.convs[step](x, adj, mask)
            x = torch.nn.functional.relu(x)
        return x


class DiffPool(BaseModel):
    def __init__(self, max_nodes=128, num_features=9, num_classes=3):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNNModel(num_features, 64, num_nodes)
        self.gnn1_embed = GNNModel(num_features, 64, 64)
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNNModel(64, 64, num_nodes)
        self.gnn2_embed = GNNModel(64, 64, 64, lin=False)
        self.gnn3_embed = GNNModel(64, 64, 64, lin=False)
        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

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
        x = x.float()
        x = torch.nn.functional.relu(self.lin1(x))
        x = self.lin2(x)
        return {
            "logits": x,
            "l": l1 + l2,
            "e": e1 + e2,
        }

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"].float())
        output["loss"] = loss
        return output
