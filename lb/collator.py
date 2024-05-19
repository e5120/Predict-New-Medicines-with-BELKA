import numpy as np
import torch
from torch_geometric.data.collate import collate


def pyg_collate_fn(batch):
    graphs, labels, non_share = [], [], []
    for i in range(len(batch)):
        graphs.append(batch[i]["graph"])
        if "label" in batch[i]:
            labels.append(batch[i]["label"])
        if "non_share" in batch[i]:
            non_share.append(batch[i]["non_share"])

    graph, slices, _ = collate(graphs[0].__class__, graphs, increment=False, add_batch=False)
    for key in slices:
        x = getattr(graph, key)
        if not isinstance(x, torch.Tensor):
            continue
        shape = x.shape
        bs = len(slices[key]) - 1
        dim_1 = shape[0] // bs
        shape = (bs, dim_1) + shape[1:]
        y = x.view(shape)
        setattr(graph, key, y)
    data = {"graph": graph}
    if len(labels):
        labels = torch.Tensor(np.array(labels))
        data["labels"] = labels
    if len(non_share):
        data["non_share"] = torch.Tensor(non_share).to(torch.bool)
    return data
