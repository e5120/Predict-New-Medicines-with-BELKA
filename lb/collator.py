import numpy as np
import torch
from torch_geometric.data.collate import collate


def pyg_collate_fn(batch):
    graphs, labels, sum_included_train = [], [], []
    for i in range(len(batch)):
        graphs.append(batch[i]["graph"])
        labels.append(batch[i]["label"])
        sum_included_train.append(batch[i]["sum_included_train"])
    labels = torch.Tensor(np.array(labels))
    sum_included_train = torch.Tensor(sum_included_train)

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
    return {
        "graph": graph,
        "labels": labels,
        "sum_included_train": sum_included_train,
    }
