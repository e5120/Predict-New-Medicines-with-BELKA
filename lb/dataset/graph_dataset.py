from torch_geometric.transforms import ToDense

from lb.dataset import LBBaseDataset


class GraphDataset(LBBaseDataset):
    def __init__(self, df, data, bb1, bb2, bb3, stage="train"):
        super().__init__(df, data, bb1, bb2, bb3, stage=stage)
        self.data = data["graph_feats"]
        self.transform = ToDense(128)

    def _generate_data(self, index):
        data = {
            "graph": self.transform(self.data[self.df[index, "id"]]),
        }
        if self.stage in ["train", "val"]:
            data["sum_included_train"] = self.df[index, "sum_included_train"]
            data["label"] = self._generate_label(index)
        return data
