from torch_geometric.transforms import ToDense

from lb.dataset import LBBaseDataset


class GraphDataset(LBBaseDataset):
    def __init__(self, df, data, bb_df, stage="train"):
        super().__init__(df, data, bb_df, stage=stage)
        assert "graph_feats" in data
        self.transform = ToDense(128)

    def _generate_data(self, index):
        data = {
            "graph": self.transform(self.data["graph_feats"][self.df[index, "id"]]),
        }
        return data
