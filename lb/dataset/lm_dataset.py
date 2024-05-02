import numpy as np
from lb.dataset import LBBaseDataset


class LMDataset(LBBaseDataset):
    def __init__(self, df, data, bb_df, stage="train"):
        super().__init__(df, data, bb_df, stage=stage)
        assert "lm_feats" in data

    def _generate_data(self, index):
        key = self.df[index, "id"]
        input_ids = self.data["lm_feats"][key]
        data = {
            "input_ids": input_ids,
            "attention_mask": np.ones(len(input_ids), dtype=np.int8),
        }
        return data
