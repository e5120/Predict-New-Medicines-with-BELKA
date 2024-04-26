import numpy as np
from lb.dataset import LBBaseDataset


class LMDataset(LBBaseDataset):
    def __init__(self, df, data, bb1, bb2, bb3, tokenizer, stage="train"):
        super().__init__(df, data, bb1, bb2, bb3, tokenizer, stage=stage)
        assert "lm_feats" in data
        self.data = data["lm_feats"]

    def _generate_data(self, index):
        key = self.df[index, "id"]
        data = {
            "input_ids": self.data[key],
            "attention_mask": np.ones(len(self.data[key]), dtype=np.int8)
        }
        if self.stage in ["train", "val"]:
            data["sum_included_train"] = self.df[index, "sum_included_train"]
            data["label"] = self._generate_label(index)
        return data
