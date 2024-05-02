from abc import abstractmethod

import numpy as np
import polars as pl
from torch.utils.data import Dataset

from lb.utils import PROTEIN_NAMES


class LBBaseDataset(Dataset):
    def __init__(self, df, data, bb_df, stage="train"):
        assert stage in ["train", "val", "test"]
        for feat_type in data:
            df = df.filter(pl.col("id").is_in(data[feat_type].keys()))
            assert len(df) <= len(data[feat_type])
        self.df = df
        self.data = data
        self.bb_df = bb_df
        self.stage = stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self._generate_data(index)
        if self.stage in ["train", "val"]:
            data["label"] = self._generate_label(index)
        if self.stage == "val":
            data["non_share"] = self.df[index, "non-share"]
        if self.stage == "train":
            data = self._augment(data)
        data = self._post_process(data)
        return data

    @abstractmethod
    def _generate_data(self, index):
        raise NotImplementedError

    def _generate_label(self, index):
        if self.stage == "test":
            return np.array([0, 0, 0])
        else:
            return self.df[index, PROTEIN_NAMES].to_numpy()[0]

    def _post_process(self, data):
        return data

    def _augment(self, data):
        return data
