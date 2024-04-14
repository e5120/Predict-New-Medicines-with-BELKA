from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset


class LBBaseDataset(Dataset):
    def __init__(self, df, bb1, bb2, bb3, stage="train"):
        assert stage in ["train", "val", "test"]
        self.df = df
        self.bb1 = bb1
        self.bb2 = bb2
        self.bb3 = bb3
        self.stage = stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self._generate_data(index)
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
            return self.df[index, ["BRD4", "HSA", "sEH"]].to_numpy()[0]

    def _post_process(self, data):
        return data

    def _augment(self, x):
        return x
