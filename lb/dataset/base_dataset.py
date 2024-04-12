from abc import abstractmethod

from torch.utils.data import Dataset


class HMSBaseDataset(Dataset):
    def __init__(self, df, stage="train"):
        assert stage in ["train", "val", "test"]
        self.df = df
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
            return -1
        else:
            return self.df[index, "binds"]

    def _post_process(self, data):
        return data

    def _augment(self, x):
        return x
