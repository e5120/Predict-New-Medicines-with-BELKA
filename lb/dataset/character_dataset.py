from lb.dataset import LBBaseDataset


class CharDataset(LBBaseDataset):
    def __init__(self, df, data, bb_df, stage="train"):
        super().__init__(df, data, bb_df, stage=stage)
        assert "char_feats" in data

    def _generate_data(self, index):
        key = self.df[index, "id"]
        input_ids = self.data["char_feats"][key]
        data = {
            "input_ids": input_ids,
            "input_lengths": len(input_ids),
        }
        return data

    def __getitem__(self, index):
        data = self._generate_data(index)
        if self.stage in ["train", "val"]:
            data["labels"] = self._generate_label(index)
        if self.stage == "val":
            data["non_share"] = self.df[index, "non-share"]
        if self.stage == "train":
            data = self._augment(data)
        data = self._post_process(data)
        return data
