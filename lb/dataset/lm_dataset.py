import numpy as np
import polars as pl
from datasets import Dataset

from lb.dataset import LBBaseDataset


def tokenize(batch, tokenizer):
    output = tokenizer(batch["molecule_smiles"], truncation=True)
    return output


class LMDataset(LBBaseDataset):
    def __init__(self, df, bb1, bb2, bb3, tokenizer, stage="train"):
        super().__init__(df, bb1, bb2, bb3, tokenizer, stage=stage)
        # df = df.select(["molecule_smiles", "BRD4", "HSA", "sEH"])
        df = (
            Dataset
            .from_pandas(df.to_pandas())
            .map(tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            .to_pandas()
        )
        self.df = pl.from_pandas(df)

    def _generate_data(self, index):
        data = {
            "input_ids": np.array(self.df[index, "input_ids"]),
            "attention_mask": np.array(self.df[index, "attention_mask"]),
        }
        if self.stage in ["train", "val"]:
            data["label"] = self._generate_label(index)
        return data
