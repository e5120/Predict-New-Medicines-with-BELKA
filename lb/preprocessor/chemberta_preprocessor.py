import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

from lb.preprocessor import BasePreprocessor


def tokenize(batch, tokenizer):
    output = tokenizer(batch["non_isomeric_molecule_smiles"], truncation=True)
    return output


class ChemBERTaPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    def _apply(self, df, data, start_idx):
        df = df.select(["id", "non_isomeric_molecule_smiles"])
        # df = df.select(["id", "molecule_smiles"])
        end_idx = start_idx + self.batch_size
        df = (
            Dataset
            .from_pandas(df[start_idx: end_idx].to_pandas())
            .map(tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            .to_pandas()
        )
        for i in range(len(df)):
            data[df.loc[i, "id"]] = df.loc[i, "input_ids"].astype(np.int16)
        return data
