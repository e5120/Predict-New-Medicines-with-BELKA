from transformers import AutoTokenizer
from datasets import Dataset

from lb.preprocessor import BasePreprocessor


def tokenize(batch, tokenizer):
    output = tokenizer(batch["molecule_smiles"], truncation=True)
    return output


class ChemBERTaPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.preprocessor.model_name)

    def _apply(self, df, data, start_idx):
        end_idx = start_idx + self.batch_size
        df = (
            Dataset
            .from_pandas(df[start_idx: end_idx].to_pandas())
            .map(tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            .to_pandas()
        )
        for i in range(len(df)):
            data[start_idx+i] = {
                "input_ids": df.loc[i, "input_ids"],
                "attention_mask": df.loc[i, "attention_mask"],
            }
        return data