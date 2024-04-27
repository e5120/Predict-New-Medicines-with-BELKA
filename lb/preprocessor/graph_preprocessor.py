import polars as pl
from torch_geometric.utils.smiles import from_smiles

from lb.preprocessor import BasePreprocessor
from lb.utils import PROTEIN_NAMES


class GraphPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stage = cfg.stage
        self.under_sampling_ratio = cfg.preprocessor.under_sampling_ratio
        self.seed = cfg.seed

    def _apply(self, df):
        if self.stage == "train" and self.under_sampling_ratio:
            df = self._under_sampling(df)
        ret = [from_smiles(smiles) for smiles in df["molecule_smiles"].to_list()]
        data = dict(zip(df["id"].to_list(), ret))
        return data

    def _under_sampling(self, df):
        df = df.with_columns(pl.sum_horizontal(PROTEIN_NAMES).alias("sum_binds"))
        pos_df = df.filter(pl.col("sum_binds") > 0)
        neg_df = df.filter(pl.col("sum_binds") == 0)
        neg_df = neg_df.sample(n=int(min(len(neg_df), len(pos_df)*self.under_sampling_ratio)), seed=self.seed)
        df = pl.concat([pos_df, neg_df])
        return df
