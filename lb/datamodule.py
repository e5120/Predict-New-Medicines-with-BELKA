from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from torch_geometric.data import DenseDataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

import lb.dataset
from lb.utils import lb_train_val_split, PROTEIN_NAMES


pl.Config.set_tbl_cols(-1)


class LBDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        if "model_name" in self.cfg.model.params:
            model_name = self.cfg.model.params["model_name"]
            if model_name == "ibm/MoLFormer-XL-both-10pct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load dataset
        self.bb1 = pl.read_parquet(Path(cfg.data_dir, "processed_bb1.parquet"))
        self.bb2 = pl.read_parquet(Path(cfg.data_dir, "processed_bb2.parquet"))
        self.bb3 = pl.read_parquet(Path(cfg.data_dir, "processed_bb3.parquet"))
        self.data = self.load_features(cfg.dataset.features)
        if cfg.stage == "train":
            df = pl.read_parquet(
                Path(cfg.data_dir, "processed_train.parquet"),
                columns=["bb1_code", "bb2_code", "bb3_code"] + PROTEIN_NAMES,
            )
            df = df.with_columns(
                pl.sum_horizontal(PROTEIN_NAMES).cast(pl.UInt8).alias("sum_binds"),
                pl.int_range(len(df)).cast(pl.UInt32).alias("id"),
            )
            for feats in self.data.values():
                if feats:
                    df = df.filter(pl.col("id").is_in(feats.keys()))
            if cfg.n_rows:
                df = df.sample(n=min(len(df), cfg.n_rows))
            trn_df, val_df = lb_train_val_split(
                df, self.bb1, self.bb2, self.bb3,
                bb1_frac=cfg.bb1_frac,
                bb2_frac=cfg.bb2_frac,
                bb3_frac=cfg.bb3_frac,
                seed=cfg.seed,
            )
            self.trn_df = trn_df
            self.val_df = val_df.sort("id")
            trn_stats = self._get_stats(self.trn_df, "train")
            val_stats = self._get_stats(self.val_df, "val")
            stats_df = pl.from_dicts([trn_stats, val_stats])
            print(stats_df)
        else:
            test_df = pl.read_parquet(Path(cfg.data_dir, "processed_test.parquet"))
            self.test_df = test_df.with_columns(
                pl.int_range(len(test_df)).cast(pl.UInt32).alias("id")
            )
            print(f"# of test: {len(self.test_df)}")
        # define Dataset class
        self.dataset_cls = getattr(lb.dataset, self.cfg.dataset.name)

    def _get_stats(self, df, data_type):
        stats = {
            "data_type": data_type,
            "data_size": len(df),
        }
        for col in PROTEIN_NAMES:
            stats[f"{col}_ratio"] = df[col].mean()
            stats[f"{col}_share_ratio"] = df.filter(pl.col("sum_included_train") > 0)[col].mean()
            stats[f"{col}_non_share_ratio"] = df.filter(pl.col("sum_included_train") == 0)[col].mean()
        stats["sum_binds_mean"] = df["sum_binds"].mean()
        return stats

    def load_features(self, feature_file_dict):
        data = {}
        for feature_type, feature_file in feature_file_dict.items():
            if feature_file is None:
                data[feature_type] = None
            else:
                filename = Path(self.cfg.data_dir, f"{feature_file}_{self.cfg.stage}.npy")
                print(f"loading {filename}")
                data[feature_type] = np.load(filename, allow_pickle=True).item()
        return data

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.trn_df
        elif stage == "val":
            df = self.val_df
        elif stage == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(
            df,
            self.data,
            self.bb1,
            self.bb2,
            self.bb3,
            stage=stage,
            **self.cfg.dataset.params,
        )
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "train":
            shuffle = True
            drop_last = True
            batch_size = self.cfg.batch_size
        else:
            shuffle = False
            drop_last = False
            batch_size = 1024
        if "lm_feats" in self.data:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.cfg.num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
                pin_memory=True,
                collate_fn=DataCollatorWithPadding(self.tokenizer),
            )
        elif "graph_feats" in self.data:
            return DenseDataLoader(
                dataset,
                batch_size=batch_size,
                # num_workers=self.cfg.num_workers,
                # shuffle=shuffle,
                # drop_last=drop_last,
                # pin_memory=True,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")
