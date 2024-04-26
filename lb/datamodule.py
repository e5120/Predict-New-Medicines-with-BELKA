from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from torch_geometric.data import DenseDataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm

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
                columns=["id", "bb1_code", "bb2_code", "bb3_code"] + PROTEIN_NAMES,
            )
            for feats in self.data.values():
                df = df.filter(pl.col("id").is_in(feats.keys()))
            df = df.with_columns(
                pl.sum_horizontal(PROTEIN_NAMES).cast(pl.UInt8).alias("sum_binds"),
            )
            self.trn_df, self.val_df = lb_train_val_split(
                df, self.bb1, self.bb2, self.bb3,
                bb1_frac=cfg.bb1_frac,
                bb2_frac=cfg.bb2_frac,
                bb3_frac=cfg.bb3_frac,
                seed=cfg.seed,
            )
            trn_stats = self._get_stats(self.trn_df, "train")
            val_stats = self._get_stats(self.val_df, "val")
            stats_df = pl.from_dicts([trn_stats, val_stats])
            print(stats_df)
        else:
            self.test_df = pl.read_parquet(Path(cfg.data_dir, "processed_test.parquet"))
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
            if feature_file is not None:
                print(f"loading {feature_type}")
                files = sorted(list(Path(self.data_dir, feature_file).glob(f"{feature_file}_{self.cfg.stage}_*.npy")))
                data[feature_type] = {}
                for filename in tqdm(files):
                    data[feature_type].update(np.load(filename, allow_pickle=True).item())
                    if self.cfg.n_rows and len(data[feature_type]) >= self.cfg.n_rows:
                        break
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
