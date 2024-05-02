from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm

import lb.dataset
from lb.utils import PROTEIN_NAMES
from lb.collator import pyg_collate_fn


pl.Config.set_tbl_cols(-1)


class LBDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.iteration = 0
        self.chunk_size = cfg.chunk_size
        self.data_dir = Path(cfg.dir.data_dir)
        self.dataset_cls = getattr(lb.dataset, self.cfg.dataset.name)
        if "model_name" in self.cfg.model.params:
            model_name = self.cfg.model.params["model_name"]
            if model_name == "ibm/MoLFormer-XL-both-10pct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load dataset
        self.bb_df = pl.read_parquet(Path(cfg.data_dir, "processed_bb.parquet"))
        self.feature_types = list(cfg.dataset.features.keys())
        self.feature_files = {"train": {}, "val": {}, "test": {}}
        for feature_type, feature_name in cfg.dataset.features.items():
            trn_files = list(Path(self.data_dir, feature_name).glob(f"{feature_name}_train_*.npy"))
            val_files = list(Path(self.data_dir, feature_name).glob(f"{feature_name}_val_fold{self.cfg.fold}.npy"))
            assert len(val_files) == 1
            test_files = list(Path(self.data_dir, feature_name).glob(f"{feature_name}_test_*.npy"))
            self.feature_files["train"][feature_type] = trn_files
            self.feature_files["val"][feature_type] = val_files
            self.feature_files["test"][feature_type] = test_files
        self.num_train_files = len(trn_files)
        if cfg.stage == "train":
            columns = ["id", "bb1_code", "bb2_code", "bb3_code"] + PROTEIN_NAMES
            trn_df = pl.read_parquet(Path(cfg.data_dir, "processed_train.parquet"), columns=columns)
            val_df = pl.read_parquet(
                Path(cfg.data_dir, f"processed_val_fold{cfg.fold}.parquet"),
                columns=columns+["non-share"],
            )
            non_share_val_df = val_df.filter(pl.col("non-share"))
            for i in range(1, 4):
                trn_df = trn_df.filter(~pl.col(f"bb{i}_code").is_in(non_share_val_df[f"bb{i}_code"]))
            self.trn_df = trn_df.sort("id")
            self.val_df = val_df.sort("id")
            trn_stats = self._get_stats(self.trn_df, "train")
            share_val_stats = self._get_stats(self.val_df.filter(~pl.col("non-share")), "share_val")
            non_share_val_stats = self._get_stats(self.val_df.filter(pl.col("non-share")), "non_share_val")
            stats_df = pl.from_dicts([trn_stats, share_val_stats, non_share_val_stats])
            print(stats_df)
        else:
            self.test_df = pl.read_parquet(Path(cfg.data_dir, "processed_test.parquet"))
            print(f"# of test: {len(self.test_df)}")

    def _get_stats(self, df, data_type):
        stats = {
            "data_type": data_type,
            "data_size": len(df),
            "num_bb1": df["bb1_code"].n_unique(),
            "num_bb2": df["bb2_code"].n_unique(),
            "num_bb3": df["bb3_code"].n_unique(),
        }
        for col in PROTEIN_NAMES:
            stats[f"{col}_ratio"] = df[col].mean()
        return stats

    def load_features(self, stage):
        data = {}
        for feature_type, feature_files in self.feature_files[stage].items():
            if stage == "train":
                start = self.iteration * self.chunk_size
                end = (self.iteration + 1) * self.chunk_size
                feature_files = feature_files[start: end]
            data[feature_type] = {}
            pbar = tqdm(feature_files)
            for filename in pbar:
                pbar.set_description(f"loading {filename.stem}")
                data[feature_type].update(np.load(filename, allow_pickle=True).item())
        return data

    def _generate_dataset(self, stage):
        if self.iteration == 0 and stage == "train":
            for key in self.feature_files[stage]:
                np.random.shuffle(self.feature_files[stage][key])
        data = self.load_features(stage)
        if stage == "train":
            df = self.trn_df
            self.iteration += 1
            if self.iteration * self.cfg.chunk_size >= self.num_train_files:
                self.iteration = 0
        elif stage == "val":
            df = self.val_df
        elif stage == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(
            df, data, self.bb_df,
            stage=stage,
            **self.cfg.dataset.params,
        )
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "train":
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        if "lm_feats" in self.feature_types:
            if "seq_len" in self.cfg.model.params:
                collate_fn = DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=self.cfg.model.params.seq_len,
                    padding="max_length",
                )
            else:
                collate_fn = DataCollatorWithPadding(self.tokenizer)
        elif "graph_feats" in self.feature_types:
            collate_fn = pyg_collate_fn
        else:
            raise NotImplementedError
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")
