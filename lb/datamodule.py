from pathlib import Path

import numpy as np
import polars as pl
from torch.utils.data import  DataLoader
import lightning as L

import lb.dataset
from lb.utils import lb_train_val_split


class LBDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.fold = 0
        # load dataset
        if cfg.stage == "train":
            self.bb1 = pl.read_parquet(Path(cfg.data_dir, "processed_bb1.parquet"))
            self.bb2 = pl.read_parquet(Path(cfg.data_dir, "processed_bb2.parquet"))
            self.bb3 = pl.read_parquet(Path(cfg.data_dir, "processed_bb3.parquet"))
            df = pl.read_parquet(Path(cfg.data_dir, "processed_train.parquet"))
            self.trn_df, self.val_df, self.aux_df = lb_train_val_split(
                df, self.bb1, self.bb2, self.bb3,
                bb1_frac=cfg.bb1_frac,
                bb2_frac=cfg.bb2_frac,
                bb3_frac=cfg.bb3_frac,
                save_dir=cfg.data_dir,
                overwrite=cfg.overwrite,
                seed=cfg.seed,
            )
            if cfg.fraction < 1.0:
               self.trn_df = self.trn_df.sample(fraction=cfg.fraction)
            print(f"# of train: {len(self.trn_df)}, # of val: {len(self.val_df)}")
        else:
            self.test_df = pl.read_parquet(Path(cfg.data_dir, "processed_test.parquet"))
            print(f"# of test: {len(self.test_df)}")
        # define Dataset class
        self.dataset_cls = getattr(lb.dataset, self.cfg.dataset.name)

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
            self.bb1,
            self.bb2,
            self.bb3,
            stage=stage,
            **self.cfg.dataset.params,
        )
        return dataset

    def _generate_dataloader(self, dataset, stage):
        if stage == "train":
            shuffle=True
            drop_last=True
        else:
            shuffle=False
            drop_last=False
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )

    def train_dataloader(self):
        train_dataset = self._generate_dataset("train")
        train_loader = self._generate_dataloader(train_dataset, "train")
        return train_loader

    def val_dataloader(self):
        val_dataset = self._generate_dataset("val")
        val_loader = self._generate_dataloader(val_dataset, "val")
        return val_loader

    def test_dataloader(self):
        test_dataset = self._generate_dataset("test")
        test_loader = self._generate_dataloader(test_dataset, "test")
        return test_loader
