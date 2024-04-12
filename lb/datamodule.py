from pathlib import Path

import numpy as np
import polars as pl
from torch.utils.data import  DataLoader
import lightning as L

import lb.dataset
from lb.utils import cross_validation


class LBDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.fold = 0
        # define Dataset class
        self.dataset_cls = getattr(lb.dataset, self.cfg.dataset.name)
        # load dataset

    def reset(self, fold):
        self.fold = fold

    def _generate_dataset(self, stage):
        if stage == "train":
            pass
        elif stage == "val":
            pass
        elif stage == "test":
            pass
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(
            df,
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
