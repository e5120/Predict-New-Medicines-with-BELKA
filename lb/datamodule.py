from pathlib import Path

import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

import lb.dataset
from lb.utils import lb_train_val_split


class LBDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        model_name = self.cfg.model.params["model_name"]
        if model_name == "ibm/MoLFormer-XL-both-10pct":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load dataset
        self.bb1 = pl.read_parquet(Path(cfg.data_dir, "processed_bb1.parquet"))
        self.bb2 = pl.read_parquet(Path(cfg.data_dir, "processed_bb2.parquet"))
        self.bb3 = pl.read_parquet(Path(cfg.data_dir, "processed_bb3.parquet"))
        if cfg.stage == "train":
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
            if cfg.val_fraction < 1.0:
               self.val_df = self.val_df.sample(fraction=cfg.val_fraction)
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
            self.tokenizer,
            stage=stage,
            **self.cfg.dataset.params,
        )
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
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
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")
