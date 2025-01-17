import hydra
import polars as pl
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def setup(cfg):
    if cfg.model_checkpoint.dirpath:
        cfg.model_checkpoint.dirpath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        cfg.model_checkpoint.dirpath = None
    L.seed_everything(cfg["seed"])
    if "model_name" in cfg.model.params and cfg.model.params["model_name"] == "ibm/MoLFormer-XL-both-10pct":
        cfg.trainer.precision = 32
    torch.backends.cudnn.benchmark = cfg.benchmark
    return cfg


def calculate_map(preds, labels):
    return average_precision_score(labels, preds, average="micro")


def bb_train_val_split(bb, no, bb_frac, seed=42):
    bb = bb.filter(pl.col("included_train"))
    new_bb = bb.sample(fraction=bb_frac, seed=seed).with_columns(pl.lit(False).alias("included_train"))
    exist_bb = bb.filter(~pl.col(f"bb{no}_code").is_in(new_bb[f"bb{no}_code"])).with_columns(pl.lit(True).alias("included_train"))
    bb = pl.concat([new_bb, exist_bb])
    return bb


def lb_train_val_split(df, bb1, bb2, bb3, bb1_frac=0.1, bb2_frac=0.2, bb3_frac=0.2, seed=42):
    print("splitting dataset now ...")
    bb1 = bb1.filter(pl.col("bb1_code").is_in(df["bb1_code"].unique()))
    bb2 = bb2.filter(pl.col("bb2_code").is_in(df["bb2_code"].unique()))
    bb3 = bb3.filter(pl.col("bb3_code").is_in(df["bb3_code"].unique()))
    bb1 = bb_train_val_split(bb1, 1, bb1_frac, seed=seed)
    bb2 = bb_train_val_split(bb2, 2, bb2_frac, seed=seed)
    bb3 = bb_train_val_split(bb3, 3, bb3_frac, seed=seed)
    df = (
        df
        .join(bb1[["bb1_code", "included_train"]], on="bb1_code", how="inner")
        .rename({"included_train": "bb1_included_train"})
        .join(bb2[["bb2_code", "included_train"]], on="bb2_code", how="inner")
        .rename({"included_train": "bb2_included_train"})
        .join(bb3[["bb3_code", "included_train"]], on="bb3_code", how="inner")
        .rename({"included_train": "bb3_included_train"})
        .with_columns(
            pl.sum_horizontal(["bb1_included_train", "bb2_included_train", "bb3_included_train"]).cast(pl.UInt8).alias("sum_included_train")
        )
        .drop(["bb1_included_train", "bb2_included_train", "bb3_included_train"])
    )
    new_val_df = df.filter(pl.col("sum_included_train") < 3)
    trn_df = df.filter(pl.col("sum_included_train") == 3)
    trn_indices, val_indices = train_test_split(np.arange(len(trn_df)), test_size=int(0.3 * len(new_val_df)), random_state=seed)
    exist_val_df = trn_df[val_indices]
    val_df = pl.concat([new_val_df, exist_val_df])
    trn_df = trn_df[trn_indices]
    print(val_df["sum_included_train"].value_counts().sort("sum_included_train"))
    return trn_df, val_df


def get_num_training_steps(n_data, cfg):
    steps_per_epoch = n_data // cfg.batch_size // len(cfg.trainer.devices) // cfg.trainer.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    return num_training_steps


def build_callbacks(cfg):
    checkpoint_callback = ModelCheckpoint(
        filename=f"model-{{val_map:.4f}}",
        **cfg.model_checkpoint,
    )
    early_stop_callback = EarlyStopping(**cfg.early_stopping)
    progress_bar_callback = TQDMProgressBar(refresh_rate=1)
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        progress_bar_callback,
    ]
    return callbacks
