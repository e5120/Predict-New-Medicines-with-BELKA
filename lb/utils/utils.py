import os
from pathlib import Path

import hydra
import polars as pl
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def setup(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])
    if cfg.model_checkpoint.dirpath:
        cfg.model_checkpoint.dirpath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        cfg.model_checkpoint.dirpath = None
    L.seed_everything(cfg["seed"])


def calculate_map(preds, labels):
    return average_precision_score(labels, preds, average="micro")


def bb_train_val_split(bb, no, bb_frac=0.1, seed=42):
    bb = bb.filter(pl.col("included_train"))
    new_bb = bb.sample(fraction=bb_frac, seed=seed).with_columns(pl.lit(False).alias("included_train"))
    exist_bb = bb.filter(~pl.col(f"bb{no}_code").is_in(new_bb[f"bb{no}_code"])).with_columns(pl.lit(True).alias("included_train"))
    bb = pl.concat([new_bb, exist_bb])
    return bb


def lb_train_val_split(df, bb1, bb2, bb3, bb1_frac=0.1, bb2_frac=0.2, bb3_frac=0.2, save_dir="", overwrite=False, seed=42):
    if overwrite or not Path(save_dir, f"lb_train_{seed}seed.parquet").exists():
        print("splitting dataset now ...")
        bb1 = bb_train_val_split(bb1, 1, bb_frac=bb1_frac, seed=seed)
        bb2 = bb_train_val_split(bb2, 2, bb_frac=bb2_frac, seed=seed)
        bb3 = bb_train_val_split(bb3, 3, bb_frac=bb3_frac, seed=seed)
        df = (
            df
            .join(bb1[["bb1_code", "included_train"]], on="bb1_code", how="inner")
            .rename({"included_train": "bb1_included_train"})
            .join(bb2[["bb2_code", "included_train"]], on="bb2_code", how="inner")
            .rename({"included_train": "bb2_included_train"})
            .join(bb3[["bb3_code", "included_train"]], on="bb3_code", how="inner")
            .rename({"included_train": "bb3_included_train"})
            .with_columns(
                pl.concat_list(["bb1_included_train", "bb2_included_train", "bb3_included_train"]).list.sum().alias("sum_included_train")
            )
            .drop(["bb1_included_train", "bb2_included_train", "bb3_included_train"])
        )
        new_val_df = df.filter(pl.col("sum_included_train") == 0)
        aux_df = df.filter((0 < pl.col("sum_included_train")) & (pl.col("sum_included_train") < 3))
        trn_df = df.filter(pl.col("sum_included_train") == 3)
        trn_indices, val_indices = train_test_split(np.arange(len(trn_df)), test_size=int(0.7 * len(new_val_df)), random_state=seed)
        exist_val_df = trn_df[val_indices]
        val_df = pl.concat([new_val_df, exist_val_df])
        trn_df = trn_df[trn_indices]
        trn_df.write_parquet(Path(save_dir, f"lb_train_{seed}seed.parquet"))
        val_df.write_parquet(Path(save_dir, f"lb_val_{seed}seed.parquet"))
        aux_df.write_parquet(Path(save_dir, f"lb_aux_{seed}seed.parquet"))
    else:
        print("loading dataset now ...")
        trn_df = pl.read_parquet(Path(save_dir, f"lb_train_{seed}seed.parquet"))
        val_df = pl.read_parquet(Path(save_dir, f"lb_val_{seed}seed.parquet"))
        aux_df = pl.read_parquet(Path(save_dir, f"lb_aux_{seed}seed.parquet"))
    return trn_df, val_df, aux_df


def get_num_training_steps(n_data, cfg):
    steps_per_epoch = n_data // cfg.batch_size // cfg.trainer.devices // cfg.trainer.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    return num_training_steps


def build_callbacks(fold, cfg):
    checkpoint_callback = ModelCheckpoint(
        filename=f"model-fold-{fold}-{{val_loss:.4f}}",
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
