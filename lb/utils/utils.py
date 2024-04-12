import os

import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from sklearn.metrics import average_precision_score


def calculate_map(preds, labels):
    return average_precision_score(labels, preds, average="micro")


def cross_validation():
    pass


def setup(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])
    if cfg.model_checkpoint.dirpath:
        cfg.model_checkpoint.dirpath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        cfg.model_checkpoint.dirpath = None
    L.seed_everything(cfg["seed"])


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
