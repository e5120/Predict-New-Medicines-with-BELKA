import torch.nn.functional as F
import lightning as L
from torchmetrics import AveragePrecision

import lb.model
import lb.optimizer
import lb.scheduler
from lb.utils import PROTEIN_NAMES


class LBModelModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(lb.model, cfg.model.name)(**cfg.model.params)
        self.map = {}
        for protein_name in PROTEIN_NAMES:
            for suffix in ["ns", "s"]:
                self.map[f"{protein_name}_{suffix}"] = AveragePrecision(task="binary")

    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, batch, batch_idx):
        return self.model.calculate_loss(batch)

    def training_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        loss = ret["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        loss = ret["loss"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metrics_update(
            F.sigmoid(ret["logits"]),
            batch["labels"].long(),
            ~batch["non_share"],
        )

    def on_validation_epoch_end(self):
        val_map = self.metrics_compute()
        self.log_dict(val_map, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def metrics_update(self, preds, labels, is_exist):
        preds = preds.detach()
        labels = labels.detach()
        is_exist = is_exist.detach()
        new_preds, new_labels = preds[~is_exist], labels[~is_exist]
        exist_preds, exist_labels = preds[is_exist], labels[is_exist]
        for i, protein_name in enumerate(PROTEIN_NAMES):
            self.map[f"{protein_name}_ns"].update(new_preds[:, i], new_labels[:, i])
            self.map[f"{protein_name}_s"].update(exist_preds[:, i], exist_labels[:, i])

    def metrics_compute(self):
        ret = {"val_map": 0.0, "val_map_prv": 0.0}
        val_pub, val_prv = 0, 0
        for k in self.map:
            ret[k] = self.map[k].compute()
            val_pub += ret[k] / 6
            if "_ns" in k:
                val_prv += 2 * ret[k] / 9
            else:
                val_prv += ret[k] / 9
            self.map[k].reset()
        assert len(ret) == 8, len(ret)
        ret["val_map"] = val_pub
        ret["val_map_prv"] = val_prv
        return ret

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward(batch)["logits"]
        probs = F.sigmoid(logits)
        return probs

    def configure_optimizers(self):
        optimizer = getattr(lb.optimizer, self.cfg.optimizer.name)(
            self.parameters(),
            **self.cfg.optimizer.params,
        )
        scheduler = getattr(lb.scheduler, self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.params,
        )
        if self.cfg.scheduler.name in ["ReduceLROnPlateau"]:
            interval = "epoch"
        else:
            interval = "step"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": "val_map",
            }
        }
