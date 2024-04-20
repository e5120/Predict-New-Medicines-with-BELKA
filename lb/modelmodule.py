import torch.nn.functional as F
import lightning as L
from torchmetrics import AveragePrecision

import lb.model
import lb.optimizer
import lb.scheduler


class LBModelModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(lb.model, cfg.model.name)(**cfg.model.params)
        self.map = AveragePrecision(task="binary")

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
        self.map.update(F.sigmoid(ret["logits"]), batch["labels"].long())

    def on_validation_epoch_end(self):
        val_map = self.map.compute()
        self.log("val_map", val_map, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map.reset()

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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val_loss",
            }
        }
