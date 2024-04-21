from abc import abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, pos_weight=[10]):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.Tensor(pos_weight),
        )

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"].float())
        output["loss"] = loss
        return output
