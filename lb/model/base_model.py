from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        logits = output["logits"]
        loss = self.loss(logits, batch["label"])
        return {
            "loss": loss,
            "logits": logits,
        }
