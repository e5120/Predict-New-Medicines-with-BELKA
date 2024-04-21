import torch.nn as nn
from transformers import AutoConfig, AutoModel

from lb.model import BaseModel


class LMModel(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        if model_name == "ibm/MoLFormer-XL-both-10pct":
            self.config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                num_labels=3,
            )
            self.lm = AutoModel.from_pretrained(
                model_name,
                add_pooling_layer=False,
                deterministic_eval=True,
                trust_remote_code=True,
            )
        else:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=3)
            self.lm = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, batch):
        last_hidden_state = self.lm(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        logits = self.classifier(
            self.dropout(last_hidden_state[:, 0])
        )
        return {
            "logits": logits,
        }
