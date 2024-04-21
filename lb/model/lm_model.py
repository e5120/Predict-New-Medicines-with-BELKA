import torch.nn as nn
from transformers import AutoConfig, AutoModel

from lb.model import BaseModel


class LMModel(BaseModel):
    def __init__(self, model_name, pos_weight=[10]):
        super().__init__(pos_weight=pos_weight)
        if model_name == "ibm/MoLFormer-XL-both-10pct":
            self.config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                deterministic_eval=True,
                num_labels=3,
            )
            self.lm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                deterministic_eval=True,
            )
        else:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=3)
            self.lm = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, batch):
        pooler_output = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).pooler_output
        logits = self.classifier(self.dropout(pooler_output))
        return {
            "logits": logits,
        }
