import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from lb.model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, model_name, embedding_dim, num_filters, num_labels=3, seq_len=142):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_embeddings = len(tokenizer)
        padding_idx = tokenizer.pad_token_id
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.conv_list = nn.Sequential(
            nn.Conv1d(seq_len, num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters, 2*num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(2*num_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*num_filters, 3*num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(3*num_filters),
            nn.ReLU(inplace=True),
        )
        self.global_max_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(3*num_filters, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_labels),
        )
        self.apply(self._init_weight)

    def _init_weight(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.xavier_normal_(x.weight, gain=nn.init.calculate_gain("relu"))
            if x.bias is not None:
                nn.init.zeros_(x.bias)

    def forward(self, batch):
        emb = self.embedding(batch["input_ids"])
        x = self.conv_list(emb)
        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = self.global_max_pool(x)
        x = x.squeeze()
        logits = self.fc(x)
        return {
            "logits": logits,
        }
