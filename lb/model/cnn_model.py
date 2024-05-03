import torch.nn as nn

from lb.model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, num_embeddings, embedding_dim, num_filters, num_labels=3, seq_len=142, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.conv_list = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, 2*num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(2*num_filters),
            nn.ReLU(),
            nn.Conv1d(2*num_filters, 3*num_filters, 3, stride=1, padding="valid"),
            nn.BatchNorm1d(3*num_filters),
            nn.ReLU(),
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(3*num_filters, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_labels),
        )
        self.apply(self._init_weight)

    def _init_weight(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.xavier_normal_(x.weight, gain=nn.init.calculate_gain("relu"))
            if x.bias is not None:
                nn.init.zeros_(x.bias)
        # elif isinstance(x, nn.Linear):
        #     nn.init.xavier_uniform_(x.weight)
        #     if x.bias is not None:
        #         nn.init.zeros_(x.bias)

    def forward(self, batch):
        emb = self.embedding(batch["input_ids"].long())
        x = emb.permute(0, 2, 1)
        x = self.conv_list(x)
        x = self.global_max_pool(x)
        x = x.squeeze()
        logits = self.fc(x)
        return {
            "logits": logits,
        }
