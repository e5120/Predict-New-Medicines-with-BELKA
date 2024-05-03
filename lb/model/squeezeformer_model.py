import torch.nn as nn
from squeezeformer.model import Squeezeformer

from lb.model import BaseModel


class SqueezeformerModel(BaseModel):
    def __init__(self, num_embeddings, embedding_dim, num_labels=3, padding_idx=0, squeezeformer_params={}):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.squeezeformer = Squeezeformer(
            num_classes=num_labels,
            input_dim=embedding_dim,
            **squeezeformer_params,
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, batch):
        emb = self.embedding(batch["input_ids"].long())
        outputs, output_lengths = self.squeezeformer(emb, batch["input_lengths"])
        logits = self.max_pool(outputs.permute(0, 2, 1)).squeeze()
        return {
            "logits": logits,
            "outputs": outputs,
            "output_lengths": output_lengths,
        }
