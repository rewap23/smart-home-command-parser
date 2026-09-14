from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from smart_home_parser.dataset import LABEL_FIELDS

# implementing transformer model for smart home command parsing
# what this does is it takes in a sequence of token IDs
# and attention masks as input
# then it outputs a dictionary of logits for each label field
# logits are raw scores for each of the classes, not probabilities
# label fields : (intent, action, device, location, value, unit)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_length: int
    num_classes: dict[str, int]
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    feedforward_dim: int = 256
    # feedforward dimension in the transformer encoder layer
    dropout: float = 0.1


class SmartHomeTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_length,
            embedding_dim=config.embedding_dim,
        )
        self.embedding_norm = nn.LayerNorm(config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.embedding_dim),
        )

        self.heads = nn.ModuleDict(
            {
                field: nn.Linear(
                    config.embedding_dim,
                    config.num_classes[field],
                )
                for field in LABEL_FIELDS
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.max_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds "
                f"max_length {self.config.max_length}."
            )

        positions = (
            torch.arange(
                sequence_length,
                device=input_ids.device,
            )
            .unsqueeze(0)
            .expand(batch_size, sequence_length)
        )

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(
            positions
        )
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        padding_mask = ~attention_mask.bool()

        encoded = self.encoder(
            src=hidden_states,
            src_key_padding_mask=padding_mask,
        )

        cls_embedding = encoded[:, 0, :]

        return {field: head(cls_embedding) for field, head in self.heads.items()}

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
