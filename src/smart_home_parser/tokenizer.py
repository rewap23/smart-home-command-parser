from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch

# constants for special tokens
PAD_TOKEN = "<pad>" 
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"


class WordTokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        max_length: int = 32,
    ) -> None:
        self.max_length = max_length
        self.token_to_id = token_to_id or {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
        }
        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }
# properties for special token IDs and vocabulary size
    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
# this function normalizes the input text by converting it to lowercase
# removing non-alphanumeric characters (except for '%')
# and collapsing multiple spaces into a single space
# tt returns the normalized text as a string
    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9%]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        return normalized.split() if normalized else []

    def fit(self, texts: list[str], min_frequency: int = 1) -> None:
        counts = Counter(
            token
            for text in texts
            for token in self.tokenize(text)
        )

        vocabulary = sorted(
            token
            for token, count in counts.items()
            if count >= min_frequency
            and token not in self.token_to_id
        )
# this function builds the vocabulary 
# by counting the frequency of tokens in the provided texts
        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
            **{
                token: index + 3
                for index, token in enumerate(vocabulary)
            },
        }
        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }
# encode function takes a string input and returns a tuple of two tensors:
# 1. token_ids tensor: contains the token IDs corresponding to the tokens in the input
# 2. attention_mask tensor: indicates which tokens are actual tokens (1) & which are padding (0)
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [CLS_TOKEN, *self.tokenize(text)]
        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens[: self.max_length]
        ]

        attention_mask = [1] * len(token_ids)
        padding = self.max_length - len(token_ids)

        token_ids.extend([self.pad_id] * padding)
        attention_mask.extend([0] * padding)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
        )
# decode function takes a list of token IDs 
# then returns the corresponding tokens as a list of strings
    def decode(self, token_ids: list[int]) -> list[str]:
        return [
            self.id_to_token.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "max_length": self.max_length,
            "token_to_id": self.token_to_id,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_id={
                token: int(token_id)
                for token, token_id in payload["token_to_id"].items()
            },
            max_length=int(payload["max_length"]),
        )