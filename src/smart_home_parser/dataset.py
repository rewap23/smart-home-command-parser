from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from smart_home_parser.tokenizer import WordTokenizer

# constants for label fields in the dataset
LABEL_FIELDS = (
    "intent",
    "action",
    "device",
    "location",
    "value",
    "unit",
)

# constants for special label values
Record = dict[str, Any]

# properties for special label values
@dataclass(frozen=True)
class LabelEncoders:
    field_to_id: dict[str, dict[str, int]]
    field_to_label: dict[str, dict[int, str]]

    @classmethod
    def fit(cls, records: list[Record]) -> "LabelEncoders":
        field_to_id: dict[str, dict[str, int]] = {}

        for field in LABEL_FIELDS:
            labels = sorted(
                {
                    str(record["labels"][field])
                    for record in records
                }
            )
            field_to_id[field] = {
                label: index
                for index, label in enumerate(labels)
            }

        field_to_label = {
            field: {
                index: label
                for label, index in mapping.items()
            }
            for field, mapping in field_to_id.items()
        }

        return cls(
            field_to_id=field_to_id,
            field_to_label=field_to_label,
        )

    def encode(self, field: str, value: str) -> int:
        return self.field_to_id[field][value]

    def decode(self, field: str, label_id: int) -> str:
        return self.field_to_label[field][label_id]

    def num_classes(self, field: str) -> int:
        return len(self.field_to_id[field])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(
            json.dumps(self.field_to_id, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabelEncoders":
        field_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
        field_to_id = {
            field: {
                str(label): int(label_id)
                for label, label_id in mapping.items()
            }
            for field, mapping in field_to_id.items()
        }

        field_to_label = {
            field: {
                label_id: label
                for label, label_id in mapping.items()
            }
            for field, mapping in field_to_id.items()
        }

        return cls(
            field_to_id=field_to_id,
            field_to_label=field_to_label,
        )


def load_records(path: str | Path) -> list[Record]:
    source = Path(path)
    with source.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


class SmartHomeCommandDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[Record],
        tokenizer: WordTokenizer,
        label_encoders: LabelEncoders,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        input_ids, attention_mask = self.tokenizer.encode(
            str(record["command"])
        )
        labels = record["labels"]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        for field in LABEL_FIELDS:
            item[field] = torch.tensor(
                self.label_encoders.encode(
                    field,
                    str(labels[field]),
                ),
                dtype=torch.long,
            )

        return item


def summarize_records(records: list[Record]) -> dict[str, dict[str, int]]:
    return {
        field: dict(
            sorted(
                Counter(
                    str(record["labels"][field])
                    for record in records
                ).items()
            )
        )
        for field in LABEL_FIELDS
    }