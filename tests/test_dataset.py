from torch.utils.data import DataLoader

from smart_home_parser.dataset import (
    LabelEncoders,
    SmartHomeCommandDataset,
)
from smart_home_parser.tokenizer import WordTokenizer


def test_dataset_returns_expected_tensor_shapes() -> None:
    records = [
        {
            "command": "turn on the kitchen lights",
            "labels": {
                "intent": "device_control",
                "action": "turn_on",
                "device": "light",
                "location": "kitchen",
                "value": "none",
                "unit": "none",
            },
        },
        {
            "command": "start a timer for ten minutes",
            "labels": {
                "intent": "unsupported",
                "action": "none",
                "device": "none",
                "location": "none",
                "value": "none",
                "unit": "none",
            },
        },
    ]

    tokenizer = WordTokenizer(max_length=12)
    tokenizer.fit([record["command"] for record in records])
    encoders = LabelEncoders.fit(records)

    dataset = SmartHomeCommandDataset(
        records=records,
        tokenizer=tokenizer,
        label_encoders=encoders,
    )

    item = dataset[0]

    assert item["input_ids"].shape == (12,)
    assert item["attention_mask"].shape == (12,)
    assert item["intent"].ndim == 0
    assert item["action"].ndim == 0


def test_dataloader_batches_dataset() -> None:
    records = [
        {
            "command": "turn on the kitchen lights",
            "labels": {
                "intent": "device_control",
                "action": "turn_on",
                "device": "light",
                "location": "kitchen",
                "value": "none",
                "unit": "none",
            },
        },
        {
            "command": "turn off the bedroom lights",
            "labels": {
                "intent": "device_control",
                "action": "turn_off",
                "device": "light",
                "location": "bedroom",
                "value": "none",
                "unit": "none",
            },
        },
    ]

    tokenizer = WordTokenizer(max_length=10)
    tokenizer.fit([record["command"] for record in records])
    encoders = LabelEncoders.fit(records)

    dataset = SmartHomeCommandDataset(
        records=records,
        tokenizer=tokenizer,
        label_encoders=encoders,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 10)
    assert batch["attention_mask"].shape == (2, 10)
    assert batch["intent"].shape == (2,)
