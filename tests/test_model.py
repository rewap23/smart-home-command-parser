import torch

from smart_home_parser.model import (
    ModelConfig,
    SmartHomeTransformer,
)


def make_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50,
        max_length=12,
        num_classes={
            "intent": 2,
            "action": 8,
            "device": 6,
            "location": 5,
            "value": 10,
            "unit": 3,
        },
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout=0.0,
    )


def test_model_returns_one_logits_tensor_per_field() -> None:
    config = make_config()
    model = SmartHomeTransformer(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(4, config.max_length),
    )
    attention_mask = torch.ones(
        (4, config.max_length),
        dtype=torch.bool,
    )

    outputs = model(input_ids, attention_mask)

    assert set(outputs) == {
        "intent",
        "action",
        "device",
        "location",
        "value",
        "unit",
    }
    assert outputs["intent"].shape == (4, 2)
    assert outputs["action"].shape == (4, 8)
    assert outputs["device"].shape == (4, 6)
    assert outputs["location"].shape == (4, 5)
    assert outputs["value"].shape == (4, 10)
    assert outputs["unit"].shape == (4, 3)


def test_model_handles_padding() -> None:
    config = make_config()
    model = SmartHomeTransformer(config)

    input_ids = torch.tensor(
        [
            [2, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    attention_mask = input_ids != 0

    outputs = model(input_ids, attention_mask)

    assert outputs["intent"].shape == (2, 2)


def test_model_rejects_too_long_sequence() -> None:
    config = make_config()
    model = SmartHomeTransformer(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 13))
    attention_mask = torch.ones((2, 13), dtype=torch.bool)

    try:
        model(input_ids, attention_mask)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("Expected ValueError for oversized sequence")
