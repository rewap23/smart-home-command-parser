import torch

from smart_home_parser.tokenizer import (
    CLS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    WordTokenizer,
)


def test_special_tokens_have_stable_ids() -> None:
    tokenizer = WordTokenizer()

    assert tokenizer.token_to_id[PAD_TOKEN] == 0
    assert tokenizer.token_to_id[UNK_TOKEN] == 1
    assert tokenizer.token_to_id[CLS_TOKEN] == 2


def test_normalization() -> None:
    tokenizer = WordTokenizer()

    assert tokenizer.tokenize(
        "  Please, TURN on the Kitchen lights!!! "
    ) == [
        "please",
        "turn",
        "on",
        "the",
        "kitchen",
        "lights",
    ]


def test_fit_and_encode_shape() -> None:
    tokenizer = WordTokenizer(max_length=8)
    tokenizer.fit(
        [
            "turn on the kitchen lights",
            "set bedroom temperature to 72",
        ]
    )

    input_ids, attention_mask = tokenizer.encode(
        "turn on the kitchen lights"
    )

    assert input_ids.shape == (8,)
    assert attention_mask.shape == (8,)
    assert input_ids[0].item() == tokenizer.cls_id
    assert attention_mask.sum().item() == 6


def test_unknown_token_maps_to_unk() -> None:
    tokenizer = WordTokenizer(max_length=5)
    tokenizer.fit(["turn on light"])

    input_ids, _ = tokenizer.encode("turn on chandelier")

    assert tokenizer.unk_id in input_ids.tolist()


def test_save_and_load_round_trip(tmp_path) -> None:
    tokenizer = WordTokenizer(max_length=12)
    tokenizer.fit(["turn on kitchen lights"])

    path = tmp_path / "vocab.json"
    tokenizer.save(path)

    loaded = WordTokenizer.load(path)

    assert loaded.max_length == 12
    assert loaded.token_to_id == tokenizer.token_to_id

    original_ids, original_mask = tokenizer.encode("turn on kitchen lights")
    loaded_ids, loaded_mask = loaded.encode("turn on kitchen lights")

    assert torch.equal(original_ids, loaded_ids)
    assert torch.equal(original_mask, loaded_mask)