from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from smart_home_parser.dataset import (
    LABEL_FIELDS,
    LabelEncoders,
    SmartHomeCommandDataset,
    load_records,
)
from smart_home_parser.model import ModelConfig, SmartHomeTransformer
from smart_home_parser.tokenizer import WordTokenizer
from smart_home_parser.train import evaluate


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return torch.device(requested_device)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Smart Home Command Parser checkpoint "
            "on a held-out test split."
        )
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    test_path = args.test_path.resolve()
    output_path = artifact_dir / "metrics.json"

    print(f"Working directory: {Path.cwd()}")
    print(f"Test split: {test_path}")
    print(f"Artifact directory: {artifact_dir}")
    print(f"Metrics output path: {output_path}")

    checkpoint_path = artifact_dir / "model.pt"
    vocab_path = artifact_dir / "vocab.json"
    encoder_path = artifact_dir / "label_encoders.json"

    for required_path in (
        test_path,
        checkpoint_path,
        vocab_path,
        encoder_path,
    ):
        require_file(required_path)

    device = resolve_device(args.device)
    print(f"Evaluation device: {device}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_checkpoint_keys = {
        "model_state_dict",
        "model_config",
        "epoch",
    }
    missing_keys = required_checkpoint_keys - set(checkpoint)
    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    config = ModelConfig(**checkpoint["model_config"])
    model = SmartHomeTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = WordTokenizer.load(vocab_path)
    label_encoders = LabelEncoders.load(encoder_path)

    test_records = load_records(test_path)
    if not test_records:
        raise ValueError(f"Test split is empty: {test_path}")

    test_dataset = SmartHomeCommandDataset(
        records=test_records,
        tokenizer=tokenizer,
        label_encoders=label_encoders,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=nn.CrossEntropyLoss(),
    )

    output: dict[str, object] = {
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "split": "test",
        "examples": len(test_dataset),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "model_name": "smart-home-command-parser-transformer",
        "label_fields": list(LABEL_FIELDS),
        **metrics,
    }

    atomic_write_json(output_path, output)

    if not output_path.is_file():
        raise RuntimeError(
            f"Evaluation completed but metrics file was not created: "
            f"{output_path}"
        )

    print("\nEvaluation metrics:")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nSUCCESS: wrote metrics to {output_path}")


if __name__ == "__main__":
    main()