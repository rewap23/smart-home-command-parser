from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_home_parser.dataset import (
    LABEL_FIELDS,
    LabelEncoders,
    SmartHomeCommandDataset,
    load_records,
)
from smart_home_parser.model import ModelConfig, SmartHomeTransformer
from smart_home_parser.tokenizer import WordTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device)
        for name, tensor in batch.items()
    }


def calculate_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    losses = [
        criterion(logits[field], batch[field])
        for field in LABEL_FIELDS
    ]
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: SmartHomeTransformer,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    correct_by_field = {field: 0 for field in LABEL_FIELDS}
    exact_matches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
        )
        loss = calculate_loss(logits, batch, criterion)

        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = {
            field: logits[field].argmax(dim=1)
            for field in LABEL_FIELDS
        }

        exact_match_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for field in LABEL_FIELDS:
            matches = predictions[field] == batch[field]
            correct_by_field[field] += matches.sum().item()
            exact_match_mask &= matches

        exact_matches += exact_match_mask.sum().item()

    return {
        "loss": total_loss / total_examples,
        **{
            f"{field}_accuracy": correct_by_field[field] / total_examples
            for field in LABEL_FIELDS
        },
        "exact_match_accuracy": exact_matches / total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.jsonl"),
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA-enabled GPU is available."
        )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.train_path)
    validation_records = load_records(args.validation_path)

    tokenizer = WordTokenizer(max_length=args.max_length)
    tokenizer.fit(
        [str(record["command"]) for record in train_records]
    )

    label_encoders = LabelEncoders.fit(train_records)

    train_dataset = SmartHomeCommandDataset(
        records=train_records,
        tokenizer=tokenizer,
        label_encoders=label_encoders,
    )
    validation_dataset = SmartHomeCommandDataset(
        records=validation_records,
        tokenizer=tokenizer,
        label_encoders=label_encoders,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        num_classes={
            field: label_encoders.num_classes(field)
            for field in LABEL_FIELDS
        },
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )

    model = SmartHomeTransformer(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(f"Using device: {device}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        processed = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for batch in progress:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
            )
            loss = calculate_loss(logits, batch, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            batch_size = batch["input_ids"].size(0)
            running_loss += loss.item() * batch_size
            processed += batch_size

            progress.set_postfix(
                train_loss=f"{running_loss / processed:.4f}"
            )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            criterion,
        )

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": running_loss / processed,
            **validation_metrics,
        }
        history.append(epoch_metrics)

        print(json.dumps(epoch_metrics, indent=2))

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                args.artifact_dir / "model.pt",
            )

    tokenizer.save(args.artifact_dir / "vocab.json")
    label_encoders.save(args.artifact_dir / "label_encoders.json")

    (args.artifact_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "model_name": "smart-home-command-parser-transformer",
        "model_version": "0.1.0",
        "task": "multi-head smart-home command semantic parsing",
        "framework": "pytorch",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_device": str(device),
        "training_seed": args.seed,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "vocab_size": tokenizer.vocab_size,
        "model_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "best_validation_loss": best_validation_loss,
        "best_epoch": min(
            history,
            key=lambda metrics: float(metrics["loss"]),
        )["epoch"],
        "label_fields": list(LABEL_FIELDS),
    }
    (args.artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved artifacts to: {args.artifact_dir}")


if __name__ == "__main__":
    main()