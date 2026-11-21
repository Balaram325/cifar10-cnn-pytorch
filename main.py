"""Reproducible CIFAR-10 CNN baseline. Run python main.py --help."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_indices(count: int, validation_fraction: float, seed: int):
    if not 0 < validation_fraction < 1 or count < 2:
        raise ValueError("Need at least two samples and a validation fraction between 0 and 1")
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    val_count = max(1, min(count - 1, int(count * validation_fraction)))
    return indices[val_count:], indices[:val_count]


def evaluate(model, loader, device, criterion, with_predictions=False):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    predictions, labels_all = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch = labels.size(0)
            loss_sum += criterion(logits, labels).item() * batch
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            count += batch
            if with_predictions:
                predictions.extend(pred.cpu().tolist())
                labels_all.extend(labels.cpu().tolist())
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}, predictions, labels_all


def run(args):
    from torchvision import datasets, transforms

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    # Separate dataset transforms prevent validation from receiving train-time augmentation.
    train_data = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    val_data = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=eval_tf)
    test_data = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_tf)
    train_indices, val_indices = split_indices(len(train_data), args.val_fraction, args.seed)
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_data, train_indices), shuffle=True, **common)
    val_loader = DataLoader(Subset(val_data, val_indices), shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_accuracy = -1.0
    checkpoint = output / "best_cnn.pt"

    print(f"Device: {device}, train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_data)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
        val, _, _ = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": train_loss_sum / seen, "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
        history.append(row)
        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]
            torch.save(model.state_dict(), checkpoint)
        print(f"Epoch {epoch}/{args.epochs} train loss={row['train_loss']:.4f} val loss={val['loss']:.4f} val accuracy={val['accuracy']:.2%}")

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test, preds, targets = evaluate(model, test_loader, device, criterion, with_predictions=True)
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(targets, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "architecture": "3-layer CNN baseline", "seed": args.seed, "epochs": args.epochs,
        "best_validation_accuracy": best_val_accuracy, "test": test,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classes": list(CLASSES), "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "classification_report": report
    }
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test accuracy={test['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}")
    print(f"Saved evaluation and best checkpoint under {output}")
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate a reproducible CIFAR-10 baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.workers < 0:
        parser.error("epochs, batch size and learning rate must be positive; workers must be nonnegative")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between 0 and 1")
    return args


if __name__ == "__main__":
    run(parse_args())
