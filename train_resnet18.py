"""Train a ResNet18 on CIFAR-10 and evaluate the best validation checkpoint.

The original 3-layer CNN is in main.py. This experiment uses ResNet18 trained
from scratch (no pretrained ImageNet weights). Tested in Google Colab on T4.
"""
import argparse
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import classification_report

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    model = resnet18(weights=None, num_classes=10)
    # Standard ResNet's 7x7/stride-2 stem is too aggressive for 32x32 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    actual = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return total_loss / total, 100 * correct / total, actual, predictions


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet18 training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_full = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=test_transform
    )
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Same split as the Colab run: 40k train / 10k validation.
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    train_data = Subset(train_full, indices[:40000])
    val_data = Subset(val_full, indices[40000:])

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_accuracy = -1.0
    best_weights = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
            }, "best_resnet18_cifar10.pth")

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    model.load_state_dict(best_weights)
    _, test_accuracy, true_labels, predictions = evaluate(
        model, test_loader, criterion, device
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions, target_names=CLASSES, zero_division=0
    ))


if __name__ == "__main__":
    main()
