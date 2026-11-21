# CIFAR-10 image classification with PyTorch

Two experiments on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): a simple CNN baseline and an improved ResNet18 trained from scratch. CIFAR-10 has 50,000 training images and 10,000 test images (32×32 RGB, ten categories).

## Results

| Experiment | Best validation accuracy | Test accuracy | Source |
| --- | ---: | ---: | --- |
| Original 3-layer CNN | not reliably recorded | 74.15% | Original `cifar10_cnn_pytorch.ipynb` Colab output |
| ResNet18 (30 epochs) | **92.81%** | **92.59%** | Colab training/evaluation output, 21 Nov 2026 |

**Improvement: +18.44 percentage points of test accuracy** between the two experiments. Both used CIFAR-10, but model architecture, augmentation and training setup changed, so this is not a controlled single-variable comparison.

For the ResNet18 run: 40,000 training images, 10,000 validation images, and a separate 10,000-image test set. Final training accuracy was 99.11%, and the classification report gave approximately 0.93 macro F1. The weakest class was **cat** (recall 0.82).

The **92.59%** figure is from a completed user-run Google Colab T4 GPU experiment, not a benchmark that was independently rerun as part of this repository update. Reproduction results may vary.

## ResNet18 experiment

The code is in [`train_resnet18.py`](train_resnet18.py). It uses:
- ResNet18 with a 3×3 stride-1 first convolution and no initial max pool, adapted for 32×32 images
- Random crop, horizontal flip, and CIFAR-10 channel normalization
- Batch normalization and residual connections from torchvision ResNet18
- 30 epochs, SGD (LR 0.1, momentum 0.9, weight decay 0.0005)
- Cosine learning rate schedule and cross-entropy label smoothing (0.1)
- Fixed-seed 40k/10k train-validation split, distinct 10k test set
- Best-validation checkpoint selection before final test evaluation

No ImageNet pretrained weights or OpenAI API key were used.

### Run in Google Colab

Choose **Runtime → Change runtime type → T4 GPU**, if available. Clone this repo and run:

```python
!git clone https://github.com/Balaram325/cifar10-cnn-pytorch.git
%cd cifar10-cnn-pytorch
!python train_resnet18.py --epochs 30 --batch-size 128 --lr 0.1
```

Colab usually already includes torch, torchvision, numpy and scikit-learn. If a package is missing, install only that package rather than reinstalling everything.

The first run downloads CIFAR-10. The best validation checkpoint is saved to `best_resnet18_cifar10.pth` (gitignored). Download or move it to Drive before the Colab runtime disconnects.

If you paste the script *directly into a Colab notebook cell*, change `parser.parse_args()` to `parser.parse_args(args=[])` so Colab's notebook arguments are ignored. Running the script as a file needs no such change.

### Original CNN

- [`cifar10_cnn_pytorch.ipynb`](cifar10_cnn_pytorch.ipynb) — initial Colab experiment, 74.15% saved test output
- [`main.py`](main.py) — reusable 3-layer CNN baseline with fixed split, saved metrics and checkpoint

```bash
python main.py --epochs 10 --batch-size 64 --seed 42
```

The newer `main.py` baseline has not been reported as reproducing the notebook's original 74.15% figure.

## Dependencies and tests

Use an existing PyTorch environment when possible to save disk space. Otherwise install a compatible PyTorch/torchvision pair for your device from [pytorch.org](https://pytorch.org/get-started/locally/) and:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

These are quick code-level tests, **not** a substitute for training and measuring accuracy.

## Limitations

The trained checkpoint and raw dataset are not in GitHub. The results are from individual experimental runs rather than repeated-seed studies. Strongest and weakest class behaviour can change with different settings. For further improvement, compare per-class errors and run additional seeds or an ablation study rather than assuming added layers always help.
