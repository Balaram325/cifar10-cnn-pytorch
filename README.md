# CIFAR-10 Image Classification — PyTorch CNN

A beginner-friendly **3-layer convolutional neural network** for classifying CIFAR-10 images into ten categories. Includes the original Google Colab notebook and a reproducible training/evaluation script.

## What is implemented
- 3 convolutional layers (32 → 64 → 128 filters), max-pooling, fully connected classifier
- Cross-entropy loss and Adam optimiser
- Fixed random seed (42 by default), **40,000 train / 10,000 validation** examples, and independent **10,000-image test set**
- Train-only random crop and horizontal flip; consistent validation/test normalisation
- Saves best checkpoint by validation accuracy; evaluates the selected model on the test set
- Outputs per-class precision, recall, F1, confusion matrix, training history, and overall test accuracy
- No API key, pretrained model, or paid service required

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 RGB 32×32 images in ten classes, with 50,000 training and 10,000 test images. The dataset is downloaded by torchvision on the first training run. **Do not commit the downloaded dataset or checkpoints.**

## Installation
Python 3.10+ recommended. Use an existing PyTorch environment or Google Colab to avoid reinstalling large dependencies. If installing locally, use the appropriate PyTorch build for your system from https://pytorch.org/get-started/locally/ and then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

On a laptop without a suitable GPU, Colab is generally much faster. The script supports CPU or CUDA and defaults to 0 dataloader workers for Windows compatibility.

## Run training and evaluation
```bash
python main.py --epochs 10 --batch-size 64 --seed 42
```

Outputs (all ignored by Git):
- `outputs/best_cnn.pt` — best validation checkpoint
- `outputs/history.csv` — epoch-level losses and validation accuracy
- `outputs/metrics.json` — test accuracy, F1 and confusion matrix

**Important:** The repository's original notebook (`cifar10_cnn_pytorch.ipynb`) shows a historical **74.15% test accuracy** from its saved Colab run, including a Tesla T4. That number is **not** a newly reproduced result from `main.py`. It is a learning baseline, not a state-of-the-art benchmark.

## Tests (no CIFAR download needed)
```bash
python -m unittest discover -s tests -v
```

These tests check model dimensions, reproducible disjoint splitting, argument defaults, and evaluation accounting. They do not replace a full training run.

## Limitations and next steps
- This is a basic CNN baseline, not a production computer-vision application.
- Random augmentation, hardware, installed PyTorch version and training time may change the obtained metrics.
- For stronger portfolio work, compare training curves, inspect class-level confusion, try ResNet18 transfer learning and document any improvement with test results.
- The notebook is preserved as the original experiment; `main.py` is the recommended reproducible route.
