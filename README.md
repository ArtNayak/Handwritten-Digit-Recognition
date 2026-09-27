# DLA Capstone — Handwritten Digit/Character Recognition

MLP vs CNN vs Transfer Learning on MNIST and EMNIST, with Dropout + L2 regularization on the baselines and a fine-tuned ResNet18 for transfer learning.

## Project layout

```
src/
  data.py               # MNIST/EMNIST loaders, EMNIST orientation fix, train-time augmentation
  models.py             # MLP, CNN, and transfer (resnet18) model builders
  train.py              # single CLI entrypoint: trains one model on one dataset
  eda.py                # dataset EDA: shape, pixel stats, class balance, NaN/blank checks
  feature_importance.py # per-model feature/prediction inspection
  compare.py             # reads results/*.json and prints a comparison table
  build_dashboard.py    # renders dashboard.html from results/
  test_smoke.py         # shape/forward-pass sanity check for models.py and data.py
results/                # one {model}_{dataset}.json (+ .pt weights) per run
docs/ADR.md             # design decisions and rationale
dashboard.html          # generated results dashboard
```

## Setup

```
pip install -r requirements.txt
```

Requires `torch`, `torchvision`, and `scikit-learn` (for precision/recall/F1). CUDA is used automatically if available, otherwise falls back to CPU — no flag needed.

## Usage

Train a model:

```
python src/train.py --model mlp|cnn|transfer --dataset mnist|emnist [--epochs 30] [--lr 1e-3] [--weight-decay 1e-4] [--batch-size 128] [--no-augment]
```

Each run writes `results/{model}_{dataset}.json` (accuracy, loss, macro precision/recall/F1, per-class metrics, confusion matrix, per-epoch history) and `results/{model}_{dataset}.pt` (weights).

Compare results:

```
python src/compare.py
```

Run the smoke test after touching `models.py` or `data.py`:

```
python src/test_smoke.py
```

Regenerate the dashboard:

```
python src/build_dashboard.py
```

## Models

- **MLP**: 2 hidden layers, Dropout + L2 (`weight_decay`).
- **CNN**: 2 conv blocks, Dropout + L2.
- **Transfer**: torchvision `resnet18` (ImageNet weights), first conv swapped for 1-channel input; `layer1`–`layer3` frozen, `layer4` + new FC head fine-tuned.

Training uses Adam, `CosineAnnealingLR`, and (for EMNIST) `RandomRotation`/`RandomAffine` augmentation. EMNIST uses the `balanced` split (47 classes).

See [docs/ADR.md](docs/ADR.md) for the full rationale behind these choices.
