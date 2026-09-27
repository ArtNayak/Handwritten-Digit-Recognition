# ADR: Handwritten Digit/Character Recognition Capstone

## Status
Accepted

## Context
DLA subject capstone. Requirement: MLP + CNN baselines with regularization
(Dropout, L2), plus a comparison against basic Transfer Learning. No novelty
needed — a working, explainable, one-night project.

## Decisions

1. **Dataset: MNIST only (EMNIST deferred).**
   MNIST is the primary deliverable (digits). torchvision ships EMNIST too —
   switching is a `--dataset emnist` flag in `data.py`, not built now.
   Why: MNIST alone already exercises every required technique; EMNIST just
   adds download/train time without changing the comparison story.

2. **Framework: PyTorch + torchvision.**
   Already installed, has MNIST/EMNIST loaders and pretrained models built
   in — no dataset wrangling or manual weight downloads needed.

3. **Models:**
   - MLP: 2 hidden layers, Dropout + `weight_decay` (L2) via optimizer.
   - CNN: 2 conv blocks + Dropout + `weight_decay`.
   - Transfer: torchvision `resnet18(weights=IMAGENET1K_V1)`, first conv
     swapped to accept 1-channel input, backbone frozen, new FC head trained.
   Why resnet18: smallest common pretrained CNN in torchvision, fits "basic
   transfer learning" requirement without picking/tuning a bigger model.

4. **Training: single `train.py` CLI, `--model mlp|cnn|transfer`.**
   Each run writes one `results/<model>.json` (test accuracy, loss, params,
   train time). `compare.py` just reads the three JSONs and prints a table.
   Why: no experiment-tracking framework needed for 3 runs.

5. **Hardware: CUDA if available, else CPU (auto-detected).**
   RTX 4050 6GB is enough for resnet18 fine-tuning at MNIST resolution;
   CPU fallback works too since MNIST is small (28x28, 60k images).

6. **No data augmentation, no LR scheduler, no config framework.**
   Fixed hyperparameters in `train.py` argparse defaults. Add a scheduler
   only if accuracy plateaus below target.

## Consequences
- Fast to build and run (each model trains in a few minutes).
- EMNIST/augmentation/scheduling are one-line extensions if asked for later,
  not built speculatively now.

## Amendment (2026-09-26): full scope requested

User asked for everything originally deferred. Added:
- `eda.py`: loads raw MNIST/EMNIST, describes shape/pixel stats/class balance,
  checks for blank images and NaNs (found none — both datasets are clean),
  documents the preprocessing pipeline. No cleaning code needed since nothing
  was dirty; describing that fact is the "clean" step for these datasets.
- `data.py`: `--dataset mnist|emnist` (EMNIST `balanced` split, 47 classes,
  digits+upper+lower merged case-insensitively where confusable — the
  standard split for combined digit+character recognition). EMNIST ships
  transposed vs MNIST convention; fixed with a rotate(-90)+hflip pre-transform.
  Train-only augmentation: `RandomRotation(10)` + `RandomAffine(translate=0.1)`.
- `models.py`: transfer model now unfreezes `layer4` (last resnet18 block) in
  addition to the new conv1/fc — a genuine "unfreeze more" step past
  head-only fine-tuning, while layer1-3 (generic low/mid-level features)
  stay frozen since re-training them needs far more data/epochs to help.
- `train.py`: default epochs 30, `CosineAnnealingLR` scheduler, and
  precision/recall/F1 (macro-averaged, via sklearn) alongside accuracy —
  macro because both datasets have imbalance or many classes and macro
  weights every class equally rather than letting frequent classes dominate.
  Per-epoch history saved for later inspection/plotting.
