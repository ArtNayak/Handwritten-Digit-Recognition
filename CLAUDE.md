# DLA Capstone — Handwritten Digit Recognition

MLP vs CNN vs Transfer Learning on MNIST. See `docs/ADR.md` for design decisions.

## Rules
- One JSON result file per model+dataset in `results/` (`{model}_{dataset}.json`), `compare.py` just reads them. Don't build a tracking framework.
- Keep `train.py` a single CLI script (`--model mlp|cnn|transfer --dataset mnist|emnist`). Don't split into a package unless it grows past this.
- GPU (CUDA) is available and installed — code must auto-detect `cuda` vs `cpu`, never hardcode either.
- Run `test_smoke.py` after touching `models.py` or `data.py` — it's the one check that catches shape/forward-pass breakage before a full 30-epoch run.
- Don't add a config framework, experiment tracker, or model beyond MLP/CNN/transfer unless asked — see ADR.
