import json
from pathlib import Path

results_dir = Path(__file__).resolve().parent.parent / "results"
rows = [
    json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))
    if not f.stem.startswith(("eda_", "features_"))
]

if not rows:
    print("No results yet. Run train.py for each model first.")
else:
    cols = ["model", "dataset", "accuracy", "precision_macro", "recall_macro", "f1_macro", "trainable_params", "train_time_sec", "device"]
    print("".join(f"{c:<16}" for c in cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            elif isinstance(v, int) and c == "trainable_params":
                v = f"{v:,}"
            vals.append(str(v))
        print("".join(f"{v:<16}" for v in vals))
