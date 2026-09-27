"""Load a trained model's checkpoint and extract what it actually looks at:
vanilla-gradient saliency maps on sample images, plus learned first-layer filters/weights.
Needs the .pt checkpoint train.py saves — run after training, no retraining involved.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import get_loaders, NUM_CLASSES
from models import build_model

RESULTS = Path(__file__).resolve().parent.parent / "results"


def arr_to_b64(arr, size=112):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.resize((size, size), Image.NEAREST).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def saliency_maps(model, images, labels, device, n_per_class=1, num_classes=10):
    model.eval()
    picked = {}
    for img, lbl in zip(images, labels):
        c = int(lbl)
        if c not in picked and len(picked) < num_classes:
            picked[c] = img
    out = {}
    for c, img in picked.items():
        x = img.unsqueeze(0).to(device).requires_grad_(True)
        score = model(x)[0, c]
        score.backward()
        saliency = x.grad.abs()[0, 0].cpu().numpy()
        out[c] = {
            "input": arr_to_b64(img[0].numpy()),
            "saliency": arr_to_b64(saliency),
        }
    return out


def first_layer_filters(model, model_name, top_k=16):
    """What the first layer's weights look like — real learned features, not a proxy."""
    if model_name == "mlp":
        w = model.net[1].weight.detach().cpu().numpy()  # (256, 784)
        norms = np.linalg.norm(w, axis=1)
        idx = np.argsort(-norms)[:top_k]
        return [arr_to_b64(w[i].reshape(28, 28)) for i in idx]
    conv = model.features[0] if model_name == "cnn" else model.conv1
    w = conv.weight.detach().cpu().numpy()[:, 0]  # (out_ch, k, k)
    idx = range(min(top_k, w.shape[0]))
    return [arr_to_b64(w[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn", "transfer"], required=True)
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = NUM_CLASSES[args.dataset]
    ckpt = RESULTS / f"{args.model}_{args.dataset}.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} — run train.py first (it now saves .pt files)")

    model = build_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    _, test_loader = get_loaders(dataset=args.dataset, batch_size=256, augment=False)
    images, labels = next(iter(test_loader))

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "saliency": saliency_maps(model, images, labels, device, num_classes=min(num_classes, 12)),
        "first_layer_filters": first_layer_filters(model, args.model),
    }
    with open(RESULTS / f"features_{args.model}_{args.dataset}.json", "w") as f:
        json.dump(out, f)
    print(f"saved results/features_{args.model}_{args.dataset}.json")


if __name__ == "__main__":
    main()
