"""Load raw data, describe it, check for issues, explain preprocessing — persisted to JSON for the dashboard."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from data import MEAN, STD, _orientation_fix

# Official EMNIST "balanced" class mapping (47 classes): 0-9 digits, 10-35 A-Z,
# 36-46 the 11 lowercase letters that look visually distinct from their uppercase form.
EMNIST_BALANCED_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")


def to_b64_png(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.resize((56, 56), Image.NEAREST).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def describe(dataset_name, split_name, ds):
    imgs = ds.data.numpy() if hasattr(ds.data, "numpy") else np.array(ds.data)
    labels = ds.targets.numpy() if hasattr(ds.targets, "numpy") else np.array(ds.targets)
    blank = int((imgs.reshape(len(imgs), -1).sum(axis=1) == 0).sum())
    nan_count = int(np.isnan(imgs.astype(np.float32)).sum())
    counts = np.bincount(labels).tolist()
    stats = {
        "samples": len(ds),
        "image_shape": list(imgs.shape[1:]),
        "dtype": str(imgs.dtype),
        "pixel_min": int(imgs.min()),
        "pixel_max": int(imgs.max()),
        "pixel_mean": float(imgs.mean()),
        "pixel_std": float(imgs.std()),
        "num_classes": len(counts),
        "blank_images": blank,
        "nan_pixels": nan_count,
        "class_counts": counts,
    }
    print(f"\n--- {dataset_name} [{split_name}] ---")
    print(f"samples: {stats['samples']}  shape: {stats['image_shape']}  dtype: {stats['dtype']}")
    print(f"pixel range: [{stats['pixel_min']}, {stats['pixel_max']}]  mean: {stats['pixel_mean']:.2f}")
    print(f"blank images: {blank}  NaN pixels: {nan_count}  classes: {stats['num_classes']}")
    return stats, imgs, labels


def sample_images(dataset, imgs, labels, num_classes, augment_tfm):
    """One raw + one augmented sample per class, base64 PNG, for the dashboard."""
    raw, augmented = {}, {}
    for c in range(num_classes):
        idx = np.where(labels == c)[0][0]
        arr = imgs[idx]
        pil = Image.fromarray(arr.astype(np.uint8))
        if dataset == "emnist":
            pil = _orientation_fix(pil)
            arr = np.array(pil)
        raw[c] = to_b64_png(arr)
        aug_pil = augment_tfm(pil)
        augmented[c] = to_b64_png(np.array(aug_pil))
    return raw, augmented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    args = ap.parse_args()

    if args.dataset == "mnist":
        train = datasets.MNIST("./data", train=True, download=True)
        test = datasets.MNIST("./data", train=False, download=True)
        labels_map = [str(i) for i in range(10)]
    else:
        train = datasets.EMNIST("./data", split="balanced", train=True, download=True)
        test = datasets.EMNIST("./data", split="balanced", train=False, download=True)
        labels_map = EMNIST_BALANCED_LABELS

    train_stats, train_imgs, train_labels = describe(args.dataset, "train", train)
    test_stats, test_imgs, test_labels = describe(args.dataset, "test", test)

    print("\n--- cleaning ---")
    total_blank = train_stats["blank_images"] + test_stats["blank_images"]
    total_nan = train_stats["nan_pixels"] + test_stats["nan_pixels"]
    if total_blank + total_nan == 0:
        clean_note = "no blank images or NaNs found — dataset is already clean, no rows dropped."
    else:
        clean_note = f"dropped {total_blank} blank images and samples with {total_nan} NaN pixels before training."
    print(clean_note)

    preprocessing_steps = [
        f"ToTensor(): uint8 [0,255] -> float32 [0,1]",
        f"Normalize(mean={MEAN}, std={STD}): standardizes to ~N(0,1), computed from MNIST train set",
        "train split only: RandomRotation(10) + RandomAffine(translate=0.1) augmentation",
    ]
    if args.dataset == "emnist":
        preprocessing_steps.insert(0, "orientation fix: rotate -90 deg + horizontal flip (EMNIST is stored transposed)")

    augment_tfm = transforms.Compose([transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.1, 0.1))])
    raw_samples, aug_samples = sample_images(args.dataset, train_imgs, train_labels, len(labels_map), augment_tfm)

    out = {
        "dataset": args.dataset,
        "class_labels": labels_map,
        "train": train_stats,
        "test": test_stats,
        "clean_note": clean_note,
        "preprocessing_steps": preprocessing_steps,
        "sample_images_raw": raw_samples,
        "sample_images_augmented": aug_samples,
    }
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"eda_{args.dataset}.json", "w") as f:
        json.dump(out, f)
    print(f"\nsaved results/eda_{args.dataset}.json")


if __name__ == "__main__":
    main()
