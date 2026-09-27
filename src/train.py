import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data import get_loaders, NUM_CLASSES
from models import build_model


def evaluate(model, loader, device, per_class=False):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    if per_class:
        p_c, r_c, f_c, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        out["per_class"] = {
            "precision": p_c.tolist(), "recall": r_c.tolist(), "f1": f_c.tolist(), "support": support.tolist(),
        }
        out["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn", "transfer"], required=True)
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)  # L2
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = NUM_CLASSES[args.dataset]
    train_loader, test_loader = get_loaders(
        dataset=args.dataset, batch_size=args.batch_size, augment=not args.no_augment
    )
    model = build_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        history.append(metrics)
        print(f"epoch {epoch+1}/{args.epochs} acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
    train_time = time.time() - start

    final = evaluate(model, test_loader, device, per_class=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results_dir / f"{args.model}_{args.dataset}.pt")
    out = {
        "model": args.model,
        "dataset": args.dataset,
        **final,
        "trainable_params": n_params,
        "train_time_sec": train_time,
        "epochs": args.epochs,
        "device": str(device),
        "augment": not args.no_augment,
        "history": history,
    }
    with open(results_dir / f"{args.model}_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)
    print({k: v for k, v in out.items() if k not in ("history", "per_class", "confusion_matrix")})


if __name__ == "__main__":
    main()
