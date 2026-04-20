"""Naive baseline: dominant-hue classifier.

For each image: extract HSV, take median hue of non-dark pixels, map to a class
by fixed hue bins. A deliberately weak baseline that measures how much color
alone buys on this task.

Usage:
    python scripts/naive.py --train data/processed/train --test data/processed/test --out results/naive.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ["cell_phone", "cup", "headphone", "laptop", "scissors", "stapler"]


def dominant_hue(path: Path) -> float:
    img = Image.open(path).convert("HSV").resize((128, 128))
    arr = np.array(img).reshape(-1, 3)
    # drop very dark and very desaturated pixels
    mask = (arr[:, 2] > 40) & (arr[:, 1] > 30)
    if mask.sum() < 50:
        return float(np.median(arr[:, 0]))
    return float(np.median(arr[mask][:, 0]))


def fit(samples):
    """Learn a per-class median hue from training samples."""
    per_class = {c: [] for c in CLASSES}
    for path, cls in samples:
        per_class[cls].append(dominant_hue(path))
    centroids = {c: float(np.median(v)) if v else 128.0 for c, v in per_class.items()}
    return centroids


def predict(path: Path, centroids: dict) -> str:
    h = dominant_hue(path)
    best = min(centroids, key=lambda c: min(abs(h - centroids[c]), 255 - abs(h - centroids[c])))
    return best


def load_split(root: Path):
    """Load (path, class) pairs from a folder layout root/<class>/*.jpg."""
    pairs = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls not in CLASSES:
            continue
        for img in cls_dir.glob("*.*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                pairs.append((img, cls))
    return pairs


def evaluate(pairs, centroids):
    correct = 0
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    for path, gt in pairs:
        pred = predict(path, centroids)
        if pred == gt:
            correct += 1
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            per_class[pred]["fp"] += 1
    accuracy = correct / max(1, len(pairs))
    f1 = {}
    for c, s in per_class.items():
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        f1[c] = 2 * p * r / max(1e-9, p + r)
    return {"accuracy": accuracy, "per_class_f1": f1, "n": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train = load_split(args.train)
    test = load_split(args.test)
    centroids = fit(train)
    report = evaluate(test, centroids)
    report["centroids"] = centroids
    report["model"] = "naive_hsv_dominant_hue"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Naive accuracy: {report['accuracy']:.3f}  (n={report['n']})")
    print(f"Per-class F1: " + ", ".join(f"{c}={report['per_class_f1'][c]:.2f}" for c in CLASSES))


if __name__ == "__main__":
    main()
