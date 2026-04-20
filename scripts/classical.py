"""Classical baseline: color histogram + HOG + gradient boosting.

Features per image:
- HSV histogram: 16 bins on H, 8 on S, 8 on V → 32-dim
- HOG: 9 orientations, 8x8 cells, 2x2 block → ~1800-dim
- Mean + std of Lab channels → 6-dim

Model: XGBoost classifier. A legitimate non-DL ML pipeline.

Usage:
    python scripts/classical.py --train data/processed/train --test data/processed/test --out results/classical.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import hog

try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier


CLASSES = ["cell_phone", "cup", "headphone", "laptop", "scissors", "stapler"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def extract_features(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB").resize((128, 128))).astype(np.float32) / 255.0

    hsv = rgb2hsv(img)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1))
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=8, range=(0, 1))
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=8, range=(0, 1))
    color = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color = color / max(1e-9, color.sum())

    gray = img.mean(axis=2)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    lab = rgb2lab(img)
    lab_stats = np.concatenate([lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)])

    return np.concatenate([color, hog_feat, lab_stats]).astype(np.float32)


def load_split(root: Path):
    pairs = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls not in CLASSES:
            continue
        for img in cls_dir.glob("*.*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                pairs.append((img, cls))
    return pairs


def featurize(pairs):
    X, y = [], []
    for path, cls in pairs:
        try:
            X.append(extract_features(path))
            y.append(CLASS_IDX[cls])
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-model", type=Path, default=None)
    args = ap.parse_args()

    print("Extracting train features...")
    X_tr, y_tr = featurize(load_split(args.train))
    print(f"  train shape: {X_tr.shape}")
    print("Extracting test features...")
    X_te, y_te = featurize(load_split(args.test))
    print(f"  test shape:  {X_te.shape}")

    print("Training classifier...")
    try:
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1)
    except TypeError:
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    accuracy = float((preds == y_te).mean())
    per_class_f1 = {}
    for i, c in enumerate(CLASSES):
        tp = int(((preds == i) & (y_te == i)).sum())
        fp = int(((preds == i) & (y_te != i)).sum())
        fn = int(((preds != i) & (y_te == i)).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        per_class_f1[c] = 2 * p * r / max(1e-9, p + r)

    report = {
        "model": "classical_colorhist_hog_gbm",
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_dim": int(X_tr.shape[1]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        with args.save_model.open("wb") as f:
            pickle.dump(clf, f)

    print(f"Classical accuracy: {accuracy:.3f}  (n_test={report['n_test']})")
    print("Per-class F1: " + ", ".join(f"{c}={per_class_f1[c]:.2f}" for c in CLASSES))


if __name__ == "__main__":
    main()
