"""DL model: MobileNetV3-small fine-tune + SSDLite-MobileNetV3 ONNX export.

Two artifacts, same MobileNet family:
- `models/classifier.onnx` — MobileNetV3-small backbone + fresh 7-class head,
  fine-tuned on Caltech-101 crops. For closed-set evaluation.
- `models/detector.onnx` — SSDLite-MobileNetV3-Large, COCO-pretrained, unchanged.
  For the live demo (bounding boxes on webcam input).

Runs on CPU (fast enough) or MPS. Trainable head is ~9K params.

Usage:
    python scripts/train_dl.py --train data/processed/train --test data/processed/test --out-dir models
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models


CLASSES = ["book", "bottle", "cell_phone", "cup", "keyboard", "laptop", "mouse"]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(train_dir, test_dir, batch=32):
    train_tf = transforms.Compose([
        transforms.Resize(232), transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(232), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)
    return train_dl, test_dl, train_ds.classes


def build_classifier(n_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    in_feat = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feat, n_classes)
    return m


def train(model, loaders, epochs=12, lr=1e-3, dev=None):
    train_dl, test_dl, _ = loaders
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.to(dev)

    best_acc = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        running, total = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); total += x.size(0)
        sched.step()
        train_loss = running / max(1, total)

        model.eval()
        correct, total_t = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(dev), y.to(dev)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total_t += x.size(0)
        acc = correct / max(1, total_t)
        history.append({"epoch": ep, "train_loss": train_loss, "test_acc": acc})
        print(f"  epoch {ep+1:2d}/{epochs}  train_loss={train_loss:.3f}  test_acc={acc:.3f}")
        best_acc = max(best_acc, acc)
    return history, best_acc


def per_class_metrics(model, test_dl, classes, dev):
    model.eval()
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(dev)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            for p, t in zip(pred, y):
                pc, tc = classes[p], classes[t]
                if p == t:
                    per[tc]["tp"] += 1
                else:
                    per[tc]["fn"] += 1
                    per[pc]["fp"] += 1
    f1 = {}
    for c, s in per.items():
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        f1[c] = 2 * p * r / max(1e-9, p + r)
    return f1


def export_classifier_onnx(model, out_path: Path):
    model.cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"  classifier.onnx -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def export_detector_onnx(out_path: Path):
    det = models.detection.ssdlite320_mobilenet_v3_large(
        weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    )
    det.eval()
    dummy = torch.randn(1, 3, 320, 320)
    torch.onnx.export(
        det, dummy, str(out_path),
        input_names=["input"], output_names=["boxes", "scores", "labels"],
        opset_version=14,
    )
    print(f"  detector.onnx  -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--skip-detector", action="store_true")
    args = ap.parse_args()

    dev = device()
    print(f"Device: {dev}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(args.train, args.test)
    n = len(loaders[2])
    print(f"Classes available: {loaders[2]} (n={n})")

    model = build_classifier(n)
    history, best_acc = train(model, loaders, epochs=args.epochs, dev=dev)
    f1 = per_class_metrics(model, loaders[1], loaders[2], dev)

    report = {
        "model": "dl_mobilenetv3_small_head_finetune",
        "classes": loaders[2],
        "accuracy": best_acc,
        "per_class_f1": f1,
        "history": history,
        "epochs": args.epochs,
    }
    (args.results_dir / "dl.json").write_text(json.dumps(report, indent=2))

    export_classifier_onnx(model, args.out_dir / "classifier.onnx")
    if not args.skip_detector:
        export_detector_onnx(args.out_dir / "detector.onnx")

    print(f"\nDL accuracy: {best_acc:.3f}")
    print("Per-class F1: " + ", ".join(f"{c}={f1[c]:.2f}" for c in loaders[2]))


if __name__ == "__main__":
    main()
