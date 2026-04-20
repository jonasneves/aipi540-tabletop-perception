"""Build a small tabletop-classification dataset from Caltech-101.

Torchvision's Caltech101 loader uses a dead Google Drive URL. This script pulls
the tarball directly from Caltech's data portal and re-organizes into an
ImageFolder layout for our 7-class taxonomy.

Usage:
    python scripts/make_dataset.py --out data/processed
"""
import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


SRC_TO_TARGET = {
    "cup": "cup",
    "laptop": "laptop",
    "cellphone": "cell_phone",
    "headphone": "headphone",
    "scissors": "scissors",
    "stapler": "stapler",
}

TARGET_CLASSES = ["cell_phone", "cup", "headphone", "laptop", "scissors", "stapler"]

DATA_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  already present: {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    subprocess.run(["curl", "-L", "--fail", "--progress-bar", "-o", str(dest), url], check=True)


def unpack(zip_path: Path, dest_dir: Path):
    """Caltech ships a zip containing a tar.gz of 101_ObjectCategories."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    # find nested tar.gz
    tar = next(dest_dir.rglob("101_ObjectCategories.tar.gz"), None)
    if tar is None:
        tar = next(dest_dir.rglob("*.tar.gz"), None)
    if tar is None:
        raise SystemExit("no inner tar.gz found under " + str(dest_dir))
    with tarfile.open(tar) as t:
        t.extractall(dest_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    zip_path = args.root / "caltech-101.zip"
    download(DATA_URL, zip_path)

    extracted_root = args.root / "caltech101_extracted"
    if not any(extracted_root.rglob("101_ObjectCategories")):
        extracted_root.mkdir(parents=True, exist_ok=True)
        unpack(zip_path, extracted_root)

    src_dir = next(extracted_root.rglob("101_ObjectCategories"), None)
    if src_dir is None:
        raise SystemExit(f"Could not locate 101_ObjectCategories under {extracted_root}")
    print(f"  source: {src_dir}")

    train_root = args.out / "train"
    test_root = args.out / "test"
    for r in (train_root, test_root):
        r.mkdir(parents=True, exist_ok=True)

    total = {"train": 0, "test": 0}
    for src_cls_dir in sorted(src_dir.iterdir()):
        if not src_cls_dir.is_dir():
            continue
        target = SRC_TO_TARGET.get(src_cls_dir.name)
        if target is None:
            continue
        imgs = sorted(src_cls_dir.glob("*.jpg"))
        n_test = max(1, int(len(imgs) * args.test_frac))
        test_imgs = imgs[:n_test]
        train_imgs = imgs[n_test:]
        for dst_root, subset, subset_name in [(train_root, train_imgs, "train"), (test_root, test_imgs, "test")]:
            cls_out = dst_root / target
            cls_out.mkdir(parents=True, exist_ok=True)
            for im in subset:
                shutil.copy(im, cls_out / f"{src_cls_dir.name}_{im.name}")
                total[subset_name] += 1
        print(f"  {src_cls_dir.name:25s} -> {target:12s}  {len(train_imgs)} train / {len(test_imgs)} test")

    print(f"\nTotal: {total['train']} train / {total['test']} test")
    missing = [c for c in TARGET_CLASSES if not (train_root / c).exists() or not any((train_root / c).iterdir())]
    if missing:
        print(f"\nMissing classes (not in Caltech-101): {missing}")
        print("Run scripts/augment_dataset.py to add these via image scrape.")


if __name__ == "__main__":
    main()
