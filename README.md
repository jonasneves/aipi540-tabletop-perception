# Tabletop Perception for Beginner Robot Kits

[![Live demo](https://img.shields.io/badge/demo-live-00539B)](https://neevs.io/aipi540-tabletop-perception/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime%20Web-WebGPU-00539B)](https://onnxruntime.ai/docs/tutorials/web/)
[![Duke AIPI 540](https://img.shields.io/badge/Duke-AIPI%20540-012169)](https://ai.meng.duke.edu/)

**[Live demo](https://neevs.io/aipi540-tabletop-perception/)** — point a webcam at a desk, watch a specialist and a generalist reason side-by-side. In-browser, no server, no API keys.

A fine-tuned 3M-parameter MobileNetV3-small hits **97% top-1 at ~15 ms/frame** on a 6-class tabletop task. A 450M-parameter open-vocab VLM (LFM2.5-VL-450M) on the same input runs **~85× slower** and collapses to 0% IoU on stylized out-of-distribution scenes, but answers open-vocabulary queries the specialist cannot attempt. Production shape: both running at once — specialist on the webcam stream, generalist on typed queries — with the generalist's honest refusals as a safety margin against silent false positives.

Choosing between them is a deployment decision, not a benchmark one.

## Results

| Tier | Model | Params | Latency | Top-1 |
|---|---|---|---|---|
| Naive | HSV threshold | 0 | ~1 ms | 24% |
| Classical | Color-hist + HOG + GBM | ~2.4K trees | ~40 ms | 76% |
| DL | MobileNetV3-small fine-tune | 2.5M frozen + 6K head | ~15 ms | **97%** |
| VLM | LFM2.5-VL-450M zero-shot | 450M | ~1300 ms | open-vocab |

Per-class F1, latency breakdown, and a five-case error analysis: [`report/report.md`](report/report.md).

## What you get

- **Live demo** — webcam in, specialist and VLM both running in-browser via ONNX Runtime Web + WebGPU. No server, no API keys.
- **Report** — 15 sections: problem, data, models, experiment, error analysis, commercial viability, ethics. See [`report/report.md`](report/report.md).
- **Reproducible training** — `scripts/train_dl.py` fine-tunes MobileNetV3-small on Caltech-101 and exports ONNX. Run end-to-end via `make eval`.

## Reproduce

```bash
git clone https://github.com/jonasneves/aipi540-tabletop-perception
cd aipi540-tabletop-perception
pip install -r requirements.txt
make dataset   # downloads + stages Caltech-101, ~2 min
make eval      # runs all three models + exports ONNX
make serve     # local demo on :8088
```

## Structure

```
.
├── README.md
├── SCOPE.md
├── requirements.txt
├── Makefile               # dataset | eval | sync | serve | deploy
├── scripts/
│   ├── make_dataset.py    # Caltech-101 download + 6-class filter
│   ├── naive.py           # HSV dominant-hue baseline
│   ├── classical.py       # color-hist + HOG + GBM
│   └── train_dl.py        # MobileNetV3-small fine-tune + ONNX export
├── notebooks/             # exploration only; not graded
├── models/                # ONNX + pickle artifacts
├── data/
│   ├── raw/
│   └── processed/
├── results/               # scores.json, plots
├── report/                # written report + figures
├── public/                # static site: ONNX Runtime Web + WebGPU
└── docs -> public         # GH Pages serves main/docs → public
```

## Team

Jonas Neves · Duke University · AIPI 540 · Spring 2026
