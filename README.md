# Tabletop Perception for Beginner Robot Kits

[![Live demo](https://img.shields.io/badge/demo-live-00539B)](https://neevs.io/aipi540-tabletop-perception/)
[![Duke AIPI 540](https://img.shields.io/badge/Duke-AIPI%20540-012169)](https://masters.pratt.duke.edu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime%20Web-WebGPU-00539B)](https://onnxruntime.ai/docs/tutorials/web/)

**[Live demo](https://neevs.io/aipi540-tabletop-perception/)** — point a webcam at a desk, watch a specialist and a generalist reason side-by-side. In-browser, no server, no API keys.

A fine-tuned MobileNetV3-small (2.5M frozen + 6K head) hits **97% top-1 at ~15 ms/frame** on a 6-class tabletop task (`cell_phone`, `cup`, `headphone`, `laptop`, `scissors`, `stapler`). A 450M-parameter open-vocab VLM (LFM2.5-VL-450M) on the same input runs **~85× slower** at detect (1.3 s/query vs 15 ms/frame) and collapses to 0% recall@IoU≥0.3 on stylized out-of-distribution synthetic scenes — though on natural photos it correctly *refuses* absent-object queries, a property the specialist cannot offer. Production shape: both running at once — specialist on the webcam stream, generalist on typed queries — with the open-vocab tier extending coverage where the closed-set head cannot reach.

Choosing between them is a deployment decision, not a benchmark one.

## Results

| Tier | Model | Params | Latency | Top-1 |
|---|---|---|---|---|
| Naive | HSV threshold | 0 | ~1 ms | 24% |
| Classical | Color-hist + HOG + GBM | ~2.4K trees | ~40 ms | 76% |
| DL | MobileNetV3-small fine-tune | 2.5M frozen + 6K head | ~15 ms | **97%** |
| VLM | LFM2.5-VL-450M zero-shot | 450M | ~1300 ms | open-vocab |

Per-class F1, latency breakdown, and a five-case error analysis: [`report/report.md`](report/report.md).

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
