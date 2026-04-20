# Tabletop Perception for Beginner Robot Kits

Vision perception layer for beginner robotics kits. Compares three models — HSV threshold (naive), color-histogram + gradient boosting (classical), and fine-tuned MobileNetV3-small (DL) — against a zero-shot LFM2.5-VL-450M baseline on common tabletop objects. Deployed as an in-browser live demo: point a webcam at a desk, see all three models reason in real time.

Final project for Duke **AIPI 540: Deep Learning Applications** (Spring 2026). Due 2026-04-21.

## Thesis

A beginner robot kit ships with a closed set of supported objects and a narrow compute budget. A 3M-parameter specialist fine-tuned on that set wins on latency and precision. A 450M VLM wins on flexibility to unseen objects, at the cost of 50× slower inference and a documented confirm-bias failure mode. Choosing between them is a deployment decision, not a benchmark one.

## What you get

- **Live web app**: webcam in, three models side-by-side, open-vocab text query, honest refusals. Deployed via GitHub Pages.
- **Report**: problem statement, data, evaluation strategy, three-model results, focused experiment on scale-vs-task-fit, error analysis, commercial viability, ethics.
- **Reproducible training**: Colab notebook fine-tunes MobileNetV3-small on a public tabletop dataset and exports ONNX.

## Structure

```
.
├── README.md
├── SCOPE.md
├── requirements.txt
├── scripts/
│   ├── naive.py           # HSV + color heuristic baseline
│   ├── classical.py       # color-hist + GBM
│   └── eval.py            # unified evaluation harness
├── notebooks/             # exploration only; not graded
├── colab/
│   └── train_mobilenet.ipynb   # runs on T4, exports ONNX
├── models/                # ONNX artifacts
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
