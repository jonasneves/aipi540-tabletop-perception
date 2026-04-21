# AIPI 540 Rubric Checklist

> **Audience:** AIPI 540 grading staff (Duke, Spring 2026). This file maps each rubric element to the specific artifact in this repository. General readers should start with [`README.md`](README.md).

## Three required models

| requirement | satisfied by | evidence |
|---|---|---|
| Naive baseline | HSV dominant-hue classifier | [`scripts/naive.py`](scripts/naive.py), [`results/naive.json`](results/naive.json) |
| Classical ML | Color-hist + HOG + XGBoost | [`scripts/classical.py`](scripts/classical.py), [`results/classical.json`](results/classical.json) |
| Deep learning | MobileNetV3-small fine-tune (head retrained on frozen backbone), exported to ONNX | [`scripts/train_dl.py`](scripts/train_dl.py), [`results/dl.json`](results/dl.json), [`models/classifier.onnx`](models/classifier.onnx) |

## Required focused experiment

**Scale-vs-task-fit.** LFM2.5-VL-450M (450M params, open-vocab, zero-shot) compared against the 3M-parameter MobileNetV3-small on the same closed-set task plus on out-of-distribution synthetic scenes.

- Evidence: report [§11 Experiment Write-Up](report/report.md#11-experiment-write-up)
- Data: 10 synthetic top-down tabletop scenes + 3 real-photo controls, graded by max-IoU per ground-truth object
- Headline finding: VLM 0% recall@IoU≥0.3 on synthetic scenes vs tight boxes on real photos. OOD failure, not a configuration issue (verified with both Q4 and FP16 decoder).

## Interactive application

| requirement | satisfied by |
|---|---|
| Publicly accessible | [neevs.io/aipi540-tabletop-perception](https://neevs.io/aipi540-tabletop-perception/) (GitHub Pages) |
| Live at submission + 1 week | GH Pages deploy on `main/docs → public/` symlink |
| Inference only, no training in app | Browser-side ONNX Runtime Web + Transformers.js. No training code ships with the app. |
| UX | Webcam input, live classifier top-3 labels, VLM open-vocab text query with bounding boxes, VLM narration side panel |

## Written report

15 required sections, all present in [`report/report.md`](report/report.md):

- §1 Problem Statement
- §2 Data Sources
- §3 Related Work
- §4 Evaluation Strategy & Metrics
- §5 Modeling Approach
- §6 Data Processing Pipeline
- §7 Hyperparameter Tuning Strategy
- §8 Models Evaluated
- §9 Results
- §10 Error Analysis (5 specific mispredictions with root causes + mitigations)
- §11 Experiment Write-Up
- §12 Conclusions
- §13 Future Work
- §14 Commercial Viability Statement
- §15 Ethics Statement

## Demo Day pitch

5-minute pitch deck at [agora/slides/aipi540-tabletop-perception](https://neevs.io/agora/slides/aipi540-tabletop-perception/).

Structure:
- Cover
- The question (specialist vs generalist)
- Approach (three closed-set models + VLM reference)
- Live demo
- Results (stats strip + interpretation)
- Questions + QR to live demo

## Repo structure (rubric-specified)

| expected | present |
|---|---|
| `README.md` | ✓ |
| `requirements.txt` | ✓ |
| `Makefile` (optional) | ✓ |
| `setup.py` | ✓ — fetch + train + export entrypoint |
| `app.py` | ✓ — local dev server for the static app in `public/` |
| `scripts/` | ✓ — `naive.py`, `classical.py`, `train_dl.py`, `make_dataset.py` |
| `models/` | ✓ — `classifier.onnx`, `classical.pkl` |
| `data/raw` + `data/processed` | ✓ |
| `data/outputs` | ✓ (via `results/`) |
| `notebooks/` | ✓ — exploration only; not graded |
| `.gitignore` | ✓ |

## Code quality

- All code modularized into functions, no loose executables at module top level
- Descriptive variable names + docstrings on every script
- External code attributed where used (pretrained MobileNetV3 weights via torchvision; LFM2.5-VL-450M model card linked in report)
- No Jupyter notebooks outside `notebooks/`

## AI policy

AI tooling (Claude Code) was used throughout development. Explicit acknowledgement in [report Appendix B](report/report.md#appendix-b-ai-tooling).
