# Scope — Tabletop Perception

> **Audience:** internal planning doc (author + collaborators). For a project overview, see [`README.md`](README.md). For the written submission, see [`report/report.md`](report/report.md). This file captures the locked scope and demo plan and may use shorthand that isn't defined elsewhere.

## Thesis

Beginner robotics kits need vision. Closed-set specialists are fast and narrow; open-vocab VLMs are flexible and slow. Chart the trade-off on a concrete tabletop task.

## Task

Given a top-down or slight-angle photo of a tabletop, classify objects from a fixed 6-class taxonomy (`cell_phone`, `cup`, `headphone`, `laptop`, `scissors`, `stapler`) *and* demonstrate open-vocab detection via VLM on the same input. Taxonomy chosen to match Caltech-101 source classes.

## Three models (540 rubric)

| Tier | Model | Features | Rationale |
|---|---|---|---|
| Naive | HSV color threshold + centroid | None | Required baseline; captures how much color alone buys |
| Classical | Color histogram + HOG + GBM | Engineered | Tests whether hand-features + a strong shallow learner closes the gap |
| DL | MobileNetV3-small, fine-tuned | Learned | Main neural-net approach; deployable in-browser via ONNX |

## Focused experiment

**Scale vs. task-fit.** LFM2.5-VL-450M as a zero-shot open-vocab baseline. Compare:

1. Closed-set accuracy (same 6 classes): MobileNetV3-small vs VLM-450M
2. Inference latency: both on MacBook browser
3. Open-vocab recall: VLM only (MobileNet can't answer this)

Expected result: MobileNet wins closed-set by large margins on latency and accuracy. VLM wins everything MobileNet can't even attempt. The trade-off is the finding.

## Data

- **Training**: YCB-Video household-object subset, or COCO-filtered subset for the 7 classes. Whichever pulls faster into Colab.
- **Evaluation**: held-out split + ~20 phone-captured photos from Jonas's actual desk (the genuine deployment condition).
- **OOD ablation**: the synthetic top-down scenes in `validation/scenes/` from the pre-build validation phase. Both MobileNet and VLM degrade on these — reported as a known generalization frontier.

## App

Single-page `public/index.html`:
- Webcam input (phone via WebRTC bridge, or laptop camera directly)
- Live MobileNet inference at ~30fps with bounding boxes
- Live VLM detection at ~0.7Hz with open-vocab text query
- Live narration side panel (VLM describe mode) every ~2s
- Fallback to static image upload if WebRTC fails

## Demo wow beat

In-person classroom demo, 5 min:
1. Phone → WebRTC → laptop. Boxes populate on screen.
2. Pick up an out-of-set object. MobileNet stops labeling. Silence.
3. Audience member calls out any object in the room. Jonas types it. VLM draws a tight box.
4. "A banana" → VLM refuses: *"There are no bananas visible."*
5. Trade-off chart. Close.

## Decisions locked

1. Fresh repo; toolmark scaffolding abandoned
2. Three models: HSV / (color-hist + HOG + GBM) / MobileNetV3-small fine-tune
3. VLM-450M is a *zero-shot baseline*, not the DL tier, to keep the DL rubric clean
4. Q4 decoder quantization (refuses absent objects; FP16 hallucinated)
5. GH Pages deployment via `docs → public` symlink
6. Training on Colab T4

## Known risks + mitigations

- **VLM confirm-bias**: open-vocab model matches the query rather than discriminates (evidence: `results/vlm_ood/grade_report.json`). Mitigation: one absent-object query in the demo script shows the refusal beat.
- **Classroom Wi-Fi**: WebRTC needs `signal.neevs.io`. If flaky, fallback to laptop webcam directly.
- **VLM color hallucination**: says "brown" for gray. Included in Limitations section.
- **No hardware robot**: paper demo only. Not required by rubric.
