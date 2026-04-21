"""Grade VLM probe output against scene ground truth.

For each (scene, query) pair:
- Convert VLM normalized boxes to pixel coords
- Compute max-IoU against any GT object whose class matches the query intent
- Report per-query mean IoU, recall@0.3, box count stats
- Also render overlays for human eyeballing
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

REPO         = Path(__file__).resolve().parent.parent
SCENES_DIR   = REPO / "data" / "synthetic_scenes"
GT_PATH      = SCENES_DIR / "ground_truth.json"
VLM_PROBE    = REPO / "results" / "vlm_ood" / "vlm_probe_q4.json"
OVERLAY_DIR  = REPO / "report" / "figures" / "vlm_overlays"
REPORT_PATH  = REPO / "results" / "vlm_ood" / "grade_report.json"
SIZE = 512

QUERY_TO_CLASS = {
    "line": {"line"},
    "obstacle": {"obstacle"},
    "goal marker": {"goal"},
    "target": {"goal"},
    "a dark line on the ground": {"line"},
    "a colored obstacle": {"obstacle"},
    "the goal marker": {"goal"},
    "something the robot should avoid": {"obstacle"},
}


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    ub = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def main():
    gt_all  = json.loads(GT_PATH.read_text())
    vlm_all = json.loads(VLM_PROBE.read_text())
    gt_by_scene = {g["scene"]: g["objects"] for g in gt_all}

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    per_query = {}  # query -> list of {max_iou, gt_count, pred_count}

    for entry in vlm_all:
        scene = entry["scene"]
        gt_objs = gt_by_scene[scene]
        img = Image.open(SCENES_DIR / scene).convert("RGB")
        draw = ImageDraw.Draw(img)

        for r in entry["results"]:
            query = r["query"]
            target_classes = QUERY_TO_CLASS.get(query, set())
            gt_targets = [o for o in gt_objs if o["class"] in target_classes]

            pred_boxes_px = []
            for pb in r["boxes"]:
                x1, y1, x2, y2 = pb["bbox"]
                pred_boxes_px.append((x1 * SIZE, y1 * SIZE, x2 * SIZE, y2 * SIZE))

            max_ious = []
            for gt in gt_targets:
                gx = tuple(gt["bbox"])
                ious = [iou(pb, gx) for pb in pred_boxes_px] or [0.0]
                max_ious.append(max(ious))

            per_query.setdefault(query, []).append({
                "scene": scene,
                "gt_count": len(gt_targets),
                "pred_count": len(pred_boxes_px),
                "per_gt_max_iou": max_ious,
                "dt_ms": r["dt"],
            })

            # Draw overlays (one image per scene, all queries combined)
            colors = {"line": "#012169", "obstacle": "#C84E00", "goal marker": "#A1B70D", "target": "#993399"}
            color = colors.get(query, "#666")
            for pb in pred_boxes_px:
                draw.rectangle(pb, outline=color, width=3)

        img.save(OVERLAY_DIR / scene)

    # Also render GT for reference
    for scene, gt_objs in gt_by_scene.items():
        img = Image.open(SCENES_DIR / scene).convert("RGB")
        draw = ImageDraw.Draw(img)
        gtcolors = {"line": "#012169", "obstacle": "#C84E00", "goal": "#A1B70D", "clutter": "#888888"}
        for o in gt_objs:
            draw.rectangle(tuple(o["bbox"]), outline=gtcolors.get(o["class"], "#666"), width=2)
        img.save(OVERLAY_DIR / f"gt_{scene}")

    # Report
    print(f"{'Query':<40} {'N':>3}  {'mean maxIoU':>12}  {'recall@0.3':>11}  {'mean preds':>11}  {'mean dt':>10}")
    print("-" * 95)
    summary = {}
    for q, entries in per_query.items():
        flat_ious = [v for e in entries for v in e["per_gt_max_iou"]]
        mean_iou = sum(flat_ious) / len(flat_ious) if flat_ious else 0.0
        recall = (sum(1 for v in flat_ious if v >= 0.3) / len(flat_ious)) if flat_ious else 0.0
        mean_preds = sum(e["pred_count"] for e in entries) / len(entries)
        mean_dt = sum(e["dt_ms"] for e in entries) / len(entries)
        print(f"{q:<40} {len(entries):>3}  {mean_iou:>12.3f}  {recall:>11.1%}  {mean_preds:>11.1f}  {mean_dt:>8.0f}ms")
        summary[q] = {"mean_iou": mean_iou, "recall_at_0.3": recall, "mean_preds": mean_preds, "mean_dt_ms": mean_dt}

    # Latency stats overall
    all_dt = [r["dt"] for e in vlm_all for r in e["results"]]
    all_dt.sort()
    p50 = all_dt[len(all_dt) // 2]
    p90 = all_dt[int(len(all_dt) * 0.9)]
    print(f"\nLatency: N={len(all_dt)}, p50={p50:.0f}ms, p90={p90:.0f}ms, mean={sum(all_dt)/len(all_dt):.0f}ms")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "per_query": summary,
        "latency_ms": {"p50": p50, "p90": p90, "mean": sum(all_dt) / len(all_dt), "n": len(all_dt)},
    }, indent=2))
    print(f"\nOverlays: {OVERLAY_DIR}/")
    print(f"Report:   {REPORT_PATH}")


if __name__ == "__main__":
    main()
