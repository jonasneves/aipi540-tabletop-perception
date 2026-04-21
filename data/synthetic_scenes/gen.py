"""Generate 10 synthetic top-down tabletop scenes for a beginner robot kit.

Each scene is 512x512 PNG. Objects: line, obstacles (cubes/cylinders), a goal
marker, and distractor clutter. Saves ground-truth JSON per-scene.
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).parent
SIZE = 512

TABLE_COLORS = [(220, 210, 195), (200, 190, 180), (210, 195, 175), (225, 220, 210)]
LINE_COLORS = [(250, 230, 60), (245, 245, 245), (40, 40, 40)]
OBSTACLE_COLORS = [(40, 40, 180), (160, 40, 40), (40, 120, 40), (120, 80, 40)]
GOAL_COLORS = [(220, 40, 120), (255, 80, 40), (80, 200, 40)]
CLUTTER_COLORS = [(150, 150, 150), (180, 170, 160), (120, 130, 140)]


def boxes_overlap(a, b, pad=10):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def propose_rect(rng, min_s=40, max_s=75, margin=20):
    s = rng.randint(min_s, max_s)
    x = rng.randint(margin, SIZE - s - margin)
    y = rng.randint(margin, SIZE - s - margin)
    return (x, y, x + s, y + s)


def place(rng, existing, min_s=40, max_s=75, margin=20, tries=40):
    for _ in range(tries):
        box = propose_rect(rng, min_s, max_s, margin)
        if not any(boxes_overlap(box, b) for _, b in existing):
            return box
    return None


def propose_line(rng):
    n = 12
    xs = [int(i * SIZE / (n - 1)) for i in range(n)]
    ys = [rng.randint(140, 380) for _ in range(n)]
    pts = list(zip(xs, ys))
    x0 = min(p[0] for p in pts) - 10
    y0 = min(p[1] for p in pts) - 10
    x1 = max(p[0] for p in pts) + 10
    y1 = max(p[1] for p in pts) + 10
    return pts, (x0, y0, x1, y1)


def render_scene(seed):
    rng = random.Random(seed)
    table = rng.choice(TABLE_COLORS)
    img = Image.new('RGB', (SIZE, SIZE), table)
    draw = ImageDraw.Draw(img)

    for _ in range(40):
        y = rng.randint(0, SIZE)
        shade = rng.randint(-10, 10)
        c = tuple(max(0, min(255, v + shade)) for v in table)
        draw.line([(0, y), (SIZE, y)], fill=c, width=1)

    items = []

    line_pts, line_box = propose_line(rng)
    line_col = rng.choice(LINE_COLORS)
    line_width = rng.randint(8, 14)
    for i in range(len(line_pts) - 1):
        draw.line([line_pts[i], line_pts[i + 1]], fill=line_col, width=line_width)
    items.append(("line", line_box))

    for _ in range(rng.randint(2, 4)):
        box = place(rng, items)
        if not box:
            continue
        col = rng.choice(OBSTACLE_COLORS)
        if rng.random() < 0.5:
            draw.rectangle(box, fill=col, outline=(20, 20, 20), width=2)
        else:
            draw.ellipse(box, fill=col, outline=(20, 20, 20), width=2)
        items.append(("obstacle", box))

    goal_box = place(rng, items, min_s=45, max_s=70)
    if goal_box:
        gcol = rng.choice(GOAL_COLORS)
        draw.ellipse(goal_box, fill=gcol, outline=(255, 255, 255), width=3)
        x, y, x2, y2 = goal_box
        cx, cy = (x + x2) // 2, (y + y2) // 2
        arm = (x2 - x) // 3
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 255, 255), width=3)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 255, 255), width=3)
        items.append(("goal", goal_box))

    for _ in range(rng.randint(3, 6)):
        box = place(rng, items, min_s=18, max_s=35, margin=5)
        if not box:
            continue
        col = rng.choice(CLUTTER_COLORS)
        draw.ellipse(box, fill=col)
        items.append(("clutter", box))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img, items


def main():
    gt = []
    for i in range(10):
        img, items = render_scene(seed=100 + i)
        img.save(OUT / f"scene_{i:02d}.png")
        gt.append({
            "scene": f"scene_{i:02d}.png",
            "objects": [{"class": c, "bbox": list(b)} for c, b in items],
        })
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print(f"Rendered 10 scenes to {OUT}")


if __name__ == '__main__':
    main()
