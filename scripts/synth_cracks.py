"""Generate synthetic wall cracks with PIXEL-EXACT thin masks.

Each sample: textured wall background + branches of a thin crack rendered as a
binary mask (255 = crack) and a composited JPG. Blob-free by construction: the
mask is only ever 2-10px wide along random-walk paths, so a fine-mask model can
be trained on it directly (converted to YOLO polygons by build_fine_dataset.py).

Usage:  python scripts/synth_cracks.py --count 700 --out fine_dataset/raw_synth --seed 7
"""
import argparse, math, os, random
import cv2
import numpy as np

def textured_bg(rng, h, w):
    base = np.array([rng.uniform(105, 175), rng.uniform(100, 168), rng.uniform(90, 160)])
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    # low-frequency shading
    shade = rng.uniform(4, 18) * np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal((h, w)).astype(np.float32)
    shade = cv2.GaussianBlur(shade, (0, 0), rng.uniform(8, 24))
    img += shade[..., None]
    # stucco patch noise (high freq, small contrast)
    rough = np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal((h, w)).astype(np.float32)
    rough = cv2.GaussianBlur(rough, (0, 0), rng.uniform(1.2, 3.5))
    img += rng.uniform(2, 8) * rough[..., None]
    # a few faint stains (drywall discolouration)
    for _ in range(rng.randint(2, 6)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rw, rh = rng.uniform(20, w * 0.35), rng.uniform(20, h * 0.35)
        ang = np.deg2rad(rng.uniform(0, 180))
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rw), int(rh)), float(np.rad2deg(ang)), 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(6, 18))
        stain = np.array([rng.uniform(-18, -3), rng.uniform(-14, -2), rng.uniform(-12, -1)])
        img += stain[None, None, :] * mask[..., None]
    img = np.clip(img, 0, 255)
    return img

def walk_path(rng, h, w, rng_len, step_range=(3, 7), turn=(0.25, 0.85)):
    x = rng.uniform(w * 0.12, w * 0.88)
    y = rng.uniform(h * 0.12, h * 0.88)
    ang = rng.uniform(0, 2 * math.pi)
    pts = [(x, y)]
    dist = 0.0
    target = rng.uniform(*rng_len)
    while dist < target:
        ang += rng.uniform(-turn[1], turn[1]) * rng.uniform(0.4, 1.0)
        x += math.cos(ang) * rng.uniform(*step_range)
        y += math.sin(ang) * rng.uniform(*step_range)
        if not (6 < x < w - 6 and 6 < y < h - 6):
            ang = rng.uniform(0, 2 * math.pi)
            continue
        pts.append((x, y))
        dist += step_range[1]
        if len(pts) > 900:
            break
    return pts

def branch_path(rng, h, w, start, base_ang, rng_len=(40, 140)):
    x, y = start
    ang = base_ang + rng.uniform(-0.9, 0.9)
    pts = [(x, y)]
    dist = 0.0
    target = rng.uniform(*rng_len)
    while dist < target:
        ang += rng.uniform(-0.7, 0.7)
        x += math.cos(ang) * 3.0
        y += math.sin(ang) * 3.0
        if not (6 < x < w - 6 and 6 < y < h - 6):
            break
        pts.append((x, y))
        dist += 3.0
    return pts

def render_crack(h, w, rng):
    """Returns mask (uint8 255=crack) and list of (pts, thickness) strokes."""
    mask = np.zeros((h, w), np.uint8)
    strokes = []
    main = walk_path(rng, h, w, (160, 520))
    t_main = rng.uniform(3.0, 9.0)
    strokes.append((main, t_main))
    # up to 3 branches off random main-path points
    nbr = rng.randint(1, 4)
    for _ in range(nbr):
        idx = rng.randint(1, max(1, len(main) - 1))
        start = main[idx]
        ang = math.atan2(main[idx][1] - main[idx - 1][1], main[idx][0] - main[idx - 1][0])
        b = branch_path(rng, h, w, start, ang)
        if len(b) >= 4:
            strokes.append((b, max(1.5, t_main * rng.uniform(0.25, 0.55))))
    # render strokes as continuous lines along the path (thin, curl-following)
    for pts, t in strokes:
        t = max(2.0, t)
        for i in range(len(pts) - 1):
            cv2.line(mask, (int(pts[i][0]), int(pts[i][1])),
                     (int(pts[i + 1][0]), int(pts[i + 1][1])), 255, int(round(t)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return mask, strokes

def compose(img, mask, rng):
    h, w = img.shape[:2]
    out = img.copy()
    crack_dark = rng.uniform(0.45, 0.78)
    crack = (mask > 0)
    # darken + tint
    out[crack] *= crack_dark
    tint = np.array([rng.uniform(-14, 6), rng.uniform(-16, 4), rng.uniform(-20, 2)])
    out[crack] += tint
    # per-pixel micro noise along crack
    nz = np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal((h, w, 3)).astype(np.float32) * rng.uniform(2, 7)
    nz = nz * mask[..., None].astype(np.float32) / 255.0
    out = out + nz
    # soft blend seam
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.1))
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=700)
    ap.add_argument("--out", default="fine_dataset/raw_synth")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    h = w = a.size
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.count):
        is_neg = rng.random() < 0.15
        bg = textured_bg(rng, h, w)
        if is_neg:
            mask = np.zeros((h, w), np.uint8)
            comp = bg.copy()
        else:
            mask, strokes = render_crack(h, w, rng)
            comp = compose(bg, mask, rng)
        fid = f"{i:05d}"
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, f"{fid}.jpg"), comp)
        cv2.imwrite(os.path.join(a.out, f"{fid}_mask.png"), mask)
        if i % 100 == 0:
            print(f"  {i}/{a.count}")
    print(f"done -> {a.out}")

if __name__ == "__main__":
    main()