"""Build fine-mask YOLO segment dataset from synthetic + public sources.

Sources:
  - synthetic  : scripts/synth_cracks.py output dir (fine_dataset/raw_synth);
                 negatives (empty masks) are kept as background-only images.
  - public     : varcoder/crack-segmentation-dataset parquet shards (image+mask bytes)

Output layout:
  fine_dataset/
    images/{train,val,test}/<stem>.jpg
    labels/{train,val,test}/<stem>.txt     (YOLO seg polys, normalized 0..1)
    data.yaml
    build_report.txt
"""
import argparse
import os
import random

import numpy as np
import cv2
import pyarrow.parquet as pq

MAX_COVERAGE = 0.25        # blob filters
MIN_AREA = 25
ELONG_AREABBOX_RATIO = 0.5
MAX_SEG_POINTS = 160
SPLIT = (0.8, 0.1, 0.1)


def read_public(paths):
    rows = []
    for p in paths:
        t = pq.read_table(p, columns=["pixel_values", "label"])
        rows.extend(t.to_pylist())
    return rows


def raster_to_polygons(mask, eps):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = m.shape[:2]
    polys = []
    for c in cnts:
        c = c.astype(np.float32)
        if len(c) >= MAX_SEG_POINTS:
            c = cv2.approxPolyDP(c, eps, True)
        c = c.reshape(-1, 2)
        if len(c) < 3:
            continue
        ring = np.vstack([c, c[0]])
        area = 0.5 * np.abs(np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]))
        if area < 1.0:
            continue
        x = np.clip(c[:, 0] / w, 0.0, 1.0)
        y = np.clip(c[:, 1] / h, 0.0, 1.0)
        polys.append(np.stack([x, y], axis=1))
    return polys


def polys_to_text(polys):
    lines = []
    for p in polys:
        lines.append("0 " + " ".join(f"{v:.6f}" for v in p.reshape(-1).tolist()))
    return "\n".join(lines)


def valid_mask(mask):
    b = mask > 0
    if not b.any():
        return None, "empty"
    cov = b.mean()
    if cov > MAX_COVERAGE:
        return None, "blob>25%"
    if b.sum() < MIN_AREA:
        return None, "speckle<25px"
    ys, xs = np.nonzero(b)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    if cov > 0.15 and b.sum() / bbox_area >= ELONG_AREABBOX_RATIO:
        return None, "non-elongated"
    return b, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="fine_dataset/raw_synth")
    ap.add_argument("--out", default="fine_dataset")
    ap.add_argument("--shards", nargs="+",
                    default=[r"C:\Users\Muhammad Zikry\AppData\Local\Temp\opencode\crackdata\train-0.parquet",
                             r"C:\Users\Muhammad Zikry\AppData\Local\Temp\opencode\crackdata\train-1.parquet"])
    ap.add_argument("--max-public", type=int, default=None)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = os.path.abspath(args.out)
    dropped = {}
    records = []

    def make_dirs():
        for s in SPLIT:
            pass
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(out, "images", s), exist_ok=True)
            os.makedirs(os.path.join(out, "labels", s), exist_ok=True)

    def emit(stem, img, mask, src):
        b, reason = valid_mask(mask)
        if b is None:
            if src == "synth" and reason == "empty":
                jpg_path = os.path.join(out, "images", pick_split(), f"{stem}.jpg")
                cv2.imwrite(jpg_path, img)
                records.append((stem, os.path.basename(os.path.dirname(jpg_path)), src, 0.0))
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
            return
        eps = max(1.0, 0.8 * (mask.shape[0] / 448.0))
        polys = raster_to_polygons(b, eps)
        if not polys:
            dropped["nopoly"] = dropped.get("nopoly", 0) + 1
            return
        split = pick_split()
        jpg_path = os.path.join(out, "images", split, f"{stem}.jpg")
        txt_path = os.path.join(out, "labels", split, f"{stem}.txt")
        cv2.imwrite(jpg_path, img)
        with open(txt_path, "w") as fh:
            fh.write(polys_to_text(polys))
        records.append((stem, split, src, b.mean()))

    def pick_split():
        r = rng.random()
        if r < SPLIT[0]:
            return "train"
        if r < SPLIT[0] + SPLIT[1]:
            return "val"
        return "test"

    make_dirs()

    # ---- synthetic ----
    synth_dir = os.path.abspath(args.synth)
    syn_names = sorted(n for n in os.listdir(synth_dir)
                       if n.lower().endswith(".jpg")
                       and os.path.exists(os.path.join(synth_dir, n[:-4] + "_mask.png")))
    for i, n in enumerate(syn_names):
        img = cv2.imread(os.path.join(synth_dir, n))
        msk = cv2.imread(os.path.join(synth_dir, n[:-4] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        emit(f"sy_{i:06d}", img, msk, "synth")

    # ---- public ----
    rows = read_public(args.shards)
    if args.max_public:
        rows = rng.sample(rows, args.max_public)
    else:
        rng.shuffle(rows)
    for i, r in enumerate(rows):
        img_b = r["pixel_values"]["bytes"]
        msk_b = r["label"]["bytes"]
        if img_b is None or msk_b is None:
            continue
        img = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
        msk = cv2.imdecode(np.frombuffer(msk_b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        if msk.shape != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]))
        emit(f"pb_{i:06d}", img, msk, "public")

    # ---- report + data.yaml ----
    cov = np.array([r[3] for r in records if r[3] > 0])
    srcs = {}
    splits = {}
    neg = 0
    for _, s, src, c in records:
        srcs[src] = srcs.get(src, 0) + 1
        splits[s] = splits.get(s, 0) + 1
        if c == 0.0:
            neg += 1
    n_pos = len(records) - neg

    data_yaml = (
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "task: segment\n"
        "names:\n  0: crack\n"
    )
    dyp = os.path.join(out, "data.yaml")
    if os.path.exists(dyp):
        with open(dyp) as fh:
            existing = fh.read()
        if "task: segment" in existing and "0: crack" in existing:
            data_yaml_text = existing
        else:
            data_yaml_text = data_yaml
    else:
        data_yaml_text = data_yaml
    with open(dyp, "w") as fh:
        fh.write(data_yaml_text)

    rows_counts = f"synth candidates: {len(syn_names)}\npublic candidates: {len(rows)}\n"
    report = rows_counts + (
        f"kept total: {len(records)}  (pos {n_pos}, neg {neg})\n"
        f"by source: {srcs}\nby split: {splits}\n"
        f"pos coverage med/P90/max: {np.median(cov)*100:.1f}% / {np.percentile(cov,90)*100:.1f}% / {cov.max()*100:.1f}%\n"
        f"dropped: {dropped}\n"
    )
    with open(os.path.join(out, "build_report.txt"), "w") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()