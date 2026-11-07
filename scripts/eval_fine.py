"""Evaluate fine-mask crack model: mAP on val + thin-mask coverage gate.

Usage:
    python scripts/eval_fine.py [--weights runs/fine_seg/segs_fine_s/weights/best.pt]

Success gate (user-defined):
  - mAP50(M) reasonably high AND
  - predicted mask pixel-coverage on real photos < 15% frame (i.e. thin, not blobs).
"""
import argparse
import os
import sys

import numpy as np
import cv2
from ultralytics import YOLO

COVERAGE_GATE = 0.15


def coverage_stats(model, jpgs, imgsz=640, conf=0.35):
    covs = []
    for p in jpgs:
        img = cv2.imread(p)
        if img is None:
            continue
        res = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)
        if len(res) == 0 or res[0].masks is None:
            continue
        data = res[0].masks.data
        if hasattr(data, "detach"):
            data = data.detach().cpu()
        masks = np.asarray(data)
        if len(masks) == 0:
            continue
        h, w = img.shape[:2]
        for m in masks:
            bin_ = cv2.resize((m > 0.5).astype(np.uint8), (w, h))
            covs.append(bin_.mean())
    return np.array(covs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"runs/fine_seg/segs_fine_s/weights/best.pt")
    ap.add_argument("--val", default="fine_dataset/data.yaml")
    ap.add_argument("--photos", nargs="+", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    epoch = os.path.basename(os.path.dirname(args.weights.split(os.sep))[-1] if False else os.path.dirname(args.weights))
    print(f"[eval] weights={args.weights}")

    val = model.val(data=args.val, imgsz=640, split="val", verbose=False)
    ms = val.results_dict
    m_box50m = ms.get("metrics/mAP50(B)", float("nan"))
    m_mask50m = ms.get("metrics/mAP50(M)", float("nan"))
    m_mask50_95 = ms.get("metrics/mAP50-95(M)", float("nan"))
    print(f"[eval] mAP50(B)={m_box50m:.3f}  mAP50(M)={m_mask50m:.3f}  mAP50-95(M)={m_mask50_95:.3f}")

    if args.photos is None:
        test_dir = "fine_dataset/images/test"
        args.photos = [os.path.join(test_dir, n) for n in sorted(os.listdir(test_dir))
                       if n.lower().endswith(".jpg")][:40]

    covs = coverage_stats(model, args.photos)
    print(f"[eval] photos={len(args.photos)}  predicted-mask counts={len(covs)}")
    if len(covs):
        print(f"[eval] mask coverage pct med/P90/max = {np.median(covs)*100:.1f}% / {np.percentile(covs,90)*100:.1f}% / {covs.max()*100:.1f}%")
        frac_thin = (covs < COVERAGE_GATE).mean()
        print(f"[eval] frac masks < {COVERAGE_GATE*100:.0f}% frame = {frac_thin:.2%}  (gate ok if >0.8)")
    else:
        print("[eval] no masks predicted on photos")


if __name__ == "__main__":
    main()