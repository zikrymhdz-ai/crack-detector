"""Score the detection pipeline against eval_set.json.

Detections are cached, so re-scoring after a change to the measurement gates is
fast. Use --refresh to re-run the segmenter.

    python scripts/score.py                 # score with current settings
    python scripts/score.py --refresh       # ignore cached model output
    python scripts/score.py --verbose
"""
import os
import sys
import json
import glob
import time
import hashlib

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import app as A  # noqa: E402

CACHE = os.path.join(ROOT, '.eval_cache')


def load_set():
    with open(os.path.join(ROOT, 'eval_set.json'), encoding='utf-8') as fh:
        return json.load(fh)


def cached_detect(path, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5((path + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    npz = os.path.join(CACHE, key + '.npz')
    if not refresh and os.path.exists(npz):
        d = np.load(npz)
        return d['boxes'], d['confs'], d['mask']
    img = load_rgb(path)
    if img is None:
        return np.zeros((0, 4), int), np.zeros(0), None
    boxes, confs, mask = A._detect_static(img, tiled=True)
    np.savez_compressed(npz, boxes=np.asarray(boxes, np.int32).reshape(-1, 4),
                        confs=np.asarray(confs, np.float32), mask=mask)
    return np.asarray(boxes), np.asarray(confs), mask


def load_rgb(path):
    """Load like analyze_image(): keep alpha and composite transparent pixels onto
    a mid-grey wall so a crack sticker is still detectable."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        img = raw[:, :, :3]
        if alpha.min() < 250:
            a = (alpha.astype(np.float32) / 255.0)[:, :, None]
            img = (img.astype(np.float32) * a + 128.0 * (1.0 - a)).astype(np.uint8)
        return img
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return raw


def score(refresh=False, verbose=False):
    cfg = load_set()
    tp = fp = fn = tn = 0
    rows = []
    for item in cfg['items']:
        path = os.path.join(cfg['source'], item['file'])
        img = load_rgb(path)
        if img is None:
            print('MISSING', path)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, _, mask = cached_detect(path, refresh)
        if mask is None or mask.shape[:2] != gray.shape[:2]:
            _, _, mask = A._detect_static(img, tiled=True)
            mask = np.zeros_like(gray)
        t0 = time.time()
        res = A.measure_combined_mask(mask, frame=img)
        dt = time.time() - t0
        total = sum(r[2] for r in res)
        pred = 1 if total >= 30 else 0
        label = int(item['label'])
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
        rows.append((item['file'], item['kind'], label, pred, len(res), round(total),
                     round(dt * 1000)))
        if verbose:
            flag = 'ok ' if pred == label else 'ERR'
            print(f'{flag} {item["file"][:44]:46s} {item["kind"]:20s} '
                  f'label={label} pred={pred} n={len(res):2d} len={total:6.0f} {dt*1000:5.0f}ms')

    tot = len(rows)
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f'\nimages={tot}  TP={tp} FP={fp} FN={fn} TN={tn}')
    print(f'recall={rec:.3f}  precision={prec:.3f}  accuracy={(tp+tn)/max(1,tot):.3f}')
    if not verbose:
        for r in rows:
            if r[2] != r[3]:
                print(f'  {"FALSE NEGATIVE" if r[2] else "FALSE POSITIVE"}: {r[0][:50]} '
                      f'({r[1]}) n={r[4]} len={r[5]}')
    return rec, prec


if __name__ == '__main__':
    score(refresh='--refresh' in sys.argv, verbose='--verbose' in sys.argv)
