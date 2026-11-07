import random
import shutil
import os

import cv2
from ultralytics import YOLO

BASE = r'C:\Users\Muhammad Zikry\Downloads\archive'
POS_DIR = os.path.join(BASE, 'Positive')
NEG_DIR = os.path.join(BASE, 'Negative')

IMG_DIR = 'crack_dataset/train/images'
LBL_DIR = 'crack_dataset/train/labels'

POS_SAMPLE = 7500
NEG_SAMPLE = 7500
CONF = 0.5

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

random.seed(42)
pos_files = random.sample([f for f in os.listdir(POS_DIR) if f.lower().endswith('.jpg')], POS_SAMPLE)
neg_files = random.sample([f for f in os.listdir(NEG_DIR) if f.lower().endswith('.jpg')], NEG_SAMPLE)

model = YOLO('crack_trained.pt')

added_pos = 0
skipped_pos = 0
for name in pos_files:
    img_path = os.path.join(POS_DIR, name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    results = model.predict(img, conf=CONF, imgsz=640, verbose=False, device=0)
    result = results[0]
    polys = []
    if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            if float(box.conf[0]) < CONF:
                continue
            m = result.masks.xy[i]
            if m is None or len(m) < 3:
                continue
            poly = [f"{x / w:.6f}" if x / w >= 0 else "0.0" for x in m[:, 0]]
            py = [f"{y / h:.6f}" if y / h >= 0 else "0.0" for y in m[:, 1]]
            pts = []
            for x, y in zip(poly, py):
                pts.append(x)
                pts.append(y)
            polys.append(pts)
    if not polys:
        skipped_pos += 1
        continue
    base = 'archive_pos_' + name[:-4]
    shutil.copy2(img_path, os.path.join(IMG_DIR, base + '.jpg'))
    with open(os.path.join(LBL_DIR, base + '.txt'), 'w') as f:
        for p in polys:
            f.write('0 ' + ' '.join(p) + '\n')
    added_pos += 1

added_neg = 0
for name in neg_files:
    img_path = os.path.join(NEG_DIR, name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    base = 'archive_neg_' + name[:-4]
    shutil.copy2(img_path, os.path.join(IMG_DIR, base + '.jpg'))
    open(os.path.join(LBL_DIR, base + '.txt'), 'w').close()
    added_neg += 1

for cache in ['crack_dataset/train/labels.cache', 'crack_dataset/test/labels.cache']:
    if os.path.exists(cache):
        os.remove(cache)

print(f"Positives added: {added_pos}, skipped (no detection): {skipped_pos}")
print(f"Negatives added: {added_neg}")
