from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import base64
import json
import socket
import shutil
import glob
import math
import yaml
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _SK_HAS = True
except Exception:
    _sk_skeletonize = None
    _SK_HAS = False
from inference_sdk import InferenceHTTPClient
try:
    from ultralytics import YOLO
    ULTRA_OK = True
except Exception as e:
    # torch/ultralytics can fail to map its CUDA DLLs (WinError 1455) when
    # another process (e.g. background training) already holds torch + the
    # paging file is low. Fall back to Edge Only mode instead of crashing.
    YOLO = None
    ULTRA_OK = False
    print(f"NOTE: ultralytics unavailable ({e}) — running in Edge Only mode")

# Force RTSP over TCP (not UDP) — kills h264 "error while decoding MB" spam
# from lost UDP packets. Must be set before any VideoCapture opens.
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')

app = Flask(__name__)

# Socket.IO push channel for the Unity VR digital twin.
# The Unity client (CrackRealtimeClient.cs) connects with a PLAIN RFC 6455
# WebSocket (flask-sock) and exchanges simple JSON envelopes, so no Socket.IO
# (Engine.IO framing) is involved on either side.
from flask_sock import Sock
sock = Sock(app)

# Live Unity clients connected on the plain /ws WebSocket route. Each entry is
# a flask_sock.WebSocket; broadcast pushes the same JSON envelope to all of them.
WS_CLIENTS = set()


@sock.route('/ws')
def crack_ws(ws):
    """Accept a Unity twin WebSocket. We only push new_crack events to the
    client, so this handler just keeps the connection alive until it drops."""
    WS_CLIENTS.add(ws)
    print(f"[WS] client connected ({len(WS_CLIENTS)} total)")
    try:
        while True:
            msg = ws.receive(timeout=None)
            if msg is None:
                break
    except Exception as e:
        print(f"[WS] client error/drop: {e}")
    finally:
        WS_CLIENTS.discard(ws)
        print(f"[WS] client disconnected ({len(WS_CLIENTS)} remaining)")


def broadcast_new_crack(payload):
    """Send {'event':'new_crack','data': payload} to every live Unity client."""
    if not WS_CLIENTS:
        return
    envelope = json.dumps({'event': 'new_crack', 'data': payload})
    dead = []
    for ws in list(WS_CLIENTS):
        try:
            ws.send(envelope)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.discard(ws)

# ─────────────────────────────────────────────
# MODEL LOADING — YOLOv8 crack segmenter
# ─────────────────────────────────────────────
MODEL_CANDIDATES = [
    'crack_fine.pt',
]

model = None
AI_READY = False
MODEL_NAME = "None"

for candidate in MODEL_CANDIDATES:
    try:
        if YOLO is None:
            print(f"ULTRA_OK False — skipping model load: {candidate}")
            break
        model = YOLO(candidate)
        AI_READY = True
        MODEL_NAME = candidate
        print(f"Model loaded: {candidate}")
        break
    except Exception as e:
        print(f"Could not load {candidate}: {e}")

# A single crack segmentation model is used for detection. MODEL_FALLBACK is
# intentionally left None (no second model / recall-rescue); all downstream
# code paths already guard on `MODEL_FALLBACK is not None` and therefore skip
# the fallback cleanly.
MODEL_FALLBACK = None
MODEL_FALLBACK_NAME = None


# ─────────────────────────────────────────────
# ROBOFLOW API — cloud fallback
# ─────────────────────────────────────────────
ROBOFLOW_API_KEY = "bHn2GXBy4Mzr96QRn18B"
ROBOFLOW_WORKSPACE = "my-crack-detection"
ROBOFLOW_WORKFLOW_ID = "wall-crack-vwall-crack-yccht-nkvrg-1-yolo11n-seg-t2-logic"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)


def roboflow_predict(frame):
    """Run inference via Roboflow workflow, return (boxes, confs, masks). Extracts polygon masks when available."""
    try:
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": frame},
            use_cache=True
        )
        boxes, confs, masks = [], [], []
        # workflow result shape varies — try predictions field
        preds = []
        if isinstance(result, list) and result and isinstance(result[0], dict):
            preds = result[0].get("predictions", []) or result[0].get("preds", [])
            # some workflows put masks under "predictions" directly
            if not preds and "predictions" in result[0]:
                preds = result[0]["predictions"]
        for pred in preds:
            x, y, w, h = pred.get("x", 0), pred.get("y", 0), pred.get("width", 0), pred.get("height", 0)
            conf = pred.get("confidence", 0)
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            boxes.append((x1, y1, x2, y2))
            confs.append(conf)
            # Try to extract polygon — Roboflow seg returns "points" [{x,y},...] or "mask" base64
            poly = None
            if "points" in pred and pred["points"]:
                try:
                    pts = pred["points"]
                    # points can be list of {x,y} or [[x,y]]
                    if isinstance(pts[0], dict):
                        poly = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
                    else:
                        poly = np.array(pts, dtype=np.float32)
                except Exception:
                    poly = None
            masks.append(poly)
        return boxes, confs, masks
    except Exception as e:
        print(f"Roboflow API error: {e}")
        return [], [], []


INFER_CONF = 0.20
INFER_IOU = 0.50
TILE_SIZE = 640   # static images are inferred as 640px tiles (20% overlap)

def predict(frame, augment=False, imgsz=640):
    """Run YOLO segmentation locally, return (boxes, confs, masks, combined_mask).

    Live/responsive path: a single pass at imgsz with a low confidence floor.
    The 3rd element is the polygon list (kept for legacy callers); the 4th is
    the union of every detected mask raster at full frame size, used by the
    measurement pipeline.
    """
    if model is not None:
        try:
            result = _run_model_once(model, frame, imgsz, augment=augment)
            boxes, confs, masks = [], [], []
            fb = None
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                    confs.append(float(box.conf[0]))
                    masks.append(result.masks.xy[i] if result.masks is not None else None)
            if not boxes and MODEL_FALLBACK is not None:
                # Primary missed everything — rescue pass with the fallback weights.
                fb = _run_model_once(MODEL_FALLBACK, frame, imgsz, augment=augment)
                if fb.boxes is not None and len(fb.boxes) > 0:
                    print("[AI] primary miss -> fallback rescued N detections")
                    for i, box in enumerate(fb.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))
                        confs.append(float(box.conf[0]))
                        masks.append(fb.masks.xy[i] if fb.masks is not None else None)
            H, W = frame.shape[:2]
            combined = _combined_mask_from_result(result, W, H)
            if fb is not None and len(fb.boxes or []) > 0:
                combined = cv2.bitwise_or(combined, _combined_mask_from_result(fb, W, H))
            return boxes, confs, masks, combined
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return [], [], [], np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    boxes, confs, masks = roboflow_predict(frame)
    if frame is None or frame.size == 0:
        return boxes, confs, masks, np.zeros((0, 0), np.uint8)
    return boxes, confs, masks, np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

# Test-time augmentation (multi-scale + flips) is not supported by this
# segmentation checkpoint: Ultralytics logs "Model does not support
# 'augment=True', reverting to single-scale prediction." on EVERY tile, which
# floods the console during a live scan while changing nothing. Keep the switch
# here so it can be re-enabled for a model that does support it.
USE_TTA = False

def _run_model_once(net, frame, imgsz, augment=False):
    return net(frame, conf=INFER_CONF, iou=INFER_IOU, imgsz=imgsz,
               augment=bool(augment and USE_TTA), verbose=False, device=0,
               retina_masks=True)[0]

def _infer_imgsz_for_image(H, W):
    """Largest tile-friendly imgsz for a full-frame pass (max 1024)."""
    longest = int(max(H, W))
    if longest <= 640:
        return 640
    return min(1024, ((longest + 31) // 32) * 32)

def _resize_mask_to_original(mask_raster, W, H):
    """YOLO raster mask (letterboxed canvas) -> binary mask at original (W,H)."""
    m = np.asarray(mask_raster)
    if m.ndim == 3:
        m = m[0]
    if m.ndim != 2:
        return np.zeros((H, W), np.uint8)
    h, w = m.shape[:2]
    if h == H and w == W:
        return ((m > 0.5).astype(np.uint8) * 255)
    scale = min(w / W, h / H)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    dw = max(0, (w - new_w) // 2)
    dh = max(0, (h - new_h) // 2)
    crop = m[dh:dh + new_h, dw:dw + new_w]
    if crop.size == 0:
        return np.zeros((H, W), np.uint8)
    res = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((res > 0.5).astype(np.uint8) * 255)

def _combined_mask_from_result(result, W, H):
    """Union EVERY predicted segment mask into ONE binary mask at (W,H).

    Cracks split across multiple boxes become a single connected region, so
    downstream measurement treats the whole crack (incl. branches) as one.
    """
    combined = np.zeros((H, W), np.uint8)
    if result is None or result.masks is None or len(result.masks) == 0:
        return combined
    try:
        data = result.masks.data                 # (N, h, w) soft mask (may be CUDA)
        if hasattr(data, 'detach'):              # torch tensor -> host numpy
            data = data.detach().cpu()
        data = np.asarray(data)
        for i in range(data.shape[0]):
            combined = cv2.bitwise_or(combined, _resize_mask_to_original(data[i], W, H))
    except Exception:
        try:
            for poly in result.masks.xy:       # polygon fallback (already frame coords)
                if poly is not None and len(poly) >= 3:
                    cnt = poly.astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(combined, [cnt], 255)
        except Exception:
            pass
    return combined

def _skeleton_length(skel):
    """Weighted skeleton length in pixels.

    Adds 1.0 per straight (4-neighbour) step and sqrt(2) per diagonal
    (8-neighbour) step. Each undirected adjacency is counted exactly once, so a
    branched crack reports its FULL path length including every branch. The
    caller converts this to real units with CM_PER_PX.
    """
    if skel is None or not np.any(skel > 0):
        return 0.0
    s = (skel > 0).astype(np.uint8)
    straight = 0.0
    diagonal = 0.0
    straight += float(np.sum(s[:, :-1] & s[:, 1:]))            # straight (right)
    straight += float(np.sum(s[:-1, :] & s[1:, :]))            # straight (down)
    diagonal += float(np.sum(s[:-1, :-1] & s[1:, 1:]))         # diagon. down-right
    diagonal += float(np.sum(s[1:, :-1] & s[:-1, 1:]))         # diagon. down-left
    return diagonal * np.sqrt(2.0) + straight

def _remove_straight_lines(mask, gray=None):
    """Erase long straight THIN runs at ANY angle (mortar joints, tile edges).

    Thick parts of the mask are protected, so wide cracks are never cut.

    When `gray` is supplied, a straight run that is a STRONG dark valley (a real
    straight crack, e.g. a vertical settlement crack in a brick wall) is kept:
    mortar joints and tile edges are not nearly as much darker than the wall as a
    crack is, so they are still erased while a straight crack survives.
    """
    H, W = mask.shape[:2]
    # Axis-aligned (fast path for the common case)
    Lh = max(60, int(0.15 * W))
    Lv = max(90, int(0.30 * H))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (Lh, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
    lines = cv2.bitwise_or(horiz, vert)

    # Rotated lines: skeletonize once, then Hough to find long straight strokes
    skel = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) \
           if hasattr(cv2, 'ximgproc') else _skeletonize(mask)
    if cv2.countNonZero(skel) > 0:
        diag = float(np.hypot(H, W))
        min_len = int(0.25 * diag)  # only erase lines spanning >25% of frame
        segs = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=max(30, min_len // 4),
                               minLineLength=min_len, maxLineGap=3)
        if segs is not None:
            eraser = np.zeros_like(mask)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            # OpenCV >= 5 returns (N, 4); older releases return (N, 1, 4).
            for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
                # stroke thickness at midpoint
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                y0, y1w = max(0, my - 2), min(H, my + 3)
                x0, x1w = max(0, mx - 2), min(W, mx + 3)
                thick = float(dist[y0:y1w, x0:x1w].max()) if dist[y0:y1w, x0:x1w].size else 0.0
                if thick > 6.0:
                    continue
                cv2.line(eraser, (x1, y1), (x2, y2), 255, int(2 * thick + 5))
            eraser = cv2.dilate(eraser, np.ones((3, 3), np.uint8), 1)
            lines = cv2.bitwise_or(lines, eraser)

    # Protect anything thicker than a mortar line
    t = max(5, int(0.008 * max(H, W)))
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((t, t), np.uint8))
    thick = cv2.dilate(thick, np.ones((t + 4, t + 4), np.uint8), iterations=1)
    lines = cv2.bitwise_and(lines, cv2.bitwise_not(thick))

    # Protect a genuinely dark straight crack (see docstring).
    if gray is not None and gray.shape[:2] == mask.shape[:2]:
        bg = _local_background(gray)
        dark = (bg - gray.astype(np.float32)) >= np.maximum(22.0, 0.28 * bg)
        dark = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1)
        lines = cv2.bitwise_and(lines, cv2.bitwise_not(dark))

    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.bitwise_and(mask, cv2.bitwise_not(lines))

CRACK_DARK_RATIO = 0.40   # piece must be darker than 60% of the wall's local brightness

# ── CRACK-vs-MASONRY GATES ───────────────────────────────────────
# The segmenter also lights up brick faces, roof tiles, mortar joints and window
# grates: regions that are dark and textured but are NOT cracks. These gates
# throw such regions away before anything is measured, drawn, counted or pushed
# to the VR twin. A crack has to satisfy ALL of them:
#
#   length  - a long thin centreline (scaled to the image, so a 480p webcam and a
#             4000px phone photo behave the same)
#   width   - average stroke width (area / skeleton length) stays hairline-ish
#   fill    - stroke occupies only part of its bounding box (a crack is a line,
#             not a filled patch)
#   compact - 4*pi*area/perimeter^2 stays low (1.0 would be a disc)
#   trough  - the stroke sits in a dark valley with BRIGHTER wall on both flanks,
#             measured perpendicular to the stroke. Brick texture has no such
#             paired flank, so its median trough contrast stays near zero.
#
# The trough gate is expressed RELATIVE to the local wall brightness as well as in
# absolute grey levels, so it behaves the same on a white plaster wall, a sunlit
# concrete facade and a dark brick wall.
CRACK_MIN_LEN_FRAC   = 0.030  # of the frame diagonal (min 30 px)
CRACK_MAX_WIDTH      = 15.0   # px
CRACK_MAX_FILL       = 0.60   # area / bounding-box area
CRACK_MAX_COMPACT    = 0.50   # 4*pi*area / perimeter^2
CRACK_MIN_TROUGH     = 5.0    # grey levels of flank contrast (absolute floor)
CRACK_MIN_TROUGH_REL = 0.055  # ...or this share of the local wall brightness
CRACK_MIN_TROUGH_FRAC = 0.42  # share of centreline points that must be a trough

# Local-contrast ("darker than the wall right here") gate, see _dark_stroke_mask.
CRACK_LOCAL_REL   = 0.10   # relative drop vs the mid-scale background
CRACK_LOCAL_ABS   = 18.0   # absolute grey-level drop that always counts
CRACK_LOCAL_K_FRAC = 0.05  # background window, as a share of the short side
CRACK_MAX_BRIGHT_RATIO = 1.02  # a crack pixel is never brighter than the wall

# ── MASONRY (brick / block) AWARENESS ────────────────────────────
# Mortar joints are thin, straight, regularly spaced lines in two roughly
# perpendicular directions. The gate below only discards a stroke that IS one of
# those joints; a real crack that runs across the same brick wall (it is longer
# than one brick, wanders, or crosses the joints) is kept.
MASONRY_MIN_SUPPORT = 0.42   # share of long-edge length aligned with the joint axis
MASONRY_ANGLE_TOL   = np.deg2rad(12.0)
MASONRY_MAX_PITCH_MAD = 0.35  # joint spacing regularity (MAD / median)
MASONRY_MAX_JOINT_FRAC = 2.6  # a single joint spans at most this many pitches


def _ellipse_kernel(px):
    px = int(max(3, px)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def _local_background(gray):
    """Brightness of the WALL AROUND each pixel, with cracks filled in.

    A morphological closing bridges narrow dark strokes (cracks, mortar) from the
    surrounding wall, so `background - pixel` is large on a crack and ~0 on a
    brick face, whatever the overall exposure of the wall is. This replaces the
    old global "darker than 40% of the frame median" rule, which failed on dark
    brick walls and on walls with a lighting gradient.

    The kernel is deliberately only a little wider than a crack (not brick-sized):
    a window wider than the masonry units would fill whole brick faces in and make
    the entire wall read as "darker than its surroundings".
    """
    g = gray.astype(np.float32)
    h, w = gray.shape[:2]
    k = int(CRACK_LOCAL_K_FRAC * min(h, w))
    k = int(np.clip(k, 11, 61))
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, _ellipse_kernel(k))


def _dark_stroke_mask(gray):
    """Binary mask of pixels that are clearly darker than the wall around them.

    Three complementary tests, unioned:
      * local  - darker than the mid-scale background (works on any exposure),
      * strong - a big absolute drop (deep, wide open fissures),
      * global - darker than a share of the frame median (the original rule; it
        is very effective on bright plaster walls where the segmenter also lights
        up the mortar joints).
    All of them are then intersected with "not brighter than the wall itself":
    the segmenter also fires on ceiling lights, screens and specular highlights,
    and an edge that is BRIGHTER than its surroundings can never be a crack.
    """
    bg = _local_background(gray)
    g = gray.astype(np.float32)
    diff = bg - g
    rel = (diff / np.maximum(bg, 1.0)) >= CRACK_LOCAL_REL
    strong = diff >= CRACK_LOCAL_ABS
    wall_med = float(np.median(gray))
    glob = g < CRACK_DARK_RATIO * wall_med
    not_bright = g <= CRACK_MAX_BRIGHT_RATIO * wall_med
    return (((rel | strong | glob) & not_bright).astype(np.uint8)) * 255


def _skeleton_normals(skel, sigma=2.0, win=4.0):
    """Unit vectors ACROSS the stroke at every pixel of a 1-px skeleton.

    Uses the structure tensor of the blurred skeleton: on a ridge the gradient
    points away from the line on BOTH sides, and `sum(g g^T)` over a window is
    sign-invariant, so its dominant eigenvector is perpendicular to the stroke
    even exactly on the medial axis. (Two earlier attempts failed here: the
    gradient of the distance transform is zero on the medial axis, and so is the
    point-wise gradient of the blurred skeleton - both made the flank direction
    random on WIDE cracks, which is why open fissures were never detected.)
    """
    s = (skel > 0).astype(np.float32)
    s = cv2.GaussianBlur(s, (0, 0), sigma)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), win)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), win)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), win)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return np.cos(theta), np.sin(theta)


def _ridge_contrast(comp, skel, gray):
    """How much BRIGHTER the wall is on both sides of this stroke.

    For every centreline pixel both flanks are sampled perpendicular to the
    stroke, just outside its edge, and compared with the pixel on the centreline:

        trough = min(left, right) - centre

    A real crack is a dark valley between two lit wall faces -> clearly positive
    everywhere along its length. Brick / plaster texture and the inside of a dark
    blob have no consistent pair of brighter flanks -> median near zero.

    Two probe distances are used and the better one is kept, so the gate works
    for hairline cracks (2-3 px) and for wide open fissures (15+ px) alike.

    `comp`, `skel` and `gray` must already be cropped to the same ROI.
    Returns (median trough contrast, share of points that are troughs), or
    (None, 0.0) when there is no centreline to sample.
    """
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return None, 0.0
    H, W = gray.shape[:2]
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    ux, uy = _skeleton_normals(skel)
    g = gray.astype(np.float32)

    def sample(xf, yf):
        mx = np.clip(xf, 0, W - 1).astype(np.float32).reshape(-1, 1)
        my = np.clip(yf, 0, H - 1).astype(np.float32).reshape(-1, 1)
        return cv2.remap(g, mx, my, cv2.INTER_LINEAR).ravel()

    half = np.clip(dist[ys, xs], 0.5, 12.0)
    centre = g[ys, xs]
    nx, ny = ux[ys, xs], uy[ys, xs]
    best = np.full(len(xs), -255.0, np.float32)
    for scale in (1.15, 1.9):
        off = half * scale + 2.0
        left = sample(xs + nx * off, ys + ny * off)
        right = sample(xs - nx * off, ys - ny * off)
        trough = np.minimum(left, right) - centre
        best = np.maximum(best, trough)
    return float(np.median(best)), float((best > 6.0).mean())


def _masonry_profile(gray):
    """Detect a brick / block wall and describe its mortar grid.

    Masonry shows up as a family of LONG, STRAIGHT, PARALLEL edges at a regular
    pitch, repeated across the whole frame (the bed joints). That is what
    `_looks_like_brick_wall` used to try to infer from the detected components;
    working from the image instead is far more reliable, because it still works
    when the segmenter only lit up the crack.

    Only ONE family is required: in a running-bond brick wall the bed joints are
    continuous, while the head joints are short and staggered, so a perfectly
    perpendicular second family often never appears. The perpendicular axis is
    therefore assumed, and the regularity of the pitch is what proves masonry.

    Returns a dict:
        detected  - True when a regular, dominant joint family was found
        axes      - joint orientation and its perpendicular (radians, mod pi)
        pitch     - median joint spacing in px (0 when it could not be measured)
        support   - share of long-edge length that lies on the dominant axis
    """
    prof = {'detected': False, 'axes': (), 'pitch': 0.0, 'support': 0.0}
    try:
        H, W = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        med = float(np.median(blur))
        lo = int(max(8, 0.25 * med))
        hi = int(max(lo + 12, min(220, 0.9 * med)))
        edges = cv2.Canny(blur, lo, hi)
        min_len = max(18, int(0.055 * float(np.hypot(H, W))))
        segs = cv2.HoughLinesP(edges, 1, np.pi / 180.0,
                               threshold=max(16, min_len // 2),
                               minLineLength=min_len, maxLineGap=4)
        if segs is None or len(segs) < 6:
            return prof
        angs, lens, mids = [], [], []
        # OpenCV >= 5 returns (N, 4); older releases return (N, 1, 4).
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            angs.append(math.atan2(dy, dx) % math.pi)
            lens.append(math.hypot(dx, dy))
            mids.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        angs = np.asarray(angs)
        lens = np.asarray(lens, np.float32)
        total = float(lens.sum())
        if total <= 0:
            return prof

        # Weighted vote for the dominant joint direction.
        bins = np.linspace(0.0, math.pi, 37)
        hist, _ = np.histogram(angs, bins=bins, weights=lens)
        a1 = float(bins[int(np.argmax(hist))] + (bins[1] - bins[0]) * 0.5)

        def circ_dist(a, b):
            d = np.abs(a - b) % math.pi
            return np.minimum(d, math.pi - d)

        near1 = circ_dist(angs, a1) <= MASONRY_ANGLE_TOL
        support = float(lens[near1].sum()) / total
        idx = np.where(near1)[0]
        if support < MASONRY_MIN_SUPPORT or len(idx) < 4:
            return prof

        # Joint pitch and, more importantly, how REGULAR it is. A textured wall
        # (concrete, stone, plaster) also produces parallel edges, but not at a
        # steady spacing - masonry is laid out brick after brick.
        nx, ny = -math.sin(a1), math.cos(a1)          # normal to the joints
        rho = np.sort(np.array([mids[i][0] * nx + mids[i][1] * ny for i in idx]))
        pitch = 0.0
        regular = 0.0
        gaps = np.diff(rho)
        gaps = gaps[gaps > max(5.0, min_len * 0.4)]
        if gaps.size >= 3:
            pitch = float(np.median(gaps))
            regular = float(np.median(np.abs(gaps - pitch)) / max(pitch, 1.0))
            if pitch > 0.55 * float(np.hypot(H, W)):
                return prof                        # "gaps" are not joints

        detected = bool(gaps.size >= 3 and regular <= MASONRY_MAX_PITCH_MAD)
        prof.update({'detected': detected,
                     'axes': (a1, (a1 + math.pi / 2.0) % math.pi),
                     'pitch': round(pitch, 1),
                     'support': round(support, 3),
                     'regularity': round(regular, 3)})
    except Exception as e:
        print(f'masonry profile error: {e}')
    return prof


def _line_geometry(skel):
    """Principal orientation + straightness (span / path length) of a skeleton."""
    if skel is None:
        return None, 0.0
    ys, xs = np.nonzero(skel)
    if len(xs) < 8:
        return None, 0.0
    p = np.column_stack((xs, ys)).astype(np.float32)
    dev = p - p.mean(axis=0)
    cov = np.cov(dev.T)
    evals, evecs = np.linalg.eigh(cov)
    # least-variance eigenvector is the line normal -> direction is the other one
    v = evecs[:, int(np.argmax(evals))]
    ang = math.atan2(float(v[1]), float(v[0])) % math.pi
    path = _skeleton_length(skel)
    span = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    straight = span / path if path > 1e-6 else 0.0
    return ang, straight


def _is_mortar_joint(skel, length, prof):
    """True when this stroke alone looks like a mortar joint (straight + aligned).

    Kept for callers that only have one stroke; `_reject_mortar_joints` is the
    version the measurement pipeline uses, because one stroke proves nothing: on
    a brick wall a straight settlement crack is also straight and aligned with the
    joints. Only a SET of such strokes is masonry.
    """
    if not prof or not prof.get('detected') or skel is None:
        return False
    ang, straight = _line_geometry(skel)
    if ang is None or straight < 0.86:
        return False
    axes = prof.get('axes') or ()
    aligned = False
    for a in axes:
        d = abs(ang - a) % math.pi
        d = min(d, math.pi - d)
        if d <= MASONRY_ANGLE_TOL:
            aligned = True
            break
    if not aligned:
        return False
    pitch = float(prof.get('pitch') or 0.0)
    if pitch > 0 and length > MASONRY_MAX_JOINT_FRAC * pitch:
        return False  # runs across several bricks -> a crack, not a joint
    return True


def _reject_mortar_joints(candidates, prof):
    """Remove mortar joints from a set of candidate cracks on a masonry wall.

    A brick / block / stone wall produces MANY straight strokes that follow the
    mortar grid. Requiring the whole set to be "jointy" before dropping anything
    keeps a single straight crack - the common settlement/diagonal crack on a
    brick wall - which the old code used to delete together with the joints.
    """
    if not prof or not prof.get('detected') or not candidates:
        return candidates
    axes = prof.get('axes') or ()
    pitch = float(prof.get('pitch') or 0.0)
    jointish = []
    for skel, contour, length, width, conf in candidates:
        ang, straight = _line_geometry(skel)
        aligned = False
        if ang is not None:
            for a in axes:
                d = abs(ang - a) % math.pi
                d = min(d, math.pi - d)
                if d <= MASONRY_ANGLE_TOL:
                    aligned = True
                    break
        is_joint = bool(aligned and straight >= 0.86)
        if is_joint and pitch > 0 and length > MASONRY_MAX_JOINT_FRAC * pitch:
            is_joint = False
        jointish.append(is_joint)
    if sum(jointish) < 3:
        return candidates            # 1-2 straight strokes: a crack, not a grid
    return [c for c, is_joint in zip(candidates, jointish) if not is_joint]

def _draw_wall_tag(img, prof, prefix='WALL'):
    """Small corner tag naming the masonry the scan is standing in front of."""
    if img is None or img.size == 0:
        return
    pitch = prof.get('pitch') or 0.0
    label = 'BRICK / MASONRY WALL'
    if pitch:
        label += f'  (joint pitch {int(pitch)}px)'
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (0, h - th - 12), (tw + 10, h), (0, 0, 0), -1)
    cv2.putText(img, label, (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (90, 220, 255), 1, cv2.LINE_AA)


def _draw_results(overlay, results, sev_color):
    """Draw measured crack(s): centerline skeleton pixels directly (1-2 px) on
    the SAME original-size image the mask was built on, so the line hugs every
    bend and branch of the crack. Contour is only a fallback."""
    for skel, contour, length, width, conf in results:
        if skel is not None and cv2.countNonZero(skel) > 0:
            halo = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=1)
            overlay[halo > 0] = (255, 255, 255)
            overlay[skel > 0] = sev_color
        elif contour is not None:
            line = contour.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [line], False, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(overlay, [line], False, sev_color, 1, cv2.LINE_AA)

def _detect_static(frame, tiled=True):
    """Detect on still frames/video.

    Large stills: 640px tiles w/ 20% overlap, imgsz=640, TTA — thin hairline
    parts that a single global pass would wash out get captured; all masks are
    unioned so a crack spanning tiles becomes one binary mask. Videos: one
    full-frame pass (tiled=False) for speed. Falls back to MODEL_FALLBACK on a
    complete miss. Returns (boxes, confs, combined_mask).
    """
    H, W = frame.shape[:2]
    combined = np.zeros((H, W), np.uint8)
    boxes, confs = [], []
    net = model
    if net is None and MODEL_FALLBACK is None:
        return boxes, confs, combined
    try:
        if tiled and max(H, W) > 768:
            step = int(TILE_SIZE * 0.8)   # 20% overlap
            for y0 in range(0, H, step):
                for x0 in range(0, W, step):
                    x1 = min(W, x0 + TILE_SIZE)
                    y1 = min(H, y0 + TILE_SIZE)
                    tile = frame[y0:y1, x0:x1]
                    if tile.shape[0] < 32 or tile.shape[1] < 32:
                        continue
                    r0 = _run_model_once(net, tile, 640, augment=True)
                    tm = _combined_mask_from_result(r0, tile.shape[1], tile.shape[0])
                    if cv2.countNonZero(tm):
                        combined[y0:y1, x0:x1] = cv2.bitwise_or(combined[y0:y1, x0:x1], tm)
                        if r0.boxes is not None and len(r0.boxes) > 0:
                            for b, c in zip(r0.boxes.xyxy, r0.boxes.conf):
                                boxes.append((int(b[0]) + x0, int(b[1]) + y0,
                                              int(b[2]) + x0, int(b[3]) + y0))
                                confs.append(float(c))
        else:
            imgsz = _infer_imgsz_for_image(H, W)
            r0 = _run_model_once(net, frame, imgsz, augment=True)
            combined = _combined_mask_from_result(r0, W, H)
            if r0.boxes is not None and len(r0.boxes) > 0:
                for b, c in zip(r0.boxes.xyxy, r0.boxes.conf):
                    boxes.append(tuple(map(int, b)))
                    confs.append(float(c))
        if not boxes and MODEL_FALLBACK is not None:
            r1 = _run_model_once(MODEL_FALLBACK, frame, 640, augment=True)
            combined = _combined_mask_from_result(r1, W, H)
            if r1.boxes is not None and len(r1.boxes) > 0:
                for b, c in zip(r1.boxes.xyxy, r1.boxes.conf):
                    boxes.append(tuple(map(int, b)))
                    confs.append(float(c))
    except Exception as e:
        print(f"Static detect error: {e}")
    return boxes, confs, combined


SEVERITY_COLORS = {
    0: (80, 200, 80),   # SAFE - green
    1: (0, 215, 255),   # Minor - gold/yellow
    2: (0, 140, 255),   # Medium - orange
    3: (0, 0, 255),     # Severe - red
}

def _is_rectangular_mask(poly, eps=0.02):
    """True if polygon is essentially a rectangle/box, OR a blob that fills most of its bounding box —
    either way it isn't a thin crack shape and should never be drawn/used directly."""
    if poly is None or len(poly) < 4:
        return False
    try:
        # Use bounding rect area vs contour area — box fills ~95-100% of its bbox, crack never does
        cnt = poly.astype(np.int32).reshape((-1, 1, 2)) if poly.ndim == 2 else poly
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = max(1, w * h)
        poly_area = cv2.contourArea(cnt)
        if poly_area <= 0:
            return True
        fill_ratio = poly_area / bbox_area
        # Box: fills almost entire bbox and approx is 4 points
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perim, True)
        if len(approx) == 4 and fill_ratio > 0.82:
            return True
        # Also catch 4-pt exact rectangles regardless of point count pre-approx
        if len(poly) == 4 and abs(poly_area - bbox_area) / bbox_area < 0.08:
            return True
        # Catch any blob that fills most of its bbox even if edges are jagged (not exactly 4 points) —
        # a real crack is thin and never covers this much of its own bounding box.
        if fill_ratio > 0.55:
            return True
        return False
    except Exception:
        return True


def _looks_like_brick_wall(components, min_count=6):
    """Deprecated: the all-or-nothing "this looks like a brick wall, report NO
    cracks" veto that used to live here threw away every real crack on a masonry
    wall (brick, block, stone) as soon as enough mortar joints were picked up as
    well. Masonry is now handled in two places instead:

      * `_masonry_profile()` reads the mortar grid from the IMAGE (two long,
        straight, regularly spaced edge families) and labels the scene;
      * `_is_mortar_joint()` drops an individual stroke only when it really is a
        single joint (straight, axis-aligned, no longer than a few bricks).

    Kept as a thin wrapper for any external caller.
    """
    prof = {'detected': False}
    return False


def measure_combined_mask(mask, conf=1.0, frame=None, masonry=None, debug=None):
    """Measure crack(s) from a single union mask.

    Returns list of (skeleton_or_None, contour, length, width, conf) ordered by
    length desc. Length is the FULL skeleton length INCLUDING all branches
    (diagonal-weighted). width = mask area / length. Speckle (<25px) is dropped,
    and every component must also pass the crack-vs-masonry gates above -- a
    brick wall or a textured plaster patch is rejected instead of being reported
    as one giant crack.

    When `frame` is given the mask is refined first: only pixels that are clearly
    darker than the LOCAL wall survive (see `_dark_stroke_mask`), which keeps the
    crack and drops mortar joints and brick faces even on a wall whose exposure
    varies across the picture. `masonry` may carry a precomputed
    `_masonry_profile` so the brick/block grid is only measured once per frame.
    """
    if mask is None or cv2.countNonZero(mask) < 25:
        return []
    H, W = mask.shape[:2]
    gray = None
    if frame is not None and frame.shape[:2] == mask.shape[:2]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if masonry is None:
            masonry = _masonry_profile(gray)
        mask = cv2.bitwise_and(mask, _dark_stroke_mask(gray))
        mask = _remove_straight_lines(mask, gray)
    else:
        mask = _remove_straight_lines(mask)
    if cv2.countNonZero(mask) < 25:
        if debug is not None:
            debug.append(('mask emptied by dark/line gates', 0, 0))
        return []

    min_len = max(30.0, CRACK_MIN_LEN_FRAC * float(np.hypot(H, W)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # One OPEN pass drops single-pixel speckle. A CLOSE would fatten a hairline
    # crack into a band (and merge it with nearby speckle), which inflates the
    # measured width and shortens the skeleton - the caller would then reject a
    # real crack for being too wide.
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clean)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        comp = np.where(lab == i, 255, 0).astype(np.uint8)
        cnts = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        if contour is None:
            continue
        skel = None
        if _SK_HAS:
            try:
                skel = (_sk_skeletonize(comp > 0).astype(np.uint8)) * 255
            except Exception:
                skel = None
        if skel is None:
            skel = _skeletonize(comp)
        skel_px = cv2.countNonZero(skel) if skel is not None else 0
        if skel_px >= 10:
            length = _skeleton_length(skel)
            skel_out = skel
        else:
            length = cv2.arcLength(contour, True) / 2.0
            skel_out = None

        if length < min_len:
            if debug is not None:
                debug.append(('too short', round(length, 1), area))
            continue
        raw_width = area / length if length > 0 else float(area)
        if raw_width > CRACK_MAX_WIDTH:
            if debug is not None:
                debug.append(('too wide', round(raw_width, 1), area))
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if area / float(max(1, bw * bh)) > CRACK_MAX_FILL:
            if debug is not None:
                debug.append(('fills bbox', round(area / float(max(1, bw * bh)), 2), area))
            continue
        perim = float(cv2.arcLength(contour, True))
        if perim > 0 and (4.0 * np.pi * area) / (perim * perim) > CRACK_MAX_COMPACT:
            if debug is not None:
                debug.append(('too compact', area, length))
            continue
        trough = trough_frac = None
        if gray is not None and skel_out is not None:
            pad = int(CRACK_MAX_WIDTH) + 8
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
            roi = gray[y0:y1, x0:x1]
            trough, trough_frac = _ridge_contrast(comp[y0:y1, x0:x1],
                                                  skel_out[y0:y1, x0:x1],
                                                  roi)
            # The required contrast scales with the wall's own brightness, so a
            # faint hairline on a bright wall and a clear crack on a dark brick
            # wall are judged the same way.
            wall_here = float(np.median(roi)) if roi.size else 0.0
            need = max(CRACK_MIN_TROUGH, CRACK_MIN_TROUGH_REL * wall_here)
            if (trough is None or trough < need
                    or trough_frac < CRACK_MIN_TROUGH_FRAC):
                if debug is not None:
                    debug.append(('no trough', round(trough or -1, 1),
                                  round(need, 1), round(trough_frac, 2), area, length))
                continue

        width = max(1.0, min((area / length if length > 0 else 1.0), 40))
        if debug is not None:
            debug.append(('accept', round(float(length), 1), round(float(width), 1),
                          round(float(trough), 1) if trough is not None else None,
                          round(float(trough_frac), 2) if trough_frac is not None else None,
                          area))
        candidates.append((skel_out, contour, round(float(length), 1),
                           round(float(width), 1), conf))

    # Masonry grid: remove mortar joints - reasoned about as a SET, so one long
    # straight settlement crack on a brick wall is not thrown away with them.
    n_before = len(candidates)
    candidates = _reject_mortar_joints(candidates, masonry)
    if debug is not None and len(candidates) != n_before:
        debug.append(('mortar grid', n_before, len(candidates)))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates

def _skeletonize(bin_img):
    """Zhang-Suen thinning — returns 1-px skeleton that exactly follows crack centerline."""
    try:
        return cv2.ximgproc.thinning(bin_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        # iterative morphological skeleton fallback
        skel = np.zeros_like(bin_img)
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = bin_img.copy()
        while True:
            eroded = cv2.erode(img, el)
            opening = cv2.dilate(eroded, el)
            temp = cv2.subtract(img, opening)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

def _order_skeleton_points(skel):
    """Walk the 1-px skeleton and return an ordered (N,1,2) polyline that follows the crack without branching."""
    pts_y, pts_x = np.where(skel > 0)
    if len(pts_x) == 0:
        return None
    # Build neighbor map 8-connected
    pts = np.column_stack((pts_x, pts_y))
    # Find endpoint (pixel with only 1 neighbor) to start walk
    from collections import defaultdict, deque
    # Use dict for fast lookup
    point_set = set(map(tuple, pts.tolist()))
    def neighbors(p):
        x, y = p
        return [(x+dx, y+dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0) and (x+dx, y+dy) in point_set]
    # Find endpoints (degree 1)
    endpoints = [p for p in point_set if len(neighbors(p)) == 1]
    start = endpoints[0] if endpoints else tuple(pts[0].tolist())
    ordered = []
    visited = set()
    cur = start
    prev = None
    while True:
        ordered.append(cur)
        visited.add(cur)
        nbrs = [n for n in neighbors(cur) if n not in visited]
        # Prefer straight continuation (avoid branching jitter) - pick neighbor closest to direction
        if not nbrs:
            break
        if prev is not None and len(nbrs) > 1:
            # choose neighbor most collinear with prev->cur direction
            vx, vy = cur[0]-prev[0], cur[1]-prev[1]
            def dot(n):
                ux, uy = n[0]-cur[0], n[1]-cur[1]
                return vx*ux + vy*uy
            nbrs.sort(key=dot, reverse=True)
        prev, cur = cur, nbrs[0]
        if len(ordered) > 8000:  # safety cap
            break
    if len(ordered) < 8:
        return None
    arr = np.array(ordered, dtype=np.int32).reshape((-1, 1, 2))
    return arr

def process_crack_region(frame, x1, y1, x2, y2, mask_polygon=None):
    """Measure crack length + width.

    Returns (line, length, width, is_open):
      - A real YOLO-seg mask polygon (thin, non-boxy) is used directly as a CLOSED outline (is_open=False).
      - Everything else is walked via the skeleton and returned as an OPEN centerline (is_open=True),
        so the drawn shape always hugs the actual crack pixels and can never render as a box or a
        filled/self-closing tangle.
    """
    if mask_polygon is not None and len(mask_polygon) > 6 and not _is_rectangular_mask(mask_polygon):
        try:
            contour = mask_polygon.astype(np.int32).reshape((-1, 1, 2))
            # Minimal smoothing only — keep every kink exactly on the crack
            epsilon = 0.0003 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            perim = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            length = perim / 2
            width = area / length if length > 0 else 0
            return contour, round(length, 1), round(width, 1), False
        except Exception:
            pass

    # Fallback: direct crack-pixel segmentation (no edge thickening → no box fill)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0, 0, False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    # Dark-crack mask via adaptive threshold — captures only the dark crack pixels, not a thickened edge
    # Crack is darker than wall, so invert after Otsu on blurred enhanced
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Keep only dark pixels that are on/near edges to avoid wall texture
    med = np.median(blur)
    lower = int(max(8, 0.30 * med))
    upper = int(min(220, 1.05 * med))
    edges = cv2.Canny(blur, lower, upper)
    # Dilate edges 1px to create a narrow band, intersect with dark mask → tight crack only
    edge_band = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    crack_mask = cv2.bitwise_and(binary, edge_band)
    # Light clean only — no CLOSE that creates box fill
    kernel = np.ones((3, 3), np.uint8)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Remove tiny speckles (<12px)
    if cv2.countNonZero(crack_mask) < 12:
        # fallback to edge-only if threshold missed faint crack
        crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if cv2.countNonZero(crack_mask) < 12:
        return None, 0, 0, False
    outer_cnts, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outer_cnts:
        return None, 0, 0, False
    # Pick the longest/thinnest contour (true crack), not the largest boxy blob
    # Score by perimeter^2/area — high for thin crack, low for compact box
    def thin_score(c):
        a = max(1, cv2.contourArea(c))
        p = cv2.arcLength(c, True)
        return (p * p) / a
    outer = max(outer_cnts, key=thin_score)
    area_raw = cv2.contourArea(outer)
    if area_raw < 12:
        return None, 0, 0, False

    # Always try the skeleton centerline FIRST, regardless of whether the raw blob looks boxy —
    # this is what actually follows the crack. The raw blob outline (outer) is never drawn.
    skel = _skeletonize(crack_mask)
    ordered = _order_skeleton_points(skel)
    if ordered is not None and len(ordered) >= 8:
        length = cv2.arcLength(ordered, False)
        width = (area_raw / length) if length > 0 else 0
        width = max(1.0, min(width, 40))
        off = ordered.copy()
        off[:, :, 0] = np.clip(off[:, :, 0] + x1, 0, frame.shape[1] - 1)
        off[:, :, 1] = np.clip(off[:, :, 1] + y1, 0, frame.shape[0] - 1)
        return off, round(float(length), 1), round(float(width), 1), True

    # Skeleton walk failed (mask too small/broken) — only now consider the raw blob outline,
    # and only if it isn't boxy. A boxy blob with no usable skeleton is discarded entirely
    # rather than drawn as a box.
    if _is_rectangular_mask(outer):
        return None, 0, 0, False
    outer = cv2.approxPolyDP(outer, 0.3, True)
    if len(outer) < 6:
        outer = max(outer_cnts, key=thin_score)
    length = cv2.arcLength(outer, True) / 2
    area = cv2.contourArea(outer)
    width = (area / length) if length > 0 else 0
    width = max(1.0, min(width, 40))
    offset = outer.copy()
    offset[:, :, 0] = np.clip(offset[:, :, 0] + x1, 0, frame.shape[1] - 1)
    offset[:, :, 1] = np.clip(offset[:, :, 1] + y1, 0, frame.shape[0] - 1)
    return offset, round(float(length), 1), round(float(width), 1), False


def _line_from_component(comp, offset_x, offset_y, min_len):
    """Skeletonize one component into an ordered centerline in frame coords.

    Returns (line, length, width, area) or None if the blob is too boxy/short
    to be a crack (never a full-size wall blob: area is hard-capped so a low-contrast
    frame that over-segments as one giant region can never hang or render as a box).
    """
    area = int(cv2.countNonZero(comp))
    if area < 25 or area > 60000:
        return None
    cnts = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return None
    per = float(cv2.arcLength(cnts[0], True))
    compact = 4 * np.pi * area / (per * per) if per > 0 else 0.0
    if compact > 0.45:
        # square/boxy blob (fill exhausts its bbox) — not a thin crack shape
        return None
    skel = _skeletonize(comp)
    ordered = _order_skeleton_points(skel)
    if ordered is None or len(ordered) < 10:
        return None
    pts = np.array(ordered, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] += offset_x
    pts[:, 1] += offset_y
    line = pts.reshape(-1, 1, 2)
    length = float(cv2.arcLength(line, False))
    if length < min_len:
        return None
    width = max(1.0, min(80, area / max(1.0, length)))
    if width > max(12.0, 0.35 * length):
        return None
    if len(line) > 2000:
        step = len(line) // 2000
        line = line[::step]
        line = np.concatenate([line, line[-1:]], axis=0) if step > 1 else line
    return line.astype(np.int32), length, width, area


def extract_crack_lines(frame, ai_boxes, ai_confs, ai_masks):
    """Produce ONE continuous end-to-end centerline per crack.

    YOLO splits one long crack into several overlapping boxes, and its seg masks
    come back as 4-pt boxy fills — neither usable as-is. So:

      1. Per detection box, two mines run with different thresholds:
         - Adaptive-only (dark/light, blockSize 41, C=-5) intersected with a
           median-derived Canny edge gate. Global Otsu is deliberately EXCLUDED
           here: on a uniform wall it saturates to near-the-whole-frame and turns
           a crack into a giant blob whose skeleton stub is useless.
         - A dilated union band around all boxes, mined with CLAHE+Otsu (the
           adaptive path can miss faint cracks on halftone tiles). Per-box and
           band fragments are merged into one candidate set.
      2. Each component is skeletonized and walked into an ordered centerline
         (never a boxy fill), then fragments that continue the same crack are
         stitched end-to-end (gap <= 30% of the shorter line, no hairpin turns).
      3. Coiled scribbles (endpoint span << path length, i.e. the walker doubled
         back on itself instead of following a real crack) are dropped.

    Component area is hard-capped so over-segmented low-contrast frames can never
    hang or render as wall-sized boxes. Returns at most 6 cracks, longest first:
    list of (line, conf, length, width, is_open=True).
    """
    if not ai_boxes:
        return []
    H, W = frame.shape[:2]
    diag = float((H * H + W * W) ** 0.5)
    min_len = max(15.0, 0.022 * diag)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Per-box adaptive path ─────────────────────────────────────────────
    # Bilateral-smoothed edges with median-derived Canny (robust on both
    # high- and very-low-contrast frames), dark + light adaptive channels.
    bf = cv2.bilateralFilter(blur, 9, 60, 60)
    med = float(np.median(bf))
    edges = cv2.Canny(bf, int(max(4, 0.20 * med)), int(min(255, 1.5 * med)))
    ed = cv2.dilate(edges, np.ones((2, 2), np.uint8), 1)
    dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, -5)
    light = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -5)

    # ── CLAHE+Otsu band path ──────────────────────────────────────────────
    # Catches faint cracks on textured/halftone walls that adaptive misses.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    bl2 = cv2.GaussianBlur(enh, (3, 3), 0)
    _, otsu = cv2.threshold(bl2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    med2 = float(np.median(bl2))
    ce_low = max(6, int(0.30 * med2)); ce_hi = max(ce_low + 10, min(220, int(1.05 * med2)))
    otsu_edges = cv2.dilate(cv2.Canny(bl2, ce_low, ce_hi), np.ones((3, 3), np.uint8), 1)
    otsu_e = cv2.bitwise_and(otsu, otsu_edges)

    def gather(mask, ox, oy, thr_len):
        """Components→ordered centerlines for one already-mined region.
        mask is cropped to a region whose top-left is (ox,oy) in frame coords;
        skeleton coords are mask-relative, so offset by the region origin only.
        Yields [line, length, width, span, area] entries."""
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < 25 or a > 60000:
                continue
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if w < 4 or h < 4:
                continue
            comp = np.where(lab == i, 255, 0).astype(np.uint8)
            res = _line_from_component(comp, ox, oy, thr_len)
            if res is not None:
                line, length, width, area = res
                span = float(np.hypot(line[0][0][0] - line[-1][0][0], line[0][0][1] - line[-1][0][1]))
                out.append([line, length, width, span, area])
        return out

    candidates = []
    # Per-box adaptive mining — the primary source; wide cracks get caught here.
    for (x1, y1, x2, y2) in ai_boxes:
        pad = max(12, int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1)))
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(W, x2 + pad), min(H, y2 + pad)
        if bx2 - bx1 < 8 or by2 - by1 < 8:
            continue
        d_r = dark[by1:by2, bx1:bx2]; l_r = light[by1:by2, bx1:bx2]; e_r = ed[by1:by2, bx1:bx2]
        thin = cv2.max(cv2.bitwise_and(d_r, e_r), cv2.bitwise_and(l_r, e_r))
        k = np.ones((3, 3), np.uint8)
        thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, k, iterations=2)
        thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, k, iterations=1)
        candidates += gather(thin, bx1, by1, max(8.0, 0.5 * min_len))

    # Union-band CLAHE+Otsu mining — catches cracks whose pixels sit between
    # box borders or are too faint for the adaptive channel.
    band = np.zeros((H, W), np.uint8)
    for (x1, y1, x2, y2) in ai_boxes:
        pad = max(20, int(0.18 * (x2 - x1)), int(0.18 * (y2 - y1)))
        cv2.rectangle(band, (max(0, x1 - pad), max(0, y1 - pad)), (min(W, x2 + pad), min(H, y2 + pad)), 255, -1)
    xs, ys = np.where(band > 0)
    if len(xs) > 0 and len(xs) < H * W * 0.9:
        y0, x0, y1, x1 = xs.min(), ys.min(), xs.max(), ys.max()
        bm = band[y0:y1 + 1, x0:x1 + 1].copy()
        m = cv2.bitwise_and(otsu_e[y0:y1 + 1, x0:x1 + 1], bm)
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        candidates += gather(m, x0, y0, max(8.0, 0.6 * min_len))

    # Stitch fragments that continue the same crack (gap <= 30% of shorter line,
    # no hairpin U-turn). Greedy: merge first match, repeat until stable.
    cands = [list(c) for c in candidates]
    changed = True
    while changed:
        changed = False
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                if cands[i] is None or cands[j] is None:
                    continue
                a, b = cands[i], cands[j]
                a_end = (a[0][0][0], a[0][-1][0])
                b_end = (b[0][0][0], b[0][-1][0])
                gap_limit = max(30.0, 0.30 * min(a[1], b[1]))
                best = None
                for pe in a_end:
                    for qe in b_end:
                        d = float(np.hypot(pe[0] - qe[0], pe[1] - qe[1]))
                        if d <= gap_limit and (best is None or d < best[0]):
                            best = (d, pe, qe)
                if best is None:
                    continue
                _, pe, qe = best
                pa = a[0].reshape(-1, 2); pb = b[0].reshape(-1, 2)
                if float(np.hypot(pe[0] - pa[0][0], pe[1] - pa[0][1])) > 1e-6:
                    pa = pa[::-1]
                if float(np.hypot(qe[0] - pb[-1][0], qe[1] - pb[-1][1])) > 1e-6:
                    pb = pb[::-1]
                if len(pa) >= 3 and len(pb) >= 3:
                    v1 = pa[-1] - pa[-2]; v2 = pb[1] - pb[0]
                    cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                    if cosang < -0.35:
                        continue  # hairpin turn — likely two crossing cracks
                joined = np.concatenate([pa, pb], axis=0).astype(np.float32).reshape(-1, 1, 2)
                length = float(cv2.arcLength(joined, False))
                cands[i] = [joined.astype(np.int32), length,
                            max(1.0, min(80, (a[4] + b[4]) / max(1.0, length))),
                            float(np.hypot(joined[0][0][0] - joined[-1][0][0], joined[0][0][1] - joined[-1][0][1])),
                            a[4] + b[4]]
                cands[j] = None
                changed = True
                break
            if changed:
                break

    # Drop coiled scribbles: a real crack keeps endpoint span >= ~28% of its
    # path length; a coiled walk-back scribble has span ~0 and is not a crack.
    cands = [c for c in cands if c is not None and c[3] >= 0.28 * c[1]]
    cands.sort(key=lambda t: t[1], reverse=True)

    # Final gate: line must overlap a real detection box, be long enough & thin.
    lines = []
    for (line, length, width, span, area) in cands:
        if length < min_len:
            continue
        x, y, w, h = cv2.boundingRect(line.reshape(-1, 1, 2))
        overlap = 0
        for (bx1, by1, bx2, by2) in ai_boxes:
            ox1, oy1 = max(x, bx1), max(y, by1)
            ox2, oy2 = min(x + w, bx2), min(y + h, by2)
            if ox2 > ox1 and oy2 > oy1:
                overlap += (ox2 - ox1) * (oy2 - oy1)
        if overlap < max(6, 0.04 * area):
            continue
        conf = 0.0
        for b, c in zip(ai_boxes, ai_confs):
            if not (b[2] < x or b[3] < y or b[0] > x + w or b[1] > y + h):
                conf = max(conf, c)
        lines.append((line, float(conf), round(float(length), 1), round(float(width), 1), True))
    lines.sort(key=lambda t: t[2], reverse=True)
    return lines[:6]

# ─────────────────────────────────────────────
# DATABASE — SQLite (works on Render, no setup)
# ─────────────────────────────────────────────
DB_PATH = 'crack_data.db'

# ── VR digital twin calibration ─────────────────────────
# px -> cm for crack_area (placeholder; replace with the
# value your photogrammetry/Meshroom pipeline measures).
CM_PER_PX = 0.05
# Demo room coordinate space (matches unity_room_scanner
# CrackifyBuilding room: 6 wide, 6 deep, 4 tall, centered
# at origin, walls at |x|=3 and |z|=3). Placeholder until
# real 3D coords ship; see crack_world_position() below.
ROOM_WALL_EXTENT = 3.0
ROOM_Y_MIN, ROOM_Y_MAX = 0.4, 3.4


def _lan_ip():
    """Best-effort LAN IP of this PC (the VR headset reaches the server through it)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _image_base():
    """Base URL for crack images that the VR headset can actually reach.

    When the request comes in via localhost (PC-side uploads / Unity editor),
    'image_url' would otherwise be http://localhost:5000/... which resolves to
    the HEADSET on the Quest — the picture 404s and never appears in Unity.
    So any localhost request is rewritten to the PC's LAN IP (e.g.
    http://192.168.0.106:5000/...) which works from both the headset and editor.
    """
    try:
        host = request.host_url.rstrip('/')
    except Exception:
        host = None
    if host and 'localhost' not in host.lower() and '127.0.0.1' not in host:
        return host
    ip = _lan_ip()
    if ip:
        port = 5000
        if host:
            try:
                port = int(host.rsplit(':', 1)[-1])
            except Exception:
                port = 5000
        return f'http://{ip}:{port}'
    return host or 'http://localhost:5000'


def _image_base64(image_path):
    """Return a data URI ('data:image/png;base64,...') for a saved crack photo so Unity
    can display the actual picture directly from the WS payload (no URL fetch needed on
    the headset). The MIME type follows the file extension, so an alpha-carrying PNG is
    announced as PNG rather than JPEG. Empty string when the file is missing."""
    try:
        p = (image_path or '').lstrip('/')
        if not os.path.exists(p):
            return ''
        ext = os.path.splitext(p)[1].lower()
        mime = {'.png': 'image/png',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'}.get(ext, 'image/jpeg')
        with open(p, 'rb') as fh:
            data = base64.b64encode(fh.read()).decode('ascii')
        return f'data:{mime};base64,{data}'
    except Exception as e:
        print(f"image_base64 error: {e}")
        return ''

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cracks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path  TEXT,
            crack_level INTEGER,
            crack_area  INTEGER,
            crack_width REAL,
            timestamp   TEXT
        )
    ''')
    try:
        conn.execute("ALTER TABLE cracks ADD COLUMN crack_width REAL")
    except Exception:
        pass  # column already exists

    # ── location columns ──────────────────────────────────────────
    # Legacy ArUco columns are KEPT so historical rows are never lost, but nothing writes to
    # them any more. Location is now assigned in Unity by raycasting the crack onto the wall.
    _legacy_location_columns = [
        ("marker_id",      "INTEGER"),   # kept, read-only (historical)
        ("location_name",  "TEXT"),      # kept, read-only
        ("offset_x_cm",    "REAL"),      # kept, read-only
        ("offset_y_cm",    "REAL"),      # kept, read-only
        ("width_cm",       "REAL"),      # kept, read-only
        ("height_cm",      "REAL"),      # kept, read-only
        ("located",        "INTEGER DEFAULT 0"),   # kept, read-only
    ]

    # Still written today - the confidence of the detection.
    _active_columns = [
        ("confidence", "REAL"),
    ]

    # Current placement columns. A row is 'pending' until Unity has raycast a position for
    # it, then 'placed'. ALTER TABLE ADD COLUMN is safe: existing rows pick up the DEFAULT,
    # so every crack saved before this migration becomes 'pending' automatically.
    _placement_columns = [
        ("status",     "TEXT DEFAULT 'pending'"),
        ("pos_x",      "REAL"),
        ("pos_y",      "REAL"),
        ("pos_z",      "REAL"),
        ("normal_x",   "REAL"),
        ("normal_y",   "REAL"),
        ("normal_z",   "REAL"),
        # Which camera produced this picture: 'rgb' or 'thermal'.
        ("camera_type", "TEXT DEFAULT 'rgb'"),
        # A single Capture writes one RGB row and one thermal row sharing this id, so the
        # VR app can treat them as ONE crack and swap its picture between the two feeds.
        ("capture_group", "TEXT"),
        # ── inspection feedback ──────────────────────────────────────────
        # Existing rows pick up the DEFAULT ('new'). Every value is read/write from Unity
        # via PATCH /api/cracks/<id>.
        ("inspection_status", "TEXT DEFAULT 'new'"),
        ("notes",   "TEXT"),
        ("repaired_at", "TEXT"),
        # Spatial mapping: which wall and where on it.
        ("wall_id",   "TEXT"),
        ("floor",     "TEXT"),
        ("local_x",   "REAL"),
        ("local_y",   "REAL"),
    ]

    for col, ctype in _legacy_location_columns + _active_columns + _placement_columns:
        try:
            conn.execute(f"ALTER TABLE cracks ADD COLUMN {col} {ctype}")
        except Exception:
            pass  # column already exists

    # Backfill status for rows that predate the column. Only touches NULLs, so re-running
    # this on every start is harmless.
    try:
        conn.execute("UPDATE cracks SET status = 'pending' WHERE status IS NULL")
    except Exception:
        pass

    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            original_image  TEXT,
            annotated_image TEXT,
            level           INTEGER,
            area            REAL,
            width           REAL,
            is_correct      INTEGER,
            timestamp       TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.after_request
def _add_cors_headers(response):
    """Permit the Unity VR app (and any browser) to read the API.

    The VR build talks to this server from a different device, and a browser-based
    debug client may sit on another origin too, so the API is opened up read-only.
    """
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type')
    # PATCH is needed by /api/cracks/<id> for inspection feedback.
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, PATCH, OPTIONS')
    return response

# ─────────────────────────────────────────────
# CRACK ANALYSIS CORE
# ─────────────────────────────────────────────
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
MAX_VIDEO_SAMPLE_FRAMES = 32  # denser sampling so brief crack frames aren't skipped; keeps uploads fast

def analyze_frame(frame, tiled=True):
    """Core detection + visualization for a single frame.

    tiled=True (images): 640px tiles w/ 20% overlap at conf 0.15 — thin, split
    hairline crack parts are caught; every mask is unioned into ONE binary mask
    so a crack is measured as a whole (full skeleton incl. branches).
    tiled=False (video): single high-res pass for speed.
    """
    original = frame.copy()
    H, W = frame.shape[:2]

    # ── AI DETECTION (tiled for large stills, single pass for video) ──
    ai_boxes, ai_confs, combined = [], [], []
    if AI_READY or model is not None or MODEL_FALLBACK is not None:
        try:
            ai_boxes, ai_confs, combined = _detect_static(frame, tiled=tiled)
        except Exception as e:
            print(f"Inference error: {e}")
        if combined is None:
            combined = np.zeros((H, W), np.uint8)

    # ── CRACK MEASUREMENT — union mask → skeleton length incl. branches ──
    # The scene profile is measured once here: it labels the wall type shown on
    # the result and lets the measurement drop mortar joints instead of cracks.
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masonry = _masonry_profile(gray_full)
    res = measure_combined_mask(combined, frame=frame, masonry=masonry)
    total_length = sum(r[2] for r in res)
    total_width = sum(r[3] for r in res)
    crack_count = len(res)
    avg_width = total_width / crack_count if crack_count > 0 else 0

    # ── SEVERITY LEVEL (based on crack length) ──
    if total_length < 30:
        level = 0; status = "SAFE"
    elif total_length < 100:
        level = 1; status = "Minor — Small cracks detected"
    elif total_length < 300:
        level = 2; status = "Medium — Inspection advised"
    else:
        level = 3; status = "SEVERE — Immediate action required"

    # ── VISUALIZATION — the actual crack pixels (skeleton), never a box ──
    overlay = original.copy()
    sev_color = SEVERITY_COLORS.get(level, (255, 0, 200))
    _draw_results(overlay, res, sev_color)
    if masonry.get('detected'):
        _draw_wall_tag(overlay, masonry)

# ── LOCATION ────────────────────────────────────────────────────
    # Location is NOT computed here any more. Detection returns geometry only; the crack's
    # position is assigned later in Unity by raycasting onto the wall, which is what
    # POST /api/cracks/<id>/position records.

    return overlay, level, round(total_length), round(avg_width, 1), status, crack_count




def analyze_image(img_path):
    """Analyse an uploaded still, keeping any alpha channel.

    IMREAD_UNCHANGED is used instead of IMREAD_COLOR so a PNG with transparency is not
    flattened onto a background colour. Detection runs on the BGR planes only; the alpha
    plane is then copied back onto the annotated result, and the caller writes it as a
    .png, so transparent regions stay transparent on the web page and in Unity.

    Fully transparent pixels carry no wall at all, and OpenCV leaves them black - which
    made a transparent crack sticker (black crack on nothing) undetectable. For detection
    only, those pixels are composited onto a mid-grey "wall" so the crack still reads as
    a dark valley; the saved overlay keeps the original alpha.
    """
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, 0, 0, 0, None, 0

    alpha = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        img = raw[:, :, :3]
    elif raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        img = raw

    det = img
    if alpha is not None and alpha.min() < 250:
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        det = (img.astype(np.float32) * a + 128.0 * (1.0 - a)).astype(np.uint8)

    overlay, level, length, width, status, cracks = analyze_frame(det)

    if alpha is not None and overlay is not None and overlay.shape[:2] == alpha.shape[:2]:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        overlay[:, :, 3] = alpha

    return overlay, level, length, width, status, cracks


def analyze_video(video_path):
    """Sample frames across the video, run detection on each, keep the worst (most severe) result."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, 0, 0, None

    sample_count = min(MAX_VIDEO_SAMPLE_FRAMES, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    best_overlay = None
    best_level   = -1
    best_area    = 0
    best_width   = 0
    best_status  = None
    best_cracks  = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue

        overlay, level, total_area, avg_width, status, crack_count = analyze_frame(frame, tiled=False)

        # Keep the frame with the highest severity level (tie-break: bigger area)
        if level > best_level or (level == best_level and total_area > best_area):
            best_overlay = overlay
            best_level   = level
            best_area    = total_area
            best_width   = avg_width
            best_status  = status
            best_cracks  = crack_count

    cap.release()

    if best_overlay is None:
        return None, 0, 0, 0, None, 0

    return best_overlay, best_level, best_area, best_width, best_status, best_cracks


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        os.makedirs('uploads', exist_ok=True)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Accept base64 (live capture) or file upload
        if 'image_data' in request.form:
            img_data  = request.form['image_data']
            img_bytes = base64.b64decode(img_data.split(',')[1])
            img_path  = f'uploads/{filename}'
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
        else:
            file     = request.files['image']
            filename = file.filename or filename
            img_path = f'uploads/{filename}'
            file.save(img_path)

        # ── Detect whether upload is an image or a video, analyze accordingly ──
        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTS:
            result_img, level, total_area, avg_width, status, crack_count = analyze_video(img_path)
        else:
            result_img, level, total_area, avg_width, status, crack_count = analyze_image(img_path)

        if result_img is None:
            return "Error reading image/video", 400

        # Keep transparency: a PNG (or WebP) upload keeps its alpha channel, so the
        # annotated picture is written as .png instead of a flattened .jpg. Everything
        # without alpha still gets the smaller JPEG.
        has_alpha = result_img.ndim == 3 and result_img.shape[2] == 4
        out_ext = '.png' if has_alpha else '.jpg'
        out_name = (f"annotated_{os.path.splitext(os.path.basename(filename))[0]}"
                    f"{out_ext}")
        out_path = f'uploads/{out_name}'
        cv2.imwrite(out_path, result_img)

# Save to DB
        try:
            conn = get_db()
            # status defaults to 'pending' - Unity assigns the position by raycasting.
            cur = conn.execute(
                "INSERT INTO cracks (image_path, crack_level, crack_area, crack_width, timestamp,"
                "                      status, confidence)"
                " VALUES (?,?,?,?,?,'pending',?)",
                (out_path, level, total_area, round(avg_width, 1),
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'), None)
            )
            conn.commit()
            record_id = cur.lastrowid
            conn.close()
            emit_saved_crack(record_id, level, total_area, out_path)
        except Exception as e:
            print(f"DB write error: {e}")

        return render_template('result.html',
                               level=level,
                               area=total_area,
                               width=avg_width,
                               status=status,
                               cracks=crack_count,
                               img=f'/uploads/{out_name}',
                               orig_img=f'/uploads/{filename}')

    # GET — show homepage with recent history
    history = []
    fb_correct = 0
    fb_total = 0
    try:
        conn    = get_db()
        records = conn.execute(
            "SELECT image_path, crack_level, crack_area, timestamp FROM cracks ORDER BY timestamp DESC LIMIT 6"
        ).fetchall()
        fb_correct = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1").fetchone()[0]
        fb_total   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        for r in records:
            history.append({
                'image_url': f'/uploads/{os.path.basename(r["image_path"])}',
                'level':     r['crack_level'],
                'area':      r['crack_area'],
                'timestamp': r['timestamp'][-8:-3] if r['timestamp'] else ''
            })
    except Exception as e:
        print(f"DB read error: {e}")

    return render_template('index.html', history=history, fb_correct=fb_correct, fb_total=fb_total)


@app.route('/live')
def live():
    return render_template('live.html', ai_ready=AI_READY, model_name=MODEL_NAME)


# ─────────────────────────────────────────────
# SAVE CAPTURE — from live.html "Capture" button
# Saves the captured frame into /uploads and inserts
# it into the same `cracks` table used by index/history.
# ─────────────────────────────────────────────
@app.route('/save_capture', methods=['POST'])
def save_capture():
    """Save one Capture as one or two history rows: one RGB and one thermal.

    Both cameras see the same crack at the same moment, so the two rows share a
    capture_group. The VR app treats a group as a SINGLE crack and offers a button to
    swap the picture between the RGB and thermal feed - which is only possible because
    they are linked rather than being two unrelated records.

    Each picture is re-analysed server-side, so the stored image is the annotated one
    with the crack line drawn on it. Rows start as status='pending'; Unity raycasts a
    world position and POSTs it to /api/cracks/<id>/position.
    """
    from flask import jsonify

    if 'image' not in request.files:
        return jsonify({'error': 'No image received'}), 400

    group = f"grp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # (form field, camera_type, label)
    sources = [('image', 'rgb', 'rgb')]
    if 'thermal' in request.files and request.files['thermal'].filename:
        sources.append(('thermal', 'thermal', 'thermal'))

    saved = []
    for field, cam_type, label in sources:
        item = request.files[field]
        if not item or not item.filename:
            continue

        os.makedirs('uploads', exist_ok=True)
        base = f"capture_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{base}.jpg"
        img_path = f'uploads/{filename}'
        tmp_path = os.path.join('uploads', f"_tmp_{filename}")
        item.save(tmp_path)

        level, total_area, avg_width = 0, 0, 0.0
        crack_count = 0
        status = 'SAFE'
        confidence = None

        try:
            img = cv2.imread(tmp_path)
            if img is None:
                raise ValueError('could not decode uploaded image')

            overlay, level, total_area, avg_width, status, crack_count = analyze_frame(img)
            if overlay is not None:
                cv2.imwrite(img_path, overlay)     # annotated frame = what history shows
            else:
                item.save(img_path)
        except Exception as e:
            print(f"save_capture({cam_type}) analyse error: {e}")
            try:
                if os.path.exists(tmp_path):
                    item.save(img_path)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        try:
            conn = get_db()
            cur = conn.execute(
                "INSERT INTO cracks (image_path, crack_level, crack_area, crack_width, timestamp,"
                "                      status, camera_type, capture_group, confidence)"
                " VALUES (?,?,?,?,?,'pending',?,?,?)",
                (img_path, level, total_area, round(float(avg_width), 1),
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 cam_type, group, confidence))
            conn.commit()
            record_id = cur.lastrowid
            conn.close()
            emit_saved_crack(record_id, level, total_area, img_path, cam_type)
            saved.append({'id': record_id, 'camera_type': cam_type,
                          'image_path': img_path, 'cracks': crack_count,
                          'level': level, 'status': status})
        except Exception as e:
            print(f"DB write error (save_capture {cam_type}): {e}")
            return jsonify({'error': str(e)}), 500

    if not saved:
        return jsonify({'error': 'Nothing was saved'}), 400

    return jsonify({
        'success': True,
        'capture_group': group,
        'saved': saved,
        'id': saved[0]['id'],
        'image_path': saved[0]['image_path'],
        'level': saved[0]['level'],
        'cracks': saved[0]['cracks'],
        'status': saved[0]['status'],
        'placement': 'pending',
    })

# ─────────────────────────────────────────────
@app.route('/vr_scan', methods=['POST'])
def vr_scan():
    from flask import jsonify
    import re
    try:
        data = request.get_json(force=True) or {}
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image received'}), 400
        level  = int(float(data.get('level', 0)))
        length = float(data.get('crack_length', 0) or 0)
        width  = round(float(data.get('crack_width', 0) or 0), 1)

        img_bytes = base64.b64decode(re.sub(r'^data:image/.+;base64,', '', image_data))
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        os.makedirs('uploads', exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"vr_scan_{timestamp_str}.jpg"
        img_path = f'uploads/{filename}'
        cv2.imwrite(img_path, frame)

        try:
            conn = get_db()
            cur = conn.execute(
                "INSERT INTO cracks (image_path, crack_level, crack_area, crack_width, timestamp) VALUES (?,?,?,?,?)",
                (img_path, level, int(round(length)), width, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
            record_id = cur.lastrowid
            conn.close()
            emit_saved_crack(record_id, level, int(round(length)), img_path)
        except Exception as e:
            print(f"DB write error (vr_scan): {e}")
            return jsonify({'error': str(e)}), 500

        return jsonify({'success': True, 'filename': filename, 'image_path': img_path})
    except Exception as e:
        print(f"vr_scan error: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# HIMICRO IP CAMERA (172.20.10.64) — RTSP relay for Unity + kiosk
# Optical (RGB) = channel 101 (2688x1520@30), Thermal = channel 201 (1280x960@25).
# Smooth MJPEG uses substream 102 for RGB (resized to 1920x1080@30).
# Unity: GET /vr_frame, /vr_rgb_frame, /vr_thermal_frame, /vr_rgb_mjpeg,
#        /vr_thermal_mjpeg, POST /vr_scan_image. Kiosk: MJPEG + /live_contours.
# NOTE: password below is for the local LAN camera only.
# ─────────────────────────────────────────────
HIMICRO_IP = '172.20.10.64'
HIMICRO_USER = 'admin'
HIMICRO_PASS = 'abcd1234'
HIMICRO_RGB_URL = f'rtsp://{HIMICRO_USER}:{HIMICRO_PASS}@{HIMICRO_IP}:554/Streaming/Channels/101'
HIMICRO_THERMAL_URL = f'rtsp://{HIMICRO_USER}:{HIMICRO_PASS}@{HIMICRO_IP}:554/Streaming/Channels/201'
HIMICRO_RGB_SUB_URL = f'rtsp://{HIMICRO_USER}:{HIMICRO_PASS}@{HIMICRO_IP}:554/Streaming/Channels/102'

# ── CAMERA SOURCE ────────────────────────────────────────────────
# The live feed is NOT hardwired to the HiMicro any more. It auto-detects whatever
# camera is actually connected (laptop webcam, USB camera, or the RTSP rig camera):
#
#   1. CAMERA_SOURCE / CRACK_CAMERA, when set:
#        "0".."9"                        a local device index
#        "himicro"                       the RTSP rig camera defined above
#        "auto"                          force detection
#        rtsp://…  http://…  file path   that exact source
#   2. AUTO-DETECT: the first local device (0..4) that returns a real frame.
#   3. The HiMicro RTSP stream, if no local camera answered.
#
# THERMAL_SOURCE does the same for the thermal channel ("none" disables it). A plain
# webcam has no thermal sensor, so the thermal routes answer 503 instead of showing a
# black frame, and /live_stats reports thermal_available: false.

# Thermal readout: the RTSP thermal stream (channel 201) is a rendered pseudo-color
# palette, NOT radiometric °C. Pixel intensity is mapped linearly across this range —
# adjust to match the camera's on-screen temperature scale for a meaningful number.
HIMICRO_THERMAL_TMIN = -20.0
HIMICRO_THERMAL_TMAX = 120.0
# Seconds between AI crack-detection passes on the RGB stream. Lower = more live.
# Inference on a 1280-wide frame typically takes longer than this, so the effective
# rate is bounded by GPU speed, not by this value.
HIMICRO_ANALYZE_INTERVAL = 0.1

_himicro = {'rgb': None, 'rgb_annotated': None,
            'contours': None, 'ai': None,
            'analyzed_at': 0.0, 'analyze_ms': 0,
            'lock': None, 'started': False}
import threading as _th
_himicro['lock'] = _th.Lock()

# ── live source resolution (auto-detect) ────────────────────────
# Resolved once per process and then cached: probing opens the device, and opening a
# camera that the live capture already owns would stall the stream, so re-probing on a
# timer is deliberately not done. Set CAMERA_SOURCE to pin a specific source instead.
_LIVE_SOURCE = {'value': None, 'lock': _th.Lock()}
_HIMICRO_LABELS = ('himicro', 'ip', 'rtsp', 'rig')


def _camera_env(*names):
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _as_source(value):
    """'2' -> 2 (a local device index); anything else stays a string."""
    return int(value) if value.isdigit() else value


def _probe_index(index):
    """True when local camera `index` opens and hands back a real frame."""
    backends = ((cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF)
                if os.name == 'nt' else (cv2.CAP_ANY,))
    for backend in backends:
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue
            # A webcam needs a moment after open() before it produces frames.
            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return True
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return False


def _detect_local_camera():
    """Index of the first working local camera, or None when there is none."""
    for index in range(5):
        if _probe_index(index):
            print(f"[camera] auto-detected local camera index {index}")
            return index
    return None


def _local_or_himicro():
    """Detected local camera, else the HiMicro RTSP stream.

    Note: index 0 is falsy, so this must never be written as
    `_detect_local_camera() or HIMICRO_RGB_URL` - that would always pick the RTSP
    stream on a machine whose only camera is device 0.
    """
    index = _detect_local_camera()
    return HIMICRO_RGB_URL if index is None else index


def resolve_live_source(force=False):
    """Source the LIVE feed should open: a device index (int) or a URL (str)."""
    with _LIVE_SOURCE['lock']:
        if not force and _LIVE_SOURCE['value'] is not None:
            return _LIVE_SOURCE['value']

    raw = _camera_env('CAMERA_SOURCE', 'CRACK_CAMERA')
    low = raw.lower()

    if raw and low in _HIMICRO_LABELS:
        value = HIMICRO_RGB_URL
    elif raw and low in ('auto', 'detect', 'webcam', 'usb', 'default'):
        value = _local_or_himicro()
    elif raw:
        value = _as_source(raw)
    else:
        value = _local_or_himicro()

    with _LIVE_SOURCE['lock']:
        if _LIVE_SOURCE['value'] != value:
            print(f"[camera] live source -> {value}")
        _LIVE_SOURCE['value'] = value
    return value


def camera_source_name(source=None):
    """Short label for /live_stats: 'himicro', 'local:0', or the raw URL."""
    src = resolve_live_source() if source is None else source
    if src == HIMICRO_RGB_URL:
        return 'himicro'
    if isinstance(src, int):
        return f'local:{src}'
    return str(src)


def resolve_thermal_source():
    """Thermal source, or None when the active camera has no thermal channel."""
    raw = _camera_env('THERMAL_SOURCE')
    if raw:
        low = raw.lower()
        if low in ('none', 'off', 'no', 'false'):
            return None
        if low in _HIMICRO_LABELS:
            return HIMICRO_THERMAL_URL
        return _as_source(raw)
    # Auto: the HiMicro thermal channel only makes sense when the live feed is the
    # HiMicro itself - a laptop webcam has no thermal sensor at all.
    return HIMICRO_THERMAL_URL if camera_source_name() == 'himicro' else None

class _FreshCapture:
    """Background reader that always keeps ONLY the newest frame.

    `source` is anything cv2.VideoCapture accepts: a local device index (int) for a
    USB / laptop camera, or a URL for the RTSP rig camera.

    cv2.VideoCapture.read() hands back the OLDEST queued frame. When the consumer
    is slower than the camera (AI inference, JPEG encode), the queue fills up and
    every read() returns a frame that is seconds old -- so detections lag behind
    the live video and look "stuck in one place". This reader drains the stream
    continuously, so read() is always current and each frame is returned once.
    """
    def __init__(self, source):
        self.source = source
        self._lock = _th.Lock()
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._stop = False
        _th.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = None
        while not self._stop:
            try:
                if cap is None:
                    cap = cv2.VideoCapture(self.source)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                else:
                    time.sleep(0.2)
                    try: cap.release()
                    except Exception: pass
                    cap = None          # reconnect
            except Exception:
                time.sleep(0.5)
                try:
                    if cap is not None: cap.release()
                except Exception:
                    pass
                cap = None
        try:
            if cap is not None: cap.release()
        except Exception:
            pass

    def read(self, timeout=1.0):
        """Return (True, newest_frame) once per new frame, else (False, None)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    return True, self._frame
            time.sleep(0.005)
        return False, None

    def release(self):
        self._stop = True

def _himicro_worker():
    # RGB stream only — one session per stream (thermal MJPEG owns 201).
    # Video relay and AI inference are decoupled: the JPEG for the stream is published
    # under the lock, then inference runs OUTSIDE the lock so /live_contours and
    # /vr_frame are never blocked by a slow GPU pass.
    rgb_cap = None
    last_analyze = 0.0
    while True:
        try:
            if rgb_cap is None:
                rgb_cap = _FreshCapture(resolve_live_source())
            ok1, rgb = rgb_cap.read(timeout=2.0)   # always the newest frame
            now = time.time()
            if ok1 and rgb is not None:
                # publish the raw frame immediately — video must never wait for AI
                _, b1 = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with _himicro['lock']:
                    _himicro['rgb'] = b1.tobytes()

                if now - last_analyze >= HIMICRO_ANALYZE_INTERVAL:
                    last_analyze = now
                    t0 = time.time()
                    try:
                        h0, w0 = rgb.shape[:2]
                        sc = 1280.0 / max(1, w0)
                        small_w, small_h = 1280, max(1, int(h0 * sc))
                        small = cv2.resize(rgb, (small_w, small_h))

                        res = run_live_analysis(small)

                        # Persist live detections so Unity can queue them and raycast a
                        # position for each one.
                        if PERSIST_LIVE and res.get('level', 0) > 0:
                            for c in res.get('contours', []):
                                try:
                                    maybe_emit_new_crack(
                                        small,
                                        np.asarray(c.get('points', []), dtype=np.float32),
                                        c.get('length', 0),
                                        res.get('level', 0),
                                        width_px=c.get('width', 0),
                                        conf=c.get('conf', 0))
                                except Exception as e:
                                    print(f"[aruco] worker persist error: {e}")

                        # contours scaled to 320x240 reference space, points capped (~150)
                        # so the browser redraws them at 60fps without freezing
                        fx, fy = 320.0 / 640, 240.0 / small_h
                        cc = []
                        for c in res.get('contours', []):
                            raw = c.get('points', [])
                            step = max(1, len(raw) // 150)
                            pts = [[int(x * fx), int(y * fy)] for [x, y] in raw[::step]]
                            if len(pts) >= 2:
                                # pts_n = same polyline normalized to 0..1 of the frame, so the
                                # browser overlay maps to the letterboxed display rect exactly
                                # regardless of stream resolution / aspect.
                                pn = [[round(x / float(small_w), 4), round(y / float(small_h), 4)]
                                      for [x, y] in raw[::step]]
                                cc.append({'points': pts, 'pts_n': pn,
                                           'conf': c.get('conf', 0),
                                           'length': c.get('length', 0), 'width': c.get('width', 0)})

                        ab = res.get('annotated_image', '')
                        annotated = base64.b64decode(ab.split(',', 1)[1]) if ',' in ab else None
                        elapsed_ms = int((time.time() - t0) * 1000)

                        # single short critical section for all AI-derived state
                        with _himicro['lock']:
                            if annotated:
                                _himicro['rgb_annotated'] = annotated
                            _himicro['contours'] = cc
                            _himicro['ai'] = {'level': res.get('level', 0),
                                              'length': res.get('crack_length', 0),
                                              'width': res.get('crack_width', 0),
                                              'boxes': res.get('ai_boxes', 0),
                                              'analyze_ms': elapsed_ms}
                            _himicro['analyzed_at'] = time.time()
                            _himicro['analyze_ms'] = elapsed_ms
                        try:
                            with stats_lock:
                                live_stats.update({'level': res.get('level', 0),
                                                   'area': res.get('crack_length', 0),
                                                   'status': res.get('status', 'SAFE'),
                                                   'ai_boxes': res.get('ai_boxes', 0),
                                                   'coverage': str(res.get('coverage', '0'))})
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"himicro analyze error: {e}")
        except Exception as e:
            print(f"himicro worker error: {e}")
            time.sleep(2)
        time.sleep(0.02)

def ensure_himicro():
    if not _himicro['started']:
        _himicro['started'] = True
        _th.Thread(target=_himicro_worker, daemon=True).start()

def _himicro_wait(seconds=10):
    ensure_himicro()
    end = time.time() + seconds
    while time.time() < end:
        with _himicro['lock']:
            if _himicro['rgb']:
                return True
        time.sleep(0.2)
    return False

@app.route('/vr_frame', methods=['GET'])
def vr_frame():
    from flask import jsonify
    import base64
    if not _himicro_wait():
        return jsonify({'error': 'camera offline: no local camera answered and the ' +
                                 'HiMicro RTSP stream is unreachable'}), 503
    with _himicro['lock']:
        raw = _himicro['rgb']
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    return jsonify({'image': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode(),
                    'width': w, 'height': h})

@app.route('/vr_rgb_frame', methods=['GET'])
def vr_rgb_frame():
    from flask import Response
    ensure_himicro()
    with _himicro['lock']:
        img = _himicro['rgb_annotated'] or _himicro['rgb']
    if not img:
        return Response(b'', status=503, mimetype='image/jpeg')
    return Response(img, mimetype='image/jpeg')

@app.route('/vr_thermal_frame', methods=['GET'])
def vr_thermal_frame():
    from flask import Response
    # On-demand snapshot (no persistent session - the thermal MJPEG endpoint owns that stream)
    src = resolve_thermal_source()
    if src is None:
        return Response(b'', status=503, mimetype='image/jpeg')
    try:
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        try:
            cap.release()
        except Exception:
            pass
        if not ok or frame is None:
            return Response(b'', status=503, mimetype='image/jpeg')
        _, buf = cv2.imencode('.jpg', _center_crosshair(frame), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype='image/jpeg')
    except Exception:
        return Response(b'', status=503, mimetype='image/jpeg')

def _center_crosshair(frame, arm=22, show_temp=True):
    """Draw a short centered '+' marker + temperature readout on a frame (returns a copy).

    arm = half-length in pixels of each arm. show_temp maps the center pixel's
    intensity to °C via HIMICRO_THERMAL_TMIN..TMAX (palette estimate — see constants).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - arm, cy), (cx + arm, cy), (0, 0, 0), 3)
    cv2.line(out, (cx, cy - arm), (cx, cy + arm), (0, 0, 0), 3)
    cv2.line(out, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1)
    cv2.line(out, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1)

    if show_temp:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        g = float(gray[cy, cx])
        temp = HIMICRO_THERMAL_TMIN + (HIMICRO_THERMAL_TMAX - HIMICRO_THERMAL_TMIN) * (g / 255.0)
        label = f"{temp:5.1f} C"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
        lx = cx - tw // 2
        ly = cy + arm + th + 10
        ly = min(ly, h - 8)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 8), (0, 0, 0), -1)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 8), (255, 255, 255), 1)
        cv2.putText(out, label, (lx, ly), font, 0.6, (120, 255, 120), 1, cv2.LINE_AA)
    return out


def _thermal_center_temp(frame):
    """Estimate °C at the frame center from palette intensity (see HIMICRO_THERMAL_* constants)."""
    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = float(gray[h // 2, w // 2])
        return round(HIMICRO_THERMAL_TMIN + (HIMICRO_THERMAL_TMAX - HIMICRO_THERMAL_TMIN) * (g / 255.0), 1)
    except Exception:
        return None


def _thermal_minmax_temp(frame):
    """Estimate min/max °C across the whole frame from palette intensity."""
    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lo, hi = int(gray.min()), int(gray.max())
        span = (HIMICRO_THERMAL_TMAX - HIMICRO_THERMAL_TMIN) / 255.0
        return (round(HIMICRO_THERMAL_TMIN + lo * span, 1),
                round(HIMICRO_THERMAL_TMIN + hi * span, 1))
    except Exception:
        return None, None


@app.route('/vr_thermal_temp')
def vr_thermal_temp():
    from flask import jsonify
    src = resolve_thermal_source()
    if src is None:
        return jsonify({'error': 'thermal not available on this camera'}), 503
    try:
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        try:
            cap.release()
        except Exception:
            pass
        if not ok or frame is None:
            return jsonify({'error': 'thermal offline'}), 503
        lo, hi = _thermal_minmax_temp(frame)
        return jsonify({'temperature_c': _thermal_center_temp(frame),
                        'min_c': lo, 'max_c': hi})
    except Exception:
        return jsonify({'error': 'thermal offline'}), 503


def _mjpeg_gen(source, out_size=None, crop_720=False, crosshair=False):
    """Raw frame relay (no AI) - smooth video, limited only by camera fps."""
    cap = _FreshCapture(source)
    try:
        while True:
            ok, frame = cap.read(timeout=2.0)
            if not ok or frame is None:
                continue
            if crop_720:
                # 1280x960 -> center-crop 1280x720 (keeps 16:9, no distortion)
                h, w = frame.shape[:2]
                if h >= 720:
                    y0 = (h - 720) // 2
                    frame = frame[y0:y0 + 720, 0:w]
            if out_size is not None:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)
            if crosshair:
                frame = _center_crosshair(frame)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    finally:
        try:
            cap.release()
        except Exception:
            pass

@app.route('/vr_rgb_mjpeg')
def vr_rgb_mjpeg():
    from flask import Response
    # Live camera. The HiMicro main stream is downscaled to 1920x1080; a local webcam is
    # relayed at its native size (upscaling it would only cost CPU).
    src = resolve_live_source()
    size = (1920, 1080) if src == HIMICRO_RGB_URL else None
    return Response(_mjpeg_gen(src, out_size=size),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vr_thermal_mjpeg')
def vr_thermal_mjpeg():
    from flask import Response
    # Thermal 1280x720 @ 25fps - center-cropped, raw (never AI-analyzed).
    # Short centered '+' marker so the camera spot-mode aim is visible.
    src = resolve_thermal_source()
    if src is None:
        return Response(b'', status=503, mimetype='image/jpeg')
    return Response(_mjpeg_gen(src, crop_720=True, crosshair=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_contours')
def live_contours():
    from flask import jsonify
    ensure_himicro()
    with _himicro['lock']:
        cc = _himicro.get('contours') or []
        ai = _himicro.get('ai') or {}
        analyzed_at = _himicro.get('analyzed_at') or 0.0
        analyze_ms = _himicro.get('analyze_ms') or 0
    # age_ms lets the browser show how fresh the geometry is, so a stalled
    # detection pass is visible instead of looking like "no cracks found".
    age_ms = int((time.time() - analyzed_at) * 1000) if analyzed_at else -1
    return jsonify({'level': ai.get('level', 0), 'length': ai.get('length', 0),
                    'width': ai.get('width', 0), 'boxes': ai.get('boxes', 0),
                    'contours': cc,
                    'analyze_ms': ai.get('analyze_ms', analyze_ms),
                    'age_ms': age_ms,
                    'interval': HIMICRO_ANALYZE_INTERVAL})

@app.route('/vr_scan_image', methods=['POST'])
def vr_scan_image():
    from flask import jsonify
    import base64
    if not _himicro_wait():
        return jsonify({'error': 'camera offline: no local camera answered and the ' +
                                 'HiMicro RTSP stream is unreachable'}), 503
    with _himicro['lock']:
        raw = _himicro['rgb']
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid frame'}), 500
    try:
        res = run_live_analysis(frame)
    except Exception as e:
        print(f"vr_scan_image error: {e}")
        return jsonify({'error': str(e)}), 500
    h, w = frame.shape[:2]
    res['raw_image'] = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
    res['image_width'] = w
    res['image_height'] = h
    return jsonify(res)


# ─────────────────────────────────────────────
# OPTION B — SERVER-SIDE AI VIDEO STREAM
# ─────────────────────────────────────────────
import threading
import time
from flask import Response, jsonify

# Shared stats updated by the frame generator
live_stats = {
    'level': 0, 'area': 0, 'status': 'SAFE',
    'ai_boxes': 0, 'coverage': '0.00', 'fps': 0,
}
stats_lock    = threading.Lock()
camera_instance = None
camera_lock   = threading.Lock()

def get_camera():
    global camera_instance
    with camera_lock:
            if camera_instance is None or not camera_instance.isOpened():
                camera_instance = cv2.VideoCapture(resolve_live_source())
            return camera_instance

def release_camera():
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            camera_instance.release()
            camera_instance = None

# ── Live crack → VR push (Socket.IO 'new_crack') ─────────
# Every crack the live loop finds is pushed to the Unity twin exactly
# ONCE. Its id is a deterministic hash of its quantised screen
# signature, so the same physical crack re-observed across frames (or
# after a server restart) keeps the same id and the Unity marker list
# stays free of duplicates.
_seen_crack_sigs = set()
_MAX_SEEN_SIGS   = 256

def _crack_signature(line):
    pts = np.asarray(line).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    y0, x0 = pts.min(0)
    y1, x1 = pts.max(0)
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    return (int(cy) // 48, int(cx) // 48, int((y1 - y0) // 40), int((x1 - x0) // 40))

def _signature_id(sig):
    h = 2166136261  # deterministic FNV-1a fold → stable marker id
    for v in sig:
        h = ((h ^ (abs(v) & 0x7fffffff)) * 16777619) & 0x7fffffff
    return h

def _save_live_crack_image(frame, cid):
    os.makedirs('uploads', exist_ok=True)
    fname = f"live_crack_{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    try:
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return ''
        with open(os.path.join('uploads', fname), 'wb') as fh:
            fh.write(buf.tobytes())
        return fname
    except Exception as e:
        print(f"save live crack image error: {e}")
        return ''

# Live detection does NOT write to the database. The operator presses Capture, and only that
# saves a picture into history - otherwise panning the camera would flood /api/cracks with a
# row per detected crack. Set PERSIST_LIVE=1 to restore the old auto-save behaviour.
PERSIST_LIVE = os.environ.get('PERSIST_LIVE', '0') not in ('0', 'false', 'False')

_live_persist_lock = _th.Lock()


def _persist_live_crack(cid, level, length_px, width_px, conf, img_path):
    """Insert one live detection as 'pending'. Returns the new row id, or None."""
    try:
        with _live_persist_lock:
            conn = get_db()
            cur = conn.execute(
                "INSERT INTO cracks (image_path, crack_level, crack_area, crack_width, timestamp,"
                "                      status, confidence)"
                " VALUES (?,?,?,?,?,'pending',?)",
                (f'uploads/{img_path}' if img_path else None,
                 level, int(round(length_px or 0)), round(float(width_px or 0), 1),
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 round(float(conf), 3) if conf is not None else None))
            conn.commit()
            rid = cur.lastrowid
            conn.close()
            return rid
    except Exception as e:
        print(f"live persist error: {e}")
        return None


def maybe_emit_new_crack(frame, line, length_px, level, width_px=None, conf=None):
    """Push a newly-seen live crack to the VR app and persist it as 'pending'.

    No location is computed here any more - Unity raycasts the crack onto the wall and
    POSTs the result back to /api/cracks/<id>/position.
    """
    sig = _crack_signature(line)
    if sig is None or sig in _seen_crack_sigs:
        return
    if len(_seen_crack_sigs) >= _MAX_SEEN_SIGS:
        _seen_crack_sigs.clear()
    _seen_crack_sigs.add(sig)

    cid   = _signature_id(sig)
    fname = _save_live_crack_image(frame, cid)
    base = _image_base()

    # Persist so Unity can queue it and raycast a position for it.
    row_id = None
    if PERSIST_LIVE:
        row_id = _persist_live_crack(cid, level, length_px, width_px, conf, fname)

    payload = {
        'id': cid,
        'row_id': row_id,          # DB row the operator positions via /api/cracks
        'position': crack_world_position(cid),
        'length_cm': round(max(0.0, float(length_px)) * CM_PER_PX, 2),
        'severity': crack_severity(level),
        'image_url': f"{base}/{fname}" if fname else '',
        'image_base64': _image_base64(os.path.join('uploads', fname)) if fname else '',
        'status': 'pending',
    }
    try:
        broadcast_new_crack(payload)
        print(f"[VR push] new_crack id={cid} severity={payload['severity']} "
              f"len={payload['length_cm']}cm row={row_id} status=pending")
    except Exception as e:
        print(f"new_crack broadcast error: {e}")

def emit_saved_crack(record_id, level, area, image_path, camera_type='rgb'):
    """Push a DB-saved crack (website capture / save_capture / vr_scan) to the
    Unity twin, with the exact /api/cracks payload shape so the client's
    snapshot backfill and live events stay identical.

    The row is 'pending': Unity raycasts a position and POSTs it to
    /api/cracks/<id>/position.
    """
    base = _image_base()
    img = (image_path or '').lstrip('/')
    payload = {
        'id': record_id,
        'position': crack_world_position(record_id),
        'length_cm': round((area or 0) * CM_PER_PX, 2),
        'severity': crack_severity(level),
        'image_url': f"{base}/{img}" if img else '',
        'image_base64': _image_base64(image_path),
        'status': 'pending',
        'camera_type': camera_type,
    }
    try:
        broadcast_new_crack(payload)
        print(f"[VR push] new_crack (capture) id={payload['id']} "
              f"camera={camera_type} severity={payload['severity']} "
              f"len={payload['length_cm']}cm status=pending")
    except Exception as e:
        print(f"new_crack (capture) emit error: {e}")

def analyze_frame_live(frame):
    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 130)
    kernel = np.ones((3, 3), np.uint8)
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    edge_area = int(np.sum(crack_mask > 0) / 4)
    coverage = round((np.sum(crack_mask > 0) / (w_img * h_img)) * 100, 2)

    # AI detection — single 640 pass, conf 0.15 keeps thin crack parts
    combined = np.zeros((h_img, w_img), np.uint8)
    ai_boxes, ai_confs = [], []
    if AI_READY or model is not None:
        try:
            ai_boxes, ai_confs, ai_masks, combined = predict(frame, imgsz=640)
        except Exception as e:
            print(f"Live AI error: {e}")

    # Crack measurements — union mask skeleton (length incl. branches).
    # One masonry profile per frame labels the wall and lets mortar joints be
    # rejected without losing a real crack running across them.
    masonry = _masonry_profile(gray)
    measurements = measure_combined_mask(combined, frame=frame, masonry=masonry)
    crack_lines = []
    total_length = 0.0
    total_width = 0.0
    crack_count = 0
    ai_area = 0
    for skel, contour, length, width, conf in measurements:
        if length > 5:
            crack_lines.append((contour, conf, length, width, True))
            total_length += length
            total_width += width
            crack_count += 1
            ai_area += int(length * width * conf * 0.4)

    avg_width = total_width / crack_count if crack_count > 0 else 0

    # Hybrid score — severity must come from the AI crack geometry. The raw Canny
    # edge count is a WALL-TEXTURE metric: brick, roof tiles, gravel and foliage all
    # produce thousands of edges, so a clean brick wall used to be scored MEDIUM or
    # SEVERE with zero cracks found. Edges now only nudge a score that already has a
    # real (filtered) crack behind it; with no crack the frame is SAFE.
    if crack_count == 0:
        total_area = 0
    else:
        total_area = int(ai_area * 1.55 + min(edge_area, ai_area) * 0.45)
    if total_area < 50:
        level = 0; status = "SAFE"
    elif total_area < 2500:
        level = 1; status = "MINOR"
    elif total_area < 8000:
        level = 2; status = "MEDIUM"
    else:
        level = 3; status = "SEVERE"

    out = original.copy()
    sev_color = SEVERITY_COLORS.get(level, (255, 0, 200))
    _draw_results(out, measurements, sev_color)

    banner = {0: (0, 160, 60), 1: (0, 180, 180), 2: (0, 130, 255), 3: (0, 0, 210)}
    cv2.rectangle(out, (0, 0), (w_img, 48), banner[level], -1)
    cv2.putText(out, f"{status}  |  Length: {round(total_length)}px  Width: {round(avg_width, 1)}px  |  AI: {crack_count} crack(s)",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    tag = f"YOLOv8-seg: {MODEL_NAME}" if model is not None else (
        f"Roboflow: {MODEL_NAME}" if AI_READY else "Edge Only")
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out, (w_img - tw - 12, h_img - th - 14), (w_img, h_img), (0, 0, 0), -1)
    cv2.putText(out, tag, (w_img - tw - 6, h_img - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    if masonry.get('detected'):
        _draw_wall_tag(out, masonry)

    # Push newly-seen cracks to the VR digital twin (idempotent per crack).
    # Only the few longest strokes of the frame are reported: a busy frame used to
    # emit a pair of crumbs for every 30-px splinter, flooding the twin with
    # cracks that are pure noise. Anything shorter than ~1.5% of the frame
    # diagonal is not a crack worth placing in the room either.
    min_emit = max(40.0, 0.015 * float(np.hypot(h_img, w_img)))
    ranked = sorted(crack_lines, key=lambda c: c[2], reverse=True)
    for line, conf, length, width, is_open in ranked[:3]:
        if length < min_emit or _is_rectangular_mask(line):
            continue
        maybe_emit_new_crack(frame, line, length, level,
                             width_px=width, conf=conf)

    return out, level, round(total_length), status, crack_count, coverage


def generate_frames():
    cam        = get_camera()
    prev_time  = time.time()
    frame_cnt  = 0
    while True:
        success, frame = cam.read()
        if not success:
            break
        annotated, level, area, status, num_boxes, coverage = analyze_frame_live(frame)
        frame_cnt += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = round(frame_cnt / (now - prev_time), 1)
            with stats_lock:
                live_stats.update({'level':level,'area':area,'status':status,
                                   'ai_boxes':num_boxes,'coverage':str(coverage),'fps':fps})
            frame_cnt = 0; prev_time = now
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_stats')
def live_stats_endpoint():
    from flask import jsonify
    with stats_lock:
        out = dict(live_stats)
    # Report the camera the live feed actually resolved to, so the UI can show it and
    # hide the thermal controls when there is no thermal sensor.
    src = resolve_live_source()
    out['camera_source'] = camera_source_name(src)
    out['thermal_available'] = resolve_thermal_source() is not None
    return jsonify(out)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    from flask import jsonify
    release_camera()
    return jsonify({'status': 'stopped'})



# ─────────────────────────────────────────────
# ANALYZE LIVE FRAME (Browser → Server → AI → Browser)
# ─────────────────────────────────────────────
def run_live_analysis(frame):
    """Shared AI pipeline: BGR ndarray -> response dict (same fields as /analyze_live).
    Used by /analyze_live and /vr_scan_image. Crack logic below is unchanged."""
    import base64
    original = frame.copy()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]

    # Edge detection — CLAHE + adaptive thresholds + better morphology
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced   = clahe.apply(gray)
    blur       = cv2.GaussianBlur(enhanced, (5,5), 0)
    med        = np.median(blur)
    lower      = int(max(10, 0.4 * med))
    upper      = int(min(200, 1.2 * med))
    edges      = cv2.Canny(blur, lower, upper)
    kernel     = np.ones((5,5), np.uint8)
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    edge_area  = int(np.sum(crack_mask > 0) / 4)
    coverage   = round((np.sum(crack_mask > 0) / (w_img * h_img)) * 100, 2)

    # AI detection — single 640 pass, conf 0.15 keeps thin crack parts
    ai_area = 0
    combined = np.zeros((h_img, w_img), np.uint8)
    ai_boxes, ai_confs = [], []
    if AI_READY or model is not None:
        try:
            ai_boxes, ai_confs, ai_masks, combined = predict(frame, imgsz=640)
        except Exception as e:
            print(f"Live AI error: {e}")

    # Crack measurements — union mask skeleton (length incl. branches).
    # One masonry profile per frame labels the wall and lets mortar joints be
    # rejected without losing a real crack running across them.
    masonry = _masonry_profile(gray)
    measurements = measure_combined_mask(combined, frame=frame, masonry=masonry)
    crack_lines = []
    total_length = 0.0
    total_width = 0.0
    crack_count = 0

    for skel, contour, length, width, conf in measurements:
        if length > 5:
            crack_lines.append((contour, conf, length, width, True))
            total_length += length
            total_width += width
            crack_count += 1
            ai_area += int(length * width * conf * 0.4)

    avg_width = total_width / crack_count if crack_count > 0 else 0

    # Hybrid score — severity comes from the AI crack geometry only, never from raw
    # Canny edge density: a brick wall, a tiled floor or a leafy tree is full of edges
    # and used to be reported MEDIUM/SEVERE with no crack at all. With no filtered
    # crack the frame is SAFE; when there is one, edges may only nudge the score.
    if crack_count == 0:
        total_area = 0
    else:
        total_area = int(ai_area * 1.55 + min(edge_area, ai_area) * 0.45)
    if total_area < 50:
        level = 0; status = "SAFE"
    elif total_area < 2500:
        level = 1; status = "MINOR"
    elif total_area < 8000:
        level = 2; status = "MEDIUM"
    else:
        level = 3; status = "SEVERE"

    out = original.copy()
    sev_color = SEVERITY_COLORS.get(level, (255, 0, 200))
    _draw_results(out, measurements, sev_color)

    banner = {0: (0, 160, 60), 1: (0, 180, 180), 2: (0, 130, 255), 3: (0, 0, 210)}
    cv2.rectangle(out, (0, 0), (w_img, 44), banner[level], -1)
    cv2.putText(out, f"{status}  |  Length: {round(total_length)}px  Width: {round(avg_width, 1)}px  |  AI: {crack_count} crack(s)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    tag = f"YOLOv8-seg: {MODEL_NAME}" if model is not None else (f"Roboflow: {MODEL_NAME}" if AI_READY else "Edge Only")
    (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out, (w_img-tw-12, h_img-th-14), (w_img, h_img), (0,0,0), -1)
    cv2.putText(out, tag, (w_img-tw-6, h_img-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)
    if masonry.get('detected'):
        _draw_wall_tag(out, masonry)

    # Encode annotated frame back to base64
    _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()

    # ── ArUco location tagging ────────────────────────────────────
    # Runs on the SAME frame as the YOLO pass above, so each crack gets a
    # marker-relative position from the markers visible in that identical image.
    locs = []
    live_contours = []
    # Best (highest-confidence) crack as the headline for this frame.
    best_conf = 0.0
    for c in crack_lines:
        conf = float(c[1]) if len(c) > 1 else 0.0
        if conf > best_conf:
            best_conf = conf

    return {
        'level':           level,
        'area':            round(total_length),
        'crack_length':    round(total_length, 1),
        'crack_width':     round(avg_width, 1),
        'status':          status,
        'ai_boxes':        crack_count,
        'confidence':      round(best_conf, 3),
        'boxes':           [{'x1': int(line[0,0,0]), 'y1': int(line[0,0,1]),
                             'x2': int(line[-1,0,0]), 'y2': int(line[-1,0,1]),
                             'length': length, 'width': width, 'conf': round(conf, 3)}
                            for line, conf, length, width, is_open in crack_lines],
        'contours':        [{'points': [[int(x), int(y)] for [x, y] in line[:, 0, :].tolist()],
                              'conf': round(conf, 3), 'length': length, 'width': width}
                             for line, conf, length, width, is_open in crack_lines],
        'coverage':        str(coverage),
        # Scene label: True when the frame is looking at a brick / block / stone
        # wall. Extra key - clients that don't know it simply ignore it.
        'masonry':         bool(masonry.get('detected')),
        'wall_type':       ('brick' if masonry.get('detected') else 'plain'),
        'joint_pitch_px':  masonry.get('pitch') or 0,
        'annotated_image': annotated_b64,
    }

@app.route('/analyze_live', methods=['POST'])
def analyze_live():
    from flask import jsonify
    import base64, re
    try:
        data       = request.get_json()
        image_data = data.get('image', '')
        # Decode base64 frame
        img_bytes  = base64.b64decode(re.sub(r'^data:image/.+;base64,', '', image_data))
        nparr      = np.frombuffer(img_bytes, np.uint8)
        frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        return jsonify(run_live_analysis(frame))
    except Exception as e:
        print(f"analyze_live error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    from flask import jsonify
    try:
        data = request.get_json()
        conn = get_db()
        conn.execute(
            "INSERT INTO feedback (original_image, annotated_image, level, area, width, is_correct, timestamp) VALUES (?,?,?,?,?,?,?)",
            (data['orig_img'], data['annotated_img'], data['level'],
             data['area'], data['width'], data['is_correct'],
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain_model():
    from flask import jsonify
    import glob, shutil, yaml

    os.makedirs('feedback_train/images', exist_ok=True)
    os.makedirs('feedback_train/labels', exist_ok=True)

    # Collect correct feedback images
    conn = get_db()
    rows = conn.execute(
        "SELECT original_image, level, area, width FROM feedback WHERE is_correct = 1"
    ).fetchall()
    conn.close()

    if not rows:
        return jsonify({'error': 'No correct feedback to train on'}), 400

    # Clear previous training data
    for f in glob.glob('feedback_train/images/*'):
        os.remove(f)
    for f in glob.glob('feedback_train/labels/*'):
        os.remove(f)

    label_count = 0
    for row in rows:
        orig = row['original_image'].lstrip('/')
        if not os.path.exists(orig):
            continue
        img = cv2.imread(orig)
        if img is None:
            continue
        h, w = img.shape[:2]

        base = os.path.splitext(os.path.basename(orig))[0]
        shutil.copy2(orig, f'feedback_train/images/{base}.jpg')

        # Re-run detection to get label coordinates (seg format: 4 polygon corners)
        boxes, confs, _ = predict(img)
        if boxes:
            with open(f'feedback_train/labels/{base}.txt', 'w') as f:
                for (x1, y1, x2, y2), conf in zip(boxes, confs):
                    if conf < 0.20:
                        continue
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w / 2
                    bh = (y2 - y1) / h / 2
                    # 4 corners as polygon
                    pts = [
                        xc - bw, yc - bh,
                        xc + bw, yc - bh,
                        xc + bw, yc + bh,
                        xc - bw, yc + bh,
                    ]
                    f.write("0 " + " ".join(f"{p:.6f}" for p in pts) + "\n")
                    label_count += 1
        else:
            open(f'feedback_train/labels/{base}.txt', 'w').close()

    if label_count == 0:
        return jsonify({'error': 'No valid labels generated'}), 400

    # Write data.yaml
    data_yaml = {
        'path': os.path.abspath('feedback_train'),
        'train': 'images',
        'val': 'images',
        'nc': 1,
        'names': ['crack']
    }
    with open('feedback_train/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # Run fine-tuning
    try:
        from ultralytics import YOLO
        model_path = 'crack_trained.pt'
        if not os.path.exists(model_path):
            return jsonify({'error': 'crack_trained.pt not found'}), 400

        finetune = YOLO(model_path)
        finetune.train(
            data='feedback_train/data.yaml',
            epochs=10,
            imgsz=640,
            batch=4,
            device=0,
            workers=0,
            exist_ok=True,
            verbose=False,
            task='segment',
            name='finetune_seg'
        )
        # Copy best weights over original model
        best = 'runs/segment/finetune_seg/weights/best.pt'
        if os.path.exists(best):
            shutil.copy2(best, model_path)
            global MODEL_NAME
            MODEL_NAME = 'crack_trained.pt (fine-tuned)'
            # Reload
            global model
            model = YOLO(model_path)
            # Cleanup training dir
            shutil.rmtree('runs/segment/finetune_seg', ignore_errors=True)
            return jsonify({'ok': True, 'labels': label_count})
        else:
            return jsonify({'error': 'Training completed but best.pt not found'}), 500
    except Exception as e:
        print(f"Retrain error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/count')
def feedback_count():
    conn = get_db()
    correct = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return {'correct': correct, 'total': total}

@app.route('/history')
def history():
    try:
        conn    = get_db()
        records = conn.execute(
            "SELECT id, image_path, crack_level, crack_area, crack_width, timestamp,"
            "       camera_type, status FROM cracks ORDER BY timestamp DESC, id DESC"
        ).fetchall()
        conn.close()
        # Convert to plain tuples so template works the same
        records = [tuple(r) for r in records]
        return render_template('history.html', records=records, db_status="Connected")
    except Exception as e:
        print(f"History error: {e}")
        return render_template('history.html', records=[], db_status=f"Error: {e}")


# ── VR digital twin — Unity-facing crack API ─────────────
# Deterministic placeholder 3D placement: no photogrammetry
# pipeline exists yet, so cracks are spread onto the four
# demo-room walls. Swap this for real Meshroom/trimesh
# coordinates when that step lands.
def crack_world_position(idx):
    hw = ROOM_WALL_EXTENT - 0.6
    y  = ROOM_Y_MIN + ((idx * 17) % 97) / 97.0 * (ROOM_Y_MAX - ROOM_Y_MIN)
    a  = ((idx * 13) % 200) / 200.0 * 2.0 * hw - hw
    wall = idx % 4
    if wall == 0:
        return {'x': round(a, 3), 'y': round(y, 3), 'z':  ROOM_WALL_EXTENT}
    if wall == 1:
        return {'x':  ROOM_WALL_EXTENT, 'y': round(y, 3), 'z': round(a, 3)}
    if wall == 2:
        return {'x': round(a, 3), 'y': round(y, 3), 'z': -ROOM_WALL_EXTENT}
    return {'x': -ROOM_WALL_EXTENT, 'y': round(y, 3), 'z': round(a, 3)}


def crack_severity(level):
    return {0: 'safe', 1: 'minor', 2: 'moderate', 3: 'severe'}.get(level, 'unknown')


# ─────────────────────────────────────────────
# CRACK PLACEMENT API (raycast placement - Unity side owns the coordinates)
# ─────────────────────────────────────────────
PLACEMENT_STATUSES = ('pending', 'placed')
INSPECTION_STATUSES = ('new', 'confirmed', 'under_repair', 'repaired')

@app.route('/api/cracks')
def api_cracks():
    """Crack records for the VR app.

    Every row starts life as status='pending'. The Unity client raycasts the crack onto a
    wall and POSTs the hit point + surface normal to /api/cracks/<id>/position, which flips
    the row to 'placed'.

    Optional ?status=pending|placed filters the placement status.
    Optional ?inspection_status=new|confirmed|under_repair|repaired filters inspection feedback.
    """
    from flask import jsonify

    want = (request.args.get('status') or '').strip().lower()
    if want and want not in PLACEMENT_STATUSES:
        return jsonify({'cracks': [],
                        'error': f"status must be one of {list(PLACEMENT_STATUSES)}"}), 400

    insp = (request.args.get('inspection_status') or '').strip().lower()
    if insp and insp not in INSPECTION_STATUSES:
        return jsonify({'cracks': [],
                        'error': f"inspection_status must be one of {list(INSPECTION_STATUSES)}"}), 400

    try:
        conn = get_db()
        sql = ("SELECT id, timestamp, confidence, crack_level, status, image_path,"
               "       camera_type, capture_group,"
               "       crack_area, crack_width,"
               "       pos_x, pos_y, pos_z, normal_x, normal_y, normal_z,"
               "       inspection_status, notes, repaired_at,"
               "       wall_id, floor, local_x, local_y "
               "FROM cracks")
        params = ()
        where = []
        if want:
            where.append("status = ?")
            params += (want,)
        if insp:
            where.append("inspection_status = ?")
            params += (insp,)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp DESC, id DESC"
        records = conn.execute(sql, params).fetchall()
        conn.close()

        base = _image_base()
        cracks = []
        for r in records:
            img = (r['image_path'] or '').lstrip('/')
            cracks.append({
                'id':              r['id'],
                'timestamp':       r['timestamp'],
                'confidence':      r['confidence'],
                'severity':        crack_severity(r['crack_level']),
                # Same two fields the website's Length and Width columns show, converted to cm
                # with the same constant the live push uses. Without these the VR marker had
                # nothing to display and rendered a literal "0.0cm".
                'length_cm':       round((r['crack_area'] or 0) * CM_PER_PX, 2),
                'width_cm':        round((r['crack_width'] or 0) * CM_PER_PX, 2),
                'status':          r['status'] or 'pending',
                'camera_type':     r['camera_type'] or 'rgb',
                'capture_group':   r['capture_group'],
                'pos_x':           r['pos_x'],
                'pos_y':           r['pos_y'],
                'pos_z':           r['pos_z'],
                'normal_x':        r['normal_x'],
                'normal_y':        r['normal_y'],
                'normal_z':        r['normal_z'],
                'inspection_status': r['inspection_status'] or 'new',
                'notes':           r['notes'],
                'repaired_at':     r['repaired_at'],
                'wall_id':         r['wall_id'],
                'floor':           r['floor'],
                'local_x':         r['local_x'],
                'local_y':         r['local_y'],
                'image_url':       f"{base}/{img}" if img else '',
                'image_base64':    _image_base64(r['image_path']),   # convenience for VR
            })

        return jsonify({'cracks': cracks})
    except Exception as e:
        return jsonify({'cracks': [], 'error': str(e)}), 500


@app.route('/api/cracks/<int:record_id>/position', methods=['POST'])
def api_set_crack_position(record_id):
    """Record where the operator placed a crack in the VR scene.

    Body: {x, y, z, nx, ny, nz} - the world hit point and the surface normal of the wall.
    Optional: wall_id, floor, local_x, local_y - spatial mapping of the wall.
    On success the row's status becomes 'placed'.
    """
    from flask import jsonify
    try:
        data = request.get_json(force=True, silent=True) or {}

        def num(key):
            v = data.get(key)
            if v is None:
                raise ValueError(f"missing {key}")
            return float(v)

        x, y, z = num('x'), num('y'), num('z')
        nx, ny, nz = num('nx'), num('ny'), num('nz')

        # A zero-length normal would make the marker orientation undefined in Unity.
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-6:
            raise ValueError("normal must not be zero-length")
        nx, ny, nz = nx / length, ny / length, nz / length

        conn = get_db()
        row = conn.execute(
            "SELECT id, status, capture_group FROM cracks WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            conn.close()
            return jsonify({'success': False, 'error': f"crack {record_id} not found"}), 404

        # Optional spatial mapping fields - old payloads without them still work.
        wall_id    = data.get('wall_id')
        floor      = data.get('floor')
        local_x    = data.get('local_x')
        local_y    = data.get('local_y')
        if local_x is not None: local_x = float(local_x)
        if local_y is not None: local_y = float(local_y)

        values = (x, y, z, nx, ny, nz, wall_id, floor, local_x, local_y)
        group = row['capture_group']
        if group:
            # One Capture writes an RGB row and a thermal row that share this group. They
            # describe the SAME physical crack, so placing either one places the whole
            # group. Without this the partner stays 'pending', Unity queues it again and
            # a second marker + photo lands on the same crack.
            cur = conn.execute(
                "UPDATE cracks SET pos_x=?, pos_y=?, pos_z=?, normal_x=?, normal_y=?, normal_z=?, "
                "wall_id=?, floor=?, local_x=?, local_y=?, status='placed'"
                " WHERE capture_group = ?",
                values + (group,))
            placed_rows = cur.rowcount
        else:
            conn.execute(
                "UPDATE cracks SET pos_x=?, pos_y=?, pos_z=?, normal_x=?, normal_y=?, normal_z=?, "
                "wall_id=?, floor=?, local_x=?, local_y=?, status='placed'"
                " WHERE id = ?",
                values + (record_id,))
            placed_rows = 1
        conn.commit()
        conn.close()

        print(f"[placement] crack #{record_id} placed at ({x:.3f}, {y:.3f}, {z:.3f}) "
              f"normal ({nx:.2f}, {ny:.2f}, {nz:.2f}) wall={wall_id} floor={floor} "
              f"local=({local_x}, {local_y}) group={group or '-'} rows={placed_rows}")
        return jsonify({'success': True, 'id': record_id, 'status': 'placed',
                        'capture_group': group,
                        'rows_placed': placed_rows,
                        'position': {'x': x, 'y': y, 'z': z},
                        'normal': {'x': nx, 'y': ny, 'z': nz}})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cracks/<int:record_id>', methods=['PATCH'])
def api_patch_crack(record_id):
    """Inspection feedback for a placed crack.

    Body: any of {inspection_status, notes}.
    inspection_status must be one of: new, confirmed, under_repair, repaired.
    When it becomes 'repaired', repaired_at is set to now; when it changes away from
    'repaired', repaired_at is cleared.
    """
    from flask import jsonify
    try:
        data = request.get_json(force=True, silent=True) or {}

        new_status = data.get('inspection_status')
        notes = data.get('notes')

        if new_status is not None and new_status not in INSPECTION_STATUSES:
            return jsonify({'success': False,
                            'error': f"inspection_status must be one of {list(INSPECTION_STATUSES)}"}), 400

        if notes is not None and not isinstance(notes, str):
            return jsonify({'success': False, 'error': 'notes must be a string'}), 400

        conn = get_db()
        row = conn.execute("SELECT id, status, inspection_status, notes, repaired_at FROM cracks WHERE id = ?",
                           (record_id,)).fetchone()
        if row is None:
            conn.close()
            return jsonify({'success': False, 'error': f"crack {record_id} not found"}), 404

        old_status = row['inspection_status']
        repairs = datetime.utcnow().isoformat()
        repaired_at = row['repaired_at']

        set_clauses = []
        params = []

        if new_status is not None:
            set_clauses.append("inspection_status = ?")
            params.append(new_status)
            if new_status == 'repaired':
                repaired_at = repairs
            elif old_status == 'repaired':
                repaired_at = None
            set_clauses.append("repaired_at = ?")
            params.append(repaired_at)

        if notes is not None:
            set_clauses.append("notes = ?")
            params.append(notes)

        if set_clauses:
            params.append(record_id)
            conn.execute("UPDATE cracks SET " + ", ".join(set_clauses) + " WHERE id = ?",
                         tuple(params))
            conn.commit()

        conn.close()

        final_notes = notes if notes is not None else row['notes']
        print(f"[inspection] crack #{record_id}: {old_status} -> {new_status or old_status} "
              f"repaired_at={repaired_at}")
        return jsonify({'success': True, 'id': record_id,
                        'inspection_status': new_status if new_status is not None else old_status,
                        'notes': final_notes,
                        'repaired_at': repaired_at})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record():
    try:
        conn = get_db()
        conn.execute("DELETE FROM cracks WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Delete error: {e}")
    return redirect(url_for('history'))


@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    from flask import jsonify
    try:
        data = request.get_json(force=True, silent=True) or {}
        ids = data.get('ids') or []
        conn = get_db()
        for rid in ids:
            conn.execute("DELETE FROM cracks WHERE id = ?", (int(rid),))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'deleted': len(ids)})
    except Exception as e:
        print(f"Delete selected error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete_all', methods=['POST'])
def delete_all():
    from flask import jsonify
    try:
        data = request.get_json(force=True, silent=True) or {}
        level = data.get('level', 'all')
        conn = get_db()
        if not (isinstance(level, int) or (isinstance(level, str) and level not in ('all', ''))):
            conn.execute("DELETE FROM cracks")
        else:
            conn.execute("DELETE FROM cracks WHERE crack_level = ?", (int(level),))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Delete all error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import os; port = int(os.environ.get('PORT', 5000))
    # app.run serves BOTH normal HTTP routes and the plain /ws WebSocket route
    # (Werkzeug 3.x dev server supports RFC 6455 upgrades used by flask-sock).
    # Reachable from the VR headset on the same Wi-Fi via http://<PC-LAN-IP>:5000.
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
