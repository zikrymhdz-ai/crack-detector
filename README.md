# CrackScan — AI Wall-Crack Inspection & VR Digital Twin

Flask service that detects, measures and reports cracks in walls (plain plaster,
concrete **and brick / block masonry**) from uploaded photos, videos or a live
camera, and pushes each finding to a Unity VR digital twin over a plain
WebSocket.

## What it does

* **Crack detection** with a YOLOv8 segmentation model (`crack_fine.pt`), tiled
  inference for large stills, single-pass inference for live/video.
* **Crack measurement** — skeletonised centreline, full length including
  branches, average width, severity level (SAFE / MINOR / MEDIUM / SEVERE).
* **Masonry aware** — the mortar grid is detected from the image itself, so a
  brick/block wall is recognised (`masonry`, `wall_type`, `joint_pitch_px` in the
  API payload) and its joints are filtered *without* discarding a real crack that
  runs across them.
* **Live inspection** — webcam, phone or RTSP camera (Hikmicro RGB + thermal),
  MJPEG streams, `/analyze_live` JSON endpoint with annotated frame, contours and
  measurements.
* **VR digital twin** — `/ws` WebSocket pushes `new_crack` events; crack records
  are stored in SQLite and their 3D placement (wall, position, normal,
  inspection status) is written back by the Unity client via
  `POST /api/cracks/<id>/position` and `PATCH /api/cracks/<id>`.
* **History & review** — `/history`, delete/select endpoints, feedback capture
  and a retrain hook.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# put the weights next to app.py (not versioned, see below)
#   crack_fine.pt

python app.py                     # http://localhost:5000
```

Set `PORT` to change the port. The server listens on `0.0.0.0` so the VR headset
can reach it over the LAN.

### Model weights

`*.pt` files are **not** committed (`crack_fine.pt` alone is ~95 MB). Place the
segmentation checkpoint in the project root:

```
crack_fine.pt
```

If it is missing the app still boots and falls back to the edge-only /
Roboflow path. Training helpers live in `scripts/` (`train_fine.py`,
`build_fine_dataset.py`, `eval_fine.py`).

## Main routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET/POST | Upload an image/video or a base64 live capture |
| `/live` | GET | Live camera page |
| `/video_feed` | GET | MJPEG stream with overlay |
| `/analyze_live` | POST | `{image: dataURI}` → level, lengths, contours, `masonry`, annotated frame |
| `/vr_scan_image` | POST | Same pipeline for the VR client |
| `/vr_frame`, `/vr_rgb_frame`, `/vr_thermal_frame` | GET | Raw / annotated frames for Unity |
| `/ws` | WebSocket | `new_crack` push channel for the VR twin |
| `/api/cracks` | GET | Crack records |
| `/api/cracks/<id>/position` | POST | Store the raycast placement from Unity |
| `/api/cracks/<id>` | PATCH | Inspection status / notes |
| `/history` | GET | Inspection history page |
| `/feedback`, `/retrain` | POST | Capture corrections, fine-tune hook |

## Mobile app (Capacitor)

`mobile/` contains an Android app built with [Capacitor](https://capacitorjs.com)
around the same UI the server renders at `/app/`. The phone captures photos or a
live camera preview and the **server** does the detection, so the model never has
to run on the device.

* **Scan** – live preview with the crack drawn on top, shutter to analyse the
  current frame, gallery / long-press for a full-resolution photo.
* **History** – every saved scan from the server, filterable by inspection status,
  with confirm / under-repair / repaired actions.
* **Server** – LAN address of the PC, connection test, model + version info.

Build and run instructions: [`mobile/README.md`](mobile/README.md).

```
cd mobile && npm install
npx cap sync android
cd android && .\gradlew.bat assembleDebug    # needs JDK 23 + Android SDK
```

### Mobile API

| Route | Method | Purpose |
|---|---|---|
| `/api/mobile/health` | GET | handshake: app version, model, masonry support |
| `/api/mobile/scan` | POST | analyse + store a photo (multipart `image` or JSON base64) |
| `/api/mobile/records/<id>` | DELETE | remove a scan |
| `/app/` | GET | the mobile web build, for testing in a phone browser |

## Accuracy notes

Detection is the model plus a measurement stage that rejects coin-sized
speckle, filled blobs, brick texture and mortar joints:

* a crack must be **longer** than a fraction of the frame diagonal, **thin**
  (area / skeleton length), non-filled, non-compact, and must sit in a **dark
  valley with brighter wall on both flanks** (measured perpendicular to the
  stroke, using a structure-tensor normal so wide cracks are handled too);
* the "darker than the wall" test is **local** (relative to a mid-scale
  background), so it behaves the same on a white plaster wall, a sunlit facade
  and a dark brick wall;
* long straight runs are erased (mortar joints, tile edges) **unless** they are
  genuinely dark — a straight settlement crack survives;
* on masonry, only strokes that repeat along the detected joint grid are dropped,
  and only when several of them do (one straight crack is not a grid).

### Evaluation harness

`eval_set.json` labels the sample images (crack / no crack, wall type) and
`scripts/score.py` scores the current pipeline against them:

```bash
python scripts/score.py --verbose     # cached model output
python scripts/score.py --refresh     # re-run the segmenter
```

Reference point on the bundled 56-image set: recall 1.00 on the crack images,
with false positives limited to non-wall scenes (people, posters, diagrams).
Brick / block / stone walls are detected normally.

## Layout

```
app.py                Flask app, detection + measurement pipeline, routes
aruco_locator.py      ArUco marker helpers
make_aruco_markers.py marker sheet generator
templates/            upload, live, result, history pages
scripts/              training, dataset building and evaluation helpers
mobile/               Capacitor Android app (see mobile/README.md)
feedback_train/       small feedback dataset used by /retrain
eval_set.json         labelled evaluation set
```

## Notes

* `feedback_train/data.yaml` uses an absolute Windows path; adjust it before
  retraining on another machine.
* Camera/API credentials are read from constants in `app.py`; move them to
  environment variables before sharing the project more widely.
