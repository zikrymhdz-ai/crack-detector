"""ArUco marker-based location tagging.

Each detected crack is located relative to an ArUco marker that is visible in the SAME
frame. The marker's ID maps to a human-readable place name (markers.json), and the crack's
pixel offset from the marker centre - measured in the marker's OWN rotated axes - gives a
position that survives the marker being slightly tilted in the photo.

Everything here is additive: no existing detection behaviour changes when no marker is
visible, cracks are simply stored with located=0 and marker_id=NULL.
"""
import json
import math
import os

import cv2
import numpy as np

# ── configuration ────────────────────────────────────────────────
MARKERS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markers.json')

# Physical width of one printed marker, in cm. Change this if you re-print at a new size -
# every px->cm conversion is derived from it, so a wrong value scales all offsets wrongly.
MARKER_SIZE_CM = 10.0

# Minimum marker edge in pixels before we trust its scale. Below this, perspective noise
# dominates and offsets get nonsense.
MIN_MARKER_PX = 24.0

# Reject markers whose corners are not convex / reasonably flat - these are usually false
# positives on busy concrete.
MIN_MARKER_ASPECT = 0.35
MAX_MARKER_ASPECT = 2.85

_aruco_dict = None
_detector = None
_marker_names = {}
_markers_loaded_at = 0.0


def _ensure_detector():
    """Build the cv2.aruco detector once. OpenCV moved this API around across versions, so
    both the modern ArucoDetector and the legacy detectMarkers path are supported."""
    global _aruco_dict, _detector
    if _detector is not None:
        return _detector

    aruco = cv2.aruco
    _aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Tuned for photos of walls: tolerant of blur/noise, strict about marker shape.
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 8
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3
    params.minMarkerDistanceRate = 0.05

    if hasattr(aruco, 'ArucoDetector'):
        _detector = aruco.ArucoDetector(_aruco_dict, params)
    else:  # pragma: no cover - very old OpenCV
        _detector = params  # legacy path uses dict+params directly
    return _detector


def load_marker_names(force=False):
    """Read markers.json -> {id: location name}. Missing file is fine (markers stay unnamed)."""
    global _marker_names, _markers_loaded_at
    if _marker_names and not force:
        return _marker_names

    names = {}
    try:
        if os.path.exists(MARKERS_JSON):
            with open(MARKERS_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Accept both {"3": "Block A"} and [{"id":3,"name":"Block A"}]
            if isinstance(data, dict):
                for k, v in data.items():
                    names[int(k)] = str(v)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'id' in item:
                        names[int(item['id'])] = str(item.get('name') or item.get('location') or item['id'])
    except Exception as e:
        print(f"[aruco] could not read {MARKERS_JSON}: {e}")

    _marker_names = names
    _markers_loaded_at = cv2.getTickCount()
    return names


def marker_name(marker_id):
    """Named location for a marker id, or a readable placeholder when markers.json has none."""
    names = load_marker_names()
    if marker_id in names:
        return names[marker_id]
    return f"Marker {marker_id}"


def detect_markers(frame_bgr):
    """Detect ArUco markers in a BGR frame.

    Returns a list of dicts:
        {id, center:(x,y), size_px, u:(unit right), v:(unit down), corners:[[x,y]x4]}

    u and v are unit vectors along the marker's own axes, taken from its top edge and left
    edge, so offsets stay correct when the marker is rotated in the photo.
    """
    if frame_bgr is None:
        return []

    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = frame_bgr

    _ensure_detector()
    aruco = cv2.aruco

    try:
        if hasattr(aruco, 'ArucoDetector'):
            corners, ids, _rejected = _detector.detectMarkers(gray)
        else:  # pragma: no cover
            corners, ids, _rejected = aruco.detectMarkers(gray, _aruco_dict, parameters=_detector)
    except Exception as e:
        print(f"[aruco] detection failed: {e}")
        return []

    if ids is None or corners is None:
        return []

    out = []
    for quad, mid in zip(corners, ids.flatten()):
        pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)

        # ArUco corner order is TL, TR, BR, BL.
        top_edge    = pts[1] - pts[0]
        left_edge   = pts[3] - pts[0]
        top_len     = float(np.linalg.norm(top_edge))
        left_len    = float(np.linalg.norm(left_edge))
        if top_len < 1e-6 or left_len < 1e-6:
            continue

        # Reject markers seen at an extreme angle - the scale would be unreliable.
        aspect = top_len / left_len
        if aspect < MIN_MARKER_ASPECT or aspect > MAX_MARKER_ASPECT:
            continue

        size_px = (top_len + left_len) / 2.0
        if size_px < MIN_MARKER_PX:
            continue

        out.append({
            'id':     int(mid),
            'center': ((pts[:, 0].mean()), (pts[:, 1].mean())),
            'size_px': size_px,
            'u':      (top_edge / top_len),    # unit vector along marker "right"
            'v':      (left_edge / left_len),   # unit vector along marker "down"
            'corners': pts.tolist(),
        })

    return out


def locate_points(markers, points, marker_size_cm=None):
    """Attach a marker-relative position to each point.

    For every point the NEAREST marker is chosen (distance is measured in that marker's own
    axes, so rotation does not skew the choice). Returns a list aligned 1:1 with `points` -
    a None point yields an unlocated entry rather than being dropped:
        {marker_id, location_name, offset_x_cm, offset_y_cm, px_per_cm, located}

    px_per_cm comes from the marker's measured pixel size, which is what makes the offset a
    real-world measurement rather than a pixel count.
    """
    size_cm = float(marker_size_cm if marker_size_cm is not None else MARKER_SIZE_CM)
    if size_cm <= 0:
        size_cm = MARKER_SIZE_CM

    results = []
    for p in points:
        if p is None:
            results.append({
                'marker_id': None, 'location_name': None,
                'offset_x_cm': None, 'offset_y_cm': None,
                'px_per_cm': None, 'located': False,
            })
            continue

        px = float(p[0])
        py = float(p[1])

        best = None
        best_d2 = float('inf')

        for m in markers:
            dx = px - m['center'][0]
            dy = py - m['center'][1]
            # Project into the marker's rotated frame before comparing, so a tilted marker
            # does not make a nearby crack look far away.
            lx = dx * m['u'][0] + dy * m['u'][1]
            ly = dx * m['v'][0] + dy * m['v'][1]
            d2 = lx * lx + ly * ly
            if d2 < best_d2:
                best_d2 = d2
                best = (m, lx, ly)

        if best is None:
            results.append({
                'marker_id':     None,
                'location_name': None,
                'offset_x_cm':   None,
                'offset_y_cm':   None,
                'px_per_cm':     None,
                'located':       False,
            })
            continue

        m, lx, ly = best
        px_per_cm = m['size_px'] / size_cm

        results.append({
            'marker_id':     int(m['id']),
            'location_name': marker_name(int(m['id'])),
            'offset_x_cm':   round(lx / px_per_cm, 2) if px_per_cm > 0 else None,
            'offset_y_cm':   round(ly / px_per_cm, 2) if px_per_cm > 0 else None,
            'px_per_cm':     round(px_per_cm, 4),
            'located':       True,
        })

    return results


def _polyline_to_points(line):
    """Coerce a crack polyline into [[x, y], ...].

    The two detection paths in this project disagree on shape: run_live_analysis emits an
    (N, 1, 2) array, while measure_combined_mask/analyze_frame emit (N, 2). Normalising both
    here means callers never have to branch on which code path produced the data.
    """
    try:
        arr = np.asarray(line, dtype=np.float64)
    except Exception:
        return []

    if arr.size == 0:
        return []

    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 1:
        if arr.size < 2:
            return []
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] < 2:
        return []

    return arr[:, :2].tolist()


def normalize_contours(res):
    """Accept either shape returned by this project's detection code and return
    [{'points': [[x, y], ...], 'conf': float}, ...].

    Two producers exist and they disagree:

    * run_live_analysis()  -> [{'points': [...], 'conf': f, 'length': l, 'width': w}]
    * analyze_frame()      -> [(skeleton_image, contour, length, width, conf), ...]
                              from measure_combined_mask(). Element 0 is a skeleton BITMAP,
                              not coordinates - the polyline is element 1 and the confidence
                              is element 4.

    Normalising here means callers never have to branch on which path produced the data.
    """
    out = []
    if res is None:
        return out

    if isinstance(res, dict):
        items = res.get('contours') or []
    else:
        items = res or []

    for it in items:
        if isinstance(it, dict):
            poly = _polyline_to_points(it.get('points'))
            try:
                conf = float(it.get('conf') or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            # measure_combined_mask tuple: (skel, contour, length, width, conf)
            poly = _polyline_to_points(it[1])
            try:
                conf = float(it[4])
            except (IndexError, TypeError, ValueError):
                conf = 1.0
        else:
            continue

        if len(poly) >= 2:
            out.append({'points': poly, 'conf': conf})

    return out


def normalize_crack_lines(crack_lines):
    """Convert this project's `crack_lines` tuples into contour dicts.

    crack_lines entries are (contour, conf, length, width, is_open) - the polyline is
    element 0 and the confidence is element 1. That is a DIFFERENT layout from
    measure_combined_mask's (skeleton, contour, length, width, conf), so it gets its own
    converter rather than being guessed at by normalize_contours().
    """
    out = []
    for it in (crack_lines or []):
        if not isinstance(it, (list, tuple)) or len(it) < 2:
            continue
        poly = _polyline_to_points(it[0])
        try:
            conf = float(it[1])
        except (TypeError, ValueError):
            conf = 0.0
        if len(poly) >= 2:
            out.append({'points': poly, 'conf': conf})
    return out


def contour_midpoints(contours):
    """One representative point (polyline midpoint) per contour.

    Returns EXACTLY one point per input contour so that its length matches and stays
    index-aligned with the caller's list. _contour_points() filters short contours, which
    shifts indices and makes locs[i] refer to a different crack than contours[i].
    """
    pts = []
    for c in (contours or []):
        poly = c.get('points') or []
        if not poly:
            pts.append(None)
            continue
        mid = poly[len(poly) // 2]
        try:
            pts.append((float(mid[0]), float(mid[1])))
        except (TypeError, ValueError):
            pts.append(None)
    return pts


def _contour_points(res, limit=400):
    """Flatten detection output into [(x, y), ...] pixel points.

    One point per crack (its polyline midpoint) is used, so a single marker-relative
    location is produced per detected crack - matching the one-row-per-crack DB shape.
    """
    pts = []
    for c in normalize_contours(res):
        poly = c['points']
        if not poly:
            continue
        mid = poly[len(poly) // 2]
        if isinstance(mid, (list, tuple)) and len(mid) >= 2:
            try:
                pts.append((float(mid[0]), float(mid[1])))
            except (TypeError, ValueError):
                pass
        if len(pts) >= limit:
            break
    return pts


def contour_extent_cm(res, loc, marker_size_cm=None):
    """Width/height of a crack in cm, using the same px_per_cm as its offset.

    `res` may be the full detection output or a single {'points': [...]} dict; the first
    contour is measured.
    """
    ppc = loc.get('px_per_cm')
    if not ppc or ppc <= 0:
        return None, None

    contours = normalize_contours(res)
    if not contours:
        return None, None

    target = contours[0]['points']
    if len(target) < 2:
        return None, None

    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    w_cm = (max(xs) - min(xs)) / ppc
    h_cm = (max(ys) - min(ys)) / ppc
    return round(w_cm, 2), round(h_cm, 2)


def annotate_frame(frame_bgr, markers, res=None, locs=None):
    """Draw marker outlines + crack offsets on a copy of the frame, for the debug image."""
    out = frame_bgr.copy()

    if locs:
        for loc in locs:
            if not loc.get('located'):
                continue
            mid = loc['marker_id']
            m = next((x for x in markers if x['id'] == mid), None)
            if not m:
                continue
            cx, cy = m['center']
            ax = cx + m['u'][0] * loc['offset_x_cm'] * loc['px_per_cm']
            ay = cy + m['u'][1] * loc['offset_x_cm'] * loc['px_per_cm']
            cv2.line(out, (int(cx), int(cy)), (int(ax), int(ay)), (0, 200, 255), 2)
            cv2.line(out, (int(cx), int(cy)), (int(ax), int(ay)), (0, 0, 255), 1)
            cv2.circle(out, (int(ax), int(ay)), 5, (0, 0, 255), -1)

    for m in markers:
        pts = np.asarray(m['corners'], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)
        c = m['center']
        cv2.circle(out, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
        cv2.putText(out, f"#{m['id']} {marker_name(m['id'])[:18]}",
                    (int(c[0]) + 8, int(c[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        # marker's own "right" axis, to show which way offsets are measured
        tip = (c[0] + m['u'][0] * m['size_px'] * 0.5,
               c[1] + m['u'][1] * m['size_px'] * 0.5)
        cv2.arrowedLine(out, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])),
                        (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)

    return out
