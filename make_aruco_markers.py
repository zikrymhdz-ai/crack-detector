"""Generate and print the ArUco markers used for crack location tagging.

Usage
-----
List what markers.json defines:
    python make_aruco_markers.py --list

Write PNGs (one per marker id) into ./aruco_markers/:
    python make_aruco_markers.py --generate

Write a single PDF sheet, 5 per A4 page, ready to print at true size:
    python make_aruco_markers.py --pdf

Then MEASURE a printed marker with a ruler and set the exact size:
    python make_aruco_markers.py --set-size 10.0

Sizing matters: the app derives pixels-per-cm from the marker's measured pixel size, so if
the printed marker is not the size you tell it about, every crack offset is scaled wrongly.
Print at 100% (no "fit to page") and measure the black square's edge, not the white border.
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS_JSON = os.path.join(HERE, 'markers.json')
OUT_DIR = os.path.join(HERE, 'aruco_markers')

# Physical width of the black marker square, in cm. Must match app.py's
# ARUCO_MARKER_SIZE_CM (or the ARUCO_MARKER_SIZE_CM environment variable).
DEFAULT_SIZE_CM = 10.0

# White quiet zone around the marker. The ArUco spec wants >= 1 module of white border on
# every side; without it detection fails on busy backgrounds.
QUIET_ZONE_MODULES = 1

# Printable resolution. 300 dpi is plenty for a 10 cm marker.
DPI = 300


def load_marker_ids():
    """Marker ids to generate, from markers.json (or 0..9 when the file is missing)."""
    try:
        with open(MARKERS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            ids = sorted(int(k) for k in data.keys())
        elif isinstance(data, list):
            ids = sorted(int(i['id']) for i in data if isinstance(i, dict) and 'id' in i)
        else:
            ids = []
        if ids:
            return ids
    except Exception as e:
        print(f"(could not read markers.json: {e})")
    return list(range(10))


def load_marker_names():
    try:
        with open(MARKERS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            return {int(i['id']): str(i.get('name', i['id']))
                    for i in data if isinstance(i, dict) and 'id' in i}
    except Exception:
        pass
    return {}


def marker_image(marker_id, pixels, quiet_modules):
    """Render one marker as a grayscale image, `pixels` per black square.

    cv2.aruco.generateImageMarker returns only the marker itself (no border), so the quiet
    zone is padded on afterwards.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)

    pad = pixels * quiet_modules
    size = pixels + 2 * pad
    canvas = np.full((size, size), 255, dtype=np.uint8)
    canvas[pad:pad + pixels, pad:pad + pixels] = marker
    return canvas


def cm_to_pixels(size_cm):
    return max(32, int(round(size_cm / 2.54 * DPI)))


def generate_pngs(size_cm):
    ids = load_marker_ids()
    os.makedirs(OUT_DIR, exist_ok=True)
    px = cm_to_pixels(size_cm)

    for mid in ids:
        img = marker_image(mid, px, QUIET_ZONE_MODULES)
        path = os.path.join(OUT_DIR, f'aruco_{mid:02d}.png')
        cv2.imwrite(path, img)
        print(f"  wrote {os.path.basename(path)}  ({px}px square, {size_cm}cm at {DPI}dpi)")

    print(f"\n{len(ids)} marker(s) in {OUT_DIR}")
    return ids


def generate_pdf(size_cm, per_page=5):
    ids = load_marker_ids()
    os.makedirs(OUT_DIR, exist_ok=True)
    px = cm_to_pixels(size_cm)
    names = load_marker_names()

    # A4 at 300 dpi
    page_w, page_h = int(8.27 * DPI), int(11.69 * DPI)
    margin = int(0.5 * DPI)
    pages = []

    for start in range(0, len(ids), per_page):
        chunk = ids[start:start + per_page]
        page = np.full((page_h, page_w), 255, dtype=np.uint8)
        cell_h = (page_h - 2 * margin) // per_page

        for k, mid in enumerate(chunk):
            img = marker_image(mid, px, QUIET_ZONE_MODULES)
            y = margin + k * cell_h
            x = margin

            # centre the marker horizontally
            if img.shape[0] < cell_h - int(0.6 * DPI):
                x = (page_w - img.shape[1]) // 2
                page[y:y + img.shape[0], x:x + img.shape[1]] = img
            else:
                # marker too big for one cell - scale the preview down (measure on paper!)
                scale = (cell_h - int(0.6 * DPI)) / img.shape[0]
                small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
                page[y:y + small.shape[0], x:x + small.shape[1]] = small

            label = f"#{mid}   {names.get(mid, '')}"
            cv2.putText(page, label, (margin, y + img.shape[0] + int(0.35 * DPI)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(page, f"{size_cm} cm square - print at 100%, do NOT fit to page",
                        (margin, y + img.shape[0] + int(0.62 * DPI)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)

        pages.append(page)

    out = os.path.join(OUT_DIR, 'aruco_sheet.pdf')
    try:
        import fitz  # PyMuPDF, optional
        doc = fitz.open()
        for pg in pages:
            buf = cv2.imencode('.png', pg)[1].tobytes()
            w, h = pg.shape[1] * 72 / DPI, pg.shape[0] * 72 / DPI
            page_pdf = doc.new_page(width=w, height=h)
            page_pdf.insert_image(fitz.Rect(0, 0, w, h), stream=buf)
        doc.save(out)
        doc.close()
        print(f"  wrote {out}  ({len(pages)} page(s))")
    except ImportError:
        # No PyMuPDF - write each page as a PNG instead so it is still printable.
        for i, pg in enumerate(pages):
            p = os.path.join(OUT_DIR, f'aruco_sheet_p{i + 1}.png')
            cv2.imwrite(p, pg)
            print(f"  wrote {p}")
        print("\n  (install PyMuPDF for a true multi-page PDF:  pip install pymupdf)")
    except Exception as e:
        for i, pg in enumerate(pages):
            p = os.path.join(OUT_DIR, f'aruco_sheet_p{i + 1}.png')
            cv2.imwrite(p, pg)
            print(f"  PDF unavailable ({e}); wrote {p} instead")

    print(f"\nPrint at 100% scale. Measure the black square - it must be {size_cm} cm.")
    return out


def set_size(size_cm):
    """Record the measured marker size so app.py picks it up on next start."""
    with open(os.path.join(HERE, 'aruco_size.txt'), 'w', encoding='utf-8') as f:
        f.write(str(size_cm))
    print(f"Wrote aruco_size.txt = {size_cm}")
    print("Set the environment variable before starting the server:")
    print(f'    set ARUCO_MARKER_SIZE_CM={size_cm}')
    print("  (PowerShell:  $env:ARUCO_MARKER_SIZE_CM=\"" + str(size_cm) + "\")")


def main():
    ap = argparse.ArgumentParser(description='Generate / print ArUco markers for crack tagging')
    ap.add_argument('--list', action='store_true', help='list marker ids and names')
    ap.add_argument('--generate', action='store_true', help='write PNG files')
    ap.add_argument('--pdf', action='store_true', help='write a printable sheet')
    ap.add_argument('--size', type=float, default=DEFAULT_SIZE_CM, help='marker size in cm')
    ap.add_argument('--set-size', type=float, metavar='CM',
                    help='record the measured size to aruco_size.txt')
    args = ap.parse_args()

    if args.set_size:
        set_size(args.set_size)
        return

    if args.list or not (args.generate or args.pdf):
        names = load_marker_names()
        ids = load_marker_ids()
        print(f"{len(ids)} marker(s) defined in markers.json:\n")
        for mid in ids:
            print(f"  #{mid:<3} {names.get(mid, '(unnamed)')}")
        print("\nUse --generate for PNGs or --pdf for a printable sheet.")
        if not args.list:
            args.generate = True
            generate_pngs(args.size)
        return

    if args.pdf:
        generate_pdf(args.size)
    if args.generate:
        generate_pngs(args.size)


if __name__ == '__main__':
    main()
