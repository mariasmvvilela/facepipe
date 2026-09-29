"""Preview the FaceMap ROIs on a frame of the stabilised video.

Draws every ROI in st3a_run_facemap.ROIS (one colour each) plus reference lines on
stabilised frame 1000 and saves stabilised_video/<session>/roi_two_panel_preview.png.
Also checks, using the stage-1 landmarks and stage-2 transforms, where the chin
and lower eyelids actually sit in the crop.

Usage (inside the facepipe env, from the project folder):
    python scripts\\diagnostics\\draw_roi.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for the stage modules
from st3a_run_facemap import ROIS, SESSION  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
STAB_DIR = PROJECT_DIR / "stabilised_video" / SESSION
LANDMARKS = PROJECT_DIR / "mediapipe_output" / SESSION / "landmarks.npy"
OUTPUT_NAME = "roi_two_panel_preview.png"
FRAME_IDX = 1000
SRC_SIZE = (640, 480)   # original video width, height (landmarks are normalised to it)

REFERENCE_LINES = {"eyes": 90, "nose tip": 145, "chin (planned)": 210}
ROI_COLOURS = [(0, 255, 0), (255, 0, 255), (0, 165, 255)]   # BGR: green, magenta, orange
CHECK = {"chin": 152, "right lower eyelid": 145, "left lower eyelid": 374}


def landmark_y_in_crop(landmarks, transforms, idx):
    pts = landmarks[:, idx, :2] * np.array(SRC_SIZE)
    return (np.einsum("nij,nj->ni", transforms[:, :, :2], pts) + transforms[:, :, 2])[:, 1]


def main():
    cap = cv2.VideoCapture(str(STAB_DIR / "stabilised.avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("ERROR: could not read frame {} of the stabilised video.".format(FRAME_IDX))

    landmarks = np.load(LANDMARKS)
    transforms = np.load(STAB_DIR / "transforms.npy")
    measured = {name: landmark_y_in_crop(landmarks, transforms, idx) for name, idx in CHECK.items()}

    # Draw on a 3x upscale so the lines and labels are readable.
    s = 3
    img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    for label, y in REFERENCE_LINES.items():
        cv2.line(img, (0, y * s), (img.shape[1], y * s), (255, 255, 0), 1)
        cv2.putText(img, "{} y={}".format(label, y), (4, y * s - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    for r, colour in zip(ROIS, ROI_COLOURS):
        x0, y0, x1, y1 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]
        cv2.rectangle(img, (x0 * s, y0 * s), (x1 * s, y1 * s), colour, 2)
        cv2.putText(img, r["label"], (x0 * s + 6, y0 * s + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    out_path = STAB_DIR / OUTPUT_NAME
    cv2.imwrite(str(out_path), img)

    for r in ROIS:
        print("ROI {label}: x={x}, y={y}, w={w}, h={h}".format(**r))
        print("     covers x={} to x={}, y={} to y={}".format(
            r["x"], r["x"] + r["w"], r["y"], r["y"] + r["h"]))
    print("Measured in stabilised crop (median / 1st pct / 99th pct y):")
    for name, y in measured.items():
        print("  {:18s} {:6.1f} / {:6.1f} / {:6.1f}".format(
            name, np.nanmedian(y), np.nanpercentile(y, 1), np.nanpercentile(y, 99)))
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()
