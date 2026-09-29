"""Plot FaceMap spatial masks (PC1-6) for each ROI over the mean stabilised face.

Saves facemap_output/<session>/spatial_masks_<suffix>.png, one 2x3 grid per ROI
(red = positive, blue = negative; the sign of an SVD component is arbitrary).

Usage (inside the facepipe env, from the project folder, after st3a_run_facemap.py):
    python scripts\\diagnostics\\plot_facemap_masks.py
"""
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for the stage modules
from st3a_run_facemap import OUTPUT_DIR, ROIS, SBIN, SESSION, STABILISED_VIDEO  # noqa: E402

FILE_SUFFIX = {"eyes_brows": "eyes", "lower_face": "lower"}
N_MEAN_FRAMES = 300
N_SHOW = 6


def mean_face(video_path):
    """Mean full-resolution grayscale frame from evenly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = None
    for f in np.linspace(0, n - 1, N_MEAN_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        acc = g if acc is None else acc + g
    cap.release()
    return acc / N_MEAN_FRAMES


def save_with_retry(fig, out, attempts=5):
    """Save via a temp file + rename. On Windows an open image preview or a
    scanner can briefly hold the target, which makes a direct save fail (Errno 22)."""
    tmp = out.with_name(out.stem + ".tmp.png")
    fig.savefig(tmp, dpi=110)
    for i in range(attempts):
        try:
            os.replace(tmp, out)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def main():
    face = mean_face(STABILISED_VIDEO)
    for r in ROIS:
        label = r["label"]
        masks = np.load(OUTPUT_DIR / "{}_masks.npy".format(label))
        varexp = np.load(OUTPUT_DIR / "{}_varexp.npy".format(label))
        # Pixels FaceMap actually used: whole sbin blocks from the start of the ROI.
        hb, wb = masks.shape[:2]
        y0, x0 = (r["y"] // SBIN) * SBIN, (r["x"] // SBIN) * SBIN
        y1, x1 = y0 + hb * SBIN, x0 + wb * SBIN
        crop = face[y0:y1, x0:x1]

        fig, axes = plt.subplots(2, 3, figsize=(11, 0.75 * 11 * (y1 - y0) / (x1 - x0) + 1.2))
        for k, ax in enumerate(axes.flat[:N_SHOW]):
            m = masks[:, :, k]
            m_up = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            lim = np.abs(m).max()
            ax.imshow(crop, cmap="gray")
            ax.imshow(m_up, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55)
            ax.set_title("PC{}  ({:.1f}% var)".format(k + 1, varexp[k] * 100))
            ax.axis("off")
        fig.suptitle("FaceMap motion SVD masks: {} (x={}-{}, y={}-{}), {}\n"
                     "red = positive, blue = negative weight (sign of each PC is arbitrary)".format(
                         label, x0, x1, y0, y1, SESSION), fontsize=11)
        fig.tight_layout()
        out = OUTPUT_DIR / "spatial_masks_{}.png".format(FILE_SUFFIX.get(label, label))
        save_with_retry(fig, out)
        plt.close(fig)
        print("saved {}".format(out))


if __name__ == "__main__":
    main()
