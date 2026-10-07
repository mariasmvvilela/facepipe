"""Plot FaceMap spatial masks (PC1-6, or --n-show) for each ROI over the mean face.

Reads stage 2's <roi>_masks.npy / <roi>_varexp.npy and the ROI's bounding box from
facemap_output/<session>/summary.json, and the mean face from stage 1's face.avi.
Saves facemap_output/<session>/spatial_masks_<roi>.png, one grid per ROI
(red = positive, blue = negative; the sign of an SVD component is arbitrary;
the ROI outline is drawn in yellow).

Usage (inside the facepipe env, from the project folder, after st2_run_facemap.py):
    python scripts\\diagnostics\\plot_facemap_masks.py --session session_YYYYMMDD_HHMMSS_<task>
    python scripts\\diagnostics\\plot_facemap_masks.py --session session_YYYYMMDD_HHMMSS_<task> --rois mouth
"""
import argparse
import json
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
from st2_run_facemap import DEFAULT_SESSION, PROJECT_DIR, SBIN  # noqa: E402

N_MEAN_FRAMES = 300


def mean_face(video_path):
    """Mean grayscale frame from evenly spaced frames."""
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION,
                        help="session folder name (or path) in raw_data/")
    parser.add_argument("--rois", nargs="+", default=None,
                        help="ROIs to plot (default: every ROI in stage 2's summary.json)")
    parser.add_argument("--n-show", type=int, default=6, help="number of PCs to plot (5 per row)")
    args = parser.parse_args()
    session = Path(args.session).name
    face_video = PROJECT_DIR / "preprocessed" / session / "face.avi"
    roi_masks = dict(np.load(PROJECT_DIR / "preprocessed" / session / "rois.npz"))
    out_dir = PROJECT_DIR / "facemap_output" / session
    with open(out_dir / "summary.json") as f:
        fm_rois = json.load(f)["rois"]
    names = args.rois or list(fm_rois)
    n_cols = 3 if args.n_show <= 6 else 5
    n_rows = -(-args.n_show // n_cols)

    face = mean_face(face_video)
    for label in names:
        masks = np.load(out_dir / "{}_masks.npy".format(label))
        varexp = np.load(out_dir / "{}_varexp.npy".format(label))
        box = fm_rois[label]["box_in_face_video"]
        # Pixels FaceMap actually used: whole sbin blocks from the box origin.
        hb, wb = masks.shape[:2]
        y0, x0 = box["y"], box["x"]
        y1, x1 = y0 + hb * SBIN, x0 + wb * SBIN
        crop = face[y0:y1, x0:x1]
        outline = roi_masks[label][y0:y1, x0:x1]

        width = 11 if n_cols == 3 else 16
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(
            width, n_rows * 1.1 * width / n_cols * (y1 - y0) / (x1 - x0) + 1.2), squeeze=False)
        for ax in axes.flat[args.n_show:]:
            ax.axis("off")
        for k, ax in enumerate(axes.flat[:args.n_show]):
            m_up = cv2.resize(masks[:, :, k], (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            lim = np.abs(m_up).max()
            ax.imshow(crop, cmap="gray")
            ax.imshow(m_up, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55)
            ax.contour(outline, levels=[0.5], colors="yellow", linewidths=0.6)
            ax.set_title("PC{}  ({:.1f}% var)".format(k + 1, varexp[k] * 100))
            ax.axis("off")
        fig.suptitle("FaceMap motion SVD masks: {}, {}\n"
                     "red = positive, blue = negative weight (sign of each PC is arbitrary)".format(
                         label, session), fontsize=11)
        fig.tight_layout()
        out = out_dir / "spatial_masks_{}.png".format(label)
        save_with_retry(fig, out)
        plt.close(fig)
        print("saved {}".format(out))


if __name__ == "__main__":
    main()
