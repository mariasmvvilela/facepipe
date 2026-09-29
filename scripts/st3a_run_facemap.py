"""Pipeline stage 3: FaceMap motion SVD on the stabilised face video.

Runs FaceMap 1.0.8 on stabilised_video/<session>/stabilised.avi with one
motion-SVD ROI per entry in ROIS and saves to facemap_output/<session>/:

    stabilised_proc.npy       FaceMap's raw output (all 500 components per ROI)
    <label>_PCs.npy           (n_frames, 100) float32, motion SVD time courses
    <label>_masks.npy         (h_bin, w_bin, 100) float32, spatial masks (binned pixels)
    <label>_varexp.npy        (100,) float32, fraction of ROI motion-energy variance per PC
    summary.json              session metadata

Notes on the FaceMap 1.0.8 API (differs from older docs / the GUI):
  * process.run() takes sbin/motSVD/movSVD as arguments; ROIs go in via `proc`.
  * A motion-SVD ROI is rind=1 (rind=0 is a pupil ROI).
  * The number of components is fixed at 500 internally; we keep the first 100.
  * With ROIs, index 0 of motSVD/motMask is the full frame and index k+1 the k-th
    motion ROI. fullSVD=False skips the full-frame SVD, leaving index 0 empty.
  * There's no variance-explained output, so it's computed here (see below).
  * Motion energy is always mean-subtracted (FaceMap subtracts the average motion frame).

Usage (inside the facepipe env, from the project folder):
    python scripts\\st3a_run_facemap.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from facemap import process

PROJECT_DIR = Path(__file__).resolve().parent.parent
SESSION = "session_20260928_161612"
STABILISED_VIDEO = PROJECT_DIR / "stabilised_video" / SESSION / "stabilised.avi"
OUTPUT_DIR = PROJECT_DIR / "facemap_output" / SESSION

# Pixel coordinates in the 256x256 stabilised crop. The split at y=130 sits
# below the lower eyelids (eyes at y=90) and above the nose tip (y=145).
ROIS = [
    {"label": "eyes_brows", "x": 70, "y": 55, "w": 126, "h": 75},    # left edge 70 drops hair strand
    {"label": "lower_face", "x": 70, "y": 130, "w": 126, "h": 90},   # below eyes to y=220 (chin)
]
SBIN = 4
N_COMPONENTS = 100


def bin_slice(r):
    return (slice(r["yrange_bin"][0], r["yrange_bin"][-1] + 1),
            slice(r["xrange_bin"][0], r["xrange_bin"][-1] + 1))


def roi_motion_sum_of_squares(video_path, rois, avgmotion):
    """Per ROI: total sum of squares of the mean-subtracted, binned motion energy.

    This is the same matrix FaceMap projects onto its masks, so
    variance explained by PC k = sum(V_k^2) / this total (masks are orthonormal).
    """
    slices = [bin_slice(r) for r in rois]
    cap = cv2.VideoCapture(str(video_path))
    Ly, Lx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Lyb, Lxb = Ly // SBIN, Lx // SBIN
    totals, prev = np.zeros(len(rois)), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[np.newaxis]
        binned = process.spatial_bin(gray, SBIN, Lyb, Lxb).reshape(Lyb, Lxb)
        if prev is not None:
            motion = np.abs(binned - prev) - avgmotion
            for k, sl in enumerate(slices):
                totals[k] += float((motion[sl] ** 2).sum())
        prev = binned
    cap.release()
    return totals


def main():
    if not STABILISED_VIDEO.is_file():
        sys.exit("ERROR: stabilised video not found (run st2_stabilise_video.py first): {}".format(
            STABILISED_VIDEO))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fm_rois = [{
        "rind": 1,                 # motion SVD ROI
        "rtype": "motion SVD",
        "ivid": 0,
        "xrange": np.arange(r["x"], r["x"] + r["w"]),
        "yrange": np.arange(r["y"], r["y"] + r["h"]),
    } for r in ROIS]
    proc_in = {
        "sbin": SBIN,
        "fullSVD": False,          # ROIs only; skip the full-frame SVD
        "save_mat": False,
        "rois": fm_rois,
        "sy": np.array([0]),
        "sx": np.array([0]),
        "savepath": str(OUTPUT_DIR),
    }

    print("Running FaceMap on {}".format(STABILISED_VIDEO))
    for r in ROIS:
        print("  ROI {label}: x={x} y={y} w={w} h={h}".format(**r))
    print("  sbin={}".format(SBIN))
    proc_path = process.run(
        filenames=[[str(STABILISED_VIDEO)]],
        sbin=SBIN,
        motSVD=True,
        movSVD=False,
        proc=proc_in,
        savepath=str(OUTPUT_DIR),
    )
    print("FaceMap output: {}".format(proc_path))

    proc = np.load(proc_path, allow_pickle=True).item()
    avgmotion = np.reshape(proc["avgmotion"][0], (proc["Lybin"][0], proc["Lxbin"][0]))

    print("Computing variance explained...")
    total_ss = roi_motion_sum_of_squares(STABILISED_VIDEO, proc["rois"], avgmotion)

    summary = {
        "session_id": SESSION.removeprefix("session_"),
        "n_frames": None,
        "n_components": N_COMPONENTS,
        "sbin": SBIN,
        "rois": {},
        "variance_definition": "fraction of total sum of squares of mean-subtracted binned "
                               "ROI motion energy captured by each PC",
    }
    print("\nSaved to {}:".format(OUTPUT_DIR))
    for k, (r, fm) in enumerate(zip(ROIS, proc["rois"])):
        V = proc["motSVD"][k + 1]                    # (n_frames, 500)
        masks = proc["motMask_reshape"][k + 1]       # (h_bin, w_bin, 500)
        # Frame 0 has no motion frame; FaceMap copies frame 1's projection into it.
        varexp = ((V[1:].astype(np.float64) ** 2).sum(axis=0) / total_ss[k])[:N_COMPONENTS]

        label = r["label"]
        pcs = V[:, :N_COMPONENTS].astype(np.float32)
        np.save(OUTPUT_DIR / "{}_PCs.npy".format(label), pcs)
        np.save(OUTPUT_DIR / "{}_masks.npy".format(label), masks[:, :, :N_COMPONENTS].astype(np.float32))
        np.save(OUTPUT_DIR / "{}_varexp.npy".format(label), varexp.astype(np.float32))

        yb, xb = fm["yrange_bin"], fm["xrange_bin"]
        summary["n_frames"] = int(pcs.shape[0])
        summary["rois"][label] = {
            "roi": {key: r[key] for key in ("x", "y", "w", "h")},
            "roi_pixels_used": {"x": [int(xb[0] * SBIN), int((xb[-1] + 1) * SBIN)],
                                "y": [int(yb[0] * SBIN), int((yb[-1] + 1) * SBIN)]},
            "mask_shape_binned": [int(yb.size), int(xb.size), N_COMPONENTS],
            "variance_explained_top5": [round(float(v), 4) for v in varexp[:5]],
            "variance_explained_cumulative_10": round(float(varexp[:10].sum()), 4),
            "variance_explained_cumulative_20": round(float(varexp[:20].sum()), 4),
            "variance_explained_cumulative_100": round(float(varexp.sum()), 4),
        }
        print("  {}: PCs {}, masks {}, cumulative var 10 = {:.3f}, 20 = {:.3f}".format(
            label, pcs.shape, tuple(summary["rois"][label]["mask_shape_binned"]),
            varexp[:10].sum(), varexp[:20].sum()))

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  summary.json")


if __name__ == "__main__":
    main()
