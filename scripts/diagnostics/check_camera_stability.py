"""Diagnostic: how still is the face in the raw video (head-mounted camera check).

With a head-mounted camera the face should sit at a fixed place in the frame, in
which case stage 2's per-frame warp is unnecessary (it would only add MediaPipe
landmark jitter) and a fixed crop + fixed face-oval mask is enough. This measures
face motion in the image two independent ways:

  1. MediaPipe: similarity fit of 4 rigid anchors (inner eye corners, nose
     bridge, nose tip) of every frame onto the session median -> translation,
     rotation, scale over time. Includes MediaPipe's own landmark noise.
  2. Image registration: phase correlation of a grayscale patch (eyes to nose
     tip) against a reference frame. Landmark-free, so it shows whether the
     motion in (1) is real camera slip or landmark jitter.

Everything is reported in pixels of the 256x256 crop the pipeline uses
(FaceMap bins it by 4, so drifts well under 4 px are below its resolution).

Writes to mediapipe_output/<session>/:
    camera_stability.png    time courses + summary
    camera_stability.json   metrics

Usage (inside the facepipe env, from the project folder):
    python scripts\\diagnostics\\check_camera_stability.py --session raw_video\\session_YYYYMMDD_HHMMSS
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LANDMARK_ROOT = PROJECT_DIR / "mediapipe_output"

ANCHOR_IDX = [133, 362, 168, 1]     # same anchors as st2_stabilise_video.py
CROP_SIZE = 256
FACE_CROP_MARGIN = 0.8              # half-width of the crop, x face extent (as in st2)
PROGRESS_EVERY = 1000


def find_video(session_dir):
    videos = sorted(session_dir.glob("*.avi"))
    if len(videos) != 1:
        sys.exit("ERROR: expected exactly one .avi in {}, found {}".format(
            session_dir, [v.name for v in videos]))
    return videos[0]


def similarity_params(M):
    """2x3 similarity matrix -> (tx, ty, rotation deg, scale)."""
    scale = np.hypot(M[0, 0], M[1, 0])
    return M[0, 2], M[1, 2], np.degrees(np.arctan2(M[1, 0], M[0, 0])), scale


def stats(x):
    x = x[np.isfinite(x)]
    return {"std": round(float(np.std(x)), 3),
            "p1_p99_range": round(float(np.percentile(x, 99) - np.percentile(x, 1)), 3),
            "max_abs_from_median": round(float(np.max(np.abs(x - np.median(x)))), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, required=True, help="session folder in raw_video/")
    args = parser.parse_args()

    session_dir = args.session if args.session.is_absolute() else PROJECT_DIR / args.session
    lm_dir = LANDMARK_ROOT / session_dir.name
    if not (lm_dir / "landmarks.npy").is_file():
        sys.exit("ERROR: run st1_mediapipe_landmarks.py first: {}".format(lm_dir))
    video_path = find_video(session_dir)
    landmarks = np.load(lm_dir / "landmarks.npy")
    detected = np.load(lm_dir / "detection_mask.npy")
    with open(lm_dir / "summary.json") as f:
        fps = json.load(f)["fps"]

    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    px = np.array([width, height], dtype=np.float32)
    n = len(landmarks)

    # Fixed crop around the median face, and its scale to the 256 crop.
    all_px = landmarks[:, :, :2] * px
    lo = np.nanmedian(np.nanmin(all_px, axis=1), axis=0)
    hi = np.nanmedian(np.nanmax(all_px, axis=1), axis=0)
    half = FACE_CROP_MARGIN * (hi - lo).max()
    to_crop = CROP_SIZE / (2 * half)

    # --- 1. MediaPipe anchors -> similarity transform onto the median -------------------
    anchors = landmarks[:, ANCHOR_IDX, :2] * px
    ref = np.nanmedian(anchors, axis=0).astype(np.float32)
    params = np.full((n, 4), np.nan)
    residual = np.full(n, np.nan)
    for i in np.flatnonzero(detected):
        M, _ = cv2.estimateAffinePartial2D(anchors[i].astype(np.float32), ref,
                                           ransacReprojThreshold=1000.0)
        if M is not None:
            params[i] = similarity_params(M)
            fit = anchors[i] @ M[:, :2].T + M[:, 2]
            residual[i] = np.linalg.norm(fit - ref, axis=1).mean()
    # Translation is how far the frame must move to match the median: flip sign so
    # it's the face's displacement. Convert px -> 256-crop px.
    lm_dx, lm_dy = -params[:, 0] * to_crop, -params[:, 1] * to_crop
    lm_rot = params[:, 2]
    lm_scale_pct = (params[:, 3] - 1) * 100
    residual_crop = residual * to_crop

    # --- 2. Phase correlation of the eyes-to-nose patch against a reference --------------
    eyes = ref[:2].mean(axis=0)
    eye_dist = np.linalg.norm(ref[0] - ref[1])
    pw = int(1.6 * eye_dist)                       # patch spans past both eyes
    x0 = int(eyes[0] - pw)
    y0 = int(eyes[1] - 0.6 * eye_dist)             # brows ...
    y1 = int(ref[3][1] + 0.1 * eye_dist)           # ... to nose tip (mouth/jaw excluded)
    x1 = int(eyes[0] + pw)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_64F)

    ref_frame = int(np.nanargmin(np.abs(lm_dx - np.nanmedian(lm_dx))
                                 + np.abs(lm_dy - np.nanmedian(lm_dy))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, f = cap.read()
    ref_patch = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float64)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    reg = np.full((n, 2), np.nan)
    reg_response = np.full(n, np.nan)
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        patch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float64)
        (dx, dy), resp = cv2.phaseCorrelate(ref_patch, patch, win)
        reg[i] = dx, dy
        reg_response[i] = resp
        i += 1
        if i % PROGRESS_EVERY == 0:
            print("registration {}/{}".format(i, n), flush=True)
    cap.release()
    reg_dx, reg_dy = reg[:, 0] * to_crop, reg[:, 1] * to_crop

    # --- Report -----------------------------------------------------------------------
    metrics = {
        "session": session_dir.name,
        "video": video_path.name,
        "n_frames": n,
        "detection_rate": round(float(detected.mean()), 4),
        "units": "pixels of the {0}x{0} crop (1 crop px = {1:.2f} raw px); FaceMap sbin=4".format(
            CROP_SIZE, 1 / to_crop),
        "crop_box_raw_px": [round(float(v), 1) for v in
                            ((lo + hi) / 2 - half).tolist() + ((lo + hi) / 2 + half).tolist()],
        "mediapipe_anchor_fit": {
            "dx": stats(lm_dx), "dy": stats(lm_dy),
            "rotation_deg": stats(lm_rot), "scale_pct": stats(lm_scale_pct),
            "residual_mean_px": round(float(np.nanmean(residual_crop)), 3),
        },
        "image_registration": {
            "patch_raw_px": [x0, y0, x1, y1],
            "reference_frame": ref_frame,
            "dx": stats(reg_dx), "dy": stats(reg_dy),
            "response_median": round(float(np.nanmedian(reg_response)), 3),
        },
        "frame_to_frame_jitter_px": {
            "mediapipe_dx": round(float(np.nanstd(np.diff(lm_dx))), 3),
            "mediapipe_dy": round(float(np.nanstd(np.diff(lm_dy))), 3),
            "registration_dx": round(float(np.nanstd(np.diff(reg_dx))), 3),
            "registration_dy": round(float(np.nanstd(np.diff(reg_dy))), 3),
        },
    }
    out_dir = lm_dir
    with open(out_dir / "camera_stability.json", "w") as f:
        json.dump(metrics, f, indent=2)

    t = np.arange(n) / fps
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for ax, (lm, rg, name) in zip(axes[:2], ((lm_dx, reg_dx, "x"), (lm_dy, reg_dy, "y"))):
        ax.plot(t, lm - np.nanmedian(lm), lw=0.5, label="MediaPipe anchors")
        ax.plot(t, rg - np.nanmedian(rg), lw=0.8, label="image registration")
        ax.axhspan(-2, 2, color="0.9", zorder=0, label="+/-2 px (half a FaceMap bin)")
        ax.set_ylabel("face {} shift\n(crop px)".format(name))
        ax.legend(loc="upper right", fontsize=8)
    axes[2].plot(t, lm_rot - np.nanmedian(lm_rot), lw=0.5)
    axes[2].set_ylabel("rotation (deg)")
    axes[3].plot(t, lm_scale_pct - np.nanmedian(lm_scale_pct), lw=0.5)
    axes[3].set_ylabel("scale (%)")
    axes[3].set_xlabel("time (s)")
    fig.suptitle("{}: face position in frame (detection {:.1%})".format(
        session_dir.name, detected.mean()))
    fig.tight_layout()
    fig.savefig(out_dir / "camera_stability.png", dpi=110)

    print(json.dumps(metrics, indent=2))
    print("Saved camera_stability.png / .json to {}".format(out_dir))


if __name__ == "__main__":
    main()
