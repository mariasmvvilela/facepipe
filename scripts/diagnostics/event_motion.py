"""Diagnostic: raw ROI motion energy around each trial, leave vs stay.

The direct equivalent of the lick-aligned 200 ms window of Cazettes et al.: for each
ROI, the mean absolute frame-to-frame pixel difference of face.avi, over the ROI's
pixels, in a 200 ms window centred on each trial's timestamp. Uses the raw pixels of
face.avi, not FaceMap, so it checks the PCs' leave signal against plain motion.
Trial alignment and leave labels are those of the pipeline (scripts/trials.py); trials
whose window doesn't fall entirely within the video are dropped.

Inputs:  raw_data/<session>/ (task events, frametimes), preprocessed/<session>/face.avi,
         rois.npz, summary.json
Saves to facemap_output/<session>/event_motion/:
    event_motion_spatial.png    per-pixel motion energy (all / stay / leave trials) on the
                                mean face, one row per ROI in rois.npz, ROI outlined in yellow
    event_motion_summary.json   per ROI: mean and SEM for leave and stay, n_leave, n_stay

Usage (inside the facepipe env, from the project folder):
    python scripts\\diagnostics\\event_motion.py --session session_YYYYMMDD_HHMMSS_<task>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline modules in scripts/
from common import FACEMAP_DIR, PREPROCESSED_DIR, PROJECT_DIR, mean_sem, save_figure, session_name  # noqa: E402
from trials import add_labels, align_trials  # noqa: E402

HALF_WINDOW_S = 0.1   # window = trial timestamp +/- 100 ms
PROGRESS_EVERY = 2000
HEATMAP_ALPHA = 0.6
CROP_MARGIN_PX = 6     # figure cells show the ROIs' bounding box plus this margin


def frame_motion(face_video, masks):
    """Per ROI, the mean |frame i - frame i-1| over the ROI's pixels, for every frame.

    Returns (n_rois, n_frames) float64; column 0 (no previous frame) is NaN.
    """
    cap = cv2.VideoCapture(str(face_video))
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video: {}".format(face_video))
    motion, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is None:
            motion.append(np.full(len(masks), np.nan))
        else:
            diff = np.abs(gray - prev)
            motion.append([diff[m].mean() for m in masks])
        prev = gray
        if len(motion) % PROGRESS_EVERY == 0:
            print("  motion energy {} frames".format(len(motion)), flush=True)
    cap.release()
    return np.array(motion).T


def window_motion_maps(face_video, windows, groups):
    """Per group, the mean over trials of each trial's per-pixel window motion, and the mean face.

    windows: (n_trials, 2) frame [start, end) per trial, all the same length.
    groups: {name: bool array over trials}.
    Every window has the same number of frame pairs, so the mean of the trials' window
    means equals a weighted sum of |frame i - frame i-1|, weighted by the number of the
    group's windows containing pair i. One pass, no per-frame images kept in memory.
    Returns ({name: (H, W) float64}, mean face (H, W) float64).
    """
    n_pairs = int(windows[0, 1] - windows[0, 0] - 1)
    n_frames_needed = int(windows[:, 1].max())
    weights = {}
    for name, sel in groups.items():
        w = np.zeros(n_frames_needed)
        for s, e in windows[sel]:
            w[s + 1:e] += 1          # pair i = (frame i-1, frame i), as in frame_motion
        weights[name] = w
    cap = cv2.VideoCapture(str(face_video))
    acc, face, prev, i = None, None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if acc is None:
            acc = {name: np.zeros_like(gray) for name in groups}
            face = np.zeros_like(gray)
        face += gray
        if prev is not None and i < n_frames_needed:
            diff = np.abs(gray - prev)
            for name, w in weights.items():
                if w[i]:
                    acc[name] += w[i] * diff
        prev = gray
        i += 1
        if i % PROGRESS_EVERY == 0:
            print("  spatial maps {} frames".format(i), flush=True)
    cap.release()
    maps = {name: acc[name] / (groups[name].sum() * n_pairs) for name in groups}
    return maps, face / i


def plot_spatial(maps, mean_face, masks, columns, session, path):
    """Rows = ROIs, columns = trial groups; heatmap inside each ROI over the mean face."""
    union = np.any(list(masks.values()), axis=0)
    ys, xs = np.nonzero(union)
    y0, y1 = max(0, ys.min() - CROP_MARGIN_PX), min(union.shape[0], ys.max() + 1 + CROP_MARGIN_PX)
    x0, x1 = max(0, xs.min() - CROP_MARGIN_PX), min(union.shape[1], xs.max() + 1 + CROP_MARGIN_PX)
    # One scale for all cells: 99th percentile over the ROI pixels of every cell.
    vmax = np.percentile(np.concatenate([maps[g][m] for m in masks.values() for g, _ in columns]), 99)
    n_rows, n_cols = len(masks), len(columns)
    cell = 3.2
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="constrained", figsize=(
        n_cols * cell * (x1 - x0) / (y1 - y0) + 1.6, n_rows * cell + 1.0))
    for r, (roi, m) in enumerate(masks.items()):
        for c, (group, header) in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(mean_face[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(~m, maps[group])[y0:y1, x0:x1], cmap="Reds",
                      vmin=0, vmax=vmax, alpha=HEATMAP_ALPHA)
            ax.contour(m[y0:y1, x0:x1], levels=[0.5], colors="yellow", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(header, fontsize=11)
            if c == 0:
                ax.set_ylabel(roi, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=matplotlib.colors.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02,
                 label="mean |frame diff| (grey levels)")
    fig.suptitle("{}: mean motion energy per pixel, {:.0f}ms window around each trial".format(
        session, 2 * HALF_WINDOW_S * 1000), fontsize=12)
    save_figure(fig, path)
    return float(vmax)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="session folder name (or path) in raw_data/")
    session = session_name(parser.parse_args().session)
    pre_dir = PREPROCESSED_DIR / session
    face_video, rois_path, pre_summary = pre_dir / "face.avi", pre_dir / "rois.npz", pre_dir / "summary.json"
    out_dir = FACEMAP_DIR / session / "event_motion"
    for path in (face_video, rois_path, pre_summary):
        if not path.is_file():
            sys.exit("ERROR: missing input (run st1 first): {}".format(path))
    masks = dict(np.load(rois_path))
    fps = json.loads(pre_summary.read_text())["fps"]
    print("ROIs: {}".format(", ".join(masks)))

    print("\nMotion energy from {}".format(face_video.relative_to(PROJECT_DIR)))
    motion = frame_motion(face_video, list(masks.values()))
    n_frames = motion.shape[1]

    al = align_trials(session, n_frames, fps)
    df = al.df
    df["frame_idx"] = al.to_frame(df["timestamp"]).astype(int)   # unclipped: outside = < 0 or >= n_frames
    df = add_labels(df)

    # Frames fidx-half .. fidx+half span 2 * half frame intervals = 200 ms. The diffs inside
    # the window are motion[:, start+1 .. end-1] (motion[:, i] = |frame i - frame i-1|).
    half = int(round(HALF_WINDOW_S * fps))
    start = df["frame_idx"].values - half
    end = df["frame_idx"].values + half + 1
    inside = (start >= 0) & (end <= n_frames)
    n_dropped = int((~inside).sum())
    df = df[inside].copy()
    trial_motion = np.array([np.nanmean(motion[:, s + 1:e], axis=1)
                             for s, e in zip(start[inside], end[inside])])   # (n_trials, n_rois)
    print("Window: +/-{} frames ({:.0f} ms); {} labelled trials dropped (window outside the video)".format(
        half, 2 * half / fps * 1000, n_dropped))

    is_leave = df["leave"].values == 1
    n_leave, n_stay = int(is_leave.sum()), int((~is_leave).sum())
    results = {}
    for k, name in enumerate(masks):
        lm, ls = mean_sem(trial_motion[is_leave, k])
        sm, ss = mean_sem(trial_motion[~is_leave, k])
        results[name] = {"leave": {"mean": lm, "sem": ls}, "stay": {"mean": sm, "sem": ss},
                         "n_leave": n_leave, "n_stay": n_stay}

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nSpatial maps from {}".format(face_video.relative_to(PROJECT_DIR)))
    windows = np.stack([start[inside], end[inside]], axis=1)
    maps, mean_face = window_motion_maps(face_video, windows,
                                         {"all": np.ones(len(df), bool), "stay": ~is_leave, "leave": is_leave})
    columns = [("all", "All trials (n={})".format(len(df))), ("stay", "Stay (n={})".format(n_stay)),
               ("leave", "Leave (n={})".format(n_leave))]
    vmax = plot_spatial(maps, mean_face, masks, columns, session, out_dir / "event_motion_spatial.png")
    summary = {
        "session_id": session.removeprefix("session_"),
        "task_session_id": al.task_session,
        "input_video": str(face_video.relative_to(PROJECT_DIR)),
        "motion_energy": "mean |frame i - frame i-1| over ROI pixels (grey levels), averaged over "
                         "the frame pairs in the window",
        "label": "leave = participant's next choice differs from this trial's choice",
        "window_ms": [-HALF_WINDOW_S * 1000, HALF_WINDOW_S * 1000],
        "window_frames": [-half, half],
        "fps": fps,
        "timing_source": al.timing_source,
        "video_start": str(al.video_start),
        "trials_dropped_window_outside_video": n_dropped,
        "rois": results,
    }
    (out_dir / "event_motion_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSaved to {}:\n  event_motion_spatial.png (colour scale 0 to {:.2f}), "
          "event_motion_summary.json".format(out_dir.relative_to(PROJECT_DIR), vmax))
    print("\nMotion energy, +/-{:.0f} ms around each trial (mean +/- SEM; leave n = {}, stay n = {}):".format(
        HALF_WINDOW_S * 1000, n_leave, n_stay))
    for name, r in results.items():
        print("  {:26s} leave {:.4f} +/- {:.4f}   stay {:.4f} +/- {:.4f}   leave/stay {:.3f}".format(
            name, r["leave"]["mean"], r["leave"]["sem"], r["stay"]["mean"], r["stay"]["sem"],
            r["leave"]["mean"] / r["stay"]["mean"]))


if __name__ == "__main__":
    main()
