"""Pipeline stage 3: raw ROI motion energy around each trial, leave vs stay.

The direct equivalent of the lick-aligned 200 ms window of Cazettes et al.: for each
ROI, the mean absolute frame-to-frame pixel difference of face.avi, over the ROI's
pixels, in a 200 ms window centred on each trial's timestamp. This uses the raw
pixels of face.avi, not the FaceMap output (which is only used to pick the ROIs).

Inputs (per session):
    raw_data/<session>/task_events.csv              trial log written by the task
    raw_data/<session>/recording_*.avi              video; its filename gives the start time
    raw_data/<session>/recording_*_frametimes.csv   per-frame times (optional, preferred)
    preprocessed/<session>/face.avi, rois.npz, summary.json (fps)
    facemap_output/<session>/<roi>_PCs.npy          ROIs are those in rois.npz with PCs here

Trial -> frame alignment, the choice of task run and the leave label are as in st4:
  * leave: the participant's choice on the next trial differs from this trial's.
  * The last trial has no next choice, so it has no label and is dropped.
  * Trials whose 200 ms window doesn't fall entirely within the video are dropped.

Saves to facemap_output/<session>/event_motion/:
    event_motion_spatial.png    per-pixel motion energy (all / stay / leave trials) on the mean
                                face, one row per ROI in SPATIAL_ROIS, ROI outlined in yellow
    event_motion_summary.json   per ROI: mean and SEM for leave and stay, n_leave, n_stay

The spatial maps don't use FaceMap, so the figure shows every ROI in SPATIAL_ROIS that
is in rois.npz, whether or not it has a _PCs.npy.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st3_event_motion.py --session session_YYYYMMDD_HHMMSS_<task>
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = "session_20260930_165233"   # same as st2

HALF_WINDOW_S = 0.1   # window = trial timestamp +/- 100 ms
PROGRESS_EVERY = 2000
SPATIAL_ROIS = ["whole_face", "upper_face", "lower_face", "upper_face_no_eyes"]   # figure rows
HEATMAP_ALPHA = 0.6
CROP_MARGIN_PX = 6     # figure cells show the ROIs' bounding box plus this margin


def video_start_time(session_dir):
    """Recording start from the video filename, recording_YYYY-MM-DD_HH-MM-SS.avi (as in st4)."""
    for video in session_dir.glob("*.avi"):
        m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", video.name)
        if m:
            return pd.Timestamp(datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")), video.name
    sys.exit("ERROR: no video with a start timestamp in its filename in {}".format(session_dir))


def load_frame_times(session_dir, video_name):
    """Per-frame wall-clock times from the recorder's sidecar, or None (as in st4)."""
    path = session_dir / (Path(video_name).stem + "_frametimes.csv")
    if not path.is_file():
        return None
    return pd.to_datetime(pd.read_csv(path)["wall_clock_timestamp"])


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


def save_figure(fig, path, attempts=5):
    """Save via a temp file + rename. On Windows an open image preview can briefly
    hold the target, which makes a direct save fail (Errno 22)."""
    tmp = path.with_name(path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(sem)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION,
                        help="session folder name (or path) in raw_data/")
    SESSION = Path(parser.parse_args().session).name
    SESSION_DIR = PROJECT_DIR / "raw_data" / SESSION
    TASK_CSV = SESSION_DIR / "task_events.csv"
    PRE_DIR = PROJECT_DIR / "preprocessed" / SESSION
    FACE_VIDEO = PRE_DIR / "face.avi"
    ROIS_PATH = PRE_DIR / "rois.npz"
    PRE_SUMMARY = PRE_DIR / "summary.json"
    FACEMAP_DIR = PROJECT_DIR / "facemap_output" / SESSION
    OUTPUT_DIR = FACEMAP_DIR / "event_motion"

    # --- Step 1: load --------------------------------------------------------------
    for path in (TASK_CSV, FACE_VIDEO, ROIS_PATH, PRE_SUMMARY):
        if not path.is_file():
            sys.exit("ERROR: missing input: {}".format(path))
    all_masks = dict(np.load(ROIS_PATH))
    roi_names = [n for n in all_masks if (FACEMAP_DIR / "{}_PCs.npy".format(n)).is_file()]
    skipped = [n for n in all_masks if n not in roi_names]
    if not roi_names:
        sys.exit("ERROR: no ROI in {} has a <roi>_PCs.npy in {} (run st2 first)".format(
            ROIS_PATH, FACEMAP_DIR))
    print("ROIs: {}".format(", ".join(roi_names)))
    if skipped:
        print("Skipped (no _PCs.npy in facemap_output): {}".format(", ".join(skipped)))
    with open(PRE_SUMMARY) as f:
        fps = json.load(f)["fps"]

    df_all = pd.read_csv(TASK_CSV)
    # Sessions recorded before the schema was standardised use the old column names
    df_all = df_all.rename(columns={"trial": "trial_number", "active_ufo": "active_site",
                                    "active_crystal": "active_site"})
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

    print("\nMotion energy from {}".format(FACE_VIDEO.relative_to(PROJECT_DIR)))
    motion = frame_motion(FACE_VIDEO, [all_masks[n] for n in roi_names])
    n_frames = motion.shape[1]
    for name in roi_names:
        n_pcs = np.load(FACEMAP_DIR / "{}_PCs.npy".format(name), mmap_mode="r").shape[0]
        if n_pcs != n_frames:
            print("WARNING: {}_PCs.npy has {} frames, face.avi {}".format(name, n_pcs, n_frames))

    video_start, video_name = video_start_time(SESSION_DIR)
    video_end = video_start + pd.Timedelta(seconds=n_frames / fps)

    # The CSV is appended to across task runs: keep the run that overlaps the video.
    runs = df_all.groupby("session_id")["timestamp"].agg(["min", "max", "size"])
    runs["overlap_s"] = [(min(r["max"], video_end) - max(r["min"], video_start)).total_seconds()
                         for _, r in runs.iterrows()]
    task_session = runs["overlap_s"].idxmax()
    if runs.loc[task_session, "overlap_s"] <= 0:
        sys.exit("ERROR: no task run in {} overlaps the video ({} to {})".format(
            TASK_CSV.name, video_start, video_end))
    df = df_all[df_all["session_id"] == task_session].reset_index(drop=True)
    print("\nUsing task run {} ({} trials)".format(task_session, len(df)))

    # --- Step 2: trials -> frames (as in st4) ---------------------------------------------
    # Preferred: exact per-frame times from the recording sidecar. Fallback: the
    # filename start time + a constant fps (1 s precision).
    frame_times = load_frame_times(SESSION_DIR, video_name)
    if frame_times is not None and len(frame_times) != n_frames:
        print("WARNING: {} frame times for {} frames; falling back to filename timing.".format(
            len(frame_times), n_frames))
        frame_times = None
    if frame_times is not None:
        ft_ns = frame_times.values.astype("datetime64[ns]").astype(np.int64)
        video_start = frame_times.iloc[0]
        timing_source = "per-frame timestamps ({})".format(Path(video_name).stem + "_frametimes.csv")

        def to_frame(times):
            """Nearest frame to each time (NaN stays NaN); -1 / n_frames if outside the video."""
            t = times.values.astype("datetime64[ns]").astype(np.int64)
            out = np.full(len(t), np.nan)
            ok = times.notna().values
            idx = np.clip(np.searchsorted(ft_ns, t[ok]), 1, n_frames - 1)
            nearest = np.where(t[ok] - ft_ns[idx - 1] < ft_ns[idx] - t[ok], idx - 1, idx)
            nearest[t[ok] < ft_ns[0]] = -1
            nearest[t[ok] > ft_ns[-1]] = n_frames
            out[ok] = nearest
            return out
    else:
        timing_source = "video filename (1 s precision) + constant fps"

        def to_frame(times):
            return ((times - video_start).dt.total_seconds() * fps).round().values

    df["frame_idx"] = to_frame(df["timestamp"]).astype(int)   # unclipped: outside = < 0 or >= n_frames
    print("Video {}: start {}, {} frames at {} fps".format(video_name, video_start, n_frames, fps))
    print("Timing source: {}".format(timing_source))

    # --- Step 3: labels (participant's choices, as in st4) ---------------------------------
    df["leave"] = (df["choice"].shift(-1) != df["choice"]).astype(int)
    df = df.iloc[:-1].copy()   # last trial: next choice unknown -> no label

    # --- Step 4: per-trial window motion energy --------------------------------------------
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

    # --- Step 5: leave vs stay ---------------------------------------------------------------
    is_leave = df["leave"].values == 1
    n_leave, n_stay = int(is_leave.sum()), int((~is_leave).sum())
    results = {}
    for k, name in enumerate(roi_names):
        lm, ls = mean_sem(trial_motion[is_leave, k])
        sm, ss = mean_sem(trial_motion[~is_leave, k])
        results[name] = {"leave": {"mean": lm, "sem": ls}, "stay": {"mean": sm, "sem": ss},
                         "n_leave": n_leave, "n_stay": n_stay}

    # --- Step 6: save ---------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_masks = {n: all_masks[n] for n in SPATIAL_ROIS if n in all_masks}
    missing = [n for n in SPATIAL_ROIS if n not in all_masks]
    if missing:
        print("WARNING: not in rois.npz, left out of the figure: {}".format(", ".join(missing)))
    print("\nSpatial maps from {}".format(FACE_VIDEO.relative_to(PROJECT_DIR)))
    windows = np.stack([start[inside], end[inside]], axis=1)
    maps, mean_face = window_motion_maps(FACE_VIDEO, windows,
                                         {"all": np.ones(len(df), bool), "stay": ~is_leave, "leave": is_leave})
    columns = [("all", "All trials (n={})".format(len(df))), ("stay", "Stay (n={})".format(n_stay)),
               ("leave", "Leave (n={})".format(n_leave))]
    vmax = plot_spatial(maps, mean_face, spatial_masks, columns, SESSION,
                        OUTPUT_DIR / "event_motion_spatial.png")
    summary = {
        "session_id": SESSION.removeprefix("session_"),
        "task_session_id": task_session,
        "input_video": str(FACE_VIDEO.relative_to(PROJECT_DIR)),
        "motion_energy": "mean |frame i - frame i-1| over ROI pixels (grey levels), averaged over "
                         "the frame pairs in the window",
        "label": "leave = participant's next choice differs from this trial's choice",
        "window_ms": [-HALF_WINDOW_S * 1000, HALF_WINDOW_S * 1000],
        "window_frames": [-half, half],
        "fps": fps,
        "timing_source": timing_source,
        "video_start": str(video_start),
        "trials_dropped_window_outside_video": n_dropped,
        "rois": results,
    }
    with open(OUTPUT_DIR / "event_motion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to {}:\n  event_motion_spatial.png (colour scale 0 to {:.2f}), "
          "event_motion_summary.json".format(OUTPUT_DIR, vmax))
    print("\nMotion energy, +/-{:.0f} ms around each trial (mean +/- SEM; leave n = {}, stay n = {}):".format(
        HALF_WINDOW_S * 1000, n_leave, n_stay))
    for name, r in results.items():
        print("  {:22s} leave {:.4f} +/- {:.4f}   stay {:.4f} +/- {:.4f}   leave/stay {:.3f}".format(
            name, r["leave"]["mean"], r["leave"]["sem"], r["stay"]["mean"], r["stay"]["sem"],
            r["leave"]["mean"] / r["stay"]["mean"]))


if __name__ == "__main__":
    main()
