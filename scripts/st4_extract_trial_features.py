"""Pipeline stage 4: align face PCs to task trials, build accumulated per-visit features
and compare leave-decision classifiers (task only / face only / task + face).

Inputs (per session):
    raw_data/<session>/task_events.csv              trial log written by the task (task/<task>.py)
    raw_data/<session>/recording_*.avi              video; its filename gives the start time
    raw_data/<session>/recording_*_frametimes.csv   per-frame times (optional, preferred)
    preprocessed/<session>/summary.json             fps, session id
    preprocessed/<session>/face.avi, rois.npz       diagnostic figure only
    facemap_output/<session>/<roi>_PCs.npy          for each ROI in ROIS that exists

Outputs in trial_data/<session>/:
    X_features.npy            (n_trials, n_features) float32, z-scored within the session
    y_labels.npy              (n_trials,) int, 1 = participant leaves after this trial, 0 = stays
    feature_names.txt         one feature name per line
    trial_metadata.csv        trial-level metadata
    summary.json              counts, ROIs used, cross-validated AUC per model
    event_motion_spatial.png  diagnostic: per-pixel motion energy (all / stay / leave), as in st3

Definitions (from the task code, not the column names):
  * leave: the participant's choice on the next trial differs from this trial's.
    `switch_occurred` is the task's hidden, random switch of the active site, which
    the participant never sees; it is NOT the label.
  * site visit: a run of consecutive trials with the same `choice` (where the
    participant is foraging), not a run of the hidden `active_site`.
  * The last trial has no next choice, so it has no label and is dropped.

Features, for trial i (1-indexed) of a site visit, using trials 1..i of that visit:
  * task: trial_in_visit (= i), consecutive_failures (failure run ending at i, reset by
    any reward), reward_rate (rewards / i).
  * face, per ROI and per PC k = 1..N_PCS, from each trial's window-mean PC value:
    <roi>_PC<k>_mean (mean over 1..i), _last3 (mean over the last min(3, i) trials),
    _slope (linear-fit slope over 1..i; 0 if i < 3).
All features are z-scored over the session's labelled trials.

Models: elastic-net logistic regression on task features (A), face features (B) and both
(C), scored by AUC-ROC with leave-one-visit-out cross-validation. Visits whose trials
are all one class (no AUC) are left out of the AUC mean, for all models alike.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st4_extract_trial_features.py --session session_YYYYMMDD_HHMMSS_<task>
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
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = "session_20260930_165233"   # same as st2

ROIS = ["whole_face", "upper_face", "lower_face", "upper_face_no_eyes"]
N_PCS = 6
FACE_STATS = ["mean", "last3", "slope"]
TASK_FEATURES = ["trial_in_visit", "consecutive_failures", "reward_rate"]
HALF_WINDOW_S = 0.25   # feature window = trial timestamp +/- 250 ms, cut at the next aim change

L1_RATIO = 0.5
MAX_ITER = 10000       # saga converges slowly
# sklearn >= 1.8 deprecates `penalty` (removed in 1.10): a float l1_ratio alone is elastic net.
SKLEARN_VERSION = tuple(int(p) for p in re.findall(r"\d+", sklearn.__version__)[:2])
PENALTY_KW = {} if SKLEARN_VERSION >= (1, 8) else {"penalty": "elasticnet"}

# Diagnostic figure, as in st3
MOTION_HALF_WINDOW_S = 0.1   # window = trial timestamp +/- 100 ms
PROGRESS_EVERY = 2000
HEATMAP_ALPHA = 0.6
CROP_MARGIN_PX = 6     # figure cells show the ROIs' bounding box plus this margin


def video_start_time(session_dir):
    """Recording start from the video filename, recording_YYYY-MM-DD_HH-MM-SS.avi.

    Only 1-second precision: alignment is uncertain by up to ~1 s (~30 frames).
    """
    for video in session_dir.glob("*.avi"):
        m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", video.name)
        if m:
            return pd.Timestamp(datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")), video.name
    sys.exit("ERROR: no video with a start timestamp in its filename in {}".format(session_dir))


def load_frame_times(session_dir, video_name):
    """Per-frame wall-clock times from the recorder's sidecar, or None."""
    path = session_dir / (Path(video_name).stem + "_frametimes.csv")
    if not path.is_file():
        return None
    return pd.to_datetime(pd.read_csv(path)["wall_clock_timestamp"])


def accumulated_features(df, trial_pcs):
    """One row per trial of df: task features, then per ROI / PC / stat face features.

    trial_pcs: {roi: (n_trials_total, N_PCS)} window means, indexed by df's index.
    Returns (rows (n_trials, n_features), feature names, trial_in_visit per trial).
    """
    names = list(TASK_FEATURES)
    for roi in trial_pcs:
        names += ["{}_PC{}_{}".format(roi, k + 1, stat) for k in range(N_PCS) for stat in FACE_STATS]
    rows, positions = [], []
    for _, visit_df in df.groupby("site_visit", sort=True):
        idx = visit_df.index.to_numpy()
        outcomes = visit_df["outcome"].to_numpy()
        consec_failures = 0
        for t in range(len(idx)):
            i = t + 1                       # position in the visit, 1-indexed
            consec_failures = consec_failures + 1 if outcomes[t] == "failure" else 0
            row = [i, consec_failures, (outcomes[:i] == "reward").sum() / i]
            for pcs in trial_pcs.values():
                vals = pcs[idx[:i]]         # (i, N_PCS)
                mean = vals.mean(axis=0)
                last3 = vals[-3:].mean(axis=0)
                slope = (np.polyfit(np.arange(i), vals, 1)[0] if i >= 3 else np.zeros(N_PCS))
                row += np.stack([mean, last3, slope], axis=1).ravel().tolist()   # PC-major
            rows.append(row)
            positions.append(i)
    return np.array(rows, dtype=np.float64), names, np.array(positions)


def leave_one_visit_out_auc(X, y, visits):
    """AUC on each held-out visit (model fit on all other visits); visits with one class skipped."""
    aucs = []
    for v in np.unique(visits):
        test = visits == v
        if len(np.unique(y[test])) < 2:
            continue
        model = LogisticRegression(solver="saga", l1_ratio=L1_RATIO, max_iter=MAX_ITER, **PENALTY_KW)
        model.fit(X[~test], y[~test])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return np.array(aucs)


def mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(sem)


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
            w[s + 1:e] += 1          # pair i = (frame i-1, frame i)
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


def plot_spatial(maps, mean_face, masks, columns, session, window_s, path):
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
        session, 2 * window_s * 1000), fontsize=12)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION,
                        help="session folder name (or path) in raw_data/")
    SESSION = Path(parser.parse_args().session).name
    SESSION_DIR = PROJECT_DIR / "raw_data" / SESSION
    TASK_CSV = SESSION_DIR / "task_events.csv"
    PRE_DIR = PROJECT_DIR / "preprocessed" / SESSION
    PRE_SUMMARY = PRE_DIR / "summary.json"
    FACE_VIDEO = PRE_DIR / "face.avi"
    ROIS_PATH = PRE_DIR / "rois.npz"
    FACEMAP_DIR = PROJECT_DIR / "facemap_output" / SESSION
    OUTPUT_DIR = PROJECT_DIR / "trial_data" / SESSION

    # --- Step 1: load --------------------------------------------------------------
    for path in (TASK_CSV, PRE_SUMMARY):
        if not path.is_file():
            sys.exit("ERROR: missing input: {}".format(path))
    with open(PRE_SUMMARY) as f:
        pre_summary = json.load(f)
    fps = pre_summary["fps"]

    roi_pcs = {}
    for roi in ROIS:
        path = FACEMAP_DIR / "{}_PCs.npy".format(roi)
        if path.is_file():
            roi_pcs[roi] = np.load(path)[:, :N_PCS]
        else:
            print("WARNING: no {} (run st2 for this ROI); skipping {}".format(
                path.relative_to(PROJECT_DIR), roi))
    if not roi_pcs:
        sys.exit("ERROR: none of {} has a _PCs.npy in {}".format(", ".join(ROIS), FACEMAP_DIR))
    frame_counts = {roi: pcs.shape[0] for roi, pcs in roi_pcs.items()}
    if len(set(frame_counts.values())) > 1:
        sys.exit("ERROR: frame counts differ between ROIs: {}".format(frame_counts))
    n_frames = next(iter(frame_counts.values()))
    print("ROIs used: {} ({} frames, {} PCs each)".format(", ".join(roi_pcs), n_frames, N_PCS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(TASK_CSV)
    # Sessions recorded before the schema was standardised use the old column names
    df_all = df_all.rename(columns={"trial": "trial_number", "active_ufo": "active_site",
                                    "active_crystal": "active_site"})
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

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

    print("\nTask runs in CSV:")
    print(runs.to_string())
    print("Using task run {} ({} trials)".format(task_session, len(df)))

    # --- Step 2: trials -> frames ------------------------------------------------------
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

    frame_raw = to_frame(df["timestamp"]).astype(int)   # unclipped: outside = < 0 or >= n_frames
    n_out = int(((frame_raw < 0) | (frame_raw >= n_frames)).sum())
    df["frame_idx"] = np.clip(frame_raw, 0, n_frames - 1)

    # When the participant re-aimed before the next firing (next trial's timestamp
    # minus its time_on_choice_ms). Windows stop there so the leaving movement
    # itself doesn't leak into the features.
    nxt = df.shift(-1)
    aim_change = nxt["timestamp"] - pd.to_timedelta(nxt["time_on_choice_ms"], unit="ms")
    # Only a change after this firing matters; otherwise it's the aim that was already held.
    aim_change = aim_change.where(aim_change > df["timestamp"])
    aim_change_frame = to_frame(aim_change)

    print("\nVideo {}: start {}, end {}".format(video_name, video_start, video_end))
    print("Timing source: {}".format(timing_source))
    print("frame_idx: min {}, max {}, mean {:.1f}; trials outside video: {}".format(
        df["frame_idx"].min(), df["frame_idx"].max(), df["frame_idx"].mean(), n_out))

    # --- Step 3: per-trial window means ---------------------------------------------------
    half = int(round(HALF_WINDOW_S * fps))
    trial_pcs = {roi: np.zeros((len(df), N_PCS)) for roi in roi_pcs}
    n_truncated = 0
    for i, fidx in enumerate(df["frame_idx"]):
        start = max(0, fidx - half)
        end = min(n_frames, fidx + half + 1)
        if not np.isnan(aim_change_frame[i]) and aim_change_frame[i] < end:
            end = max(fidx + 1, int(aim_change_frame[i]))
            n_truncated += 1
        for roi, pcs in roi_pcs.items():
            trial_pcs[roi][i] = pcs[start:end].mean(axis=0)
    print("Window: +/-{} frames; {} windows cut short at an aim change".format(half, n_truncated))

    # --- Labels and site visits (participant's choices) ----------------------------------
    df["leave"] = (df["choice"].shift(-1) != df["choice"]).astype(int)
    df["site_visit"] = (df["choice"] != df["choice"].shift()).cumsum()
    keep = df.index[:-1]   # last trial: next choice unknown -> no label
    df = df.loc[keep].copy()

    # --- Step 4: accumulated features within each visit, z-scored over the session --------
    X_raw, feature_names, trial_in_visit = accumulated_features(df, trial_pcs)
    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    y = df["leave"].values.astype(int)
    visits = df["site_visit"].values
    df["trial_in_visit"] = trial_in_visit
    task_cols = np.array([n in TASK_FEATURES for n in feature_names])

    # --- Step 5: save features ------------------------------------------------------------
    np.save(OUTPUT_DIR / "X_features.npy", X)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    (OUTPUT_DIR / "feature_names.txt").write_text("\n".join(feature_names) + "\n")
    df[["trial_number", "choice", "outcome", "leave", "site_visit", "trial_in_visit",
        "frame_idx"]].to_csv(OUTPUT_DIR / "trial_metadata.csv", index=False)
    print("\nFeatures: {} trials x {} features ({} task + {} face)".format(
        X.shape[0], X.shape[1], int(task_cols.sum()), int((~task_cols).sum())))

    # --- Step 6: diagnostic spatial motion figure (as in st3; not a model input) ----------
    figure_path = OUTPUT_DIR / "event_motion_spatial.png"
    if FACE_VIDEO.is_file() and ROIS_PATH.is_file():
        all_masks = dict(np.load(ROIS_PATH))
        spatial_masks = {n: all_masks[n] for n in ROIS if n in all_masks}
        missing = [n for n in ROIS if n not in all_masks]
        if missing:
            print("WARNING: not in rois.npz, left out of the figure: {}".format(", ".join(missing)))
        # Frames fidx-half_m .. fidx+half_m span 2 * half_m frame intervals = 200 ms; trials
        # whose window isn't entirely within the video are left out of the figure.
        half_m = int(round(MOTION_HALF_WINDOW_S * fps))
        m_start = frame_raw[keep] - half_m
        m_end = frame_raw[keep] + half_m + 1
        inside = (m_start >= 0) & (m_end <= n_frames)
        is_leave = y[inside] == 1
        n_leave_fig, n_stay_fig = int(is_leave.sum()), int((~is_leave).sum())
        print("\nSpatial maps from {} (+/-{} frames; {} labelled trials outside the video left out)".format(
            FACE_VIDEO.relative_to(PROJECT_DIR), half_m, int((~inside).sum())))
        maps, mean_face = window_motion_maps(
            FACE_VIDEO, np.stack([m_start[inside], m_end[inside]], axis=1),
            {"all": np.ones(len(is_leave), bool), "stay": ~is_leave, "leave": is_leave})
        columns = [("all", "All trials (n={})".format(len(is_leave))),
                   ("stay", "Stay (n={})".format(n_stay_fig)),
                   ("leave", "Leave (n={})".format(n_leave_fig))]
        plot_spatial(maps, mean_face, spatial_masks, columns, SESSION, MOTION_HALF_WINDOW_S, figure_path)
    else:
        print("WARNING: {} or {} missing; skipping event_motion_spatial.png".format(
            FACE_VIDEO.relative_to(PROJECT_DIR), ROIS_PATH.relative_to(PROJECT_DIR)))
        figure_path = None

    # --- Step 7: leave-one-visit-out elastic-net logistic regression -----------------------
    models = {
        "A": ("task only", task_cols),
        "B": ("face PCs only", ~task_cols),
        "C": ("task + face", np.ones(len(feature_names), bool)),
    }
    n_folds = len(np.unique(visits))
    print("\nFitting models: {} visits, leave-one-visit-out".format(n_folds))
    auc = {}
    for key, (label, cols) in models.items():
        fold_aucs = leave_one_visit_out_auc(X[:, cols], y, visits)
        m, s = mean_sem(fold_aucs)
        auc[key] = {"label": label, "n_features": int(cols.sum()), "mean": round(m, 4),
                    "sem": round(s, 4), "n_folds_scored": len(fold_aucs)}

    summary = {
        "session_id": pre_summary.get("session_id", SESSION.removeprefix("session_")),
        "task_session_id": task_session,
        "n_trials": int(len(y)),
        "n_leave": int(y.sum()),
        "leave_rate": round(float(y.mean()), 4),
        "n_visits": int(n_folds),
        "rois_used": list(roi_pcs),
        "n_pcs_per_roi": N_PCS,
        "n_features": int(X.shape[1]),
        "label": "leave = participant's next choice differs from this trial's choice",
        "model": "LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio={}), "
                 "leave-one-visit-out CV, AUC-ROC per held-out visit (single-class visits "
                 "not scored)".format(L1_RATIO),
        "auc": auc,
        "n_folds_total": int(n_folds),
        "fps": fps,
        "video_start": str(video_start),
        "timing_source": timing_source,
        "window_ms": [-HALF_WINDOW_S * 1000, HALF_WINDOW_S * 1000],
        "windows_truncated_at_aim_change": n_truncated,
        "trials_outside_video": n_out,
        "diagnostic_figure_window_ms": [-MOTION_HALF_WINDOW_S * 1000, MOTION_HALF_WINDOW_S * 1000],
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to {}:".format(OUTPUT_DIR))
    print("  X_features.npy {}, y_labels.npy {}, feature_names.txt, trial_metadata.csv, summary.json{}".format(
        X.shape, y.shape, ", event_motion_spatial.png" if figure_path else ""))
    print("  {} leave trials ({:.1%}), {} site visits; AUC from {} of {} visits (others have one class)".format(
        summary["n_leave"], summary["leave_rate"], n_folds, auc["A"]["n_folds_scored"], n_folds))
    print()
    for key, header in (("A", "Model A (task only):"), ("B", "Model B (face PCs only):"),
                        ("C", "Model C (task + face):")):
        print("{:26s}AUC = {:.3f} ± {:.3f}".format(header, auc[key]["mean"], auc[key]["sem"]))


if __name__ == "__main__":
    main()
