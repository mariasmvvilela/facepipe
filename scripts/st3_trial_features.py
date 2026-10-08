"""Pipeline stage 3: align face PCs to task trials and build accumulated per-visit features.

Builds features for every ROI in roi_definitions.ROIS that has FaceMap PCs, so stage 4
can compare any set of ROIs by picking columns, without rebuilding features.

Inputs (per session):
    raw_data/<session>/task_events.csv, recording_*.avi, recording_*_frametimes.csv
    preprocessed/<session>/summary.json             fps
    facemap_output/<session>/<roi>_PCs.npy          stage 2

Outputs in trial_data/<session>/:
    X_features.npy            (n_trials, n_features) float32, each column z-scored over the session
    y_labels.npy              (n_trials,) int, 1 = participant leaves after this trial, 0 = stays
    feature_names.txt         one feature name per line
    trial_metadata.csv        trial_number, choice, outcome, leave, site_visit, trial_in_visit, frame_idx
    features_summary.json     ROIs, alignment and window details

Leave labels and site visits: see trials.py.

Features, for trial i (1-indexed) of a site visit, using trials 1..i of that visit:
  * task: consecutive_failures (failure run ending at i, reset by any reward),
    reward_rate (rewards / i).
  * face, per ROI and per PC k = 1..N_PCS, from each trial's window-mean PC value:
    <roi>_PC<k>_mean (mean over 1..i), _last3 (mean over the last min(3, i) trials),
    _slope (linear-fit slope over 1..i; 0 if i < 3).
The window is the trial timestamp +/- HALF_WINDOW_S, cut short where the participant
re-aims before the next firing, so the leaving movement itself doesn't leak in.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st3_trial_features.py --session session_YYYYMMDD_HHMMSS_<task>
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import FACEMAP_DIR, PREPROCESSED_DIR, PROJECT_DIR, TRIAL_DIR, session_name
from roi_definitions import ROIS
from trials import add_labels, align_trials

N_PCS = 6
FACE_STATS = ["mean", "last3", "slope"]
TASK_FEATURES = ["consecutive_failures", "reward_rate"]
HALF_WINDOW_S = 0.25   # feature window = trial timestamp +/- 250 ms, cut at the next aim change


def load_pcs(session):
    """{roi: (n_frames, N_PCS)} for every ROI in ROIS with a _PCs.npy, in ROIS order."""
    roi_pcs, missing = {}, []
    for roi in ROIS:
        path = FACEMAP_DIR / session / "{}_PCs.npy".format(roi)
        if path.is_file():
            roi_pcs[roi] = np.load(path)[:, :N_PCS]
        else:
            missing.append(roi)
    if not roi_pcs:
        sys.exit("ERROR: no ROI has a _PCs.npy in {} (run st2 first)".format(FACEMAP_DIR / session))
    if missing:
        print("No PCs (run st2 for them to include them): {}".format(", ".join(missing)))
    frame_counts = {roi: pcs.shape[0] for roi, pcs in roi_pcs.items()}
    if len(set(frame_counts.values())) > 1:
        sys.exit("ERROR: frame counts differ between ROIs: {}".format(frame_counts))
    return roi_pcs


def window_means(df, roi_pcs, to_frame, fps):
    """Per ROI, each trial's mean PC values over its window. Returns ({roi: (n_trials, N_PCS)},
    frame_idx per trial (clipped to the video), n trials outside the video, n windows cut short)."""
    n_frames = next(iter(roi_pcs.values())).shape[0]
    frame_raw = to_frame(df["timestamp"]).astype(int)
    n_out = int(((frame_raw < 0) | (frame_raw >= n_frames)).sum())
    frame_idx = np.clip(frame_raw, 0, n_frames - 1)

    # When the participant re-aimed before the next firing (next trial's timestamp minus its
    # time_on_choice_ms). Only a change after this firing matters; otherwise it's the aim
    # that was already held.
    nxt = df.shift(-1)
    aim_change = nxt["timestamp"] - pd.to_timedelta(nxt["time_on_choice_ms"], unit="ms")
    aim_change = aim_change.where(aim_change > df["timestamp"])
    aim_change_frame = to_frame(aim_change)

    half = int(round(HALF_WINDOW_S * fps))
    trial_pcs = {roi: np.zeros((len(df), N_PCS)) for roi in roi_pcs}
    n_truncated = 0
    for i, fidx in enumerate(frame_idx):
        start = max(0, fidx - half)
        end = min(n_frames, fidx + half + 1)
        if not np.isnan(aim_change_frame[i]) and aim_change_frame[i] < end:
            end = max(fidx + 1, int(aim_change_frame[i]))
            n_truncated += 1
        for roi, pcs in roi_pcs.items():
            trial_pcs[roi][i] = pcs[start:end].mean(axis=0)
    return trial_pcs, frame_idx, n_out, n_truncated


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
            row = [consec_failures, (outcomes[:i] == "reward").sum() / i]
            for pcs in trial_pcs.values():
                vals = pcs[idx[:i]]         # (i, N_PCS)
                mean = vals.mean(axis=0)
                last3 = vals[-3:].mean(axis=0)
                slope = (np.polyfit(np.arange(i), vals, 1)[0] if i >= 3 else np.zeros(N_PCS))
                row += np.stack([mean, last3, slope], axis=1).ravel().tolist()   # PC-major
            rows.append(row)
            positions.append(i)
    return np.array(rows, dtype=np.float64), names, np.array(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="session folder name (or path) in raw_data/")
    session = session_name(parser.parse_args().session)
    pre_summary_path = PREPROCESSED_DIR / session / "summary.json"
    if not pre_summary_path.is_file():
        sys.exit("ERROR: missing input (run st1 first): {}".format(pre_summary_path))
    fps = json.loads(pre_summary_path.read_text())["fps"]
    out_dir = TRIAL_DIR / session

    roi_pcs = load_pcs(session)
    n_frames = next(iter(roi_pcs.values())).shape[0]
    print("ROIs: {} ({} frames, {} PCs each)\n".format(", ".join(roi_pcs), n_frames, N_PCS))

    al = align_trials(session, n_frames, fps)
    df = al.df
    trial_pcs, df["frame_idx"], n_out, n_truncated = window_means(df, roi_pcs, al.to_frame, fps)
    print("Trials outside the video: {}; window +/-{} frames, {} cut short at an aim change".format(
        n_out, int(round(HALF_WINDOW_S * fps)), n_truncated))

    df = add_labels(df)
    X_raw, feature_names, df["trial_in_visit"] = accumulated_features(df, trial_pcs)
    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    y = df["leave"].values.astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "y_labels.npy", y)
    (out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n")
    df[["trial_number", "choice", "outcome", "leave", "site_visit", "trial_in_visit",
        "frame_idx"]].to_csv(out_dir / "trial_metadata.csv", index=False)
    summary = {
        "session_id": session.removeprefix("session_"),
        "task_session_id": al.task_session,
        "n_trials": int(len(y)),
        "n_leave": int(y.sum()),
        "n_visits": int(df["site_visit"].nunique()),
        "rois": list(roi_pcs),
        "n_pcs_per_roi": N_PCS,
        "task_features": TASK_FEATURES,
        "face_stats": FACE_STATS,
        "n_features": int(X.shape[1]),
        "scaling": "each column z-scored over the session's labelled trials",
        "label": "leave = participant's next choice differs from this trial's choice",
        "fps": fps,
        "video_start": str(al.video_start),
        "timing_source": al.timing_source,
        "window_ms": [-HALF_WINDOW_S * 1000, HALF_WINDOW_S * 1000],
        "windows_truncated_at_aim_change": n_truncated,
        "trials_outside_video": n_out,
    }
    (out_dir / "features_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSaved to {}:".format(out_dir.relative_to(PROJECT_DIR)))
    print("  X_features.npy {} ({} task + {} face features), y_labels.npy, feature_names.txt, "
          "trial_metadata.csv, features_summary.json".format(
              X.shape, len(TASK_FEATURES), X.shape[1] - len(TASK_FEATURES)))
    print("  {} leave trials ({:.1%}), {} site visits".format(
        summary["n_leave"], y.mean(), summary["n_visits"]))
    print("\nNext step:\n  python scripts\\st4_classify_leave.py --session {}".format(session))


if __name__ == "__main__":
    main()
