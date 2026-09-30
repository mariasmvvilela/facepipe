"""Pipeline stage 4: align face features to task trials and build the accumulated
feature matrix for leave-decision classification.

Inputs (per session):
    raw_video/<session>/task_events.csv          trial log from task/alien_energy_forager.py
    raw_video/<session>/recording_*.avi          video; its filename gives the start time
    facemap_output/<session>/{eyes_brows,lower_face}_PCs.npy
    mediapipe_output/<session>/head_pose.npy, summary.json (fps)

Outputs in trial_data/<session>/:
    X_features.npy      (n_trials, n_features) float32
    y_labels.npy        (n_trials,) int, 1 = participant leaves after this trial, 0 = stays
    feature_names.txt   one feature name per line
    trial_metadata.csv  trial-level metadata
    summary.json        counts, feature count, leave rate

Definitions (from the task code, not the column names):
  * leave: the participant's choice on the next trial differs from this trial's.
    `switch_occurred` is the task's hidden, random crystal depletion, which the
    participant never sees; it's kept in the metadata but is NOT the label.
  * site visit: a run of consecutive trials with the same `choice` (where the
    participant is foraging), not a run of the hidden `active_crystal`.
  * The last trial has no next choice, so it has no label and is dropped.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st4_extract_trial_features.py
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
SESSION = "session_20260928_161612"
SESSION_DIR = PROJECT_DIR / "raw_video" / SESSION
TASK_CSV = SESSION_DIR / "task_events.csv"
EYES_PCS = PROJECT_DIR / "facemap_output" / SESSION / "eyes_brows_PCs.npy"
LOWER_PCS = PROJECT_DIR / "facemap_output" / SESSION / "lower_face_PCs.npy"
HEAD_POSE = PROJECT_DIR / "mediapipe_output" / SESSION / "head_pose.npy"
MEDIAPIPE_SUMMARY = PROJECT_DIR / "mediapipe_output" / SESSION / "summary.json"
OUTPUT_DIR = PROJECT_DIR / "trial_data" / SESSION

N_PCS = 10
HALF_WINDOW_S = 0.25   # window = trial timestamp +/- 250 ms, cut at the next aim change


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
    """Per-frame wall-clock times from record_session.py's sidecar, or None."""
    path = session_dir / (Path(video_name).stem + "_frametimes.csv")
    if not path.is_file():
        return None
    return pd.to_datetime(pd.read_csv(path)["wall_clock_timestamp"])


def window_mean(values, start, end):
    return values[start:end].mean(axis=0)


def main():
    # --- Step 1: load --------------------------------------------------------------
    for path in (TASK_CSV, EYES_PCS, LOWER_PCS, HEAD_POSE, MEDIAPIPE_SUMMARY):
        if not path.is_file():
            sys.exit("ERROR: missing input: {}".format(path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(TASK_CSV)
    # task/space_shooter.py calls the hidden active site active_ufo
    df_all = df_all.rename(columns={"active_ufo": "active_crystal"})
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    eyes_pcs = np.load(EYES_PCS)[:, :N_PCS]
    lower_pcs = np.load(LOWER_PCS)[:, :N_PCS]
    head_pose = np.load(HEAD_POSE)
    with open(MEDIAPIPE_SUMMARY) as f:
        fps = json.load(f)["fps"]
    n_frames = eyes_pcs.shape[0]
    if not (lower_pcs.shape[0] == head_pose.shape[0] == n_frames):
        sys.exit("ERROR: frame counts differ: eyes {}, lower {}, head pose {}".format(
            n_frames, lower_pcs.shape[0], head_pose.shape[0]))

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

    print("Task runs in CSV:")
    print(runs.to_string())
    print("Using task run {} ({} trials)\n".format(task_session, len(df)))
    print(df.head().to_string())
    print()
    print(df.dtypes.to_string())

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

    df["time_since_video_start_ms"] = (df["timestamp"] - video_start).dt.total_seconds() * 1000
    frame_raw = to_frame(df["timestamp"]).astype(int)
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
    trial_eyes = np.zeros((len(df), N_PCS))
    trial_lower = np.zeros((len(df), N_PCS))
    trial_pose = np.zeros((len(df), 3))
    n_truncated = 0
    for i, fidx in enumerate(df["frame_idx"]):
        start = max(0, fidx - half)
        end = min(n_frames, fidx + half + 1)
        if not np.isnan(aim_change_frame[i]) and aim_change_frame[i] < end:
            end = max(fidx + 1, int(aim_change_frame[i]))
            n_truncated += 1
        trial_eyes[i] = window_mean(eyes_pcs, start, end)
        trial_lower[i] = window_mean(lower_pcs, start, end)
        trial_pose[i] = window_mean(head_pose, start, end)
    print("Window: +/-{} frames; {} windows cut short at an aim change".format(half, n_truncated))

    # --- Labels and site visits (participant's choices) ----------------------------------
    df["leave"] = (df["choice"].shift(-1) != df["choice"]).astype(int)
    df["site_visit"] = (df["choice"] != df["choice"].shift()).cumsum()
    keep = df.index[:-1]   # last trial: next choice unknown -> no label

    # --- Step 4: within-session z-scores (over labelled trials) ---------------------------
    trial_eyes_z = StandardScaler().fit_transform(trial_eyes[keep])
    trial_lower_z = StandardScaler().fit_transform(trial_lower[keep])
    trial_pose_z = StandardScaler().fit_transform(trial_pose[keep])
    df = df.loc[keep].copy()

    # --- Step 5: accumulation features within each visit ----------------------------------
    accum_features = []
    for _, visit_df in df.groupby("site_visit", sort=True):
        visit_idx = visit_df.index.tolist()
        rewards_cum, failures_run = [], []   # per trial so far in this visit
        for t in range(len(visit_idx)):
            so_far = visit_idx[:t + 1]
            outcomes = visit_df.loc[so_far, "outcome"].values
            # Consecutive failures up to and including trial t; resets on any reward.
            consec_failures = 0
            for o in reversed(outcomes):
                if o == "failure":
                    consec_failures += 1
                else:
                    break
            rewards_cum.append(int((outcomes == "reward").sum()))
            failures_run.append(consec_failures)

            row = {
                "trial_in_visit": t,
                "n_rewards_so_far": rewards_cum[-1],
            }
            row["reward_rate"] = row["n_rewards_so_far"] / (t + 1)
            row["consecutive_failures"] = consec_failures
            for name, series in (("n_rewards_so_far", rewards_cum),
                                 ("consecutive_failures", failures_run)):
                row["{}_slope".format(name)] = (
                    np.polyfit(np.arange(t + 1), series, 1)[0] if t >= 2 else 0.0)
            for label, feat_z in (("eyes", trial_eyes_z), ("lower", trial_lower_z)):
                vals = feat_z[so_far]
                for pc in range(N_PCS):
                    col = vals[:, pc]
                    row["{}_PC{}_mean".format(label, pc + 1)] = col.mean()
                    row["{}_PC{}_last3".format(label, pc + 1)] = col[-3:].mean()
                    row["{}_PC{}_slope".format(label, pc + 1)] = (
                        np.polyfit(np.arange(len(col)), col, 1)[0] if len(col) >= 3 else 0.0)
            for j, axis in enumerate(("yaw", "pitch", "roll")):
                col = trial_pose_z[so_far, j]
                row["pose_{}_mean".format(axis)] = col.mean()
                row["pose_{}_last3".format(axis)] = col[-3:].mean()
            accum_features.append(row)
    accum_df = pd.DataFrame(accum_features, index=df.index)

    # --- Step 6: final matrices -------------------------------------------------------------
    y = df["leave"].values.astype(int)
    X = accum_df.values.astype(np.float32)
    feature_names = list(accum_df.columns)
    meta = df[["trial", "choice", "active_crystal", "outcome", "switch_occurred", "leave",
               "timestamp", "site_visit", "frame_idx"]].copy()
    meta["trial_in_visit"] = accum_df["trial_in_visit"]

    # --- Step 7: save -------------------------------------------------------------------------
    np.save(OUTPUT_DIR / "X_features.npy", X)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    (OUTPUT_DIR / "feature_names.txt").write_text("\n".join(feature_names) + "\n")
    meta.to_csv(OUTPUT_DIR / "trial_metadata.csv", index=False)

    groups = {
        "eyes_brows": sum(n.startswith("eyes_") for n in feature_names),
        "lower_face": sum(n.startswith("lower_") for n in feature_names),
        "head_pose": sum(n.startswith("pose_") for n in feature_names),
        "task": sum(not n.startswith(("eyes_", "lower_", "pose_")) for n in feature_names),
    }
    summary = {
        "session_id": SESSION.removeprefix("session_"),
        "task_session_id": task_session,
        "n_trials": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_leave_trials": int(y.sum()),
        "leave_rate": round(float(y.mean()), 4),
        "n_site_visits": int(meta["site_visit"].nunique()),
        "n_pcs_per_roi": N_PCS,
        "feature_groups": groups,
        "label": "leave = participant's next choice differs from this trial's choice",
        "n_depletion_events": int(df["switch_occurred"].sum()),
        "video_start": str(video_start),
        "timing_source": timing_source,
        "fps": fps,
        "window_ms": [-HALF_WINDOW_S * 1000, HALF_WINDOW_S * 1000],
        "windows_truncated_at_aim_change": n_truncated,
        "trials_outside_video": n_out,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to {}:".format(OUTPUT_DIR))
    print("  X_features.npy {}, y_labels.npy {}, feature_names.txt, trial_metadata.csv, summary.json".format(
        X.shape, y.shape))
    print("  {} leave trials ({:.1%}), {} site visits, feature groups {}".format(
        summary["n_leave_trials"], summary["leave_rate"], summary["n_site_visits"], groups))


if __name__ == "__main__":
    main()
