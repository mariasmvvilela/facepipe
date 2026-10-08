"""Task trials -> video frames, and the leave labels. Shared by st3 and the diagnostics.

Definitions (from the task code, not the column names):
  * leave: the participant's choice on the next trial differs from this trial's.
    `switch_occurred` is the task's hidden, random switch of the active site, which
    the participant never sees; it is NOT the label.
  * site visit: a run of consecutive trials with the same `choice` (where the
    participant is foraging), not a run of the hidden `active_site`.
  * The last trial has no next choice, so it has no label and is dropped.
"""
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import RAW_DIR


@dataclass
class Alignment:
    df: pd.DataFrame            # trials of the task run recorded in the video
    task_session: str           # that run's session_id in task_events.csv
    to_frame: Callable          # pd.Series of times -> float array of frame indices (see align_trials)
    video_name: str
    video_start: pd.Timestamp
    video_end: pd.Timestamp
    timing_source: str


def video_start_time(session_dir):
    """Recording start from the video filename, recording_YYYY-MM-DD_HH-MM-SS.avi.

    Only 1-second precision: used to pick the task run, and for timing only when there
    is no frametimes file.
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


def align_trials(session, n_frames, fps):
    """Load raw_data/<session>/task_events.csv, keep the task run recorded in the video,
    and build the time -> frame mapping.

    to_frame(times) returns the nearest frame to each time as floats (NaN stays NaN);
    times outside the video map to < 0 or >= n_frames, so callers decide whether to clip.
    """
    session_dir = RAW_DIR / session
    task_csv = session_dir / "task_events.csv"
    if not task_csv.is_file():
        sys.exit("ERROR: missing input: {}".format(task_csv))
    df_all = pd.read_csv(task_csv)
    # Sessions recorded before the schema was standardised use the old column names
    df_all = df_all.rename(columns={"trial": "trial_number", "active_ufo": "active_site",
                                    "active_crystal": "active_site"})
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

    video_start, video_name = video_start_time(session_dir)
    video_end = video_start + pd.Timedelta(seconds=n_frames / fps)

    # The CSV is appended to across task runs: keep the run that overlaps the video.
    runs = df_all.groupby("session_id")["timestamp"].agg(["min", "max", "size"])
    runs["overlap_s"] = [(min(r["max"], video_end) - max(r["min"], video_start)).total_seconds()
                         for _, r in runs.iterrows()]
    task_session = runs["overlap_s"].idxmax()
    if runs.loc[task_session, "overlap_s"] <= 0:
        sys.exit("ERROR: no task run in {} overlaps the video ({} to {})".format(
            task_csv.name, video_start, video_end))
    df = df_all[df_all["session_id"] == task_session].reset_index(drop=True)
    if len(runs) > 1:
        print("Task runs in {}:\n{}".format(task_csv.name, runs.to_string()))
    print("Using task run {} ({} trials)".format(task_session, len(df)))

    # Preferred: exact per-frame times from the recording sidecar. Fallback: the
    # filename start time + a constant fps (1 s precision).
    frame_times = load_frame_times(session_dir, video_name)
    if frame_times is not None and len(frame_times) != n_frames:
        print("WARNING: {} frame times for {} frames; falling back to filename timing.".format(
            len(frame_times), n_frames))
        frame_times = None
    if frame_times is not None:
        ft_ns = frame_times.values.astype("datetime64[ns]").astype(np.int64)
        video_start = frame_times.iloc[0]
        timing_source = "per-frame timestamps ({})".format(Path(video_name).stem + "_frametimes.csv")

        def to_frame(times):
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

    print("Video {}: start {}, {} frames at {} fps".format(video_name, video_start, n_frames, fps))
    print("Timing source: {}".format(timing_source))
    return Alignment(df, task_session, to_frame, video_name, video_start, video_end, timing_source)


def add_labels(df):
    """Add `leave` and `site_visit` (1, 2, ...) and drop the last trial (no next choice)."""
    df = df.copy()
    df["leave"] = (df["choice"].shift(-1) != df["choice"]).astype(int)
    df["site_visit"] = (df["choice"] != df["choice"].shift()).cumsum()
    return df.iloc[:-1].copy()
