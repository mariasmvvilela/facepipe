"""Pipeline stage 3b: OpenFace 2.2 action units, head pose and gaze from the raw video.

Alternative to stages 1-3a. OpenFace does its own face tracking and 3D alignment,
so it runs on the raw recording, not on the stabilised video. Runs
FeatureExtraction.exe (a standalone binary, outside the conda env) and converts its
CSV into per-frame arrays in openface_output/<session folder name>/:

    raw/<video stem>.csv         OpenFace's untouched output (+ _of_details.txt)
    au_intensity.npy             (n_frames, 17) float32, AU intensity 0-5, NaN = not valid
    au_presence.npy              (n_frames, 18) float32, AU present 0/1, NaN = not valid
    au_intensity_names.txt       one AU name per line (AU01 ... AU45)
    au_presence_names.txt        one AU name per line (AU01 ... AU45, incl. AU28)
    head_pose.npy                (n_frames, 3) float32, OpenFace Ry Rx Rz in degrees, NaN = not valid
    gaze.npy                     (n_frames, 2) float32, gaze angle x, y in degrees, NaN = not valid
    landmarks_2d.npy             (n_frames, 68, 2) float32, pixel coords, NaN = not valid
    valid_mask.npy               (n_frames,) bool, success == 1 and confidence >= MIN_CONFIDENCE
    summary.json                 session metadata, OpenFace command, tracking quality

Notes on OpenFace 2.2.0:
  * On a video, FeatureExtraction uses its dynamic AU models: each AU is calibrated
    against the person's own neutral face (running median), which is what we want
    within one participant. -au_static would disable this.
  * head_pose columns are ordered yaw pitch roll like stage 1, but are OpenFace's own
    Ry/Rx/Rz (camera coordinates, radians -> degrees). The signs are not converted to
    stage 1's convention; diagnostics/compare_facemap_openface.py checks them.
  * Camera intrinsics are not given, so OpenFace guesses (fx = fy = 500): rotations
    are fine, translation (pose_T*) is only approximate and isn't saved.
  * OpenFace frames are 1-based; they are re-indexed to match the video's frame 0.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st2b_run_openface.py
    python scripts\\st2b_run_openface.py --session raw_data\\session_YYYYMMDD_HHMMSS_<task>
    python scripts\\st2b_run_openface.py --rerun      (run OpenFace again even if its CSV exists)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = PROJECT_DIR / "raw_data" / "session_20260928_161612_alien_energy_forager"
OUTPUT_ROOT = PROJECT_DIR / "openface_output"
# OpenFace lives outside the repo; set OPENFACE_DIR to override.
OPENFACE_DIR = Path(os.environ.get("OPENFACE_DIR", r"C:\Users\Maria\tools\OpenFace_2.2.0_win_x64"))
OPENFACE_EXE = OPENFACE_DIR / "FeatureExtraction.exe"
OPENFACE_VERSION = "2.2.0"
OPENFACE_FLAGS = ["-aus", "-pose", "-gaze", "-2Dfp"]

MIN_CONFIDENCE = 0.8   # OpenFace's own recommended threshold for a reliable track
N_LANDMARKS_2D = 68


def find_video(session_dir):
    videos = sorted(session_dir.glob("*.avi"))
    if len(videos) != 1:
        sys.exit("ERROR: expected exactly one .avi in {}, found {}".format(
            session_dir, [v.name for v in videos]))
    return videos[0]


def run_openface(video_path, raw_dir):
    cmd = [str(OPENFACE_EXE), "-f", str(video_path), "-out_dir", str(raw_dir), *OPENFACE_FLAGS, "-q"]
    print("Running OpenFace (this takes several minutes):\n  {}".format(" ".join(cmd)), flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
        sys.exit("ERROR: OpenFace exited with code {}".format(result.returncode))
    return cmd, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="session folder in raw_data/")
    parser.add_argument("--rerun", action="store_true",
                        help="run OpenFace again even if its CSV already exists")
    args = parser.parse_args()

    session_dir = args.session if args.session.is_absolute() else PROJECT_DIR / args.session
    if not session_dir.is_dir():
        sys.exit("ERROR: session folder not found: {}".format(session_dir))
    if not OPENFACE_EXE.is_file():
        sys.exit("ERROR: OpenFace not found: {} (set OPENFACE_DIR)".format(OPENFACE_EXE))
    video_path = find_video(session_dir)
    output_dir = OUTPUT_ROOT / session_dir.name
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / (video_path.stem + ".csv")

    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print("Video: {}  ({} frames, {:.2f} fps)".format(video_path.name, n_video, fps))

    # --- Step 1: run OpenFace (or reuse its CSV) ---------------------------------
    if csv_path.is_file() and not args.rerun:
        print("Reusing existing OpenFace output: {}".format(csv_path))
        cmd, runtime = None, None
    else:
        cmd, runtime = run_openface(video_path, raw_dir)
        print("OpenFace finished in {:.0f} s".format(runtime))
    if not csv_path.is_file():
        sys.exit("ERROR: OpenFace did not write {}".format(csv_path))

    # --- Step 2: parse and index by video frame --------------------------------------
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if df["face_id"].nunique() > 1:
        print("WARNING: OpenFace reported {} face ids; keeping face 0".format(df["face_id"].nunique()))
        df = df[df["face_id"] == 0]
    n_frames = n_video
    landmarks_path = PROJECT_DIR / "preprocessed" / session_dir.name / "landmarks.npy"
    if landmarks_path.is_file():
        n_mp = len(np.load(landmarks_path, mmap_mode="r"))
        if n_mp != n_video:
            print("WARNING: video reports {} frames, stage 1 has {}; using stage 1's count".format(
                n_video, n_mp))
            n_frames = n_mp
    idx = df["frame"].to_numpy().astype(int) - 1          # OpenFace frames are 1-based
    keep = (idx >= 0) & (idx < n_frames)
    df, idx = df[keep], idx[keep]
    if len(df) != n_frames:
        print("WARNING: OpenFace has rows for {} of {} frames; the rest are NaN".format(
            len(df), n_frames))

    def to_frames(values, fill=np.nan):
        out = np.full((n_frames,) + values.shape[1:], fill, dtype=np.float32)
        out[idx] = values
        return out

    success = to_frames(df["success"].to_numpy(), fill=0) == 1
    confidence = to_frames(df["confidence"].to_numpy())
    valid = success & (confidence >= MIN_CONFIDENCE)

    au_r_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    au_c_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]
    au_intensity = to_frames(df[au_r_cols].to_numpy())
    au_presence = to_frames(df[au_c_cols].to_numpy())
    head_pose = to_frames(np.degrees(df[["pose_Ry", "pose_Rx", "pose_Rz"]].to_numpy()))
    gaze = to_frames(np.degrees(df[["gaze_angle_x", "gaze_angle_y"]].to_numpy()))
    lm = np.stack([df[["x_{}".format(i) for i in range(N_LANDMARKS_2D)]].to_numpy(),
                   df[["y_{}".format(i) for i in range(N_LANDMARKS_2D)]].to_numpy()], axis=2)
    landmarks_2d = to_frames(lm)
    for arr in (au_intensity, au_presence, head_pose, gaze, landmarks_2d):
        arr[~valid] = np.nan

    # --- Step 3: save -------------------------------------------------------------------
    np.save(output_dir / "au_intensity.npy", au_intensity)
    np.save(output_dir / "au_presence.npy", au_presence)
    np.save(output_dir / "head_pose.npy", head_pose)
    np.save(output_dir / "gaze.npy", gaze)
    np.save(output_dir / "landmarks_2d.npy", landmarks_2d)
    np.save(output_dir / "valid_mask.npy", valid)
    (output_dir / "au_intensity_names.txt").write_text("\n".join(c[:-2] for c in au_r_cols) + "\n")
    (output_dir / "au_presence_names.txt").write_text("\n".join(c[:-2] for c in au_c_cols) + "\n")

    conf_found = confidence[np.isfinite(confidence)]
    summary = {
        "session_id": session_dir.name.removeprefix("session_"),
        "video_file": video_path.name,
        "openface_version": OPENFACE_VERSION,
        "openface_command": cmd if cmd else "reused existing CSV",
        "openface_runtime_s": None if runtime is None else round(runtime, 1),
        "total_frames": int(n_frames),
        "openface_rows": int(len(df)),
        "success_frames": int(success.sum()),
        "valid_frames": int(valid.sum()),
        "valid_rate": round(float(valid.mean()), 4),
        "min_confidence": MIN_CONFIDENCE,
        "confidence_median": round(float(np.median(conf_found)), 3),
        "confidence_p05": round(float(np.percentile(conf_found, 5)), 3),
        "fps": round(fps, 3),
        "au_intensity_names": [c[:-2] for c in au_r_cols],
        "au_presence_names": [c[:-2] for c in au_c_cols],
        "au_models": "dynamic (person-specific neutral calibration, OpenFace default for video)",
        "head_pose_columns": ["yaw (Ry)", "pitch (Rx)", "roll (Rz)"],
        "head_pose_units": "degrees, OpenFace camera coordinates (signs not matched to stage 1)",
        "gaze_columns": ["gaze_angle_x", "gaze_angle_y"],
        "gaze_units": "degrees",
        "landmarks_2d_units": "pixels in the original frame",
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. {} frames, {} valid ({:.1%}), median confidence {:.2f}".format(
        n_frames, valid.sum(), valid.mean(), summary["confidence_median"]))
    print("Mean AU intensity over valid frames:")
    means = np.nanmean(au_intensity, axis=0)
    print("  " + "  ".join("{} {:.2f}".format(c[:-2], m) for c, m in zip(au_r_cols, means)))
    print("Saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
