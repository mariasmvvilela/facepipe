"""Pipeline stage 1: MediaPipe face landmarks + head pose for a recorded session.

Reads the session's .avi, runs the MediaPipe Tasks FaceLandmarker in VIDEO mode
and saves to mediapipe_output/<session folder name>/:

    landmarks.npy       (n_frames, 478, 3) float32, normalised image coords, NaN = no face
    head_pose.npy       (n_frames, 3) float32, yaw pitch roll in degrees, NaN = no face
    detection_mask.npy  (n_frames,) bool, True where a face was detected
    summary.json        session metadata

Usage (inside the facepipe env, from the project folder):
    python scripts\\st1_mediapipe_landmarks.py
    python scripts\\st1_mediapipe_landmarks.py --session raw_video\\session_YYYY-MM-DD_HH-MM-SS
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = PROJECT_DIR / "raw_video" / "session_20260928_161612"
MODEL_PATH = PROJECT_DIR / "scripts" / "face_landmarker.task"
OUTPUT_ROOT = PROJECT_DIR / "mediapipe_output"

N_LANDMARKS = 478 # maximum number of landmarks returned by MediaPipe FaceLandmarker
PROGRESS_EVERY = 100 # print progress every N frames

# Head-pose correspondences: MediaPipe landmark index -> canonical 3D point.
# The 3D points are in OpenCV's camera convention (x = image right, y = image down,
# z = away from camera), so a face looking straight at the camera gives R ~= identity
# and yaw/pitch/roll ~= 0. "left"/"right" are the participant's own sides; the
# participant's left eye (263) appears on the image's right, hence +x.
POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]
MODEL_POINTS = np.array([
    [0.0,    0.0,   0.0],    # nose tip           - landmark 1
    [0.0,   63.6,  12.5],    # chin               - landmark 152
    [43.3, -32.7,  26.0],    # left eye corner    - landmark 263
    [-43.3, -32.7, 26.0],    # right eye corner   - landmark 33
    [28.9,  28.9,  24.1],    # left mouth corner  - landmark 287
    [-28.9, 28.9,  24.1],    # right mouth corner - landmark 57
], dtype=np.float64)


def rotation_to_euler(R):
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch), camera frame. Returns degrees.

    Signs: +yaw = participant turns to their right (image left),
    +pitch = looks down, +roll = head tilts clockwise on screen.
    """
    pitch = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    roll = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([yaw, pitch, roll])


def head_pose(landmarks_norm, width, height, camera_matrix, dist_coeffs):
    image_points = landmarks_norm[POSE_LANDMARKS, :2] * np.array([width, height])
    # SQPNP, not ITERATIVE: with these 6 near-planar points ITERATIVE often converges
    # to the mirrored solution with the face behind the camera (tz < 0, roll ~ 180).
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points.astype(np.float64),
                                  camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok or tvec[2, 0] <= 0:
        return np.full(3, np.nan)
    R, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(R)


def find_video(session_dir):
    videos = sorted(session_dir.glob("*.avi"))
    if len(videos) != 1:
        sys.exit("ERROR: expected exactly one .avi in {}, found {}".format(
            session_dir, [v.name for v in videos]))
    return videos[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="session folder in raw_video/")
    parser.add_argument("--video", type=Path, default=None,
                        help="video file (default: the single .avi in the session folder)")
    args = parser.parse_args()

    session_dir = args.session if args.session.is_absolute() else PROJECT_DIR / args.session
    if not session_dir.is_dir():
        sys.exit("ERROR: session folder not found: {}".format(session_dir))
    video_path = args.video or find_video(session_dir)
    if not MODEL_PATH.is_file():
        sys.exit("ERROR: model not found: {}".format(MODEL_PATH))
    if not (session_dir / "task_events.csv").is_file():
        print("WARNING: no task_events.csv in session folder (not needed for this stage).")

    output_dir = OUTPUT_ROOT / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_dir.name.removeprefix("session_")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video: {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video: {}".format(video_path.name))
    print("  frames {}, {}x{}, {:.2f} fps, {:.1f} s".format(
        reported_frames, width, height, fps, reported_frames / fps))
    print("Output: {}\n".format(output_dir))

    camera_matrix = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )

    landmarks, poses, mask = [], [], []
    nan_landmarks = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    nan_pose = np.full(3, np.nan, dtype=np.float32)

    with FaceLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / fps))
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]],
                              dtype=np.float32)
                landmarks.append(lm)
                poses.append(head_pose(lm, width, height, camera_matrix,
                                       dist_coeffs).astype(np.float32))
                mask.append(True)
            else:
                landmarks.append(nan_landmarks)
                poses.append(nan_pose)
                mask.append(False)

            frame_idx += 1
            if frame_idx % PROGRESS_EVERY == 0:
                print("Frame {}/{} ({:.1f}%) — detections: {}/{}".format(
                    frame_idx, reported_frames, 100 * frame_idx / reported_frames,
                    sum(mask), frame_idx), flush=True)
    cap.release()

    n_frames = len(mask)
    if n_frames == 0:
        sys.exit("ERROR: no frames could be read from the video.")
    if n_frames != reported_frames:
        print("NOTE: decoded {} frames; container reported {}.".format(n_frames, reported_frames))

    landmarks = np.stack(landmarks)
    poses = np.stack(poses)
    mask = np.array(mask, dtype=bool)
    np.save(output_dir / "landmarks.npy", landmarks)
    np.save(output_dir / "head_pose.npy", poses)
    np.save(output_dir / "detection_mask.npy", mask)

    summary = {
        "session_id": session_id,
        "video_file": video_path.name,
        "total_frames": n_frames,
        "detected_frames": int(mask.sum()),
        "detection_rate": round(float(mask.mean()), 4),
        "fps": round(fps, 3),
        "duration_seconds": round(n_frames / fps, 2),
        "resolution": "{}x{}".format(width, height),
        "landmark_shape": list(landmarks.shape),
        "head_pose_shape": list(poses.shape),
        "landmark_units": "x, y normalised to [0, 1] by image width/height; z relative depth (MediaPipe)",
        "head_pose_columns": ["yaw", "pitch", "roll"],
        "head_pose_signs": "+yaw = turn to participant's right, +pitch = look down, +roll = clockwise on screen",
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. {}/{} frames with a face ({:.1%}).".format(
        summary["detected_frames"], n_frames, mask.mean()))
    print("Saved to {}:".format(output_dir))
    for name in ("landmarks.npy", "head_pose.npy", "detection_mask.npy", "summary.json"):
        print("  " + name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
