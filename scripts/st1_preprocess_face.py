"""Pipeline stage 1: face landmarks, stability check, clean face video and ROIs.

For head-mounted-camera sessions: the face sits still in the frame, so one fixed
crop and mask (from the session-median landmarks) is used for every frame. A
per-frame warp would only add MediaPipe landmark noise, and that noise follows gaze
and blinks (the eye-corner anchors shift), which would put choice-correlated motion
into the video.

Steps:
  1. MediaPipe FaceLandmarker on every raw frame -> 478 landmarks per frame.
  2. Stability check (stops here if the camera slipped). The face's shift in the
     image is measured by phase correlation of an eyes-to-nose patch against a
     reference frame; this is landmark-free, so blinks and gaze don't affect it.
     The session fails if the shift's 1st-99th percentile range exceeds
     STABILITY_LIMIT_PX (crop px; half a FaceMap bin at sbin 4). The MediaPipe
     anchor fit is reported alongside for comparison, but includes landmark noise.
  3. Clean video: one similarity transform from the median anchors onto a 256x256
     template, grayscale, everything outside the face oval set to grey.
  4. ROIs (as in Cazettes et al. 2025): horizontal bands of the face oval, bounded by
     landmarks of the median face (see ROIS in roi_definitions.py): whole_face, upper_face (forehead top to
     mid-nose, eyes included), lower_face (mid-nose to chin), and upper_face_no_eyes (upper_face
     with the visible eyes greyed, to separate brow / periorbital skin from blinks and gaze).

Writes to preprocessed/<session>/:
    landmarks.npy        (n_frames, 478, 3) float32, normalised raw-image coords, NaN = no face
    detection_mask.npy   (n_frames,) bool, True where a face was detected
    stability.json/.png  face shift over time (registration and MediaPipe), pass/fail
    face.avi             256x256 grayscale clean face video, lossless (FFV1), input fps
    rois.npz             one (256, 256) bool mask per ROI, in face.avi coordinates
    rois.json            ROI definitions and pixel counts
    roi_preview.png      every ROI outlined on the mean clean face
    summary.json         session metadata

After editing ROIS (roi_definitions.py) / EYE_MARGIN, --rois-only rebuilds just the ROIs (and their preview)
from the saved landmarks in a few seconds, without redoing MediaPipe or the video.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st1_preprocess_face.py --session raw_data\\session_YYYYMMDD_HHMMSS_<task>
    python scripts\\st1_preprocess_face.py --session raw_data\\session_YYYYMMDD_HHMMSS_<task> --rois-only
"""
import argparse
import json
import sys

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import PREPROCESSED_DIR, PROJECT_DIR, RAW_DIR, save_figure, session_name  # noqa: E402
from roi_definitions import ROIS  # noqa: E402

MODEL_PATH = PROJECT_DIR / "scripts" / "face_landmarker.task"

N_LANDMARKS = 478
PROGRESS_EVERY = 500

# --- Stability -------------------------------------------------------------------
STABILITY_LIMIT_PX = 2.0      # max 1st-99th percentile range of the face shift, crop px
MIN_DETECTION_RATE = 0.95     # warn below this

# --- Clean video (template as in the former st2) ------------------------------------
OUT_SIZE = 256
# Anchors: rigid points only. "right"/"left" are the participant's own sides, so 133
# (participant's right eye) appears on the image's left.
ANCHOR_IDX = {"right_eye_inner": 133, "left_eye_inner": 362, "nose_tip": 1, "nose_bridge": 168}
# Template: eye midpoint at (128, 90), nose tip straight below it at y = 145. The
# anchors' relative proportions come from the session's median face (build_template).
EYE_MID = np.array([128.0, 90.0])
EYE_TO_NOSE_TIP = 55.0
FIT_THRESHOLD_PX = 1000.0     # huge RANSAC threshold = plain least squares on all 4 anchors
MASK_GREY = 128
FACE_ERODE_PX = 2             # trim the jagged edge of the face oval (crop px)

# --- MediaPipe landmark sets (participant's own left / right) ------------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
# Eye cut-out: the eye outline is the lid margin (median, i.e. open, eye), so this is the
# visible eye only, plus a small margin for the lashes (fraction of the distance between
# the eye centres). The skin around the eyes stays in.
EYE_MARGIN = 0.1

def find_video(session_dir):
    videos = sorted(session_dir.glob("*.avi"))
    if len(videos) != 1:
        sys.exit("ERROR: expected exactly one .avi in {}, found {}".format(
            session_dir, [v.name for v in videos]))
    return videos[0]


def transform_points(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def build_template(median_anchors):
    """Map the median anchor configuration (pixels) into the 256x256 template layout."""
    right_eye, left_eye, nose_tip, _ = median_anchors
    eye_mid = (right_eye + left_eye) / 2
    v = nose_tip - eye_mid
    down = v / np.linalg.norm(v)
    right = np.array([down[1], -down[0]])   # image x axis when the face is upright
    scale = EYE_TO_NOSE_TIP / np.linalg.norm(v)
    rel = median_anchors - eye_mid
    return (EYE_MID + scale * np.stack([rel @ right, rel @ down], axis=1)).astype(np.float32)


def stats(x):
    x = x[np.isfinite(x)]
    return {"std": round(float(np.std(x)), 3),
            "p1_p99_range": round(float(np.percentile(x, 99) - np.percentile(x, 1)), 3),
            "max_abs_from_median": round(float(np.max(np.abs(x - np.median(x)))), 3)}


def disk(radius_px):
    d = 2 * int(round(radius_px)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


# --- Step 1 ----------------------------------------------------------------------------
def detect_landmarks(video_path, fps, n_reported):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    nan_lm = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    landmarks, mask = [], []
    cap = cv2.VideoCapture(str(video_path))
    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # VIDEO mode only needs increasing timestamps; exact times are in the frametimes file.
            result = detector.detect_for_video(image, int(len(mask) * 1000 / fps))
            if result.face_landmarks:
                landmarks.append(np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]],
                                          dtype=np.float32))
                mask.append(True)
            else:
                landmarks.append(nan_lm)
                mask.append(False)
            if len(mask) % PROGRESS_EVERY == 0:
                print("  landmarks {}/{} - detections {}".format(len(mask), n_reported, sum(mask)),
                      flush=True)
    cap.release()
    return np.stack(landmarks), np.array(mask, dtype=bool)


# --- Step 2 ----------------------------------------------------------------------------
def check_stability(video_path, anchors_px, detected, crop_scale, fps, out_dir):
    """Face shift in the image over time, in crop px. Returns (passed, metrics)."""
    n = len(anchors_px)
    ref = np.nanmedian(anchors_px, axis=0).astype(np.float32)

    # MediaPipe: similarity fit of each frame's anchors onto the median (includes landmark noise)
    lm_shift = np.full((n, 2), np.nan)
    for i in np.flatnonzero(detected):
        M, _ = cv2.estimateAffinePartial2D(anchors_px[i].astype(np.float32), ref,
                                           ransacReprojThreshold=FIT_THRESHOLD_PX)
        if M is not None:
            lm_shift[i] = -M[:, 2] * crop_scale     # the face's displacement, not the correction

    # Image registration: phase correlation of a brows-to-nose-tip patch (landmark-free)
    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eyes = ref[:2].mean(axis=0)
    eye_dist = np.linalg.norm(ref[0] - ref[1])
    pw = 1.6 * eye_dist
    x0, x1 = int(max(0, eyes[0] - pw)), int(min(width, eyes[0] + pw))
    y0, y1 = int(max(0, eyes[1] - 0.6 * eye_dist)), int(min(height, ref[2][1] + 0.1 * eye_dist))
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_64F)
    ref_frame = int(np.nanargmin(np.abs(lm_shift - np.nanmedian(lm_shift, axis=0)).sum(axis=1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    _, f = cap.read()
    ref_patch = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float64)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    reg_shift = np.full((n, 2), np.nan)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        patch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float64)
        (dx, dy), _ = cv2.phaseCorrelate(ref_patch, patch, win)
        reg_shift[i] = np.array([dx, dy]) * crop_scale
        if (i + 1) % (4 * PROGRESS_EVERY) == 0:
            print("  stability {}/{}".format(i + 1, n), flush=True)
    cap.release()

    reg_stats = {"dx": stats(reg_shift[:, 0]), "dy": stats(reg_shift[:, 1])}
    worst = max(reg_stats["dx"]["p1_p99_range"], reg_stats["dy"]["p1_p99_range"])
    passed = worst <= STABILITY_LIMIT_PX
    metrics = {
        "passed": bool(passed),
        "criterion": "image-registration shift, 1st-99th percentile range <= {} crop px (dx and dy)".format(
            STABILITY_LIMIT_PX),
        "worst_p1_p99_range_px": worst,
        "units": "pixels of the {0}x{0} face crop (1 crop px = {1:.2f} raw px); FaceMap sbin=4".format(
            OUT_SIZE, 1 / crop_scale),
        "detection_rate": round(float(detected.mean()), 4),
        "image_registration": {**reg_stats, "patch_raw_px": [x0, y0, x1, y1], "reference_frame": ref_frame},
        "mediapipe_anchor_fit": {"dx": stats(lm_shift[:, 0]), "dy": stats(lm_shift[:, 1]),
                                 "note": "includes MediaPipe landmark noise (blinks, gaze); not used for pass/fail"},
    }
    with open(out_dir / "stability.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    t = np.arange(n) / fps
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, k, name in zip(axes, (0, 1), ("x", "y")):
        ax.plot(t, lm_shift[:, k] - np.nanmedian(lm_shift[:, k]), lw=0.5, color="0.6",
                label="MediaPipe anchors (incl. landmark noise)")
        ax.plot(t, reg_shift[:, k] - np.nanmedian(reg_shift[:, k]), lw=0.8, label="image registration")
        ax.axhspan(-STABILITY_LIMIT_PX / 2, STABILITY_LIMIT_PX / 2, color="0.9", zorder=0,
                   label="+/-{} px".format(STABILITY_LIMIT_PX / 2))
        ax.set_ylabel("face {} shift\n(crop px)".format(name))
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("{}: {} (registration p1-p99 range {:.2f} px, limit {})".format(
        out_dir.name, "STABLE" if passed else "NOT STABLE", worst, STABILITY_LIMIT_PX))
    fig.tight_layout()
    save_figure(fig, out_dir / "stability.png")
    plt.close(fig)
    return passed, metrics


# --- Step 3 ----------------------------------------------------------------------------
def write_face_video(video_path, M, face_mask, fps, n_frames, out_path):
    """Warp every frame with M, grayscale, grey outside the face. Returns the mean frame."""
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"FFV1"), fps,
                             (OUT_SIZE, OUT_SIZE), isColor=False)
    if not writer.isOpened():
        sys.exit("ERROR: could not create {}".format(out_path))
    cap = cv2.VideoCapture(str(video_path))
    acc = np.zeros((OUT_SIZE, OUT_SIZE))
    written = 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = cv2.warpAffine(gray, M, (OUT_SIZE, OUT_SIZE), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        out[~face_mask] = MASK_GREY
        writer.write(out)
        acc += out
        written += 1
        if written % (4 * PROGRESS_EVERY) == 0:
            print("  face video {}/{}".format(written, n_frames), flush=True)
    cap.release()
    writer.release()
    return acc / max(written, 1), written


# --- Step 4 ----------------------------------------------------------------------------
def region_mask(lm_crop, outlines, margin_px):
    mask = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
    for idx in outlines:
        hull = cv2.convexHull(np.round(lm_crop[idx]).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    if margin_px > 0:
        mask = cv2.dilate(mask, disk(margin_px))
    return mask.astype(bool)


def landmark_row(lm_crop, idx):
    return None if idx is None else int(round(lm_crop[idx, 1]))


def build_rois(lm_crop):
    """ROI masks from the median landmarks in crop coordinates. Returns (rois, face mask,
    eye-centre distance, {roi: [first row, end row)} of each band)."""
    eye_dist = np.linalg.norm(lm_crop[RIGHT_EYE].mean(axis=0) - lm_crop[LEFT_EYE].mean(axis=0))
    face = cv2.erode(region_mask(lm_crop, [FACE_OVAL], 0).astype(np.uint8),
                     disk(FACE_ERODE_PX)).astype(bool)
    eyes = region_mask(lm_crop, [RIGHT_EYE, LEFT_EYE], EYE_MARGIN * eye_dist)
    rows = np.arange(OUT_SIZE)[:, None]
    rois, bands = {}, {}
    for name, (top, bottom, no_eyes) in ROIS.items():
        y0 = landmark_row(lm_crop, top)
        y1 = landmark_row(lm_crop, bottom)
        y0, y1 = (0 if y0 is None else y0), (OUT_SIZE if y1 is None else y1)
        m = face & (rows >= y0) & (rows < y1)
        if no_eyes:
            m &= ~eyes
        rois[name], bands[name] = m, [y0, y1]
    return rois, face, eye_dist, bands


def mean_video_frame(video_path, n_samples=300):
    """Mean grayscale frame from evenly spaced frames of a video."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc, used = np.zeros((OUT_SIZE, OUT_SIZE)), 0
    for f in np.linspace(0, n - 1, min(n, n_samples)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            acc += cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            used += 1
    cap.release()
    return acc / max(used, 1)


def save_rois(out_dir, rois, eye_dist, bands, mean_face, session_name):
    np.savez_compressed(out_dir / "rois.npz", **rois)
    roi_info = {
        "units": "pixels of face.avi ({0}x{0})".format(OUT_SIZE),
        "eye_centre_distance_px": round(float(eye_dist), 2),
        "eye_cutout": {"landmark_outlines": [RIGHT_EYE, LEFT_EYE],
                       "margin_fraction_of_eye_distance": EYE_MARGIN},
        "rois": {name: {"top_landmark": top, "bottom_landmark": bottom,
                        "rows": "{} <= y < {}".format(*bands[name]),
                        "eyes_cut_out": no_eyes, "within_face_oval": True,
                        "n_pixels": int(rois[name].sum())}
                 for name, (top, bottom, no_eyes) in ROIS.items()},
    }
    with open(out_dir / "rois.json", "w") as fh:
        json.dump(roi_info, fh, indent=2)
    save_roi_preview(mean_face, rois, out_dir / "roi_preview.png",
                     "{}: ROIs on the mean face".format(session_name))
    for name in ROIS:
        print("  {:22s} {:6d} px".format(name, int(rois[name].sum())))


def save_roi_preview(mean_face, rois, path, title):
    names = list(rois)
    n_cols = 3
    n_rows = -(-len(names) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        m = rois[name]
        ax.imshow(mean_face, cmap="gray", vmin=0, vmax=255)
        ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.35, vmin=0, vmax=1)
        ax.contour(m, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title("{} ({} px)".format(name, int(m.sum())), fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def fixed_transform(landmarks, width, height):
    """Median landmarks (raw px), template, and the one transform median anchors -> template."""
    px = np.array([width, height], dtype=np.float32)
    median_lm = np.nanmedian(landmarks[:, :, :2], axis=0) * px          # (478, 2) raw px
    anchor_idx = list(ANCHOR_IDX.values())
    template = build_template(median_lm[anchor_idx])
    M, _ = cv2.estimateAffinePartial2D(median_lm[anchor_idx].astype(np.float32), template,
                                       ransacReprojThreshold=FIT_THRESHOLD_PX)
    return median_lm, template, M


def rebuild_rois(session_dir, out_dir, width, height):
    """--rois-only: ROIs from the saved landmarks and face.avi, without redoing steps 1-3."""
    for name in ("landmarks.npy", "face.avi"):
        if not (out_dir / name).is_file():
            sys.exit("ERROR: --rois-only needs a full st1 run first (missing {})".format(out_dir / name))
    median_lm, _, M = fixed_transform(np.load(out_dir / "landmarks.npy"), width, height)
    rois, _, eye_dist, bands = build_rois(transform_points(M, median_lm))
    print("ROIs (rebuilt from saved landmarks):")
    save_rois(out_dir, rois, eye_dist, bands, mean_video_frame(out_dir / "face.avi"), session_dir.name)
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["rois"] = list(ROIS)
        summary_path.write_text(json.dumps(summary, indent=2))
    print("Saved rois.npz, rois.json, roi_preview.png to {}".format(out_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="session folder name (or path) in raw_data/")
    parser.add_argument("--rois-only", action="store_true",
                        help="only rebuild the ROIs from the saved landmarks (after editing ROIS / EYE_MARGIN)")
    args = parser.parse_args()

    session_dir = RAW_DIR / session_name(args.session)
    if not MODEL_PATH.is_file():
        sys.exit("ERROR: model not found: {}".format(MODEL_PATH))
    video_path = find_video(session_dir)
    out_dir = PREPROCESSED_DIR / session_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video: {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps > 0:
        sys.exit("ERROR: video reports fps {}; cannot continue".format(fps))
    print("Video: {}  ({} frames, {}x{}, {:.2f} fps)".format(video_path.name, n_reported, width, height, fps))
    print("Output: {}\n".format(out_dir))
    if args.rois_only:
        rebuild_rois(session_dir, out_dir, width, height)
        return

    # 1. Landmarks
    print("Step 1/4: MediaPipe landmarks")
    landmarks, detected = detect_landmarks(video_path, fps, n_reported)
    n_frames = len(detected)
    if n_frames == 0 or not detected.any():
        sys.exit("ERROR: no face detected in any frame")
    np.save(out_dir / "landmarks.npy", landmarks)
    np.save(out_dir / "detection_mask.npy", detected)
    print("  {}/{} frames with a face ({:.1%})".format(detected.sum(), n_frames, detected.mean()))
    if detected.mean() < MIN_DETECTION_RATE:
        print("WARNING: face detected in only {:.1%} of frames (expected >= {:.0%})".format(
            detected.mean(), MIN_DETECTION_RATE))

    # The fixed transform: median anchors -> template
    px = np.array([width, height], dtype=np.float32)
    anchor_idx = list(ANCHOR_IDX.values())
    median_lm, template, M = fixed_transform(landmarks, width, height)
    crop_scale = float(np.hypot(M[0, 0], M[1, 0]))                      # crop px per raw px

    # 2. Stability
    print("\nStep 2/4: stability check")
    passed, stability = check_stability(video_path, landmarks[:, anchor_idx, :2] * px, detected,
                                        crop_scale, fps, out_dir)
    print("  registration shift p1-p99 range: dx {:.2f}, dy {:.2f} crop px (limit {})".format(
        stability["image_registration"]["dx"]["p1_p99_range"],
        stability["image_registration"]["dy"]["p1_p99_range"], STABILITY_LIMIT_PX))
    if not passed:
        sys.exit("\nSTOPPED: the face moved in the frame more than the limit, so a fixed crop and "
                 "mask would not stay on the face.\nSee {} and stability.json. Landmarks are saved; "
                 "no face video or ROIs were made.".format(out_dir / "stability.png"))
    print("  stable")

    # 3. Clean face video + 4. ROIs (both from the median landmarks in crop coordinates)
    lm_crop = transform_points(M, median_lm)
    rois, face_mask, eye_dist, bands = build_rois(lm_crop)
    print("\nStep 3/4: clean face video")
    mean_face, written = write_face_video(video_path, M, face_mask, fps, n_frames, out_dir / "face.avi")
    if written != n_frames:
        print("WARNING: wrote {} frames, landmarks have {}".format(written, n_frames))

    print("\nStep 4/4: ROIs")
    save_rois(out_dir, rois, eye_dist, bands, mean_face, session_dir.name)

    summary = {
        "session_id": session_dir.name.removeprefix("session_"),
        "input_video": video_path.name,
        "total_frames": n_frames,
        "detected_frames": int(detected.sum()),
        "detection_rate": round(float(detected.mean()), 4),
        "fps": round(fps, 3),
        "input_resolution": "{}x{}".format(width, height),
        "output_video": "face.avi",
        "output_resolution": "{0}x{0}".format(OUT_SIZE),
        "output_codec": "FFV1 (lossless), grayscale",
        "crop_px_per_raw_px": round(crop_scale, 4),
        "transform": np.round(M, 5).tolist(),
        "anchor_landmarks": ANCHOR_IDX,
        "template_points": np.round(template, 2).tolist(),
        "mask_fill_value": MASK_GREY,
        "stability_passed": True,
        "stability_worst_p1_p99_range_px": stability["worst_p1_p99_range_px"],
        "rois": list(ROIS),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nDone. Saved to {}:".format(out_dir))
    print("  landmarks.npy, detection_mask.npy, stability.json/.png, face.avi, rois.npz, "
          "rois.json, roi_preview.png, summary.json")
    print("Next: python scripts\\st2_run_facemap.py --session {}".format(session_dir.name))


if __name__ == "__main__":
    main()
