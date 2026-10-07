"""Pipeline stage 2: geometric stabilisation of the face video.

For each frame, a similarity transform (rotation + uniform scale + translation)
maps 4 MediaPipe anchor landmarks onto a fixed canonical template, so head motion
is removed and only facial movement remains. Pixels outside the face oval are then
set to neutral grey. Writes to stabilised_video/<session>/:

    stabilised.avi        256x256 grayscale (stored as 3-channel), face-oval masked, XVID, input fps
    diagnostic_strip.png  10 evenly spaced frames: original crop (top) vs stabilised + masked (bottom)
    transforms.npy        (n_frames, 2, 3) float32 per-frame warp matrices, NaN = failed
    summary.json          session metadata + stabilisation quality metrics

--fixed (head-mounted camera): one transform and one face-oval mask, both from the
session-median landmarks, for every frame. The face doesn't move in the image, so a
per-frame warp would only add MediaPipe landmark noise, and with these anchors that
noise follows gaze (the eye corners shift when the participant looks left/right),
which would put a choice-correlated shift into the video. Check stability first with
diagnostics/check_camera_stability.py.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st2_stabilise_video.py
    python scripts\\st2_stabilise_video.py --session raw_data\\session_YYYYMMDD_HHMMSS_<task>
    python scripts\\st2_stabilise_video.py --session raw_data\\session_YYYYMMDD_HHMMSS_<task> --fixed
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = PROJECT_DIR / "raw_data" / "session_20260928_161612_alien_energy_forager"
LANDMARK_ROOT = PROJECT_DIR / "mediapipe_output"
OUTPUT_ROOT = PROJECT_DIR / "stabilised_video"

OUT_SIZE = 256
PROGRESS_EVERY = 500
N_STRIP = 10

# Anchor landmarks. "right"/"left" are the participant's own sides, so 133
# (participant's right eye) appears on the image's left.
ANCHOR_IDX = {
    "right_eye_inner": 133,
    "left_eye_inner":  362,
    "nose_tip":        1,
    "nose_bridge":     168,   # was 152 (chin), which moves with the jaw
}
# Template layout: eye midpoint at (128, 90), nose tip straight below it at y = 145.
# The anchors' relative proportions come from the session's median face (see
# build_template): a fixed hand-made template can't be matched by a similarity
# transform unless its proportions match the face's, and then the fit silently
# keeps only the two eye points. (168 sits at eye level, not halfway down the nose.)
EYE_MID = np.array([128.0, 90.0])
EYE_TO_NOSE_TIP = 55.0
# Large RANSAC threshold = every anchor is always an inlier, i.e. a plain
# least-squares fit on all 4 points (the default 3 px drops points frame to frame).
FIT_THRESHOLD_PX = 1000.0
# Non-anchor landmarks used only to measure how still the face is after warping.
# The chin is expected to move a little (jaw), the others shouldn't.
CHECK_IDX = {"forehead": 10, "right_eye_outer": 33, "left_eye_outer": 263, "chin": 152}

# MediaPipe face-oval landmarks. Everything outside the (warped) oval is set to
# neutral grey so background, hair and clothing motion never reaches FaceMap.
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
MASK_GREY = 128
MASK_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def apply_face_oval(frame, oval_pts_px, M):
    """Grey out everything outside the face oval in a stabilised frame.

    oval_pts_px: (n, 2) face-oval landmarks in original-frame pixels
    M:           2x3 affine matrix used to warp this frame
    """
    oval = np.round(transform_points(M, oval_pts_px)).astype(np.int32)
    mask = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(oval), 1)   # hull guards against a non-convex outline
    mask = cv2.erode(mask, MASK_ERODE_KERNEL, iterations=1)   # trim the jagged boundary
    result = frame.copy()
    result[mask == 0] = MASK_GREY
    return result


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


def transform_points(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def find_video(session_dir):
    videos = sorted(session_dir.glob("*.avi"))
    if len(videos) != 1:
        sys.exit("ERROR: expected exactly one .avi in {}, found {}".format(
            session_dir, [v.name for v in videos]))
    return videos[0]


def to_gray_bgr(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="session folder in raw_data/")
    parser.add_argument("--fixed", action="store_true",
                        help="head-mounted camera: one median transform + mask for all frames")
    args = parser.parse_args()

    session_dir = args.session if args.session.is_absolute() else PROJECT_DIR / args.session
    if not session_dir.is_dir():
        sys.exit("ERROR: session folder not found: {}".format(session_dir))
    video_path = find_video(session_dir)
    landmarks_path = LANDMARK_ROOT / session_dir.name / "landmarks.npy"
    if not landmarks_path.is_file():
        sys.exit("ERROR: landmarks not found (run mediapipe_landmarks.py first): {}".format(
            landmarks_path))
    output_dir = OUTPUT_ROOT / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_dir.name.removeprefix("session_")

    landmarks = np.load(landmarks_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video: {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames != len(landmarks):
        print("WARNING: video reports {} frames but landmarks have {}; "
              "using the landmark count.".format(n_frames, len(landmarks)))
        n_frames = len(landmarks)

    px = np.array([width, height], dtype=np.float32)
    anchor_idx = list(ANCHOR_IDX.values())
    anchors_px = landmarks[:, anchor_idx, :2] * px                 # (n, 4, 2)
    template = build_template(np.nanmedian(anchors_px, axis=0))
    if args.fixed:
        median_lm_px = np.nanmedian(landmarks[:, :, :2], axis=0) * px    # (478, 2)
        M_fixed, _ = cv2.estimateAffinePartial2D(median_lm_px[anchor_idx].astype(np.float32),
                                                 template, ransacReprojThreshold=FIT_THRESHOLD_PX)
        print("Mode: fixed (one median transform and face-oval mask for all frames)")
    print("Video: {}  ({} frames, {}x{}, {:.2f} fps)".format(
        video_path.name, n_frames, width, height, fps))
    print("Template anchors (px in 256x256 crop):")
    for name, (x, y) in zip(ANCHOR_IDX, template):
        print("  {:16s} ({:6.1f}, {:6.1f})".format(name, x, y))
    print("Output: {}\n".format(output_dir))

    # Fixed crop of the original frame around the median face, for the diagnostic strip.
    all_px = landmarks[:, :, :2] * px
    lo, hi = np.nanmedian(all_px.min(axis=1), axis=0), np.nanmedian(all_px.max(axis=1), axis=0)
    centre, half = (lo + hi) / 2, 0.8 * (hi - lo).max()
    crop_box = [int(max(0, centre[0] - half)), int(max(0, centre[1] - half)),
                int(min(width, centre[0] + half)), int(min(height, centre[1] + half))]
    strip_frames = set(np.linspace(0, n_frames - 1, N_STRIP).astype(int).tolist())
    strip_orig, strip_stab = {}, {}

    writer = cv2.VideoWriter(str(output_dir / "stabilised.avi"),
                             cv2.VideoWriter_fourcc(*"XVID"), fps, (OUT_SIZE, OUT_SIZE))
    if not writer.isOpened():
        sys.exit("ERROR: could not create output video.")

    failed = []
    transforms = np.full((n_frames, 2, 3), np.nan, dtype=np.float32)
    residuals = np.full((n_frames, len(anchor_idx)), np.nan)
    check_before = np.full((n_frames, len(CHECK_IDX), 2), np.nan)
    check_after = np.full((n_frames, len(CHECK_IDX), 2), np.nan)
    blank = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            print("WARNING: video ended early at frame {}".format(frame_idx))
            n_frames = frame_idx
            break

        src = anchors_px[frame_idx].astype(np.float32)
        M = None
        if args.fixed:
            M = M_fixed
        elif not np.isnan(src).any():
            M, _ = cv2.estimateAffinePartial2D(src, template,
                                               ransacReprojThreshold=FIT_THRESHOLD_PX)
        if M is None:
            failed.append(frame_idx)
            out = blank
        else:
            transforms[frame_idx] = M
            out = to_gray_bgr(cv2.warpAffine(frame, M, (OUT_SIZE, OUT_SIZE),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE))
            oval_px = (median_lm_px[FACE_OVAL_IDX] if args.fixed
                       else landmarks[frame_idx, FACE_OVAL_IDX, :2] * px)
            out = apply_face_oval(out, oval_px, M)
            residuals[frame_idx] = np.linalg.norm(transform_points(M, src) - template, axis=1)
            chk = landmarks[frame_idx, list(CHECK_IDX.values()), :2] * px
            # "before" in the same 256-crop scale as "after", but without removing motion
            check_before[frame_idx] = chk * np.sqrt(abs(np.linalg.det(M[:, :2])))
            check_after[frame_idx] = transform_points(M, chk)
        writer.write(out)

        if frame_idx in strip_frames:
            x0, y0, x1, y1 = crop_box
            strip_orig[frame_idx] = to_gray_bgr(cv2.resize(frame[y0:y1, x0:x1], (OUT_SIZE, OUT_SIZE)))
            strip_stab[frame_idx] = out.copy()

        done = frame_idx + 1
        if done % PROGRESS_EVERY == 0:
            print("Frame {}/{} ({:.1f}%) — failed warps: {}".format(
                done, n_frames, 100 * done / n_frames, len(failed)), flush=True)

    writer.release()
    cap.release()
    np.save(output_dir / "transforms.npy", transforms[:n_frames])

    # Diagnostic strip: template anchor positions marked in red on the stabilised row.
    order = sorted(strip_orig)
    for f in order:
        for x, y in template:
            cv2.circle(strip_stab[f], (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        for img in (strip_orig[f], strip_stab[f]):
            cv2.putText(img, str(f), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
    strip = np.vstack([np.hstack([strip_orig[f] for f in order]),
                       np.hstack([strip_stab[f] for f in order])])
    cv2.imwrite(str(output_dir / "diagnostic_strip.png"), strip)

    # Stillness of non-anchor landmarks (std of position, px in the 256 crop).
    jitter_before = np.nanstd(check_before, axis=0).mean(axis=1)
    jitter_after = np.nanstd(check_after, axis=0).mean(axis=1)

    summary = {
        "session_id": session_id,
        "mode": "fixed" if args.fixed else "per_frame",
        "input_video": video_path.name,
        "output_video": "stabilised.avi",
        "total_frames": n_frames,
        "failed_warps": len(failed),
        "failed_frame_indices": failed,
        "output_resolution": "{}x{}".format(OUT_SIZE, OUT_SIZE),
        "fps": round(fps, 3),
        "grayscale": True,
        "face_oval_mask": {"landmarks": FACE_OVAL_IDX, "fill_value": MASK_GREY,
                           "erode_kernel_px": 5},
        "anchor_landmarks": anchor_idx,
        "template_points": np.round(template, 2).tolist(),
        "anchor_residual_px_median": np.round(np.nanmedian(residuals, axis=0), 2).tolist(),
        "anchor_residual_px_p99": np.round(np.nanpercentile(residuals, 99, axis=0), 2).tolist(),
        "check_landmark_jitter_px": {
            name: {"before": round(float(b), 2), "after": round(float(a), 2)}
            for name, b, a in zip(CHECK_IDX, jitter_before, jitter_after)
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. {} frames, {} failed warps.".format(n_frames, len(failed)))
    print("Anchor residual after warp (median px): {}".format(
        dict(zip(ANCHOR_IDX, summary["anchor_residual_px_median"]))))
    print("Non-anchor landmark jitter, std px (before -> after):")
    for name, v in summary["check_landmark_jitter_px"].items():
        print("  {:16s} {:6.2f} -> {:5.2f}".format(name, v["before"], v["after"]))
    print("Saved: stabilised.avi, diagnostic_strip.png, transforms.npy, summary.json")


if __name__ == "__main__":
    main()
