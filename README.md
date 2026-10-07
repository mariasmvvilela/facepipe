# facepipe

Facial and eye analysis pipeline for predicting foraging site-leave decisions.
Human replication of Cazettes et al. (2025, Nature Neuroscience) — Mainen Lab.

## Overview
Participants perform a probabilistic foraging task while face video and eye tracking
data (Pupil Labs) are recorded. The pipeline extracts facial motion features using
MediaPipe geometric stabilisation and FaceMap SVD decomposition, then uses
accumulated trial-level features to predict leave decisions.

## Recording a session
```
conda activate facepipe
python scripts\recording.py --task space_shooter --participant P01      # player-paced tasks
python scripts\recording_alien_forager.py --participant P01              # Alien Energy Forager
```
`recording.py --task` takes the stem of any player-paced task in `task/` (space_shooter and
future tasks like it). The forager fires at a constant rate, so it has its own recorder
(same recording, plus trial X/N progress and a firing-rate check in `session_info.txt`).
Both check camera framing, launch `task/<task>.py --out-dir <session folder>`, record the
webcam from the moment the task is ready (start screen) until the task window is closed,
and write a `raw_data/session_YYYYMMDD_HHMMSS_<task>/` folder: video, per-frame timestamps,
`task_events.csv` and `session_info.txt`. `task_events.csv` has one row per event
(`timestamp, frame_idx, event, side`; events `trial`, `reward`, `fail`, `switch`,
`system_flip`). The task writes it as it runs, and the recorder adds `frame_idx` at the end.
Parse folder names with `name.split("_", 3)` → `["session", date, time, task]`.

## Pipeline stages
Two branches turn the raw video into per-frame face features; both feed stage 4.

```
raw video ─┬─ st1 preprocess face ─ st2 FaceMap ──┬─ st4 trial features ─ st5 classify
           └─ st2b OpenFace (own 3D alignment) ───┘
```

1. `st1_preprocess_face.py` — head-mounted camera sessions. MediaPipe landmarks on every frame;
   stability check (stops if the face moved in the frame); one fixed crop to a 256×256
   grayscale face video with the background greyed (`preprocessed/<session>/face.avi`); anatomical
   ROI masks from the landmarks (`rois.npz`, preview in `roi_preview.png`). ROIs are defined in
   `REGIONS` / `ROIS` at the top of the script: `whole_face`, `face_no_eyes` (visible eye only
   removed), `face_no_eyes_or_lids`, `eyes`, `brows`,
   `nose`, `mouth`.
2. `st2_run_facemap.py` — FaceMap motion SVD, one run per ROI (`--rois`, default `whole_face
   face_no_eyes`, or `all`); pixels outside each ROI are greyed so they contribute no motion
2b. `st2b_run_openface.py` — OpenFace 2.2 action units, head pose and gaze from the raw video
   *(optional; kept in case it is useful)*
4. `st4_extract_trial_features.py` — align to task events, build accumulated feature matrix
   *(not yet updated for the new st1/st2 outputs or the event-based task_events.csv)*
5. `st5_classify_leave.py` — predict leave decision from face + eye features *(planned)*

Diagnostics (not part of the pipeline) live in `scripts/diagnostics/`: `plot_facemap_masks.py`
(FaceMap spatial masks per ROI), `compare_au_head_pose.py`, `compare_facemap_openface.py`
*(still expects the old st1/st3a outputs)*. Replaced scripts (old MediaPipe st1, stabilisation
st2, rectangle-ROI st3a, camera-stability and head-pose diagnostics) are in `scripts/legacy/`
for reference; they are not maintained and may need path fixes to run.

## Environment setup
Requires conda and an NVIDIA GPU driver supporting CUDA 12.8+. To recreate the exact environment
(from the Anaconda Prompt, inside the project folder):

```
conda env create -f environment.yml
conda activate facepipe
pip install --force-reinstall --no-deps opencv-contrib-python==4.9.0.80
python scripts/verify_env.py
```

The third line makes sure MediaPipe's full OpenCV build is the one in place
(FaceMap and MediaPipe depend on two different OpenCV packages that share the `cv2` folder).
`verify_env.py` should end with `RESULT: ALL CHECKS PASSED`.

> Do not `pip install opencv-python` or upgrade NumPy in this environment —
> FaceMap 1.0.8 requires `numpy<2` and `opencv-python-headless<4.10`.

## Project structure

```
facepipe_project/
├── raw_data/              # original recordings (not tracked by git)
├── stabilised_video/       # affine-stabilised face crops (not tracked)
├── facemap_output/         # FaceMap SVD results (not tracked)
├── mediapipe_output/       # landmark arrays and head pose (not tracked)
├── openface_output/        # OpenFace action units, pose, gaze (not tracked)
├── trial_data/             # trial-level feature matrices (not tracked)
├── notebooks/              # exploratory Jupyter notebooks
├── scripts/                # pipeline stages (st1–st5)
│   └── diagnostics/        # quality checks and plots
├── environment.yml         # conda environment specification
└── requirements-lock.txt   # exact pinned pip versions
```

## Dependencies
See `requirements-lock.txt` for exact pinned versions.
Key packages: PyTorch 2.11 (CUDA 12.8), FaceMap 1.0.8, MediaPipe 1.0.1, OpenCV 4.9.

## Reference
Cazettes et al. (2025). Facial expressions in mice reveal latent cognitive variables
and their neural correlates. Nature Neuroscience.
https://doi.org/10.1038/s41593-025-02071-5
