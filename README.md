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
python scripts\record_space_shooter.py --participant P01
```
Checks camera framing, launches the Space Shooter task (`task/space_shooter.py`), records
the webcam from the moment SPACE is pressed on the start screen until the game-over screen,
and writes a pipeline-ready `raw_video/session_YYYYMMDD_HHMMSS/` folder (video, per-frame
timestamps, `task_events.csv`, `task_markers.csv`, `session_info.txt`).
`record_session.py` is the manual (S/Q keys) recorder for other tasks.

## Pipeline stages
Two branches turn the raw video into per-frame face features; both feed stage 4.

```
raw video ─┬─ st1 MediaPipe ─ st2 stabilise ─ st3a FaceMap ──┬─ st4 trial features ─ st5 classify
           └─ st3b OpenFace (own 3D alignment) ──────────────┘
```

1. `st1_mediapipe_landmarks.py` — detect 478 face landmarks + head pose per frame
2. `st2_stabilise_video.py` — similarity-warp each frame to a canonical face template, mask the face oval
3. a. `st3a_run_facemap.py` — SVD on motion energy, extract movement PCs
   b. `st3b_run_openface.py` — OpenFace 2.2 action units, head pose and gaze from the raw video *(planned)*
4. `st4_extract_trial_features.py` — align to task events, build accumulated feature matrix
5. `st5_classify_leave.py` — predict leave decision from face + eye features *(planned)*

Diagnostics (not part of the pipeline) live in `scripts/diagnostics/`:
`check_pc_head_motion.py` (FaceMap PCs vs head pose), `plot_facemap_masks.py`, `draw_roi.py`.

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
├── raw_video/              # original recordings (not tracked by git)
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
