# facepipe

Face-video pipeline for predicting foraging site-leave decisions.
Human replication of Cazettes et al. (2025, Nature Neuroscience) — Mainen Lab.

## Overview
Participants perform a probabilistic foraging task while a head-mounted camera records
their face. The pipeline crops a fixed, landmark-aligned face video, extracts facial
motion components per face region with FaceMap's motion SVD, accumulates them over each
site visit, and tests whether they predict the decision to leave.

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
Run each stage with `--session session_YYYYMMDD_HHMMSS_<task>` (a folder name in `raw_data/`,
or its path).

```
raw_data ─ st1 preprocess face ─ st2 FaceMap ─ st3 trial features ─ st4 classify leave
           preprocessed/          facemap_output/  trial_data/         trial_data/<session>/models/
```

1. `st1_preprocess_face.py` — MediaPipe landmarks on every frame; stability check (stops if the
   face moved in the frame); one fixed crop to a 256×256 grayscale face video with the
   background greyed (`face.avi`); ROI masks from the median landmarks (`rois.npz`, preview in
   `roi_preview.png`). `--rois-only` rebuilds just the ROIs after editing them.
2. `st2_run_facemap.py` — FaceMap motion SVD, one run per ROI (`--rois`, default: the main
   four, or `all`); pixels outside each ROI are greyed so they contribute no motion.
   Writes `<roi>_PCs.npy` and plots of each ROI's motion and PC spatial masks.
3. `st3_trial_features.py` — aligns trials to video frames, labels leave/stay, and builds
   accumulated per-visit features (task + face PCs) for every ROI that has PCs.
4. `st4_classify_leave.py` — elastic-net logistic regression on task only / face only /
   task + face features, leave-one-visit-out AUC. `--rois` picks which ROIs' face features to
   use (default: the main four), so ROI sets are compared without rebuilding features.

ROIs are defined once, in `scripts/roi_definitions.py`: horizontal bands of the face between
two MediaPipe landmarks, at full face width. `DEFAULT_ROIS` are the main analysis
(`whole_face`, `upper_face`, `lower_face`, `upper_face_no_eyes`); the others are experimental
bands within the upper face.

Shared code: `common.py` (project paths, `--session` handling, figure saving) and `trials.py`
(task run selection, trial → frame alignment, leave labels and site visits).

Diagnostics (not part of the pipeline) live in `scripts/diagnostics/`:
`event_motion.py` — raw pixel motion energy around each trial, leave vs stay, per ROI, with
per-pixel maps; a check on the FaceMap features using plain motion.
Replaced scripts (old MediaPipe st1, stabilisation, rectangle ROIs, OpenFace and its
comparisons) are in `scripts/legacy/` for reference; they are not maintained and may need
path fixes to run.

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
├── raw_data/               # recordings: video, frametimes, task_events.csv (not tracked by git)
├── preprocessed/           # st1: face.avi, landmarks, ROI masks (not tracked)
├── facemap_output/         # st2: FaceMap PCs and plots per ROI (not tracked)
├── trial_data/             # st3 features, st4 model results (not tracked)
├── notebooks/              # analysis notebooks
├── scripts/                # pipeline stages st1–st4, recorders, shared modules
│   ├── diagnostics/        # quality checks, not part of the pipeline
│   └── legacy/             # replaced scripts, for reference
├── task/                   # foraging tasks (see task/README.md)
├── docs/                   # environment verification log
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
