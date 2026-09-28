# facepipe

Facial and eye analysis pipeline for predicting foraging site-leave decisions.
Human replication of Cazettes et al. (2025, Nature Neuroscience) — Mainen Lab.

## Overview
Participants perform a probabilistic foraging task while face video and eye tracking
data (Pupil Labs) are recorded. The pipeline extracts facial motion features using
MediaPipe geometric stabilisation and FaceMap SVD decomposition, then uses
accumulated trial-level features to predict leave decisions.

## Pipeline stages
1. `mediapipe_landmarks.py` — detect 478 face landmarks + head pose per frame
2. `stabilise_video.py` — affine warp each frame to a canonical face template
3. `run_facemap.py` — SVD on motion energy, extract movement PCs
4. `extract_trial_features.py` — align to task events, build accumulated feature matrix
5. `classify_leave.py` — predict leave decision from face + eye features

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
├── trial_data/             # trial-level feature matrices (not tracked)
├── notebooks/              # exploratory Jupyter notebooks
├── scripts/                # pipeline scripts
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
