"""Diagnostic: how much residual head movement is left in the FaceMap PCs?

Correlates the first N_PCS motion-SVD time courses of one FaceMap ROI with head
pose from stage 1. Motion SVD is built from frame differences, so a rigid head
movement shows up as *change* in pose, not pose itself. Both are tested:

    pose   yaw, pitch, roll (deg)             -> slow drifts, e.g. looking down
    speed  |d yaw|, |d pitch|, |d roll| (deg/frame) -> movement, what motion energy sees

Frame t of the motion PCs is |frame(t) - frame(t-1)|, so d pose(t) = pose(t) - pose(t-1)
lines up with it exactly. Frame 0 (a FaceMap copy of frame 1) is dropped.

PC signs are arbitrary, so read |r|. The time series are strongly autocorrelated,
so p-values would be meaningless and aren't reported.

Writes to facemap_output/<session>/ (<roi> = the --roi label):
    <roi>_head_motion_timeseries.png   pose + speed + PCs on one time axis
    <roi>_head_motion_corr.png         correlation heatmaps (pose, speed) + R^2 per PC
    <roi>_head_motion_corr.json        the same numbers

Usage (inside the facepipe env, from the project folder):
    python scripts\\diagnostics\\check_pc_head_motion.py
    python scripts\\diagnostics\\check_pc_head_motion.py --session session_YYYYMMDD_HHMMSS --roi eyes_brows
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SESSION = "session_20260928_161612"
N_PCS = 6
POSE_NAMES = ["yaw", "pitch", "roll"]


def corr_matrix(a, b):
    """Pearson r between every column of a (n, i) and every column of b (n, j) -> (i, j)."""
    a = (a - a.mean(axis=0)) / a.std(axis=0)
    b = (b - b.mean(axis=0)) / b.std(axis=0)
    return a.T @ b / len(a)


def r_squared(X, y):
    """Fraction of y's variance explained by a linear fit on X (with intercept)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    return 1 - resid.var() / y.var()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--roi", default="lower_face", help="FaceMap ROI label (lower_face, eyes_brows)")
    args = parser.parse_args()

    fm_dir = PROJECT_DIR / "facemap_output" / args.session
    pcs_path = fm_dir / "{}_PCs.npy".format(args.roi)
    varexp_path = fm_dir / "{}_varexp.npy".format(args.roi)
    pose_path = PROJECT_DIR / "mediapipe_output" / args.session / "head_pose.npy"
    mp_summary_path = PROJECT_DIR / "mediapipe_output" / args.session / "summary.json"
    for p in (pcs_path, pose_path, mp_summary_path):
        if not p.is_file():
            sys.exit("ERROR: not found: {}".format(p))

    pcs = np.load(pcs_path)[:, :N_PCS].astype(np.float64)
    pose = np.load(pose_path).astype(np.float64)
    varexp = np.load(varexp_path) if varexp_path.is_file() else None
    with open(mp_summary_path) as f:
        fps = json.load(f)["fps"]
    if len(pcs) != len(pose):
        sys.exit("ERROR: PCs have {} frames but head pose has {}".format(len(pcs), len(pose)))

    speed = np.abs(np.diff(pose, axis=0, prepend=np.nan))
    valid = np.isfinite(pose).all(axis=1) & np.isfinite(speed).all(axis=1)
    valid[0] = False   # FaceMap's duplicated first frame
    print("{} ROI, {} frames, {} used ({} dropped: frame 0 / no face / neighbour of no face)".format(
        args.roi, len(pcs), valid.sum(), (~valid).sum()))

    p, s, v = pose[valid], speed[valid], pcs[valid]
    r_pose = corr_matrix(p, v)      # (3, N_PCS)
    r_speed = corr_matrix(s, v)
    regressors = np.column_stack([p, s])
    r2 = np.array([r_squared(regressors, v[:, k]) for k in range(N_PCS)])

    pc_names = ["PC{}".format(k + 1) for k in range(N_PCS)]
    speed_names = ["|d {}|".format(n) for n in POSE_NAMES]

    # --- Time series ---------------------------------------------------------------
    t = np.arange(len(pcs)) / fps
    rows = [(POSE_NAMES[i], pose[:, i], "deg", "tab:blue") for i in range(3)]
    rows += [(speed_names[i], speed[:, i], "deg/fr", "tab:orange") for i in range(3)]
    rows += [(pc_names[k], np.where(np.arange(len(pcs)) == 0, np.nan, pcs[:, k]), "a.u.", "0.2")
             for k in range(N_PCS)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 1.1 * len(rows)), sharex=True)
    for ax, (name, y, unit, colour) in zip(axes, rows):
        ax.plot(t, y, lw=0.5, color=colour)
        ax.set_ylabel("{}\n({})".format(name, unit), rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for k in range(N_PCS):
        ax = axes[6 + k]
        best = max([(abs(r_pose[i, k]), POSE_NAMES[i], r_pose[i, k]) for i in range(3)] +
                   [(abs(r_speed[i, k]), speed_names[i], r_speed[i, k]) for i in range(3)])
        ax.text(1.002, 0.5, "max |r|: {} {:+.2f}\nR² all 6: {:.2f}".format(best[1], best[2], r2[k]),
                transform=ax.transAxes, fontsize=7, va="center")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("{}: head pose vs {} motion PCs".format(args.session, args.roi))
    fig.tight_layout()
    fig.savefig(fm_dir / "{}_head_motion_timeseries.png".format(args.roi), dpi=130)
    plt.close(fig)

    # --- Correlation heatmaps --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, r, names, title in ((axes[0], r_pose, POSE_NAMES, "pose"),
                                (axes[1], r_speed, speed_names, "speed (|Δ pose|)")):
        im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(N_PCS), pc_names)
        ax.set_yticks(range(3), names)
        ax.set_title("r: {} vs PC".format(title), fontsize=10)
        for i in range(3):
            for k in range(N_PCS):
                ax.text(k, i, "{:+.2f}".format(r[i, k]), ha="center", va="center", fontsize=8,
                        color="white" if abs(r[i, k]) > 0.5 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.05, label="r")
    axes[2].bar(pc_names, r2, color="0.4")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("R²: PC ~ pose + speed", fontsize=10)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("{} ({})".format(args.session, args.roi), fontsize=10)
    fig.savefig(fm_dir / "{}_head_motion_corr.png".format(args.roi), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Numbers ---------------------------------------------------------------------
    result = {
        "session": args.session,
        "roi": args.roi,
        "frames_used": int(valid.sum()),
        "note": "PC signs are arbitrary; read |r|. No p-values: series are autocorrelated.",
        "pcs": {
            pc_names[k]: {
                "variance_explained": None if varexp is None else round(float(varexp[k]), 4),
                "r_pose": {n: round(float(r_pose[i, k]), 3) for i, n in enumerate(POSE_NAMES)},
                "r_speed": {n: round(float(r_speed[i, k]), 3) for i, n in enumerate(POSE_NAMES)},
                "r2_pose_and_speed": round(float(r2[k]), 3),
            } for k in range(N_PCS)
        },
    }
    with open(fm_dir / "{}_head_motion_corr.json".format(args.roi), "w") as f:
        json.dump(result, f, indent=2)

    print("\n{:5s} {:>7s}  {:>20s}  {:>26s}  {:>5s}".format(
        "", "varexp", "r pose (y, p, r)", "r speed (|dy|, |dp|, |dr|)", "R²"))
    for k in range(N_PCS):
        print("{:5s} {:>7s}  {:>6.2f} {:>6.2f} {:>6.2f}  {:>8.2f} {:>8.2f} {:>8.2f}  {:>5.2f}".format(
            pc_names[k], "-" if varexp is None else "{:.3f}".format(varexp[k]),
            *r_pose[:, k], *r_speed[:, k], r2[k]))
    print("\nSaved to {}: {r}_head_motion_timeseries.png, {r}_head_motion_corr.png, "
          "{r}_head_motion_corr.json".format(fm_dir, r=args.roi))


if __name__ == "__main__":
    main()
