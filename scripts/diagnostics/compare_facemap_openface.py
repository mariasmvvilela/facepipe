"""Diagnostic: what do the FaceMap PCs contain? Compared against OpenFace action units.

Three checks on one session:

1. PC x AU correlations, per FaceMap ROI. Motion SVD is built from frame differences,
   so each AU is tested both as its level and as its change |d AU| (the motion-energy
   equivalent). A PC that tracks |d AU45| is picking up blinks.

2. Variance partition per PC: how much of the PC's variance is explained by
       head  = MediaPipe yaw/pitch/roll + |d yaw/pitch/roll|         (6 regressors)
       AUs   = OpenFace AU intensities + |d AU intensities|           (2 per active AU)
   fitted separately and together (linear, in-sample R^2):
       unique to AUs  = R2(both) - R2(head)
       unique to head = R2(both) - R2(AUs)
       shared         = R2(head) + R2(AUs) - R2(both)
   A PC with a large "unique to head" and little AU share is most likely stabilisation
   artifact; a large "shared" part means facial movement and head movement happen
   together (e.g. blinks during head turns) and one session can't separate them.
   ~40 regressors on ~9000 frames inflates in-sample R^2 by < 0.01. AUs that are
   almost never active (MIN_ACTIVE_FRAC) are left out.

3. Head pose agreement between MediaPipe (stage 1) and OpenFace (stage 3b), the
   pose estimate the head-motion checks rely on.

Writes to openface_output/<session>/compare_facemap/:
    pc_au_corr_<roi>.png        |d AU| and AU-level correlation heatmaps
    variance_partition.png      stacked R^2 bars per PC, both ROIs
    head_pose_agreement.png     MediaPipe vs OpenFace yaw/pitch/roll over time
    comparison.json             all numbers

Usage (inside the facepipe env, from the project folder, after st3a and st3b):
    python scripts\\diagnostics\\compare_facemap_openface.py
    python scripts\\diagnostics\\compare_facemap_openface.py --session session_YYYYMMDD_HHMMSS_<task>
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
DEFAULT_SESSION = "session_20260928_161612_alien_energy_forager"
ROIS = ["eyes_brows", "lower_face"]
N_PCS = 10            # same number stage 4 uses
POSE_NAMES = ["yaw", "pitch", "roll"]
MIN_ACTIVE_FRAC = 0.01   # leave out AUs with intensity > 0 on fewer than 1% of frames


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def corr_matrix(a, b):
    """Pearson r between every column of a (n, i) and every column of b (n, j) -> (i, j)."""
    return zscore(a).T @ zscore(b) / len(a)


def r_squared(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return 1 - (y - X1 @ coef).var() / y.var()


def with_change(x):
    """Stack a signal with its absolute frame-to-frame change: (n, k) -> (n, 2k)."""
    return np.column_stack([x, np.abs(np.diff(x, axis=0, prepend=np.nan))])


def heatmap(ax, r, xlabels, ylabels, title):
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if abs(r[i, j]) >= 0.2:
                ax.text(j, i, "{:.2f}".format(r[i, j]).replace("0.", "."), ha="center",
                        va="center", fontsize=6, color="white" if abs(r[i, j]) > 0.5 else "black")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION)
    args = parser.parse_args()

    fm_dir = PROJECT_DIR / "facemap_output" / args.session
    of_dir = PROJECT_DIR / "openface_output" / args.session
    mp_dir = PROJECT_DIR / "mediapipe_output" / args.session
    out_dir = of_dir / "compare_facemap"
    paths = {"au": of_dir / "au_intensity.npy", "au_names": of_dir / "au_intensity_names.txt",
             "of_valid": of_dir / "valid_mask.npy", "of_pose": of_dir / "head_pose.npy",
             "mp_pose": mp_dir / "head_pose.npy", "mp_summary": mp_dir / "summary.json",
             **{roi: fm_dir / "{}_PCs.npy".format(roi) for roi in ROIS}}
    for p in paths.values():
        if not p.is_file():
            sys.exit("ERROR: not found: {} (run st3a/st3b first)".format(p))
    out_dir.mkdir(exist_ok=True)

    au = np.load(paths["au"]).astype(np.float64)
    au_names = paths["au_names"].read_text().split()
    # AUs that are (almost) never active carry no signal and have ~zero variance.
    active = np.nanmean(au > 0, axis=0) >= MIN_ACTIVE_FRAC
    dropped_aus = [a for a, keep in zip(au_names, active) if not keep]
    au, au_names = au[:, active], [a for a, keep in zip(au_names, active) if keep]
    print("AUs active on < {:.0%} of frames, left out: {}".format(MIN_ACTIVE_FRAC, dropped_aus or "none"))
    of_valid = np.load(paths["of_valid"])
    of_pose = np.load(paths["of_pose"]).astype(np.float64)
    mp_pose = np.load(paths["mp_pose"]).astype(np.float64)
    with open(paths["mp_summary"]) as f:
        fps = json.load(f)["fps"]
    pcs = {roi: np.load(paths[roi])[:, :N_PCS].astype(np.float64) for roi in ROIS}
    n = len(au)
    if any(len(a) != n for a in (mp_pose, *pcs.values())):
        sys.exit("ERROR: frame counts differ: OpenFace {}, MediaPipe {}, FaceMap {}".format(
            n, len(mp_pose), [len(v) for v in pcs.values()]))

    head_X = with_change(mp_pose)          # (n, 6)
    au_X = with_change(au)                 # (n, 2 * n_aus): levels then |d|
    # Frame t needs t and t-1 valid in both trackers (the change terms use both);
    # frame 0 is FaceMap's copy of frame 1.
    valid = np.isfinite(head_X).all(axis=1) & np.isfinite(au_X).all(axis=1)
    valid[0] = False
    print("{} frames, {} used ({:.1%}); OpenFace valid on {:.1%} of frames".format(
        n, valid.sum(), valid.mean(), of_valid.mean()))

    pc_names = ["PC{}".format(k + 1) for k in range(N_PCS)]
    result = {"session": args.session, "frames_used": int(valid.sum()),
              "aus_used": au_names, "aus_dropped_inactive": dropped_aus, "rois": {}}

    # --- 1 + 2: correlations and variance partition per ROI ---------------------------
    partitions = {}
    for roi in ROIS:
        v = pcs[roi][valid]
        r_level = corr_matrix(v, au_X[valid, :len(au_names)])    # (N_PCS, 17)
        r_change = corr_matrix(v, au_X[valid, len(au_names):])
        part = []
        for k in range(N_PCS):
            r2_h = r_squared(head_X[valid], v[:, k])
            r2_a = r_squared(au_X[valid], v[:, k])
            r2_b = r_squared(np.column_stack([head_X[valid], au_X[valid]]), v[:, k])
            part.append({"r2_head": r2_h, "r2_aus": r2_a, "r2_both": r2_b,
                         "unique_aus": r2_b - r2_h, "unique_head": r2_b - r2_a,
                         "shared": r2_h + r2_a - r2_b})
        partitions[roi] = part

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
        heatmap(axes[0], r_change, ["|d {}|".format(a) for a in au_names], pc_names,
                "{}: PC vs |d AU| (change)".format(roi))
        im = heatmap(axes[1], r_level, au_names, pc_names, "{}: PC vs AU level".format(roi))
        fig.colorbar(im, ax=axes[1], fraction=0.03, label="r")
        fig.suptitle("{} — values shown where |r| >= 0.2".format(args.session), fontsize=10)
        fig.savefig(out_dir / "pc_au_corr_{}.png".format(roi), dpi=130)
        plt.close(fig)

        top = lambda r, k: sorted(zip(au_names, r[k]), key=lambda t: -abs(t[1]))[:3]
        result["rois"][roi] = {
            pc_names[k]: {
                "top_au_change": {a: round(float(x), 3) for a, x in top(r_change, k)},
                "top_au_level": {a: round(float(x), 3) for a, x in top(r_level, k)},
                **{key: round(float(val), 3) for key, val in part[k].items()},
            } for k in range(N_PCS)
        }

    # Variance partition figure
    fig, axes = plt.subplots(1, len(ROIS), figsize=(13, 3.8), sharey=True, layout="constrained")
    for ax, roi in zip(axes, ROIS):
        part = partitions[roi]
        x = np.arange(N_PCS)
        ua = np.array([p["unique_aus"] for p in part])
        sh = np.clip([p["shared"] for p in part], 0, None)
        uh = np.array([p["unique_head"] for p in part])
        ax.bar(x, ua, color="tab:green", label="unique to AUs")
        ax.bar(x, sh, bottom=ua, color="0.65", label="shared")
        ax.bar(x, uh, bottom=ua + sh, color="tab:red", label="unique to head motion")
        ax.set_xticks(x, pc_names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(roi, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("R² of PC")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("{}: FaceMap PC variance explained by OpenFace AUs vs head motion".format(
        args.session), fontsize=10)
    fig.savefig(out_dir / "variance_partition.png", dpi=130)
    plt.close(fig)

    # --- 3: head pose agreement ------------------------------------------------------------
    both = np.isfinite(mp_pose).all(axis=1) & np.isfinite(of_pose).all(axis=1)
    r_pose = corr_matrix(mp_pose[both], of_pose[both])     # (3 MediaPipe, 3 OpenFace)
    t = np.arange(n) / fps
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, layout="constrained")
    agreement = {}
    for i, (ax, name) in enumerate(zip(axes, POSE_NAMES)):
        sign = np.sign(r_pose[i, i])
        of_aligned = sign * of_pose[:, i]
        # Offset only (same units, degrees); mean difference over frames both tracked.
        offset = np.nanmean(mp_pose[both, i] - of_aligned[both])
        ax.plot(t, mp_pose[:, i], lw=0.6, label="MediaPipe")
        ax.plot(t, of_aligned + offset, lw=0.6, alpha=0.8,
                label="OpenFace ({}1 x, {:+.1f}° offset)".format("+" if sign > 0 else "−", offset))
        ax.set_ylabel("{} (deg)".format(name), fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        agreement[name] = {"r": round(float(r_pose[i, i]), 3), "sign": int(sign),
                           "offset_deg": round(float(offset), 2),
                           "rmse_after_offset_deg": round(float(np.sqrt(np.nanmean(
                               (mp_pose[both, i] - of_aligned[both] - offset) ** 2))), 2)}
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("{}: head pose, MediaPipe vs OpenFace".format(args.session), fontsize=10)
    fig.savefig(out_dir / "head_pose_agreement.png", dpi=130)
    plt.close(fig)
    result["head_pose_agreement"] = agreement
    result["head_pose_corr_matrix"] = {"rows_mediapipe": POSE_NAMES, "cols_openface": POSE_NAMES,
                                       "r": np.round(r_pose, 3).tolist()}

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    # --- Console summary ------------------------------------------------------------------
    for roi in ROIS:
        print("\n{}  (R²: head / AUs / both;  unique AU / shared / unique head;  top |d AU|)".format(roi))
        for k in range(N_PCS):
            p = partitions[roi][k]
            best = next(iter(result["rois"][roi][pc_names[k]]["top_au_change"].items()))
            print("  {:5s} {:.2f} / {:.2f} / {:.2f}   {:.2f} / {:.2f} / {:.2f}   {} {:+.2f}".format(
                pc_names[k], p["r2_head"], p["r2_aus"], p["r2_both"],
                p["unique_aus"], p["shared"], p["unique_head"], best[0], best[1]))
    print("\nHead pose agreement (MediaPipe vs OpenFace):")
    for name, a in agreement.items():
        print("  {:5s} r = {:+.2f}, RMSE after offset {:.1f}°".format(name, a["r"], a["rmse_after_offset_deg"]))
    print("\nSaved to {}".format(out_dir))


if __name__ == "__main__":
    main()
