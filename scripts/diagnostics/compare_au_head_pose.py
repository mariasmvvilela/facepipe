"""Diagnostic: how do OpenFace's action units relate to its own head pose estimate?

OpenFace estimates head pose and AUs from the same fit (AUs are read off a face that
is aligned with the estimated pose), so the two are not independent: a pose error, or
the AU models' limited robustness to large rotations, can make an AU look like it
follows the head. It can also be real behaviour, e.g. blinks (AU45) during head turns
or brow raises (AU01/02) when looking up. This checks how much of each AU is
predictable from head pose and head movement in one session:

1. Correlations. AU level vs pose (yaw, pitch, roll) and AU change |d AU| vs head
   speed |d yaw|, |d pitch|, |d roll|.

2. R^2 per AU (linear, in-sample) from
       pose      yaw, pitch, roll and their squares (centred)        (6 regressors)
       movement  |d yaw|, |d pitch|, |d roll|                          (3 regressors)
   fitted separately and together. The squares catch U-shaped effects, e.g. an AU
   that rises whenever the head turns away from frontal in either direction.
   The |d AU| of each AU is fitted on movement alone.

3. Pose tuning: mean AU intensity in pose-decile bins of each axis (not linear, so
   it shows thresholds, e.g. an AU that only switches on beyond 20 deg of yaw).

4. Lagged correlation between total head speed and |d AU| (+-LAG_S seconds):
   a peak at lag 0 that is narrow is typical of tracking artefacts (same frame);
   a peak before/after 0 is more likely behaviour (e.g. a blink leading a turn).

5. Time series: pose, head speed and the AUs most predictable from the head.

The time series are strongly autocorrelated, so p-values would be meaningless and
are not reported. AUs that are almost never active (MIN_ACTIVE_FRAC) are left out.

Writes to openface_output/<session>/au_vs_head_pose/:
    au_head_corr.png         correlation heatmaps (AU vs pose, |d AU| vs head speed)
    au_head_r2.png           R^2 per AU from pose / movement / both, and |d AU| from movement
    au_pose_tuning.png       mean AU intensity by pose-decile bin, one panel per axis
    au_head_speed_xcorr.png  lagged correlation, head speed vs |d AU|
    au_head_timeseries.png   pose, head speed and the most head-related AUs over time
    au_head_pose.json        all numbers

Usage (inside the facepipe env, from the project folder, after st3b):
    python scripts\\diagnostics\\compare_au_head_pose.py
    python scripts\\diagnostics\\compare_au_head_pose.py --session session_YYYYMMDD_HHMMSS
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
POSE_NAMES = ["yaw", "pitch", "roll"]
MIN_ACTIVE_FRAC = 0.01   # leave out AUs with intensity > 0 on fewer than 1% of frames
N_BINS = 10              # pose deciles for the tuning curves
LAG_S = 2.0              # +- range of the lagged correlation
N_TIMESERIES_AUS = 4     # AUs shown in the time series figure


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def corr_matrix(a, b):
    """Pearson r between every column of a (n, i) and every column of b (n, j) -> (i, j)."""
    return zscore(a).T @ zscore(b) / len(a)


def r_squared(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return 1 - (y - X1 @ coef).var() / y.var()


def lagged_corr(x, Y, valid, max_lag):
    """r(x[t], Y[t + lag]) for lag in -max_lag..max_lag, only where both frames are valid.

    x (n,), Y (n, k), valid (n,) -> (2 * max_lag + 1, k). Positive lag = Y after x.
    """
    n = len(x)
    out = np.full((2 * max_lag + 1, Y.shape[1]), np.nan)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        a = slice(max(0, -lag), n - max(0, lag))
        b = slice(max(0, lag), n - max(0, -lag))
        ok = valid[a] & valid[b]
        out[i] = corr_matrix(x[a][ok, None], Y[b][ok])[0]
    return out


def heatmap(ax, r, xlabels, ylabels, title):
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            ax.text(j, i, "{:.2f}".format(r[i, j]).replace("0.", "."), ha="center",
                    va="center", fontsize=7, color="white" if abs(r[i, j]) > 0.5 else "black")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION)
    args = parser.parse_args()

    of_dir = PROJECT_DIR / "openface_output" / args.session
    out_dir = of_dir / "au_vs_head_pose"
    paths = {"au": of_dir / "au_intensity.npy", "au_names": of_dir / "au_intensity_names.txt",
             "pose": of_dir / "head_pose.npy", "summary": of_dir / "summary.json"}
    for p in paths.values():
        if not p.is_file():
            sys.exit("ERROR: not found: {} (run st3b first)".format(p))
    out_dir.mkdir(exist_ok=True)

    au = np.load(paths["au"]).astype(np.float64)
    au_names = paths["au_names"].read_text().split()
    active = np.nanmean(au > 0, axis=0) >= MIN_ACTIVE_FRAC
    dropped_aus = [a for a, keep in zip(au_names, active) if not keep]
    au, au_names = au[:, active], [a for a, keep in zip(au_names, active) if keep]
    print("AUs active on < {:.0%} of frames, left out: {}".format(MIN_ACTIVE_FRAC, dropped_aus or "none"))
    pose = np.load(paths["pose"]).astype(np.float64)
    with open(paths["summary"]) as f:
        fps = json.load(f)["fps"]
    n, n_aus = au.shape

    d_pose = np.abs(np.diff(pose, axis=0, prepend=np.nan))       # deg/frame
    d_au = np.abs(np.diff(au, axis=0, prepend=np.nan))
    head_speed = np.sqrt(np.sum(np.diff(pose, axis=0, prepend=np.nan) ** 2, axis=1)) * fps  # deg/s
    # Frame t needs t and t-1 valid (the change terms use both).
    valid = (np.isfinite(pose).all(axis=1) & np.isfinite(au).all(axis=1)
             & np.isfinite(d_pose).all(axis=1) & np.isfinite(d_au).all(axis=1))
    print("{} frames, {} used ({:.1%})".format(n, valid.sum(), valid.mean()))

    pose_v, au_v, d_pose_v, d_au_v = pose[valid], au[valid], d_pose[valid], d_au[valid]
    pose_c = pose_v - pose_v.mean(axis=0)
    X_pose = np.column_stack([pose_c, pose_c ** 2])
    X_move = d_pose_v
    X_both = np.column_stack([X_pose, X_move])

    # --- 1: correlations ------------------------------------------------------------
    r_level = corr_matrix(au_v, pose_v)          # (n_aus, 3)
    r_change = corr_matrix(d_au_v, d_pose_v)     # (n_aus, 3)
    fig, axes = plt.subplots(1, 2, figsize=(8, 0.32 * n_aus + 1.6), layout="constrained")
    heatmap(axes[0], r_level, POSE_NAMES, au_names, "AU level vs pose")
    im = heatmap(axes[1], r_change, ["|d {}|".format(p) for p in POSE_NAMES],
                 ["|d {}|".format(a) for a in au_names], "AU change vs head speed")
    fig.colorbar(im, ax=axes[1], fraction=0.05, label="r")
    fig.suptitle("{}: OpenFace AUs vs OpenFace head pose".format(args.session), fontsize=10)
    fig.savefig(out_dir / "au_head_corr.png", dpi=130)
    plt.close(fig)

    # --- 2: R^2 per AU ----------------------------------------------------------------
    r2 = {}
    for j, name in enumerate(au_names):
        r2[name] = {"pose": r_squared(X_pose, au_v[:, j]),
                    "movement": r_squared(X_move, au_v[:, j]),
                    "both": r_squared(X_both, au_v[:, j]),
                    "change_from_movement": r_squared(X_move, d_au_v[:, j])}
    x = np.arange(n_aus)
    w = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for off, key, color in ((-w, "pose", "tab:blue"), (0, "movement", "tab:orange"), (w, "both", "0.35")):
        axes[0].bar(x + off, [r2[a][key] for a in au_names], w, color=color, label=key)
    axes[0].set_ylabel("R² of AU level")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].bar(x, [r2[a]["change_from_movement"] for a in au_names], 0.6, color="tab:orange")
    axes[1].set_ylabel("R² of |d AU|\nfrom head speed")
    axes[1].set_xticks(x, au_names, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("{}: AU variance explained by head pose (yaw/pitch/roll + squares) "
                 "and head movement (|d pose|)".format(args.session), fontsize=10)
    fig.savefig(out_dir / "au_head_r2.png", dpi=130)
    plt.close(fig)

    # --- 3: pose tuning ---------------------------------------------------------------
    tuning = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.3 * n_aus + 1.8), sharey=True, layout="constrained")
    au_max = au_v.max(axis=0)
    for i, (ax, pname) in enumerate(zip(axes, POSE_NAMES)):
        edges = np.percentile(pose_v[:, i], np.linspace(0, 100, N_BINS + 1))
        bin_idx = np.clip(np.digitize(pose_v[:, i], edges[1:-1]), 0, N_BINS - 1)
        means = np.array([au_v[bin_idx == b].mean(axis=0) for b in range(N_BINS)]).T   # (n_aus, bins)
        centres = np.array([np.median(pose_v[bin_idx == b, i]) for b in range(N_BINS)])
        # Colour scaled per AU (row) so weak AUs are visible; the numbers are raw intensity.
        scaled = means / np.where(au_max > 0, au_max, 1)[:, None]
        ax.imshow(scaled / np.maximum(scaled.max(axis=1, keepdims=True), 1e-9),
                  cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for a in range(n_aus):
            for b in range(N_BINS):
                ax.text(b, a, "{:.2f}".format(means[a, b]).replace("0.", "."), ha="center",
                        va="center", fontsize=6, color="white" if scaled[a, b] < 0.6 * scaled[a].max() else "black")
        ax.set_xticks(range(N_BINS), ["{:.0f}".format(c) for c in centres], fontsize=8)
        ax.set_xlabel("{} decile, median (deg)".format(pname), fontsize=9)
        ax.set_yticks(range(n_aus), au_names, fontsize=8)
        tuning[pname] = {"bin_median_deg": np.round(centres, 2).tolist(),
                         "mean_au": {a: np.round(means[k], 3).tolist() for k, a in enumerate(au_names)}}
    fig.suptitle("{}: mean AU intensity per pose decile (colour scaled per AU)".format(args.session),
                 fontsize=10)
    fig.savefig(out_dir / "au_pose_tuning.png", dpi=130)
    plt.close(fig)

    # --- 4: lagged correlation, head speed vs |d AU| ----------------------------------------
    max_lag = int(round(LAG_S * fps))
    lags_s = np.arange(-max_lag, max_lag + 1) / fps
    xc = lagged_corr(head_speed, d_au, valid, max_lag)          # (lags, n_aus)
    fig, ax = plt.subplots(figsize=(9, 0.3 * n_aus + 1.6), layout="constrained")
    lim = max(0.1, np.nanmax(np.abs(xc)))
    im = ax.imshow(xc.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                   extent=[lags_s[0], lags_s[-1], n_aus - 0.5, -0.5])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(n_aus), ["|d {}|".format(a) for a in au_names], fontsize=8)
    ax.set_xlabel("lag (s); > 0 = AU change after head movement")
    fig.colorbar(im, ax=ax, fraction=0.04, label="r")
    ax.set_title("{}: head speed vs AU change, lagged correlation".format(args.session), fontsize=10)
    fig.savefig(out_dir / "au_head_speed_xcorr.png", dpi=130)
    plt.close(fig)
    xcorr = {}
    for k, a in enumerate(au_names):
        peak = int(np.nanargmax(np.abs(xc[:, k])))
        xcorr[a] = {"r_lag0": round(float(xc[max_lag, k]), 3), "r_peak": round(float(xc[peak, k]), 3),
                    "peak_lag_s": round(float(lags_s[peak]), 3)}

    # --- 5: time series ----------------------------------------------------------------------
    top_aus = sorted(au_names, key=lambda a: -r2[a]["both"])[:N_TIMESERIES_AUS]
    t = np.arange(n) / fps
    fig, axes = plt.subplots(2 + len(top_aus), 1, figsize=(14, 1.5 * (2 + len(top_aus)) + 1),
                             sharex=True, layout="constrained")
    for i, pname in enumerate(POSE_NAMES):
        axes[0].plot(t, pose[:, i] - np.nanmedian(pose[:, i]), lw=0.6, label=pname)
    axes[0].set_ylabel("pose − median\n(deg)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=3)
    axes[1].plot(t, head_speed, lw=0.5, color="k")
    axes[1].set_ylabel("head speed\n(deg/s)", fontsize=8)
    for ax, a in zip(axes[2:], top_aus):
        ax.plot(t, au[:, au_names.index(a)], lw=0.6, color="tab:green")
        ax.set_ylabel("{}\nR²={:.2f}".format(a, r2[a]["both"]), fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("{}: head pose and the AUs most predictable from it".format(args.session), fontsize=10)
    fig.savefig(out_dir / "au_head_timeseries.png", dpi=130)
    plt.close(fig)

    # --- Save ----------------------------------------------------------------------------------
    result = {
        "session": args.session, "frames_used": int(valid.sum()), "fps": fps,
        "aus_used": au_names, "aus_dropped_inactive": dropped_aus,
        "pose_summary_deg": {p: {"median": round(float(np.median(pose_v[:, i])), 2),
                                 "p05": round(float(np.percentile(pose_v[:, i], 5)), 2),
                                 "p95": round(float(np.percentile(pose_v[:, i], 95)), 2)}
                             for i, p in enumerate(POSE_NAMES)},
        "head_speed_deg_s": {"median": round(float(np.median(head_speed[valid])), 2),
                             "p95": round(float(np.percentile(head_speed[valid], 95)), 2)},
        "per_au": {a: {"r_level_vs_pose": dict(zip(POSE_NAMES, np.round(r_level[k], 3).tolist())),
                       "r_change_vs_speed": dict(zip(POSE_NAMES, np.round(r_change[k], 3).tolist())),
                       "r2": {key: round(float(v), 3) for key, v in r2[a].items()},
                       "xcorr_head_speed": xcorr[a]}
                   for k, a in enumerate(au_names)},
        "pose_tuning": tuning,
    }
    with open(out_dir / "au_head_pose.json", "w") as f:
        json.dump(result, f, indent=2)

    # --- Console summary ------------------------------------------------------------------------
    print("\nPose (deg, 5-95%): " + "  ".join("{} {:+.1f}..{:+.1f}".format(
        p, v["p05"], v["p95"]) for p, v in result["pose_summary_deg"].items()))
    print("Head speed: median {:.1f} deg/s, 95th pct {:.1f} deg/s".format(
        result["head_speed_deg_s"]["median"], result["head_speed_deg_s"]["p95"]))
    print("\n       r vs yaw/pitch/roll    R² pose/move/both   R² |dAU|   xcorr peak")
    for k, a in enumerate(au_names):
        q, xcr = r2[a], xcorr[a]
        print("  {}  {:+.2f} {:+.2f} {:+.2f}      {:.2f} / {:.2f} / {:.2f}     {:.2f}     {:+.2f} at {:+.2f} s".format(
            a, *r_level[k], q["pose"], q["movement"], q["both"], q["change_from_movement"],
            xcr["r_peak"], xcr["peak_lag_s"]))
    print("\nSaved to {}".format(out_dir))


if __name__ == "__main__":
    main()
