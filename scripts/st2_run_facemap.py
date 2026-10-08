"""Pipeline stage 2: FaceMap motion SVD, one run per ROI from stage 1.

FaceMap's motion-SVD ROIs are rectangles, but the stage-1 ROIs follow the face oval
(bands of the face; upper_face_no_eyes also has the eyes cut out). So for each ROI
this writes a temporary copy of preprocessed/<session>/face.avi, cropped to the ROI's bounding box with every pixel outside the ROI set to constant
grey, and runs FaceMap on that whole frame. Constant pixels have zero frame-to-frame
difference, so they contribute no motion. The copy is lossless (FFV1), so no
compression noise appears in the grey area; it is deleted afterwards.

Saves to facemap_output/<session>/:

    <roi>_proc.npy            FaceMap's raw output (all 500 components)
    <roi>_PCs.npy             (n_frames, 100) float32, motion SVD time courses
    <roi>_masks.npy           (h_bin, w_bin, 100) float32, spatial masks (binned pixels of the box)
    <roi>_varexp.npy          (100,) float32, fraction of ROI motion-energy variance per PC
    <roi>_motion.png          total motion energy in the ROI + PC1-6 time courses over the session
    <roi>_masks.png           PC1-6 spatial masks on the mean face (what each component moves)
    summary.json              per ROI: bounding box in face.avi pixels, pixel count, variance explained

Notes on the FaceMap 1.0.8 API (differs from older docs / the GUI):
  * process.run() takes sbin/motSVD/movSVD as arguments; ROIs go in via `proc`.
  * A motion-SVD ROI is rind=1 (rind=0 is a pupil ROI).
  * The number of components is fixed at 500 internally; we keep the first 100.
  * With ROIs, index 0 of motSVD/motMask is the full frame and index k+1 the k-th
    motion ROI. fullSVD=False skips the full-frame SVD, leaving index 0 empty.
  * There's no variance-explained output, so it's computed here (see below).
  * Motion energy is always mean-subtracted (FaceMap subtracts the average motion frame).
  * FaceMap 1.0.8's utils.svdecon runs sklearn PCA on the (pixels x frames) matrix,
    which also subtracts each frame's mean over pixels. That forces every spatial
    mask to sum to zero over the whole box, so the constant grey pixels outside an
    ROI get a large constant weight (43% of PC1's norm for an earlier face-minus-eyes ROI).
    FaceMap's earlier implementation (still in the source, commented out) did a plain
    SVD; svdecon_uncentred restores that, with the same randomized solver and seed.

Usage (inside the facepipe env, from the project folder):
    python scripts\\st2_run_facemap.py --session session_YYYYMMDD_HHMMSS_<task>
    python scripts\\st2_run_facemap.py --session session_YYYYMMDD_HHMMSS_<task> --rois upper_face upper_face_no_eyes
    python scripts\\st2_run_facemap.py --session session_YYYYMMDD_HHMMSS_<task> --rois all
    python scripts\\st2_run_facemap.py --session session_YYYYMMDD_HHMMSS_<task> --subfolder rectangular_rois
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from facemap import process, utils
from sklearn.utils.extmath import randomized_svd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = "session_20260930_165233"
DEFAULT_ROIS = ["whole_face", "upper_face", "lower_face", "upper_face_no_eyes"]
SBIN = 4
N_COMPONENTS = 100
MASK_GREY = 128          # same as stage 1
N_PLOT = 6               # PCs shown in the figures
N_MEAN_FRAMES = 300      # frames averaged for the mean face behind the masks


def svdecon_uncentred(X, k=100):
    """Drop-in for facemap.utils.svdecon without PCA's centring: (U, S, Vt) of X itself."""
    return randomized_svd(X, n_components=k, random_state=0)


# process.py calls utils.svdecon at run time, so replacing it here is enough.
utils.svdecon = svdecon_uncentred


def bin_slice(r):
    return (slice(r["yrange_bin"][0], r["yrange_bin"][-1] + 1),
            slice(r["xrange_bin"][0], r["xrange_bin"][-1] + 1))


def roi_motion_sum_of_squares(video_path, rois, avgmotion):
    """Per ROI: total sum of squares of the mean-subtracted, binned motion energy.

    This is the same matrix FaceMap projects onto its masks, so
    variance explained by PC k = sum(V_k^2) / this total (masks are orthonormal).
    """
    slices = [bin_slice(r) for r in rois]
    cap = cv2.VideoCapture(str(video_path))
    Ly, Lx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Lyb, Lxb = Ly // SBIN, Lx // SBIN
    totals, prev = np.zeros(len(rois)), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[np.newaxis]
        binned = process.spatial_bin(gray, SBIN, Lyb, Lxb).reshape(Lyb, Lxb)
        if prev is not None:
            motion = np.abs(binned - prev) - avgmotion
            for k, sl in enumerate(slices):
                totals[k] += float((motion[sl] ** 2).sum())
        prev = binned
    cap.release()
    return totals


def save_figure(fig, path, attempts=5):
    """Save via a temp file + rename. On Windows an open image preview can briefly
    hold the target, which makes a direct save fail (Errno 22)."""
    tmp = path.with_name(path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def mean_face(video_path):
    """Mean grayscale frame from evenly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc, used = None, 0
    for f in np.linspace(0, n - 1, min(n, N_MEAN_FRAMES)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            acc = g if acc is None else acc + g
            used += 1
    cap.release()
    return acc / used


def plot_motion(label, motion, pcs, varexp, fps, session, path, n_show=N_PLOT):
    """Total ROI motion energy + PC time courses on one time axis."""
    t = np.arange(len(motion)) / fps
    fig, axes = plt.subplots(n_show + 1, 1, figsize=(14, 1.3 * (n_show + 1) + 0.8), sharex=True)
    rows = [("motion\nenergy", motion, "tab:orange")]
    rows += [("PC{}\n{:.1f}%".format(k + 1, varexp[k] * 100), pcs[:, k], "0.2") for k in range(n_show)]
    for ax, (name, y, colour) in zip(axes, rows):
        y = y.astype(np.float64).copy()
        y[0] = np.nan                     # frame 0 has no motion frame (FaceMap copies frame 1)
        ax.plot(t, y, lw=0.5, color=colour)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("{}: {} - total motion energy and PC time courses (% = variance explained; "
                 "PC sign is arbitrary)".format(session, label), fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def plot_masks(label, masks, varexp, box, roi_mask, face, session, path, n_show=N_PLOT):
    """PC spatial masks over the mean face, ROI outline in yellow."""
    hb, wb = masks.shape[:2]
    y0, x0 = box["y"], box["x"]
    y1, x1 = y0 + hb * SBIN, x0 + wb * SBIN     # pixels FaceMap used: whole sbin blocks
    crop, outline = face[y0:y1, x0:x1], roi_mask[y0:y1, x0:x1]
    n_cols = 3 if n_show <= 6 else 5
    n_rows = -(-n_show // n_cols)
    width = 11 if n_cols == 3 else 16
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(
        width, n_rows * 1.1 * width / n_cols * (y1 - y0) / (x1 - x0) + 1.2))
    for ax in axes.flat[n_show:]:
        ax.axis("off")
    for k, ax in enumerate(axes.flat[:n_show]):
        m_up = cv2.resize(masks[:, :, k], (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        lim = np.abs(m_up).max()
        ax.imshow(crop, cmap="gray")
        ax.imshow(m_up, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55)
        ax.contour(outline, levels=[0.5], colors="yellow", linewidths=0.6)
        ax.set_title("PC{}  ({:.1f}% var)".format(k + 1, varexp[k] * 100))
        ax.axis("off")
    fig.suptitle("{}: {} - where each PC's motion is\nred = positive, blue = negative weight "
                 "(sign of each PC is arbitrary)".format(session, label), fontsize=11)
    fig.tight_layout()
    save_figure(fig, path)


def write_roi_video(face_video, mask, out_path):
    """Copy of face_video cropped to the mask's bounding box (whole SBIN blocks), grey outside the mask."""
    ys, xs = np.nonzero(mask)
    y0, x0 = (ys.min() // SBIN) * SBIN, (xs.min() // SBIN) * SBIN
    y1 = min(mask.shape[0], -(-(ys.max() + 1) // SBIN) * SBIN)
    x1 = min(mask.shape[1], -(-(xs.max() + 1) // SBIN) * SBIN)
    box_mask = mask[y0:y1, x0:x1]
    cap = cv2.VideoCapture(str(face_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"FFV1"), fps,
                             (x1 - x0, y1 - y0), isColor=False)
    if not writer.isOpened():
        sys.exit("ERROR: could not create {}".format(out_path))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        crop[~box_mask] = MASK_GREY
        writer.write(crop)
        n += 1
    cap.release()
    writer.release()
    return {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=DEFAULT_SESSION,
                        help="session folder name (or path) in raw_data/")
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS,
                        help="ROI names from stage 1's rois.npz, or 'all' (default: {})".format(
                            " ".join(DEFAULT_ROIS)))
    parser.add_argument("--subfolder", default=None,
                        help="save into facemap_output/<session>/<subfolder>/ (e.g. to keep runs with "
                             "different ROI definitions apart)")
    args = parser.parse_args()
    session = Path(args.session).name
    pre_dir = PROJECT_DIR / "preprocessed" / session
    face_video = pre_dir / "face.avi"
    rois_path = pre_dir / "rois.npz"
    out_dir = PROJECT_DIR / "facemap_output" / session
    if args.subfolder:
        out_dir = out_dir / args.subfolder
    for p in (face_video, rois_path):
        if not p.is_file():
            sys.exit("ERROR: not found (run st1_preprocess_face.py first): {}".format(p))

    all_masks = dict(np.load(rois_path))
    names = list(all_masks) if args.rois == ["all"] else args.rois
    unknown = [n for n in names if n not in all_masks]
    if unknown:
        sys.exit("ERROR: unknown ROI(s) {}; stage 1 made: {}".format(unknown, ", ".join(all_masks)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update({
        "session_id": session.removeprefix("session_"),
        "input_video": str(face_video.relative_to(PROJECT_DIR)),
        "n_components": N_COMPONENTS,
        "sbin": SBIN,
        "variance_definition": "fraction of total sum of squares of mean-subtracted binned "
                               "ROI motion energy captured by each PC",
    })
    # Keep ROIs from earlier st2 runs (e.g. a single --rois run after the defaults), but drop
    # entries left by the old rectangle-ROI stage, which have no box.
    summary["rois"] = {name: info for name, info in summary.get("rois", {}).items()
                       if "box_in_face_video" in info}
    cap = cv2.VideoCapture(str(face_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    face = mean_face(face_video)

    for label in names:
        mask = all_masks[label]
        print("\n=== ROI {} ({} px) ===".format(label, int(mask.sum())))
        roi_video = tmp_dir / "{}.avi".format(label)
        box, n_frames = write_roi_video(face_video, mask, roi_video)
        print("  box x={x} y={y} w={w} h={h}, {n} frames".format(n=n_frames, **box))

        proc_in = {
            "sbin": SBIN,
            "fullSVD": False,          # ROI only; skip the full-frame SVD
            "save_mat": False,
            "rois": [{"rind": 1, "rtype": "motion SVD", "ivid": 0,
                      "xrange": np.arange(box["w"]), "yrange": np.arange(box["h"])}],
            "sy": np.array([0]),
            "sx": np.array([0]),
            "savepath": str(out_dir),
        }
        proc_path = process.run(filenames=[[str(roi_video)]], sbin=SBIN, motSVD=True, movSVD=False,
                                proc=proc_in, savepath=str(out_dir))
        proc = np.load(proc_path, allow_pickle=True).item()
        avgmotion = np.reshape(proc["avgmotion"][0], (proc["Lybin"][0], proc["Lxbin"][0]))
        print("  computing variance explained...")
        total_ss = roi_motion_sum_of_squares(roi_video, proc["rois"], avgmotion)[0]

        V = proc["motSVD"][1]                    # (n_frames, 500)
        masks = proc["motMask_reshape"][1]       # (h_bin, w_bin, 500)
        # Frame 0 has no motion frame; FaceMap copies frame 1's projection into it.
        varexp = ((V[1:].astype(np.float64) ** 2).sum(axis=0) / total_ss)[:N_COMPONENTS]
        np.save(out_dir / "{}_PCs.npy".format(label), V[:, :N_COMPONENTS].astype(np.float32))
        np.save(out_dir / "{}_masks.npy".format(label), masks[:, :, :N_COMPONENTS].astype(np.float32))
        np.save(out_dir / "{}_varexp.npy".format(label), varexp.astype(np.float32))
        roi_video.unlink()

        plot_motion(label, proc["motion"][1], V, varexp, fps, session, out_dir / "{}_motion.png".format(label))
        plot_masks(label, masks, varexp, box, mask, face, session, out_dir / "{}_masks.png".format(label))
        print("  plots: {0}_motion.png, {0}_masks.png".format(label))

        summary["n_frames"] = int(V.shape[0])
        summary["rois"][label] = {
            "box_in_face_video": box,
            "n_pixels": int(mask.sum()),
            "mask_shape_binned": list(masks.shape[:2]) + [N_COMPONENTS],
            "variance_explained_top5": [round(float(v), 4) for v in varexp[:5]],
            "variance_explained_cumulative_10": round(float(varexp[:10].sum()), 4),
            "variance_explained_cumulative_20": round(float(varexp[:20].sum()), 4),
            "variance_explained_cumulative_100": round(float(varexp.sum()), 4),
        }
        print("  {}: PCs {}, cumulative var 10 = {:.3f}, 20 = {:.3f}".format(
            label, (V.shape[0], N_COMPONENTS), varexp[:10].sum(), varexp[:20].sum()))

    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSaved to {}:\n  plots: <roi>_motion.png, <roi>_masks.png\n"
          "  data:  <roi>_PCs/_masks/_varexp.npy, <roi>_proc.npy, summary.json".format(out_dir))


if __name__ == "__main__":
    main()
