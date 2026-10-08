"""Pipeline stage 4: compare leave-decision classifiers (task only / face only / task + face).

Uses stage 3's feature table, keeping the face features of the chosen ROIs (--rois), so
ROI sets can be compared without rebuilding features.

Models: elastic-net logistic regression on task features (A), face features (B) and both
(C), scored by AUC-ROC with leave-one-visit-out cross-validation. Visits whose trials
are all one class (no AUC) are left out of the AUC mean, for all models alike.

Inputs:  trial_data/<session>/X_features.npy, y_labels.npy, feature_names.txt, trial_metadata.csv
Output:  trial_data/<session>/models/<roi>+<roi>....json   AUC per model, per-fold AUCs

Usage (inside the facepipe env, from the project folder):
    python scripts\\st4_classify_leave.py --session session_YYYYMMDD_HHMMSS_<task>
    python scripts\\st4_classify_leave.py --session session_YYYYMMDD_HHMMSS_<task> --rois upper_face
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PROJECT_DIR, TRIAL_DIR, mean_sem, session_name
from roi_definitions import DEFAULT_ROIS

L1_RATIO = 0.5
MAX_ITER = 10000       # saga converges slowly
# sklearn >= 1.8 deprecates `penalty` (removed in 1.10): a float l1_ratio alone is elastic net.
SKLEARN_VERSION = tuple(int(p) for p in re.findall(r"\d+", sklearn.__version__)[:2])
PENALTY_KW = {} if SKLEARN_VERSION >= (1, 8) else {"penalty": "elasticnet"}
FACE_FEATURE = re.compile(r"^(?P<roi>.+)_PC\d+_(mean|last3|slope)$")


def load_features(session):
    """(X, y, feature names, trial metadata) from stage 3."""
    d = TRIAL_DIR / session
    if not (d / "X_features.npy").is_file():
        sys.exit("ERROR: no features in {} (run st3 first)".format(d))
    names = (d / "feature_names.txt").read_text().split()
    return np.load(d / "X_features.npy"), np.load(d / "y_labels.npy"), names, pd.read_csv(d / "trial_metadata.csv")


def feature_columns(names, rois):
    """Boolean masks over the feature columns: (task features, face features of `rois`)."""
    face_roi = [m.group("roi") if (m := FACE_FEATURE.match(n)) else None for n in names]
    missing = [r for r in rois if r not in face_roi]
    if missing:
        sys.exit("ERROR: no features for ROI(s) {}; available: {} (run st2 and st3 for them)".format(
            ", ".join(missing), ", ".join(dict.fromkeys(r for r in face_roi if r))))
    task = np.array([r is None for r in face_roi])
    face = np.array([r in rois for r in face_roi])
    return task, face


def leave_one_visit_out_auc(X, y, visits):
    """AUC on each held-out visit (model fit on all other visits); visits with one class skipped."""
    aucs = []
    for v in np.unique(visits):
        test = visits == v
        if len(np.unique(y[test])) < 2:
            continue
        model = LogisticRegression(solver="saga", l1_ratio=L1_RATIO, max_iter=MAX_ITER, **PENALTY_KW)
        model.fit(X[~test], y[~test])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return np.array(aucs)


def compare_models(X, y, visits, names, rois):
    """{"A"/"B"/"C": {label, n_features, mean, sem, n_folds_scored, fold_aucs}}."""
    task, face = feature_columns(names, rois)
    models = {"A": ("task only", task), "B": ("face PCs only", face), "C": ("task + face", task | face)}
    results = {}
    for key, (label, cols) in models.items():
        fold_aucs = leave_one_visit_out_auc(X[:, cols], y, visits)
        m, s = mean_sem(fold_aucs)
        results[key] = {"label": label, "n_features": int(cols.sum()), "mean": round(m, 4),
                        "sem": round(s, 4), "n_folds_scored": len(fold_aucs),
                        "fold_aucs": [round(float(a), 4) for a in fold_aucs]}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="session folder name (or path) in raw_data/")
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS,
                        help="ROIs whose face features to use (default: {})".format(" ".join(DEFAULT_ROIS)))
    args = parser.parse_args()
    session = session_name(args.session)
    rois = list(dict.fromkeys(args.rois))

    X, y, names, meta = load_features(session)
    visits = meta["site_visit"].values
    n_visits = len(np.unique(visits))
    print("{}: {} trials, {} leave, {} visits; ROIs: {}".format(
        session, len(y), int(y.sum()), n_visits, ", ".join(rois)))
    auc = compare_models(X, y, visits, names, rois)

    out_path = TRIAL_DIR / session / "models" / "{}.json".format("+".join(rois))
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "session_id": session.removeprefix("session_"),
        "rois": rois,
        "n_trials": int(len(y)),
        "n_leave": int(y.sum()),
        "n_visits": n_visits,
        "model": "LogisticRegression(elastic net, solver='saga', l1_ratio={}), leave-one-visit-out CV, "
                 "AUC-ROC per held-out visit (single-class visits not scored)".format(L1_RATIO),
        "auc": auc,
    }, indent=2))

    print("AUC from {} of {} visits (others have one class)\n".format(auc["A"]["n_folds_scored"], n_visits))
    for key, r in auc.items():
        print("  {} {:16s} {:3d} features   AUC = {:.3f} +/- {:.3f}".format(
            key, r["label"], r["n_features"], r["mean"], r["sem"]))
    print("\nSaved {}".format(out_path.relative_to(PROJECT_DIR)))


if __name__ == "__main__":
    main()
