"""Project paths and small helpers shared by the pipeline stages."""
import os
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "raw_data"
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
FACEMAP_DIR = PROJECT_DIR / "facemap_output"
TRIAL_DIR = PROJECT_DIR / "trial_data"


def session_name(arg):
    """Session folder name from --session, given as a name or a path (raw_data\\session_...)."""
    name = Path(arg).name
    if not (RAW_DIR / name).is_dir():
        raise SystemExit("ERROR: session folder not found: {}".format(RAW_DIR / name))
    return name


def save_figure(fig, path, attempts=5):
    """Save via a temp file + rename, then close the figure. On Windows an open image
    preview can briefly hold the target, which makes a direct save fail (Errno 22)."""
    import matplotlib.pyplot as plt
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


def mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(sem)
