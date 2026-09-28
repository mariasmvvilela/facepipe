"""Verify the facepipe conda environment: import every pipeline dependency,
print versions in a table, and confirm PyTorch can use the GPU."""
import importlib
import importlib.metadata as md
import platform
import subprocess
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# (label, import name, distribution name for metadata fallback)
PACKAGES = [
    ("Python", None, None),
    ("PyTorch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("FaceMap", "facemap", "facemap"),
    ("MediaPipe", "mediapipe", "mediapipe"),
    ("OpenCV (cv2)", "cv2", "opencv-contrib-python"),
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn", "scikit-learn"),
    ("pandas", "pandas", "pandas"),
    ("JupyterLab", "jupyterlab", "jupyterlab"),
    ("PyQt6", "PyQt6.QtCore", "PyQt6"),
    ("pyqtgraph", "pyqtgraph", "pyqtgraph"),
    ("pip", "pip", "pip"),
]


def version_of(module, dist):
    v = getattr(module, "__version__", None)
    if v is None:
        try:
            v = md.version(dist)
        except md.PackageNotFoundError:
            v = "?"
    return v


rows, failures = [], 0
for label, mod_name, dist in PACKAGES:
    if mod_name is None:
        rows.append((label, "OK", platform.python_version()))
        continue
    try:
        mod = importlib.import_module(mod_name)
        rows.append((label, "OK", version_of(mod, dist)))
    except Exception as e:  # report, don't stop
        failures += 1
        rows.append((label, "FAIL", f"{type(e).__name__}: {e}"[:60]))

print(f"facepipe environment verification — {datetime.now():%Y-%m-%d %H:%M}")
print(f"Interpreter: {sys.executable}")
print(f"OS: {platform.platform()}\n")
w = max(len(r[0]) for r in rows)
print(f"{'Package':<{w}}  {'Status':<6}  Version")
print(f"{'-' * w}  {'-' * 6}  {'-' * 30}")
for label, status, ver in rows:
    print(f"{label:<{w}}  {status:<6}  {ver}")

# OpenCV: exactly one cv2 provider should be installed at a given version
cv_dists = {}
for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python",
          "opencv-contrib-python-headless"):
    try:
        cv_dists[d] = md.version(d)
    except md.PackageNotFoundError:
        pass
print(f"\nOpenCV distributions installed: {cv_dists}")
if len(set(cv_dists.values())) > 1:
    failures += 1
    print("  FAIL: mismatched OpenCV versions share the cv2 folder")

print("\nGPU check")
try:
    import torch
    ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {ok}")
    if ok:
        props = torch.cuda.get_device_properties(0)
        print(f"  Device:        {props.name}")
        print(f"  VRAM:          {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA (torch):  {torch.version.cuda}")
        print(f"  cuDNN:         {torch.backends.cudnn.version()}")
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        print(f"  GPU matmul:    OK {tuple((x @ x).shape)}")
    else:
        failures += 1
except Exception as e:
    failures += 1
    print(f"  FAIL: {e}")

pip_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                           capture_output=True, text=True)
print(f"\npip check: {pip_check.stdout.strip() or pip_check.stderr.strip()}")
if pip_check.returncode != 0:
    failures += 1

print(f"\nRESULT: {'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
