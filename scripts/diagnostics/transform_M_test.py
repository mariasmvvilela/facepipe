import json
import numpy as np

summary = json.load(open(r"preprocessed\session_20260930_165233\summary.json"))
M = np.array(summary["transform"])
print("M =", M)

# Decompose into interpretable parameters
s = np.hypot(M[0,0], M[1,0])
theta = np.degrees(np.arctan2(M[1,0], M[0,0]))
tx, ty = M[0,2], M[1,2]
print(f"scale:    {s:.4f}  (crop px per raw px)")
print(f"rotation: {theta:.3f} deg")
print(f"translation: tx={tx:.1f}, ty={ty:.1f} px")