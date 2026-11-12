"""Design the baseline LQR and record it in gain_table.json (one entry per name)."""
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from params import load_params
from control import design_lqr

root = Path(__file__).resolve().parent.parent
p = load_params()
Q = [5, 1, 50, 1]
R = 100

K, Acl = design_lqr(p, Q, R)

try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=root, stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = "unknown"

entry = {
    "name": "lqr_baseline_R100",
    "git_commit": commit,
    "state_order": ["z", "v", "theta", "thetadot"],
    "control_law": "u = -K @ state, u is total wheel torque in N m",
    "Q_diag": Q,
    "R": R,
    "K": [round(float(k), 6) for k in K.ravel()],
    "fastest_closed_loop_pole_rad_s": round(float(max(abs(e.real) for e in np.linalg.eigvals(Acl))), 2),
    "params": asdict(p),
}

out = root / "gain_table.json"
table = json.loads(out.read_text()) if out.exists() else {"entries": []}
table["entries"] = [e for e in table["entries"] if e["name"] != entry["name"]] + [entry]
out.write_text(json.dumps(table, indent=2))
print("saved", out)
print("K =", entry["K"])
