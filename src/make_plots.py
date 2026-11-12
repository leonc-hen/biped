import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import mujoco
from params import load_params
from model import deriv, rk4
from control import design_lqr
from sim_est import simulate_est
from mujoco_model import make, apply_torque

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)
p = load_params()
K, _ = design_lqr(p, [5, 1, 50, 1], 100)
DT = 2e-4
EST = dict(estimator="complementary", tau_c=0.5, compensate=True)

def tvec(n):
    return (np.arange(n) + 1) * DT

# ---- Plot 1: model versus MuJoCo, open loop ----
th0, T = 0.001, 0.5
s = np.array([0.0, 0.0, th0, 0.0]); dt = 1e-4
tm, thm = [0.0], [th0]
for k in range(int(T / dt)):
    s = rk4(lambda q: deriv(p, q, 0.0), s, dt); tm.append((k + 1) * dt); thm.append(s[2])
model, data, ids, vids = make(p, mode="contact")
data.qpos[ids["pitch"]] = th0
mujoco.mj_forward(model, data)
tj, thj = [], []
for _ in range(int(T / 0.0005)):
    apply_torque(data, 0.0); mujoco.mj_step(model, data); tj.append(data.time); thj.append(data.qpos[ids["pitch"]])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tm, thm, label="Step 2 model", linewidth=2)
ax.plot(tj, thj, "--", label="MuJoCo (contact)", linewidth=2)
ax.set_xlabel("time (s)"); ax.set_ylabel("tilt (rad)")
ax.set_title("Open loop fall from 0.001 rad: model vs MuJoCo"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "model_vs_mujoco.png", dpi=150); plt.close(fig)
print("saved model_vs_mujoco.png")

# ---- Plot 2: recovery and push, full stack (estimator, noise, limits) ----
cases = [("Recover from 0.3 rad lean", dict(theta0=0.3, v_ref=0.0, t_end=3.0)),
         ("Survive a 0.6 m/s push", dict(theta0=0.0, v0=0.6, v_ref=0.0, t_end=3.0))]
fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex="col")
for col, (title, kw) in enumerate(cases):
    r = simulate_est(p, K, dt=DT, **EST, **kw)
    t = tvec(len(r["theta"]))
    axes[0, col].plot(t, np.degrees(r["theta"])); axes[0, col].set_ylabel("tilt (deg)")
    axes[1, col].plot(t, r["v"]); axes[1, col].set_ylabel("velocity (m/s)")
    axes[2, col].plot(t, r["tau_applied"]); axes[2, col].set_ylabel("total wheel torque (N m)")
    axes[2, col].set_xlabel("time (s)"); axes[0, col].set_title(title)
    for a in axes[:, col]: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "recovery_and_push.png", dpi=150); plt.close(fig)
print("saved recovery_and_push.png")

# ---- Plot 3: estimator comparison while driving ----
cases = [("true state", dict(estimator=None)),
         ("plain filter, 0.5 s", dict(estimator="complementary", tau_c=0.5, compensate=False)),
         ("plain filter, 2 s", dict(estimator="complementary", tau_c=2.0, compensate=False)),
         ("0.5 s + encoder compensation", dict(estimator="complementary", tau_c=0.5, compensate=True))]
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for name, kw in cases:
    r = simulate_est(p, K, v_ref=1.0, t_end=10.0, dt=DT, **kw)
    t = tvec(len(r["v"]))
    axes[0].plot(t, r["v"], label=name)
    axes[1].plot(t, np.degrees(r["theta"]))
axes[0].axhline(1.0, color="k", linestyle=":", linewidth=1)
axes[0].set_ylabel("velocity (m/s)"); axes[1].set_ylabel("tilt (deg)"); axes[1].set_xlabel("time (s)")
axes[0].set_title("Driving to 1 m/s with different estimators"); axes[0].legend()
for a in axes: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "estimator_comparison.png", dpi=150); plt.close(fig)
print("saved estimator_comparison.png")

# ---- Plot 4: why a large R ----
Rs = np.logspace(-1, 3.2, 40)
fast = []
for R in Rs:
    _, Acl = design_lqr(p, [5, 1, 50, 1], R)
    fast.append(max(abs(e.real) for e in np.linalg.eigvals(Acl)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(Rs, fast, linewidth=2)
ax.axvline(100, color="k", linestyle=":", linewidth=1); ax.text(110, max(fast) * 0.5, "chosen R = 100")
ax.set_xlabel("R (torque cost)"); ax.set_ylabel("fastest closed loop pole (rad/s)")
ax.set_title("A small R makes the loop very fast, and delay sensitive"); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "R_tradeoff.png", dpi=150); plt.close(fig)
print("saved R_tradeoff.png")