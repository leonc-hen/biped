"""Step 5 checks: controllability, LQR gains versus R, delay tolerance, velocity step."""
import numpy as np
from params import load_params
from control import design_lqr, controllability_rank
from sim import simulate

p = load_params()
Q = [5, 1, 50, 1]

print("Check 1: controllability rank (need 4):", controllability_rank(p))

print("Check 2: LQR gains and closed loop poles, Q = diag(5, 1, 50, 1)")
print("   R        K = [z, v, theta, thetadot]            fastest pole (rad/s)")
for R in [1, 10, 100, 1000]:
    K, Acl = design_lqr(p, Q, R)
    fast = min(np.linalg.eigvals(Acl).real)
    print(f"  {R:5d}   {np.round(K.ravel(), 3)}   {fast:9.1f}")

print("Check 3: delay test, 500 Hz loop, nonlinear sim with limits and friction")
print("  F = fell or did not settle, . = recovered.  delays in ms: 0 4 8 12 16 20")
delays = [0.0, 0.004, 0.008, 0.012, 0.016, 0.020]
for th0 in [0.05, 0.2]:
    print("  start tilt", th0, "rad")
    for R in [1, 10, 100, 1000]:
        K, _ = design_lqr(p, Q, R)
        row = ""
        for dl in delays:
            r = simulate(p, K, theta0=th0, delay_s=dl, rate_hz=500.0, t_end=1.5)
            row += "F" if (r["fell"] or abs(r["final_theta"]) > 0.02) else "."
        print(f"    R = {R:5d}   {row}")

print("Check 4: velocity step to 0.3 m/s from rest, R = 100, 500 Hz, no delay")
K, _ = design_lqr(p, Q, 100)
r = simulate(p, K, theta0=0.0, v_ref=0.3, rate_hz=500.0, t_end=4.0)
v, th, tau = r["v"], r["theta"], r["tau"]
t = np.arange(len(v)) * 1e-4
idx = np.where(np.abs(v - 0.3) > 0.015)[0]
print("  final velocity (m/s):", round(float(v[-1]), 3))
print("  5 percent settling time (s):", round(float(t[idx[-1]] + 1e-4), 2))
print("  peak velocity overshoot (m/s):", round(float(v.max() - 0.3), 3))
print("  peak tilt (rad):", round(float(np.max(np.abs(th))), 3))
print("  peak total wheel torque (N m):", round(float(np.max(np.abs(tau))), 3))
