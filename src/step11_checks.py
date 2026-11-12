import numpy as np
import mujoco
from params import load_params
from model import deriv, rk4
from control import design_lqr
from sim import simulate
from mujoco_model import make, apply_torque

p = load_params()
r = p.wheel_radius_m

def my_model(tau_fn, s0, T, dt=5e-5, wrong=False):
    """Step 2 nonlinear model, fine step. Returns time and state arrays.
    wrong=True drops the reaction torque on the body (the mistake we are guarding against)."""
    def f(s, tau):
        if not wrong:
            return deriv(p, s, tau)
        a, b, J = p.a, p.b, p.J
        k = p.body_mass_kg * p.gravity_ms2 * p.axle_to_com_m
        M = np.array([[a, b * np.cos(s[2])], [b * np.cos(s[2]), J]])
        xdd, thdd = np.linalg.solve(M, np.array([tau / r + b * np.sin(s[2]) * s[3]**2, k * np.sin(s[2])]))
        return np.array([s[1], xdd, s[3], thdd])
    s = np.array(s0, dtype=float)
    t, out = [0.0], [s.copy()]
    for i in range(int(round(T / dt))):
        tau = tau_fn((i + 0.5) * dt)
        s = rk4(lambda q: f(q, tau), s, dt)
        t.append((i + 1) * dt); out.append(s.copy())
    return np.array(t), np.array(out)

def mj_open_loop(mode, ts, solref, tau_fn, th0, T):
    model, data, ids, vids = make(p, mode=mode, timestep=ts, solref=solref)
    data.qpos[ids["pitch"]] = th0
    mujoco.mj_forward(model, data)
    t, x, th = [], [], []
    for _ in range(int(round(T / ts))):
        apply_torque(data, tau_fn(data.time))
        mujoco.mj_step(model, data)
        t.append(data.time); x.append(data.qpos[ids["x"]]); th.append(data.qpos[ids["pitch"]])
    slip = abs(data.qpos[ids["x"]] - r * data.qpos[ids["wheel"]])
    return np.array(t), np.array(x), np.array(th), slip

def err_pct(t, y, tr, yr):
    return 100 * np.max(np.abs(y - np.interp(t, tr, yr))) / np.max(np.abs(yr))

print("Check 1: open loop tilt, start 0.001 rad, zero torque, 0.5 s (pass: under 2 percent)")
tr, ref = my_model(lambda t: 0.0, [0, 0, 0.001, 0], 0.5)
for ts, sr in [(0.001, "0.002 1"), (0.0005, "0.001 1"), (0.00025, "0.0005 1")]:
    t, x, th, slip = mj_open_loop("constraint", ts, sr, lambda t: 0.0, 0.001, 0.5)
    print(f"  constraint model, timestep {ts*1000:.2f} ms   tilt error {err_pct(t, th, tr, ref[:, 2]):.2f} percent")
t, x, th, slip = mj_open_loop("contact", 0.0005, "0.001 1", lambda t: 0.0, 0.001, 0.5)
print(f"  contact model,    timestep 0.50 ms   tilt error {err_pct(t, th, tr, ref[:, 2]):.2f} percent   rolling slip {slip:.1e} m")

print("Check 2: torque step, 0.05 N m total from rest, 0.15 s (tests the reaction term)")
tau_fn = lambda t: 0.05
tr, ref = my_model(tau_fn, [0, 0, 0, 0], 0.15)
_, refw = my_model(tau_fn, [0, 0, 0, 0], 0.15, wrong=True)
for mode in ["constraint", "contact"]:
    t, x, th, slip = mj_open_loop(mode, 0.0005, "0.001 1", tau_fn, 0.0, 0.15)
    print(f"  {mode:10s} model: tilt error {err_pct(t, th, tr, ref[:, 2]):.2f} percent, position error {err_pct(t, x, tr, ref[:, 0]):.2f} percent")
print(f"  my model WITHOUT the reaction torque vs MuJoCo constraint: tilt error "
      f"{err_pct(t, th, tr, refw[:, 2]):.0f} percent (should be large, this is what the check catches)")

print("Check 3: closed loop LQR, start tilt 0.2 rad, 500 Hz, true state, 1 s")
K, _ = design_lqr(p, [5, 1, 50, 1], 100)
mine = simulate(p, K, theta0=0.2, limits=False, rate_hz=500.0, t_end=1.0, dt=1e-4)
tm = (np.arange(len(mine["theta"])) + 1) * 1e-4
for mode in ["constraint", "contact"]:
    ts = 0.0005
    model, data, ids, vids = make(p, mode=mode, timestep=ts, solref="0.001 1")
    data.qpos[ids["pitch"]] = 0.2
    mujoco.mj_forward(model, data)
    z, tau = 0.0, 0.0
    tt, thh, vv = [], [], []
    sub = int(round(0.002 / ts))
    for k in range(int(round(1.0 / ts))):
        if k % sub == 0:
            v, th_, thd_ = data.qvel[vids["x"]], data.qpos[ids["pitch"]], data.qvel[vids["pitch"]]
            z += v * 0.002
            tau = float((-K @ np.array([z, v, th_, thd_])).item())
        apply_torque(data, tau)
        mujoco.mj_step(model, data)
        tt.append(data.time); thh.append(data.qpos[ids["pitch"]]); vv.append(data.qvel[vids["x"]])
    tt, thh, vv = np.array(tt), np.array(thh), np.array(vv)
    print(f"  {mode:10s} model: tilt error {err_pct(tt, thh, tm, mine['theta']):.2f} percent of peak tilt, "
          f"velocity error {err_pct(tt, vv, tm, mine['v']):.2f} percent of peak velocity")