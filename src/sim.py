"""Closed loop nonlinear simulation using the true state."""
import numpy as np
from model import deriv, rk4
from actuator import rel_speed, torque_mode

def simulate(p, K, theta0=0.2, v_ref=0.0, rate_hz=500.0, delay_s=0.0,
             t_end=2.0, dt=1e-4, fall_rad=None, limits=True, latency_samples=0, v0=0.0,
             com_offset_rad=0.0):
    """Closed loop nonlinear simulation with a discrete controller, a pure delay,
    and the torque mode actuator (current and voltage limits, friction).
    Returns a dict of results."""
    if fall_rad is None:
        fall_rad = p.fall_threshold_rad
    Ts = 1.0 / rate_hz
    steps_per_ctrl = int(round(Ts / dt))
    delta = com_offset_rad
    s = np.array([0.0, v0, theta0 + delta, 0.0])     # s[2] is the angle of the CoM line (body tilt + delta)
    z, u_cmd = 0.0, 0.0
    pending = []                      # (apply_time, command)
    applied = 0.0
    th_hist, v_hist, u_hist, w_hist = [], [], [], []
    n = int(t_end / dt)
    fell = False
    for k in range(n):
        t = k * dt
        if k % steps_per_ctrl == 0:
            z += (s[1] - v_ref) * Ts
            u_cmd = float((-K @ np.array([z, s[1], s[2] - delta, s[3]])).item())
            pending.append((t + delay_s + latency_samples * Ts, u_cmd))
        while pending and pending[0][0] <= t + 1e-12:
            applied = pending.pop(0)[1]
        if limits:
            tau, _ = torque_mode(p, s, applied)
        else:
            tau = applied                 # ideal actuator, no limits, no friction
        s = rk4(lambda q: deriv(p, q, tau), s, dt)
        th_hist.append(s[2] - delta); v_hist.append(s[1]); u_hist.append(tau)
        w_hist.append(rel_speed(p, s))
        if abs(s[2] - delta) > fall_rad:
            fell = True
            break
    th = np.array(th_hist)
    return {"fell": fell, "theta": th, "v": np.array(v_hist), "tau": np.array(u_hist),
            "w_rel": np.array(w_hist), "final_theta": th[-1], "final_v": v_hist[-1],
            "peak_tau": float(np.max(np.abs(u_hist)))}
