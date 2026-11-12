"""Closed loop nonlinear simulation with simulated sensors and the estimator in the loop."""
import numpy as np
from model import deriv, rk4
from actuator import rel_speed, torque_mode
from sensors import read_sensors
from estimator import ComplementaryEstimator, calibrate_gyro_bias


def _skip(values, n):
    """Drop the first n samples, unless that would leave nothing."""
    values = np.asarray(values)
    return values[n:] if len(values) > n else values


def simulate_est(p, K, estimator=None, tau_c=0.5, compensate=False, v_ref=0.3,
                 rate_hz=500.0, t_end=8.0, dt=2e-4, fall_rad=1.3, seed=1, theta0=0.0,
                 calibrate=True, delay_s=0.0, v0=0.0):
    """Closed loop with sensors and estimator in the loop. estimator=None uses the true state.
    The applied torque is computed before the sensors are read because the accelerometer
    model needs it, so a new command takes effect on the next physics step."""
    rng = np.random.default_rng(seed)
    Ts = 1.0 / rate_hz
    steps_per_ctrl = int(round(Ts / dt))
    s = np.array([0.0, v0, theta0, 0.0])
    est = None
    if estimator == "complementary":
        bias_est = calibrate_gyro_bias(p, rng) if calibrate else 0.0
        est = ComplementaryEstimator(p, Ts, tau_c, compensate, bias_est)
        est.theta = theta0
    z, applied, cmd = 0.0, 0.0, 0.0
    pending = []                      # (apply_time, command)
    th_err, th_true, v_hist, tau_hist = [], [], [], []
    tau_app, w_hist = [], []
    fell = False
    for k in range(int(t_end / dt)):
        t = k * dt
        while pending and pending[0][0] <= t + 1e-12:
            applied = pending.pop(0)[1]
        tau, _ = torque_mode(p, s, applied)
        if k % steps_per_ctrl == 0:
            if est is None:
                th_hat, thd_hat, v_hat = s[2], s[3], s[1]
            else:
                gyro, fx, fz, w_rel = read_sensors(p, s, tau, rng)
                th_hat, thd_hat, v_hat = est.update(gyro, fx, fz, w_rel)
            z += (v_hat - v_ref) * Ts
            cmd = float((-K @ np.array([z, v_hat, th_hat, thd_hat])).item())
            pending.append((t + delay_s, cmd))
            th_err.append(th_hat - s[2])
            tau_hist.append(cmd)
        tau_app.append(tau)
        w_hist.append(rel_speed(p, s))
        s = rk4(lambda q: deriv(p, q, tau), s, dt)
        th_true.append(s[2])
        v_hist.append(s[1])
        if abs(s[2]) > fall_rad:
            fell = True
            break
    n_skip = int(1.0 * rate_hz)       # ignore the first second
    tail = _skip(th_err, n_skip)
    tau_tail = _skip(tau_hist, n_skip)
    return {"fell": fell, "v": np.array(v_hist), "theta": np.array(th_true),
            "tau_applied": np.array(tau_app), "w_rel": np.array(w_hist),
            "tilt_err_rms_deg": float(np.degrees(np.sqrt(np.mean(tail**2)))),
            "final_v": float(v_hist[-1]), "final_theta_deg": float(np.degrees(th_true[-1])),
            "tau_rms": float(np.sqrt(np.mean(tau_tail**2))),
            "tau_peak": float(np.max(np.abs(tau_tail))),
            "peak_tilt_deg": float(np.degrees(np.max(np.abs(th_true))))}
