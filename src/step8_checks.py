"""Step 8 checks: actuator torque, current, power and speed demanded by the balance controller."""
import numpy as np
from params import load_params
from control import design_lqr
from sim import simulate
from sim_est import simulate_est

p = load_params()
K, _ = design_lqr(p, [5, 1, 50, 1], 100)
n, Kt, r = p.motor_count, p.motor_kt_nm_per_a, p.wheel_radius_m

def demands(**kw):
    """Run with an ideal actuator so the numbers are what the controller ASKS for."""
    res = simulate(p, K, limits=False, rate_hz=500.0, t_end=4.0, dt=2e-4, **kw)
    tau, w = res["tau"], res["w_rel"]
    return {"fell": res["fell"],
            "tau_total_peak": float(np.max(np.abs(tau))),
            "tau_motor_peak": float(np.max(np.abs(tau)) / n),
            "tau_motor_rms": float(np.sqrt(np.mean((tau / n) ** 2))),
            "current_peak": float(np.max(np.abs(tau)) / n / Kt),
            "power_peak": float(np.max(np.abs(tau * w))),
            "w_peak": float(np.max(np.abs(w)))}

def show(label, d):
    print(f"  {label:26s} total {d['tau_total_peak']:.3f} N m   per motor {d['tau_motor_peak']:.3f} N m "
          f"(rms {d['tau_motor_rms']:.3f})   {d['current_peak']:.2f} A   power {d['power_peak']:.2f} W   "
          f"wheel speed vs body {d['w_peak']:.1f} rad/s" + ("   FELL" if d["fell"] else ""))

print("Check 1: velocity pushes (robot starts upright, upright command, ideal actuator)")
for v0 in [0.3, 0.6, 1.0]:
    show(f"push of {v0} m/s", demands(theta0=0.0, v0=v0))

print("Check 2: tilt recovery from rest")
for th0 in [0.1, 0.3, 0.6]:
    show(f"start tilt {th0} rad", demands(theta0=th0))

print("Check 3: driving, velocity step from rest")
for vr in [0.3, 1.0]:
    show(f"velocity command {vr} m/s", demands(theta0=0.0, v_ref=vr))

print("Check 4: speed needed to drive")
for v in [0.3, 1.0, 2.0]:
    w = v / r
    print(f"  {v} m/s needs {w:.1f} rad/s at the wheel = {w * 60 / (2 * np.pi):.0f} rpm")

print("Check 5: standing still with noisy sensors and the estimator in the loop (8 s, 500 Hz)")
res = simulate_est(p, K, estimator="complementary", tau_c=0.5, compensate=True, v_ref=0.0, t_end=8.0)
print(f"  torque rms (total) {res['tau_rms']:.4f} N m, peak {res['tau_peak']:.4f} N m, "
      f"per motor peak current {res['tau_peak'] / n / Kt:.2f} A")

print("Placeholder motor for comparison: torque limit", Kt * p.current_limit_a, "N m per motor,",
      p.current_limit_a, "A, no load speed", p.battery_voltage_v / p.motor_ke_vs_per_rad, "rad/s")
