import numpy as np
from dataclasses import replace
from params import load_params
from control import design_lqr
from sensors import read_sensors
from sim_est import simulate_est

p = load_params()
K, _ = design_lqr(p, [5, 1, 50, 1], 100)

print("Check 1: sensor model with noise and bias switched off, robot upright, rolling at 0.5 m/s")
p0 = replace(p, gyro_noise_rad_s=0.0, accel_noise_ms2=0.0, encoder_noise_rad_s=0.0, gyro_bias_rad_s=0.0)
gyro, fx, fz, w_rel = read_sensors(p0, np.array([0.0, 0.5, 0.0, 0.0]), 0.0, np.random.default_rng(0))
print(f"  gyro {gyro:.3f} rad/s (expect 0), accel forward {fx:.3f} (expect 0), "
      f"accel up {fz:.3f} (expect 9.81), wheel speed {w_rel:.2f} rad/s (expect {0.5 / p.wheel_radius_m:.2f})")

print("Check 2: estimators in the closed loop, 500 Hz, nonlinear sim with limits")
print("  driving to a velocity command, 10 s run, error measured after the first second")
cases = [("true state (reference)", dict(estimator=None)),
         ("comp filter 0.5 s", dict(estimator="complementary", tau_c=0.5, compensate=False)),
         ("comp filter 2 s", dict(estimator="complementary", tau_c=2.0, compensate=False)),
         ("0.5 s + encoder compensation", dict(estimator="complementary", tau_c=0.5, compensate=True))]
for vref in [0.3, 0.6, 1.0]:
    print("  velocity command", vref, "m/s")
    for name, kw in cases:
        r = simulate_est(p, K, v_ref=vref, t_end=10.0, **kw)
        print(f"    {name:30s} fell={str(r['fell']):5s} tilt err rms={r['tilt_err_rms_deg']:5.2f} deg   "
              f"final v={r['final_v']:.3f}   max v={r['v'].max():.3f}")

print("Check 3: gyro bias, calibrated at startup or not (velocity command 0.3 m/s, 8 s)")
for cal in [True, False]:
    print("  calibrated at startup:", cal)
    for name, kw in [("0.5 s + compensation", dict(tau_c=0.5, compensate=True)),
                     ("2 s plain", dict(tau_c=2.0, compensate=False)),
                     ("2 s + compensation", dict(tau_c=2.0, compensate=True))]:
        r = simulate_est(p, K, estimator="complementary", v_ref=0.3, t_end=8.0, calibrate=cal, **kw)
        print(f"    {name:24s} tilt err rms={r['tilt_err_rms_deg']:.2f} deg   final v={r['final_v']:.3f}")