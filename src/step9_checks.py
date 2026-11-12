"""Step 9 checks: robustness of the nominal LQR gains to parameter, payload, noise and delay changes."""
import numpy as np
from dataclasses import replace
from params import load_params
from model import linear_AB
from control import design_lqr
from sim import simulate
from sim_est import simulate_est

p = load_params()
Q, R = [5, 1, 50, 1], 100
K_nom, _ = design_lqr(p, Q, R)             # gains designed ONCE on the nominal robot

def with_radius(p, scale):
    r = p.wheel_radius_m * scale           # solid disc wheels: inertia follows mass * r^2
    return replace(p, wheel_radius_m=r, wheel_inertia_total_kgm2=0.5 * p.wheel_mass_total_kg * r**2)

def with_payload(p, mp, h_p, e):
    """Add a point mass mp at axial height h_p and lateral offset e (body frame).
    Returns the new params and delta, the angle between the body axis and the new CoM line."""
    m, l, I = p.body_mass_kg, p.axle_to_com_m, p.body_pitch_inertia_kgm2
    mt = m + mp
    ca, cl = (m * l + mp * h_p) / mt, mp * e / mt
    I_new = I + m * ((l - ca) ** 2 + cl ** 2) + mp * ((h_p - ca) ** 2 + (e - cl) ** 2)
    return (replace(p, body_mass_kg=mt, axle_to_com_m=float(np.hypot(ca, cl)),
                    body_pitch_inertia_kgm2=I_new), float(np.arctan2(cl, ca)))

def evaluate(pt, delta=0.0, gain_scale=1.0):
    """Nominal gains on a perturbed robot. Returns (open loop unstable pole, worst closed loop
    pole real part, largest recovered tilt, tilt gain if you redesigned LQR for this robot)."""
    A, B = linear_AB(pt)
    Kc = K_nom * gain_scale
    ol = float(max(np.linalg.eigvals(A).real))
    cl = float(max(np.linalg.eigvals(A - B @ Kc).real))
    best = 0.0
    for th in [0.3, 0.6, 0.9, 1.2]:
        r = simulate(pt, Kc, theta0=th, com_offset_rad=delta, rate_hz=500.0, t_end=4.0, dt=2e-4)
        if (not r["fell"]) and abs(r["final_theta"] + delta) < 0.05:
            best = th
        else:
            break
    K_re, _ = design_lqr(pt, Q, R)
    return ol, cl, best, float(K_re[0, 2])

def row(label, pt, **kw):
    ol, cl, best, kth = evaluate(pt, **kw)
    status = "stable" if cl < 0 else "UNSTABLE"
    print(f"  {label:34s} open loop pole {ol:5.1f}   closed loop {status:8s} ({cl:6.2f})   "
          f"recovers {best:.1f} rad   redesigned tilt gain {kth:6.2f}")

print(f"Nominal tilt gain for reference: {K_nom[0, 2]:.2f}")

print("Check 1: body mass")
for s in [0.8, 1.0, 1.2]:
    row(f"mass x{s}", replace(p, body_mass_kg=p.body_mass_kg * s))

print("Check 2: axle to center of mass distance (inertia held fixed)")
for l in [0.08, 0.10, 0.12, 0.16, 0.20]:
    row(f"l = {l} m", replace(p, axle_to_com_m=l))

print("Check 3: body pitch inertia")
for s in [0.5, 0.8, 1.2, 2.0]:
    row(f"inertia x{s}", replace(p, body_pitch_inertia_kgm2=p.body_pitch_inertia_kgm2 * s))

print("Check 4: wheel radius")
for s in [0.8, 1.2]:
    row(f"radius x{s}", with_radius(p, s))

print("Check 5: motor torque constant error (delivered torque scales the gains)")
for g in [0.5, 0.8, 1.2, 1.5, 2.0]:
    row(f"Kt error, torque x{g}", p, gain_scale=g)

print("Check 6: viscous friction (placeholder is 0.001)")
for f in [0.0, 0.001, 0.002, 0.003]:
    row(f"friction {f}", replace(p, viscous_friction_nm_s_per_rad=f))

print("Check 7: 0.2 kg payload on the arm (height above axle, forward offset)")
for h, e in [(0.10, 0.0), (0.20, 0.0), (0.30, 0.0), (0.20, 0.10), (0.30, 0.10), (0.30, 0.20)]:
    pp, d = with_payload(p, 0.2, h, e)
    row(f"payload h={h} fwd={e}", pp, delta=d)
    print(f"{'':36s} CoM line leans {np.degrees(d):4.1f} deg from the body axis")

print("Check 8: sensor noise, estimator in the loop (command 0.3 m/s, 8 s)")
for k in [1, 3, 10]:
    pn = replace(p, gyro_noise_rad_s=p.gyro_noise_rad_s * k, accel_noise_ms2=p.accel_noise_ms2 * k,
                 encoder_noise_rad_s=p.encoder_noise_rad_s * k)
    r = simulate_est(pn, K_nom, estimator="complementary", tau_c=0.5, compensate=True, v_ref=0.3, t_end=8.0)
    print(f"  noise x{k:2d}   fell={str(r['fell']):5s} tilt err rms {r['tilt_err_rms_deg']:.2f} deg   "
          f"torque rms {r['tau_rms']:.4f} N m   final v {r['final_v']:.3f}")

print("Check 9: gyro bias, not calibrated")
for b in [0.02, 0.05, 0.10]:
    pb = replace(p, gyro_bias_rad_s=b)
    r = simulate_est(pb, K_nom, estimator="complementary", tau_c=0.5, compensate=True, v_ref=0.3,
                     t_end=8.0, calibrate=False)
    print(f"  bias {b} rad/s   fell={str(r['fell']):5s} tilt err rms {r['tilt_err_rms_deg']:.2f} deg   final v {r['final_v']:.3f}")

print("Check 10: delay margin WITH the estimator and noise in the loop (start tilt 0.3 rad)")
best = -1.0
for ms in range(0, 31, 2):
    r = simulate_est(p, K_nom, estimator="complementary", tau_c=0.5, compensate=True, v_ref=0.0,
                     t_end=5.0, theta0=0.3, delay_s=ms / 1000.0)
    if (not r["fell"]) and abs(r["final_theta_deg"]) < 3.0:
        best = ms
    else:
        break
print(f"  largest passing extra delay: {best:.0f} ms")
