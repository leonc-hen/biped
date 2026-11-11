import json
import subprocess
import datetime
import numpy as np
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from params import load_params
from control import design_lqr
from sim_est import simulate_est

ROOT = Path(__file__).resolve().parent.parent
p = load_params()
Q, R = [5, 1, 50, 1], 100
K, _ = design_lqr(p, Q, R)
n, Kt, Rw = p.motor_count, p.motor_kt_nm_per_a, p.winding_resistance_ohm
bf = p.viscous_friction_nm_s_per_rad
RATE, DT = 500.0, 2e-4
EST = dict(estimator="complementary", tau_c=0.5, compensate=True)

def run(pp=None, **kw):
    args = dict(EST)
    args.update(kw)
    return simulate_est(pp or p, K, rate_hz=RATE, dt=DT, **args)

def motor_signals(res):
    """Per motor torque (N m) and current (A) over time, rebuilt from the applied torque."""
    tau_m = res["tau_applied"] / n + bf * res["w_rel"]     # undo the friction loss
    return tau_m, tau_m / Kt

# ---- 1. Largest recovered start tilt ----
def tilt_limit():
    best = 0.0
    for i in range(1, 13):
        th = round(0.1 * i, 1)
        r = run(theta0=th, v_ref=0.0, t_end=4.0)
        if (not r["fell"]) and abs(r["final_theta_deg"]) < 3.0:
            best = th
        else:
            break
    return best

def recovery_time(th0):
    """First time after which the true tilt stays within 0.05 rad of upright."""
    r = run(theta0=th0, v_ref=0.0, t_end=6.0)
    t = (np.arange(len(r["theta"])) + 1) * DT
    outside = np.where(np.abs(r["theta"]) > 0.05)[0]
    if r["fell"]:
        return None
    return float(t[outside[-1]] + DT) if len(outside) else 0.0

# ---- 2. Largest survived push ----
def push_limit():
    best = 0.0
    for v0 in [0.5 * i for i in range(1, 17)]:
        r = run(theta0=0.0, v0=v0, v_ref=0.0, t_end=6.0)
        if (not r["fell"]) and abs(r["final_theta_deg"]) < 3.0 and abs(r["final_v"]) < 0.1:
            best = v0
        else:
            break
    return best

# ---- 3. Velocity tracking ----
def tracking(vref):
    r = run(theta0=0.0, v_ref=vref, t_end=12.0)
    v = r["v"]
    t = (np.arange(len(v)) + 1) * DT
    outside = np.where(np.abs(v - vref) > 0.05 * vref)[0]
    settle = float(t[outside[-1]] + DT) if len(outside) else 0.0
    return {"settling_time_5pct_s": round(settle, 2),
            "overshoot_pct": round(float(100 * (v.max() - vref) / vref), 1),
            "final_error_m_s": round(float(v[-1] - vref), 4), "fell": r["fell"]}

# ---- 4. Torque and current in standard scenarios ----
def scenario_demand(**kw):
    r = run(**kw)
    tau_m, cur = motor_signals(r)
    return {"peak_torque_per_motor_nm": round(float(np.max(np.abs(tau_m))), 3),
            "rms_torque_per_motor_nm": round(float(np.sqrt(np.mean(tau_m**2))), 4),
            "peak_current_per_motor_a": round(float(np.max(np.abs(cur))), 2),
            "rms_current_per_motor_a": round(float(np.sqrt(np.mean(cur**2))), 3),
            "fell": r["fell"]}

# ---- 5. Delay margin ----
def delay_margin():
    best = -1
    for ms in range(0, 41, 2):
        r = run(theta0=0.3, v_ref=0.0, t_end=5.0, delay_s=ms / 1000.0)
        if (not r["fell"]) and abs(r["final_theta_deg"]) < 3.0:
            best = ms
        else:
            break
    return best

# ---- 6. RMS tilt under noise, standing still ----
def noise_rms(scale):
    vals = []
    for seed in [1, 2, 3]:
        pn = replace(p, gyro_noise_rad_s=p.gyro_noise_rad_s * scale, accel_noise_ms2=p.accel_noise_ms2 * scale,
                     encoder_noise_rad_s=p.encoder_noise_rad_s * scale)
        r = run(pn, theta0=0.0, v_ref=0.0, t_end=8.0, seed=seed)
        th = r["theta"][int(1.0 / DT):]
        vals.append(float(np.degrees(np.sqrt(np.mean(th**2)))))
    return round(float(np.mean(vals)), 4)

# ---- 7. Energy per meter, cruising at 1 m/s ----
def energy_per_meter(vref=1.0):
    r = run(theta0=0.0, v_ref=vref, t_end=12.0)
    tau_m, cur = motor_signals(r)
    w = r["w_rel"]
    copper = n * Rw * cur**2
    mech = np.maximum(n * tau_m * w, 0.0)          # no credit for regeneration
    P = copper + mech
    lo, hi = int(8.0 / DT), len(P)                   # cruise window, 8 s to 12 s
    energy = float(np.sum(P[lo:hi]) * DT)
    dist = float(np.sum(r["v"][lo:hi]) * DT)
    return {"joules_per_meter": round(energy / dist, 3), "mean_power_w": round(energy / (hi - lo) / DT, 3),
            "copper_w": round(float(np.mean(copper[lo:hi])), 3), "mechanical_w": round(float(np.mean(mech[lo:hi])), 3)}

def git_info():
    def g(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    try:
        return g("rev-parse", "--short", "HEAD"), bool(g("status", "--porcelain"))
    except Exception:
        return "unknown", None

print("Running baseline metrics (this takes several minutes)...")
res = {}
res["1_max_recovered_tilt_rad"] = tilt_limit();                  print("1 tilt limit", res["1_max_recovered_tilt_rad"])
res["1b_recovery_time_s"] = {"0.3_rad": recovery_time(0.3), "0.9_rad": recovery_time(0.9)}; print("1b", res["1b_recovery_time_s"])
res["2_max_push_m_s"] = push_limit();                            print("2 push limit", res["2_max_push_m_s"])
res["3_velocity_tracking"] = {"0.3_m_s": tracking(0.3), "1.0_m_s": tracking(1.0)}; print("3", res["3_velocity_tracking"])
res["4_demands"] = {"recover_0.3_rad": scenario_demand(theta0=0.3, v_ref=0.0, t_end=4.0),
                    "push_0.6_m_s": scenario_demand(theta0=0.0, v0=0.6, v_ref=0.0, t_end=4.0),
                    "drive_1.0_m_s": scenario_demand(theta0=0.0, v_ref=1.0, t_end=12.0)}; print("4", res["4_demands"])
res["5_delay_margin_ms"] = delay_margin();                       print("5 delay margin", res["5_delay_margin_ms"])
res["6_rms_tilt_deg_standing"] = {"noise_x1": noise_rms(1), "noise_x3": noise_rms(3)}; print("6", res["6_rms_tilt_deg_standing"])
res["7_energy"] = energy_per_meter();                            print("7", res["7_energy"])

commit, dirty = git_info()
out = {
    "meta": {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "git_commit": commit, "git_uncommitted_changes": dirty,
             "note": "All values use PLACEHOLDER parameters. Regenerate after every params.yaml change."},
    "conditions": {"loop_rate_hz": RATE, "sim_step_s": DT, "estimator": EST, "actuator": "torque mode with limits and friction",
                   "fall_threshold_rad": 1.3, "noise_seed_list": [1, 2, 3], "recovery_criterion":
                   "no fall and final tilt under 3 deg, final speed under 0.1 m/s for pushes"},
    "controller": {"type": "LQR, state [z, v, theta, thetadot]", "Q_diag": Q, "R": R, "K": [round(float(k), 6) for k in K.ravel()]},
    "params": asdict(p),
    "results": res,
}
(ROOT / "baseline_lqr.json").write_text(json.dumps(out, indent=2))
print("saved", ROOT / "baseline_lqr.json", " commit", commit, " uncommitted changes:", dirty)