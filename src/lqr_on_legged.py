"""Scripted balance test on the training plant, no RL. Legs are held at the nominal
IK pose by the same per motor PD the env uses; wheels run the Phase 1 LQR from
baseline_lqr.json on true state. Separates "the robot can't do it" from "the
policy didn't learn it". Run from the repo root:
python src/lqr_on_legged.py [--kp 30 --kd 0.4 --dt 0.002]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params, leg_ik  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.leg_actuator import pd_torque, torque_envelope  # noqa: E402

LEG_QI, LEG_VI, WHEEL_VI = [7, 8, 10, 11], [6, 7, 9, 10], [8, 11]
LEG_C, WHEEL_C = [0, 1, 3, 4], [2, 5]


def to_motor(v, coaxial):
    v = np.array(v, dtype=float)
    if coaxial:
        v[1] += v[0]
        v[3] += v[2]
    return v


def simulate(p, model, K, kp, kd, theta0=0.0, v_ref=0.0, push=0.0, t_end=8.0):
    coaxial = p.get("legs", {}).get("topology", "serial") == "coaxial_belt"
    lm, wm = p["leg_motor"], p["wheel_motor"]
    r = p["wheel"]["radius_m"]
    h = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, h)
    tgt_m = to_motor([q1, q2, q1, q2], coaxial)

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    h_t = h - r
    data.qpos[0] = h_t * math.sin(theta0)
    data.qpos[2] = r + h_t * math.cos(theta0) + p["sim"]["spawn_clearance_m"]
    data.qpos[3:7] = (math.cos(theta0 / 2), 0.0, math.sin(theta0 / 2), 0.0)
    mujoco.mj_forward(model, data)

    wheels = [model.body("wheel_l").id, model.body("wheel_r").id]
    others = [b for b in range(1, model.nbody) if b not in wheels]
    m_o = model.body_mass[others]
    torso = model.body("torso").id

    def axle_and_theta():
        axle = data.xpos[wheels].mean(0)
        com = (m_o[:, None] * data.xipos[others]).sum(0) / m_o.sum()
        return axle[0], math.atan2(com[0] - axle[0], com[2] - axle[2])

    dt_ctrl = p["sim"]["policy_dt_s"]
    n_sub = round(dt_ctrl / model.opt.timestep)
    x0, th_prev = axle_and_theta()
    x_prev, z = x0, 0.0
    log = []
    for k in range(int(t_end / dt_ctrl)):
        t = k * dt_ctrl
        if push and abs(t - 1.0) < dt_ctrl / 2:
            data.qvel[0] += push
        x, th = axle_and_theta()
        v = (x - x_prev) / dt_ctrl
        thd = (th - th_prev) / dt_ctrl
        x_prev, th_prev = x, th
        z = (x - x0) if v_ref == 0.0 else z + (v - v_ref) * dt_ctrl
        u = -float(K @ np.array([z, v, th, thd]))
        for _ in range(n_sub):
            q_m = to_motor(data.qpos[LEG_QI], coaxial)
            qd_m = to_motor(data.qvel[LEG_VI], coaxial)
            data.ctrl[LEG_C] = pd_torque(q_m, qd_m, tgt_m, kp, kd,
                                         lm["peak_torque_Nm"], lm["max_speed_rad_s"])
            wv = data.qvel[WHEEL_VI]
            drive, brake = torque_envelope(wv, wm["peak_torque_Nm"], wm["max_speed_rad_s"])
            tau = np.full(2, u / 2)
            lim = np.where(tau * wv > 0, drive, brake)
            data.ctrl[WHEEL_C] = np.clip(tau, -lim, lim)
            mujoco.mj_step(model, data)
        zax = data.xmat[torso].reshape(3, 3)[:, 2]
        log.append((t, th, v, math.degrees(math.acos(max(-1.0, min(1.0, zax[2]))))))
        if abs(th) > 1.0:
            return np.array(log), True
    return np.array(log), False


def wobble(log, seconds=3.0, dt=0.01):
    seg = log[-int(seconds / dt):, 3]
    s = seg - seg.mean()
    freq = np.count_nonzero(np.diff(np.sign(s)) != 0) / 2 / seconds
    return seg.std(), freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    args = ap.parse_args()
    ok = True
    p = load_params()
    kp = args.kp if args.kp is not None else p["leg_motor"]["kp_Nm_per_rad"]
    kd = args.kd if args.kd is not None else p["leg_motor"]["kv_Nms_per_rad"]
    K = np.array(json.load(open(ROOT / "baseline_lqr.json"))["controller"]["K"])
    model = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    if args.dt:
        model.opt.timestep = args.dt
    print(f"LQR K {K}   leg PD kp {kp} kd {kd}   physics dt {model.opt.timestep} s")

    log, fell = simulate(p, model, K, kp, kd)
    sd, f = wobble(log)
    print(f"stand still       fell {fell}   tilt wobble {sd:.3f} deg at {f:.1f} Hz   "
          f"creep {log[-200:, 2].mean():+.3f} m/s")
    ok &= (not fell) and sd < 0.05 and abs(log[-200:, 2].mean()) < 0.02
    for th0 in (0.2, 0.35):
        log, fell = simulate(p, model, K, kp, kd, theta0=th0)
        print(f"tilt {th0:.2f} rad     fell {fell}   final lean "
              f"{math.degrees(log[-50:, 1].mean()):+.2f} deg")
        ok &= (not fell) and abs(math.degrees(log[-50:, 1].mean())) < 1.0
    log, fell = simulate(p, model, K, kp, kd, push=0.4)
    print(f"push 0.4 m/s      fell {fell}   final speed {log[-100:, 2].mean():+.3f} m/s")
    ok &= (not fell) and abs(log[-100:, 2].mean()) < 0.05
    for vr in (0.3, 1.0):
        log, fell = simulate(p, model, K, kp, kd, v_ref=vr, t_end=12.0)
        sd, f = wobble(log)
        print(f"drive {vr:.1f} m/s     fell {fell}   speed over last 2 s "
              f"{log[-200:, 2].mean():+.3f} m/s   wobble {sd:.3f} deg at {f:.1f} Hz")
        ok &= (not fell) and abs(log[-200:, 2].mean() - vr) < 0.05 * vr and sd < 0.05
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()