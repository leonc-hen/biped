"""Phase 2 step 14 checks: the torque mode model driven by src/leg_actuator.py
must reproduce the position mode model (run from the repo root).

1. Identical ramped crouch on the rail in both models: joint trajectories match.
2. Envelope: under a full torque command, free leg speed saturates at or below
   max_speed_rad_s.
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params, leg_ik  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.leg_actuator import pd_torque  # noqa: E402

HIP_Q, KNEE_Q = [7, 10], [8, 11]
LEG_V = [6, 7, 9, 10]


def rail(data):
    data.qpos[0:2] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[0:2] = 0.0
    data.qvel[3:6] = 0.0


def run_ramp(model, data, p, mode):
    lm = p["leg_motor"]
    coaxial = p.get("legs", {}).get("topology", "serial") == "coaxial_belt"
    q1, q2 = leg_ik(p, p["pose"]["nominal_hip_height_m"])
    c1, c2 = leg_ik(p, 0.16)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    n = int(1.0 / model.opt.timestep)
    traj = np.zeros((n, 4))
    leg_ctrl, wheel_ctrl = [0, 1, 3, 4], [2, 5]
    for i in range(n):
        a = min(1.0, 2.0 * (i + 1) / n)          # ramp over the first half, hold after
        tgt = np.array([(1 - a) * q1 + a * c1, (1 - a) * q2 + a * c2] * 2)
        if mode == "position":
            data.ctrl[leg_ctrl] = tgt
        else:
            q = data.qpos[[HIP_Q[0], KNEE_Q[0], HIP_Q[1], KNEE_Q[1]]]
            v = data.qvel[LEG_V]
            tau = pd_torque(q, v, tgt, lm["kp_Nm_per_rad"], lm["kv_Nms_per_rad"],
                            lm["peak_torque_Nm"], lm["max_speed_rad_s"])
            if coaxial:   # belt motor carries the knee torque; hip motor makes up the rest
                tau = np.array([tau[0] - tau[1], tau[1], tau[2] - tau[3], tau[3]])
            data.ctrl[leg_ctrl] = tau
        data.ctrl[wheel_ctrl] = 0.0
        rail(data)
        mujoco.mj_step(model, data)
        traj[i] = data.qpos[[HIP_Q[0], KNEE_Q[0], HIP_Q[1], KNEE_Q[1]]]
    return traj


def main():
    p = load_params()
    coaxial = p.get("legs", {}).get("topology", "serial") == "coaxial_belt"
    m_pos = mujoco.MjModel.from_xml_string(build_xml(p, mode="position"))
    m_trq = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    ok = True

    # 1 trajectory parity
    t_pos = run_ramp(m_pos, mujoco.MjData(m_pos), p, "position")
    t_trq = run_ramp(m_trq, mujoco.MjData(m_trq), p, "torque")
    err = np.max(np.abs(t_pos - t_trq))
    good = err < 0.02
    ok &= good
    print(f"1 PD parity          max joint difference {err * 1000:6.2f} mrad   "
          f"{'PASS' if good else 'FAIL'}")

    # 2 speed envelope on an airborne leg
    data = mujoco.MjData(m_trq)
    mujoco.mj_resetDataKeyframe(m_trq, data, m_trq.key("stand").id)
    data.qpos[2] = 0.6
    mujoco.mj_forward(m_trq, data)
    lm = p["leg_motor"]
    for _ in range(int(1.0 / m_trq.opt.timestep)):
        v = data.qvel[LEG_V]
        if coaxial:
            v = np.array([v[0], v[0] + v[1], v[2], v[2] + v[3]])
        from src.leg_actuator import torque_envelope
        drive, _ = torque_envelope(v, lm["peak_torque_Nm"], lm["max_speed_rad_s"])
        data.ctrl[[0, 1, 3, 4]] = drive          # push as hard as the envelope allows
        data.ctrl[[2, 5]] = 0.0
        data.qpos[0:7] = (0, 0, 0.6, 1, 0, 0, 0)
        data.qvel[0:6] = 0.0
        mujoco.mj_step(m_trq, data)
    v = data.qvel[LEG_V]
    if coaxial:
        v = np.array([v[0], v[0] + v[1], v[2], v[2] + v[3]])
    v_max = np.max(np.abs(v))
    good = v_max <= lm["max_speed_rad_s"] * 1.02
    ok &= good
    print(f"2 speed envelope     peak leg speed {v_max:5.1f} rad/s   "
          f"limit {lm['max_speed_rad_s']:.1f}   {'PASS' if good else 'FAIL'}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()