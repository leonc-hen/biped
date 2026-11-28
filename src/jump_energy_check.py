"""The training env's scripted jump replayed in CPU MuJoCo on the same torque mode
XML (coaxial belt tendon), same motor space PD and trajectory, torso on a rail at
every physics step, with exact motor work. Compare its CoM rise with the MJX value
from jump_reach_check.py. Run from the repo root: python src/jump_energy_check.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params, leg_ik  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.leg_actuator import pd_torque  # noqa: E402

LEG_QI, LEG_VI, LEG_C, WHEEL_C = [7, 8, 10, 11], [6, 7, 9, 10], [0, 1, 3, 4], [2, 5]


def to_motor(v):
    v = np.array(v, dtype=float)
    v[1] += v[0]
    v[3] += v[2]
    return v


def main():
    p = load_params()
    lm = p["leg_motor"]
    model = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("stand").id)
    mujoco.mj_forward(model, d)
    torso = model.body("torso").id
    stand = np.array(leg_ik(p, 0.25) * 2)
    crouch = np.array(leg_ik(p, 0.17) * 2)
    land = np.array(leg_ik(p, 0.22) * 2)
    ext = np.array([-0.15, 0.30, -0.15, 0.30])

    def base(t):
        if t < 0.4:
            a = t / 0.4
            return (1 - a) * stand + a * crouch
        if t < 0.55:
            return ext
        b = min(max((t - 0.85) / 0.4, 0.0), 1.0)
        return (1 - b) * land + b * stand

    dt = model.opt.timestep
    com0 = float(d.subtree_com[torso][2])
    m, g = mujoco.mj_getTotalmass(model), p["sim"]["gravity_mps2"]
    work = best = work_at_best = peak_speed = 0.0
    for k in range(int(1.4 / dt)):
        tgt = to_motor(base(k * dt))
        q, qd = to_motor(d.qpos[LEG_QI]), to_motor(d.qvel[LEG_VI])
        d.ctrl[LEG_C] = pd_torque(q, qd, tgt, lm["kp_Nm_per_rad"], lm["kv_Nms_per_rad"],
                                  lm["peak_torque_Nm"], lm["max_speed_rad_s"])
        d.ctrl[WHEEL_C] = 0.0
        d.qpos[0:2] = 0.0
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[0:2] = 0.0
        d.qvel[3:6] = 0.0
        mujoco.mj_step(model, d)
        power = d.actuator_force[LEG_C] * d.actuator_velocity[LEG_C]
        work += float(np.maximum(power, 0.0).sum()) * dt
        peak_speed = max(peak_speed, float(np.abs(d.actuator_velocity[LEG_C]).max()))
        rise = float(d.subtree_com[torso][2]) - com0
        if rise > best:
            best, work_at_best = rise, work
    print(f"CPU replay of the env jump: CoM rise {best * 100:.1f} cm, positive motor work "
          f"{work_at_best:.2f} J, efficiency {m * g * best / work_at_best:.2f}, "
          f"peak motor speed {peak_speed:.1f} rad/s")


if __name__ == "__main__":
    main()