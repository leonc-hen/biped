"""Phase 2 step 12 checks: legged model sanity (run from the repo root).

The stand pose has no balance controller and the CoM sits about 9 mm forward of
the axle, so the free robot tips within a second. Checks therefore either use a
short window (1), expect the fall (1b), remove contact (3a), start from an
exactly upright seated state (3b), or pin the base to a vertical rail (4).
"""
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params, leg_ik, total_mass  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.planar_from_legged import derive  # noqa: E402

HIP_L, KNEE_L, WHEEL_L, HIP_R, KNEE_R, WHEEL_R = 7, 8, 9, 10, 11, 12   # qpos
V_WHEELS = [8, 11]                                                     # qvel


def reset_key(model, data, name):
    mujoco.mj_resetDataKeyframe(model, data, model.key(name).id)
    mujoco.mj_forward(model, data)


def step_free(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def step_rail(model, data, seconds):
    """Base pinned in x, y and rotation, free in z: a vertical rail fixture."""
    for _ in range(int(seconds / model.opt.timestep)):
        data.qpos[0:2] = 0.0
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[0:2] = 0.0
        data.qvel[3:6] = 0.0
        mujoco.mj_step(model, data)


def leg_extension(p, q_hip, q_knee):
    l1, l2 = p["thigh"]["length_m"], p["shank"]["length_m"]
    return l1 * math.cos(q_hip) + l2 * math.cos(q_hip + q_knee)


def tilt_rad(model, data):
    zaxis = data.xmat[model.body("torso").id].reshape(3, 3)[:, 2]
    return math.acos(max(-1.0, min(1.0, zaxis[2])))


def wheels_on_floor(model, data):
    floor = model.geom("floor").id
    touching = set()
    for i in range(data.ncon):
        pair = {data.contact[i].geom1, data.contact[i].geom2}
        if floor in pair:
            name = model.geom((pair - {floor}).pop()).name
            if name.startswith("wheel_"):
                touching.add(name)
    return touching


def main():
    p = load_params()
    model = mujoco.MjModel.from_xml_string(build_xml(p))
    data = mujoco.MjData(model)
    ok = True

    # 1 leg stiffness and wheel contact on the rail (no balance controller exists yet)
    reset_key(model, data, "stand")
    q1, q2 = leg_ik(p, p["pose"]["nominal_hip_height_m"])
    ext_cmd = leg_extension(p, q1, q2)
    step_rail(model, data, 0.5)
    ext_now = 0.5 * (
        leg_extension(p, data.qpos[HIP_L], data.qpos[KNEE_L])
        + leg_extension(p, data.qpos[HIP_R], data.qpos[KNEE_R])
    )
    sag = ext_cmd - ext_now
    touch = wheels_on_floor(model, data)
    good = abs(sag) < 0.005 and len(touch) == 2
    ok &= good
    print(f"1  leg sag on rail    {sag * 1000:+5.1f} mm   wheels {len(touch)}/2   "
          f"{'PASS' if good else 'FAIL'}")

    # 1b it must tip over without a controller
    reset_key(model, data, "stand")
    step_free(model, data, 1.7)
    tilt = tilt_rad(model, data)
    good = tilt > 0.5
    ok &= good
    print(f"1b tips over by 2 s   tilt {math.degrees(tilt):5.1f} deg   "
          f"{'PASS (unstable as expected)' if good else 'FAIL (should have fallen)'}")

    # 2 free fall timing
    reset_key(model, data, "stand")
    drop = 0.05
    data.qpos[2] += drop
    mujoco.mj_forward(model, data)
    t = 0.0
    while data.ncon == 0 and t < 0.5:
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    expected = math.sqrt(2 * drop / p["sim"]["gravity_mps2"])
    good = abs(t - expected) / expected < 0.10
    ok &= good
    print(f"2  free fall          contact {t * 1000:5.1f} ms   expected {expected * 1000:5.1f} ms   "
          f"{'PASS' if good else 'FAIL'}")

    # 3a airborne wheel spin: exact first order response with damping
    reset_key(model, data, "stand")
    data.qpos[2] = 0.6
    mujoco.mj_forward(model, data)
    tau = 0.02
    acts = [model.actuator("wheel_l_torque").id, model.actuator("wheel_r_torque").id]
    t_run = 0.1
    w0 = data.qvel[V_WHEELS[0]]
    for _ in range(int(t_run / model.opt.timestep)):
        data.ctrl[acts] = tau
        data.qpos[0:7] = (0, 0, 0.6, 1, 0, 0, 0)
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)
    w_meas = data.qvel[V_WHEELS[0]] - w0
    I_spin = (p["wheel"]["inertia_kgm2"][1]
              + p["wheel_motor"]["rotor_inertia_kgm2"] * p["wheel_motor"]["gear_ratio"] ** 2)
    b_w = p["joints"]["wheel_damping_Nms_per_rad"]
    w_pred = (tau / b_w) * (1.0 - math.exp(-b_w * t_run / I_spin))
    good = abs(w_meas - w_pred) / w_pred < 0.05
    ok &= good
    print(f"3a wheel spin (air)   {w_meas:6.2f} rad/s after {t_run * 1000:.0f} ms   "
          f"predicted {w_pred:6.2f}   terminal {tau / b_w:5.1f} rad/s   "
          f"{'PASS' if good else 'FAIL'}")

    # 3b ground rolling against the planar two body prediction
    d, _ = derive(p)
    r = d["wheel_radius_m"]
    m = d["body_mass_kg"]
    l = d["axle_to_com_m"]
    m_eff = d["wheel_mass_total_kg"] + d["wheel_inertia_total_kgm2"] / r ** 2
    a_ = m_eff + m
    b_ = m * l
    J_ = d["body_pitch_inertia_kgm2"] + m * l ** 2
    D_ = a_ * J_ - b_ ** 2
    tau_tot = 2 * tau
    xdd_axle = tau_tot * (J_ / r + b_) / D_
    thdd = -tau_tot * (a_ + b_ / r) / D_
    h_t = p["pose"]["nominal_hip_height_m"] - r
    a_pred = xdd_axle + h_t * thdd
    # 3b ground rolling, differential: a torqued run minus a free run from the
    # same state cancels the gravity tipping from the forward CoM offset
    t_run = 0.05

    def torso_accel(tau_cmd):
        reset_key(model, data, "stand")
        step_rail(model, data, 0.2)            # seat the contacts, exactly upright
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v0 = data.qvel[0]
        for _ in range(int(t_run / model.opt.timestep)):
            data.ctrl[acts] = tau_cmd
            mujoco.mj_step(model, data)
        return (data.qvel[0] - v0) / t_run

    a_meas = torso_accel(tau) - torso_accel(0.0)
    good = abs(a_meas - a_pred) / abs(a_pred) < 0.20
    ok &= good
    print(f"3b rolling (ground)   torso {a_meas:5.3f} m/s2   planar predicts {a_pred:5.3f}   "
          f"(axle term {xdd_axle:5.3f}, pitch reaction {h_t * thdd:+5.3f})   "
          f"{'PASS' if good else 'FAIL'}")

    # 4 static holding torque on the vertical rail, wheels carrying full weight
    leg_acts = [model.actuator(n).id for n in ("hip_l_pos", "knee_l_pos", "hip_r_pos", "knee_r_pos")]
    W_half = total_mass(p) * p["sim"]["gravity_mps2"] / 2
    l1 = p["thigh"]["length_m"]
    rows = []
    print("4  holding torque on the rail (prediction: knee = W/2 * l1*|sin q_hip|, hip near 0)")
    for H in (0.16, 0.19, 0.23):
        qa, qb = leg_ik(p, H)
        reset_key(model, data, "stand")
        data.qpos[2] = H + p["sim"]["spawn_clearance_m"]
        data.qpos[HIP_L], data.qpos[KNEE_L] = qa, qb
        data.qpos[HIP_R], data.qpos[KNEE_R] = qa, qb
        data.ctrl[:] = [qa, qb, 0.0, qa, qb, 0.0]
        mujoco.mj_forward(model, data)
        step_rail(model, data, 1.0)
        f = data.actuator_force[leg_acts]
        knee_pred = W_half * l1 * abs(math.sin(qa))
        good = abs(abs(f[1]) - knee_pred) / knee_pred < 0.40
        ok &= good
        rows.append([H, *np.round(f, 4)])
        print(f"   H {H:.2f} m   hip {f[0]:+.3f}  knee {f[1]:+.3f}  N m   "
              f"knee predicted {knee_pred:.3f}   {'PASS' if good else 'FAIL'}")
    out = ROOT / "results" / "holding_torque.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["hip_height_m", "hip_l_Nm", "knee_l_Nm", "hip_r_Nm", "knee_r_Nm"])
        w.writerows(rows)
    print(f"   wrote {out.relative_to(ROOT)}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()