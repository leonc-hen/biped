"""Jump feasibility sweep (plan 2.9 pass 1, done in simulation). How high can the
robot raise its CoM for a given leg actuator torque, speed limit and leg length,
before any policy exists?

Fixture: the vertical rail (base pinned in x, y and rotation, free in z), so no
balance is needed. Controller: scripted maximum effort from the deepest legal
crouch. Knees push with all the torque the speed envelope allows until nearly
straight; hips hold each wheel under its hip, where the ground force passes
through the hip, so the knee does essentially all the work.

Gives an upper bound on what a policy can reach (a policy also balances and
lands) and a lower bound on what the actuator must supply. Landing loads are
not covered. Rise is the apex of the whole robot CoM above its CoM when standing
at the nominal leg angles. Writes results/jump_feasibility.csv.
Run from the repo root: python src/jump_feasibility.py
"""
import copy
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.leg_actuator import torque_envelope  # noqa: E402

TORQUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]        # leg actuator peak, N m
SPEEDS = [15.0, 20.0, 30.0, 40.0, 60.0]         # leg actuator max output speed, rad/s
LENGTHS = [0.11, 0.13, 0.155]                   # thigh = shank, m
KNEE_START = 2.3       # deepest crouch, 0.1 rad inside the knee limit
KNEE_END = 0.3         # stop pushing before the straight leg singularity
TARGET_RISE = 0.25     # R06
DT = 0.0005            # finer than training: the push lasts under 100 ms

HIP_Q, KNEE_Q = [7, 10], [8, 11]
HIP_V, KNEE_V = [6, 9], [7, 10]
HIP_C, KNEE_C, WHEEL_C = [0, 3], [1, 4], [2, 5]


def scaled_params(p, length):
    """Thigh and shank at a new length; mass scales with length, inertia like a rod."""
    p = copy.deepcopy(p)
    k = length / p["thigh"]["length_m"]
    for part in ("thigh", "shank"):
        p[part]["length_m"] = length
        p[part]["mass_kg"] *= k
        i = p[part]["inertia_kgm2"]
        p[part]["inertia_kgm2"] = [i[0] * k ** 3, i[1] * k ** 3, i[2] * k]
    return p


def limited(tau, w, tau_pk, w_max):
    drive, brake = torque_envelope(w, tau_pk, w_max)
    lim = float(drive if tau * w > 0 else brake)
    return float(np.clip(tau, -lim, lim))


def wheels_touch(data, floor_id, wheel_ids):
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == floor_id and c.geom2 in wheel_ids) or \
           (c.geom2 == floor_id and c.geom1 in wheel_ids):
            return True
    return False


def run_jump(model, p, tau_pk, w_max, ids):
    floor_id, wheel_ids, torso_id = ids
    data = mujoco.MjData(model)
    length, r = p["thigh"]["length_m"], p["wheel"]["radius_m"]
    a0 = KNEE_START / 2
    data.qpos[2] = r + 2 * length * math.cos(a0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for hq, kq in zip(HIP_Q, KNEE_Q):
        data.qpos[hq], data.qpos[kq] = -a0, KNEE_START
    mujoco.mj_forward(model, data)

    def rail():
        data.qpos[0:2] = 0.0
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[0:2] = 0.0
        data.qvel[3:6] = 0.0

    def hip_tau(i):
        target = -data.qpos[KNEE_Q[i]] / 2           # keeps the wheel under the hip
        tau = 40.0 * (target - data.qpos[HIP_Q[i]]) - 1.0 * data.qvel[HIP_V[i]]
        return limited(tau, data.qvel[HIP_V[i]], tau_pk, w_max)

    # settle into the crouch for 0.3 s
    for _ in range(int(0.3 / DT)):
        for i in range(2):
            data.ctrl[HIP_C[i]] = hip_tau(i)
            tau = 40.0 * (KNEE_START - data.qpos[KNEE_Q[i]]) - 1.0 * data.qvel[KNEE_V[i]]
            data.ctrl[KNEE_C[i]] = limited(tau, data.qvel[KNEE_V[i]], tau_pk, w_max)
        data.ctrl[WHEEL_C] = 0.0
        rail()
        mujoco.mj_step(model, data)

    com0 = float(data.subtree_com[torso_id][2])
    apex, work, takeoff_t = com0, 0.0, None
    peak_tk = peak_wk = peak_p = 0.0
    pushing = [True, True]
    for k in range(int(0.8 / DT)):
        touch = wheels_touch(data, floor_id, wheel_ids)
        for i in range(2):
            kq, kv = data.qpos[KNEE_Q[i]], data.qvel[KNEE_V[i]]
            if pushing[i] and kq <= KNEE_END:
                pushing[i] = False
            if pushing[i]:
                drive, brake = torque_envelope(kv, tau_pk, w_max)
                tk = -float(drive if kv < 0 else brake)   # extension is negative knee motion
            else:
                tk = limited(20.0 * (KNEE_END - kq) - 0.5 * kv, kv, tau_pk, w_max)
            th = hip_tau(i)
            data.ctrl[KNEE_C[i]], data.ctrl[HIP_C[i]] = tk, th
            hv = data.qvel[HIP_V[i]]
            if pushing[i] and touch:
                work += (tk * kv + th * hv) * DT
            peak_tk = max(peak_tk, abs(tk))
            peak_wk = max(peak_wk, abs(kv))
            peak_p = max(peak_p, abs(tk * kv) + abs(th * hv))
        data.ctrl[WHEEL_C] = 0.0
        rail()
        mujoco.mj_step(model, data)
        apex = max(apex, float(data.subtree_com[torso_id][2]))
        if takeoff_t is None and k > 10 and not wheels_touch(data, floor_id, wheel_ids):
            takeoff_t = k * DT

    m, g = mujoco.mj_getTotalmass(model), p["sim"]["gravity_mps2"]
    return dict(apex=apex, com0=com0, work=work, takeoff_s=takeoff_t,
                peak_knee_Nm=peak_tk, peak_knee_rad_s=peak_wk, peak_leg_W=peak_p,
                speed_limited=peak_wk >= 0.97 * w_max,
                efficiency=(m * g * (apex - com0) / work) if work > 1e-6 else None)


def main():
    p0 = load_params()
    l0, r = p0["thigh"]["length_m"], p0["wheel"]["radius_m"]
    a_nom = math.acos((p0["pose"]["nominal_hip_height_m"] - r) / (2 * l0))
    rows, worst_eff = [], 0.0
    for length in LENGTHS:
        p = scaled_params(p0, length)
        model = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
        model.opt.timestep = DT
        ids = (model.geom("floor").id,
               {model.geom("wheel_l_geom").id, model.geom("wheel_r_geom").id},
               model.body("torso").id)

        ds = mujoco.MjData(model)                     # standing reference
        ds.qpos[2] = r + 2 * length * math.cos(a_nom)
        ds.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for hq, kq in zip(HIP_Q, KNEE_Q):
            ds.qpos[hq], ds.qpos[kq] = -a_nom, 2 * a_nom
        mujoco.mj_forward(model, ds)
        com_stand = float(ds.subtree_com[ids[2]][2])

        print(f"\nlegs {length * 1000:.0f} mm, total mass {mujoco.mj_getTotalmass(model):.2f} kg"
              f"   CoM rise above standing, cm   (* reaches R06 25 cm, s = speed limited)")
        print("  peak N m " + "".join(f"{w:>9.0f} r/s" for w in SPEEDS))
        for tau in TORQUES:
            cells = []
            for w in SPEEDS:
                res = run_jump(model, p, tau, w, ids)
                rise = res["apex"] - com_stand
                if res["efficiency"] is not None:
                    worst_eff = max(worst_eff, res["efficiency"])
                rows.append(dict(leg_m=length, peak_Nm=tau, max_rad_s=w,
                                 rise_m=round(rise, 4),
                                 rise_from_crouch_m=round(res["apex"] - res["com0"], 4),
                                 **{k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in res.items() if k not in ("apex", "com0")}))
                mark = ("*" if rise >= TARGET_RISE else " ") + ("s" if res["speed_limited"] else " ")
                cells.append(f"{rise * 100:10.1f}{mark}")
            print(f"  {tau:8.1f} " + "".join(cells))

    out = ROOT / "results" / "jump_feasibility.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"highest energy efficiency in the sweep: {worst_eff:.2f} "
          f"({'OK, below 1' if worst_eff < 1.0 else 'ABOVE 1: the fixture is adding energy, stop'})")


if __name__ == "__main__":
    main()