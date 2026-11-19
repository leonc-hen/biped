"""Actuator pass 2 (plan 2.9): real catalog leg actuators on the jump fixture.

Same scripted maximum effort jump as src/jump_feasibility.py, at 155 mm legs,
but each candidate brings its own peak torque, speed limit and mass. Speed is
the datasheet no-load output speed scaled from 24 V to the 6S nominal 22.2 V.
Hip actuators sit on the torso. Knee actuators either sit at the knee (serial
leg) or on the torso driving the knee through a belt or linkage (previews Step
15). The torso is then resized so the whole robot hits R01's target (2.0 kg) or
cap (2.25 kg), so placeholder masses don't decide the answer. Also reports the
static knee torque to stand at the nominal pose against each continuous rating.

Specs and prices are reseller listings checked 2026-11-19; recheck before buying.
Writes results/actuator_candidates.csv. Run from the repo root.
"""
import csv
import math
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.jump_feasibility import scaled_params, run_jump, HIP_Q, KNEE_Q  # noqa: E402

LEG_M = 0.155
V_BATT = 22.2                 # 6S nominal
TARGET, STRETCH = 0.10, 0.12  # R06 as amended (D18)
MASSES = [2.0, 2.25]          # R01 target and cap

# name, peak N m, no-load output rpm at 24 V, rated N m, mass kg with driver, USD per unit
CANDIDATES = [
    ("CubeMars AK40-10", 4.1, 435, 1.3, 0.195, "136-163"),
    ("CubeMars AKE40-6 + driver", 5.5, 330, 1.8, 0.170, "not listed"),  # 140 g + assumed 30 g driver and encoder
    ("SteadyWin GIM4305-10", 3.47, 366, 1.0, 0.150, "70-132"),
    ("SteadyWin GIM4310-10 V1.1", 5.6, 228, 2.05, 0.227, "82 + driver"),
    ("DJI M2006 + C610", 1.8, 500, 1.0, 0.107, "41-95 + ESC"),
]


def place_actuators(model, m_act, knee_at_hip, total_target):
    """Add actuator masses, then resize the torso so the robot weighs total_target.
    Returns the torso mass left for everything that is not an actuator."""
    torso = model.body("torso").id
    on_torso = 2 * m_act + (2 * m_act if knee_at_hip else 0.0)
    if not knee_at_hip:
        for name in ("thigh_l", "thigh_r"):
            b = model.body(name).id
            m_t, z_t = float(model.body_mass[b]), float(model.body_ipos[b][2])
            z_a = 2 * z_t                                  # the knee, at the far end of the thigh
            m_new = m_t + m_act
            z_new = (m_t * z_t + m_act * z_a) / m_new
            extra = m_t * (z_t - z_new) ** 2 + m_act * (z_a - z_new) ** 2
            model.body_mass[b] = m_new
            model.body_ipos[b][2] = z_new
            model.body_inertia[b][0] += extra
            model.body_inertia[b][1] += extra
    model.body_mass[torso] += on_torso
    excess = float(sum(model.body_mass)) - total_target
    new_torso = float(model.body_mass[torso]) - excess
    model.body_inertia[torso] *= new_torso / float(model.body_mass[torso])
    model.body_mass[torso] = new_torso
    return new_torso - on_torso


def stand_com(model, p, a_nom, torso_id):
    d = mujoco.MjData(model)
    d.qpos[2] = p["wheel"]["radius_m"] + 2 * p["thigh"]["length_m"] * math.cos(a_nom)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for hq, kq in zip(HIP_Q, KNEE_Q):
        d.qpos[hq], d.qpos[kq] = -a_nom, 2 * a_nom
    mujoco.mj_forward(model, d)
    return float(d.subtree_com[torso_id][2])


def main():
    p0 = load_params()
    p = scaled_params(p0, LEG_M)
    g = p["sim"]["gravity_mps2"]
    a_nom = math.acos((p0["pose"]["nominal_hip_height_m"] - p0["wheel"]["radius_m"])
                      / (2 * p0["thigh"]["length_m"]))
    xml = build_xml(p, mode="torque")
    rows = []

    for knee_at_hip in (False, True):
        title = ("coaxial belt leg: both motors at the hip, knee driven by a belt" if knee_at_hip
                 else "knee motors at the knee (serial leg)")
        print(f"\n{title}, legs {LEG_M * 1000:.0f} mm   "
              f"(CoM rise above standing, cm: * reaches 10, ** reaches 12)")
        print(f"  {'candidate':28s} {'N m':>5s} {'rad/s':>6s} {'g':>5s}  "
              + "  ".join(f"{m:>7.2f} kg" for m in MASSES) + "   torso left at 2.0 kg")
        for name, tau, rpm, rated, m_act, price in CANDIDATES:
            w = rpm * 2 * math.pi / 60 * V_BATT / 24.0
            cells, left_at_target = [], None
            for M in MASSES:
                model = mujoco.MjModel.from_xml_string(xml)
                model.opt.timestep = 0.0005
                left = place_actuators(model, m_act, knee_at_hip, M)
                if M == MASSES[0]:
                    left_at_target = left
                ids = (model.geom("floor").id,
                       {model.geom("wheel_l_geom").id, model.geom("wheel_r_geom").id},
                       model.body("torso").id)
                res = run_jump(model, p, tau, w, ids, coaxial=knee_at_hip)
                rise = res["apex"] - stand_com(model, p, a_nom, ids[2])
                clr = res["wheel_clearance_m"]
                mark = "**" if rise >= STRETCH else ("* " if rise >= TARGET else "  ")
                cells.append(f"{rise * 100:6.1f}/{clr * 100:4.1f}{mark}")
                rows.append(dict(candidate=name, knee_at_hip=knee_at_hip, total_kg=M,
                                 peak_Nm=tau, speed_rad_s=round(w, 1), mass_g=round(m_act * 1000),
                                 rise_cm=round(rise * 100, 2), wheel_clear_cm=round(clr * 100, 2),
                                 torso_left_kg=round(left, 3), price_usd=price))
            print(f"  {name:28s} {tau:5.2f} {w:6.1f} {m_act * 1000:5.0f}  " + "  ".join(cells)
                  + f"   {left_at_target:.2f} kg")

    print("\ncontinuous load: static knee torque to stand at the nominal pose, percent of rated")
    for name, tau, rpm, rated, m_act, price in CANDIDATES:
        cells = []
        for M in MASSES:
            hold = M * g / 2 * LEG_M * math.sin(a_nom)
            pct = hold / rated * 100
            flag = "over" if pct > 100 else ("tight" if pct > 80 else "ok")
            cells.append(f"{M:.2f} kg: {hold:.2f} N m = {pct:4.0f}% {flag:5s}")
        print(f"  {name:28s} rated {rated:4.2f}   " + "   ".join(cells))
    print("  note: in the coaxial belt leg the hip motor carries this same static torque too")

    out = ROOT / "results" / "actuator_candidates.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()