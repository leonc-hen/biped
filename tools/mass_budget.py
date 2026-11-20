"""Mass budget v1: does the robot close against R01 with each actuator candidate?

Reads docs/mass_budget.csv (everything except the four leg actuators), adds the
actuators where the D20 coaxial belt leg puts them (all four on the torso), adds
a contingency on items not yet measured, checks R01 (2.0 kg target, 2.25 kg cap),
and reruns the coaxial belt jump with the budget's own mass distribution instead
of placeholder masses. Run from the repo root: python tools/mass_budget.py
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
from src.jump_feasibility import scaled_params, run_jump  # noqa: E402
from src.actuator_candidates import CANDIDATES, LEG_M, V_BATT, stand_com  # noqa: E402

R01_TARGET, R01_CAP = 2.0, 2.25
CONTINGENCY = 0.10        # on every item not yet measured
BUDGET = ROOT / "docs" / "mass_budget.csv"


def load_budget():
    by_body = {"torso": 0.0, "thigh": 0.0, "shank": 0.0, "wheel": 0.0}
    unmeasured = 0.0
    with open(BUDGET) as f:
        for row in csv.DictReader(f):
            kg = float(row["qty"]) * float(row["unit_g"]) / 1000.0
            by_body[row["body"]] += kg
            if row["status"].strip() != "measured":
                unmeasured += kg
    return by_body, unmeasured


def set_body(model, name, new_mass):
    b = model.body(name).id
    model.body_inertia[b] *= new_mass / float(model.body_mass[b])
    model.body_mass[b] = new_mass


def apply_budget(model, torso_group, thigh, shank, wheel):
    """torso_group covers the torso body plus its shell and arm bodies; legs are per side."""
    attached = sum(float(model.body_mass[model.body(n).id]) for n in ("shell_l", "shell_r", "arm_mass"))
    set_body(model, "torso", torso_group - attached)
    for side in ("l", "r"):
        set_body(model, f"thigh_{side}", thigh)
        set_body(model, f"shank_{side}", shank)
        set_body(model, f"wheel_{side}", wheel)


def main():
    by_body, unmeasured = load_budget()
    contingency = CONTINGENCY * unmeasured
    base = sum(by_body.values())
    print(f"budget without leg actuators: {base:.3f} kg "
          f"(torso {by_body['torso']:.3f}, thighs {by_body['thigh']:.3f}, "
          f"shanks {by_body['shank']:.3f}, wheels {by_body['wheel']:.3f}) "
          f"+ {contingency:.3f} kg contingency on unmeasured items")

    p0 = load_params()
    p = scaled_params(p0, LEG_M)
    g = p["sim"]["gravity_mps2"]
    a_nom = math.acos((p0["pose"]["nominal_hip_height_m"] - p0["wheel"]["radius_m"])
                      / (2 * p0["thigh"]["length_m"]))
    xml = build_xml(p, mode="torque")

    print(f"\ncoaxial belt leg, {LEG_M * 1000:.0f} mm, all four leg actuators on the torso")
    print(f"  {'candidate':28s} {'total kg':>8s}  {'R01':12s} {'jump CoM/clear cm':>18s}"
          f"  {'standing load':>14s}  USD per unit")
    for name, tau, rpm, rated, m_act, price in CANDIDATES:
        total = base + contingency + 4 * m_act
        r01 = ("under target" if total <= R01_TARGET else
               "under cap" if total <= R01_CAP else "OVER CAP")
        model = mujoco.MjModel.from_xml_string(xml)
        model.opt.timestep = 0.0005
        apply_budget(model, by_body["torso"] + contingency + 4 * m_act,
                     by_body["thigh"] / 2, by_body["shank"] / 2, by_body["wheel"] / 2)
        ids = (model.geom("floor").id,
               {model.geom("wheel_l_geom").id, model.geom("wheel_r_geom").id},
               model.body("torso").id)
        w = rpm * 2 * math.pi / 60 * V_BATT / 24.0
        res = run_jump(model, p, tau, w, ids, coaxial=True)
        rise = res["apex"] - stand_com(model, p, a_nom, ids[2])
        hold = total * g / 2 * LEG_M * math.sin(a_nom)
        print(f"  {name:28s} {total:8.3f}  {r01:12s} "
              f"{rise * 100:9.1f} /{res['wheel_clearance_m'] * 100:5.1f}  "
              f"{hold / rated * 100:10.0f} %     {price}")
    print("\nstanding load: static torque on each leg motor at the nominal pose, percent of rated")


if __name__ == "__main__":
    main()