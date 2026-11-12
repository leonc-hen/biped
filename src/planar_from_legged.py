"""Derive the Phase 1 planar parameters from params_legged.yaml at the nominal pose.

Default: check mode. Prints the derived values next to the current params.yaml.
--write: overwrites the derived keys in params.yaml. Only use this together with
regenerating gain_table.json and baseline_lqr.json, or the baseline becomes unfair.
Hand measured and electrical keys (motor constants, noise, friction, battery,
current limit, imu height) are never touched; they stay as params.yaml has them.
"""
import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.params import load_params, leg_ik  # noqa: E402

PLANAR_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


def rot_y(q, x, z):
    """Rotate a vector in the x z plane about +y by q (x forward, z up)."""
    return (x * math.cos(q) + z * math.sin(q), -x * math.sin(q) + z * math.cos(q))


def derive(p):
    b, th, sh, wh = p["body"], p["thigh"], p["shank"], p["wheel"]
    wm = p["wheel_motor"]
    H = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, H)
    l1, l2 = th["length_m"], sh["length_m"]

    hip_z = H - wh["radius_m"]  # everything measured from the wheel axle, x forward z up
    dx1, dz1 = rot_y(q1, 0.0, -l1 / 2)        # hip to thigh CoM
    dxk, dzk = rot_y(q1, 0.0, -l1)            # hip to knee
    dx2, dz2 = rot_y(q1 + q2, 0.0, -l2 / 2)   # knee to shank CoM

    # (mass, x, z, I_yy about own CoM); legs counted twice, wheels excluded (they are
    # the planar model's wheel term, not part of the pendulum body)
    parts = [
        (b["mass_kg"], 0.0, hip_z + b["com_height_above_hips_m"], b["inertia_kgm2"][1]),
        (p["shell"]["mass_kg"], 0.0, hip_z + p["shell"]["z_offset_m"], 0.0),
        (p["arm"]["mass_kg"], 0.0, hip_z + p["arm"]["com_above_hips_m"], 0.0),
        (2 * th["mass_kg"], dx1, hip_z + dz1, 2 * th["inertia_kgm2"][1]),
        (2 * sh["mass_kg"], dxk + dx2, hip_z + dzk + dz2, 2 * sh["inertia_kgm2"][1]),
    ]

    m = sum(row[0] for row in parts)
    x_com = sum(row[0] * row[1] for row in parts) / m
    z_com = sum(row[0] * row[2] for row in parts) / m
    inertia = sum(
        Iy + mi * ((x - x_com) ** 2 + (z - z_com) ** 2) for mi, x, z, Iy in parts
    )
    wheel_spin = 2 * (wh["inertia_kgm2"][1] + wm["rotor_inertia_kgm2"] * wm["gear_ratio"] ** 2)

    derived = {
        "wheel_mass_total_kg": round(2 * wh["mass_kg"], 6),
        "wheel_radius_m": wh["radius_m"],
        "wheel_inertia_total_kgm2": round(wheel_spin, 8),
        "body_mass_kg": round(m, 6),
        "axle_to_com_m": round(z_com, 6),
        "body_pitch_inertia_kgm2": round(inertia, 8),
        "track_width_m": b["hip_separation_m"],
    }
    return derived, x_com


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    derived, x_com = derive(load_params())
    with open(PLANAR_PATH) as f:
        planar = yaml.safe_load(f)

    print(f"{'key':28s} {'params.yaml':>14s} {'derived':>14s}")
    for k, v in derived.items():
        old = planar.get(k, float("nan"))
        print(f"{k:28s} {old:>14.6g} {v:>14.6g}")
    print(
        f"\nStanding CoM forward offset at the nominal pose: {x_com * 1000:.1f} mm.\n"
        "The planar model assumes 0; on the real robot the body leans slightly back\n"
        "to cancel it. Fine for now, revisit if it grows past about 15 mm."
    )

    if args.write:
        planar.update(derived)
        with open(PLANAR_PATH, "w") as f:
            yaml.safe_dump(planar, f, sort_keys=False)
        print(
            "\nWrote params.yaml. Now regenerate gain_table.json and baseline_lqr.json\n"
            "(src/save_gains.py, src/make_baseline.py), or the baseline is no longer fair."
        )


if __name__ == "__main__":
    main()