"""Phase 2 step 13 checks: D08 get up geometry (run from the repo root).

Quasi static leg sweeps from the settled lying poses, no policy involved.
1. lying_front and lying_back: does some leg pose put both wheels on the floor
   and pitch the hips up at least 15 degrees? That is the lever arm a get up
   needs. If no grid point achieves it, S4 fails for geometric reasons and the
   knee range or geometry must change before any training.
2. lying_side: does some leg pose roll the body at least 30 degrees off the
   shell rail toward front or back? That is the D08 self righting route. If
   not, resize the shell (bigger side_radius_m or smaller y_offset_m) and rerun.
Writes every grid point to results/step13_getup_geometry.csv. Takes a minute or two.
"""
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

HIP_CTRL, KNEE_CTRL, WHEEL_CTRL = [0, 3], [1, 4], [2, 5]


def settle(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def axis_vs_up_deg(model, data, col):
    """Angle between a torso body axis (col 0=x, 1=y, 2=z) and world up."""
    v = data.xmat[model.body("torso").id].reshape(3, 3)[:, col]
    return math.degrees(math.acos(max(-1.0, min(1.0, v[2]))))


def wheels_on_floor(model, data):
    floor = model.geom("floor").id
    n = set()
    for i in range(data.ncon):
        pair = {data.contact[i].geom1, data.contact[i].geom2}
        if floor in pair:
            g = model.geom((pair - {floor}).pop()).name
            if g.startswith("wheel_"):
                n.add(g)
    return len(n)


def spawn_penetration_mm(model, data):
    if data.ncon == 0:
        return 0.0
    return 1000.0 * min(data.contact[i].dist for i in range(data.ncon))


def place_on_floor(model, data):
    """Lower the robot from high up until any part touches the floor, then back off
    3 mm, so lying poses start resting instead of dropped or interpenetrating."""
    floor = model.geom("floor").id

    def touching():
        return any(floor in (data.contact[i].geom1, data.contact[i].geom2)
                   for i in range(data.ncon))

    data.qpos[2] = 0.5
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    while not touching() and data.qpos[2] > -0.5:
        data.qpos[2] -= 0.002
        mujoco.mj_forward(model, data)
    data.qpos[2] += 0.003
    mujoco.mj_forward(model, data)


RAMP_S, HOLD_S = 0.6, 0.6
SETTLE_S = 2.5
HIP_Q, KNEE_Q = [7, 10], [8, 11]


def sweep(model, data, key_name, metric_col, writer):
    """Settle the keyframe, then ramp each symmetric (hip, knee) command from the
    settled joint angles over RAMP_S and hold for HOLD_S, quasi statically."""
    jr = load_params()["joints"]
    mujoco.mj_resetDataKeyframe(model, data, model.key(key_name).id)
    mujoco.mj_forward(model, data)
    place_on_floor(model, data)
    pen = spawn_penetration_mm(model, data)
    settle(model, data, SETTLE_S)
    rest_w = wheels_on_floor(model, data)
    q0, v0 = data.qpos.copy(), data.qvel.copy()
    base_angle = axis_vs_up_deg(model, data, metric_col)

    best = (-1e9, None, None, 0)
    n_ramp = int(RAMP_S / model.opt.timestep)
    for hip in np.linspace(*jr["hip"]["range_rad"], 6):
        for knee in np.linspace(*jr["knee"]["range_rad"], 6):
            data.qpos[:], data.qvel[:] = q0, v0
            mujoco.mj_forward(model, data)
            hip_now = 0.5 * (data.qpos[HIP_Q[0]] + data.qpos[HIP_Q[1]])
            knee_now = 0.5 * (data.qpos[KNEE_Q[0]] + data.qpos[KNEE_Q[1]])
            data.ctrl[:] = 0.0
            for i in range(n_ramp):
                a = (i + 1) / n_ramp
                data.ctrl[HIP_CTRL] = (1 - a) * hip_now + a * hip
                data.ctrl[KNEE_CTRL] = (1 - a) * knee_now + a * knee
                mujoco.mj_step(model, data)
            settle(model, data, HOLD_S)
            change = base_angle - axis_vs_up_deg(model, data, metric_col)
            w = wheels_on_floor(model, data)
            writer.writerow([key_name, round(hip, 3), round(knee, 3),
                             round(hip, 3), round(knee, 3), w, round(change, 1)])
            if change > best[0]:
                best = (change, hip, knee, w)
    return base_angle, best, pen, rest_w


def sweep_side(model, data, writer):
    """Asymmetric sweep from lying_side. The airborne top leg swinging through its
    range is the main roll authority, which symmetric commands cannot express.
    Tracks the MINIMUM attitude change, the most rolled off point."""
    jr = load_params()["joints"]
    mujoco.mj_resetDataKeyframe(model, data, model.key("lying_side").id)
    mujoco.mj_forward(model, data)
    place_on_floor(model, data)
    pen = spawn_penetration_mm(model, data)
    settle(model, data, SETTLE_S)
    q0, v0 = data.qpos.copy(), data.qvel.copy()
    base_angle = axis_vs_up_deg(model, data, 1)
    rest_z = axis_vs_up_deg(model, data, 2)

    grid_h = np.linspace(*jr["hip"]["range_rad"], 4)
    grid_k = np.linspace(*jr["knee"]["range_rad"], 4)
    n_ramp = int(RAMP_S / model.opt.timestep)
    worst = (1e9, (None,) * 4)
    for hl in grid_h:
        for kl in grid_k:
            for hr in grid_h:
                for kr in grid_k:
                    data.qpos[:], data.qvel[:] = q0, v0
                    mujoco.mj_forward(model, data)
                    start = [data.qpos[HIP_Q[0]], data.qpos[KNEE_Q[0]],
                             data.qpos[HIP_Q[1]], data.qpos[KNEE_Q[1]]]
                    target = [hl, kl, hr, kr]
                    data.ctrl[:] = 0.0
                    for i in range(n_ramp):
                        a = (i + 1) / n_ramp
                        for c, s0, tg in zip((0, 1, 3, 4), start, target):
                            data.ctrl[c] = (1 - a) * s0 + a * tg
                        mujoco.mj_step(model, data)
                    settle(model, data, HOLD_S)
                    change = base_angle - axis_vs_up_deg(model, data, 1)
                    writer.writerow(["lying_side", round(hl, 3), round(kl, 3),
                                     round(hr, 3), round(kr, 3),
                                     wheels_on_floor(model, data), round(change, 1)])
                    if change < worst[0]:
                        worst = (change, (hl, kl, hr, kr))
    return base_angle, worst, pen, rest_z


def main():
    p = load_params()
    model = mujoco.MjModel.from_xml_string(build_xml(p))
    data = mujoco.MjData(model)
    out = ROOT / "results" / "step13_getup_geometry.csv"
    out.parent.mkdir(exist_ok=True)
    ok = True

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pose", "hip_l_rad", "knee_l_rad", "hip_r_rad", "knee_r_rad",
                    "wheels_on_floor", "attitude_change_deg"])

        # 1 front and back: body z axis vs up; flat is near 90, pitching up reduces it
        for key in ("lying_front", "lying_back"):
            base, (gain, hip, knee, wh), pen, rest_w = sweep(model, data, key, 2, w)
            self_righted = base < 45.0 and rest_w == 2      # rolled most of the way up onto its wheels
            valid = pen > -2.0 and (45.0 <= base <= 135.0 or self_righted)
            good = valid and (self_righted or (gain >= 15.0 and wh == 2))
            ok &= good
            note = "self-righted onto its wheels" if self_righted else ""
            print(f"{key:12s} spawn pen {pen:+5.1f} mm   rest tilt {base:5.1f} deg   "
                  f"best pitch up {gain:+5.1f} deg at hip {hip:+.2f} knee {knee:+.2f} rad, "
                  f"wheels {wh}/2   {note}   "
                  f"{'PASS' if good else ('INVALID START' if not valid else 'FAIL')}")

        # 2 side: body y axis vs up, near 0 on the side; the minimum change (most
        # negative) is the largest roll off the rail
        base, (gain, pose), pen, rest_z = sweep_side(model, data, w)
        valid = pen > -2.0
        rolled_ok = base >= 30.0 and 45.0 <= rest_z <= 135.0   # landed on front or back, not upside down
        good = valid and (gain <= -30.0 or rolled_ok)
        ok &= good
        note = "rolled onto front or back during settle" if rolled_ok else ""
        print(f"{'lying_side':12s} spawn pen {pen:+5.1f} mm   rest roll {base:5.1f} deg   "
              f"best roll {-gain:+5.1f} deg at hl {pose[0]:+.2f} kl {pose[1]:+.2f} "
              f"hr {pose[2]:+.2f} kr {pose[3]:+.2f}   {note}   "
              f"{'PASS' if good else ('INVALID START' if not valid else 'FAIL')}")

        # upside down must not be a resting pose (D08): it should roll onto front or back
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        data.qpos[3:7] = (0.0436, 0.0, 0.999, 0.0)      # 175 deg about y, slightly off balance
        mujoco.mj_forward(model, data)
        place_on_floor(model, data)
        settle(model, data, SETTLE_S)
        z_ang = axis_vs_up_deg(model, data, 2)
        good = 45.0 <= z_ang <= 135.0
        ok &= good
        print(f"{'upside_down':12s} rests at {z_ang:5.1f} deg   "
              f"{'PASS (rolled onto front or back)' if good else 'FAIL (stays inverted)'}")

    print(f"wrote {out.relative_to(ROOT)}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()