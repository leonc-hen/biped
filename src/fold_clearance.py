"""Fold clearance (D01 fallback 2 input): with the body held upright in the air, sweep the hip
and knee through their ranges and report where a shank or wheel hits the torso or shell.
Also prints the lateral layout: leg plane, wheel faces and rail extent.
Run from the repo root: python -m src.fold_clearance
"""
import mujoco
import numpy as np

from model.build_mjcf import build_xml
from tools.params import load_params


def main():
    p = load_params()
    m = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    d.qpos[2] = 1.0                                     # in the air, nothing touches the floor
    gid = lambda n: m.geom(n).id
    legs = [f"{a}_{s}_geom" for a in ("shank", "wheel") for s in "lr"]
    body = [n for n in ("torso_geom", "shell_l_geom", "shell_r_geom", "shell_top_geom")
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
    adr = {j: m.joint(j).qposadr[0] for j in ("hip_l", "knee_l", "hip_r", "knee_r")}
    (h_lo, h_hi), (k_lo, k_hi) = m.joint("hip_l").range, m.joint("knee_l").range
    ft = np.zeros(6)

    mujoco.mj_forward(m, d)
    wy, ry = d.geom_xpos[gid("wheel_l_geom"), 1], d.geom_xpos[gid("shell_l_geom"), 1]
    w_half = p["wheel"]["width_m"] / 2
    print("lateral layout, left side (m from the midline):")
    print(f"  leg plane (hip) {p['body']['hip_separation_m'] / 2:.3f}   "
          f"wheel faces {wy - w_half:.3f} to {wy + w_half:.3f}   "
          f"rail spans {ry - m.geom_size[gid('shell_l_geom')][0]:.3f} to "
          f"{ry + m.geom_size[gid('shell_l_geom')][0]:.3f}")

    hips = np.arange(h_lo, h_hi + 1e-9, 0.05)
    print(f"\n{'knee':>6s}  {'hip angles clear':>17s}  worst contact")
    for k in np.arange(1.5, k_hi + 1e-9, 0.15).tolist() + [k_hi]:
        clear, worst, which = 0, 1.0, ""
        for h in hips:
            for s in "lr":
                d.qpos[adr[f"hip_{s}"]], d.qpos[adr[f"knee_{s}"]] = h, k
            mujoco.mj_forward(m, d)
            dmin = 1.0
            for a in legs:
                for b in body:
                    dist = mujoco.mj_geomDistance(m, d, gid(a), gid(b), 0.05, ft)
                    if dist < dmin:
                        dmin, w = dist, f"{a} / {b} at hip {h:+.2f}"
            clear += dmin > 0.002
            if dmin < worst:
                worst, which = dmin, w
        print(f"{k:6.2f}  {clear:6d} of {len(hips):3d}      "
              f"{worst * 1000:+6.1f} mm  {which if worst < 0.002 else ''}")
    print("\nclear = more than 2 mm between every shank or wheel and the torso or shell")


if __name__ == "__main__":
    main()