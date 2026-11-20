"""Watch each lying pose during the 2.5 s settle: attitude every 0.1 s, and which
geoms rest on the floor at the end. Run from the repo root."""
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.step13_checks import place_on_floor, axis_vs_up_deg, settle  # noqa: E402


def floor_contacts(model, data):
    floor = model.geom("floor").id
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            names.add(model.geom(other).name or f"geom {other}")
    return sorted(names)


def main():
    model = mujoco.MjModel.from_xml_string(build_xml(load_params()))
    data = mujoco.MjData(model)
    for key in ("lying_front", "lying_back", "lying_side"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        mujoco.mj_forward(model, data)
        place_on_floor(model, data)
        print(f"\n{key}: starts touching {floor_contacts(model, data)}")
        for k in range(25):
            settle(model, data, 0.1)
            print(f"  t {0.1 * (k + 1):.1f} s   up axis {axis_vs_up_deg(model, data, 2):6.1f} deg   "
                  f"side axis {axis_vs_up_deg(model, data, 1):6.1f} deg   height {data.qpos[2]:.3f} m")
        print(f"  resting on: {floor_contacts(model, data)}")


if __name__ == "__main__":
    main()