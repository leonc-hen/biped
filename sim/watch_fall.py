import time

import mujoco
import mujoco.viewer

from model.build_mjcf import build_xml
from tools.params import load_params


def main():
    model = mujoco.MjModel.from_xml_string(build_xml(load_params()))
    data = mujoco.MjData(model)
    data.qvel[4] = 0.3
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            wait = model.opt.timestep - (time.time() - start)
            if wait > 0:
                time.sleep(wait)


if __name__ == "__main__":
    main()
