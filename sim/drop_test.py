import mujoco

from model.build_mjcf import build_xml
from tools.params import load_params


def main():
    model = mujoco.MjModel.from_xml_string(build_xml(load_params()))
    data = mujoco.MjData(model)
    data.qvel[4] = 0.3
    steps_per_print = int(0.25 / model.opt.timestep)
    for i in range(int(3.0 / model.opt.timestep) + 1):
        if i % steps_per_print == 0:
            print(f"t={data.time:5.2f} s  torso height={data.qpos[2]:.3f} m")
        mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
