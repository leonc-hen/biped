import mujoco
import numpy as np
import pytest

from model.build_mjcf import build_xml
from tools.params import load_params, total_mass


def make_model():
    p = load_params()
    model = mujoco.MjModel.from_xml_string(build_xml(p))
    return p, model, mujoco.MjData(model)


def test_mass_matches_params():
    p, model, _ = make_model()
    assert abs(mujoco.mj_getTotalmass(model) - total_mass(p)) < 1e-6


def test_free_fall():
    _, model, data = make_model()
    data.qpos[2] += 2.0
    mujoco.mj_forward(model, data)
    z_start = data.qpos[2]
    steps = int(0.2 / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    t = steps * model.opt.timestep
    expected_drop = 0.5 * 9.81 * t**2
    assert abs((z_start - data.qpos[2]) - expected_drop) < 0.01


def test_tips_over_without_control():
    _, model, data = make_model()
    data.qvel[4] = 0.3
    z_start = data.qpos[2]
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert data.qpos[2] < z_start - 0.1


@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize("joint", ["hip", "knee"])
def test_positive_joint_swings_foot_backward(side, joint):
    _, model, data = make_model()
    adr = model.jnt_qposadr[model.joint(f"{joint}_{side}").id]
    data.qpos[adr] = 0.3
    mujoco.mj_forward(model, data)
    assert data.body(f"wheel_{side}").xpos[0] < data.body(f"thigh_{side}").xpos[0]