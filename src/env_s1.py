"""S1 training environment: balance and drive at fixed height (docs/interface.md v0.1).

Pure functional env: make_env() returns (reset, step, sizes). reset(rng) -> EnvState,
step(EnvState, action) -> EnvState. No auto reset and no randomization yet; both are
wrappers added in later steps. Reward weights are placeholders to tune in training.
"""
import math
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params, leg_ik  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402
from src.leg_actuator_jax import pd_torque  # noqa: E402

# qpos: 0:7 free | 7 hip_l 8 knee_l 9 wheel_l 10 hip_r 11 knee_r 12 wheel_r
# qvel: 0:6 free | 6 hip_l 7 knee_l 8 wheel_l 9 hip_r 10 knee_r 11 wheel_r
# ctrl: 0 hip_l 1 knee_l 2 wheel_l 3 hip_r 4 knee_r 5 wheel_r
LEG_Q = jnp.array([7, 8, 10, 11])
LEG_V = jnp.array([6, 7, 9, 10])
WHEEL_V = jnp.array([8, 11])
LEG_CTRL = jnp.array([0, 1, 3, 4])
WHEEL_CTRL = jnp.array([2, 5])

OBS_SIZE, ACT_SIZE = 27, 6
EPISODE_S = 10.0
W = dict(track_v=1.0, track_yaw=0.5, upright=0.5, height=0.3,
         torque=1e-3, action_rate=1e-2, fall=5.0)   # placeholder weights


class EnvState(NamedTuple):
    data: object
    obs: jnp.ndarray
    prev_action: jnp.ndarray     # tanh clipped, as the interface defines
    cmd: jnp.ndarray             # vx m/s, yaw rate rad/s, hip height m
    reward: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray


def gravity_body(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return -jnp.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def make_env():
    p = load_params()
    cpu_model = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    model = mjx.put_model(cpu_model)
    lm = p["leg_motor"]
    n_sub = round(p["sim"]["policy_dt_s"] / p["sim"]["physics_dt_s"])
    max_steps = round(EPISODE_S / p["sim"]["policy_dt_s"])
    h_nom = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, h_nom)
    stand_leg = jnp.array([q1, q2, q1, q2])
    qpos0 = jnp.array(cpu_model.key_qpos[cpu_model.key("stand").id])
    wheel_peak = p["wheel_motor"]["peak_torque_Nm"]
    # observation scales, docs/interface.md v0.1
    scale = jnp.concatenate([
        jnp.ones(3), jnp.full(3, 0.25), jnp.ones(4), jnp.full(4, 0.05),
        jnp.full(2, 0.025), jnp.ones(6), jnp.array([1.0, 0.33, 4.0]), jnp.ones(2),
    ])

    def get_obs(data, prev_action, cmd):
        g_b = gravity_body(data.qpos[3:7])
        raw = jnp.concatenate([
            g_b, data.qvel[3:6], data.qpos[LEG_Q], data.qvel[LEG_V],
            data.qvel[WHEEL_V], prev_action, cmd, jnp.zeros(2),
        ])
        return raw * scale

    def reset(rng):
        r1, r2 = jax.random.split(rng)
        cmd = jnp.array([
            jax.random.uniform(r1, (), minval=-0.5, maxval=1.0),
            jax.random.uniform(r2, (), minval=-1.0, maxval=1.0),
            h_nom,
        ])
        data = mjx.make_data(model).replace(qpos=qpos0, qvel=jnp.zeros(model.nv))
        data = mjx.forward(model, data)
        a0 = jnp.zeros(ACT_SIZE)
        return EnvState(data, get_obs(data, a0, cmd), a0, cmd,
                        jnp.zeros(()), jnp.zeros(()), jnp.zeros((), dtype=jnp.int32))

    def step(state, action):
        a = jnp.tanh(action)
        leg_target = stand_leg + 0.5 * a[:4]
        wheel_tau = wheel_peak * a[4:6]

        def sub(d, _):
            tau_leg = pd_torque(d.qpos[LEG_Q], d.qvel[LEG_V], leg_target,
                                lm["kp_Nm_per_rad"], lm["kv_Nms_per_rad"],
                                lm["peak_torque_Nm"], lm["max_speed_rad_s"])
            ctrl = jnp.zeros(ACT_SIZE).at[LEG_CTRL].set(tau_leg).at[WHEEL_CTRL].set(wheel_tau)
            d = d.replace(ctrl=ctrl)
            return mjx.step(model, d), d.ctrl

        data, ctrl_hist = jax.lax.scan(sub, state.data, None, length=n_sub)

        g_b = gravity_body(data.qpos[3:7])
        upright = -g_b[2]
        height = data.qpos[2]
        track_v = jnp.exp(-jnp.square(data.qvel[0] - state.cmd[0]) / 0.25)
        track_yaw = jnp.exp(-jnp.square(data.qvel[5] - state.cmd[1]) / 0.25)
        height_r = jnp.exp(-jnp.square(height - state.cmd[2]) / 0.01)
        cost_tau = jnp.mean(jnp.sum(jnp.square(ctrl_hist), axis=-1))
        cost_rate = jnp.sum(jnp.square(a - state.prev_action))

        steps = state.steps + 1
        fell = (upright < math.cos(1.0)) | (height < 0.08)
        done = (fell | (steps >= max_steps)).astype(jnp.float32)
        reward = (W["track_v"] * track_v + W["track_yaw"] * track_yaw
                  + W["upright"] * upright + W["height"] * height_r
                  - W["torque"] * cost_tau - W["action_rate"] * cost_rate
                  - W["fall"] * fell.astype(jnp.float32))

        return EnvState(data, get_obs(data, a, state.cmd), a, state.cmd,
                        reward, done, steps)

    return reset, step, dict(obs=OBS_SIZE, act=ACT_SIZE, n_sub=n_sub,
                             max_steps=max_steps)