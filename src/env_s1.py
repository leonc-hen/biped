"""Training environment for S1 and S2 (docs/interface.md v0.2).

make_env(cfg) returns (reset, step, sizes, aux). Pure functional:
reset(rng) -> EnvState, step(EnvState, action) -> EnvState. No auto reset.
S1 reproduces reward v0.3 exactly (run s1_v2). S2 adds height commands, mid
episode command resampling, random pushes, a leg action low pass filter and a
stronger torso level reward. No domain randomization yet (Step 11).
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
W = dict(track_v=1.0, track_v_fine=1.0, track_yaw=0.5, upright=0.5,
         torque=1e-3, action_rate=1e-2, angvel=0.02, fall=5.0)

S1 = dict(name="s1", height_cmd=False, resample_s=None, push_every_s=None,
          push_dv=(0.0, 0.0), leg_filter_alpha=1.0,
          w_tilt=0.3, tilt_sigma2=0.006, w_height=0.3, height_sigma2=0.01,
          height_sigma2_fine=None)
S2 = dict(name="s2", height_cmd=True, resample_s=3.0, push_every_s=6.0,
          push_dv=(0.6, 0.4), leg_filter_alpha=0.5,
          w_tilt=0.5, tilt_sigma2=0.003, w_height=1.0, height_sigma2=0.0025,
          height_sigma2_fine=0.0004)
CONFIGS = dict(s1=S1, s2=S2)


class EnvState(NamedTuple):
    data: object
    obs: jnp.ndarray
    prev_action: jnp.ndarray     # applied action: tanh, leg part filtered
    cmd: jnp.ndarray             # vx m/s, yaw rate rad/s, hip height m
    reward: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray
    rng: jnp.ndarray


def gravity_body(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return -jnp.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def forward_velocity(data):
    """Speed along the torso's heading, horizontal plane. Body 1 is the torso."""
    fwd = data.xmat[1].reshape(3, 3)[:, 0]
    fwd_h = fwd.at[2].set(0.0)
    fwd_h = fwd_h / (jnp.linalg.norm(fwd_h) + 1e-6)
    return jnp.dot(fwd_h, data.qvel[0:3])


def make_env(cfg=None):
    cfg = dict(S1 if cfg is None else cfg)
    p = load_params()
    cpu_model = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    model = mjx.put_model(cpu_model)
    lm = p["leg_motor"]
    dt = p["sim"]["policy_dt_s"]
    n_sub = round(dt / p["sim"]["physics_dt_s"])
    max_steps = round(EPISODE_S / dt)
    h_nom = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, h_nom)
    stand_leg = jnp.array([q1, q2, q1, q2])
    h_lo, h_hi = p["pose"]["height_cmd_range_m"]
    r_w = p["wheel"]["radius_m"]
    l12 = p["thigh"]["length_m"] + p["shank"]["length_m"]
    qpos0 = jnp.array(cpu_model.key_qpos[cpu_model.key("stand").id])
    wheel_peak = p["wheel_motor"]["peak_torque_Nm"]
    scale = jnp.concatenate([
        jnp.ones(3), jnp.full(3, 0.25), jnp.ones(4), jnp.full(4, 0.05),
        jnp.full(2, 0.025), jnp.ones(6), jnp.array([1.0, 0.33, 4.0]), jnp.ones(2),
    ])

    def ik_jnp(h):
        qh = -jnp.arccos(jnp.clip((h - r_w) / l12, -1.0, 1.0))
        return jnp.array([qh, -2.0 * qh, qh, -2.0 * qh])

    def sample_cmd(rng):
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        vx = jax.random.uniform(r1, (), minval=-0.5, maxval=1.0)
        vx = jnp.where(jax.random.uniform(r4, ()) < 0.2, 0.0, vx)   # 20 percent pure standing
        yaw = jax.random.uniform(r2, (), minval=-1.0, maxval=1.0)
        h = (jax.random.uniform(r3, (), minval=h_lo, maxval=h_hi)
             if cfg["height_cmd"] else jnp.asarray(h_nom))
        return jnp.array([vx, yaw, h])

    def get_obs(data, prev_action, cmd):
        g_b = gravity_body(data.qpos[3:7])
        raw = jnp.concatenate([
            g_b, data.qvel[3:6], data.qpos[LEG_Q], data.qvel[LEG_V],
            data.qvel[WHEEL_V], prev_action, cmd, jnp.zeros(2),
        ])
        return raw * scale

    def reset(rng):
        rng, rc, rv = jax.random.split(rng, 3)
        cmd = sample_cmd(rc)
        dv = jax.random.uniform(rv, (2,), minval=-1.0, maxval=1.0) * jnp.array([0.5, 1.0])
        qvel0 = jnp.zeros(model.nv).at[0].set(dv[0]).at[4].set(dv[1])  # base vx, pitch rate
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos0, qvel=qvel0))
        a0 = jnp.zeros(ACT_SIZE)
        z = jnp.zeros(())
        return EnvState(data, get_obs(data, a0, cmd), a0, cmd, z, z,
                        jnp.zeros((), dtype=jnp.int32), rng)

    def step(state, action):
        rng = state.rng
        a_raw = jnp.tanh(action)
        al = cfg["leg_filter_alpha"]
        a = a_raw.at[:4].set(al * a_raw[:4] + (1.0 - al) * state.prev_action[:4])
        leg_target = ik_jnp(state.cmd[2]) + 0.5 * a[:4]
        wheel_tau = wheel_peak * a[4:6]

        data = state.data
        if cfg["push_every_s"]:
            rng, kp, kd = jax.random.split(rng, 3)
            hit = jax.random.uniform(kp, ()) < dt / cfg["push_every_s"]
            dvxy = (jax.random.uniform(kd, (2,), minval=-1.0, maxval=1.0)
                    * jnp.array(cfg["push_dv"]))
            data = data.replace(qvel=data.qvel.at[0:2].add(jnp.where(hit, dvxy, 0.0)))

        def sub(d, _):
            tau_leg = pd_torque(d.qpos[LEG_Q], d.qvel[LEG_V], leg_target,
                                lm["kp_Nm_per_rad"], lm["kv_Nms_per_rad"],
                                lm["peak_torque_Nm"], lm["max_speed_rad_s"])
            ctrl = jnp.zeros(ACT_SIZE).at[LEG_CTRL].set(tau_leg).at[WHEEL_CTRL].set(wheel_tau)
            d = d.replace(ctrl=ctrl)
            return mjx.step(model, d), d.ctrl

        data, ctrl_hist = jax.lax.scan(sub, data, None, length=n_sub)

        g_b = gravity_body(data.qpos[3:7])
        upright = -g_b[2]
        tilt_ang = jnp.arccos(jnp.clip(upright, -1.0, 1.0))
        height = data.qpos[2]
        err_v = forward_velocity(data) - state.cmd[0]
        track_v = jnp.exp(-jnp.square(err_v) / 0.25)
        track_v_fine = jnp.exp(-jnp.square(err_v) / 0.04)
        track_yaw = jnp.exp(-jnp.square(data.qvel[5] - state.cmd[1]) / 0.09)
        tilt_r = jnp.exp(-jnp.square(tilt_ang) / cfg["tilt_sigma2"])
        e_h = height - state.cmd[2]
        height_r = jnp.exp(-jnp.square(e_h) / cfg["height_sigma2"])
        if cfg["height_sigma2_fine"]:
            height_r = (0.5 * height_r
                        + 0.5 * jnp.exp(-jnp.square(e_h) / cfg["height_sigma2_fine"]))
        cost_tau = jnp.mean(jnp.sum(jnp.square(ctrl_hist), axis=-1))
        cost_rate = jnp.sum(jnp.square(a - state.prev_action))
        cost_angvel = jnp.sum(jnp.square(data.qvel[3:6]))

        steps = state.steps + 1
        fell = (upright < math.cos(1.0)) | (height < 0.08)
        done = (fell | (steps >= max_steps)).astype(jnp.float32)
        reward = (W["track_v"] * track_v + W["track_v_fine"] * track_v_fine
                  + W["track_yaw"] * track_yaw + W["upright"] * upright
                  + cfg["w_tilt"] * tilt_r + cfg["w_height"] * height_r
                  - W["torque"] * cost_tau - W["action_rate"] * cost_rate
                  - W["angvel"] * cost_angvel
                  - W["fall"] * fell.astype(jnp.float32))

        cmd = state.cmd
        if cfg["resample_s"]:
            rng, kc = jax.random.split(rng)
            every = round(cfg["resample_s"] / dt)
            cmd = jnp.where(steps % every == 0, sample_cmd(kc), cmd)

        return EnvState(data, get_obs(data, a, cmd), a, cmd, reward, done, steps, rng)

    aux = dict(p=p, cfg=cfg, model=model, qpos0=qpos0, stand_leg=stand_leg, h_nom=h_nom,
               h_range=(h_lo, h_hi), lm=lm, wheel_peak=wheel_peak, get_obs=get_obs,
               ik=ik_jnp)
    return reset, step, dict(obs=OBS_SIZE, act=ACT_SIZE, n_sub=n_sub,
                             max_steps=max_steps), aux
