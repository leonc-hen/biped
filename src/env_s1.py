"""Training environment for S1, S2 and S2R (docs/interface.md v0.2).

make_env(cfg) returns (reset, step, sizes, aux). Pure functional:
reset(rng) -> EnvState, step(EnvState, action) -> EnvState. No auto reset.
S1 reproduces reward v0.3 (s1_v2). S2 adds height commands, resampling, pushes,
a leg action filter and a torso level reward (s2_v0 to s2_v3). S2R is S2 plus
the plan 2.7 domain randomization (D16) and the wheel torque speed envelope.
With dr False every randomization path is skipped, so S1 and S2 are unchanged.
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
from src.leg_actuator_jax import pd_torque, torque_envelope  # noqa: E402

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
Q_RES = 2 * math.pi / 16384          # 14 bit joint encoder
W = dict(track_v=1.0, track_v_fine=1.0, track_yaw=0.5, upright=0.5,
         torque=1e-3, action_rate=1e-2, fall=5.0)

S1 = dict(name="s1", height_cmd=False, resample_s=None, push_every_s=None,
          push_dv=(0.0, 0.0), leg_filter_alpha=1.0,
          w_tilt=0.3, tilt_sigma2=0.006, w_height=0.3, height_sigma2=0.01,
          height_sigma2_fine=None, dr=False, dr_scale=0.0, wheel_envelope=False,
          cmd_fixed=None, reset_perturb=True, noise_scale=None, delay_fixed_ms=None,
          w_angvel=0.02, angvel_roll_pitch_only=False,
          w_track_v_sharp=0.0, track_v_sharp_sigma2=0.0025, push_max_norm=None)
S2 = dict(S1, name="s2", height_cmd=True, resample_s=3.0, push_every_s=6.0,
          push_dv=(0.6, 0.4), leg_filter_alpha=0.5,
          w_tilt=0.5, tilt_sigma2=0.003, w_height=1.0, height_sigma2=0.0025,
          height_sigma2_fine=0.0004)
S2R = dict(S2, name="s2r", dr=True, dr_scale=1.0, wheel_envelope=True,
           w_angvel=0.1, angvel_roll_pitch_only=True, w_track_v_sharp=0.5)
CONFIGS = dict(s1=S1, s2=S2, s2r=S2R)


class EnvState(NamedTuple):
    data: object
    obs: jnp.ndarray
    prev_action: jnp.ndarray     # applied action: tanh, leg part filtered
    cmd: jnp.ndarray             # vx m/s, yaw rate rad/s, hip height m
    reward: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray
    rng: jnp.ndarray
    dr: dict                     # this episode's randomization (nominal when dr is off)
    act_hist: jnp.ndarray        # applied actions from the previous two ticks, for delay


def quat_mat(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def gravity_body(quat):
    """World down in the body frame. The evaluators use it for tilt."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return -jnp.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def grav_dir_body(quat, g_unit):
    """The actual gravity direction in the body frame. Exactly gravity_body(quat)
    when g_unit is (0, 0, -1)."""
    R = quat_mat(quat)
    return g_unit[0] * R[0] + g_unit[1] * R[1] + g_unit[2] * R[2]


def rot_x(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


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
    lm, wm = p["leg_motor"], p["wheel_motor"]
    coaxial = p.get("legs", {}).get("topology", "serial") == "coaxial_belt"

    def to_motor(v):
        """Joint order [hip_l, knee_l, hip_r, knee_r] to motor order. In the coaxial
        belt leg motor 2 turns the shank's absolute angle, hip + knee."""
        return v.at[1].add(v[0]).at[3].add(v[2]) if coaxial else v
    dt = p["sim"]["policy_dt_s"]
    phys_dt = p["sim"]["physics_dt_s"]
    n_sub = round(dt / phys_dt)
    max_steps = round(EPISODE_S / dt)
    g_mag = p["sim"]["gravity_mps2"]
    h_nom = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, h_nom)
    stand_leg = jnp.array([q1, q2, q1, q2])
    h_lo, h_hi = p["pose"]["height_cmd_range_m"]
    r_w = p["wheel"]["radius_m"]
    l12 = p["thigh"]["length_m"] + p["shank"]["length_m"]
    qpos0 = jnp.array(cpu_model.key_qpos[cpu_model.key("stand").id])
    wheel_peak = wm["peak_torque_Nm"]
    torso = cpu_model.body("torso").id
    nbody, nv = cpu_model.nbody, cpu_model.nv
    max_delay_sub = round(0.020 / phys_dt)     # 20 ms in physics substeps
    assert max_delay_sub <= 2 * n_sub, "delay model keeps two ticks of history"
    scale = jnp.concatenate([
        jnp.ones(3), jnp.full(3, 0.25), jnp.ones(4), jnp.full(4, 0.05),
        jnp.full(2, 0.025), jnp.ones(6), jnp.array([1.0, 0.33, 4.0]), jnp.ones(2),
    ])

    nominal_dr = dict(
        mass_scale=jnp.ones(nbody), com_offset=jnp.zeros(3), inertia_scale=jnp.ones((nbody, 1)),
        mu=jnp.asarray(1.0), kt_leg=jnp.ones(4), kt_w=jnp.ones(2),
        kp_s=jnp.ones(4), kd_s=jnp.ones(4),
        arm_scale=jnp.ones(nv), damp_scale=jnp.ones(nv), fric_scale=jnp.ones(nv),
        delay_sub=jnp.asarray(0, dtype=jnp.int32), gyro_bias=jnp.zeros(3),
        mount=jnp.eye(3), v_scale=jnp.asarray(1.0), g_unit=jnp.array([0.0, 0.0, -1.0]),
    )

    def sample_dr(rng):
        s = cfg["dr_scale"]
        k = jax.random.split(rng, 16)

        def U(key, shape, lo, hi):
            return jax.random.uniform(key, shape, minval=lo, maxval=hi)

        def near1(key, shape, frac):
            return U(key, shape, 1.0 - frac * s, 1.0 + frac * s)

        deg3 = math.radians(3.0) * s
        mount_rp = U(k[13], (2,), -deg3, deg3)
        slope = U(k[14], (2,), -deg3, deg3)
        sx, sy = jnp.sin(slope[0]), jnp.sin(slope[1])
        return dict(
            mass_scale=near1(k[0], (nbody,), 0.15),
            com_offset=U(k[1], (3,), -1.0, 1.0) * jnp.array([0.015, 0.01, 0.01]) * s,
            inertia_scale=near1(k[2], (nbody, 1), 0.15),
            mu=U(k[3], (), 1.0 - 0.6 * s, 1.0 + 0.25 * s),
            kt_leg=near1(k[4], (4,), 0.10), kt_w=near1(k[5], (2,), 0.10),
            kp_s=near1(k[6], (4,), 0.10), kd_s=near1(k[7], (4,), 0.10),
            arm_scale=near1(k[8], (nv,), 0.20), damp_scale=near1(k[9], (nv,), 0.20),
            fric_scale=near1(k[10], (nv,), 0.20),
            delay_sub=(jnp.asarray(round(cfg["delay_fixed_ms"] / 1000 / phys_dt), dtype=jnp.int32)
                       if cfg["delay_fixed_ms"] is not None else
                       jax.random.randint(k[11], (), 0, int(round(max_delay_sub * s)) + 1)),
            gyro_bias=U(k[12], (3,), -0.02 * s, 0.02 * s),
            mount=rot_x(mount_rp[0]) @ rot_y(mount_rp[1]),
            v_scale=U(k[15], (), 1.0 - 0.25 * s, 1.0 + 0.05 * s),
            g_unit=jnp.array([sx, sy, -jnp.sqrt(1.0 - sx * sx - sy * sy)]),
        )

    def apply_dr(dr):
        return model.replace(
            body_mass=model.body_mass * dr["mass_scale"],
            body_ipos=model.body_ipos.at[torso].add(dr["com_offset"]),
            body_inertia=model.body_inertia * dr["inertia_scale"],
            # MuJoCo uses the larger of the two geoms' friction, so every geom gets mu
            geom_friction=model.geom_friction.at[:, 0].set(dr["mu"]),
            dof_armature=model.dof_armature * dr["arm_scale"],
            dof_damping=model.dof_damping * dr["damp_scale"],
            dof_frictionloss=model.dof_frictionloss * dr["fric_scale"],
            opt=model.opt.replace(gravity=g_mag * dr["g_unit"]),
        )

    def ik_jnp(h):
        qh = -jnp.arccos(jnp.clip((h - r_w) / l12, -1.0, 1.0))
        return jnp.array([qh, -2.0 * qh, qh, -2.0 * qh])

    def sample_cmd(rng):
        if cfg["cmd_fixed"] is not None:
            return jnp.array(cfg["cmd_fixed"], dtype=jnp.float32)
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        vx = jax.random.uniform(r1, (), minval=-0.5, maxval=1.0)
        vx = jnp.where(jax.random.uniform(r4, ()) < 0.2, 0.0, vx)
        yaw = jax.random.uniform(r2, (), minval=-1.0, maxval=1.0)
        h = (jax.random.uniform(r3, (), minval=h_lo, maxval=h_hi)
             if cfg["height_cmd"] else jnp.asarray(h_nom))
        return jnp.array([vx, yaw, h])

    def get_obs(data, prev_action, cmd, dr, rng):
        g_b = grav_dir_body(data.qpos[3:7], dr["g_unit"])
        gyro = data.qvel[3:6]
        q, qd, wd = data.qpos[LEG_Q], data.qvel[LEG_V], data.qvel[WHEEL_V]
        if cfg["dr"]:
            s = cfg["dr_scale"] if cfg["noise_scale"] is None else cfg["noise_scale"]
            k = jax.random.split(rng, 5)
            g_b = dr["mount"] @ g_b + s * 0.01 * jax.random.normal(k[0], (3,))
            gyro = (dr["mount"] @ gyro + dr["gyro_bias"]
                    + s * 0.01 * jax.random.normal(k[1], (3,)))
            q = jnp.round(q / Q_RES) * Q_RES + s * 0.002 * jax.random.normal(k[2], (4,))
            qd = qd + s * 0.05 * jax.random.normal(k[3], (4,))
            wd = wd + s * 0.1 * jax.random.normal(k[4], (2,))
        raw = jnp.concatenate([g_b, gyro, q, qd, wd, prev_action, cmd, jnp.zeros(2)])
        return raw * scale

    def init_from_data(data, cmd):
        """A full EnvState from given physics data. For evaluators."""
        a0, z = jnp.zeros(ACT_SIZE), jnp.zeros(())
        key = jax.random.PRNGKey(0)
        dr0 = sample_dr(key) if cfg["dr"] else nominal_dr
        return EnvState(data, get_obs(data, a0, cmd, dr0, key), a0, cmd, z, z,
                        jnp.zeros((), dtype=jnp.int32), key, dr0,
                        jnp.zeros((2, ACT_SIZE)))

    def reset(rng):
        rng, rc, rv, rd, ro = jax.random.split(rng, 5)
        cmd = sample_cmd(rc)
        dr = sample_dr(rd) if cfg["dr"] else nominal_dr
        m = apply_dr(dr) if cfg["dr"] else model
        if cfg["reset_perturb"]:
            dv = jax.random.uniform(rv, (2,), minval=-1.0, maxval=1.0) * jnp.array([0.5, 1.0])
            qvel0 = jnp.zeros(model.nv).at[0].set(dv[0]).at[4].set(dv[1])
        else:
            qvel0 = jnp.zeros(model.nv)
        data = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos0, qvel=qvel0))
        a0, z = jnp.zeros(ACT_SIZE), jnp.zeros(())
        return EnvState(data, get_obs(data, a0, cmd, dr, ro), a0, cmd, z, z,
                        jnp.zeros((), dtype=jnp.int32), rng, dr, jnp.zeros((2, ACT_SIZE)))

    def step(state, action):
        rng, dr = state.rng, state.dr
        m = apply_dr(dr) if cfg["dr"] else model
        a_raw = jnp.tanh(action)
        al = cfg["leg_filter_alpha"]
        a = a_raw.at[:4].set(al * a_raw[:4] + (1.0 - al) * state.prev_action[:4])
        hist = jnp.stack([a, state.act_hist[0], state.act_hist[1]])   # now, 1 and 2 ticks ago
        base = ik_jnp(state.cmd[2])
        kp = lm["kp_Nm_per_rad"] * dr["kp_s"]
        kd = lm["kv_Nms_per_rad"] * dr["kd_s"]
        leg_speed = lm["max_speed_rad_s"] * dr["v_scale"]
        wheel_speed = wm["max_speed_rad_s"] * dr["v_scale"]

        data = state.data
        if cfg["push_every_s"]:
            rng, kp_, kd_ = jax.random.split(rng, 3)
            hit = jax.random.uniform(kp_, ()) < dt / cfg["push_every_s"]
            if cfg["push_max_norm"] is None:
                dvxy = (jax.random.uniform(kd_, (2,), minval=-1.0, maxval=1.0)
                        * jnp.array(cfg["push_dv"]))
            else:
                ka, km = jax.random.split(kd_)
                ang = jax.random.uniform(ka, (), minval=0.0, maxval=2 * math.pi)
                mag = jax.random.uniform(km, (), minval=0.0, maxval=cfg["push_max_norm"])
                dvxy = mag * jnp.array([jnp.cos(ang), jnp.sin(ang)])
            data = data.replace(qvel=data.qvel.at[0:2].add(jnp.where(hit, dvxy, 0.0)))

        def sub(d, j):
            # substep j: the action in force was issued 0, 1 or 2 ticks ago
            a_eff = hist[jnp.clip((dr["delay_sub"] - j + n_sub - 1) // n_sub, 0, 2)]
            tau_leg = pd_torque(to_motor(d.qpos[LEG_Q]), to_motor(d.qvel[LEG_V]),
                                to_motor(base + 0.5 * a_eff[:4]),
                                kp, kd, lm["peak_torque_Nm"], leg_speed) * dr["kt_leg"]
            tau_w = wheel_peak * a_eff[4:6]
            if cfg["wheel_envelope"]:
                wv = d.qvel[WHEEL_V]
                drive, brake = torque_envelope(wv, wheel_peak, wheel_speed)
                lim = jnp.where(tau_w * wv > 0.0, drive, brake)
                tau_w = jnp.clip(tau_w, -lim, lim)
            tau_w = tau_w * dr["kt_w"]
            ctrl = jnp.zeros(ACT_SIZE).at[LEG_CTRL].set(tau_leg).at[WHEEL_CTRL].set(tau_w)
            d = d.replace(ctrl=ctrl)
            return mjx.step(m, d), d.ctrl

        data, ctrl_hist = jax.lax.scan(sub, data, jnp.arange(n_sub))

        g_b = grav_dir_body(data.qpos[3:7], dr["g_unit"])
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
        ang = data.qvel[3:5] if cfg["angvel_roll_pitch_only"] else data.qvel[3:6]
        cost_angvel = jnp.sum(jnp.square(ang))
        track_v_sharp = jnp.exp(-jnp.square(err_v) / cfg["track_v_sharp_sigma2"])

        steps = state.steps + 1
        fell = (upright < math.cos(1.0)) | (height < 0.08)
        done = (fell | (steps >= max_steps)).astype(jnp.float32)
        reward = (W["track_v"] * track_v + W["track_v_fine"] * track_v_fine
                  + W["track_yaw"] * track_yaw + W["upright"] * upright
                  + cfg["w_tilt"] * tilt_r + cfg["w_height"] * height_r
                  - W["torque"] * cost_tau - W["action_rate"] * cost_rate
                  - cfg["w_angvel"] * cost_angvel
                  + cfg["w_track_v_sharp"] * track_v_sharp
                  - W["fall"] * fell.astype(jnp.float32))

        cmd = state.cmd
        if cfg["resample_s"]:
            rng, kc = jax.random.split(rng)
            every = round(cfg["resample_s"] / dt)
            cmd = jnp.where(steps % every == 0, sample_cmd(kc), cmd)

        if cfg["dr"]:
            rng, ko = jax.random.split(rng)
        else:
            ko = rng
        return EnvState(data, get_obs(data, a, cmd, dr, ko), a, cmd, reward, done, steps,
                        rng, dr, jnp.stack([a, state.act_hist[0]]))

    aux = dict(p=p, cfg=cfg, model=model, qpos0=qpos0, stand_leg=stand_leg, h_nom=h_nom,
               h_range=(h_lo, h_hi), lm=lm, wheel_peak=wheel_peak, ik=ik_jnp,
               nominal_dr=nominal_dr, sample_dr=sample_dr, apply_dr=apply_dr,
               init_from_data=init_from_data, to_motor=to_motor, coaxial=coaxial)
    return reset, step, dict(obs=OBS_SIZE, act=ACT_SIZE, n_sub=n_sub,
                             max_steps=max_steps), aux
