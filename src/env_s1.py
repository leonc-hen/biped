"""Training environment for S1, S2, S2R and S3 (docs/interface.md v0.2).

make_env(cfg) returns (reset, step, sizes, aux). Pure functional:
reset(rng) -> EnvState, step(EnvState, action) -> EnvState. No auto reset.
S1 reproduces reward v0.3 (s1_v2). S2 adds height commands, resampling, pushes,
a leg action filter and a torso level reward (s2_v0 to s2_v3). S2R is S2 plus
the plan 2.7 domain randomization (D16) and the wheel torque speed envelope.
S3 is S2R plus one jump command per episode (R06).
With dr False every randomization path is skipped, so S1 and S2 are unchanged.
"""
import math
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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
TARGET_MARGIN = 0.05
W = dict(track_v=1.0, track_v_fine=1.0, track_yaw=0.5, upright=0.5,
         torque=1e-3, action_rate=1e-2, fall=5.0)

S1 = dict(name="s1", height_cmd=False, resample_s=None, push_every_s=None,
          push_dv=(0.0, 0.0), leg_filter_alpha=1.0,
          w_tilt=0.3, tilt_sigma2=0.006, w_height=0.3, height_sigma2=0.01,
          height_sigma2_fine=None, dr=False, dr_scale=0.0, wheel_envelope=False,
          cmd_fixed=None, reset_perturb=True, noise_scale=None, delay_fixed_ms=None,
          w_angvel=0.02, angvel_roll_pitch_only=False,
          w_track_v_sharp=0.0, track_v_sharp_sigma2=0.0025, push_max_norm=None,
          w_posture=0.0, w_hip_sym=0.0, w_stagger=0.0, delay_max_ms=20.0,
          jump=False, jump_window_s=1.5, w_jump=2.0, w_jump_vz=1.0, w_apex=100.0, w_impact=0.0,
          jump_action_scale=0.5,
          jump_primitive=False, jump_crouch_h=0.17, jump_land_h=0.22, jump_ext_knee=0.30,
          reset_tilt=0.0, w_tilt_broad=0.0, tilt_broad_sigma2=0.05,
          jump_push_hip_scale=0.0,
          getup=False, getup_action_scale=1.5, w_getup_success=20.0,
          getup_handoff="terminate", episode_s=10.0,
          handoff_tilt=0.26, handoff_height=0.18, handoff_rate=1.0,
          getup_fall_frac=0.0,
          getup_demo_frac=0.0, getup_demo_file="results/getup_demo_states.npz")
S2 = dict(S1, name="s2", height_cmd=True, resample_s=3.0, push_every_s=6.0,
          push_dv=(0.6, 0.4), leg_filter_alpha=0.5,
          w_tilt=0.5, tilt_sigma2=0.003, w_height=1.0, height_sigma2=0.0025,
          height_sigma2_fine=0.0004)
S2R = dict(S2, name="s2r", dr=True, dr_scale=1.0, wheel_envelope=True,
           w_angvel=0.5, angvel_roll_pitch_only=True, w_track_v_sharp=0.5,
           w_posture=0.3, w_hip_sym=0.0, w_stagger=400.0)
S3 = dict(S2R, name="s3", jump=True, w_impact=1e-3, jump_action_scale=0.5,
          jump_primitive=True, jump_ext_knee=1.25, reset_tilt=0.4,
          w_tilt_broad=0.5, jump_push_hip_scale=0.25)
S4 = dict(S2R, name="s4", getup=True, episode_s=8.0, push_every_s=None, resample_s=None,
          cmd_fixed=(0.0, 0.0, 0.25), reset_perturb=False,
          handoff_tilt=0.35, handoff_height=0.15, handoff_rate=2.0,
          getup_fall_frac=0.2, getup_demo_frac=0.5)
CONFIGS = dict(s1=S1, s2=S2, s2r=S2R, s3=S3, s4=S4)


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
    jump_at: jnp.ndarray         # policy step when the jump command starts, -1 for none
    jump_apex: jnp.ndarray       # highest wheel clearance in this window, m
    vz_prev: jnp.ndarray         # torso vertical speed last step, for the impact penalty
    mode: jnp.ndarray            # 1.0 in get up mode, 0.0 otherwise


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
    if cfg.get("physics_dt"):
        cpu_model.opt.timestep = cfg["physics_dt"]
    if cfg.get("solver_iterations"):
        cpu_model.opt.iterations, cpu_model.opt.ls_iterations = cfg["solver_iterations"]
    model = mjx.put_model(cpu_model)
    lm, wm = p["leg_motor"], p["wheel_motor"]
    coaxial = p.get("legs", {}).get("topology", "serial") == "coaxial_belt"

    def to_motor(v):
        """Joint order [hip_l, knee_l, hip_r, knee_r] to motor order. In the coaxial
        belt leg motor 2 turns the shank's absolute angle, hip + knee."""
        return v.at[1].add(v[0]).at[3].add(v[2]) if coaxial else v
    dt = p["sim"]["policy_dt_s"]
    phys_dt = float(cpu_model.opt.timestep)
    n_sub = round(dt / phys_dt)
    max_steps = round(cfg["episode_s"] / dt)
    g_mag = p["sim"]["gravity_mps2"]
    h_nom = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, h_nom)
    stand_leg = jnp.array([q1, q2, q1, q2])
    jr = p["joints"]
    j_lo = jnp.array([jr["hip"]["range_rad"][0], jr["knee"]["range_rad"][0]] * 2) + TARGET_MARGIN
    j_hi = jnp.array([jr["hip"]["range_rad"][1], jr["knee"]["range_rad"][1]] * 2) - TARGET_MARGIN
    h_lo, h_hi = p["pose"]["height_cmd_range_m"]
    r_w = p["wheel"]["radius_m"]
    l12 = p["thigh"]["length_m"] + p["shank"]["length_m"]
    l1, l2 = p["thigh"]["length_m"], p["shank"]["length_m"]

    def foot_x(qh, qk):
        """Wheel axle position ahead of the hip, in the torso frame. Positive hip and
        knee swing the foot backward, so forward is negative sine."""
        return -(l1 * jnp.sin(qh) + l2 * jnp.sin(qh + qk))
    qpos0 = jnp.array(cpu_model.key_qpos[cpu_model.key("stand").id])
    wheel_peak = wm["peak_torque_Nm"]
    torso = cpu_model.body("torso").id
    wheel_ids = jnp.array([cpu_model.body("wheel_l").id, cpu_model.body("wheel_r").id])

    def settled_pose(name):
        """A lying keyframe placed on the floor and settled limp until at rest (CPU MuJoCo)."""
        d = mujoco.MjData(cpu_model)
        mujoco.mj_resetDataKeyframe(cpu_model, d, cpu_model.key(name).id)
        d.qvel[:], d.ctrl[:] = 0.0, 0.0
        floor = cpu_model.geom("floor").id
        d.qpos[2] = 0.5
        mujoco.mj_forward(cpu_model, d)
        while d.qpos[2] > -0.5 and not any(
                floor in (d.contact[i].geom1, d.contact[i].geom2) for i in range(d.ncon)):
            d.qpos[2] -= 0.002
            mujoco.mj_forward(cpu_model, d)
        d.qpos[2] += 0.003
        mujoco.mj_forward(cpu_model, d)
        q = d.qpos[3:7]
        up_placed = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        t, dt_p = 0.0, cpu_model.opt.timestep
        while t < 3.0:
            mujoco.mj_step(cpu_model, d)
            t += dt_p
            if t > 0.3 and (d.qvel[:6] ** 2).sum() ** 0.5 < 0.02:
                break
        q = d.qpos[3:7]
        up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        touching = sorted({mujoco.mj_id2name(cpu_model, mujoco.mjtObj.mjOBJ_GEOM,
                                             d.contact[i].geom1 + d.contact[i].geom2 - floor)
                           for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2)})
        print(f"get up start {name}: upright {up_placed:+.2f} as placed, {up:+.2f} at rest "
              f"after {t:.2f} s, torso z {d.qpos[2]:.3f} m, touching the floor: {', '.join(touching)}",
              flush=True)
        return d.qpos.copy()

    getup_starts = (jnp.array([settled_pose(k) for k in ("lying_back", "lying_front", "lying_side")])
                    if cfg["getup"] else None)

    def fall_library(n_falls=80, every_s=0.04, seed=0):
        """States along limp falls from standing, from slightly tipped to lying (D28). Start
        states for the get up curriculum: success comes early from the easy ones and the
        policy learns backward toward the rest poses. Training only; eval uses rest poses."""
        rs = np.random.default_rng(seed)
        floor = cpu_model.geom("floor").id
        leg_q = np.asarray(LEG_Q)
        qs, vs = [], []
        for _ in range(n_falls):
            d = mujoco.MjData(cpu_model)
            mujoco.mj_resetDataKeyframe(cpu_model, d, cpu_model.key("stand").id)
            d.qpos[leg_q] += rs.uniform(-0.4, 0.4, 4)
            d.qvel[:], d.ctrl[:] = 0.0, 0.0
            d.qpos[2] = 0.6
            mujoco.mj_forward(cpu_model, d)
            while d.qpos[2] > -0.5 and not any(
                    floor in (d.contact[i].geom1, d.contact[i].geom2) for i in range(d.ncon)):
                d.qpos[2] -= 0.002
                mujoco.mj_forward(cpu_model, d)
            d.qpos[2] += 0.003
            ang = rs.uniform(0.0, 2 * np.pi)
            d.qvel[0:2] = rs.uniform(0.3, 1.0) * np.array([np.cos(ang), np.sin(ang)])
            d.qvel[3:6] = rs.uniform(-2.0, 2.0, 3)
            t, rec = 0.0, 0.0
            while t < 3.0:
                mujoco.mj_step(cpu_model, d)
                t += cpu_model.opt.timestep
                if t >= rec:
                    rec += every_s
                    q = d.qpos[3:7]
                    if 1 - 2 * (q[1] ** 2 + q[2] ** 2) < 0.88 or d.qpos[2] < 0.12:
                        qs.append(d.qpos.copy())
                        vs.append(d.qvel.copy())
                if t > 0.5 and (d.qvel[:6] ** 2).sum() ** 0.5 < 0.02:
                    break
        print(f"fall library: {len(qs)} start states from {n_falls} limp falls", flush=True)
        return jnp.array(np.array(qs)), jnp.array(np.array(vs))

    lib_q, lib_v = (fall_library() if cfg["getup"] and cfg["getup_fall_frac"] > 0
                    else (None, None))
    demo_q = demo_v = None
    if cfg["getup"] and cfg["getup_demo_frac"] > 0:
        dz = np.load(ROOT / cfg["getup_demo_file"])
        demo_q, demo_v = jnp.array(dz["qpos"]), jnp.array(dz["qvel"])
        print(f"demo start states: {demo_q.shape[0]}", flush=True)
    nbody, nv = cpu_model.nbody, cpu_model.nv
    max_delay_sub = round(cfg["delay_max_ms"] / 1000 / phys_dt)
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

    k_ext = cfg["jump_ext_knee"]
    EXT = jnp.array([-k_ext / 2, k_ext, -k_ext / 2, k_ext])   # push target, wheel under the hip

    def jump_base(t_s, h_cmd):
        """Scripted jump in joint space: crouch, push, land, return to the stance."""
        stand, crouch, land = ik_jnp(h_cmd), ik_jnp(cfg["jump_crouch_h"]), ik_jnp(cfg["jump_land_h"])
        a = jnp.clip(t_s / 0.4, 0.0, 1.0)
        b = jnp.clip((t_s - 0.85) / 0.4, 0.0, 1.0)
        pre = (1.0 - a) * stand + a * crouch
        post = (1.0 - b) * land + b * stand
        return jnp.where(t_s < 0.4, pre, jnp.where(t_s < 0.55, EXT, post))

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

    def get_obs(data, prev_action, cmd, dr, rng, flags=None):
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
        raw = jnp.concatenate([g_b, gyro, q, qd, wd, prev_action, cmd,
                               jnp.zeros(2) if flags is None else flags])
        return raw * scale

    def init_from_data(data, cmd):
        """A full EnvState from given physics data. For evaluators."""
        a0, z = jnp.zeros(ACT_SIZE), jnp.zeros(())
        key = jax.random.PRNGKey(0)
        dr0 = sample_dr(key) if cfg["dr"] else nominal_dr
        return EnvState(data, get_obs(data, a0, cmd, dr0, key), a0, cmd, z, z,
                        jnp.zeros((), dtype=jnp.int32), key, dr0,
                        jnp.zeros((2, ACT_SIZE)), jnp.asarray(-1, dtype=jnp.int32),
                        jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))

    def reset(rng):
        rng, rc, rv, rd, ro, rj, rt = jax.random.split(rng, 7)
        cmd = sample_cmd(rc)
        dr = sample_dr(rd) if cfg["dr"] else nominal_dr
        m = apply_dr(dr) if cfg["dr"] else model
        if cfg["reset_perturb"]:
            dv = jax.random.uniform(rv, (2,), minval=-1.0, maxval=1.0) * jnp.array([0.5, 1.0])
            qvel0 = jnp.zeros(model.nv).at[0].set(dv[0]).at[4].set(dv[1])
        else:
            qvel0 = jnp.zeros(model.nv)
        qpos_r = qpos0
        if cfg["reset_tilt"]:
            k1, k2 = jax.random.split(rt)
            th = jnp.where(jax.random.uniform(k1, ()) < 0.3,
                           jax.random.uniform(k2, (), minval=-cfg["reset_tilt"],
                                              maxval=cfg["reset_tilt"]), 0.0)
            h_t = qpos0[2] - r_w            # tilt the whole robot about the wheel axle
            qpos_r = (qpos0.at[0].set(h_t * jnp.sin(th))
                      .at[2].set(r_w + h_t * jnp.cos(th))
                      .at[3:7].set(jnp.array([jnp.cos(th / 2), 0.0, jnp.sin(th / 2), 0.0])))
        if cfg["getup"]:
            k1, k2, k3, k4, k5 = jax.random.split(rt, 5)
            pose = jax.random.randint(k1, (), 0, 3)           # 0 back, 1 front, 2 side
            qpos_r = getup_starts[pose].at[LEG_Q].add(
                jax.random.uniform(k2, (4,), minval=-0.2, maxval=0.2))
            qvel0 = jnp.zeros(model.nv)
            u = jax.random.uniform(k3, ())
            f_demo, f_fall = cfg["getup_demo_frac"], cfg["getup_fall_frac"]
            if lib_q is not None:
                use = (u >= f_demo) & (u < f_demo + f_fall)
                li = jax.random.randint(k4, (), 0, lib_q.shape[0])
                qpos_r = jnp.where(use, lib_q[li], qpos_r)
                qvel0 = jnp.where(use, lib_v[li], qvel0)
                pose = jnp.where(use, 3, pose)                 # 3 = fall library
            if demo_q is not None:
                use = u < f_demo
                di = jax.random.randint(k5, (), 0, demo_q.shape[0])
                qpos_r = jnp.where(use, demo_q[di], qpos_r)
                qvel0 = jnp.where(use, demo_v[di], qvel0)
                pose = jnp.where(use, 4, pose)                 # 4 = demo state
            dr = dict(dr, pose=pose)
        data = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos_r, qvel=qvel0))
        a0, z = jnp.zeros(ACT_SIZE), jnp.zeros(())
        jump_at = (jax.random.randint(rj, (), round(2.0 / dt), round(6.0 / dt)) if cfg["jump"]
                   else jnp.asarray(-1)).astype(jnp.int32)
        flags0 = jnp.array([0.0, 1.0 if cfg["getup"] else 0.0])
        return EnvState(data, get_obs(data, a0, cmd, dr, ro, flags0), a0, cmd, z, z,
                        jnp.zeros((), dtype=jnp.int32), rng, dr, jnp.zeros((2, ACT_SIZE)),
                        jump_at, jnp.zeros(()), jnp.zeros(()),
                        jnp.asarray(1.0 if cfg["getup"] else 0.0))

    def step(state, action):
        rng, dr = state.rng, state.dr
        steps = state.steps + 1
        win = round(cfg["jump_window_s"] / dt)
        in_win = (state.jump_at >= 0) & (steps >= state.jump_at) & (steps < state.jump_at + win)
        win_end = (state.jump_at >= 0) & (steps == state.jump_at + win)
        fw = in_win.astype(jnp.float32)
        leg_scale = 0.5 + (cfg["jump_action_scale"] - 0.5) * fw
        m = apply_dr(dr) if cfg["dr"] else model
        a_raw = jnp.tanh(action)
        al = cfg["leg_filter_alpha"]
        a = a_raw.at[:4].set(al * a_raw[:4] + (1.0 - al) * state.prev_action[:4])
        hist = jnp.stack([a, state.act_hist[0], state.act_hist[1]])   # now, 1 and 2 ticks ago
        base = ik_jnp(state.cmd[2])
        if cfg["jump_primitive"]:
            t_j = (steps - state.jump_at).astype(jnp.float32) * dt
            base = jnp.where(in_win, jump_base(t_j, state.cmd[2]), base)
            push_phase = in_win & (t_j >= 0.25) & (t_j < 0.55)
            hp = cfg["jump_push_hip_scale"]
            leg_scale = jnp.where(push_phase, jnp.array([hp, 0.0, hp, 0.0]), leg_scale)
        kp = lm["kp_Nm_per_rad"] * dr["kp_s"]
        kd = lm["kv_Nms_per_rad"] * dr["kd_s"]
        leg_speed = lm["max_speed_rad_s"] * dr["v_scale"]
        wheel_speed = wm["max_speed_rad_s"] * dr["v_scale"]

        data = state.data
        if cfg["push_every_s"]:
            rng, kp_, kd_ = jax.random.split(rng, 3)
            hit = (jax.random.uniform(kp_, ()) < dt / cfg["push_every_s"]) & ~in_win
            if cfg["push_max_norm"] is None:
                dvxy = (jax.random.uniform(kd_, (2,), minval=-1.0, maxval=1.0)
                        * jnp.array(cfg["push_dv"]))
            else:
                ka, km = jax.random.split(kd_)
                ang = jax.random.uniform(ka, (), minval=0.0, maxval=2 * math.pi)
                mag = jax.random.uniform(km, (), minval=0.0, maxval=cfg["push_max_norm"])
                dvxy = mag * jnp.array([jnp.cos(ang), jnp.sin(ang)])
            data = data.replace(qvel=data.qvel.at[0:2].add(jnp.where(hit, dvxy, 0.0)))

        leg_scale = jnp.where(state.mode > 0.5, cfg["getup_action_scale"], leg_scale)

        def sub(d, j):
            # substep j: the action in force was issued 0, 1 or 2 ticks ago
            a_eff = hist[jnp.clip((dr["delay_sub"] - j + n_sub - 1) // n_sub, 0, 2)]
            tgt_j = jnp.clip(base + leg_scale * a_eff[:4], j_lo, j_hi)
            tau_leg = pd_torque(to_motor(d.qpos[LEG_Q]), to_motor(d.qvel[LEG_V]), to_motor(tgt_j),
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
        tilt_broad = jnp.exp(-jnp.square(tilt_ang) / cfg["tilt_broad_sigma2"])
        e_h = height - state.cmd[2]
        height_r = jnp.exp(-jnp.square(e_h) / cfg["height_sigma2"])
        if cfg["height_sigma2_fine"]:
            height_r = (0.5 * height_r
                        + 0.5 * jnp.exp(-jnp.square(e_h) / cfg["height_sigma2_fine"]))
        cost_tau = jnp.mean(jnp.sum(jnp.square(ctrl_hist), axis=-1))
        cost_rate = jnp.sum(jnp.square(a - state.prev_action))
        ang = data.qvel[3:5] if cfg["angvel_roll_pitch_only"] else data.qvel[3:6]
        cost_angvel = jnp.sum(jnp.square(ang))
        q_j = data.qpos[LEG_Q]
        cost_posture = jnp.sum(jnp.square(q_j - ik_jnp(state.cmd[2])))
        cost_hip_sym = jnp.square(q_j[0] - q_j[2])
        cost_stagger = jnp.square(foot_x(q_j[0], q_j[1]) - foot_x(q_j[2], q_j[3]))
        clear = jnp.maximum(jnp.min(data.xpos[wheel_ids, 2]) - r_w, 0.0)
        apex = jnp.where(in_win, jnp.maximum(state.jump_apex, clear), state.jump_apex)
        vz = data.qvel[2]
        az = (vz - state.vz_prev) / dt
        track_v_sharp = jnp.exp(-jnp.square(err_v) / cfg["track_v_sharp_sigma2"])

        mode1 = state.mode > 0.5
        handoff = mode1 & (tilt_ang < cfg["handoff_tilt"]) & (height > cfg["handoff_height"]) & \
            (jnp.linalg.norm(data.qvel[3:6]) < cfg["handoff_rate"])
        fell = ((upright < math.cos(1.0)) | (height < 0.08)) & ~mode1
        end_on_handoff = handoff if cfg["getup_handoff"] == "terminate" else False
        done = (fell | (steps >= max_steps) | end_on_handoff).astype(jnp.float32)
        new_mode = (jnp.where(handoff, 0.0, state.mode) if cfg["getup_handoff"] == "switch"
                    else state.mode)
        reward = (W["track_v"] * track_v + W["track_v_fine"] * track_v_fine
                  + W["track_yaw"] * track_yaw + W["upright"] * upright
                  + cfg["w_tilt"] * tilt_r + cfg["w_height"] * height_r * (1.0 - fw)
                  + cfg["w_tilt_broad"] * tilt_broad
                  - W["torque"] * cost_tau - W["action_rate"] * cost_rate
                  - cfg["w_angvel"] * cost_angvel
                  + cfg["w_track_v_sharp"] * track_v_sharp
                  - cfg["w_posture"] * cost_posture * (1.0 - fw) - cfg["w_hip_sym"] * cost_hip_sym
                  - cfg["w_stagger"] * cost_stagger
                  + cfg["w_jump"] * fw * jnp.minimum(clear, 0.12) / 0.06
                  + cfg["w_jump_vz"] * fw * jnp.clip(vz, 0.0, 2.5)
                  + cfg["w_apex"] * win_end.astype(jnp.float32) * jnp.clip(apex / 0.06, 0.0, 2.0)
                  - cfg["w_impact"] * (state.vz_prev < -0.2).astype(jnp.float32)
                    * jnp.square(jnp.maximum(jnp.abs(az) - 30.0, 0.0))
                  - W["fall"] * fell.astype(jnp.float32))
        if cfg["getup"]:
            up01 = (upright + 1.0) / 2.0
            h_prog = jnp.clip((height - 0.08) / (state.cmd[2] - 0.08), 0.0, 1.0)
            r_getup = (-0.5 - (1.0 - up01) - (1.0 - h_prog)
                       + cfg["w_getup_success"] * handoff.astype(jnp.float32)
                       - W["torque"] * cost_tau - W["action_rate"] * cost_rate)
            reward = jnp.where(mode1, r_getup, reward)

        cmd = state.cmd
        if cfg["resample_s"]:
            rng, kc = jax.random.split(rng)
            every = round(cfg["resample_s"] / dt)
            cmd = jnp.where(steps % every == 0, sample_cmd(kc), cmd)

        if cfg["dr"]:
            rng, ko = jax.random.split(rng)
        else:
            ko = rng
        return EnvState(data, get_obs(data, a, cmd, dr, ko, jnp.array([fw, new_mode])), a, cmd,
                        reward, done, steps,
                        rng, dr, jnp.stack([a, state.act_hist[0]]), state.jump_at, apex, vz,
                        new_mode)

    aux = dict(p=p, cfg=cfg, model=model, qpos0=qpos0, stand_leg=stand_leg, h_nom=h_nom,
               h_range=(h_lo, h_hi), lm=lm, wheel_peak=wheel_peak, ik=ik_jnp,
               nominal_dr=nominal_dr, sample_dr=sample_dr, apply_dr=apply_dr,
               init_from_data=init_from_data, to_motor=to_motor, coaxial=coaxial)
    return reset, step, dict(obs=OBS_SIZE, act=ACT_SIZE, n_sub=n_sub,
                             max_steps=max_steps), aux
