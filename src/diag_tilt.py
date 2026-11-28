"""After a 0.3 rad backward tilt, where does the policy end up? Prints the pitch over
time, the final posture and what touches the floor. Also checks that the training
resets include tilted starts. Run from the repo root: python src/diag_tilt.py --run s3v6_a
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import argparse
import pickle
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS, LEG_Q  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402


def pitch_of(quat):
    return 2.0 * np.arcsin(np.clip(np.asarray(quat)[..., 2], -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    saved = pickle.load(open(ROOT / "runs" / args.run / "ckpt.pkl", "rb"))
    cfg = dict(CONFIGS[saved["cfg"].get("env", "s1")])
    cfg.update(saved["cfg"].get("env_overrides", {}))

    reset_tr, _, _, _ = make_env(cfg)
    st0 = jax.jit(jax.vmap(reset_tr))(jax.random.split(jax.random.PRNGKey(0), 2000))
    pt = pitch_of(st0.data.qpos[:, 3:7])
    print(f"training resets: {np.mean(np.abs(pt) > 0.05) * 100:.0f}% tilted, largest "
          f"{np.degrees(np.abs(pt).max()):.0f} deg (expect about 30% and 23 deg)")

    det = dict(cfg, push_every_s=None, resample_s=None, dr=False)
    _, step, sizes, aux = make_env(det)
    p, model = aux["p"], aux["model"]
    r_w, th = p["wheel"]["radius_m"], -0.3
    h_t = aux["h_nom"] - r_w
    qpos = (aux["qpos0"].at[0].set(h_t * np.sin(th))
            .at[2].set(r_w + h_t * np.cos(th) + p["sim"]["spawn_clearance_m"])
            .at[3:7].set(jnp.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0])))
    data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos, qvel=jnp.zeros(model.nv)))
    st = aux["init_from_data"](data, jnp.array([0.0, 0.0, aux["h_nom"]], dtype=jnp.float32))
    net, params = ActorCritic(sizes["act"]), saved["params"]

    @jax.jit
    def go(s):
        def body(s, _):
            mean, _, _ = net.apply(params, s.obs)
            s = step(s, mean)
            return s, (s.data.qpos[3:7], s.data.qpos[2], jnp.tanh(mean))
        return jax.lax.scan(body, s, None, length=500)

    st, (quat, z, act) = go(st)
    pitch, z, act = pitch_of(quat), np.array(z), np.array(act)
    print("pitch (deg, positive = forward) at " + "   ".join(
        f"{t:.1f} s: {np.degrees(pitch[int(t / 0.01) - 1]):+.1f}" for t in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0)))
    q = np.array(st.data.qpos[LEG_Q])
    print(f"final: torso height {z[-1]:.3f} m, hip_l {q[0]:+.2f} knee_l {q[1]:+.2f} "
          f"hip_r {q[2]:+.2f} knee_r {q[3]:+.2f} rad, leg actions {np.round(act[-50:, :4].mean(0), 2)}")

    cpu = mujoco.MjModel.from_xml_string(build_xml(p, mode="torque"))
    d = mujoco.MjData(cpu)
    d.qpos[:] = np.array(st.data.qpos)
    mujoco.mj_forward(cpu, d)
    floor = cpu.geom("floor").id
    touching = sorted({cpu.geom(d.contact[i].geom2 if d.contact[i].geom1 == floor
                                else d.contact[i].geom1).name
                       for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2)})
    print(f"touching the floor at the end: {touching}")


if __name__ == "__main__":
    main()