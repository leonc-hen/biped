"""Realism check for the open loop get up motions in results/getup_search.npz.
Per pose: time to handoff, time with no floor contact, peak body rate, fastest downward
torso speed, and the deepest overlap between a shank or wheel and the torso or shell
(the model has no self collision, so the search can pass legs through the body).
Run from the repo root: python src/getup_realism.py
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402

DT = 0.01
POSES = ["back", "front", "side"]


def main():
    cfg = dict(CONFIGS["s4"], dr=False)
    reset, step, sizes, aux = make_env(cfg)
    N = round(cfg["episode_s"] / DT)
    z = np.load(ROOT / "results" / "getup_search.npz")
    knot_s = float(z["knot_s"])
    K = z["back"].shape[0]
    t_knot = jnp.arange(K) * knot_s
    t_step = jnp.arange(N) * DT

    # same start states as getup_search.py (seed 0)
    cand = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(0), 64))
    pose = np.array(cand.dr["pose"])
    starts = [jax.tree.map(lambda x, i=int(np.where(pose == j)[0][0]): x[i], cand)
              for j in range(3)]

    def actions(knots):
        u = jax.vmap(lambda col: jnp.interp(t_step, t_knot, col), in_axes=1, out_axes=1)(
            jnp.clip(knots, -1.0, 1.0))
        return jnp.arctanh(jnp.clip(u, -0.995, 0.995))

    @jax.jit
    def trajectory(s0, knots):
        def body(s, a):
            s = step(s, a)
            return s, (s.data.qpos, s.data.qvel, s.done)
        return jax.lax.scan(body, s0, actions(knots))[1]

    cpu = mujoco.MjModel.from_xml_string(build_xml(aux["p"], mode="torque"))
    d = mujoco.MjData(cpu)
    floor = cpu.geom("floor").id

    def gid(n):
        return mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_GEOM, n)
    legs = [n for n in (f"{p}_{s}_geom" for p in ("shank", "wheel") for s in "lr") if gid(n) >= 0]
    body = [n for n in ("torso_geom", "shell_l_geom", "shell_r_geom", "shell_top_geom") if gid(n) >= 0]
    pairs = [(a, b) for a in legs for b in body]
    ft = np.zeros(6)

    def worst_overlap():
        best, which = 1.0, ""
        for a, b in pairs:
            dist = mujoco.mj_geomDistance(cpu, d, gid(a), gid(b), 0.05, ft)
            if dist < best:
                best, which = dist, f"{a} / {b}"
        return best, which

    print(f"checking {len(pairs)} leg/body geom pairs: {', '.join(legs)} against {', '.join(body)}")
    for key in ("stand", "crouch"):
        mujoco.mj_resetDataKeyframe(cpu, d, cpu.key(key).id)
        mujoco.mj_forward(cpu, d)
        dist, which = worst_overlap()
        print(f"reference keyframe {key:6s}: closest leg/body distance {dist * 1000:+6.1f} mm ({which})")

    print(f"\n{'pose':6s} {'handoff':>8s} {'airborne':>9s} {'peak rate':>10s} {'max down v':>11s} "
          f"{'worst overlap':>14s}  pair")
    for j, name in enumerate(POSES):
        qpos, qvel, done = (np.array(v) for v in
                            trajectory(starts[j], jnp.asarray(z[name], dtype=jnp.float32)))
        end = int(np.argmax(done > 0.5)) if (done > 0.5).any() else N - 1
        air, worst, which = 0, 1.0, ""
        for k in range(end + 1):
            d.qpos[:] = qpos[k]
            mujoco.mj_forward(cpu, d)
            if not any(floor in (d.contact[i].geom1, d.contact[i].geom2) for i in range(d.ncon)):
                air += 1
            dist, w = worst_overlap()
            if dist < worst:
                worst, which = dist, w
        rate = np.linalg.norm(qvel[:end + 1, 3:6], axis=1).max()
        vdown = qvel[:end + 1, 2].min()
        print(f"{name:6s} {(end + 1) * DT:7.2f}s {air * DT:8.2f}s {rate:8.1f}r/s {vdown:9.2f}m/s "
              f"{worst * 1000:+12.1f}mm  {which}")
    print("\nnegative overlap = the leg passes into the body by that much; positive = clearance.")
    print("airborne = time with nothing touching the floor; peak rate = body rotation speed.")


if __name__ == "__main__":
    main()