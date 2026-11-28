"""Is MJX enforcing the leg/body self collision? Replays a motion from results/getup_search.npz,
takes the frame where a leg goes deepest into the body, and compares the self collision
contacts CPU MuJoCo and MJX report at that frame.
Run from the repo root: python src/selfcol_check.py [--pose side]
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import argparse
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402

DT = 0.01
POSES = ["back", "front", "side"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="side", choices=POSES)
    args = ap.parse_args()
    cfg = dict(CONFIGS["s4"], dr=False)
    reset, step, sizes, aux = make_env(cfg)
    N = round(cfg["episode_s"] / DT)
    z = np.load(ROOT / "results" / "getup_search.npz")
    knots = jnp.asarray(z[args.pose], dtype=jnp.float32)
    K = knots.shape[0]
    t_knot, t_step = jnp.arange(K) * float(z["knot_s"]), jnp.arange(N) * DT
    cand = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(0), 64))
    pose = np.array(cand.dr["pose"])
    s0 = jax.tree.map(lambda x: x[int(np.where(pose == POSES.index(args.pose))[0][0])], cand)
    u = jax.vmap(lambda col: jnp.interp(t_step, t_knot, col), in_axes=1, out_axes=1)(
        jnp.clip(knots, -1, 1))
    acts = jnp.arctanh(jnp.clip(u, -0.995, 0.995))
    qpos = np.array(jax.jit(lambda s, a: jax.lax.scan(
        lambda s, a: (step(s, a),) * 2, s, a)[1].data.qpos)(s0, acts))

    cpu = mujoco.MjModel.from_xml_string(build_xml(aux["p"], mode="torque"))
    d = mujoco.MjData(cpu)
    name = lambda g: mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_GEOM, int(g))
    print(f"pairs in the model: {cpu.npair}")
    for i in range(cpu.npair):
        print(f"  {name(cpu.pair_geom1[i])} / {name(cpu.pair_geom2[i])}  "
              f"condim {cpu.pair_dim[i]}  solref {np.round(cpu.pair_solref[i], 4)}")

    legs = [n for n in (f"{p}_{s}_geom" for p in ("shank", "wheel") for s in "lr")]
    body = ["torso_geom", "shell_l_geom", "shell_r_geom", "shell_top_geom"]
    ids = {n: cpu.geom(n).id for n in legs + body if mujoco.mj_name2id(
        cpu, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
    selfset = {frozenset((ids[a], ids[b])) for a in legs for b in body if a in ids and b in ids}
    ft = np.zeros(6)
    worst, kw = 1.0, 0
    for k in range(N):
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(cpu, d)
        for pr in selfset:
            a, b = tuple(pr)
            dist = mujoco.mj_geomDistance(cpu, d, a, b, 0.05, ft)
            if dist < worst:
                worst, kw = dist, k
    print(f"\ndeepest leg/body overlap {worst * 1000:+.1f} mm at t = {(kw + 1) * DT:.2f} s")

    d.qpos[:] = qpos[kw]
    mujoco.mj_forward(cpu, d)
    rows = [(name(c.geom1), name(c.geom2), c.dist) for c in d.contact[:d.ncon]
            if frozenset((c.geom1, c.geom2)) in selfset]
    print(f"CPU MuJoCo self contacts at that frame: {len(rows)}")
    for a, b, dist in rows:
        print(f"  {a} / {b}  dist {dist * 1000:+.1f} mm")

    mx = mjx.put_model(cpu)
    dx = mjx.forward(mx, mjx.make_data(mx).replace(qpos=jnp.asarray(qpos[kw])))
    try:
        con = dx.contact
    except AttributeError:
        con = dx._impl.contact
    g, dist = np.array(con.geom), np.array(con.dist)
    rows = [(name(a), name(b), dd) for (a, b), dd in zip(g, dist) if frozenset((a, b)) in selfset]
    print(f"MJX self contact slots at that frame: {len(rows)} "
          f"({sum(dd < 0 for *_, dd in rows)} penetrating)")
    for a, b, dd in rows:
        if dd < 0.01:
            print(f"  {a} / {b}  dist {dd * 1000:+.1f} mm")


if __name__ == "__main__":
    main()