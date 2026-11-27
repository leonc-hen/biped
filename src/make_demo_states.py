"""Demo start states for the get up curriculum (D29). Replays the open loop get up motions in
results/getup_search.npz from several start variants per pose and keeps the states along each
motion up to its best moment (most upright and high, or the handoff). Training starts sampled
from these put the policy partway along a known get up path.
Run from the repo root: python src/make_demo_states.py
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402

DT = 0.01
POSES = ["back", "front", "side"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--src", default="getup_search")
    args = ap.parse_args()
    cfg = dict(CONFIGS["s4"], dr=False, getup_fall_frac=0.0, getup_demo_frac=0.0)
    reset, step, sizes, aux = make_env(cfg)
    N = round(cfg["episode_s"] / DT)
    z = np.load(ROOT / "results" / f"{args.src}.npz")
    K = z["back"].shape[0]
    t_knot, t_step = jnp.arange(K) * float(z["knot_s"]), jnp.arange(N) * DT

    def actions(knots):
        u = jax.vmap(lambda col: jnp.interp(t_step, t_knot, col), in_axes=1, out_axes=1)(
            jnp.clip(knots, -1.0, 1.0))
        return jnp.arctanh(jnp.clip(u, -0.995, 0.995))

    def traj(s0, acts):
        def body(s, a):
            s = step(s, a)
            return s, (s.data.qpos, s.data.qvel, s.done)
        return jax.lax.scan(body, s0, acts)[1]

    trajs = jax.jit(jax.vmap(traj, in_axes=(0, None)))
    cand = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(11), 128))
    pose = np.array(cand.dr["pose"])
    Q, V, P = [], [], []
    for j, name in enumerate(POSES):
        ix = np.where(pose == j)[0][:args.variants]
        s0 = jax.tree.map(lambda x: x[ix], cand)
        qpos, qvel, done = (np.array(v) for v in
                            trajs(s0, actions(jnp.asarray(z[name], dtype=jnp.float32))))
        up = 1 - 2 * (qpos[..., 4] ** 2 + qpos[..., 5] ** 2)
        pq = np.clip(up, 0, 1) * np.clip((qpos[..., 2] - 0.08) / 0.07, 0, 1)
        kept, bests = 0, []
        for v in range(len(ix)):
            end = int(np.argmax(done[v] > 0.5)) if (done[v] > 0.5).any() else N
            k_best = int(np.argmax(pq[v, :max(end, 1)]))
            bests.append(pq[v, k_best])
            sel = np.arange(0, k_best + 1, args.every)
            Q.append(qpos[v, sel])
            V.append(qvel[v, sel])
            P.append(np.full(len(sel), j))
            kept += len(sel)
        print(f"{name:5s}: {len(ix)} variants, best pose reached: median {np.median(bests):.2f} "
              f"min {np.min(bests):.2f}, {kept} states kept", flush=True)
    out = ROOT / "results" / "getup_demo_states.npz"
    np.savez(out, qpos=np.concatenate(Q), qvel=np.concatenate(V), pose=np.concatenate(P))
    print(f"wrote {out.relative_to(ROOT)}: {sum(len(q) for q in Q)} states")


if __name__ == "__main__":
    main()