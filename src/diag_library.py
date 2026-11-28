"""Get up policy versus the locomotion policy from fall library starts, grouped by how far
over the start is (D28 check). Separates a get up mode problem from a physical limit.
Run from the repo root: python src/diag_library.py --run s4_a --loco s3v9
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
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402

DT = 0.01
BUCKETS = [(-1.01, -0.5, "upside down"), (-0.5, 0.0, "past flat"),
           (0.0, 0.54, "flat to 57 deg"), (0.54, 0.89, "28 to 57 deg")]


def set_mode(s, v):
    return s._replace(mode=v) if hasattr(s, "_replace") else s.replace(mode=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--loco", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    sg = pickle.load(open(ROOT / "runs" / args.run / "ckpt.pkl", "rb"))
    sl = pickle.load(open(ROOT / "runs" / args.loco / "ckpt.pkl", "rb"))
    cfg = dict(CONFIGS[sg["cfg"].get("env", "s4")])
    cfg.update(sg["cfg"].get("env_overrides", {}))
    cfg.update(dr=False, getup_fall_frac=1.0, getup_demo_frac=0.0)
    reset, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    N = round(cfg["episode_s"] / DT)

    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(3), args.n))
    q0 = np.array(states.data.qpos)
    up0 = 1 - 2 * (q0[:, 4] ** 2 + q0[:, 5] ** 2)

    def make_roll(params, loco_mode):
        @jax.jit
        def roll(states):
            def one(s):
                if loco_mode:
                    s = set_mode(s, jnp.zeros(()))
                def body(s, _):
                    m, _, _ = net.apply(params, s.obs)
                    s = step(s, m)
                    return s, s.done
                return jax.lax.scan(body, s, None, length=N)[1]
            return jax.vmap(one)(states)
        return roll

    print(f"{args.n} fall library starts, nominal (compiles twice, a few minutes each)", flush=True)
    dg = np.array(make_roll(sg["params"], False)(states))
    dl = np.array(make_roll(sl["params"], True)(states))
    first_g = np.where(dg.any(1), dg.argmax(1), N)
    first_l = np.where(dl.any(1), dl.argmax(1), N)
    hand = first_g < N - 1                 # in get up mode a done before the timeout is a handoff
    stays = first_l >= 300                 # in locomotion mode a done is a fall

    print(f"\n{'start':16s} {'n':>5s} {args.run + ' handoff':>14s} {'median t':>9s} "
          f"{args.loco + ' stays up 3 s':>18s}")
    for lo, hi, label in BUCKETS:
        m = (up0 >= lo) & (up0 < hi)
        if not m.any():
            continue
        t = np.median(first_g[m & hand]) * DT if (m & hand).any() else float("nan")
        ctrl = f"{stays[m].mean() * 100:17.1f}%" if lo >= 0.54 else f"{'(falls at once)':>18s}"
        print(f"{label:16s} {m.sum():5d} {hand[m].mean() * 100:13.1f}% {t:8.2f}s {ctrl}")
    print(f"\nlocomotion mode ends an episode once tilt passes 57 deg, so its column only means "
          f"something for the 28 to 57 deg group")


if __name__ == "__main__":
    main()