"""Train S1 (balance and drive at fixed height). Run from the repo root:
python src/train_s1.py --name s1_v0 --steps 100000000 [--resume]
Monitor with: tensorboard --logdir runs
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env  # noqa: E402
from src.ppo import train  # noqa: E402

CFG = dict(num_envs=2048, unroll=32, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
           epochs=4, minibatches=8, vf_coef=0.5, ent_coef=1e-3, seed=0,
           log_every=5, ckpt_every=50, resume=False, total_steps=100_000_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=CFG["total_steps"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(CFG, total_steps=args.steps, resume=args.resume, seed=args.seed)
    reset_fn, step_fn, sizes, aux = make_env()
    train(cfg, reset_fn, step_fn, sizes, ROOT / "runs" / args.name)


if __name__ == "__main__":
    main()