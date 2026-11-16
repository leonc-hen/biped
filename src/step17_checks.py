"""Phase 2 step 17 check: PPO actually learns on S1 (run from the repo root).

One short run (about 4M steps). Pass: the fall rate in the last logged updates
drops to under half of the first logged updates, and reward per step rises.
This is a smoke test, not the parity test; it answers 'is the trainer wired
correctly', nothing about policy quality.
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env  # noqa: E402
from src.ppo import train  # noqa: E402
from src.train_s1 import CFG  # noqa: E402


def main():
    cfg = dict(CFG, total_steps=4_000_000, num_envs=1024, log_every=2,
               ckpt_every=10_000, resume=False)
    reset_fn, step_fn, sizes, aux = make_env()
    _, hist = train(cfg, reset_fn, step_fn, sizes, ROOT / "runs" / "step17_smoke")

    head, tail = hist[:3], hist[-3:]
    r0 = sum(h[1] for h in head) / 3
    r1 = sum(h[1] for h in tail) / 3
    d0 = sum(h[2] for h in head) / 3
    d1 = sum(h[2] for h in tail) / 3
    print(f"\nreward/step {r0:+.3f} -> {r1:+.3f}   fall rate {d0:.4f} -> {d1:.4f}")
    ok = (d1 < 0.5 * d0 + 1e-5) and (r1 > r0)
    print("PASS: learning (fall rate collapsed, reward rose)" if ok
          else "FAIL: no clear learning signal")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()