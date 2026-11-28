"""R08 get up evaluation. Episodes start lying on the back, front or side. The get up
policy runs until the handoff (tilt under 15 deg, torso above 0.18 m, body rates under
1 rad/s) or 8 s; then the locomotion policy takes over and must stay up for 3 s.
Success per start pose; nominal with --a_only, full randomization otherwise.
Run from the repo root: python src/eval_getup.py --run s4_c --loco s3v9 [--a_only]
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import json
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402
from src.eval_batch import wilson  # noqa: E402

DT = 0.01
POSES = ["back", "front", "side"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--loco", required=True)
    ap.add_argument("--a_only", action="store_true")
    args = ap.parse_args()
    sg = pickle.load(open(ROOT / "runs" / args.run / "ckpt.pkl", "rb"))
    sl = pickle.load(open(ROOT / "runs" / args.loco / "ckpt.pkl", "rb"))
    cfg = dict(CONFIGS[sg["cfg"].get("env", "s4")])
    cfg.update(sg["cfg"].get("env_overrides", {}))
    cfg.update(getup_handoff="switch", episode_s=11.5, getup_fall_frac=0.0, getup_demo_frac=0.0)
    cfg.update(dr=False) if args.a_only else cfg.update(dr=True, dr_scale=1.0, delay_max_ms=10.0)
    reset, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    n, N = 3000, round(11.0 / DT)

    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(77), n))
    pose = np.array(states.dr["pose"])

    @jax.jit
    def roll(states):
        def one(s):
            def body(s, _):
                mg, _, _ = net.apply(sg["params"], s.obs)
                ml, _, _ = net.apply(sl["params"], s.obs)
                s = step(s, jnp.where(s.mode > 0.5, mg, ml))
                return s, (s.mode, s.done)
            _, out = jax.lax.scan(body, s, None, length=N)
            return out
        return jax.vmap(one)(states)

    print(f"{n} get up episodes, {'nominal' if args.a_only else 'randomized'} "
          f"(the first run compiles for several minutes)", flush=True)
    mode, done = (np.array(x) for x in roll(states))
    ok = np.zeros(n, dtype=bool)
    t_up = np.full(n, np.nan)
    for i in range(n):
        sw = np.where(mode[i] < 0.5)[0]
        if len(sw) == 0:
            continue
        k = int(sw[0])
        t_up[i] = (k + 1) * DT
        if t_up[i] <= 8.0:
            ok[i] = not (done[i, k:k + 300] > 0.5).any()

    res = {"run": args.run, "loco": args.loco, "randomized": not args.a_only}
    checks = {}
    for j, name in enumerate(POSES):
        m = pose == j
        res[f"getup_{name}"] = wilson(int(ok[m].sum()), int(m.sum()))
        up = m & ~np.isnan(t_up)
        res[f"getup_{name}_reached_handoff"] = round(float(up.sum() / m.sum()), 3)
        res[f"getup_{name}_time_s_median"] = (round(float(np.median(t_up[m & ok])), 2)
                                              if (m & ok).any() else None)
        checks[f"R08 up from {name} within 8 s, stays up 3 s, at least 90 percent"] = \
            res[f"getup_{name}"]["rate"] >= 0.90
    gate_ok = all(checks.values())
    res["gate"] = {"get up": checks}
    res["gate_pass"] = gate_ok
    tag = "_Aonly" if args.a_only else ""
    out = ROOT / "results" / f"evalg_{args.run}{tag}.json"
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}\nGATE {'PASS' if gate_ok else 'FAIL'}")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()