"""Phase 2 step 16 checks: the S1 environment (run from the repo root).

1. Observation size matches docs/interface.md v0.1.
2. JAX PD matches the numpy PD on random inputs (dual implementation parity).
3. Batched rollout with random actions: finite obs and rewards throughout.
4. Zero action from stand: the robot falls and done triggers within 2 s.
5. Environment throughput at batch 2048 -> the real policy steps/s for planning.
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import leg_actuator as pd_np  # noqa: E402
from src import leg_actuator_jax as pd_jx  # noqa: E402
from src.env_s1 import make_env, OBS_SIZE  # noqa: E402


def main():
    ok = True
    reset, step, sizes, aux = make_env()

    # 1 observation size
    s = reset(jax.random.PRNGKey(0))
    good = s.obs.shape == (OBS_SIZE,) and sizes["obs"] == OBS_SIZE
    ok &= good
    print(f"1 obs size           {s.obs.shape[0]} (interface v0.1: {OBS_SIZE})   "
          f"{'PASS' if good else 'FAIL'}")

    # 2 PD parity numpy vs jax
    rng = np.random.default_rng(0)
    q, v, t = (rng.uniform(-2, 2, 64) for _ in range(3))
    a_np = pd_np.pd_torque(q, v, t, 15.0, 0.4, 3.0, 20.0, 0.9)
    a_jx = np.array(pd_jx.pd_torque(jnp.array(q), jnp.array(v), jnp.array(t),
                                    15.0, 0.4, 3.0, 20.0, 0.9))
    err = np.max(np.abs(a_np - a_jx))
    good = err < 1e-5
    ok &= good
    print(f"2 PD parity          max difference {err:.2e} N m   {'PASS' if good else 'FAIL'}")

    # 3 batched random rollout, 200 policy steps
    nb, n_steps = 256, 200
    states = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(1), nb))

    @jax.jit
    def rollout(states, rng):
        def body(carry, _):
            st, r = carry
            r, k = jax.random.split(r)
            acts = jax.random.normal(k, (nb, sizes["act"]))
            st = jax.vmap(step)(st, acts)
            return (st, r), (st.obs, st.reward)
        (st, _), (obs, rew) = jax.lax.scan(body, (states, rng), None, length=n_steps)
        return st, obs, rew

    _, obs, rew = rollout(states, jax.random.PRNGKey(2))
    good = bool(jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.isfinite(rew)))
    ok &= good
    print(f"3 random rollout     finite obs {bool(jnp.all(jnp.isfinite(obs)))}   "
          f"finite reward {bool(jnp.all(jnp.isfinite(rew)))}   "
          f"reward range [{float(rew.min()):.2f}, {float(rew.max()):.2f}]   "
          f"{'PASS' if good else 'FAIL'}")

    # 4 zero action: must fall and signal done within 2 s
    st = reset(jax.random.PRNGKey(3))
    fell_at = -1.0

    @jax.jit
    def one(st):
        return step(st, jnp.zeros(sizes["act"]))

    for i in range(int(2.0 / 0.01)):
        st = one(st)
        if float(st.done) > 0.5:
            fell_at = (i + 1) * 0.01
            break
    good = 0.0 < fell_at <= 2.0
    ok &= good
    print(f"4 zero action falls  done at {fell_at:.2f} s   {'PASS' if good else 'FAIL'}")

    # 5 throughput at batch 2048
    nb2 = 2048
    states2 = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(4), nb2))

    @jax.jit
    def burst(states, rng):
        def body(carry, _):
            st, r = carry
            r, k = jax.random.split(r)
            st = jax.vmap(step)(st, jax.random.normal(k, (nb2, sizes["act"])))
            return (st, r), ()
        (st, _), _ = jax.lax.scan(body, (states, rng), None, length=100)
        return st

    burst(states2, jax.random.PRNGKey(5)).obs.block_until_ready()   # compile
    t0 = time.perf_counter()
    burst(states2, jax.random.PRNGKey(6)).obs.block_until_ready()
    dt = time.perf_counter() - t0
    pps = nb2 * 100 / dt
    print(f"5 env throughput     {pps:,.0f} policy steps/s at batch {nb2}   "
          f"(= {pps * sizes['n_sub']:,.0f} physics steps/s)")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()