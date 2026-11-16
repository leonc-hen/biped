"""Phase 2 step 15 checks: MJX runs the legged model on the GPU (run from repo root).

1. The torque mode model loads into MJX at all (geom and feature support).
2. A batched rollout from the stand pose runs without NaN and keeps wheel contact.
3. Throughput: steps/second at several batch sizes -> informs num_envs and
   training time estimates. Writes results/mjx_throughput.csv.
"""
import csv
import sys
import time
from pathlib import Path
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.params import load_params  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402


def main():
    p = load_params()
    xml = build_xml(p, mode="torque")
    cpu_model = mujoco.MjModel.from_xml_string(xml)
    print(f"backend: {jax.default_backend()}   device: {jax.devices()[0]}")
    if jax.default_backend() != "gpu":
        print("FAIL: not on the GPU, stop here (see D10)")
        sys.exit(1)

    # 1 load into MJX
    try:
        model = mjx.put_model(cpu_model)
        print("1 MJX load           PASS")
    except Exception as e:
        print(f"1 MJX load           FAIL: {type(e).__name__}: {e}")
        print("  If this names the cylinder geom: set wheel.geom_type: sphere in")
        print("  params_legged.yaml, rebuild, rerun, and record the swap in decisions.md.")
        sys.exit(1)

    key_id = cpu_model.key("stand").id
    qpos0 = jnp.array(cpu_model.key_qpos[key_id])

    def init(rng):
        data = mjx.make_data(model)
        return data.replace(qpos=qpos0, qvel=jnp.zeros(model.nv))

    def make_batched_rollout(n_steps):
        def rollout(data):
            def body(d, _):
                return mjx.step(model, d), ()
            d, _ = jax.lax.scan(body, data, None, length=n_steps)
            return d
        return jax.jit(jax.vmap(rollout))

    # 2 batched sanity rollout, 0.5 s
    n_steps = int(0.5 / cpu_model.opt.timestep)
    roll_sanity = make_batched_rollout(n_steps)
    batch = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(0), 256))
    out = roll_sanity(batch)
    finite = bool(jnp.all(jnp.isfinite(out.qpos)))
    heights = np.array(out.qpos[:, 2])
    sane = finite and heights.min() > 0.0 and heights.max() < 0.5
    print(f"2 batched rollout    finite {finite}   height range "
          f"[{heights.min():.3f}, {heights.max():.3f}] m   "
          f"{'PASS' if sane else 'FAIL'}")

    # 3 throughput
    roll_warm = make_batched_rollout(50)
    roll_timed = make_batched_rollout(200)
    rows = []
    print("3 throughput (physics steps/s, after jit warmup)")
    for nb in (512, 1024, 2048, 4096):
        try:
            b = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(1), nb))
            roll_warm(b).qpos.block_until_ready()      # compile for this batch shape
            roll_timed(b).qpos.block_until_ready()     # compile the timed length too
            t0 = time.perf_counter()
            roll_timed(b).qpos.block_until_ready()
            dt = time.perf_counter() - t0
            sps = nb * 200 / dt
            rows.append([nb, round(sps)])
            print(f"   batch {nb:5d}   {sps:12,.0f} steps/s")
        except Exception as e:
            rows.append([nb, f"OOM or error: {type(e).__name__}"])
            print(f"   batch {nb:5d}   {type(e).__name__} (expected eventually on 8 GB)")
            break
    out_csv = ROOT / "results" / "mjx_throughput.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["batch", "steps_per_s"])
        w.writerows(rows)
    print(f"  wrote {out_csv.relative_to(ROOT)}")
    print("DONE" if sane else "SOME CHECKS FAILED")
    sys.exit(0 if sane else 1)


if __name__ == "__main__":
    main()