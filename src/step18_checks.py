"""Phase 2 step 18 checks: domain randomization plumbing (run from the repo root).

1. Sampled parameters stay inside the D16 ranges at scale 1.0.
2. apply_dr really changes the MJX model: the weight force scales with mass, and
   tilted gravity produces a horizontal force.
3. Command delay: the last substep applies the action from 0, 1 or 2 ticks ago
   depending on the delay.
4. Batched S2R rollout with random actions stays finite; throughput vs S2.
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS, ACT_SIZE, WHEEL_CTRL  # noqa: E402


def within(x, lo, hi):
    x = np.asarray(x)
    return float(x.min()) >= lo - 1e-6 and float(x.max()) <= hi + 1e-6


def main():
    ok = True
    reset, step, sizes, aux = make_env(CONFIGS["s2r"])

    # 1 ranges
    d = jax.vmap(aux["sample_dr"])(jax.random.split(jax.random.PRNGKey(0), 4096))
    max_tilt_deg = math.degrees(3.0 * math.sqrt(2)) * math.pi / 180 * 180 / math.pi
    checks = {
        "mass within 15 percent": within(d["mass_scale"], 0.85, 1.15),
        "inertia within 15 percent": within(d["inertia_scale"], 0.85, 1.15),
        "CoM x within 15 mm": within(d["com_offset"][:, 0], -0.015, 0.015),
        "friction 0.4 to 1.25": within(d["mu"], 0.4, 1.25),
        "motor kt within 10 percent": within(d["kt_leg"], 0.9, 1.1) and within(d["kt_w"], 0.9, 1.1),
        "PD gains within 10 percent": within(d["kp_s"], 0.9, 1.1) and within(d["kd_s"], 0.9, 1.1),
        "armature, damping, friction within 20 percent":
            all(within(d[k], 0.8, 1.2) for k in ("arm_scale", "damp_scale", "fric_scale")),
        "delay 0 to 20 ms, both ends sampled":
            within(d["delay_sub"], 0, 10) and int(np.max(d["delay_sub"])) == 10
            and int(np.min(d["delay_sub"])) == 0,
        "battery 18 to 25.2 V": within(np.asarray(d["v_scale"]) * 24.0, 18.0, 25.2),
        "slope at most 3 deg per axis":
            float(np.degrees(np.arccos(-np.asarray(d["g_unit"])[:, 2])).max()) <= 3.0 * math.sqrt(2) + 1e-3,
        "IMU mount tilt at most 3 deg per axis":
            float(np.degrees(np.arccos(np.asarray(d["mount"])[:, 2, 2])).max()) <= 3.0 * math.sqrt(2) + 1e-3,
    }
    for name, passed in checks.items():
        ok &= passed
        print(f"1 {'PASS' if passed else 'FAIL'}  {name}")

    # 2 the model actually changes
    nom = aux["nominal_dr"]

    def bias_force(dr):
        m = aux["apply_dr"](dr)
        dd = mjx.forward(m, mjx.make_data(m).replace(qpos=aux["qpos0"], qvel=jnp.zeros(m.nv)))
        return np.array(dd.qfrc_bias[0:3])

    f0 = bias_force(nom)
    f_heavy = bias_force(dict(nom, mass_scale=nom["mass_scale"] * 1.15))
    f_slope = bias_force(dict(nom, g_unit=jnp.array(
        [math.sin(math.radians(3.0)), 0.0, -math.cos(math.radians(3.0))])))
    ratio = f_heavy[2] / f0[2]
    good = abs(ratio - 1.15) < 0.01
    ok &= good
    print(f"2 {'PASS' if good else 'FAIL'}  weight force ratio with 15 percent more mass: {ratio:.4f}")
    good = abs(f_slope[0]) > 0.04 * abs(f0[2])
    ok &= good
    print(f"2 {'PASS' if good else 'FAIL'}  horizontal force on a 3 deg slope: "
          f"{abs(f_slope[0]) / abs(f0[2]) * 100:.1f} percent of weight (expect about 5.2)")

    # 3 delay: wheel command in force on the last substep
    reset2, step2, _, _ = make_env(CONFIGS["s2"])
    st = reset2(jax.random.PRNGKey(1))
    two_ago = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    one_ago = jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5])
    st = st._replace(act_hist=jnp.stack([one_ago, two_ago]))
    act = jnp.array([0.0, 0.0, 0.0, 0.0, math.atanh(0.25), math.atanh(0.25)])
    step_j = jax.jit(step2)
    peak = aux["wheel_peak"]
    for delay, expect in ((0, 0.25), (5, -0.5), (10, 0.5)):
        s = st._replace(dr=dict(st.dr, delay_sub=jnp.asarray(delay, dtype=jnp.int32)))
        got = float(step_j(s, act).data.ctrl[WHEEL_CTRL][0])
        good = abs(got - peak * expect) < 1e-4
        ok &= good
        print(f"3 {'PASS' if good else 'FAIL'}  delay {delay * 2:2d} ms: last substep wheel "
              f"torque {got:+.3f} N m, expect {peak * expect:+.3f}")

    # 4 batched rollout and throughput
    sps = {}
    for name in ("s2", "s2r"):
        r_, s_, _, _ = make_env(CONFIGS[name])
        nb = 2048
        states = jax.jit(jax.vmap(r_))(jax.random.split(jax.random.PRNGKey(4), nb))

        def make_burst(step_fn):
            @jax.jit
            def burst(states, rng):
                def body(carry, _):
                    st_, k = carry
                    k, ka = jax.random.split(k)
                    st_ = jax.vmap(step_fn)(st_, jax.random.normal(ka, (nb, ACT_SIZE)))
                    return (st_, k), ()
                (st_, _), _ = jax.lax.scan(body, (states, rng), None, length=100)
                return st_
            return burst

        burst = make_burst(s_)
        print(f"4 compiling the {name} rollout", flush=True)
        burst(states, jax.random.PRNGKey(5)).obs.block_until_ready()
        t0 = time.perf_counter()
        out = burst(states, jax.random.PRNGKey(6))
        out.obs.block_until_ready()
        sps[name] = nb * 100 / (time.perf_counter() - t0)
        if name == "s2r":
            good = bool(jnp.all(jnp.isfinite(out.obs)))
            ok &= good
            print(f"4 {'PASS' if good else 'FAIL'}  S2R random action rollout finite")
    print(f"4 throughput: S2 {sps['s2']:,.0f}  S2R {sps['s2r']:,.0f} policy steps/s "
          f"(randomization costs {(1 - sps['s2r'] / sps['s2']) * 100:.0f} percent)")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()