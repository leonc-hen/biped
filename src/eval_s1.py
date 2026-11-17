"""S1 evaluation harness: the trained policy against the Phase 1 baseline scenarios.
Run from the repo root: python src/eval_s1.py --run s1_v0

Known unfairness vs baseline_lqr.json, to be closed in later steps and rerun:
policy at 100 Hz vs LQR at 500; true state observations, no sensor noise or
estimator; no command delay; legged plant vs planar plant; env fall threshold
1.0 rad vs planar 1.3. Until then, R07 is the pass bar and the baseline table
is context, not a verdict.
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import json
import math
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import (make_env, EnvState, gravity_body, forward_velocity,  # noqa: E402
                        ACT_SIZE, LEG_Q, LEG_V)
from src.leg_actuator_jax import pd_torque  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402

DT = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    with open(ROOT / "runs" / args.run / "ckpt.pkl", "rb") as f:
        params = pickle.load(f)["params"]

    reset, step, sizes, aux = make_env()
    net = ActorCritic(sizes["act"])
    p, model = aux["p"], aux["model"]
    lm = aux["lm"]
    r_w = p["wheel"]["radius_m"]
    h_t = aux["h_nom"] - r_w

    def init_state(theta=0.0, cmd=(0.0, 0.0, None)):
        c = jnp.array([cmd[0], cmd[1], aux["h_nom"] if cmd[2] is None else cmd[2]])
        qpos = aux["qpos0"]
        qpos = qpos.at[0].set(h_t * math.sin(theta))
        qpos = qpos.at[2].set(r_w + h_t * math.cos(theta) + p["sim"]["spawn_clearance_m"])
        qpos = qpos.at[3:7].set(jnp.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]))
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos, qvel=jnp.zeros(model.nv)))
        a0 = jnp.zeros(ACT_SIZE)
        return EnvState(data, aux["get_obs"](data, a0, c), a0, c,
                        jnp.zeros(()), jnp.zeros(()), jnp.zeros((), dtype=jnp.int32))

    @partial(jax.jit, static_argnums=1)
    def rollout(state, n):
        def body(st, _):
            mean, _, _ = net.apply(params, st.obs)
            d = st.data
            tau_leg = pd_torque(d.qpos[LEG_Q], d.qvel[LEG_V],
                                aux["stand_leg"] + 0.5 * jnp.tanh(mean)[:4],
                                lm["kp_Nm_per_rad"], lm["kv_Nms_per_rad"],
                                lm["peak_torque_Nm"], lm["max_speed_rad_s"])
            st = step(st, mean)
            g = gravity_body(st.data.qpos[3:7])
            tilt = jnp.arccos(jnp.clip(-g[2], -1.0, 1.0))
            tau_w = aux["wheel_peak"] * jnp.tanh(mean)[4:6]
            power = (jnp.sum(jnp.abs(tau_leg * d.qvel[LEG_V]))
                     + jnp.sum(jnp.abs(tau_w * d.qvel[jnp.array([8, 11])])))
            return st, (tilt, forward_velocity(st.data), st.done, jnp.max(jnp.abs(tau_leg)),
                        jnp.max(jnp.abs(tau_w)), power)
        _, out = jax.lax.scan(body, state, None, length=n)
        return out

    def rollout_np(state, n):
        return [np.array(x) for x in rollout(state, n)]

    def until_fall(arrs):
        done = arrs[2]
        fell = bool((done > 0.5).any())
        n = int(np.argmax(done > 0.5)) + 1 if fell else len(done)
        return [a[:n] for a in arrs], fell

    N5 = int(5.0 / DT)
    results = {}

    # 0 standing equilibrium. The legged robot's resting torso tilt is not zero
    # (forward CoM offset), so recovery is judged relative to it (D12).
    out, st_fell = until_fall(rollout_np(init_state(), N5))
    tilt_s = out[0][int(1.0 / DT):]
    tilt_eq = float(tilt_s.mean()) if len(tilt_s) else math.nan   # nan if it fell within 1 s
    results["standing_fell"] = st_fell
    results["standing_tilt_mean_deg"] = round(math.degrees(tilt_eq), 3) if len(tilt_s) else None
    results["standing_tilt_std_deg"] = (round(math.degrees(float(tilt_s.std())), 3)
                                        if len(tilt_s) else None)

    def settled(tail):
        return (abs(tail.mean() - tilt_eq) < math.radians(2.0)
                and tail.std() < math.radians(1.0))

    # 1 tilt recovery sweep
    max_tilt, t_rec_03 = 0.0, None
    for th in np.arange(0.1, 1.01, 0.1):
        out, fell = until_fall(rollout_np(init_state(theta=float(th)), N5))
        tilt, _, _, tau_l, tau_w, _ = out
        okr = (not fell) and settled(tilt[-50:])
        if okr:
            max_tilt = float(th)
            if abs(th - 0.3) < 1e-6:
                below = np.where(tilt < math.radians(3))[0]
                t_rec_03 = float(below[0] * DT) if len(below) else None
                results["tau_peak_recover03_leg_wheel_Nm"] = [float(tau_l.max()), float(tau_w.max())]
        else:
            break
    results["max_recovered_tilt_rad"] = max_tilt
    results["recovery_time_from_0.3rad_s"] = t_rec_03

    # 2 push sweep: settle 1 s, then a sudden base velocity change
    @jax.jit
    def settle_1s(state):
        def body(st, _):
            mean, _, _ = net.apply(params, st.obs)
            return step(st, mean), ()
        st, _ = jax.lax.scan(body, state, None, length=int(1.0 / DT))
        return st

    max_push = 0.0
    for dv in np.arange(0.1, 1.51, 0.1):
        st = settle_1s(init_state())
        st = st._replace(data=st.data.replace(qvel=st.data.qvel.at[0].add(float(dv))),
                         done=jnp.zeros(()), steps=jnp.zeros((), dtype=jnp.int32))
        out, fell = until_fall(rollout_np(st, N5))
        tilt, vx, _, _, _, _ = out
        if (not fell) and settled(tilt[-50:]) and abs(vx[-50:].mean()) < 0.1:
            max_push = float(dv)
        else:
            break
    results["max_push_m_s"] = max_push

    # 3 velocity tracking
    for v_ref in (0.3, 1.0):
        out, fell = until_fall(rollout_np(init_state(cmd=(v_ref, 0.0, None)), int(8.0 / DT)))
        tilt, vx, _, tau_l, tau_w, power = out
        results[f"track_{v_ref}_fell"] = fell
        results[f"track_{v_ref}_mean_v"] = round(float(vx[-200:].mean()), 3)
        idx = None
        if not fell:
            band = np.abs(vx - v_ref) < 0.1 * v_ref
            idx = next((i for i in range(len(band)) if band[i:].all()), None)
        results[f"track_{v_ref}_settle_s"] = None if idx is None else round(idx * DT, 2)
        results[f"track_{v_ref}_overshoot_pct"] = round(
            max(0.0, (vx.max() - v_ref) / v_ref * 100), 1)
        if v_ref == 1.0 and not fell:
            steady = slice(int(4.0 / DT), None)
            dist = float(np.sum(vx[steady]) * DT)
            results["energy_J_per_m_at_1m_s"] = (
                round(float(np.sum(power[steady]) * DT) / dist, 2) if dist > 1.0 else None)
            results["tau_peak_drive1_leg_wheel_Nm"] = [float(tau_l.max()), float(tau_w.max())]

    out = ROOT / "results" / "eval_s1.json"
    with open(out, "w") as f:
        json.dump({"run": args.run, "metrics": results}, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))
    print("\nbaseline_lqr.json for comparison:")
    print(json.dumps(json.load(open(ROOT / "baseline_lqr.json")), indent=2))

    ok = (max_tilt >= 0.35 and max_push >= 0.4
          and results.get("track_1.0_settle_s") is not None
          and not st_fell and tilt_eq < math.radians(6.0))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("R07 PARITY PASS" if ok else "R07 PARITY FAIL (20 deg tilt, 0.4 m/s push, tracks 1.0 m/s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()