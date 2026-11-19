"""Policy evaluation harness for S1 and S2 (run from the repo root):
python src/eval_policy.py --run s2_v0

Reads the env name from the checkpoint and evaluates on a deterministic copy of
that env (no random pushes, no command resampling). Recovery is judged relative
to the measured standing equilibrium (D12). Remaining unfairness vs
baseline_lqr.json: 100 Hz vs 500 Hz, true state observations, no delay, legged
vs planar plant, 1.0 vs 1.3 rad fall threshold.
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
from src.env_s1 import (make_env, CONFIGS, EnvState, gravity_body, forward_velocity,  # noqa: E402
                        ACT_SIZE, LEG_CTRL, WHEEL_CTRL, LEG_V, WHEEL_V)
from src.ppo import ActorCritic  # noqa: E402

DT = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    with open(ROOT / "runs" / args.run / "ckpt.pkl", "rb") as f:
        saved = pickle.load(f)
    params = saved["params"]
    env_name = saved["cfg"].get("env", "s1")
    cfg = dict(CONFIGS[env_name], push_every_s=None, resample_s=None)
    reset, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    p, model = aux["p"], aux["model"]
    r_w = p["wheel"]["radius_m"]
    h_t = aux["h_nom"] - r_w

    def init_state(theta=0.0, cmd=(0.0, 0.0, None)):
        h = aux["h_nom"] if cmd[2] is None else cmd[2]
        c = jnp.array([cmd[0], cmd[1], h])
        qpos = aux["qpos0"]
        qpos = qpos.at[0].set(h_t * math.sin(theta))
        qpos = qpos.at[2].set(r_w + h_t * math.cos(theta) + p["sim"]["spawn_clearance_m"])
        qpos = qpos.at[3:7].set(jnp.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]))
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos, qvel=jnp.zeros(model.nv)))
        a0, z = jnp.zeros(ACT_SIZE), jnp.zeros(())
        return EnvState(data, aux["get_obs"](data, a0, c), a0, c, z, z,
                        jnp.zeros((), dtype=jnp.int32), jax.random.PRNGKey(0))

    @partial(jax.jit, static_argnums=1)
    def rollout(state, n):
        def body(st, _):
            mean, _, _ = net.apply(params, st.obs)
            st = step(st, mean)
            d = st.data
            g = gravity_body(d.qpos[3:7])
            tilt = jnp.arccos(jnp.clip(-g[2], -1.0, 1.0))
            tau_l, tau_w = d.ctrl[LEG_CTRL], d.ctrl[WHEEL_CTRL]
            power = (jnp.sum(jnp.abs(tau_l * d.qvel[LEG_V]))
                     + jnp.sum(jnp.abs(tau_w * d.qvel[WHEEL_V])))
            return st, (tilt, forward_velocity(d), st.done, jnp.max(jnp.abs(tau_l)),
                        jnp.max(jnp.abs(tau_w)), power, d.qpos[2])
        _, out = jax.lax.scan(body, state, None, length=n)
        return out

    def rollout_np(state, n):
        return [np.array(x) for x in rollout(state, n)]

    def until_fall(arrs):
        done = arrs[2]
        fell = bool((done > 0.5).any())
        n = int(np.argmax(done > 0.5)) + 1 if fell else len(done)
        return [a[:n] for a in arrs], fell

    @jax.jit
    def settle_1s(state):
        def body(st, _):
            mean, _, _ = net.apply(params, st.obs)
            return step(st, mean), ()
        st, _ = jax.lax.scan(body, state, None, length=int(1.0 / DT))
        return st

    N5 = int(5.0 / DT)
    res = {"env": env_name}
    out_path = ROOT / "results" / f"eval_{env_name}_{args.run}.json"

    def finish(ok):
        with open(out_path, "w") as f:
            json.dump({"run": args.run, "metrics": res}, f, indent=2, default=float)
        print(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {out_path.relative_to(ROOT)}")
        sys.exit(0 if ok else 1)

    # 0 standing equilibrium (D12)
    (tilt_s, *_), st_fell = until_fall(rollout_np(init_state(), N5))
    tilt_s = tilt_s[int(1.0 / DT):]
    res["standing_fell"] = st_fell
    if st_fell or len(tilt_s) < 50:
        print("FAIL: fell while standing, nothing else is meaningful")
        finish(False)
    tilt_eq = float(tilt_s.mean())
    res["standing_tilt_mean_deg"] = round(math.degrees(tilt_eq), 3)
    res["standing_tilt_std_deg"] = round(math.degrees(float(tilt_s.std())), 3)

    def settled(tail):
        return (abs(tail.mean() - tilt_eq) < math.radians(2.0)
                and tail.std() < math.radians(1.0))

    # 1 tilt recovery sweep
    max_tilt, t_rec = 0.0, None
    for th in np.arange(0.1, 1.01, 0.1):
        (tilt, _, _, tau_l, tau_w, _, _), fell = until_fall(
            rollout_np(init_state(theta=float(th)), N5))
        if fell or not settled(tilt[-50:]):
            break
        max_tilt = float(th)
        if abs(th - 0.3) < 1e-6:
            near = np.where(np.abs(tilt - tilt_eq) < math.radians(2.0))[0]
            t_rec = float(near[0] * DT) if len(near) else None
            res["tau_peak_recover03_leg_wheel_Nm"] = [float(tau_l.max()), float(tau_w.max())]
    res["max_recovered_tilt_rad"] = round(max_tilt, 2)
    res["recovery_time_from_0.3rad_s"] = t_rec

    # 2 push sweeps: settle 1 s, then a sudden base velocity change on one axis
    def push_sweep(axis):
        best = 0.0
        for dv in np.arange(0.1, 1.51, 0.1):
            st = settle_1s(init_state())
            st = st._replace(data=st.data.replace(qvel=st.data.qvel.at[axis].add(float(dv))),
                             done=jnp.zeros(()), steps=jnp.zeros((), dtype=jnp.int32))
            (tilt, v, *_), fell = until_fall(rollout_np(st, N5))
            if (not fell) and settled(tilt[-50:]) and abs(v[-50:].mean()) < 0.1:
                best = float(dv)
            else:
                break
        return round(best, 2)

    res["max_push_fwd_m_s"] = push_sweep(0)
    if env_name != "s1":
        res["max_push_lat_m_s"] = push_sweep(1)

    # 3 velocity tracking
    for v_ref in (0.3, 1.0):
        (tilt, vx, _, tau_l, tau_w, power, _), fell = until_fall(
            rollout_np(init_state(cmd=(v_ref, 0.0, None)), int(8.0 / DT)))
        res[f"track_{v_ref}_fell"] = fell
        res[f"track_{v_ref}_mean_v"] = round(float(vx[-200:].mean()), 3)
        band = np.abs(vx - v_ref) < 0.1 * v_ref
        idx = None if fell else next((i for i in range(len(band)) if band[i:].all()), None)
        res[f"track_{v_ref}_settle_s"] = None if idx is None else round(idx * DT, 2)
        res[f"track_{v_ref}_overshoot_pct"] = round(
            max(0.0, (float(vx.max()) - v_ref) / v_ref * 100), 1)
        if v_ref == 1.0 and not fell:
            steady = slice(int(4.0 / DT), None)
            dist = float(np.sum(vx[steady]) * DT)
            res["energy_J_per_m_at_1m_s"] = (
                round(float(np.sum(power[steady]) * DT) / dist, 2) if dist > 1.0 else None)
            res["tau_peak_drive1_leg_wheel_Nm"] = [float(tau_l.max()), float(tau_w.max())]

    # 4 height tracking (S2 onward)
    height_errs = []
    if env_name != "s1":
        lo, hi = aux["h_range"]
        for h in (lo, aux["h_nom"], hi):
            (*_, hh), fell = until_fall(rollout_np(init_state(cmd=(0.0, 0.0, h)), N5))
            e = None if fell else round(abs(float(hh[-100:].mean()) - h) * 1000, 1)
            res[f"height_{h:.3f}_signed_mm"] = (
                None if fell else round((float(hh[-100:].mean()) - h) * 1000, 1))
            res[f"height_{h:.3f}_err_mm"] = e
            height_errs.append(e)

    checks = {
        "recovers 0.35 rad tilt (R07)": max_tilt >= 0.35,
        "forward push 0.4 m/s (R07)": res["max_push_fwd_m_s"] >= 0.4,
        "tracks 1.0 m/s": res.get("track_1.0_settle_s") is not None,
        "standing lean under 6 deg (D12)": tilt_eq < math.radians(6.0),
    }
    if env_name != "s1":
        checks["lateral push 0.4 m/s (R07)"] = res["max_push_lat_m_s"] >= 0.4
        checks["torso level, lean under 3 deg"] = tilt_eq < math.radians(3.0)
        checks["height within 10 mm at all three"] = all(
            e is not None and e <= 10.0 for e in height_errs)
    print()
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(checks.values())
    print(f"\n{env_name.upper()} {'PASS' if ok else 'FAIL'}")
    finish(ok)


if __name__ == "__main__":
    main()
