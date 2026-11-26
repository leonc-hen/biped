"""Batched evaluation harness (plan 2.8). Run from the repo root:
python src/eval_batch.py --run s2_v3 [--compare]

Part A: deterministic sweeps in one parallel batch. Tilt forward and back, pushes
in four directions up to 2.0 m/s, tracking, heights, standing. Same criteria and
time windows as eval_policy.py, so --compare checks that the two agree.
Part A2: 1000 R07 pushes (0.4 m/s) in random directions at random times, success
rate with a 95 percent Wilson interval, also split into 8 direction sectors.
Part B: 1000 full training distribution episodes (random commands and pushes),
fall rate with interval, tracking and height RMSE, per joint torque and speed.
Model randomization arrives in Step 12; then Part B runs under it.
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
import time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import (make_env, CONFIGS, gravity_body, forward_velocity,  # noqa: E402
                        LEG_CTRL, WHEEL_CTRL, LEG_Q, LEG_V, WHEEL_V)
from src.ppo import ActorCritic  # noqa: E402

DT = 0.01
N_DET = 800            # 8 s, the longest deterministic test
N_STAT = 1000          # one full training episode
N_PUSH_RAND = 1000
N_EPISODES_B = 1000
DEG1, DEG2 = math.radians(1.0), math.radians(2.0)
LEG_NAMES = ["hip_l", "knee_l", "hip_r", "knee_r"]
SECTORS = ["fwd", "fwd_left", "left", "back_left", "back", "back_right", "right", "fwd_right"]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": round(p, 4), "ci95": [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)],
            "n": n}


def make_rollout(step_fn, net, params, n, to_motor):
    """Batched rollout. Each episode can receive one external push (push_k < 0: none)."""
    def one(state, push_k, push_dv):
        def body(st, t):
            d = st.data
            d = d.replace(qvel=d.qvel.at[0:2].add(jnp.where(t == push_k, push_dv, 0.0)))
            st = st._replace(data=d)
            mean, _, _ = net.apply(params, st.obs)
            st = step_fn(st, mean)
            d = st.data
            g = gravity_body(d.qpos[3:7])
            return st, dict(
                tilt=jnp.arccos(jnp.clip(-g[2], -1.0, 1.0)), v=forward_velocity(d),
                done=st.done, tau_leg=d.ctrl[LEG_CTRL], tau_w=d.ctrl[WHEEL_CTRL],
                v_leg=to_motor(d.qvel[LEG_V]), v_w=d.qvel[WHEEL_V], h=d.qpos[2], cmd=st.cmd,
                q_leg=d.qpos[LEG_Q], com_z=d.subtree_com[1, 2], jump_apex=st.jump_apex)
        _, out = jax.lax.scan(body, state, jnp.arange(n))
        return out
    return jax.jit(jax.vmap(one))


def power_of(o):
    return (np.abs(o["tau_leg"] * o["v_leg"]).sum(-1)
            + np.abs(o["tau_w"] * o["v_w"]).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--dr_scale", type=float, default=None)
    ap.add_argument("--a_only", action="store_true", help="Part A only, skip B and C")
    ap.add_argument("--a_noise", action="store_true", help="Part A with full sensor noise")
    ap.add_argument("--a_delay_ms", type=float, default=None, help="Part A with a fixed delay")
    args = ap.parse_args()
    t0 = time.time()

    with open(ROOT / "runs" / args.run / "ckpt.pkl", "rb") as f:
        saved = pickle.load(f)
    params = saved["params"]
    env_name = saved["cfg"].get("env", "s1")
    train_cfg = dict(CONFIGS[env_name])
    train_cfg.update(saved["cfg"].get("env_overrides", {}))
    det_cfg = dict(train_cfg, push_every_s=None, resample_s=None, dr=False)
    if args.a_noise or args.a_delay_ms is not None:
        det_cfg.update(dr=True, dr_scale=0.0, noise_scale=1.0 if args.a_noise else 0.0,
                       delay_fixed_ms=args.a_delay_ms)
    b_cfg = dict(train_cfg)
    if args.dr_scale is not None:
        b_cfg.update(dr=True, dr_scale=args.dr_scale)
    _, step_det, sizes, aux = make_env(det_cfg)
    reset_tr, step_tr, _, _ = make_env(b_cfg)
    net = ActorCritic(sizes["act"])
    p, model = aux["p"], aux["model"]
    r_w = p["wheel"]["radius_m"]
    h_nom = aux["h_nom"]
    h_t = h_nom - r_w
    clr = p["sim"]["spawn_clearance_m"]
    h_lo, h_hi = aux["h_range"]
    max_leg_speed = p["leg_motor"]["max_speed_rad_s"]

    def init_state(theta, cmd):
        qpos = aux["qpos0"]
        qpos = qpos.at[0].set(h_t * jnp.sin(theta))
        qpos = qpos.at[2].set(r_w + h_t * jnp.cos(theta) + clr)
        zero = jnp.zeros_like(theta)
        qpos = qpos.at[3:7].set(jnp.stack([jnp.cos(theta / 2), zero, jnp.sin(theta / 2), zero]))
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos, qvel=jnp.zeros(model.nv)))
        return aux["init_from_data"](data, cmd)

    # ---- episode table: (kind, value, theta, cmd, push_k, push_dv, end_step)
    specs = []

    def add(kind, value, theta=0.0, cmd=(0.0, 0.0, h_nom), push_k=-1,
            push_dv=(0.0, 0.0), end=500):
        specs.append((kind, float(value), float(theta), tuple(float(c) for c in cmd),
                      int(push_k), tuple(float(d) for d in push_dv), int(end)))

    add("stand", 0.0)
    for th in np.round(np.arange(0.1, 1.01, 0.1), 2):
        add("tilt_fwd", th, theta=th)
        add("tilt_back", th, theta=-th)
    for kind, ax, sgn in (("push_fwd", 0, 1), ("push_back", 0, -1),
                          ("push_lat", 1, 1), ("push_lat_neg", 1, -1)):
        for dv in np.round(np.arange(0.1, 2.01, 0.1), 2):
            d = [0.0, 0.0]
            d[ax] = sgn * dv
            add(kind, dv, push_k=100, push_dv=d, end=600)   # settle 1 s, push, judge 5 s later
    for v_ref in (0.3, 1.0):
        add("track", v_ref, cmd=(v_ref, 0.0, h_nom), end=800)
    for h in (h_lo, h_nom, h_hi):
        add("height", h, cmd=(0.0, 0.0, h), end=500)
    rng = np.random.default_rng(0)
    for a, k in zip(rng.uniform(0, 2 * math.pi, N_PUSH_RAND),
                    rng.integers(100, 301, N_PUSH_RAND)):
        add("push_rand", 0.4, push_k=k, push_dv=(0.4 * math.cos(a), 0.4 * math.sin(a)), end=800)

    # ---- Part A and A2: one batch
    print(f"Part A: {len(specs)} episodes in one batch "
          f"(the first run compiles for several minutes)", flush=True)
    states = jax.jit(jax.vmap(init_state))(
        jnp.array([s[2] for s in specs]), jnp.array([s[3] for s in specs]))
    roll_det = make_rollout(step_det, net, params, N_DET, aux["to_motor"])
    out = {k: np.array(v) for k, v in roll_det(
        states, jnp.array([s[4] for s in specs]), jnp.array([s[5] for s in specs])).items()}
    tilt, v, done, h = out["tilt"], out["v"], out["done"], out["h"]
    power = power_of(out)

    def alive_len(i, end):
        d = done[i, :end] > 0.5
        return int(np.argmax(d)) + 1 if d.any() else end

    def fell_before(i, end):
        return bool((done[i, :end] > 0.5).any())

    res = {"env": env_name, "run": args.run}
    if fell_before(0, 500):
        print("FAIL: fell while standing, nothing else is meaningful")
        sys.exit(1)
    tilt_eq = float(tilt[0, 100:500].mean())
    res["standing_tilt_mean_deg"] = round(math.degrees(tilt_eq), 3)
    res["standing_tilt_std_deg"] = round(math.degrees(float(tilt[0, 100:500].std())), 3)
    seg = tilt[0, 100:500] - tilt[0, 100:500].mean()
    res["standing_wobble_hz"] = round(np.count_nonzero(np.diff(np.sign(seg)) != 0) / 2 / 4.0, 1)
    l1, l2 = p["thigh"]["length_m"], p["shank"]["length_m"]
    hips = out["q_leg"][0, 100:500]
    foot = lambda qh, qk: -(l1 * np.sin(qh) + l2 * np.sin(qh + qk))
    stagger_mm = float(np.abs(foot(hips[:, 0], hips[:, 1]) - foot(hips[:, 2], hips[:, 3])).mean() * 1000)
    res["standing_wheel_stagger_mm"] = round(stagger_mm, 1)

    def success(i):
        kind, _, _, _, _, _, end = specs[i]
        if fell_before(i, end):
            return False
        tail = tilt[i, end - 50:end]
        ok = abs(tail.mean() - tilt_eq) < DEG2 and tail.std() < DEG1
        if kind.startswith("push"):
            ok = ok and abs(v[i, end - 50:end].mean()) < 0.1
        return ok

    def sweep_max(kind):
        best = 0.0
        for i, s in enumerate(specs):
            if s[0] != kind:
                continue
            if success(i):
                best = s[1]
            else:
                break
        return round(best, 2)

    def find(kind, value):
        return next(i for i, s in enumerate(specs) if s[0] == kind and abs(s[1] - value) < 1e-6)

    res["max_recovered_tilt_rad"] = sweep_max("tilt_fwd")
    res["max_recovered_tilt_back_rad"] = sweep_max("tilt_back")
    i03 = find("tilt_fwd", 0.3)
    near = np.where(np.abs(tilt[i03, :500] - tilt_eq) < DEG2)[0]
    res["recovery_time_from_0.3rad_s"] = (
        float(near[0] * DT) if success(i03) and len(near) else None)
    res["max_push_fwd_m_s"] = sweep_max("push_fwd")
    res["max_push_back_m_s"] = sweep_max("push_back")
    res["max_push_lat_m_s"] = sweep_max("push_lat")
    res["max_push_lat_neg_m_s"] = sweep_max("push_lat_neg")
    res["standing_mean_v_m_s"] = round(float(v[0, 100:500].mean()), 4)

    def why(i):
        end = specs[i][6]
        if fell_before(i, end):
            return f"fell at {np.argmax(done[i, :end] > 0.5) * DT:.2f} s"
        tail = tilt[i, end - 50:end]
        seg = tilt[i, end - 200:end] - tilt[i, end - 200:end].mean()
        freq = np.count_nonzero(np.diff(np.sign(seg)) != 0) / 2 / (200 * DT)
        return (f"lean offset {math.degrees(tail.mean() - tilt_eq):+.2f} deg, "
                f"wobble {math.degrees(tail.std()):.2f} deg at {freq:.1f} Hz, "
                f"final speed {v[i, end - 50:end].mean():+.3f} m/s")

    res["first_failures"] = {}
    for kind in ("tilt_fwd", "tilt_back", "push_fwd", "push_back", "push_lat", "push_lat_neg"):
        i = next((j for j, s in enumerate(specs) if s[0] == kind and not success(j)), None)
        if i is not None:
            res["first_failures"][f"{kind}_{specs[i][1]:.1f}"] = why(i)

    for v_ref in (0.3, 1.0):
        i = find("track", v_ref)
        n_ = alive_len(i, 800)
        fell = fell_before(i, 800)
        vx = v[i, :n_]
        res[f"track_{v_ref}_fell"] = fell
        res[f"track_{v_ref}_mean_v"] = round(float(vx[-200:].mean()), 3)
        band = np.abs(vx - v_ref) < 0.1 * v_ref
        suffix_ok = np.flip(np.logical_and.accumulate(np.flip(band)))
        idx = None if fell or not suffix_ok.any() else int(np.argmax(suffix_ok))
        res[f"track_{v_ref}_settle_s"] = None if idx is None else round(idx * DT, 2)
        res[f"track_{v_ref}_overshoot_pct"] = round(
            max(0.0, (float(vx.max()) - v_ref) / v_ref * 100), 1)
        if v_ref == 1.0 and not fell:
            dist = float(np.sum(vx[400:]) * DT)
            res["energy_J_per_m_at_1m_s"] = (
                round(float(np.sum(power[i, 400:800]) * DT) / dist, 2) if dist > 1.0 else None)

    height_errs = []
    if env_name != "s1":
        for hc in (h_lo, h_nom, h_hi):
            i = find("height", hc)
            n_ = alive_len(i, 500)
            signed = None if fell_before(i, 500) else (float(h[i, :n_][-100:].mean()) - hc) * 1000
            res[f"height_{hc:.3f}_signed_mm"] = None if signed is None else round(signed, 1)
            res[f"height_{hc:.3f}_err_mm"] = None if signed is None else round(abs(signed), 1)
            height_errs.append(None if signed is None else abs(signed))

    rand = [i for i, s in enumerate(specs) if s[0] == "push_rand"]
    ok_rand = np.array([success(i) for i in rand])
    res["push_rand_0.4_success"] = wilson(int(ok_rand.sum()), len(rand))
    ang = np.array([math.atan2(specs[i][5][1], specs[i][5][0]) for i in rand])
    sector = (((ang + math.pi / 8) % (2 * math.pi)) // (math.pi / 4)).astype(int)
    res["push_rand_0.4_by_sector"] = {
        SECTORS[s]: round(float(ok_rand[sector == s].mean()), 3) for s in range(8)}

    if not args.a_only:
        # ---- Part B: the training distribution
        print("Part B: training distribution episodes", flush=True)
        nb = N_EPISODES_B
        states_b = jax.jit(jax.vmap(reset_tr))(jax.random.split(jax.random.PRNGKey(123), nb))
        roll_b = make_rollout(step_tr, net, params, N_STAT, aux["to_motor"])
        ob = {k: np.array(x) for k, x in roll_b(
            states_b, jnp.full(nb, -1), jnp.zeros((nb, 2))).items()}
        d = ob["done"] > 0.5
        first = np.where(d.any(1), d.argmax(1), N_STAT)
        fell_b = d[:, :N_STAT - 1].any(1)                  # done before the timeout step
        t_idx = np.arange(N_STAT)[None, :]
        alive = t_idx < first[:, None]
        period = round(b_cfg["resample_s"] / DT) if b_cfg["resample_s"] else N_STAT
        settled_mask = alive & ((t_idx % period) >= 100)   # skip 1 s after every command change
        cmd_used = np.concatenate([np.array(states_b.cmd)[:, None, :], ob["cmd"][:, :-1, :]], axis=1)
        err_v = ob["v"] - cmd_used[..., 0]
        err_h = ob["h"] - cmd_used[..., 2]

        def rms(x, m):
            return round(float(np.sqrt(np.mean(np.square(x[m])))), 4)

        res["train_dist_fall"] = wilson(int(fell_b.sum()), nb)
        res["train_dist_v_rmse_m_s"] = {"all": rms(err_v, alive), "settled": rms(err_v, settled_mask)}
        res["train_dist_h_rmse_mm"] = {"all": round(rms(err_h, alive) * 1000, 1),
                                       "settled": round(rms(err_h, settled_mask) * 1000, 1)}
        tl, tw = np.abs(ob["tau_leg"]), np.abs(ob["tau_w"])
        vl, vw = np.abs(ob["v_leg"]), np.abs(ob["v_w"])
        res["train_dist_leg_torque_Nm"] = {
            n: {"peak": round(float(tl[..., j][alive].max()), 3),
                "p99": round(float(np.percentile(tl[..., j][alive], 99)), 3),
                "rms": round(float(np.sqrt(np.mean(np.square(tl[..., j][alive])))), 3)}
            for j, n in enumerate(LEG_NAMES)}
        res["train_dist_wheel_torque_Nm"] = {
            "peak": round(float(tw[alive].max()), 3),
            "p99": round(float(np.percentile(tw[alive], 99)), 3),
            "rms": round(float(np.sqrt(np.mean(np.square(tw[alive])))), 3)}
        res["train_dist_leg_speed"] = {
            "peak_rad_s": round(float(vl[alive].max()), 2),
            "frac_above_90pct_limit": round(float(np.mean(vl[alive] > 0.9 * max_leg_speed)), 5)}
        res["train_dist_wheel_speed_peak_rad_s"] = round(float(vw[alive].max()), 1)
        res["train_dist_mean_power_W"] = round(float(power_of(ob)[alive].mean()), 2)

        # ---- Part C: R07 pushes under randomization
        if b_cfg["dr"]:
            print(f"Part C: R07 pushes under randomization, scale {b_cfg['dr_scale']}", flush=True)
            c_cfg = dict(b_cfg, push_every_s=None, resample_s=None,
                         cmd_fixed=(0.0, 0.0, h_nom), reset_perturb=False, delay_max_ms=10.0,
                         jump=False)
            reset_c, step_c, _, _ = make_env(c_cfg)
            nc = 1000
            rc = np.random.default_rng(1)
            ang_c = rc.uniform(0, 2 * math.pi, nc)
            k_c = rc.integers(100, 301, nc)
            dv_c = np.stack([0.4 * np.cos(ang_c), 0.4 * np.sin(ang_c)], 1)
            states_c = jax.jit(jax.vmap(reset_c))(jax.random.split(jax.random.PRNGKey(7), nc))
            roll_c = make_rollout(step_c, net, params, N_DET, aux["to_motor"])
            oc = {k: np.array(x) for k, x in roll_c(
                states_c, jnp.array(k_c), jnp.array(dv_c)).items()}
            oc0 = {k: np.array(x) for k, x in roll_c(
                states_c, jnp.full(nc, -1), jnp.zeros((nc, 2))).items()}    # same robots, no push
            ok_pre = np.zeros(nc, dtype=bool)
            drift = np.zeros(nc)
            okc = np.zeros(nc, dtype=bool)
            reasons = {"fell": 0, "lean offset": 0, "wobble": 0, "still moving": 0}
            for i in range(nc):
                if (oc["done"][i] > 0.5).any():
                    reasons["fell"] += 1
                    continue
                k = int(k_c[i])
                pre = oc["tilt"][i, k - 50:k].mean()      # each robot's own equilibrium
                tail = oc["tilt"][i, N_DET - 50:]
                twin_fell = (oc0["done"][i] > 0.5).any()
                ref = pre if twin_fell else oc0["tilt"][i, N_DET - 50:].mean()
                drift[i] = abs(ref - pre)
                lean_ok = abs(tail.mean() - ref) < DEG2
                ok_pre[i] = abs(tail.mean() - pre) < DEG2 and tail.std() < DEG1 and \
                    abs(oc["v"][i, N_DET - 50:].mean()) < 0.1
                wob_ok = tail.std() < DEG1
                spd_ok = abs(oc["v"][i, N_DET - 50:].mean()) < 0.1
                okc[i] = lean_ok and wob_ok and spd_ok
                if not okc[i]:
                    reasons["lean offset" if not lean_ok else "wobble" if not wob_ok else "still moving"] += 1
            res["dr_push_fail_reasons"] = reasons
            res["dr_push_0.4_success_vs_prepush"] = round(float(ok_pre.mean()), 3)
            res["dr_nopush_drift_deg"] = {"median": round(float(np.degrees(np.median(drift))), 2),
                                          "p90": round(float(np.degrees(np.percentile(drift, 90))), 2)}
            res["dr_push_0.4_success"] = wilson(int(okc.sum()), nc)
            sec_c = (((ang_c + math.pi / 8) % (2 * math.pi)) // (math.pi / 4)).astype(int)
            res["dr_push_0.4_by_sector"] = {
                SECTORS[s]: round(float(okc[sec_c == s].mean()), 3) for s in range(8)}

            print("Part D: training distribution with pushes limited to R07", flush=True)
            d_cfg = dict(b_cfg, push_max_norm=0.4, delay_max_ms=10.0, jump=False)
            reset_d, step_d, _, _ = make_env(d_cfg)
            states_d = jax.jit(jax.vmap(reset_d))(jax.random.split(jax.random.PRNGKey(321), nb))
            roll_d = make_rollout(step_d, net, params, N_STAT, aux["to_motor"])
            out_d = {k: np.array(x) for k, x in roll_d(
                states_d, jnp.full(nb, -1), jnp.zeros((nb, 2))).items()}
            fell_d = (out_d["done"] > 0.5)[:, :N_STAT - 1].any(1)
            res["within_spec_fall"] = wilson(int(fell_d.sum()), nb)
            drd = {k: np.array(v) for k, v in states_d.dr.items()}
            cmd_d = out_d["cmd"]
            factors = {
                "battery_V": drd["v_scale"] * 24.0,
                "friction": drd["mu"],
                "delay_ms": drd["delay_sub"] * 1000.0 * p["sim"]["physics_dt_s"],
                "slope_deg": np.degrees(np.arccos(np.clip(-drd["g_unit"][:, 2], -1.0, 1.0))),
                "torso_com_x_mm": drd["com_offset"][:, 0] * 1000.0,
                "max_cmd_speed_m_s": np.abs(cmd_d[..., 0]).max(1),
                "max_cmd_yaw_rad_s": np.abs(cmd_d[..., 1]).max(1),
                "max_cmd_height_m": cmd_d[..., 2].max(1),
            }
            table = {}
            print("  within spec fall rate by factor (thirds of the 1000 episodes):")
            for name, x in factors.items():
                b = np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))
                row = {f"{x[b == k].min():.3g} to {x[b == k].max():.3g}": round(float(fell_d[b == k].mean()), 3)
                       for k in range(3) if (b == k).any()}
                table[name] = row
                print(f"    {name:20s} " + "   ".join(f"{r}: {v:.3f}" for r, v in row.items()))
            res["within_spec_fall_by_factor"] = table

    # ---- Part E: jumps (R06), nominal with --a_only, randomized otherwise
    if train_cfg.get("jump"):
        n_e = 200 if args.a_only else 1000
        base_e = det_cfg if args.a_only else dict(b_cfg, delay_max_ms=10.0)
        e_cfg = dict(base_e, jump=True, push_every_s=None, resample_s=None,
                     cmd_fixed=(0.0, 0.0, h_nom), reset_perturb=False)
        print(f"Part E: {n_e} jumps, {'nominal' if args.a_only else 'randomized'}", flush=True)
        reset_e, step_e, _, _ = make_env(e_cfg)
        states_e = jax.jit(jax.vmap(reset_e))(jax.random.split(jax.random.PRNGKey(55), n_e))
        roll_e = make_rollout(step_e, net, params, N_STAT, aux["to_motor"])
        oe = {k: np.array(x) for k, x in roll_e(
            states_e, jnp.full(n_e, -1), jnp.zeros((n_e, 2))).items()}
        ja = np.array(states_e.jump_at)
        win_e = round(e_cfg["jump_window_s"] / DT)
        rise = np.zeros(n_e)
        fell_e = np.zeros(n_e, dtype=bool)
        settled_e = np.zeros(n_e, dtype=bool)
        land_tau = np.zeros(n_e)
        for i in range(n_e):
            k = int(ja[i])
            rise[i] = oe["com_z"][i, k:k + win_e].max() - oe["com_z"][i, k - 50:k].mean()
            fell_e[i] = (oe["done"][i, :N_STAT - 1] > 0.5).any()
            settled_e[i] = (not fell_e[i]) and oe["tilt"][i, -50:].std() < DEG1
            land_tau[i] = np.abs(oe["tau_leg"][i, k:k + win_e]).max()
        clear_e = oe["jump_apex"][:, -1]
        tag = "jump_nominal" if args.a_only else "jump_dr"
        res[f"{tag}_success_10cm"] = wilson(int((settled_e & (rise >= 0.10)).sum()), n_e)
        res[f"{tag}_success_12cm"] = round(float((settled_e & (rise >= 0.12)).mean()), 3)
        res[f"{tag}_fell"] = round(float(fell_e.mean()), 3)
        res[f"{tag}_com_rise_cm"] = {q: round(float(np.percentile(rise, p)) * 100, 1)
                                     for q, p in (("p10", 10), ("median", 50), ("p90", 90))}
        res[f"{tag}_wheel_clearance_cm_median"] = round(float(np.median(clear_e)) * 100, 1)
        res[f"{tag}_landing_leg_torque_Nm"] = {"p99": round(float(np.percentile(land_tau, 99)), 2),
                                               "peak": round(float(land_tau.max()), 2)}
        fell_t = np.array([(np.argmax(oe["done"][i] > 0.5) - int(ja[i])) * DT if fell_e[i] else np.nan
                           for i in range(n_e)])
        tilt_win = np.array([np.degrees(oe["tilt"][i, int(ja[i]):int(ja[i]) + win_e].max())
                             for i in range(n_e)])
        if fell_e.any():
            res[f"{tag}_fall_time_after_cmd_s"] = {
                q: round(float(np.nanpercentile(fell_t, p)), 2)
                for q, p in (("p10", 10), ("median", 50), ("p90", 90))}
        res[f"{tag}_max_tilt_in_window_deg"] = {
            "falls": round(float(np.median(tilt_win[fell_e])), 1) if fell_e.any() else None,
            "successes": round(float(np.median(tilt_win[~fell_e])), 1) if (~fell_e).any() else None}
        if not args.a_only and fell_e.any():
            dre = {k: np.array(v) for k, v in states_e.dr.items()}
            factors = {
                "battery_V": dre["v_scale"] * 24.0,
                "friction": dre["mu"],
                "delay_ms": dre["delay_sub"] * 1000.0 * p["sim"]["physics_dt_s"],
                "slope_deg": np.degrees(np.arccos(np.clip(-dre["g_unit"][:, 2], -1.0, 1.0))),
                "torso_com_x_mm": dre["com_offset"][:, 0] * 1000.0,
                "torso_mass_scale": dre["mass_scale"][:, 1],
                "leg_torque_const": dre["kt_leg"].mean(1),
            }
            print("  jump fall rate by factor (thirds of the episodes):")
            table = {}
            for name, x in factors.items():
                b = np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))
                row = {f"{x[b == k].min():.3g} to {x[b == k].max():.3g}": round(float(fell_e[b == k].mean()), 3)
                       for k in range(3) if (b == k).any()}
                table[name] = row
                print(f"    {name:20s} " + "   ".join(f"{r}: {v:.3f}" for r, v in row.items()))
            res[f"{tag}_fall_by_factor"] = table

    res["runtime_s"] = round(time.time() - t0, 1)
    tag = "" if args.dr_scale is None else f"_dr{args.dr_scale}"
    tag += "_Anoise" if args.a_noise else ""
    tag += f"_Adelay{args.a_delay_ms:g}" if args.a_delay_ms is not None else ""
    tag += "_Aonly" if args.a_only else ""
    out_path = ROOT / "results" / f"evalb_{env_name}_{args.run}{tag}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out_path.relative_to(ROOT)}")

    nominal = {
        "tilt 0.35 rad forward and back (R07)":
            min(res["max_recovered_tilt_rad"], res["max_recovered_tilt_back_rad"]) >= 0.35,
        "push 0.4 m/s in all four directions (R07)":
            min(res[k] for k in ("max_push_fwd_m_s", "max_push_back_m_s",
                                 "max_push_lat_m_s", "max_push_lat_neg_m_s")) >= 0.4,
        "random direction 0.4 m/s push, CI lower bound 0.90":
            res["push_rand_0.4_success"]["ci95"][0] >= 0.90,
        "tracks 1.0 m/s": res.get("track_1.0_settle_s") is not None,
        "torso lean under 3 deg": tilt_eq < math.radians(3.0),
        "wheels within 15 mm front to back standing": stagger_mm < 15.0,
    }
    if env_name != "s1":
        nominal["height within 10 mm at all three"] = all(
            e is not None and e <= 10.0 for e in height_errs)
    groups = [("nominal model", nominal)]
    if b_cfg["dr"] and not args.a_only:
        groups.append((f"randomized, scale {b_cfg['dr_scale']}", {
            "R07 push success at least 95 percent (Gate A)": res["dr_push_0.4_success"]["rate"] >= 0.95,
            "within spec fall rate at most 3 percent (R07 pushes)":
                res["within_spec_fall"]["rate"] <= 0.03,
        }))
    if train_cfg.get("jump"):
        tag = "jump_nominal" if args.a_only else "jump_dr"
        groups.append((f"jumps ({'nominal' if args.a_only else 'randomized'})", {
            "R06 10 cm jump success at least 90 percent":
                res[f"{tag}_success_10cm"]["rate"] >= 0.90,
        }))
    gate_ok = True
    for title, checks in groups:
        print(f"\n{title}")
        for name, passed in checks.items():
            gate_ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    res["gate"] = {title: {name: bool(v) for name, v in checks.items()} for title, checks in groups}
    res["gate_pass"] = bool(gate_ok)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nGATE {'PASS' if gate_ok else 'FAIL'}")

    if args.compare:
        ref_path = ROOT / "results" / f"eval_{env_name}_{args.run}.json"
        ref = json.load(open(ref_path))["metrics"]
        checks = [("standing_tilt_mean_deg", 0.1), ("max_recovered_tilt_rad", 0.1),
                  ("recovery_time_from_0.3rad_s", 0.02), ("max_push_fwd_m_s", 0.1),
                  ("max_push_lat_m_s", 0.1), ("track_0.3_mean_v", 0.01),
                  ("track_1.0_mean_v", 0.01), ("track_1.0_settle_s", 0.05)]
        checks += [(k, 1.0) for k in ref if k.endswith("_signed_mm")]
        print(f"\nparity against {ref_path.relative_to(ROOT)}")
        all_ok = True
        for k, tol in checks:
            a, b = res.get(k), ref.get(k)
            if k.startswith("max_push") and a is not None:
                a = min(a, 1.5)                      # eval_policy's sweep stops at 1.5
            same = (a is None and b is None) or (
                a is not None and b is not None and abs(a - b) <= tol + 1e-9)
            all_ok &= same
            print(f"  {'OK  ' if same else 'DIFF'}  {k:32s} batch {a}   reference {b}")
        print("PARITY OK" if all_ok else "PARITY FAIL")
        sys.exit(0 if all_ok else 1)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()