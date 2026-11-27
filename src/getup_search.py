"""Open loop get up search (cross entropy method) with the locomotion policy catching.
Each candidate is a piecewise linear leg and wheel command (knots every 0.5 s, 8 s). It plays
from the settled start pose until the handoff (S4 config: tilt < 20 deg, torso > 0.15 m, body
rate < 2 rad/s); then the locomotion policy takes over and must stay up for 3 s. Every
candidate is scored over several start variants (leg noise) so the motion is robust.
Run from the repo root: python src/getup_search.py --loco s3v9 [--pop 256 --iters 20 --starts 4]
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import json
import pickle
import sys
import time

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402

DT = 0.01
KNOT_S = 0.5
SCRIPT_S, HOLD_S = 8.0, 3.0
POSES = ["back", "front", "side"]
RATE_OK = 8.0      # rad/s of body rotation before the penalty starts
DOWN_OK = 0.5      # m/s of downward torso speed before the penalty starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", required=True)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--elite", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["catch", "pose"], default="catch")
    ap.add_argument("--out", default="getup_search")
    args = ap.parse_args()

    sl = pickle.load(open(ROOT / "runs" / args.loco / "ckpt.pkl", "rb"))
    cfg = dict(CONFIGS["s4"], dr=False, getup_handoff="switch", episode_s=SCRIPT_S + HOLD_S)
    reset, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    A = sizes["act"]
    N = round((SCRIPT_S + HOLD_S) / DT)
    K = round(SCRIPT_S / KNOT_S) + 1
    t_knot = jnp.arange(K) * KNOT_S
    t_step = jnp.arange(N) * DT

    cand = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(args.seed), 96))
    pose = np.array(cand.dr["pose"])
    starts = [jax.tree.map(lambda x, ix=np.where(pose == j)[0][:args.starts]: x[ix], cand)
              for j in range(3)]

    def actions(knots):
        u = jax.vmap(lambda col: jnp.interp(t_step, t_knot, col), in_axes=1, out_axes=1)(
            jnp.clip(knots, -1.0, 1.0))
        return jnp.arctanh(jnp.clip(u, -0.995, 0.995))   # the env applies tanh

    def rollout(s0, knots):
        def body(c, a):
            s, alive, hand, t_h, t_f, best, pr, vd = c
            ml, _, _ = net.apply(sl["params"], s.obs)
            scripted = alive * (s.mode > 0.5)
            s = step(s, jnp.where(s.mode > 0.5, a, ml))
            q, v = s.data.qpos, s.data.qvel
            up = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
            pose_q = jnp.clip(up, 0.0, 1.0) * jnp.clip((q[2] - 0.08) / 0.07, 0.0, 1.0)
            best = jnp.maximum(best, scripted * pose_q)
            pr = jnp.maximum(pr, scripted * jnp.linalg.norm(v[3:6]))
            vd = jnp.minimum(vd, scripted * v[2])
            sw = scripted * (s.mode < 0.5)
            hand, t_h = hand + sw, t_h + sw * s.steps * DT
            fall = alive * (s.done > 0.5) * (s.steps < N)
            t_f = jnp.where(fall > 0, s.steps * DT, t_f)
            alive = alive * (1.0 - (s.done > 0.5))
            return (s, alive, hand, t_h, t_f, best, pr, vd), None
        z = jnp.zeros(())
        init = (s0, jnp.ones(()), z, z, jnp.asarray(N * DT + 1.0), z, z, z)
        c, _ = jax.lax.scan(body, init, actions(knots))
        _, _, hand, t_h, t_f, best, pr, vd = c
        held = jnp.clip((jnp.minimum(t_f, N * DT) - t_h) / HOLD_S, 0.0, 1.0) * hand
        ok = hand * (t_h <= SCRIPT_S) * (t_f > t_h + HOLD_S)
        return ok, hand, t_h, held, best, pr, vd

    def trajectory(s0, knots):
        def body(s, a):
            ml, _, _ = net.apply(sl["params"], s.obs)
            s = step(s, jnp.where(s.mode > 0.5, a, ml))
            return s, (s.data.qpos, s.done, s.mode)
        return jax.lax.scan(body, s0, actions(knots))[1]

    ev = jax.jit(jax.vmap(jax.vmap(rollout, in_axes=(0, None)), in_axes=(None, 0)))
    trj = jax.jit(trajectory)
    rng = np.random.default_rng(args.seed)
    n_el = max(4, int(args.pop * args.elite))
    summary, saved = {}, {}
    print(f"{K} knots x {A} commands, population {args.pop} x {args.starts} starts, "
          f"{args.iters} rounds per pose (the first round compiles for several minutes)", flush=True)

    for j, name in enumerate(POSES):
        mean, sd = np.zeros((K, A)), np.full((K, A), 0.6)
        top = None
        t0 = time.time()
        for it in range(args.iters):
            X = np.clip(mean + sd * rng.standard_normal((args.pop, K, A)), -1.0, 1.0)
            if top is not None:
                X[0] = top["knots"]
            ok, hand, t_h, held, best, pr, vd = (
                np.array(v) for v in ev(starts[j], jnp.asarray(X, dtype=jnp.float32)))
            pen = 20 * np.maximum(pr - RATE_OK, 0.0) + 500 * np.maximum(-vd - DOWN_OK, 0.0)
            if args.objective == "pose":
                score = (1000 * best - pen).mean(1)
            else:
                score = (600 * hand + 600 * held + 300 * best - 20 * t_h - pen).mean(1)
            order = np.argsort(-score)
            el = X[order[:n_el]]
            mean = 0.8 * el.mean(0) + 0.2 * mean
            sd = np.maximum(0.8 * el.std(0) + 0.2 * sd, 0.05)
            i = int(order[0])
            if top is None or score[i] > top["score"]:
                th = t_h[i][hand[i] > 0.5]
                top = dict(score=float(score[i]), knots=X[i].copy(),
                           best_pose=float(best[i].mean()), best_pose_min=float(best[i].min()),
                           success=float(ok[i].mean()), handoff=float(hand[i].mean()),
                           t_handoff=float(th.mean()) if th.size else None,
                           peak_rate=float(pr[i].max()), max_down_v=float(vd[i].min()))
            print(f"{name:5s} round {it + 1:2d}/{args.iters}  population success "
                  f"{ok.mean() * 100:5.1f}%  best: success {top['success'] * 100:3.0f}% "
                  f"handoff {top['handoff'] * 100:3.0f}% "
                  f"pose {top['best_pose']:.2f}/{top['best_pose_min']:.2f}  "
                  f"rate {top['peak_rate']:4.1f}  down {top['max_down_v']:+.2f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
        summary[name] = {k: v for k, v in top.items() if k != "knots"}
        saved[name] = top["knots"]

    np.savez(ROOT / "results" / f"{args.out}.npz", knot_s=KNOT_S, **saved)
    json.dump(summary, open(ROOT / "results" / f"{args.out}.json", "w"), indent=2)

    cpu = mujoco.MjModel.from_xml_string(build_xml(aux["p"], mode="torque"))
    d = mujoco.MjData(cpu)
    ren = mujoco.Renderer(cpu, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = cpu.body("torso").id
    cam.distance, cam.elevation, cam.azimuth = 1.1, -15.0, 70.0
    for j, name in enumerate(POSES):
        s0 = jax.tree.map(lambda x: x[0], starts[j])
        qpos, done, mode = (np.array(v) for v in trj(s0, jnp.asarray(saved[name], dtype=jnp.float32)))
        end = int(np.argmax(done > 0.5)) if (done > 0.5).any() else N - 1
        out = ROOT / "results" / f"{args.out}_{name}.mp4"
        with imageio.get_writer(out, fps=50) as w:
            for k in range(0, end + 1, 2):
                d.qpos[:] = qpos[k]
                mujoco.mj_forward(cpu, d)
                ren.update_scene(d, cam)
                w.append_data(ren.render())
        sw = np.where(mode < 0.5)[0]
        print(f"wrote {out.relative_to(ROOT)}"
              + (f" (handoff at {(sw[0] + 1) * DT:.2f} s)" if len(sw) else " (no handoff)"))

    print("\npose   success  handoff  best pose (mean/min)  peak rate  max down v")
    for name, s in summary.items():
        print(f"{name:6s} {s['success'] * 100:6.0f}%  {s['handoff'] * 100:6.0f}%  "
              f"{s['best_pose']:9.2f} / {s['best_pose_min']:.2f}  "
              f"{s['peak_rate']:6.1f}r/s  {s['max_down_v']:+7.2f}m/s")
    print("\nsuccess = handoff within 8 s and the locomotion policy stays up 3 s after it")


if __name__ == "__main__":
    main()