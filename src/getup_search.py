"""Open loop get up search (cross entropy method) from each settled start pose, nominal robot.
Leg and wheel commands are piecewise linear between knots every 0.5 s. Body rotation above
8 rad/s and downward torso speed above 0.5 m/s are penalized, so motions stay hardware friendly.
Saves the best motion per pose as the base for a scripted get up.
Run from the repo root: python src/getup_search.py [--pop 512 --iters 30]
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import json
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
from model.build_mjcf import build_xml  # noqa: E402

DT = 0.01
KNOT_S = 0.5
POSES = ["back", "front", "side"]
RATE_OK = 8.0      # rad/s of body rotation before the penalty starts
DOWN_OK = 0.5      # m/s of downward torso speed before the penalty starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--elite", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(CONFIGS["s4"], dr=False)
    reset, step, sizes, aux = make_env(cfg)
    A = sizes["act"]
    N = round(cfg["episode_s"] / DT)
    K = N // round(KNOT_S / DT) + 1
    t_knot = jnp.arange(K) * KNOT_S
    t_step = jnp.arange(N) * DT

    cand = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(args.seed), 64))
    pose = np.array(cand.dr["pose"])
    starts = [jax.tree.map(lambda x, i=int(np.where(pose == j)[0][0]): x[i], cand)
              for j in range(3)]

    def actions(knots):
        u = jax.vmap(lambda col: jnp.interp(t_step, t_knot, col), in_axes=1, out_axes=1)(
            jnp.clip(knots, -1.0, 1.0))
        return jnp.arctanh(jnp.clip(u, -0.995, 0.995))   # the env applies tanh

    def rollout(s0, knots):
        def body(c, a):
            s, alive, ret, hand, t_h, best, mz, pr, vd = c
            s = step(s, a)
            q, v = s.data.qpos, s.data.qvel
            up = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
            pose_q = jnp.clip(up, 0.0, 1.0) * jnp.clip((q[2] - 0.08) / 0.10, 0.0, 1.0)
            ret = ret + alive * s.reward
            best = jnp.maximum(best, alive * pose_q)
            mz = jnp.maximum(mz, alive * q[2])
            pr = jnp.maximum(pr, alive * jnp.linalg.norm(v[3:6]))
            vd = jnp.minimum(vd, alive * v[2])
            h = alive * (s.done > 0.5) * (s.steps < N)
            hand, t_h = hand + h, t_h + h * s.steps * DT
            alive = alive * (1.0 - (s.done > 0.5))
            return (s, alive, ret, hand, t_h, best, mz, pr, vd), None
        z = jnp.zeros(())
        c, _ = jax.lax.scan(body, (s0, jnp.ones(()), z, z, z, z, z, z, z), actions(knots))
        return c[2:]

    def trajectory(s0, knots):
        def body(s, a):
            s = step(s, a)
            return s, (s.data.qpos, s.done)
        return jax.lax.scan(body, s0, actions(knots))[1]

    ev = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
    trj = jax.jit(trajectory)
    rng = np.random.default_rng(args.seed)
    n_el = max(4, int(args.pop * args.elite))
    summary, saved = {}, {}
    print(f"{K} knots x {A} commands, population {args.pop}, {args.iters} rounds per pose "
          f"(the first round compiles for several minutes)", flush=True)

    for j, name in enumerate(POSES):
        mean, sd = np.zeros((K, A)), np.full((K, A), 0.6)
        top = None
        t0 = time.time()
        for it in range(args.iters):
            X = np.clip(mean + sd * rng.standard_normal((args.pop, K, A)), -1.0, 1.0)
            if top is not None:
                X[0] = top["knots"]
            ret, hand, t_h, best, mz, pr, vd = (
                np.array(v) for v in ev(starts[j], jnp.asarray(X, dtype=jnp.float32)))
            score = (1000 * hand - 20 * t_h + 500 * best + 0.1 * ret
                     - 20 * np.maximum(pr - RATE_OK, 0.0) - 500 * np.maximum(-vd - DOWN_OK, 0.0))
            order = np.argsort(-score)
            el = X[order[:n_el]]
            mean = 0.8 * el.mean(0) + 0.2 * mean
            sd = np.maximum(0.8 * el.std(0) + 0.2 * sd, 0.05)
            i = int(order[0])
            if top is None or score[i] > top["score"]:
                top = dict(score=float(score[i]), knots=X[i].copy(), handoff=bool(hand[i] > 0.5),
                           t_handoff=float(t_h[i]) if hand[i] > 0.5 else None,
                           best_pose=float(best[i]), max_z=float(mz[i]),
                           peak_rate=float(pr[i]), max_down_v=float(vd[i]))
            print(f"{name:5s} round {it + 1:2d}/{args.iters}  handoff in population "
                  f"{hand.mean() * 100:5.1f}%  best: handoff {top['handoff']!s:5s} "
                  f"t {top['t_handoff'] or 0:4.2f}s  pose {top['best_pose']:.2f}  "
                  f"rate {top['peak_rate']:4.1f}  down {top['max_down_v']:+.2f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
        summary[name] = {k: v for k, v in top.items() if k != "knots"}
        saved[name] = top["knots"]

    np.savez(ROOT / "results" / "getup_search.npz", knot_s=KNOT_S, **saved)
    json.dump(summary, open(ROOT / "results" / "getup_search.json", "w"), indent=2)

    cpu = mujoco.MjModel.from_xml_string(build_xml(aux["p"], mode="torque"))
    d = mujoco.MjData(cpu)
    ren = mujoco.Renderer(cpu, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = cpu.body("torso").id
    cam.distance, cam.elevation, cam.azimuth = 1.1, -15.0, 70.0
    for j, name in enumerate(POSES):
        qpos, done = (np.array(v) for v in trj(starts[j], jnp.asarray(saved[name], dtype=jnp.float32)))
        end = int(np.argmax(done > 0.5)) if (done > 0.5).any() else len(done) - 1
        out = ROOT / "results" / f"getup_search_{name}.mp4"
        with imageio.get_writer(out, fps=50) as w:
            for k in range(0, end + 1, 2):
                d.qpos[:] = qpos[k]
                mujoco.mj_forward(cpu, d)
                ren.update_scene(d, cam)
                w.append_data(ren.render())

    print("\npose   handoff  time    pose  peak rate  max down v")
    for name, s in summary.items():
        print(f"{name:6s} {str(s['handoff']):7s}  {s['t_handoff'] or 0:4.2f}s  {s['best_pose']:4.2f}  "
              f"{s['peak_rate']:6.1f}r/s  {s['max_down_v']:+7.2f}m/s")
    print("\npose: best of upright x height progress at the same moment, 1.0 = upright at 0.18 m")
    print("wrote results/getup_search.json, results/getup_search.npz and getup_search_<pose>.mp4")


if __name__ == "__main__":
    main()