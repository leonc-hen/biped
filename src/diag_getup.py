"""S4 diagnosis: how far the get up policy gets from each start pose, plus one video per pose.
Run after training stops: python src/diag_getup.py --run s4_a
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import argparse
import pickle
import sys

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
POSES = ["back", "front", "side"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    sg = pickle.load(open(ROOT / "runs" / args.run / "ckpt.pkl", "rb"))
    cfg = dict(CONFIGS[sg["cfg"].get("env", "s4")])
    cfg.update(sg["cfg"].get("env_overrides", {}))
    cfg.update(dr=False, getup_fall_frac=0.0, getup_demo_frac=0.0)
    reset, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    N = round(cfg["episode_s"] / DT)

    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(5), args.n))
    pose = np.array(states.dr["pose"])

    @jax.jit
    def roll(states):
        def one(s):
            def body(s, _):
                m, _, _ = net.apply(sg["params"], s.obs)
                s = step(s, m)
                return s, (s.data.qpos, s.done)
            return jax.lax.scan(body, s, None, length=N)[1]
        return jax.vmap(one)(states)

    print(f"{args.n} episodes of {args.run}, nominal (compiles for a few minutes)", flush=True)
    qpos, done = (np.array(x) for x in roll(states))
    up = 1 - 2 * (qpos[..., 4] ** 2 + qpos[..., 5] ** 2)
    z = qpos[..., 2]
    first = np.where(done.any(1), done.argmax(1), N - 1)
    live = np.arange(N)[None, :] <= first[:, None]
    handoff = first < N - 1
    last = slice(N - 100, N)

    print(f"\n{'pose':6s} {'handoff':>8s} {'max up':>7s} {'max z':>6s} "
          f"{'up at 8s':>9s} {'z at 8s':>8s} {'t up>0.5':>9s}")
    for j, name in enumerate(POSES):
        m = pose == j
        stuck = m & ~handoff
        mu = np.median(np.where(live, up, -2).max(1)[m])
        mz = np.median(np.where(live, z, 0).max(1)[m])
        fu = np.median(up[stuck][:, last].mean(1)) if stuck.any() else float("nan")
        fz = np.median(z[stuck][:, last].mean(1)) if stuck.any() else float("nan")
        tu = np.mean((np.where(live, up, -2)[m] > 0.5).sum(1) * DT)
        print(f"{name:6s} {handoff[m].mean() * 100:7.1f}% {mu:+7.2f} {mz:6.3f} "
              f"{fu:+9.2f} {fz:8.3f} {tu:8.2f}s")
    print("\nup: +1 standing, 0 flat, -1 inverted.  z: torso height, m (handoff needs 0.18).")
    print("max = best reached in the episode (median over episodes); at 8s = last second, "
          "episodes that never handed off; t up>0.5 = mean seconds spent tilted less than 60 deg")

    cpu = mujoco.MjModel.from_xml_string(build_xml(aux["p"], mode="torque"))
    d = mujoco.MjData(cpu)
    ren = mujoco.Renderer(cpu, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = cpu.body("torso").id
    cam.distance, cam.elevation, cam.azimuth = 1.1, -15.0, 70.0
    for j, name in enumerate(POSES):
        i = int(np.where(pose == j)[0][0])
        out = ROOT / "results" / f"getup_{args.run}_{name}.mp4"
        with imageio.get_writer(out, fps=50) as w:
            for k in range(0, int(first[i]) + 1, 2):
                d.qpos[:] = qpos[i, k]
                mujoco.mj_forward(cpu, d)
                ren.update_scene(d, cam)
                w.append_data(ren.render())
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()