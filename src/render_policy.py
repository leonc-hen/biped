"""Watch a trained policy. Runs a scripted demo in the training simulation (MJX, no
randomization), then replays it in CPU MuJoCo with a tracking camera: a live window
with --live, otherwise an MP4. Run from the repo root:
python src/render_policy.py --run s3v9 [--live]
"""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))

import argparse
import pickle
import sys
import time
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402
from model.build_mjcf import build_xml  # noqa: E402

DT = 0.01
T_TOTAL = 12.0
# (start s, forward m/s, yaw rad/s, hip height m)
SCHEDULE = [(0.0, 0.0, 0.0, 0.25), (1.5, 0.8, 0.0, 0.25), (4.0, 0.0, 0.0, 0.25),
            (5.0, 0.0, 1.0, 0.25), (6.5, 0.0, 0.0, 0.15), (7.3, 0.0, 0.0, 0.30),
            (8.0, 0.0, 0.0, 0.25)]
PUSH_AT, PUSH_DV = 8.5, (0.0, 0.4)        # sideways 0.4 m/s (R07)
JUMP_AT = 9.5


def render_xml(p):
    """The training model plus a checkered floor, a light and a video sized offscreen
    buffer, visuals only."""
    root = ET.fromstring(build_xml(p, mode="torque"))
    ET.SubElement(ET.SubElement(root, "visual"), "global", offwidth="960", offheight="540")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1=".85 .85 .85", rgb2=".65 .65 .65", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="20 20")
    floor = root.find(".//geom[@name='floor']")
    floor.set("material", "grid")
    floor.attrib.pop("rgba", None)
    ET.SubElement(root.find("worldbody"), "light", pos="0 -1 3", dir="0 0.3 -1",
                  diffuse=".8 .8 .8")
    return ET.tostring(root, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--live", action="store_true", help="show a window instead of writing an MP4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    saved = pickle.load(open(ROOT / "runs" / args.run / "ckpt.pkl", "rb"))
    env_name = saved["cfg"].get("env", "s1")
    cfg = dict(CONFIGS[env_name])
    cfg.update(saved["cfg"].get("env_overrides", {}))
    cfg.update(push_every_s=None, resample_s=None, dr=False)
    _, step, sizes, aux = make_env(cfg)
    net, params = ActorCritic(sizes["act"]), saved["params"]
    n = round(T_TOTAL / DT)

    cmds = np.zeros((n, 3), dtype=np.float32)
    for t0, vx, yaw, h in SCHEDULE:
        cmds[round(t0 / DT):] = (vx, yaw, h)
    push = np.zeros((n, 2), dtype=np.float32)
    push[round(PUSH_AT / DT)] = PUSH_DV

    model = aux["model"]
    data = mjx.forward(model, mjx.make_data(model).replace(qpos=aux["qpos0"], qvel=jnp.zeros(model.nv)))
    st = aux["init_from_data"](data, jnp.array(cmds[0]))
    if cfg.get("jump"):
        st = st._replace(jump_at=jnp.asarray(round(JUMP_AT / DT), dtype=jnp.int32))

    @jax.jit
    def run(st, cmds, push):
        def body(s, x):
            c, dv = x
            d = s.data
            s = s._replace(cmd=c, data=d.replace(qvel=d.qvel.at[0:2].add(dv)))
            mean, _, _ = net.apply(params, s.obs)
            s = step(s, mean)
            return s, (s.data.qpos, s.done)
        return jax.lax.scan(body, st, (cmds, push))

    print(f"simulating {T_TOTAL:.0f} s of {args.run} (first run compiles for a minute or two)", flush=True)
    _, (qpos, done) = run(st, jnp.array(cmds), jnp.array(push))
    qpos, done = np.array(qpos), np.array(done)
    first = int(np.argmax(done > 0.5)) if (done > 0.5).any() else None
    if first is not None and first < sizes["max_steps"] - 1:
        print(f"note: the robot fell at {first * DT:.2f} s")

    cpu = mujoco.MjModel.from_xml_string(render_xml(aux["p"]))
    d = mujoco.MjData(cpu)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = cpu.body("torso").id
    cam.distance, cam.elevation, cam.azimuth = 1.4, -12.0, 70.0

    if args.live:
        with mujoco.viewer.launch_passive(cpu, d) as v:
            v.cam.type, v.cam.trackbodyid = cam.type, cam.trackbodyid
            v.cam.distance, v.cam.elevation, v.cam.azimuth = cam.distance, cam.elevation, cam.azimuth
            while v.is_running():
                for q in qpos:
                    t0 = time.perf_counter()
                    d.qpos[:] = q
                    mujoco.mj_forward(cpu, d)
                    v.sync()
                    if not v.is_running():
                        break
                    time.sleep(max(0.0, DT - (time.perf_counter() - t0)))
                time.sleep(1.0)            # pause, then loop the demo
        return

    import imageio
    out = Path(args.out) if args.out else ROOT / "results" / f"demo_{args.run}.mp4"
    renderer = mujoco.Renderer(cpu, height=540, width=960)
    with imageio.get_writer(out, fps=50) as w:
        for q in qpos[::2]:                    # 100 Hz policy steps, 50 fps video
            d.qpos[:] = q
            mujoco.mj_forward(cpu, d)
            renderer.update_scene(d, camera=cam)
            w.append_data(renderer.render())
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()