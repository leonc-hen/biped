"""Why are the leg motors saturated? Rolls a policy on the nominal robot (no
randomization), standing at the nominal height and at the top of the range, and
reports per motor: torque, time at the torque limit, PD tracking error, the
policy's action, and how close each joint sits to its limits.
Run from the repo root: python src/diag_saturation.py --run s2r2_b
"""
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import argparse
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env_s1 import make_env, CONFIGS, LEG_Q, LEG_CTRL  # noqa: E402
from src.ppo import ActorCritic  # noqa: E402

JOINTS = ["hip_l", "knee_l", "hip_r", "knee_r"]
MOTORS = ["m1_l hip", "m2_l belt", "m1_r hip", "m2_r belt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    with open(ROOT / "runs" / args.run / "ckpt.pkl", "rb") as f:
        saved = pickle.load(f)
    params = saved["params"]
    cfg = dict(CONFIGS[saved["cfg"].get("env", "s1")])
    cfg.update(saved["cfg"].get("env_overrides", {}))
    cfg.update(push_every_s=None, resample_s=None, dr=False)
    _, step, sizes, aux = make_env(cfg)
    net = ActorCritic(sizes["act"])
    p, model = aux["p"], aux["model"]
    peak = aux["lm"]["peak_torque_Nm"]
    jr = p["joints"]
    lo = np.array([jr["hip"]["range_rad"][0], jr["knee"]["range_rad"][0]] * 2)
    hi = np.array([jr["hip"]["range_rad"][1], jr["knee"]["range_rad"][1]] * 2)

    @jax.jit
    def rollout(state):
        def body(st, _):
            mean, _, _ = net.apply(params, st.obs)
            st = step(st, mean)
            d = st.data
            tgt = aux["ik"](st.cmd[2]) + 0.5 * st.prev_action[:4]
            err = aux["to_motor"](tgt) - aux["to_motor"](d.qpos[LEG_Q])
            return st, (d.qpos[LEG_Q], d.ctrl[LEG_CTRL], err, jnp.tanh(mean[:4]))
        _, out = jax.lax.scan(body, state, None, length=500)
        return out

    for h in (aux["h_nom"], aux["h_range"][1]):
        data = mjx.forward(model, mjx.make_data(model).replace(
            qpos=aux["qpos0"], qvel=jnp.zeros(model.nv)))
        st = aux["init_from_data"](data, jnp.array([0.0, 0.0, h], dtype=jnp.float32))
        q, tau, err, act = (np.array(x)[100:] for x in rollout(st))
        print(f"\nstanding, commanded height {h:.3f} m (last 4 s, no randomization)")
        for j in range(4):
            margin = min(q[:, j].min() - lo[j], hi[j] - q[:, j].max())
            print(f"  {JOINTS[j]:6s} q {q[:, j].mean():+.3f} rad, margin to limit {margin:+.3f}   "
                  f"{MOTORS[j]:9s} |tau| {np.abs(tau[:, j]).mean():4.2f} N m, "
                  f"at limit {np.mean(np.abs(tau[:, j]) >= 0.95 * peak) * 100:3.0f}%, "
                  f"PD error {err[:, j].mean():+.3f} rad, action {act[:, j].mean():+.2f}")


if __name__ == "__main__":
    main()