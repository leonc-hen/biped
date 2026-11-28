"""Can the S3 action space produce a 10 cm jump at all? On a vertical rail (as in
src/jump_feasibility.py: base pinned in x, y and rotation after every policy step,
free in z), so balance is taken out. Runs the scripted jump primitive with zero
residual for several push target knee angles, to pick the one that gives about
14 cm. Run from the repo root."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_s1 import make_env, CONFIGS  # noqa: E402

# action order: hip_l, knee_l, hip_r, knee_r, wheel_l, wheel_r
CROUCH = jnp.array([-3.0, 3.0, -3.0, 3.0, 0.0, 0.0])
EXTEND = jnp.array([3.0, -3.0, 3.0, -3.0, 0.0, 0.0])


def rail(st):
    d = st.data
    return st._replace(data=d.replace(
        qpos=d.qpos.at[0:2].set(0.0).at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0])),
        qvel=d.qvel.at[0:2].set(0.0).at[3:6].set(0.0)))


for k_ext in (1.25,):
    label = f"push to knee {k_ext:.2f} rad"
    cfg = dict(CONFIGS["s3"], dr=False, push_every_s=None, resample_s=None,
               cmd_fixed=(0.0, 0.0, 0.25), reset_perturb=False, jump_ext_knee=k_ext)
    reset, step, _, _ = make_env(cfg)
    st = reset(jax.random.PRNGKey(0))
    st = st._replace(jump_at=jnp.asarray(1, dtype=jnp.int32))
    step = jax.jit(step)
    com0, best = float(st.data.subtree_com[1, 2]), 0.0
    for t in range(140):
        st = rail(step(st, jnp.zeros(6)))
        best = max(best, float(st.data.subtree_com[1, 2]) - com0)
    print(f"{label:22s} scripted jump: CoM rise {best * 100:.1f} cm, "
          f"wheel clearance {float(st.jump_apex) * 100:.1f} cm")
