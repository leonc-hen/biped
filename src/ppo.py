"""Minimal PPO in JAX for the S1 environment family. Self contained on purpose:
the deployed actor must be tiny and fully understood, so the trainer is too.
flax networks, optax optimizer, pickle checkpoints with resume.

Known simplification: episode timeouts are treated as terminal in GAE (no
bootstrap through the time limit). Revisit if value estimates look biased.
"""
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorboardX import SummaryWriter


class ActorCritic(nn.Module):
    act_size: int

    @nn.compact
    def __call__(self, obs):
        h = obs
        for n in (128, 128):            # actor stays small, 2.10 export budget
            h = nn.tanh(nn.Dense(n)(h))
        mean = nn.Dense(self.act_size)(h)
        log_std = jnp.clip(
            self.param("log_std", nn.initializers.constant(-0.5), (self.act_size,)), -3.0, 0.0)
        v = obs
        for n in (256, 256):            # critic can be bigger, it never ships
            v = nn.tanh(nn.Dense(n)(v))
        return mean, log_std, nn.Dense(1)(v)[..., 0]


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def train(cfg, reset_fn, step_fn, sizes, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    rng = jax.random.PRNGKey(cfg["seed"])
    net = ActorCritic(sizes["act"])
    rng, k = jax.random.split(rng)
    params = net.init(k, jnp.zeros(sizes["obs"]))
    if cfg.get("init_from"):
        with open(cfg["init_from"], "rb") as f:
            params = pickle.load(f)["params"]
        print(f"initialized from {cfg['init_from']}")
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg["lr"]))
    opt_state = tx.init(params)
    B, T = cfg["num_envs"], cfg["unroll"]

    rng, k = jax.random.split(rng)
    env_states = jax.vmap(reset_fn)(jax.random.split(k, B))

    update_start = 0
    ckpt_path = run_dir / "ckpt.pkl"
    if cfg["resume"] and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            saved = pickle.load(f)
        params, opt_state, update_start, rng = (
            saved["params"], saved["opt_state"], saved["update"], saved["rng"])
        print(f"resumed from update {update_start}")

    def dist_logp(mean, log_std, a):
        return -0.5 * jnp.sum(((a - mean) / jnp.exp(log_std)) ** 2
                              + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)

    @jax.jit
    def collect(params, env_states, rng):
        def body(carry, _):
            st, r = carry
            r, ka, kr = jax.random.split(r, 3)
            mean, log_std, value = net.apply(params, st.obs)
            a = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            logp = dist_logp(mean, log_std, a)
            ns = jax.vmap(step_fn)(st, a)
            tr = Transition(st.obs, a, logp, value, ns.reward, ns.done)
            fresh = jax.vmap(reset_fn)(jax.random.split(kr, B))
            nxt = jax.tree.map(
                lambda f, n: jnp.where(
                    ns.done.reshape((B,) + (1,) * (n.ndim - 1)) > 0.5, f, n),
                fresh, ns)
            return (nxt, r), tr
        (st, r), traj = jax.lax.scan(body, (env_states, rng), None, length=T)
        _, _, last_v = net.apply(params, st.obs)
        return st, traj, last_v

    def gae(traj, last_v):
        def body(carry, tr):
            adv_next, v_next = carry
            nonterm = 1.0 - tr.done
            delta = cfg.get("reward_scale", 0.1) * tr.reward + cfg["gamma"] * v_next * nonterm - tr.value
            adv = delta + cfg["gamma"] * cfg["lam"] * nonterm * adv_next
            return (adv, tr.value), adv
        _, advs = jax.lax.scan(body, (jnp.zeros(B), last_v), traj, reverse=True)
        return advs, advs + traj.value

    def loss_fn(params, batch, adv, ret):
        mean, log_std, value = net.apply(params, batch.obs)
        logp = dist_logp(mean, log_std, batch.action)
        ratio = jnp.exp(logp - batch.logp)
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(
            ratio * a, jnp.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * a))
        vf = jnp.mean((value - ret) ** 2)
        ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        return pg + cfg["vf_coef"] * vf - cfg["ent_coef"] * ent, (pg, vf, ent)

    @jax.jit
    def update(params, opt_state, traj, last_v, rng):
        advs, rets = gae(traj, last_v)
        flat = jax.tree.map(lambda x: x.reshape((T * B,) + x.shape[2:]), traj)
        advs, rets = advs.reshape(-1), rets.reshape(-1)
        mb = T * B // cfg["minibatches"]

        def epoch(carry, _):
            params, opt_state, r = carry
            r, k = jax.random.split(r)
            perm = jax.random.permutation(k, T * B)

            def mini(carry, i):
                params, opt_state = carry
                sel = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                batch = jax.tree.map(lambda x: x[sel], flat)
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, batch, advs[sel], rets[sel])
                upd, opt_state = tx.update(g, opt_state)
                return (optax.apply_updates(params, upd), opt_state), aux
            (params, opt_state), aux = jax.lax.scan(
                mini, (params, opt_state), jnp.arange(cfg["minibatches"]))
            return (params, opt_state, r), aux
        (params, opt_state, _), aux = jax.lax.scan(
            epoch, (params, opt_state, rng), None, length=cfg["epochs"])
        return params, opt_state, jax.tree.map(jnp.mean, aux)

    steps_per_update = B * T
    n_updates = cfg["total_steps"] // steps_per_update
    history = []
    t0 = time.time()
    for u in range(update_start, n_updates):
        rng, kc, ku = jax.random.split(rng, 3)
        env_states, traj, last_v = collect(params, env_states, kc)
        params, opt_state, (pg, vf, ent) = update(params, opt_state, traj, last_v, ku)

        if (u + 1) % cfg["log_every"] == 0 or u == update_start:
            r_mean = float(traj.reward.mean())
            d_rate = float(traj.done.mean())
            ep_len = 1.0 / max(d_rate, 1e-4)
            sps = steps_per_update * (u + 1 - update_start) / (time.time() - t0)
            step_count = (u + 1) * steps_per_update
            for name, val in (("reward_per_step", r_mean), ("done_rate", d_rate),
                              ("ep_len_steps", ep_len), ("pg_loss", float(pg)),
                              ("vf_loss", float(vf)), ("entropy", float(ent)),
                              ("steps_per_s", sps)):
                writer.add_scalar(name, val, step_count)
            history.append((u + 1, r_mean, d_rate))
            print(f"update {u + 1:5d}/{n_updates}   steps {step_count / 1e6:6.2f}M   "
                  f"reward/step {r_mean:+.3f}   ep len {ep_len:7.1f}   {sps:,.0f} sps")

        if (u + 1) % cfg["ckpt_every"] == 0 or (u + 1) == n_updates:
            with open(ckpt_path, "wb") as f:
                pickle.dump(dict(params=params, opt_state=opt_state,
                                 update=u + 1, rng=rng, cfg=cfg), f)
        if (u + 1) % 250 == 0:
            with open(run_dir / f"ckpt_{u + 1:05d}.pkl", "wb") as f:
                pickle.dump(dict(params=params, opt_state=opt_state,
                                 update=u + 1, rng=rng, cfg=cfg), f)
    writer.close()
    return params, history