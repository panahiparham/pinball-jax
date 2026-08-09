"""Benchmark environment and agent throughput: numpy vs. JAX, single vs. vmapped.

Measures steps/second for:
- Numpy Pinball with random actions in a plain Python loop
- JAX Pinball with random actions in jitted scan, single and vmapped
- DQN and random agents on JAX Pinball

Run with::

    uv run --group benchmark python benchmark_throughput.py
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as bm
from pinball_jax import Pinball, PinballParams

# The numpy baseline is the vendored reference implementation that also serves
# as the test oracle, so there is one copy of it in the repo.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from _reference_pinball import PinballModel

SETTING = "easy"
EPISODE_CUTOFF = 1_000
CONFIGS_DIR = Path(__file__).parent / "src" / "pinball_jax" / "configs"
CONFIG_PATH = CONFIGS_DIR / f"{SETTING}.cfg"

env_jax = Pinball(SETTING)
env_params = PinballParams(max_steps_in_episode=EPISODE_CUTOFF)
ACTION_DIM = env_jax.action_space(env_params).n
OBS_DIM = int(np.prod(env_jax.observation_space(env_params).shape))


def format_throughput(steps, elapsed_s):
    """Format throughput with thousands separators and appropriate precision."""
    throughput = steps / elapsed_s
    if throughput >= 100_000:
        return f"{int(round(throughput / 1000)) * 1000:,}"
    elif throughput >= 10_000:
        return f"{int(round(throughput / 100)) * 100:,}"
    else:
        return f"{int(round(throughput)):,}"


def format_speedup(speedup):
    """Format speedup with appropriate precision."""
    if speedup >= 10:
        return f"{int(round(speedup))}x"
    else:
        return f"{speedup:.1f}x"


def benchmark_numpy(num_steps):
    """Benchmark the numpy reference in a plain Python loop."""
    env = PinballModel(str(CONFIG_PATH), np.random.default_rng(0))
    rng = np.random.default_rng(0)

    for _ in range(num_steps):
        env.take_action(int(rng.integers(0, ACTION_DIM)))

    return num_steps


def benchmark_jax_single(num_steps):
    """Benchmark JAX Pinball, single env, in jitted scan with random policy.

    Excludes compilation time via warmup, includes blocking time.
    """
    train_fn = partial(bm.random_train, env=env_jax, env_params=env_params,
                       action_dim=ACTION_DIM, total_timesteps=num_steps)

    rng = jax.random.key(0)
    jitted_fn = jax.jit(train_fn)

    jax.block_until_ready(jitted_fn(rng))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_fn(rng))
    elapsed = time.perf_counter() - t_start

    return num_steps, elapsed


def benchmark_jax_vmapped(num_envs, num_steps):
    """Benchmark JAX Pinball, vmapped over num_envs envs, in jitted scan.

    Excludes compilation time via warmup, includes blocking time.
    """
    train_fn = partial(bm.random_train, env=env_jax, env_params=env_params,
                       action_dim=ACTION_DIM, total_timesteps=num_steps)

    jitted_vmapped = jax.jit(jax.vmap(train_fn))
    keys = jax.random.split(jax.random.key(0), num_envs)

    jax.block_until_ready(jitted_vmapped(keys))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_vmapped(keys))
    elapsed = time.perf_counter() - t_start

    return num_envs * num_steps, elapsed


def benchmark_agent(num_steps, n_seeds, agent_name):
    """Benchmark an agent (random or DQN) for throughput.

    Excludes compilation time via warmup, includes blocking time.
    """
    if agent_name == "Random":
        train_fn = partial(
            bm.random_train, env=env_jax, env_params=env_params,
            action_dim=ACTION_DIM, total_timesteps=num_steps,
        )
    else:
        train_fn = partial(
            bm.dqn_train, env=env_jax, env_params=env_params, obs_dim=OBS_DIM,
            action_dim=ACTION_DIM, total_timesteps=num_steps,
        )

    keys = jax.vmap(jax.random.key)(jnp.arange(n_seeds))
    jitted_fn = jax.jit(jax.vmap(train_fn))

    jax.block_until_ready(jitted_fn(keys))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_fn(keys))
    elapsed = time.perf_counter() - t_start

    return n_seeds * num_steps, elapsed


def main():
    """Runs both throughput tables and prints them as markdown."""
    print("\n" + "=" * 80)
    print("THROUGHPUT BENCHMARKS: numpy vs. JAX Pinball")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    numpy_steps = 10_000
    print(f"Benchmarking numpy Pinball with {numpy_steps:,} steps...")
    t_start = time.perf_counter()
    benchmark_numpy(numpy_steps)
    numpy_elapsed = time.perf_counter() - t_start
    numpy_throughput = numpy_steps / numpy_elapsed
    print(f"  Numpy: {format_throughput(numpy_steps, numpy_elapsed)} steps/sec")
    print()

    print("Table 1: Environment Throughput (Random Policy)")
    print("-" * 80)

    rows = [{
        "impl": "reference (numpy)",
        "n_envs": 1,
        "throughput": format_throughput(numpy_steps, numpy_elapsed),
        "speedup": "1x",
    }]

    jax_single_steps = 100_000
    print(f"Benchmarking JAX single env with {jax_single_steps:,} steps...")
    jax_single_total, jax_single_elapsed = benchmark_jax_single(jax_single_steps)
    jax_single_throughput = jax_single_total / jax_single_elapsed
    jax_single_speedup = jax_single_throughput / numpy_throughput
    throughput_str = format_throughput(jax_single_total, jax_single_elapsed)
    speedup_str = format_speedup(jax_single_speedup)
    print(f"  JAX 1x1: {throughput_str} steps/sec ({speedup_str})")
    rows.append({
        "impl": "pinball-jax", "n_envs": 1,
        "throughput": throughput_str, "speedup": speedup_str,
    })

    # Steps/sec is a rate, independent of step count once compiled, so step count
    # per config is chosen to keep each config's wall time bounded (collision
    # detection against several polygon obstacles doesn't vectorize as cheaply
    # across a wide vmap batch as e.g. a simple grid update, so larger n_envs
    # need proportionally fewer steps to finish in a reasonable time).
    vmapped_configs = [(8, 100_000), (64, 20_000), (512, 4_000), (4096, 1_000)]

    for n_envs, steps in vmapped_configs:
        print(f"Benchmarking JAX vmapped {n_envs} envs with {steps:,} steps per env...")
        total_steps, elapsed = benchmark_jax_vmapped(n_envs, steps)
        throughput = total_steps / elapsed
        throughput_str = format_throughput(total_steps, elapsed)
        speedup_str = format_speedup(throughput / numpy_throughput)
        print(f"  JAX {n_envs}x: {throughput_str} steps/sec ({speedup_str})")
        rows.append({
            "impl": "pinball-jax", "n_envs": n_envs,
            "throughput": throughput_str, "speedup": speedup_str,
        })

    print()
    print("| Implementation | Environments | Steps/sec | Speedup vs. numpy |")
    print("|---|---|---|---|")
    for row in rows:
        cells = [row["impl"], row["n_envs"], row["throughput"], row["speedup"]]
        print("| " + " | ".join(str(c) for c in cells) + " |")

    print()
    print("Table 2: Agent Throughput on pinball-jax")
    print("-" * 80)

    agent_results = {}
    agent_steps = 50_000
    for name in ["Random", "DQN"]:
        print(f"Benchmarking {name} agent with {agent_steps:,} steps...")
        total_steps, elapsed = benchmark_agent(agent_steps, 1, name)
        print(f"  {name}: {format_throughput(total_steps, elapsed)} steps/sec")
        agent_results[name] = (total_steps, elapsed)

    print()
    print("| Agent | Steps/sec |")
    print("|---|---|")
    for name in ["Random", "DQN"]:
        total_steps, elapsed = agent_results[name]
        print(f"| {name} | {format_throughput(total_steps, elapsed)} |")

    print()
    print("=" * 80)
    print("Benchmarking complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
