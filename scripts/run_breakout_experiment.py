from __future__ import annotations

import argparse
import pathlib
import sys
import time

import ale_py
import gymnasium as gym
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_priors_ai import CorePriorActiveInferenceAgent


def run_episode(
    env: gym.Env,
    agent: CorePriorActiveInferenceAgent,
    max_steps: int,
    seed: int,
    episode_idx: int,
) -> tuple[float, int, float, float, float, float, float, float, float, float, float]:
    observation, info = env.reset(seed=seed + episode_idx)
    agent.reset()

    total_reward = 0.0
    running_efe = 0.0
    running_uncertainty = 0.0
    target_seen_ratio = 0.0
    running_object_prior = 0.0
    running_agent_prior = 0.0
    running_geometry_prior = 0.0
    running_number_prior = 0.0
    running_numerosity = 0.0
    running_number_surprise = 0.0
    step = 0

    for step in range(1, max_steps + 1):
        action, diagnostics = agent.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        running_efe += diagnostics.free_energy
        running_uncertainty += diagnostics.uncertainty
        target_seen_ratio += 1.0 if diagnostics.has_target else 0.0
        running_object_prior += diagnostics.object_prior
        running_agent_prior += diagnostics.agent_prior
        running_geometry_prior += diagnostics.geometry_prior
        running_number_prior += diagnostics.number_prior
        running_numerosity += diagnostics.numerosity_mean
        running_number_surprise += diagnostics.number_surprise
        agent.observe_transition(info)

        if terminated or truncated:
            break

    steps = step
    avg_efe = running_efe / max(steps, 1)
    avg_uncertainty = running_uncertainty / max(steps, 1)
    tracked_ratio = target_seen_ratio / max(steps, 1)
    avg_object_prior = running_object_prior / max(steps, 1)
    avg_agent_prior = running_agent_prior / max(steps, 1)
    avg_geometry_prior = running_geometry_prior / max(steps, 1)
    avg_number_prior = running_number_prior / max(steps, 1)
    avg_numerosity = running_numerosity / max(steps, 1)
    avg_number_surprise = running_number_surprise / max(steps, 1)
    return (
        total_reward,
        steps,
        avg_efe,
        avg_uncertainty,
        tracked_ratio,
        avg_object_prior,
        avg_agent_prior,
        avg_geometry_prior,
        avg_number_prior,
        avg_numerosity,
        avg_number_surprise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Core-prior active inference Breakout test")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    args = parser.parse_args()

    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Breakout-v5",
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    action_meanings = env.unwrapped.get_action_meanings()
    agent = CorePriorActiveInferenceAgent(
        action_meanings=action_meanings,
        seed=args.seed,
        horizon=args.horizon,
        num_samples=args.num_samples,
        elite_fraction=args.elite_fraction,
        iterations=args.iterations,
    )

    rewards: list[float] = []
    times: list[float] = []

    for episode_idx in range(args.episodes):
        start = time.time()
        (
            reward,
            steps,
            avg_efe,
            avg_uncertainty,
            tracked_ratio,
            avg_object_prior,
            avg_agent_prior,
            avg_geometry_prior,
            avg_number_prior,
            avg_numerosity,
            avg_number_surprise,
        ) = run_episode(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            seed=args.seed,
            episode_idx=episode_idx,
        )
        duration = time.time() - start
        rewards.append(reward)
        times.append(duration)
        print(
            f"episode={episode_idx + 1} reward={reward:.1f} steps={steps} avg_efe={avg_efe:.3f} "
            f"avg_uncertainty={avg_uncertainty:.3f} target_tracked={tracked_ratio:.2%} "
            f"obj={avg_object_prior:.3f} agent={avg_agent_prior:.3f} "
            f"geom={avg_geometry_prior:.3f} num={avg_number_prior:.3f} "
            f"n_mean={avg_numerosity:.2f} n_surprise={avg_number_surprise:.2f} sec={duration:.2f}"
        )

    rewards_np = np.asarray(rewards, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.float32)
    print(
        "final "
        f"mean_reward={rewards_np.mean():.3f} std_reward={rewards_np.std():.3f} "
        f"min_reward={rewards_np.min():.3f} max_reward={rewards_np.max():.3f} "
        f"mean_sec={times_np.mean():.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
