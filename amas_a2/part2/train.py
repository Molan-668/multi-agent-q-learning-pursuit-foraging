"""
Part 2: Training MARL agents (IQL) on the custom Level-Based Foraging environment
with two different reward functions.
"""

import numpy as np
import torch
from amas_a2.part1.agent import IQLAgent
from amas_a2.part2.foraging_env import (
    LevelBasedForaging,
    make_global_reward,
    make_mixed_reward,
    set_reward_function,
)


def evaluate_iql_foraging(agents, env, n_episodes=5):
    """Evaluate IQL agents on foraging env without exploration."""
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        while env.agents and steps < env.max_steps * 2:
            active_agents = list(env.agents)
            actions = {}
            for aid in active_agents:
                agent_idx = int(aid.split("_")[-1])
                actions[aid] = agents[agent_idx].act(obs[aid], eval_mode=True)
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for aid in active_agents:
                episode_reward += rewards.get(aid, 0.0)
            if all(terminations.values()) or all(truncations.values()):
                break
            steps += 1
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def train_iql_foraging(
    reward_fn="global",
    grid_size=10,
    n_agents=3,
    n_crops=5,
    max_level=5,
    max_steps=200,
    static=False,
    n_episodes=3000,
    eval_interval=100,
    eval_episodes=5,
    lr=1e-4,
    gamma=0.99,
    batch_size=64,
    target_update_freq=500,
    epsilon_decay=0.9995,
    device="cpu",
    seed=None,
    verbose=True,
):
    """Train IQL agents on the foraging environment."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment
    env = LevelBasedForaging(
        grid_size=grid_size,
        n_agents=n_agents,
        n_crops=n_crops,
        max_level=max_level,
        max_steps=max_steps,
        static=static,
    )

    # Set reward function
    if reward_fn == "global":
        r_fn = make_global_reward(grid_size, n_agents, n_crops, max_steps)
    elif reward_fn == "mixed":
        r_fn = make_mixed_reward(grid_size, n_agents, n_crops, max_steps)
    else:
        raise ValueError(f"Unknown reward function: {reward_fn}")
    set_reward_function(env, r_fn)

    obs_shape = env.observation_space(env.possible_agents[0]).shape  # (grid, grid, 3)
    n_actions = env.action_space(env.possible_agents[0]).n

    # Create IQL agents
    agents = []
    for i in range(n_agents):
        agent = IQLAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            agent_id=i,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            epsilon_decay=epsilon_decay,
            device=device,
        )
        agents.append(agent)

    eval_episodes_list = []
    eval_rewards_list = []
    episode_rewards = []
    losses = []

    obs, _ = env.reset()
    total_steps = 0
    episode = 1

    while episode <= n_episodes:
        episode_reward = 0
        episode_loss_sum = 0
        episode_update_count = 0

        while env.agents:
            # Save current agents list BEFORE step (step may clear it)
            active_agents = list(env.agents)
            actions = {}
            for aid in active_agents:
                agent_idx = int(aid.split("_")[-1])
                actions[aid] = agents[agent_idx].act(obs[aid])

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Store experiences for all agents that acted
            stored_agents = list(actions.keys())
            for aid in stored_agents:
                agent_idx = int(aid.split("_")[-1])
                done = terminations.get(aid, False) or truncations.get(aid, False)
                agents[agent_idx].replay_buffer.push(
                    obs[aid], actions[aid], rewards.get(aid, 0.0), 
                    next_obs[aid], done
                )

            for aid in stored_agents:
                agent_idx = int(aid.split("_")[-1])
                loss_val = agents[agent_idx].update()
                if loss_val is not None:
                    episode_loss_sum += loss_val
                    episode_update_count += 1

            for aid in stored_agents:
                episode_reward += rewards.get(aid, 0.0)
            total_steps += 1

            if all(terminations.values()) or all(truncations.values()):
                break

            obs = next_obs

        # Episode finished
        episode_rewards.append(episode_reward)
        if episode_update_count > 0:
            losses.append(episode_loss_sum / max(1, episode_update_count))

        if episode % eval_interval == 0:
            mean_reward, std_reward = evaluate_iql_foraging(agents, env, eval_episodes)
            eval_episodes_list.append(episode)
            eval_rewards_list.append(mean_reward)
            if verbose:
                print(
                    f"[Foraging-{reward_fn}] Ep {episode:4d}/{n_episodes} | "
                    f"Train Reward: {episode_reward:.2f} | "
                    f"Eval Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                    f"Eps: {agents[0].epsilon:.3f} | "
                    f"Steps: {total_steps}"
                )

        obs, _ = env.reset()
        episode += 1

    return {
        "eval_episodes": np.array(eval_episodes_list),
        "eval_rewards": np.array(eval_rewards_list),
        "episode_rewards": np.array(episode_rewards),
        "losses": np.array(losses),
        "total_steps": total_steps,
    }
