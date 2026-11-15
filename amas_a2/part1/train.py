"""
Part 1: Training and evaluation pipeline for IQL and CQL on Pursuit.

Key design:
- Training: agents interact with the environment, store experiences, learn
- Evaluation: every N episodes, run eval episodes with epsilon=0, record cumulative reward
- Averaging: train with 3-5 seeds per algorithm, average the eval curves
- Visualization: produce line charts (x=episode, y=cumulative reward)
"""

import numpy as np
import torch
from pettingzoo.sisl import pursuit_v4
from collections import defaultdict
import time

from amas_a2.part1.agent import IQLAgent, CQLAgent


class PursuitEnvWrapper:
    """Wrapper to simplify the pursuit parallel env interface."""

    def __init__(self, **kwargs):
        self.env = pursuit_v4.parallel_env(**kwargs)
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)

    def reset(self):
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        return self.env.step(actions)

    @property
    def action_space(self):
        return self.env.action_space(self.agents[0])

    @property
    def observation_space(self):
        return self.env.observation_space(self.agents[0])

    @property
    def n_actions(self):
        return self.env.action_space(self.agents[0]).n

    def close(self):
        self.env.close()


def evaluate_iql(agents, env, n_episodes=5):
    """Evaluate IQL agents without exploration."""
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        while env.agents and steps < 1000:
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


def evaluate_cql(agent, env, n_episodes=5):
    """Evaluate CQL agents without exploration."""
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        while env.agents and steps < 1000:
            active_agents = list(env.agents)
            actions = {}
            for aid in active_agents:
                actions[aid] = agent.act(obs, aid, eval_mode=True)
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for aid in active_agents:
                episode_reward += rewards.get(aid, 0.0)
            if all(terminations.values()) or all(truncations.values()):
                break
            steps += 1
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def train_iql(
    env_config=None,
    n_episodes=2000,
    eval_interval=50,
    eval_episodes=5,
    lr=1e-4,
    gamma=0.99,
    buffer_size=100000,
    batch_size=64,
    target_update_freq=500,
    epsilon_decay=0.9995,
    device="cpu",
    seed=None,
    verbose=True,
):
    """Train IQL agents and return evaluation results."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if env_config is None:
        env_config = {}

    env = PursuitEnvWrapper(**env_config)
    obs_shape = env.observation_space.shape
    n_actions = env.n_actions
    n_agents = env.n_agents

    agents = []
    for i in range(n_agents):
        agent = IQLAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            agent_id=i,
            lr=lr,
            gamma=gamma,
            buffer_size=buffer_size,
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
        # Run one full episode (multiple env steps until done)
        episode_reward = 0
        episode_loss_sum = 0
        episode_update_count = 0

        while env.agents:
            # Save agents list before step (step may clear it)
            active_agents = list(env.agents)
            actions = {}
            for aid in active_agents:
                agent_idx = int(aid.split("_")[-1])
                actions[aid] = agents[agent_idx].act(obs[aid])

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            for aid in active_agents:
                agent_idx = int(aid.split("_")[-1])
                done = terminations.get(aid, False) or truncations.get(aid, False)
                agents[agent_idx].replay_buffer.push(
                    obs[aid], actions[aid], rewards.get(aid, 0.0), next_obs[aid], done
                )

            for aid in active_agents:
                agent_idx = int(aid.split("_")[-1])
                loss_val = agents[agent_idx].update()
                if loss_val is not None:
                    episode_loss_sum += loss_val
                    episode_update_count += 1

            for aid in active_agents:
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
            mean_reward, std_reward = evaluate_iql(agents, env, eval_episodes)
            eval_episodes_list.append(episode)
            eval_rewards_list.append(mean_reward)
            if verbose:
                print(
                    f"[IQL] Ep {episode:4d}/{n_episodes} | "
                    f"Train Reward: {episode_reward:.2f} | "
                    f"Eval Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                    f"Eps: {agents[0].epsilon:.3f} | "
                    f"Steps: {total_steps}"
                )

        obs, _ = env.reset()
        episode += 1

    env.close()

    return {
        "eval_episodes": np.array(eval_episodes_list),
        "eval_rewards": np.array(eval_rewards_list),
        "episode_rewards": np.array(episode_rewards),
        "losses": np.array(losses),
        "total_steps": total_steps,
    }


def train_cql(
    env_config=None,
    n_episodes=2000,
    eval_interval=50,
    eval_episodes=5,
    lr=1e-4,
    gamma=0.99,
    buffer_size=100000,
    batch_size=64,
    target_update_freq=500,
    epsilon_decay=0.9995,
    device="cpu",
    seed=None,
    verbose=True,
):
    """Train CQL and return evaluation results."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if env_config is None:
        env_config = {}

    env = PursuitEnvWrapper(**env_config)
    obs_shape = env.observation_space.shape
    n_actions = env.n_actions
    n_agents = env.n_agents

    agent = CQLAgent(
        obs_shape=obs_shape,
        n_agents=n_agents,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        epsilon_decay=epsilon_decay,
        device=device,
    )

    eval_episodes_list = []
    eval_rewards_list = []
    episode_rewards = []
    losses = []

    obs, _ = env.reset()
    total_steps = 0
    episode = 1

    while episode <= n_episodes:
        episode_reward = 0
        episode_loss_val = None

        while env.agents:
            active_agents = list(env.agents)
            actions = {}
            for aid in active_agents:
                actions[aid] = agent.act(obs, aid)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            dones_dict = {aid: terminations.get(aid, False) or truncations.get(aid, False) for aid in active_agents}
            agent.store_experience(obs, actions, rewards, next_obs, dones_dict)

            loss_val = agent.update()

            for aid in active_agents:
                episode_reward += rewards.get(aid, 0.0)
            total_steps += 1
            if loss_val is not None:
                episode_loss_val = loss_val

            if all(terminations.values()) or all(truncations.values()):
                break

            obs = next_obs

        # Episode finished
        episode_rewards.append(episode_reward)
        if episode_loss_val is not None:
            losses.append(episode_loss_val)

        if episode % eval_interval == 0:
            mean_reward, std_reward = evaluate_cql(agent, env, eval_episodes)
            eval_episodes_list.append(episode)
            eval_rewards_list.append(mean_reward)
            if verbose:
                print(
                    f"[CQL] Ep {episode:4d}/{n_episodes} | "
                    f"Train Reward: {episode_reward:.2f} | "
                    f"Eval Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Steps: {total_steps}"
                )

        obs, _ = env.reset()
        episode += 1

    env.close()

    return {
        "eval_episodes": np.array(eval_episodes_list),
        "eval_rewards": np.array(eval_rewards_list),
        "episode_rewards": np.array(episode_rewards),
        "losses": np.array(losses),
        "total_steps": total_steps,
    }
