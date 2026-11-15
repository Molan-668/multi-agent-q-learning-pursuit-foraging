"""
Part 1: IQL and CQL agent implementations for the Pursuit environment.

IQL (Independent Q-Learning):
  Each agent independently learns its own Q-network, treating all other agents
  as part of the environment. Agents share network architecture but maintain
  separate parameters.

CQL (Centralized Q-Learning):
  A single global Q-network takes the concatenated observations of ALL agents
  as input and produces joint Q-values. During execution, each agent selects
  its action by marginalizing over other agents' actions (or using the
  centralized Q with argmax over its own actions given others' greedy actions).

In practice for discrete action spaces, CQL here means:
  - One Q-network with observation dimension = sum of all agent obs
  - Output dimension = per-agent action space (same actions for all)
  - Each agent's action selected by argmax over its own Q-values
  - This is actually a form of parameter sharing with centralized critic
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network with convolutional layers for Pursuit env obs (7x7x3)."""

    def __init__(self, obs_shape, n_actions, hidden_dim=256):
        super().__init__()
        c, h, w = obs_shape  # (3, 7, 7)
        # Small conv net for the 7x7 grid observations
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        # x shape: (batch, c, h, w)
        features = self.conv(x)
        return self.fc(features)


class CentralizedDQN(nn.Module):
    """
    Centralized Q-network for CQL.
    Takes concatenated observations from ALL agents as input.
    Outputs Q-values for each agent's actions.
    """

    def __init__(self, obs_shape, n_agents, n_actions, hidden_dim=512):
        super().__init__()
        c, h, w = obs_shape

        # Per-agent conv encoder - shared weights
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        # Centralized head - takes all agent features concatenated
        self.fc = nn.Sequential(
            nn.Linear(conv_out * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * n_agents),
        )
        self.n_agents = n_agents
        self.n_actions = n_actions

    def forward(self, x):
        # x shape: (batch, n_agents, c, h, w)
        batch_size = x.shape[0]
        n_agents = x.shape[1]

        # Encode each agent's observation independently (shared conv)
        x_reshaped = x.view(batch_size * n_agents, *x.shape[2:])
        features = self.conv(x_reshaped)  # (batch*n_agents, conv_out)
        features = features.view(batch_size, n_agents * features.shape[-1])

        # Centralized processing
        q_values = self.fc(features)  # (batch, n_actions * n_agents)
        q_values = q_values.view(batch_size, n_agents, self.n_actions)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class IQLAgent:
    """
    Independent Q-Learning agent.
    Each pursuer has its own DQN, trained on its local observations.
    """

    def __init__(
        self,
        obs_shape,
        n_actions,
        agent_id,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=500,
        tau=0.005,  # For soft update
        device="cpu",
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.device = device
        self.step_count = 0

        # Swap channels to first for PyTorch conv: HWC -> CHW
        self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        self.q_network = DQN(self.obs_shape, n_actions).to(device)
        self.target_network = DQN(self.obs_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _preprocess_obs(self, obs):
        """Convert HWC to CHW and add batch/device."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0)
        return obs.to(self.device)

    def act(self, obs, eval_mode=False):
        """Select action. If eval_mode, no exploration."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()

    def update(self):
        """Single gradient update step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        # states/next_states: (batch, h, w, c) -> permute to (batch, c, h, w)
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Target Q values (double DQN style to reduce overestimation)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return loss.item()


class CQLAgent:
    """
    Centralized Q-Learning agent.
    A single DQN takes ALL agents' observations and outputs per-agent Q-values.
    During training, the centralized Q is updated using all agents' experiences
    simultaneously. Each agent selects its action from its slice of the output.

    Note: In the Pursuit environment, all agents share identical action/obs spaces.
    """

    def __init__(
        self,
        obs_shape,
        n_agents,
        n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=500,
        tau=0.005,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.device = device
        self.step_count = 0

        # obs_shape is (h, w, c) from pettingzoo, convert to (c, h, w)
        self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        self.q_network = CentralizedDQN(
            self.obs_shape, n_agents, n_actions
        ).to(device)
        self.target_network = CentralizedDQN(
            self.obs_shape, n_agents, n_actions
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _preprocess_obs(self, obs_dict):
        """
        Convert dict of {agent_id: obs_array} to a single tensor.
        obs_dict keys are sorted by agent order.
        Returns: (1, n_agents, c, h, w)
        """
        # Sort agents to ensure consistent ordering
        agent_ids = sorted(obs_dict.keys())
        obs_list = []
        for aid in agent_ids:
            o = obs_dict[aid]
            # HWC -> CHW
            o_t = torch.FloatTensor(o).permute(2, 0, 1)
            obs_list.append(o_t)
        return torch.stack(obs_list, dim=0).unsqueeze(0).to(self.device)  # (1, n_agents, c, h, w)

    def act(self, obs_dict, agent_id, eval_mode=False):
        """
        Select action for a specific agent using the centralized Q.
        Agent_id is used to index into the Q output.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs_dict)
            all_q = self.q_network(obs_tensor)  # (1, n_agents, n_actions)
            # Find this agent's index
            agent_ids = sorted(obs_dict.keys())
            agent_idx = agent_ids.index(agent_id)
            q_agent = all_q[0, agent_idx]  # (n_actions,)
            return q_agent.argmax().item()

    def update(self):
        """
        Single centralized update using a batch of joint experiences.
        Each experience contains all agents' obs, actions, rewards.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(
            self.replay_buffer.buffer, self.batch_size
        )
        # Unpack: each entry is (obs_dict, action_dict, reward_dict, next_obs_dict, done_dict)
        # But we stored flattened vectors for simplicity

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        # states[i] is array of shape (n_agents, h, w, c)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        # Permute: (batch, n_agents, h, w, c) -> (batch, n_agents, c, h, w)
        states = states.permute(0, 1, 4, 2, 3)

        actions = torch.LongTensor(np.array(actions)).to(self.device)  # (batch, n_agents)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)  # (batch, n_agents)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)  # (batch, n_agents)

        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        next_states = next_states.permute(0, 1, 4, 2, 3)

        # Current Q values for all agents
        current_q = self.q_network(states)  # (batch, n_agents, n_actions)
        current_q = current_q.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (batch, n_agents)

        # Target Q values
        with torch.no_grad():
            # Double DQN: use online net for action selection, target for value
            next_q_online = self.q_network(next_states)  # (batch, n_agents, n_actions)
            next_actions = next_q_online.argmax(dim=2)  # (batch, n_agents)
            next_q_target = self.target_network(next_states)  # (batch, n_agents, n_actions)
            next_q = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)

            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return loss.item()

    def store_experience(self, obs_dict, action_dict, reward_dict, next_obs_dict, done_dict):
        """
        Store a joint experience.
        For efficiency, we store flattened arrays:
        - obs: (n_agents, h, w, c)
        - actions: (n_agents,)
        - rewards: (n_agents,)
        - next_obs: (n_agents, h, w, c)
        - dones: (n_agents,)
        """
        agent_ids = sorted(obs_dict.keys())
        n = len(agent_ids)

        obs_array = np.stack([obs_dict[aid] for aid in agent_ids], axis=0)
        action_array = np.array([action_dict[aid] for aid in agent_ids])
        reward_array = np.array([reward_dict[aid] for aid in agent_ids])
        next_obs_array = np.stack([next_obs_dict[aid] for aid in agent_ids], axis=0)
        done_array = np.array([done_dict[aid] for aid in agent_ids], dtype=np.float32)

        self.replay_buffer.push(
            obs_array, action_array, reward_array, next_obs_array, done_array
        )
