"""
Part 2: Level-Based Foraging Environment (PettingZoo Parallel API).

Environment Description (as per assignment):
- Grid world with crops (apples) and forager agents.
- Each agent can move N/S/E/W, harvest adjacent crop, or stay (6 actions).
- Each agent and crop has a skill level.
- Agents cooperate to harvest: sum of skill levels >= crop level to succeed.
- Goal: harvest all crops as quickly as possible.
- Agents randomized on reset (unless static mode is used).

Reward Functions (two candidates):
1. Global reward: shared reward based on global progress (crops remaining)
2. Mixed reward: global progress + local per-agent reward for successful harvests
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class LevelBasedForaging(ParallelEnv):
    """
    Level-Based Foraging environment with PettingZoo Parallel API.

    Key features:
    - Grid world with crops and foragers
    - Agents have skill levels, crops have difficulty levels
    - Cooperation required: sum(agent levels) >= crop level
    - 6 actions per agent: N, E, S, W, Harvest, Stay
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "level_based_foraging_v0"}

    def __init__(
        self,
        grid_size=10,
        n_agents=3,
        n_crops=5,
        max_level=5,
        max_steps=200,
        static=False,
        render_mode=None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_crops = n_crops
        self.max_level = max_level
        self.max_steps = max_steps
        self.static = static
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        # Action space: 0=North, 1=East, 2=South, 3=West, 4=Harvest, 5=Stay
        self._action_space = spaces.Discrete(6)
        # Observation: agent's grid-size channels + self features
        # Channels: agents_layer, crops_layer, own_level, own_pos
        # (grid_size, grid_size, 3) + (3,) flattened -> we do (grid_size, grid_size, 4)
        self._observation_space = spaces.Box(
            low=0,
            high=max(1, max_level),
            shape=(grid_size, grid_size, 3),
            dtype=np.float32,
        )
        # Plus a small vector for agent-specific info (level, x, y)
        self._observation_space_vec = spaces.Box(
            low=0, high=max(grid_size, max_level), shape=(3,), dtype=np.float32
        )

        # Internal state
        self.agent_levels = None
        self.agent_positions = None
        self.crop_levels = None
        self.crop_positions = None
        self.crop_alive = None
        self._prev_crop_alive = None
        self.step_count = 0
        self._reward_fn = None

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def _generate_entities(self):
        """Generate random agent/crop levels and positions."""
        rng = np.random.RandomState()

        if self.static and hasattr(self, "_static_state"):
            # Use pre-generated static state
            self.agent_levels = self._static_state["agent_levels"].copy()
            self.agent_positions = self._static_state["agent_positions"].copy()
            self.crop_levels = self._static_state["crop_levels"].copy()
            self.crop_positions = self._static_state["crop_positions"].copy()
            return

        # Generate agents
        self.agent_levels = rng.randint(1, self.max_level + 1, size=self.n_agents)
        self.crop_levels = rng.randint(1, self.max_level + 1, size=self.n_crops)

        # Generate non-overlapping positions
        all_positions = set()
        positions = []

        for _ in range(self.n_agents + self.n_crops):
            while True:
                pos = (rng.randint(0, self.grid_size), rng.randint(0, self.grid_size))
                if pos not in all_positions:
                    all_positions.add(pos)
                    positions.append(pos)
                    break

        self.agent_positions = np.array(positions[: self.n_agents])
        self.crop_positions = np.array(positions[self.n_agents :])

        if self.static:
            self._static_state = {
                "agent_levels": self.agent_levels.copy(),
                "agent_positions": self.agent_positions.copy(),
                "crop_levels": self.crop_levels.copy(),
                "crop_positions": self.crop_positions.copy(),
            }

    def _get_obs(self, agent_id):
        """
        Build observation for a specific agent.
        Returns grid (grid_size, grid_size, 3) + agent info vector (3,).
        We return them as a tuple (grid, vector).
        """
        agent_idx = int(agent_id.split("_")[-1])
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Channel 0: own agent position, Channel 1: other agents
        for i, pos in enumerate(self.agent_positions):
            x, y = pos
            if i == agent_idx:
                grid[x, y, 0] = self.agent_levels[i]
            else:
                grid[x, y, 1] = self.agent_levels[i]

        # Channel 2: crop levels
        for i, pos in enumerate(self.crop_positions):
            if self.crop_alive[i]:
                x, y = pos
                grid[x, y, 2] = self.crop_levels[i]

        return grid

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._generate_entities()
        self.crop_alive = np.ones(self.n_crops, dtype=bool)
        self._prev_crop_alive = self.crop_alive.copy()
        self.step_count = 0

        observations = {}
        for aid in self.agents:
            observations[aid] = self._get_obs(aid)

        infos = {aid: {} for aid in self.agents}
        return observations, infos

    def step(self, actions):
        """Execute parallel actions for all agents."""
        # Ensure deterministic conflict resolution
        if not self.agents:
            return {}, {}, {}, {}, {}

        # 1. Process movement
        new_positions = self.agent_positions.copy()
        dx = [0, 1, 0, -1, 0, 0]  # N, E, S, W, Harvest, Stay
        dy = [-1, 0, 1, 0, 0, 0]

        for i, aid in enumerate(self.agents):
            if aid not in actions:
                continue
            action = actions[aid]
            if action < 4:  # Movement
                nx = self.agent_positions[i, 0] + dx[action]
                ny = self.agent_positions[i, 1] + dy[action]
                nx = np.clip(nx, 0, self.grid_size - 1)
                ny = np.clip(ny, 0, self.grid_size - 1)

                # Check if target is already claimed by another agent
                # Conflict resolution: agent with lower index wins
                # We'll resolve after all movements
                new_positions[i] = [nx, ny]

        # Resolve position conflicts
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if np.all(new_positions[i] == new_positions[j]):
                    # Both try to occupy same cell: agent with higher index stays
                    # But to be more fair, both stay (simpler deterministic rule)
                    new_positions[i] = self.agent_positions[i]
                    new_positions[j] = self.agent_positions[j]

        self.agent_positions = new_positions

        # 2. Process harvest actions
        rewards = {aid: 0.0 for aid in self.agents}
        harvest_events = []

        for i, aid in enumerate(self.agents):
            if aid not in actions:
                continue
            action = actions[aid]
            if action == 4:  # Harvest
                ax, ay = self.agent_positions[i]
                # Check adjacent tiles for crops
                for ci in range(self.n_crops):
                    if not self.crop_alive[ci]:
                        continue
                    cx, cy = self.crop_positions[ci]
                    if abs(ax - cx) + abs(ay - cy) <= 1:  # Adjacent or same tile
                        harvest_events.append((ci, i))

        # Resolve harvest conflicts: group by crop
        harvest_groups = {}
        for ci, ai in harvest_events:
            if ci not in harvest_groups:
                harvest_groups[ci] = []
            harvest_groups[ci].append(ai)

        successful_harvests = {}
        attempted_harvests = {}
        for ci, agent_indices in harvest_groups.items():
            attempted_harvests[ci] = agent_indices
            total_level = sum(self.agent_levels[ai] for ai in agent_indices)
            if total_level >= self.crop_levels[ci]:
                self.crop_alive[ci] = False
                successful_harvests[ci] = agent_indices

        # 3. Compute rewards using the active reward function
        crops_remaining = np.sum(self.crop_alive)
        all_harvested = crops_remaining == 0

        # Apply reward function (set externally, or use default global)
        if self._reward_fn is not None:
            rewards = self._reward_fn(
                self.agent_levels,
                self.agent_positions,
                self.crop_levels,
                self.crop_positions,
                self.crop_alive,
                self.step_count,
                self.max_steps,
                successful_harvests,
                attempted_harvests,
            )
        else:
            # Default: simple global reward
            for aid in self.agents:
                progress_reward = (self.n_crops - crops_remaining) / self.n_crops
                time_penalty = -0.01
                rewards[aid] = progress_reward + time_penalty

        # 4. Check termination
        self.step_count += 1
        all_crops_harvested = all_harvested
        time_up = self.step_count >= self.max_steps

        terminations = {aid: all_crops_harvested for aid in self.agents}
        truncations = {aid: time_up for aid in self.agents}

        # If done, clear agents
        if all_crops_harvested or time_up:
            self.agents = []

        # 5. Build observations
        observations = {}
        for aid in self.possible_agents[:self.n_agents]:
            if aid in self.agents or True:
                observations[aid] = self._get_obs(aid)
            else:
                observations[aid] = self._get_obs(aid)

        infos = {aid: {} for aid in self.possible_agents[:self.n_agents]}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render the environment to console."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        # Place agents
        for i, (x, y) in enumerate(self.agent_positions):
            grid[x, y] = f"A{self.agent_levels[i]}"

        # Place crops
        for i, (x, y) in enumerate(self.crop_positions):
            if self.crop_alive[i]:
                if grid[x, y] != ".":
                    grid[x, y] = f"C{self.crop_levels[i]}"
                else:
                    grid[x, y] = f"C{self.crop_levels[i]}"

        print("+" + "---+" * self.grid_size)
        for row in grid:
            print("| " + " | ".join(f"{c:2}" for c in row) + " |")
            print("+" + "---+" * self.grid_size)
        print(f"\nStep: {self.step_count} | Crops remaining: {np.sum(self.crop_alive)}")

    def close(self):
        pass


# Reward function factories
def make_global_reward(grid_size=10, n_agents=3, n_crops=5, max_steps=200):
    """
    Reward function 1: Global reward only.
    All agents share a single reward based on harvest progress.
    Encourages cooperation but may cause credit assignment issues.
    """

    def reward_fn(agent_levels, agent_positions, crop_levels, crop_positions, crop_alive, step, max_steps, successful_harvests, attempted_harvests):
        crops_remaining = np.sum(crop_alive)
        n_crops = len(crop_levels)
        rewards = {}

        # Progress-based global reward
        harvest_ratio = (n_crops - crops_remaining) / n_crops
        for i in range(len(agent_levels)):
            aid = f"agent_{i}"
            # Shared global progress reward with a time penalty
            rewards[aid] = harvest_ratio - 0.01

        # Bonus when all harvested
        if crops_remaining == 0:
            for i in range(len(agent_levels)):
                rewards[f"agent_{i}"] += 1.0

        return rewards

    return reward_fn


def make_mixed_reward(grid_size=10, n_agents=3, n_crops=5, max_steps=200):
    """
    Reward function 2: Mixed global + local reward.
    Global: shared reward based on progress.
    Local: individual reward based on participation in harvests.
    """

    def reward_fn(agent_levels, agent_positions, crop_levels, crop_positions, crop_alive, step, max_steps, successful_harvests, attempted_harvests):
        crops_remaining = np.sum(crop_alive)
        n_crops = len(crop_levels)
        rewards = {}

        harvest_ratio = (n_crops - crops_remaining) / n_crops
        time_penalty = -0.01

        for i in range(len(agent_levels)):
            aid = f"agent_{i}"
            global_reward = harvest_ratio * 0.5 + time_penalty

            # Local reward: reward participation and proximity
            local_reward = 0.0
            if any(i in contributors for contributors in successful_harvests.values()):
                # Agent helped harvest a crop this step
                local_reward += 1.0

            # Proximity to remaining crops encourages exploration
            ax, ay = agent_positions[i]
            for ci in range(len(crop_levels)):
                if crop_alive[ci]:
                    cx, cy = crop_positions[ci]
                    dist = abs(ax - cx) + abs(ay - cy)
                    local_reward += max(0, (1.0 - dist / (grid_size * 2))) * 0.05

            rewards[aid] = global_reward + local_reward

        # Add explicit local reward for attempted harvests that were unsuccessful
        for ci, contributors in attempted_harvests.items():
            if ci not in successful_harvests:
                for ai in contributors:
                    rewards[f"agent_{ai}"] += -0.1

        # Big bonus when all harvested
        if crops_remaining == 0:
            for i in range(len(agent_levels)):
                rewards[f"agent_{i}"] += 2.0

        return rewards

    return reward_fn


def set_reward_function(env, reward_fn):
    """Set the reward function for the environment.

    The reward function signature must be:
        (agent_levels, agent_positions, crop_levels,
         crop_positions, crop_alive, step, max_steps,
         successful_harvests, attempted_harvests)

    - successful_harvests: dict crop_index -> list of agent indices that jointly harvested it
    - attempted_harvests: dict crop_index -> list of agent indices that attempted harvest
    """
    env._reward_fn = reward_fn
