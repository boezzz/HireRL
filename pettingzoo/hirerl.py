"""
Environment Structure:
- Strategic Agents: Companies (RL agents that learn hiring policies)
- Environment: Worker pool (non-strategic, greedy wage-takers)
- Information Asymmetry: Workers know true ability σ_j, firms observe noisy signals
- Screening: Firms can invest cost c to get better ability estimates
- Matching: Greedy (myopic) vs Stable matching mechanisms

Action Space:
- no-op: Do nothing
- fire(j): Fire worker j
- offer(j, w): Make wage offer w to unemployed worker j
- interview(j, c): Screen worker j with cost c (before hiring)

Observation Space (Partial Observability):
- Public signals σ̂_j,t for all workers
- Employment status and wages (public)
- Experience levels exp_j,t (public)
- Firm's private beliefs about abilities (from screening/performance)
- Firm's current workforce and profits

Reward:
r_i,t = Σ_{j ∈ E_i,t} (p_ij,t - w_ij,t) - c_fire - c_hire

where p_ij,t = σ_j + β*log(1 + exp_j,t) depends on TRUE ability (private info)
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict as GymDict, Tuple as GymTuple, MultiBinary
from pettingzoo.utils.env import ParallelEnv
from typing import Dict, List, Any, Tuple, Optional, Set
from enum import IntEnum

from workers import WorkerPool, WorkerState
from screening import ScreeningMechanism, FirmBeliefs, ScreeningTechnology


class ActionType(IntEnum):
    """
    Company action types.

    We encode actions as integers for compatibility with standard RL algorithms.
    Action encoding:
        0: NO_OP
        1 to N: FIRE worker (j = action - 1)
        N+1 to 2N: OFFER to worker (j = action - N - 1) with default wage
        2N+1 to 3N: INTERVIEW worker (j = action - 2N - 1) with default cost
    """
    NO_OP = 0
    FIRE_BASE = 1
    OFFER_BASE = -1  # Computed dynamically
    INTERVIEW_BASE = -1  # Computed dynamically


class JobMarketEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for job market with screening and matching.

    This implements the full MARL specification including:
    - Partial observability (firms don't see true ability)
    - Costly screening mechanism
    - Employer learning from performance
    - Deterministic quit rule
    - Comparison with stable matching
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "job_market_v2"}

    def __init__(
        self,
        num_companies: int = 3,
        num_workers: int = 10,
        ability_dim: int = 1,
        max_workers_per_company: int = 5,
        # Worker dynamics
        gamma: float = 0.1,  # Tenure signal growth: σ̂_j,t = σ̂_j,0 + γ*τ_j,t
        g0: float = 0.1,     # Base experience growth
        g1: float = 0.05,    # Ability-dependent experience growth
        # Costs
        base_firing_cost: float = 0.5, # firing cost should be >>> hiring cost
        base_hiring_cost: float = 0.2,
        base_screening_cost: float = 0.5,  # Default interview cost
        # Screening technology
        screening_tech: ScreeningTechnology = ScreeningTechnology.SQRT,
        screening_c_max: float = 1.0,
        # Wage parameters
        worker_bargaining_power: float = 0.6,  # α in wage = α * expected_profit
        # Episode length
        max_timesteps: int = 100,
        # Rendering
        render_mode: Optional[str] = None,
        # Seed
        seed: Optional[int] = None
    ):
        """
        Initialize job market environment.

        Args:
            num_companies: Number of company agents
            num_workers: Total workers in the market
            ability_dim: Dimensionality of ability vector (default 1)
            max_workers_per_company: Capacity constraint per company
            gamma: Public signal growth rate with tenure
            g0, g1: Experience accumulation parameters
            base_firing_cost: Cost to fire a worker
            base_hiring_cost: Cost to hire a worker (onboarding, etc.)
            base_screening_cost: Default interview/screening cost
            screening_tech: Screening technology type
            screening_c_max: Maximum meaningful screening cost
            worker_bargaining_power: Fraction of surplus captured by workers
            max_timesteps: Episode length
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            seed: Random seed
        """
        super().__init__()

        self.render_mode = render_mode
        self.num_companies = num_companies
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_workers_per_company = max_workers_per_company

        self.gamma = gamma
        self.g0 = g0
        self.g1 = g1

        self.base_firing_cost = base_firing_cost
        self.base_hiring_cost = base_hiring_cost
        self.base_screening_cost = base_screening_cost

        self.worker_bargaining_power = worker_bargaining_power
        self.max_timesteps = max_timesteps

        # Initialize worker pool
        self.worker_pool = WorkerPool(
            num_workers=num_workers,
            ability_dim=ability_dim,
            gamma=gamma,
            g0=g0,
            g1=g1,
            seed=seed
        )

        # Initialize screening mechanism
        self.screening = ScreeningMechanism(
            technology=screening_tech,
            c_max=screening_c_max,
            seed=seed
        )

        # Company agent setup
        self.agents = [f"company_{i}" for i in range(num_companies)]
        self.possible_agents = self.agents.copy()

        # Each company maintains beliefs about worker abilities
        self.firm_beliefs: Dict[str, FirmBeliefs] = {
            agent: FirmBeliefs(num_workers, ability_dim)
            for agent in self.agents
        }

        # Action space: NO_OP + FIRE*N + OFFER*N + INTERVIEW*N
        self.action_size = 1 + 3 * num_workers
        self._action_spaces = {
            agent: Discrete(self.action_size)
            for agent in self.agents
        }

        # Observation space: (publicly observable + private beliefs)
        # Public: sigma_hat (N,d), experience (N,), tenure (N,), employed_by (N,), wages (N,)
        # Private: belief_mean (N,d), belief_var (N,d), own_workforce (N binary), own_profit (1,)
        obs_size = (
            num_workers * ability_dim +  # sigma_hat
            num_workers +                 # experience
            num_workers +                 # tenure
            num_workers +                 # employed_by
            num_workers +                 # wages
            num_workers * ability_dim +  # belief_mean
            num_workers * ability_dim +  # belief_var
            num_workers +                 # own_workforce (binary)
            1                             # own_profit
        )

        self._observation_spaces = {
            agent: GymDict({
                'observation': Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_size,),
                    dtype=np.float32
                ),
                'action_mask': MultiBinary(self.action_size)
            })
            for agent in self.agents
        }

        # State tracking
        self.timestep = 0
        self.company_profits = {agent: [] for agent in self.agents}  # Track profit history

        # RNG
        self.rng = np.random.RandomState(seed)

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state.

        Returns:
            observations: Dict of observations for each agent
            infos: Dict of info dicts for each agent
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset worker pool
        self.worker_pool.reset(seed=seed)

        # Reset firm beliefs (initialized from public signals)
        public_state = self.worker_pool.get_public_state()
        for agent in self.agents:
            self.firm_beliefs[agent].initialize_from_public_signals(
                public_state['sigma_hat'],
                signal_noise_var=0.25
            )

        # Reset state
        self.timestep = 0
        self.company_profits = {agent: [0.0] for agent in self.possible_agents}

        # Reset agents to all possible agents (alive at start of episode)
        self.agents = self.possible_agents.copy()

        # Generate initial observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def _generate_action_mask(self, agent: str) -> np.ndarray:
        """
        Generate action mask for a specific agent.

        Action validity rules:
        - NO_OP (0): Always valid
        - FIRE (1 to N): Valid only if worker is employed by this company
        - OFFER (N+1 to 2N): Valid only if worker is unemployed AND company has capacity
        - INTERVIEW (2N+1 to 3N): Valid only if worker is unemployed

        Args:
            agent: Company agent name

        Returns:
            Binary mask where 1 = valid action, 0 = invalid action
        """
        company_idx = int(agent.split("_")[1])
        action_mask = np.zeros(self.action_size, dtype=np.int8)

        # NO_OP is always valid
        action_mask[0] = 1

        # Check company capacity
        current_workforce = len(self.worker_pool.get_employed_by_company(company_idx))
        has_capacity = current_workforce < self.max_workers_per_company

        # Process each worker
        for worker_id in range(self.num_workers):
            worker = self.worker_pool.workers[worker_id]

            # FIRE action: valid only if worker is employed by this company
            fire_action_id = 1 + worker_id
            if worker.employed_by == company_idx:
                action_mask[fire_action_id] = 1

            # OFFER action: valid only if worker is unemployed AND company has capacity
            offer_action_id = self.num_workers + 1 + worker_id
            if worker.employed_by == -1 and has_capacity:
                action_mask[offer_action_id] = 1

            # INTERVIEW action: valid only if worker is unemployed
            interview_action_id = 2 * self.num_workers + 1 + worker_id
            if worker.employed_by == -1:
                action_mask[interview_action_id] = 1

        return action_mask

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Get observation for a specific agent.

        Observation includes:
        - Public information (all workers)
        - Private beliefs (this firm's estimates of abilities)
        - Own workforce and profit
        - Action mask indicating valid actions

        Returns:
            Dictionary with 'observation' and 'action_mask' keys
        """
        company_idx = int(agent.split("_")[1])

        # Get public state
        public = self.worker_pool.get_public_state()

        # Get private beliefs
        beliefs = self.firm_beliefs[agent]

        # Own workforce (binary indicator per worker)
        own_workforce = np.array([
            1.0 if w.employed_by == company_idx else 0.0
            for w in self.worker_pool.workers
        ], dtype=np.float32)

        # Own profit (most recent)
        own_profit = np.array([self.company_profits[agent][-1]], dtype=np.float32)

        # Concatenate all components
        obs = np.concatenate([
            public['sigma_hat'].flatten(),
            public['experience'],
            public['tenure'].astype(np.float32),
            public['employed_by'].astype(np.float32),
            public['wages'],
            beliefs.belief_mean.flatten(),
            beliefs.belief_var.flatten(),
            own_workforce,
            own_profit
        ])

        # Generate action mask
        action_mask = self._generate_action_mask(agent)

        return {
            'observation': obs.astype(np.float32),
            'action_mask': action_mask
        }

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get info dict for agent (debugging/logging info)."""
        company_idx = int(agent.split("_")[1])
        workforce = self.worker_pool.get_employed_by_company(company_idx)

        return {
            'workforce_size': len(workforce),
            'total_profit': sum(self.company_profits[agent]),
            'unemployment_rate': self.worker_pool.get_unemployment_rate(),
            'avg_wage': self.worker_pool.get_average_wage(),
            'timestep': self.timestep
        }

    def _decode_action(self, agent: str, action: int) -> Tuple[str, Optional[int], Optional[float]]:
        """
        Decode integer action into (action_type, worker_id, value).

        Returns:
            (action_type, worker_id, value)
            where:
                action_type in ['noop', 'fire', 'offer', 'interview']
                worker_id: index of worker (or None for noop)
                value: wage for offer, cost for interview (or None)
        """
        if action == 0:
            return ('noop', None, None)

        elif 1 <= action <= self.num_workers:
            # Fire action
            worker_id = action - 1
            return ('fire', worker_id, None)

        elif self.num_workers + 1 <= action <= 2 * self.num_workers:
            # Offer action (with computed wage)
            worker_id = action - self.num_workers - 1
            # Compute wage based on beliefs and bargaining power
            wage = self._compute_wage_offer(agent, worker_id)
            return ('offer', worker_id, wage)

        elif 2 * self.num_workers + 1 <= action <= 3 * self.num_workers:
            # Interview action (with default cost)
            worker_id = action - 2 * self.num_workers - 1
            cost = self.base_screening_cost
            return ('interview', worker_id, cost)

        else:
            # Invalid action, treat as noop
            return ('noop', None, None)

    def _compute_wage_offer(self, agent: str, worker_id: int) -> float:
        """
        Compute wage offer based on firm's beliefs and worker bargaining power.

        Wage = α * E[profit | beliefs]
             = α * (E[σ_j | beliefs] + β*log(1 + exp_j))

        Args:
            agent: Company agent
            worker_id: Worker to make offer to

        Returns:
            Wage offer
        """
        worker = self.worker_pool.workers[worker_id]
        beliefs = self.firm_beliefs[agent]

        expected_profit = beliefs.get_expected_profit(
            worker_id,
            worker.experience,
            beta=0.5
        )

        wage = self.worker_bargaining_power * expected_profit

        return max(0.1, wage)  # Minimum wage floor

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminations
        Dict[str, bool],         # truncations
        Dict[str, dict]          # infos
    ]:
        """
        Execute one timestep of the environment.

        Phase 1: Decode actions
        Phase 2: Resolve screening/interviews (update beliefs)
        Phase 3: Resolve firings
        Phase 4: Resolve offers (workers accept highest wage)
        Phase 5: Compute rewards (production from current workforce)
        Phase 6: Update worker states (experience, tenure, public signals)
        Phase 7: Apply deterministic quit rule
        Phase 8: Update beliefs from performance observations

        Args:
            actions: Dict mapping agent -> action_id

        Returns:
            Standard PettingZoo ParallelEnv step return tuple
        """
        # Phase 1: Decode actions
        decoded_actions = {}
        for agent, action in actions.items():
            decoded_actions[agent] = self._decode_action(agent, action)

        # Phase 2: Process screening/interviews
        screening_costs = {agent: 0.0 for agent in self.agents}

        for agent, (action_type, worker_id, value) in decoded_actions.items():
            if action_type == 'interview' and worker_id is not None:
                cost = value if value is not None else self.base_screening_cost
                screening_costs[agent] += cost

                # Perform screening
                worker = self.worker_pool.workers[worker_id]
                sigma_estimate, precision = self.screening.screen_worker(
                    worker.sigma_true,
                    worker.sigma_hat_0,
                    cost
                )

                # Update beliefs
                self.firm_beliefs[agent].update_from_screening(
                    worker_id,
                    sigma_estimate,
                    cost,
                    self.screening
                )

        # Phase 3: Process firings
        for agent, (action_type, worker_id, _) in decoded_actions.items():
            if action_type == 'fire' and worker_id is not None:
                company_idx = int(agent.split("_")[1])
                worker = self.worker_pool.workers[worker_id]

                # Can only fire own employees
                if worker.employed_by == company_idx:
                    self.worker_pool.fire_worker(worker_id)

        # Phase 4: Process offers (workers choose best offer)
        # Collect all offers
        offers_by_worker: Dict[int, List[Tuple[int, float]]] = {}  # worker_id -> [(company_idx, wage), ...]

        for agent, (action_type, worker_id, value) in decoded_actions.items():
            if action_type == 'offer' and worker_id is not None:
                company_idx = int(agent.split("_")[1])
                worker = self.worker_pool.workers[worker_id]

                # Can only make offers to unemployed workers
                if worker.employed_by == -1:
                    wage = value if value is not None else self._compute_wage_offer(agent, worker_id)

                    if worker_id not in offers_by_worker:
                        offers_by_worker[worker_id] = []
                    offers_by_worker[worker_id].append((company_idx, wage))

        # Workers accept highest wage offer (greedy)
        hiring_costs = {agent: 0.0 for agent in self.agents}

        for worker_id, offers in offers_by_worker.items():
            if not offers:
                continue

            # Choose highest wage
            best_company, best_wage = max(offers, key=lambda x: x[1])

            # Check company capacity
            current_workforce = len(self.worker_pool.get_employed_by_company(best_company))
            if current_workforce < self.max_workers_per_company:
                self.worker_pool.hire_worker(worker_id, best_company, best_wage)
                agent_name = f"company_{best_company}"
                hiring_costs[agent_name] += self.base_hiring_cost

        # Phase 5: Compute rewards (production from current workforce BEFORE updates)
        rewards = {}
        firing_costs = {agent: 0.0 for agent in self.agents}

        # Count firings
        for agent, (action_type, worker_id, _) in decoded_actions.items():
            if action_type == 'fire' and worker_id is not None:
                firing_costs[agent] += self.base_firing_cost

        for agent in self.agents:
            company_idx = int(agent.split("_")[1])
            workforce = self.worker_pool.get_employed_by_company(company_idx)

            # Compute profit from current employees
            total_profit = 0.0
            total_wages = 0.0

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                profit = self.worker_pool.compute_match_profit(worker_id, company_idx)
                total_profit += profit
                total_wages += worker.wage

            # Net reward
            reward = (
                total_profit -
                total_wages -
                firing_costs[agent] -
                hiring_costs[agent] -
                screening_costs[agent]
            )

            rewards[agent] = float(reward)
            self.company_profits[agent].append(reward)

        # Phase 6: Update worker states
        self.worker_pool.update_experience_and_tenure()

        # Phase 7: Apply deterministic quit rule
        quit_workers = self.worker_pool.apply_deterministic_quit_rule()

        # Phase 8: Update beliefs from performance (employer learning)
        for agent in self.agents:
            company_idx = int(agent.split("_")[1])
            workforce = self.worker_pool.get_employed_by_company(company_idx)

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                observed_profit = self.worker_pool.compute_match_profit(worker_id, company_idx)

                # Update beliefs based on performance
                self.firm_beliefs[agent].update_from_performance(
                    worker_id,
                    observed_profit,
                    worker.experience,
                    profit_noise_var=0.1
                )

        # Increment time
        self.timestep += 1

        # Generate new observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_timesteps for agent in self.agents}

        # Remove done agents from self.agents (PettingZoo Parallel API requirement)
        self.agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """
        Render the current state.

        Displays:
        - Time step
        - Unemployment rate
        - Average wage
        - Company workforces and profits
        """
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Time Step: {self.timestep}/{self.max_timesteps}")
            print(f"Unemployment Rate: {self.worker_pool.get_unemployment_rate():.2%}")
            print(f"Average Wage: {self.worker_pool.get_average_wage():.2f}")
            print(f"{'='*60}")

            # Use possible_agents instead of agents (agents may be empty after episode ends)
            for agent in self.possible_agents:
                company_idx = int(agent.split("_")[1])
                workforce = self.worker_pool.get_employed_by_company(company_idx)
                recent_profit = self.company_profits[agent][-1] if self.company_profits[agent] else 0.0
                total_profit = sum(self.company_profits[agent])

                print(f"\n{agent}:")
                print(f"  Workforce: {len(workforce)}/{self.max_workers_per_company}")
                print(f"  Recent Profit: {recent_profit:.2f}")
                print(f"  Total Profit: {total_profit:.2f}")

            print(f"{'='*60}\n")

    def close(self):
        """Clean up resources."""
        pass
