"""
HireRL Parallel Environment aligned with paper timing:

1. Firms deterministically target the highest public-signal workers for interviews.
2. Agents decide how much to invest in the interview (cost -> signal precision).
3. Wage offers for newly interviewed workers depend only on interview signals.
4. After production, firms update beliefs from realized profits.
5. Existing matches adjust wages using the wage rule with past profits.
6. Deterministic firing rule: fire if p - w < -c_fire.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from gymnasium.spaces import Discrete, Box, Dict as GymDict, MultiBinary
from pettingzoo.utils.env import ParallelEnv

from workers import WorkerPool
from interview0 import ScreeningMechanism
from matching1 import greedy_wage_matching_from_signals
from after_hiring_update2 import FirmBeliefs
from generated_profit3 import generate_profit, update_belief_from_profit
from post_hiring_adjust_wage4 import (
    default_g_bounded,
    adjust_wage_post_hire,
    firing_decision,
)


class JobMarketEnv(ParallelEnv):
    """
    Parallel environment where the only strategic choice is interview cost.

    Deterministic structure:
        - Each firm (ordered by size) targets the highest-signal unemployed workers.
        - Workers accept the interview from the largest interested firm.
        - Hiring is myopic and wage offers for new hires follow step (3) of the paper.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "job_market_v3"}

    def __init__(
        self,
        num_companies: int = 1,
        num_workers: int = 10,
        ability_dim: int = 1,
        max_workers_per_company: int = 5,
        gamma: float = 0.1,
        g0: float = 0.1,
        g1: float = 0.05,
        base_firing_cost: float = 1.0,
        base_screening_cost: float = 0.5,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        profit_theta: float = 0.05,
        profit_noise_var: float = 0.1,
        profit_function_type: str = "linear",
        wage_profit_share: float = 0.5,
        initial_offer_vx: float = 0.0,
        max_timesteps: int = 100,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
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
        self.base_screening_cost = base_screening_cost
        self.max_interview_cost = max_interview_cost
        self.num_interview_cost_levels = max(2, num_interview_cost_levels)

        self.wage_profit_share = wage_profit_share
        self.initial_offer_vx = float(np.clip(initial_offer_vx, 0.0, 0.99))
        self.profit_theta = profit_theta
        self.delta_eps_sq = profit_noise_var
        self.profit_function_type = profit_function_type
        self.max_timesteps = max_timesteps

        self.worker_pool = WorkerPool(
            num_workers=num_workers,
            ability_dim=ability_dim,
            gamma=gamma,
            g0=g0,
            g1=g1,
            seed=seed,
        )

        self.screening = ScreeningMechanism(
            delta0_sq=1.0,
            lam=1.0,
            seed=seed,
        )

        self.agents = [f"company_{i}" for i in range(num_companies)]
        self.possible_agents = self.agents.copy()
        self.firm_priority = list(range(num_companies))  # smaller index = larger firm

        self.firm_beliefs: Dict[str, FirmBeliefs] = {
            agent: FirmBeliefs(num_workers=num_workers, ability_dim=ability_dim)
            for agent in self.agents
        }

        # Track interview information and profits for each firm-worker pair
        def _init_matrix(value: float = 0.0) -> Dict[str, np.ndarray]:
            return {agent: np.full(num_workers, value, dtype=np.float32) for agent in self.agents}

        base_var = self.screening.interview_var(0.0)
        self._interview_signal_at_hire = _init_matrix(0.0)
        self._interview_vars = _init_matrix(base_var)
        self._last_profit = _init_matrix(0.0)

        # Action space: choose among discrete interview costs
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            self.num_interview_cost_levels,
            dtype=np.float32,
        )
        self.idle_action = 0
        self.action_size = self.num_interview_cost_levels
        self._action_spaces = {agent: Discrete(self.action_size) for agent in self.agents}

        obs_size = (
            num_workers * ability_dim  # sigma_hat
            + num_workers  # experience
            + num_workers  # tenure
            + num_workers  # employed_by
            + num_workers  # wages
            + num_workers * ability_dim  # belief mean
            + num_workers * ability_dim  # belief variance
            + num_workers  # own workforce indicator
            + 1  # own profit
        )
        obs_space = GymDict(
            {
                "observation": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_size,),
                    dtype=np.float32,
                ),
                "action_mask": MultiBinary(self.action_size),
            }
        )
        self._observation_spaces = {agent: obs_space for agent in self.agents}

        self.timestep = 0
        self.company_profits: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def _company_index(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _deterministic_interview_assignments(self) -> Dict[int, int]:
        """Assign each firm to the highest public-signal unemployed worker.
        Firms are prioritized by current firm size (descending)."""
        public = self.worker_pool.get_public_state()
        sigma_hat = public["sigma_hat"][:, 0] if self.ability_dim == 1 else np.mean(public["sigma_hat"], axis=1)

        # Compute current firm sizes
        firm_sizes = {
            firm_idx: len(self.worker_pool.get_employed_by_company(firm_idx))
            for firm_idx in range(self.num_companies)
        }
        # Sort firms by descending size (largest first); break ties by firm index
        firm_order = sorted(firm_sizes.keys(), key=lambda k: (-firm_sizes[k], k))

        unemployed = [
            w.worker_id
            for w in self.worker_pool.workers
            if w.employed_by == -1
        ]
        unemployed_sorted = sorted(
            unemployed,
            key=lambda j: float(sigma_hat[j]),
            reverse=True,
        )

        assignments: Dict[int, int] = {}
        idx = 0

        for firm_idx in firm_order:
            current_workforce = firm_sizes[firm_idx]
            if current_workforce >= self.max_workers_per_company:
                continue
            if idx >= len(unemployed_sorted):
                break
            assignments[firm_idx] = unemployed_sorted[idx]
            idx += 1

        return assignments

    def _cost_from_action(self, action: int) -> float:
        action = int(np.clip(action, 0, self.action_size - 1))
        return float(self.cost_levels[action])

    def _compute_vx(self, exp_t: float, delta_interview_sq: float) -> float:
        exp_t = max(0.0, float(exp_t))
        denom = delta_interview_sq + self.delta_eps_sq
        if denom <= 0.0:
            return 0.0
        K1 = delta_interview_sq / denom
        if K1 <= 0.0:
            return 0.0
        return (exp_t * K1) / (1.0 + (exp_t - 1.0) * K1)

    def _update_wages_existing_employees(self):
        """Apply step (5) wage rule for continuing employees before new hires."""
        for agent in self.agents:
            company_idx = self._company_index(agent)
            workforce = self.worker_pool.get_employed_by_company(company_idx)

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                last_profit = float(self._last_profit[agent][worker_id])
                tilde_sigma = float(self._interview_signal_at_hire[agent][worker_id])
                delta_interview_sq = float(self._interview_vars[agent][worker_id])

                result = adjust_wage_post_hire(
                    tilde_sigma_interview=tilde_sigma,
                    p_ij_tm1=last_profit,
                    exp_t=worker.experience,
                    delta_interview_sq=delta_interview_sq,
                    delta_eps_sq=self.delta_eps_sq,
                    psi=self.wage_profit_share,
                )

                worker.wage = result.wage_t

    def _generate_action_mask(self, agent: str) -> np.ndarray:
        assignments = self._deterministic_interview_assignments()
        company_idx = self._company_index(agent)
        mask = np.zeros(self.action_size, dtype=np.int8)

        if company_idx not in assignments:
            mask[self.idle_action] = 1
        else:
            mask[:] = 1
        return mask

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        company_idx = self._company_index(agent)
        public = self.worker_pool.get_public_state()
        beliefs = self.firm_beliefs[agent]

        own_workforce = np.array(
            [1.0 if w.employed_by == company_idx else 0.0 for w in self.worker_pool.workers],
            dtype=np.float32,
        )
        own_profit = np.array([self.company_profits[agent][-1]], dtype=np.float32)

        obs = np.concatenate(
            [
                public["sigma_hat"].flatten(),
                public["experience"],
                public["tenure"].astype(np.float32),
                public["employed_by"].astype(np.float32),
                public["wages"],
                beliefs.belief_mean.flatten(),
                beliefs.belief_var.flatten(),
                own_workforce,
                own_profit,
            ]
        ).astype(np.float32)

        return {
            "observation": obs,
            "action_mask": self._generate_action_mask(agent),
        }

    def _get_info(self, agent: str) -> Dict[str, Any]:
        company_idx = self._company_index(agent)
        workforce = self.worker_pool.get_employed_by_company(company_idx)
        return {
            "workforce_size": len(workforce),
            "total_profit": float(sum(self.company_profits[agent])),
            "unemployment_rate": self.worker_pool.get_unemployment_rate(),
            "avg_wage": self.worker_pool.get_average_wage(),
            "timestep": self.timestep,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.worker_pool.reset(seed=seed)
        public_state = self.worker_pool.get_public_state()

        base_var = self.screening.interview_var(0.0)

        self.agents = self.possible_agents.copy()
        self.timestep = 0
        self.company_profits = {agent: [0.0] for agent in self.agents}

        for agent in self.agents:
            beliefs = FirmBeliefs(num_workers=self.num_workers, ability_dim=self.ability_dim)
            for worker_id in range(self.num_workers):
                sigma_init = public_state["sigma_hat"][worker_id]
                init_val = float(sigma_init[0]) if self.ability_dim == 1 else float(np.mean(sigma_init))
                beliefs.initialize_from_interview_signal(worker_id, init_val, signal_noise_var=base_var)
            self.firm_beliefs[agent] = beliefs
            self._interview_signal_at_hire[agent] = np.zeros(self.num_workers, dtype=np.float32)
            self._interview_vars[agent] = np.full(self.num_workers, base_var, dtype=np.float32)
            self._last_profit[agent] = np.zeros(self.num_workers, dtype=np.float32)

        # --- Assign each firm a random initial workforce drawn from the worker pool ---
        all_workers = list(range(self.num_workers))
        self.rng.shuffle(all_workers)
        ptr = 0
        for firm_idx in range(self.num_companies):
            if ptr >= len(all_workers):
                break
            max_assignable = min(self.max_workers_per_company, len(all_workers) - ptr)
            init_n = self.rng.randint(0, max_assignable + 1)
            for _ in range(init_n):
                if ptr >= len(all_workers):
                    break
                worker_id = all_workers[ptr]
                ptr += 1
                initial_wage = 0.0
                self.worker_pool.hire_worker(worker_id, firm_idx, initial_wage)
                agent_name = f"company_{firm_idx}"
                sigma_init = public_state["sigma_hat"][worker_id]
                signal_scalar = float(sigma_init[0]) if self.ability_dim == 1 else float(np.mean(sigma_init))
                self._interview_signal_at_hire[agent_name][worker_id] = signal_scalar
                self._interview_vars[agent_name][worker_id] = base_var

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        decoded_actions = {agent: int(actions.get(agent, self.idle_action)) for agent in self.agents}

        prev_state = [
            {
                "experience": worker.experience,
                "employed": worker.employed_by >= 0,
                "sigma": float(worker.sigma_true[0] if self.ability_dim == 1 else np.mean(worker.sigma_true)),
            }
            for worker in self.worker_pool.workers
        ]

        # Update wages for continuing employees using last period's profits.
        # Conceptually, this implements the wage rule
        #   w_{j,t} = (1 - v_x) g(tilde_sigma_{ij,interview}) + v_x * psi * p_{ij,t-1},
        # so wages for period t are set at the beginning of period t based on profits from t-1.
        self._update_wages_existing_employees()

        assignments = self._deterministic_interview_assignments()
        screening_costs = {agent: 0.0 for agent in self.agents}
        tilde_matrix = np.full((self.num_companies, self.num_workers), -np.inf, dtype=np.float32)
        targeted_workers: set[int] = set()

        for agent in self.agents:
            company_idx = self._company_index(agent)
            if company_idx not in assignments:
                continue

            worker_id = assignments[company_idx]
            worker = self.worker_pool.workers[worker_id]
            cost = self._cost_from_action(decoded_actions[agent])

            screening_costs[agent] += cost

            tilde_sigma, _ = self.screening.screen_worker(
                sigma_true=worker.sigma_true,
                sigma_hat_0=worker.sigma_hat,
                cost=cost,
            )
            signal_scalar = float(tilde_sigma[0]) if self.ability_dim == 1 else float(np.mean(tilde_sigma))
            var_val = self.screening.interview_var(cost)

            self.firm_beliefs[agent].initialize_from_interview_signal(
                worker_id,
                signal_scalar,
                signal_noise_var=var_val,
            )
            self._interview_signal_at_hire[agent][worker_id] = signal_scalar
            self._interview_vars[agent][worker_id] = var_val

            tilde_matrix[company_idx, worker_id] = signal_scalar
            targeted_workers.add(worker_id)

        if targeted_workers:
            matching_result = greedy_wage_matching_from_signals(
                tilde_sigma=tilde_matrix,
                v_x=self.initial_offer_vx,
                g=default_g_bounded,
                eligible_workers=sorted(targeted_workers),
            )

            for firm_idx, worker_id in matching_result.firm_to_worker.items():
                if worker_id is None:
                    continue
                current_workforce = len(self.worker_pool.get_employed_by_company(firm_idx))
                if current_workforce >= self.max_workers_per_company:
                    continue
                worker = self.worker_pool.workers[worker_id]
                if worker.employed_by != -1:
                    continue
                wage_offer = matching_result.worker_wage.get(worker_id)
                if wage_offer is None:
                    continue
                self.worker_pool.hire_worker(worker_id, firm_idx, wage_offer)

        # Step 4: production and belief updates
        total_profits = {agent: 0.0 for agent in self.agents}
        total_wages = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            company_idx = self._company_index(agent)
            workforce = self.worker_pool.get_employed_by_company(company_idx)

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                prev = prev_state[worker_id]

                profit = generate_profit(
                    exp_tm1=prev["experience"],
                    sigma_j=prev["sigma"],
                    employed_tm1=prev["employed"],
                    g0=self.g0,
                    g1=self.g1,
                    theta=self.profit_theta,
                    delta_eps_sq=self.delta_eps_sq,
                    f_type=self.profit_function_type,
                    rng=self.rng,
                )

                total_profits[agent] += profit
                total_wages[agent] += worker.wage

                tilde_sigma_interview = float(self._interview_signal_at_hire[agent][worker_id])
                delta_interview_sq = float(self._interview_vars[agent][worker_id])

                new_belief, vx = update_belief_from_profit(
                    tilde_sigma_interview=tilde_sigma_interview,
                    p_ijt=profit,
                    exp_t=worker.experience,
                    delta_interview_sq=delta_interview_sq,
                    delta_eps_sq=self.delta_eps_sq,
                )
                self.firm_beliefs[agent].belief_mean[worker_id, 0] = new_belief
                self._last_profit[agent][worker_id] = profit

        # Experience update occurs after production, before firing
        self.worker_pool.update_experience_and_tenure()

        # Step 6: firing decision
        firing_costs = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            company_idx = self._company_index(agent)
            workforce = list(self.worker_pool.get_employed_by_company(company_idx))

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                profit = float(self._last_profit[agent][worker_id])
                wage_paid = float(worker.wage)

                decision = firing_decision(
                    p_ijt=profit,
                    w_ijt=wage_paid,
                    c_fire_t=self.base_firing_cost,
                )

                if decision.fire:
                    self.worker_pool.fire_worker(worker_id)
                    firing_costs[agent] += self.base_firing_cost
                    self._last_profit[agent][worker_id] = 0.0
                    self._interview_signal_at_hire[agent][worker_id] = 0.0

        rewards = {}
        for agent in self.agents:
            reward = (
                total_profits[agent]
                - total_wages[agent]
                - screening_costs[agent]
                - firing_costs[agent]
            )
            rewards[agent] = float(reward)
            self.company_profits[agent].append(float(reward))

        self.timestep += 1

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_timesteps for agent in self.agents}

        if any(truncations.values()):
            self.agents = [agent for agent in self.agents if not truncations[agent]]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "human":
            return

        print(f"\n{'=' * 60}")
        print(f"Time Step: {self.timestep}/{self.max_timesteps}")
        print(f"Unemployment Rate: {self.worker_pool.get_unemployment_rate():.2%}")
        print(f"Average Wage: {self.worker_pool.get_average_wage():.2f}")
        print(f"{'=' * 60}")

        for agent in self.possible_agents:
            company_idx = self._company_index(agent)
            workforce = self.worker_pool.get_employed_by_company(company_idx)
            recent_profit = self.company_profits[agent][-1] if self.company_profits[agent] else 0.0
            total_profit = float(sum(self.company_profits[agent]))
            print(f"\n{agent}:")
            print(f"  Workforce: {len(workforce)}/{self.max_workers_per_company}")
            print(f"  Recent Profit: {recent_profit:.2f}")
            print(f"  Total Profit: {total_profit:.2f}")

        print(f"{'=' * 60}\n")

    def close(self):
        return None
