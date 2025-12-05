from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from gymnasium.spaces import Box, Dict as GymDict, MultiBinary, Discrete
from pettingzoo.utils.env import ParallelEnv

from interview0 import ScreeningMechanism
from matching1 import greedy_wage_matching_from_signals, WageMatchingResult
from generated_profit2 import generate_profit_array, update_beliefs_and_experience
from post_hiring_adjust_wage3 import default_g_bounded, adjust_wage_post_hire


class JobMarketEnv(ParallelEnv):
    """
    Minimal 1-2-3 hiring pipeline:
      1) Interview: firms draw from σ̂ and refine signals via interview0.screen_worker.
      2) Matching: firms offer to 20% of interviewed workers (capacity-capped).
      3) Profit/learning: realized profit, wage update, and belief/experience updates.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "job_market_v3"}

    def __init__(
        self,
        num_companies: int = 1,
        num_workers: int = 10,
        ability_dim: int = 1,
        max_workers_per_company: int = 5,
        firm_capacities: Optional[List[int]] = None,
        g0: float = 0.1,
        g1: float = 0.05,
        profit_theta: float = 0.05,
        profit_noise_var: float = 0.05,
        wage_profit_share: float = 0.5,
        wage_scale: float = 1.0,
        initial_offer_vx: float = 0.0,
        public_signal_variance: float = 1.0,
        max_timesteps: int = 100,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
        firm_types: Optional[List[str]] = None,
        firm_type_premia: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        if ability_dim != 1:
            raise ValueError("This simplified pipeline currently supports ability_dim=1.")
        self.num_companies = num_companies
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_workers_per_company = max_workers_per_company
        self.firm_capacities = firm_capacities

        self.g0 = g0
        self.g1 = g1
        self.profit_theta = profit_theta
        self.delta_eps_sq = profit_noise_var
        self.wage_profit_share = wage_profit_share
        self.wage_scale = wage_scale
        self.initial_offer_vx = float(np.clip(initial_offer_vx, 0.0, 0.99))
        self.public_signal_variance = public_signal_variance
        self.max_timesteps = max_timesteps

        # Cap interview cost at 2.7 as requested
        max_cost = float(min(2.7, max_interview_cost))
        self.max_interview_cost = max_cost

        self.firm_types = firm_types if firm_types is not None else ["generic"] * num_companies
        default_premia = {"small": 1.0, "medium": 1.0, "large": 1.0, "generic": 1.0}
        if firm_type_premia:
            default_premia.update(firm_type_premia)
        self.firm_type_premia = default_premia

        self.screening = ScreeningMechanism(delta0_sq=0.4, lam=1.0)

        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )
        if self.action_mode == "continuous":
            self.action_low = 0.0
            self.action_high = self.max_interview_cost
            self.action_size = 1
            self.idle_action = None
            self._action_spaces = {
                agent: Box(
                    low=np.array([self.action_low], dtype=np.float32),
                    high=np.array([self.action_high], dtype=np.float32),
                    dtype=np.float32,
                )
                for agent in [f"company_{i}" for i in range(num_companies)]
            }
        else:
            self.action_low = 0.0
            self.action_high = float(max_interview_cost)
            self.action_size = max(2, num_interview_cost_levels)
            self.idle_action = 0
            self._action_spaces = {
                agent: Discrete(self.action_size) for agent in [f"company_{i}" for i in range(num_companies)]
            }

        obs_size = (
            num_workers * ability_dim  # sigma_hat
            + num_workers  # experience
            + num_workers  # tenure
            + num_workers  # employed_by
            + num_workers  # wages
            + num_workers * ability_dim  # belief_mean
            + num_workers * ability_dim  # belief_var (placeholder zeros)
            + num_workers  # own_workforce
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
        self.obs_size = obs_size
        self._observation_spaces = {f"company_{i}": obs_space for i in range(num_companies)}

        self.agents = [f"company_{i}" for i in range(num_companies)]
        self.possible_agents = self.agents.copy()
        self.rng = np.random.RandomState(seed)

        self._init_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def _capacity(self, firm_idx: int) -> int:
        if self.firm_capacities is None:
            return self.max_workers_per_company
        return int(self.firm_capacities[firm_idx])

    def _wage_multiplier(self, firm_idx: int) -> float:
        if firm_idx < len(self.firm_types):
            return float(self.firm_type_premia.get(self.firm_types[firm_idx], 1.0))
        return 1.0

    def _generate_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_size, dtype=np.int8)
        return mask

    def _cost_from_action(self, action: Any) -> float:
        if self.action_mode == "discrete":
            idx = int(np.clip(int(action), 0, len(self.cost_levels) - 1))
            return float(self.cost_levels[idx])
        arr = np.asarray(action).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(np.clip(arr[0], self.action_low, self.action_high))

    def _init_state(self):
        self.timestep = 0
        self.sigma_true = self.rng.randn(self.num_workers).astype(np.float32)
        self.sigma_hat = ScreeningMechanism.init_sigma_hat(
            sigma_true=self.sigma_true,
            noise_std=np.sqrt(self.public_signal_variance),
            rng=self.rng,
        ).reshape(-1)
        self.sigma_tilde = np.tile(self.sigma_hat, (self.num_companies, 1)).astype(np.float32)
        base_var = self.screening.interview_var(0.0)
        self.interview_vars = np.full((self.num_companies, self.num_workers), base_var, dtype=np.float32)
        self.experience = np.zeros(self.num_workers, dtype=np.float32)
        self.tenure = np.zeros(self.num_workers, dtype=np.float32)
        self.employed_by = np.full(self.num_workers, -1, dtype=int)
        self.wages = np.zeros(self.num_workers, dtype=np.float32)
        self.company_profits: Dict[str, List[float]] = {agent: [0.0] for agent in self.agents}
        self.last_step_finance = {
            agent: {
                "profit": 0.0,
                "wage": 0.0,
                "screening_cost": 0.0,
                "reward": 0.0,
            }
            for agent in self.agents
        }

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        firm_idx = int(agent.split("_")[1])
        belief_mean = self.sigma_tilde[firm_idx]
        belief_var = np.zeros_like(belief_mean)
        own_workforce = (self.employed_by == firm_idx).astype(np.float32)
        own_profit = np.array([self.company_profits[agent][-1]], dtype=np.float32)

        obs = np.concatenate(
            [
                self.sigma_hat.reshape(-1, self.ability_dim).flatten(),
                self.experience,
                self.tenure,
                self.employed_by.astype(np.float32),
                self.wages,
                belief_mean.reshape(-1, self.ability_dim).flatten(),
                belief_var.reshape(-1, self.ability_dim).flatten(),
                own_workforce,
                own_profit,
            ]
        ).astype(np.float32)

        return {
            "observation": obs,
            "action_mask": self._generate_action_mask(),
        }

    def _get_info(self, agent: str) -> Dict[str, Any]:
        firm_idx = int(agent.split("_")[1])
        workforce_size = int(np.sum(self.employed_by == firm_idx))
        return {
            "workforce_size": workforce_size,
            "unemployment_rate": float(np.mean(self.employed_by < 0)),
            "avg_wage": float(np.mean(self.wages)) if self.wages.size else 0.0,
            "timestep": self.timestep,
            "last_step_profit": float(self.last_step_finance[agent]["profit"]),
            "last_step_wage": float(self.last_step_finance[agent]["wage"]),
            "last_step_screening_cost": float(self.last_step_finance[agent]["screening_cost"]),
            "last_step_reward": float(self.last_step_finance[agent]["reward"]),
            "worker_metrics": [
                {
                    "worker_id": j,
                    "sigma_hat": float(self.sigma_hat[j]),
                    "sigma_tilde": float(self.sigma_tilde[firm_idx, j]),
                    "sigma_true": float(self.sigma_true[j]),
                    "experience": float(self.experience[j]),
                    "interview_cost": float(self.interview_vars[firm_idx, j]),
                    "profit": 0.0,
                    "wage": float(self.wages[j]),
                }
                for j in range(self.num_workers)
            ],
        }

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._init_state()
        self.agents = self.possible_agents.copy()
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Map actions -> interview cost
        costs = {agent: self._cost_from_action(actions.get(agent, self.idle_action)) for agent in self.agents}
        screening_costs = {agent: 0.0 for agent in self.agents}

        interviewed_mask = np.zeros((self.num_companies, self.num_workers), dtype=bool)
        unemployed_workers = np.where(self.employed_by < 0)[0]

        # 1) Interview: draw from sigma_hat, update via interview0
        for firm_idx, agent in enumerate(self.agents):
            remaining_capacity = max(0, self._capacity(firm_idx) - int(np.sum(self.employed_by == firm_idx)))
            if remaining_capacity <= 0 or unemployed_workers.size == 0:
                continue
            n_interviews = min(
                unemployed_workers.size,
                max(1, int(np.ceil(0.5 * remaining_capacity))),
            )
            candidates_sorted = unemployed_workers[np.argsort(self.sigma_hat[unemployed_workers])[::-1]]
            candidates = candidates_sorted[:n_interviews]
            cost_val = float(costs[agent])
            for worker_id in candidates:
                sigma_tilde_draw = self.screening.screen_worker(
                    sigma_true=np.array([self.sigma_true[worker_id]], dtype=np.float32),
                    interview_costs=np.array([cost_val], dtype=np.float32),
                    sigma_hat=np.array([self.sigma_hat[worker_id]], dtype=np.float32),
                )[0]
                self.sigma_tilde[firm_idx, worker_id] = sigma_tilde_draw
                self.interview_vars[firm_idx, worker_id] = float(self.screening.interview_var(cost_val))
                interviewed_mask[firm_idx, worker_id] = True
                screening_costs[agent] += cost_val

        # 2) Matching: firms offer to 20% of the workers they interviewed
        remaining_capacity = [
            max(0, self._capacity(i) - int(np.sum(self.employed_by == i)))
            for i in range(self.num_companies)
        ]
        matching_result: WageMatchingResult = greedy_wage_matching_from_signals(
            sigma_tilde=self.sigma_tilde,
            interviewed_mask=interviewed_mask,
            capacities=remaining_capacity,
            eligible_workers=unemployed_workers,
            v_x=self.initial_offer_vx,
            g=default_g_bounded,
            firm_multipliers=[self._wage_multiplier(i) for i in range(self.num_companies)],
        )

        for worker_id, firm_id in matching_result.worker_to_firm.items():
            if firm_id is None:
                continue
            if self.employed_by[worker_id] != -1:
                continue
            if remaining_capacity[firm_id] <= 0:
                continue
            wage_offer = matching_result.worker_wage.get(worker_id, 0.0)
            self.employed_by[worker_id] = firm_id
            self.wages[worker_id] = float(wage_offer) * self.wage_scale * self._wage_multiplier(firm_id)
            self.tenure[worker_id] = 0.0
            remaining_capacity[firm_id] -= 1

        # 3) Profit draw + wage adjustment + belief/experience update
        profits_per_worker = generate_profit_array(
            exp_tm1=self.experience,
            sigma_j=self.sigma_true,
            employed_by=self.employed_by,
            g0=self.g0,
            g1=self.g1,
            theta=self.profit_theta,
            delta_eps_sq=self.delta_eps_sq,
            rng=self.rng,
        )

        for worker_id, firm_id in enumerate(self.employed_by):
            if firm_id < 0:
                continue
            delta_interview_sq = float(self.interview_vars[firm_id, worker_id])
            wage_res = adjust_wage_post_hire(
                sigma_tilde_initial=float(self.sigma_tilde[firm_id, worker_id]),
                p_ij_tm1=float(profits_per_worker[worker_id]),
                exp_t=float(self.experience[worker_id]),
                delta_interview_sq=delta_interview_sq,
                delta_eps_sq=self.delta_eps_sq,
                psi=self.wage_profit_share,
                g=default_g_bounded,
            )
            self.wages[worker_id] = float(wage_res.wage_t) * self.wage_scale * self._wage_multiplier(firm_id)

        (
            self.sigma_tilde,
            self.sigma_hat,
            self.experience,
            vx_per_worker,
        ) = update_beliefs_and_experience(
            sigma_tilde=self.sigma_tilde,
            sigma_hat=self.sigma_hat,
            sigma_true=self.sigma_true,
            employed_by=self.employed_by,
            experience=self.experience,
            profits=profits_per_worker,
            interview_vars=self.interview_vars,
            delta_eps_sq=self.delta_eps_sq,
            g0=self.g0,
            g1=self.g1,
            theta=self.profit_theta,
        )

        self.tenure = self.tenure + (self.employed_by >= 0).astype(np.float32)

        total_profits = {agent: 0.0 for agent in self.agents}
        total_wages = {agent: 0.0 for agent in self.agents}

        for worker_id, firm_id in enumerate(self.employed_by):
            if firm_id < 0:
                continue
            agent = f"company_{firm_id}"
            total_profits[agent] += float(profits_per_worker[worker_id])
            total_wages[agent] += float(self.wages[worker_id])

        rewards = {}
        for agent in self.agents:
            reward = total_profits[agent] - total_wages[agent] - screening_costs[agent]
            rewards[agent] = float(reward)
            self.company_profits[agent].append(float(reward))
            self.last_step_finance[agent] = {
                "profit": float(total_profits[agent]),
                "wage": float(total_wages[agent]),
                "screening_cost": float(screening_costs[agent]),
                "reward": float(reward),
            }

        self.timestep += 1

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_timesteps for agent in self.agents}

        if self.timestep >= self.max_timesteps:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n{'=' * 40}")
        print(f"Time Step: {self.timestep}/{self.max_timesteps}")
        print(f"Unemployment Rate: {np.mean(self.employed_by < 0):.2%}")
        for agent in self.possible_agents:
            firm_idx = int(agent.split("_")[1])
            workforce = int(np.sum(self.employed_by == firm_idx))
            recent_profit = self.company_profits[agent][-1] if self.company_profits[agent] else 0.0
            total_profit = float(sum(self.company_profits[agent]))
            print(f"{agent}: workforce={workforce}, last_reward={recent_profit:.2f}, total_reward={total_profit:.2f}")
        print(f"{'=' * 40}\n")

    def close(self):
        return None
