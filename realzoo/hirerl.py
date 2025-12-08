from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from gymnasium.spaces import Box, Dict as GymDict, MultiBinary, Discrete
from pettingzoo.utils.env import ParallelEnv

from interview0 import ScreeningMechanism
from matching1 import firm_offer, worker_wage_accepted, FirmWageOffers, FinalOffers
from generated_profit2 import (
    generate_profit_array,
    update_sigma_tilde_from_profit,
    update_sigma_hat_accepted,
    update_sigma_no_offer,
    update_experience, normal_between_0_1,
)
from post_hiring_adjust_wage3 import default_g_bounded, adjust_wage_post_hire, firing_decision


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
        firing_cost_multiplier: float = 1.0,
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
        self.firing_cost_multiplier = firing_cost_multiplier
        self.offer_accept_greedy_chance = 0.8

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
            self.action_size = num_workers  # Now a vector: one cost per worker
            self.idle_action = None
            self._action_spaces = {
                agent: Box(
                    low=np.zeros(num_workers, dtype=np.float32),
                    high=np.full(num_workers, self.max_interview_cost, dtype=np.float32),
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

    def _costs_from_action(self, action: Any) -> np.ndarray:
        """Extract per-worker interview costs from action.

        Returns:
            Array of shape (num_workers,) with interview cost for each worker
        """
        if self.action_mode == "discrete":
            # For discrete mode, action is still a scalar - apply to all workers
            idx = int(np.clip(int(action), 0, len(self.cost_levels) - 1))
            cost = float(self.cost_levels[idx])
            return np.full(self.num_workers, cost, dtype=np.float32)

        # Continuous mode: action is a vector of costs per worker
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros(self.num_workers, dtype=np.float32)
        if arr.size != self.num_workers:
            # Fallback: broadcast or pad
            if arr.size == 1:
                return np.full(self.num_workers, float(arr[0]), dtype=np.float32)
            else:
                arr = np.resize(arr, self.num_workers)
        return np.clip(arr, self.action_low, self.action_high).astype(np.float32)

    def _init_state(self):
        self.timestep = 0
        # limits sigma_true to [0,1]
        self.sigma_true = np.array([normal_between_0_1(self.rng) for _ in range(self.num_workers)]).astype(np.float32)
        self.sigma_hat = ScreeningMechanism.init_sigma_hat(
            sigma_true=self.sigma_true,
            noise_std=np.sqrt(self.public_signal_variance),
            rng=self.rng,
        ).reshape(-1)
        self.sigma_tilde = np.tile(self.sigma_hat, (self.num_companies, 1)).astype(np.float32)
        base_var = self.screening.interview_var(0.0)
        self.interview_vars = np.full((self.num_companies, self.num_workers), base_var, dtype=np.float32)
        self.experience = np.ones(self.num_workers, dtype=np.float32)
        self.tenure = np.zeros(self.num_workers, dtype=np.float32)
        self.employed_by = np.full(self.num_workers, -1, dtype=int)
        self.wages = np.zeros(self.num_workers, dtype=np.float32)
        self.company_profits: Dict[str, List[float]] = {agent: [0.0] for agent in self.possible_agents}
        self.last_step_finance = {
            agent: {
                "profit": 0.0,
                "wage": 0.0,
                "screening_cost": 0.0,
                "firing_cost": 0.0,
                "reward": 0.0,
            }
            for agent in self.possible_agents
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
            "last_step_firing_cost": float(self.last_step_finance[agent]["firing_cost"]),
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

        base_var = self.screening.interview_var(0.0)
        self.interview_vars = np.full((self.num_companies, self.num_workers), base_var, dtype=np.float32)

        # Map actions -> per-worker interview costs
        # costs_per_worker[agent] = array of shape (num_workers,)
        costs_per_worker = {
            agent: self._costs_from_action(actions.get(agent, self.idle_action))
            for agent in self.agents
        }
        screening_costs = {agent: 0.0 for agent in self.agents}

        interviewed_mask = np.zeros((self.num_companies, self.num_workers), dtype=bool)
        unemployed_workers = np.where(self.employed_by < 0)[0]

        # 1) Interview with per-worker costs
        # Each firm can interview any worker; multiple firms can interview the same worker
        # Each firm that interviews a worker updates their own private belief σ̃_{ij}
        self.step_interview(costs_per_worker, interviewed_mask, screening_costs, unemployed_workers)

        # 2) Matching: firms offer to 30% of the workers they interviewed
        self.step_offers(interviewed_mask, unemployed_workers)

        # 3) Profit draw + wage adjustment + belief/experience update
        # Create binary mask: 1 if employed by any firm, 0 if unemployed
        employed_mask = (self.employed_by >= 0).astype(np.int8)
        profits_per_worker = generate_profit_array(
            exp_tm1=self.experience,
            sigma_true=self.sigma_true,
            employed_by=employed_mask,
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
            wage = adjust_wage_post_hire(
                sigma_tilde_interview=float(self.sigma_tilde[firm_id, worker_id]),
                p_ij_tm1=float(profits_per_worker[worker_id]),
                exp_t=self.experience[worker_id],
                delta_interview_sq=delta_interview_sq,
                delta_eps_sq=self.delta_eps_sq,
                psi=self.wage_profit_share,
                sigma_true=float(self.sigma_true[worker_id]),
                g_fn=default_g_bounded,
            )
            self.wages[worker_id] = float(wage) * self.wage_scale

        # 4) Firing decision: fire workers with excessive losses
        firing_costs = {agent: 0.0 for agent in self.agents}
        for worker_id, firm_id in enumerate(self.employed_by):
            if firm_id < 0:
                continue
            agent = f"company_{firm_id}"
            profit = float(profits_per_worker[worker_id])
            wage = float(self.wages[worker_id])

            fire_result = firing_decision(
                profit=profit,
                wage=wage,
                c_fire_t=0.0,
            )

            if fire_result.fire:
                # Fire the worker: set employment status to unemployed
                c_fire = self.firing_cost_multiplier * wage
                self.employed_by[worker_id] = -1
                self.tenure[worker_id] = 0.0
                self.wages[worker_id] = 0.0
                # Add firing cost (severance payment)
                firing_costs[agent] += c_fire

        # 5) Update beliefs and experience using new separate functions
        sigma_hat_next = self.sigma_hat.copy()
        sigma_tilde_next = self.sigma_tilde.copy()

        for worker_id in range(self.num_workers):
            firm_id = self.employed_by[worker_id]

            if firm_id < 0:
                # worker unemployed
                sigma_hat_no_offer, sigma_tilde_no_offer = update_sigma_no_offer(self.sigma_hat[worker_id])
                sigma_hat_next[worker_id] = float(sigma_hat_no_offer)
                # all firms set their sigma_tilde to sigma_tilde_no_offer
                sigma_tilde_next[:, worker_id] = float(sigma_tilde_no_offer)
            else:
                # worker employed, update beliefs based on profit
                delta_interview_sq = float(self.interview_vars[firm_id, worker_id])

                sigma_tilde_new, sigma_update, vx = update_sigma_tilde_from_profit(
                    sigma_tilde_interview=float(self.sigma_tilde[firm_id, worker_id]),
                    sigma_true=float(self.sigma_true[worker_id]),
                    exp_t=self.experience[worker_id],
                    delta_interview_sq=delta_interview_sq,
                    delta_eps_sq=self.delta_eps_sq,
                )

                # update hiring firm's sigma_tilde
                sigma_tilde_next[firm_id, worker_id] = float(sigma_tilde_new)

                # update public signal sigma_hat
                sigma_hat_next[worker_id] = float(
                    update_sigma_hat_accepted(
                        sigma_tilde=float(self.sigma_tilde[firm_id, worker_id]),
                        sigma_update=float(sigma_update),
                    )
                )

        self.sigma_hat = sigma_hat_next
        self.sigma_tilde = sigma_tilde_next

        # update experience for all workers
        for firm_idx in range(self.num_companies):
            employed_mask = (self.employed_by == firm_idx).astype(np.int8)
            self.experience = update_experience(
                exp_t=self.experience,
                sigma_true=self.sigma_true,
                employed_by=employed_mask,
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
            reward = total_profits[agent] - total_wages[agent] - screening_costs[agent] - firing_costs[agent]
            rewards[agent] = float(reward)
            self.company_profits[agent].append(float(reward))
            self.last_step_finance[agent] = {
                "profit": float(total_profits[agent]),
                "wage": float(total_wages[agent]),
                "screening_cost": float(screening_costs[agent]),
                "firing_cost": float(firing_costs[agent]),
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

    def step_offers(self, interviewed_mask: np.ndarray[tuple[int, int], np.dtype[Any]],
                    unemployed_workers: np.ndarray[tuple[Any, ...], np.dtype[np.int32 | np.int64]]):
        remaining_capacity = [
            max(0, self._capacity(i) - int(np.sum(self.employed_by == i)))
            for i in range(self.num_companies)
        ]

        # Generate offers from each firm
        firm_offers = []
        for firm_idx in range(self.num_companies):
            offer = firm_offer(
                sigma_tilde=self.sigma_tilde[firm_idx],
                interviewed_mask=interviewed_mask[firm_idx],
                offer_rate=0.3,
                g=default_g_bounded,
                firm_multiplier=self._wage_multiplier(firm_idx),
                firm_id=firm_idx,
            )
            firm_offers.append(offer)

        # Workers accept best offers
        matching_results = worker_wage_accepted(firm_offers=firm_offers, num_workers=self.num_workers, rnd=self.rng, greedy_chance=self.offer_accept_greedy_chance)

        # Process accepted offers
        for result in matching_results:
            firm_id = result.firm_id
            new_hires_mask = result.employeed_by.astype(bool)

            for worker_id in np.where(new_hires_mask)[0]:
                # Check constraints
                if self.employed_by[worker_id] != -1:
                    continue
                if worker_id not in unemployed_workers:
                    continue
                if remaining_capacity[firm_id] <= 0:
                    continue

                # Get wage from firm's offer
                wage_offer = firm_offers[firm_id].wage_array[worker_id]

                # Hire the worker
                self.employed_by[worker_id] = firm_id
                self.wages[worker_id] = float(wage_offer) * self.wage_scale
                self.tenure[worker_id] = 0.0
                remaining_capacity[firm_id] -= 1

    def step_interview(self, costs_per_worker: dict[Any, np.ndarray[tuple[Any, ...], np.dtype[Any]]],
                       interviewed_mask: np.ndarray[tuple[int, int], np.dtype[Any]], screening_costs: dict[Any, float],
                       unemployed_workers: np.ndarray[tuple[Any, ...], np.dtype[np.int32 | np.int64]]):
        for firm_idx, agent in enumerate(self.agents):
            remaining_capacity = max(0, self._capacity(firm_idx) - int(np.sum(self.employed_by == firm_idx)))
            if remaining_capacity <= 0:
                continue

            worker_costs = costs_per_worker[agent]  # (num_workers,)

            # Firm interviews any unemployed worker with cost > 0
            for worker_id in unemployed_workers:
                cost_val = float(worker_costs[worker_id])
                if cost_val > 0.0:  # Firm wants to interview this worker
                    # Generate firm i's private signal about worker j: σ̃_{ij}
                    # This is independent across firms (each gets their own noisy assessment)
                    sigma_tilde_draw = self.screening.screen_worker(
                        sigma_true=np.array([self.sigma_true[worker_id]], dtype=np.float32),
                        interview_costs=np.array([cost_val], dtype=np.float32),
                        sigma_hat=np.array([self.sigma_hat[worker_id]], dtype=np.float32),
                    )[0]

                    # Update firm i's private belief about worker j
                    self.sigma_tilde[firm_idx, worker_id] = sigma_tilde_draw
                    self.interview_vars[firm_idx, worker_id] = float(self.screening.interview_var(cost_val))
                    interviewed_mask[firm_idx, worker_id] = True
                    screening_costs[agent] += cost_val

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
