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
from gymnasium.spaces import Box, Dict as GymDict, MultiBinary, Discrete
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
        firm_capacities: Optional[List[int]] = None,
        gamma: float = 0.1,
        g0: float = 0.1,
        g1: float = 0.05,
        experience_theta: float = 0.2,
        base_firing_cost: float = 1.0,
        base_screening_cost: float = 0.5,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
        profit_theta: float = 0.05,
        profit_noise_var: float = 0.05,
        profit_function_type: str = "diminishing",
        wage_profit_share: float = 0.5,
        initial_offer_vx: float = 0.0,
        max_timesteps: int = 100,
        firm_types: Optional[List[str]] = None,
        firm_type_premia: Optional[Dict[str, float]] = None,
        module_toggles: Optional[Dict[str, bool]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the job-market environment.

        Key tensor shapes:
            - Worker belief/state arrays live in `(num_workers, ability_dim)` form.
            - Observations flatten these arrays into a 1-D vector of length
              `obs_size = num_workers * (ability_dim*3 + 5) + num_workers + 1`
              (see `_get_obs` for the exact breakdown).
            - Continuous action space is a Box with shape `(1,)`; discrete mode
              uses `Discrete(num_interview_cost_levels)`.
        """
        super().__init__()

        self.render_mode = render_mode
        self.num_companies = num_companies
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_workers_per_company = max_workers_per_company
        if firm_capacities is not None:
            if len(firm_capacities) != num_companies:
                raise ValueError("firm_capacities length must equal num_companies")
            if min(firm_capacities) <= 0:
                raise ValueError("firm_capacities must all be positive")
            self.firm_capacities = [int(x) for x in firm_capacities]
        else:
            self.firm_capacities = None

        self.gamma = gamma
        self.g0 = g0
        self.g1 = g1

        self.base_firing_cost = base_firing_cost
        self.base_screening_cost = base_screening_cost
        self.max_interview_cost = max_interview_cost
        self.num_interview_cost_levels = max(2, num_interview_cost_levels)
        self.action_mode = action_mode.lower()
        if self.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be 'continuous' or 'discrete'")

        self.wage_profit_share = wage_profit_share
        self.initial_offer_vx = float(np.clip(initial_offer_vx, 0.0, 0.99))
        self.profit_theta = profit_theta
        self.delta_eps_sq = profit_noise_var
        self.profit_function_type = profit_function_type
        self.max_timesteps = max_timesteps

        # Firm type tags (for wage premia) and premia map
        if firm_types is not None:
            if len(firm_types) != num_companies:
                raise ValueError("firm_types length must equal num_companies")
            self.firm_types = list(firm_types)
        else:
            self.firm_types = ["generic"] * num_companies

        default_premia = {"small": 1.0, "medium": 1.0, "large": 1.0, "generic": 1.0}
        if firm_type_premia:
            default_premia.update(firm_type_premia)
        self.firm_type_premia = default_premia

        self.rng = np.random.RandomState(seed)

        self.worker_pool = WorkerPool(
            num_workers=num_workers,
            ability_dim=ability_dim,
            gamma=gamma,
            g0=g0,
            g1=g1,
            seed=seed,
        )

        self.screening = ScreeningMechanism(
            delta0_sq=0.4,
            lam=1.0,
        )

        default_modules = {
            "wage_adjustment": True,
            "interview": True,
            "matching": True,
            "production": True,
            "experience": True,
            "firing": True,
        }
        if module_toggles:
            default_modules.update(module_toggles)
        self.module_toggles = default_modules

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
        self._current_interview_costs = _init_matrix(0.0)
        self._last_vx = _init_matrix(0.0)
        self._last_k1 = _init_matrix(0.0)

        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            self.num_interview_cost_levels,
            dtype=np.float32,
        ) #这还是discrete的。

        if self.action_mode == "continuous":
            self.action_low = 0.0
            self.action_high = float(self.max_interview_cost)
            self.action_size = 1
            self.idle_action = None
            self._action_spaces = {
                agent: Box(
                    low=np.array([self.action_low], dtype=np.float32),
                    high=np.array([self.action_high], dtype=np.float32),
                    dtype=np.float32,
                )
                for agent in self.agents
            }
        else:
            self.action_low = 0.0
            self.action_high = float(self.max_interview_cost)
            self.action_size = self.num_interview_cost_levels
            self.idle_action = 0
            self._action_spaces = {agent: Discrete(self.action_size) for agent in self.agents}

        obs_size = (
            num_workers * ability_dim  # sigma_hat public belief.
            + num_workers  # experience
            + num_workers  # tenure
            + num_workers  # employed_by
            + num_workers  # wages
            + num_workers * ability_dim  # private belief,
            + num_workers * ability_dim  # private belief variance.
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
        self.obs_size = obs_size
        self.last_step_finance = {
            agent: {
                "profit": 0.0,
                "wage": 0.0,
                "screening_cost": 0.0,
                "firing_cost": 0.0,
                "reward": 0.0,
            }
            for agent in self.agents
        }

    def _capacity(self, firm_idx: int) -> int:
        """
        Per-firm hiring capacity. Falls back to a global max when no list is provided.
        """
        if self.firm_capacities is None:
            return self.max_workers_per_company
        return self.firm_capacities[firm_idx]

    def _wage_multiplier(self, firm_idx: int) -> float:
        """Return phi_type wage multiplier for this firm."""
        if firm_idx < len(self.firm_types):
            t = self.firm_types[firm_idx]
            return float(self.firm_type_premia.get(t, 1.0))
        return 1.0

        self._observation_spaces = {agent: obs_space for agent in self.agents}

        self.timestep = 0
        self.company_profits: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def observation_space(self, agent: str):
        """Return the Dict observation space (`observation` vector + action mask) for `agent`."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        """Return the action space (Box or Discrete) for `agent`."""
        return self._action_spaces[agent]

    def _company_index(self, agent: str) -> int:
        """Extract integer firm index from agent name like 'company_2'."""
        return int(agent.split("_")[1])

    def _deterministic_interview_assignments(self) -> Dict[int, int]:
        """
        Assign each firm to the highest public-signal unemployed worker.

        Returns:
            Mapping `firm_idx -> worker_id` for firms that receive an interview.
            Only unemployed workers are considered, sorted by public signal.
        """
        public = self.worker_pool.get_public_state()
        sigma_hat = public["sigma_hat"][:, 0] #shape is (10,)

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
            if current_workforce >= self._capacity(firm_idx):
                continue
            if idx >= len(unemployed_sorted):
                break
            assignments[firm_idx] = unemployed_sorted[idx]
            idx += 1

        return assignments

    def _cost_from_action(self, action: Any) -> float:
        """
        Map a policy action (index or float) into an interview cost scalar.

        Returns:
            float cost in [action_low, action_high]; if discrete, looks up
            `cost_levels[idx]`.

            no matter is discrete or cns, it returns a scalar (which is correct).
        """
        if self.action_mode == "discrete":
            idx = int(np.clip(int(action), 0, self.action_size - 1))
            return float(self.cost_levels[idx])
        if isinstance(action, (list, tuple, np.ndarray)):
            value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        else:
            value = float(action)
        return float(np.clip(value, self.action_low, self.action_high))

    def _compute_vx(self, exp_t: float, delta_interview_sq: float) -> float:
        """
        Compute the weight `v_x` used in belief/wage updates.

        Args:
            exp_t: scalar experience at time t.
            delta_interview_sq: interview noise variance.
        Returns:
            Scalar between 0 and 1.
        """
        exp_t = max(0.0, float(exp_t))
        denom = delta_interview_sq + self.delta_eps_sq
        if denom <= 0.0:
            return 0.0
        K1 = delta_interview_sq / denom
        if K1 <= 0.0:
            return 0.0
        return (exp_t * K1) / (1.0 + (exp_t - 1.0) * K1)

    def _update_wages_existing_employees(self):
        """
        Apply step (5) wage rule for continuing employees before new hires.

        Iterates over each current worker and updates their wage using the last
        stored interview signal (`_interview_signal_at_hire`) and profit (`_last_profit`).
        """
        for agent in self.agents:
            company_idx = self._company_index(agent)
            workforce = self.worker_pool.get_employed_by_company(company_idx)

            for worker_id in workforce:
                worker = self.worker_pool.workers[worker_id]
                last_profit = float(self._last_profit[agent][worker_id])
                sigma_tilde_interview = float(self._interview_signal_at_hire[agent][worker_id])
                delta_interview_sq = float(self._interview_vars[agent][worker_id])

                result = adjust_wage_post_hire(
                    sigma_tilde_interview=sigma_tilde_interview,
                    p_ij_tm1=last_profit,
                    exp_t=worker.experience,
                    delta_interview_sq=delta_interview_sq,
                    delta_eps_sq=self.delta_eps_sq,
                    psi=self.wage_profit_share,
                    profit_norm_method="tanh",
                    profit_norm_scale=500,
                )

                phi = self._wage_multiplier(company_idx)
                worker.wage = result.wage_t * phi

    def _generate_action_mask(self, agent: str) -> np.ndarray:
        """
        Build the valid-action mask for `agent`.

        Returns:
            In continuous mode: all ones of shape `(1,)`.
            In discrete mode: binary vector of length `action_size` where only
            NO_OP is valid for firms without an assignment.
        """
        if self.action_mode == "continuous":
            return np.ones(self.action_size, dtype=np.int8)

        assignments = self._deterministic_interview_assignments()
        company_idx = self._company_index(agent)
        mask = np.zeros(self.action_size, dtype=np.int8)

        if company_idx not in assignments:
            mask[self.idle_action] = 1
        else:
            mask[:] = 1
        return mask

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Construct observation dictionary for `agent`.

        The flattened observation concatenates:
            sigma_hat (num_workers * ability_dim),
            experience (num_workers),
            tenure (num_workers),
            employed_by indicators (num_workers),
            wages (num_workers),
            belief mean/var (each num_workers * ability_dim),
            own_workforce indicator (num_workers),
            recent profit (1).
        """
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
                public["sigma_hat"].flatten(),  # (num_workers, ability_dim) -> (num_workers*ability_dim,)
                public["experience"],  # (num_workers,)
                public["tenure"].astype(np.float32),  # (num_workers,)
                public["employed_by"].astype(np.float32),  # (num_workers,)
                public["wages"],  # (num_workers,)
                beliefs.belief_mean.flatten(),  # (num_workers, ability_dim) -> (num_workers*ability_dim,)
                beliefs.belief_var.flatten(),  # (num_workers, ability_dim) -> (num_workers*ability_dim,)
                own_workforce,  # (num_workers,)
                own_profit,  # (1,)
            ]
        ).astype(np.float32)

        return {
            "observation": obs,
            "action_mask": self._generate_action_mask(agent),
        }

    def set_module_toggles(self, **overrides: bool) -> None:
        """Enable or disable pipeline modules (interview, matching, etc.)."""
        for key, value in overrides.items():
            if key in self.module_toggles:
                self.module_toggles[key] = bool(value)

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """
        Collect diagnostic info for `agent`.

        Includes workforce size, aggregate profits, unemployment stats,
        current timestep, and per-worker metrics (see `worker_metrics` list).
        """
        company_idx = self._company_index(agent)
        workforce = self.worker_pool.get_employed_by_company(company_idx)
        public = self.worker_pool.get_public_state()
        metrics = []
        for worker_id in range(self.num_workers):
            sigma_hat = public["sigma_hat"][worker_id]
            metrics.append(
                {
                    "worker_id": worker_id,
                    "public_tenure": float(public["tenure"][worker_id]),
                    "sigma_hat": float(sigma_hat[0]),
                    "experience": float(self.worker_pool.workers[worker_id].experience),
                    "interview_cost": float(self._current_interview_costs[agent][worker_id]),
                    "profit": float(self._last_profit[agent][worker_id]),
                    "sigma_tilde": float(self.firm_beliefs[agent].belief_mean[worker_id, 0]),
                    "sigma_true": float(self.worker_pool.workers[worker_id].sigma_true[0]),
                    "wage": float(self.worker_pool.workers[worker_id].wage),
                    "vx": float(self._last_vx[agent][worker_id]),
                    "k1": float(self._last_k1[agent][worker_id]),
                }
            )
        return {
            "workforce_size": len(workforce),
            "total_profit": float(sum(self.company_profits[agent])),
            "unemployment_rate": self.worker_pool.get_unemployment_rate(),
            "avg_wage": self.worker_pool.get_average_wage(),
            "timestep": self.timestep,
            "last_step_profit": float(self.last_step_finance[agent]["profit"]),
            "last_step_wage": float(self.last_step_finance[agent]["wage"]),
            "last_step_screening_cost": float(self.last_step_finance[agent]["screening_cost"]),
            "last_step_firing_cost": float(self.last_step_finance[agent]["firing_cost"]),
            "last_step_reward": float(self.last_step_finance[agent]["reward"]),
            "worker_metrics": metrics,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment state and return initial observations/infos.

        Re-samples worker pool, reinitializes beliefs, and assigns each firm
        a random workforce (up to `max_workers_per_company`).
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.worker_pool.reset(seed=seed)
        public_state = self.worker_pool.get_public_state()

        base_var = self.screening.interview_var(0.0)

        self.agents = self.possible_agents.copy()
        self.timestep = 0
        self.company_profits = {agent: [0.0] for agent in self.agents}
        self.last_step_finance = {
            agent: {
                "profit": 0.0,
                "wage": 0.0,
                "screening_cost": 0.0,
                "firing_cost": 0.0,
                "reward": 0.0,
            }
            for agent in self.agents
        }

        for agent in self.agents:
            beliefs = FirmBeliefs(num_workers=self.num_workers, ability_dim=self.ability_dim)
            for worker_id in range(self.num_workers):
                sigma_hat_init = public_state["sigma_hat"][worker_id]
                init_val = float(sigma_hat_init[0])
                beliefs.initialize_from_interview_signal(worker_id, init_val, signal_noise_var=base_var)
            self.firm_beliefs[agent] = beliefs
            sigma_hat_flat = public_state["sigma_hat"].flatten().astype(np.float32)
            self._interview_signal_at_hire[agent] = sigma_hat_flat.copy()
            self._interview_vars[agent] = np.full(self.num_workers, base_var, dtype=np.float32)
            self._last_profit[agent] = np.zeros(self.num_workers, dtype=np.float32)
            self._current_interview_costs[agent] = np.zeros(self.num_workers, dtype=np.float32)
            self._last_vx[agent] = np.zeros(self.num_workers, dtype=np.float32)
            self._last_k1[agent] = np.zeros(self.num_workers, dtype=np.float32)

        # --- Assign each firm a random initial workforce drawn from the worker pool ---
        all_workers = list(range(self.num_workers))
        self.rng.shuffle(all_workers)
        ptr = 0
        for firm_idx in range(self.num_companies):
            if ptr >= len(all_workers):
                break
            max_assignable = min(self._capacity(firm_idx), len(all_workers) - ptr)
            init_n = self.rng.randint(0, max_assignable + 1)
            for _ in range(init_n):
                if ptr >= len(all_workers):
                    break
                worker_id = all_workers[ptr]
                ptr += 1
                initial_wage = 0.0
                self.worker_pool.hire_worker(worker_id, firm_idx, initial_wage)
                agent_name = f"company_{firm_idx}"
                self._interview_vars[agent_name][worker_id] = base_var

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        # record firings during this step
        for agent, info in infos.items():
            info["firings"] = []
            info["hirings"] = []
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
        """
        Execute one environment timestep.

        Args:
            actions: dict mapping agent name to either interview cost (float) or
                     discrete index, depending on action mode.
        Returns:
            tuple of (observations, rewards, terminations, truncations, infos),
            each a dict keyed by agent names.
        """
        decoded_actions: Dict[str, float] = {}
        for agent in self.agents:
            default = 0 if self.action_mode == "discrete" else 0.0
            decoded_actions[agent] = self._cost_from_action(actions.get(agent, default))

        for agent in self.agents:
            self._current_interview_costs[agent].fill(0.0)

        prev_state = [
            {
                "experience": worker.experience,
                "employed": worker.employed_by >= 0,
                "sigma": float(worker.sigma_true[0])
            }
            for worker in self.worker_pool.workers
        ]

        if self.module_toggles["wage_adjustment"]:
            self._update_wages_existing_employees()

        assignments: Dict[int, int] = {}
        if self.module_toggles["interview"]:
            assignments = self._deterministic_interview_assignments()
        screening_costs = {agent: 0.0 for agent in self.agents}
        tilde_matrix = np.full((self.num_companies, self.num_workers), -np.inf, dtype=np.float32)
        targeted_workers: set[int] = set()
        hirings_record = {agent: [] for agent in self.agents}

        if self.module_toggles["interview"]:
            for agent in self.agents:
                company_idx = self._company_index(agent)
                if company_idx not in assignments:
                    continue

                worker_id = assignments[company_idx]
                worker = self.worker_pool.workers[worker_id]
                cost = decoded_actions[agent]

                screening_costs[agent] += cost
                self._current_interview_costs[agent][worker_id] = cost

                tilde_sigma, _ = self.screening.screen_worker(
                    sigma_true=worker.sigma_true,
                    sigma_hat_0=worker.sigma_hat,
                    cost=cost,
                )
                signal_scalar = float(tilde_sigma[0])
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

        if targeted_workers and self.module_toggles["matching"]:
            phi_list = [self._wage_multiplier(idx) for idx in range(self.num_companies)]
            matching_result = greedy_wage_matching_from_signals(
                tilde_sigma=tilde_matrix,
                v_x=self.initial_offer_vx,
                g=default_g_bounded,
                eligible_workers=sorted(targeted_workers),
                firm_multipliers=phi_list,
            )

            for firm_idx, worker_id in matching_result.firm_to_worker.items():
                if worker_id is None:
                    continue
                current_workforce = len(self.worker_pool.get_employed_by_company(firm_idx))
                if current_workforce >= self._capacity(firm_idx):
                    continue
                worker = self.worker_pool.workers[worker_id]
                if worker.employed_by != -1:
                    continue
                wage_offer = matching_result.worker_wage.get(worker_id)
                if wage_offer is None:
                    continue
                self.worker_pool.hire_worker(worker_id, firm_idx, wage_offer)
                agent_name = f"company_{firm_idx}"
                if agent_name in hirings_record:
                    hirings_record[agent_name].append(worker_id)

        total_profits = {agent: 0.0 for agent in self.agents}
        total_wages = {agent: 0.0 for agent in self.agents}

        if self.module_toggles["production"]:
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

                    sigma_tilde_interview = float(self._interview_signal_at_hire[agent][worker_id])
                    delta_interview_sq = float(self._interview_vars[agent][worker_id])

                    new_belief, vx = update_belief_from_profit(
                        sigma_tilde_interview=sigma_tilde_interview,
                        sigma_true=prev["sigma"],
                        sigma_hat=float(worker.sigma_hat[0]),
                        exp_t=worker.experience,
                        delta_interview_sq=delta_interview_sq,
                        delta_eps_sq=self.delta_eps_sq,
                    )
                    self.firm_beliefs[agent].belief_mean[worker_id, 0] = new_belief
                    # Store vx and k1 for diagnostics
                    if delta_interview_sq + self.delta_eps_sq > 0:
                        k1 = float(delta_interview_sq / (delta_interview_sq + self.delta_eps_sq))
                    else:
                        k1 = 0.0
                    self._last_k1[agent][worker_id] = k1
                    self._last_vx[agent][worker_id] = vx
                    self._last_profit[agent][worker_id] = profit

        if self.module_toggles["experience"]:
            self.worker_pool.update_experience_and_tenure()

        firing_costs = {agent: 0.0 for agent in self.agents}
        firings_record = {agent: [] for agent in self.agents}

        if self.module_toggles["firing"]:
            for agent in self.agents:
                company_idx = self._company_index(agent)
                workforce = list(self.worker_pool.get_employed_by_company(company_idx))

                for worker_id in workforce:
                    worker = self.worker_pool.workers[worker_id]
                    profit = float(self._last_profit[agent][worker_id])
                    wage_paid = float(worker.wage)
                    c_fire_t = 6.0 * wage_paid

                    decision = firing_decision(
                        p_ijt=profit,
                        w_ijt=wage_paid,
                        c_fire_t=c_fire_t,
                    )

                    if decision.fire:
                        self.worker_pool.fire_worker(worker_id)
                        firing_costs[agent] += c_fire_t
                        self._last_profit[agent][worker_id] = 0.0
                        self._interview_signal_at_hire[agent][worker_id] = 0.0
                        firings_record[agent].append(worker_id)

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
        for agent in self.agents:
            infos[agent]["hirings"] = hirings_record[agent]
            infos[agent]["firings"] = firings_record[agent]
        terminations = {agent: False for agent in self.agents}
        # we need these all to be true to generate the episodic_return chart
        truncations = {agent: self.timestep >= self.max_timesteps for agent in self.agents}

        if self.timestep >= self.max_timesteps:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Print a textual summary of key metrics when `render_mode='human'`."""
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
            print(f"  Workforce: {len(workforce)}/{self._capacity(company_idx)}")
            print(f"  Recent Profit: {recent_profit:.2f}")
            print(f"  Total Profit: {total_profit:.2f}")

        print(f"{'=' * 60}\n")

    def close(self):
        """PettingZoo compatibility method; nothing persistent to clean up."""
        return None
