"""
JobMarketEnv variant using Stable Matching (Gale-Shapley) instead of greedy matching.

This allows comparing greedy matching vs stable matching as baseline approaches.
"""

from typing import Dict, Tuple, Any, Optional, List

import numpy as np

from hirerl import JobMarketEnv
from matching1 import stable_matching_from_signals, WageMatchingResult
from post_hiring_adjust_wage3 import default_g_bounded


class JobMarketEnvStableMatching(JobMarketEnv):
    """
    JobMarketEnv using Gale-Shapley stable matching instead of greedy matching.

    This variant uses the stable_matching_from_signals algorithm to create
    stable matches between firms and workers, providing a baseline for comparison
    with the greedy matching approach.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "job_market_stable_v3"}

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Step function using stable matching instead of greedy matching.

        The only difference from the parent class is that we use
        stable_matching_from_signals instead of greedy_wage_matching_from_signals
        in step 2 (matching).
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Import here to avoid circular dependency
        from matching1 import stable_matching_from_signals
        from generated_profit2 import generate_profit_array, update_beliefs_and_experience
        from post_hiring_adjust_wage3 import adjust_wage_post_hire, firing_decision

        # Map actions -> per-worker interview costs
        costs_per_worker = {
            agent: self._costs_from_action(actions.get(agent, self.idle_action))
            for agent in self.agents
        }
        screening_costs = {agent: 0.0 for agent in self.agents}

        interviewed_mask = np.zeros((self.num_companies, self.num_workers), dtype=bool)
        unemployed_workers = np.where(self.employed_by < 0)[0]

        # 1) Interview with per-worker costs
        for firm_idx, agent in enumerate(self.agents):
            remaining_capacity = max(0, self._capacity(firm_idx) - int(np.sum(self.employed_by == firm_idx)))
            if remaining_capacity <= 0:
                continue

            worker_costs = costs_per_worker[agent]

            for worker_id in unemployed_workers:
                cost_val = float(worker_costs[worker_id])
                if cost_val > 0.0:
                    sigma_tilde_draw = self.screening.screen_worker(
                        sigma_true=np.array([self.sigma_true[worker_id]], dtype=np.float32),
                        interview_costs=np.array([cost_val], dtype=np.float32),
                        sigma_hat=np.array([self.sigma_hat[worker_id]], dtype=np.float32),
                    )[0]

                    self.sigma_tilde[firm_idx, worker_id] = sigma_tilde_draw
                    self.interview_vars[firm_idx, worker_id] = float(self.screening.interview_var(cost_val))
                    interviewed_mask[firm_idx, worker_id] = True
                    screening_costs[agent] += cost_val

        # 2) STABLE MATCHING: Use Gale-Shapley instead of greedy matching
        remaining_capacity = [
            max(0, self._capacity(i) - int(np.sum(self.employed_by == i)))
            for i in range(self.num_companies)
        ]
        matching_result: WageMatchingResult = stable_matching_from_signals(
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

        # 4) Firing decision
        firing_costs = {agent: 0.0 for agent in self.agents}
        for worker_id, firm_id in enumerate(self.employed_by):
            if firm_id < 0:
                continue
            agent = f"company_{firm_id}"
            profit = float(profits_per_worker[worker_id])
            wage = float(self.wages[worker_id])
            c_fire = self.firing_cost_multiplier * wage

            fire_result = firing_decision(
                p_ijt=profit,
                w_ijt=wage,
                c_fire_t=c_fire,
            )

            if fire_result.fire:
                self.employed_by[worker_id] = -1
                self.tenure[worker_id] = 0.0
                self.wages[worker_id] = 0.0
                firing_costs[agent] += c_fire

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
