"""
1. Greedy Matching: Firms make offers, workers accept highest wage (MARL approach)
2. Stable Matching: Deferred acceptance algorithm (Gale-Shapley)

A matching is STABLE if:
1. Individual Rationality: No firm or worker prefers unemployment to their match
2. No Blocking Pairs: No (firm, worker) pair would both prefer to match with each other
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class MatchingOutcome:
    """
    Result of a matching mechanism.

    Attributes:
        matches: Dict mapping company_id -> set of worker_ids
        wages: Dict mapping worker_id -> wage
        unmatched_workers: Set of worker IDs not matched
        unmatched_companies: Set of company IDs not matched
        is_stable: Whether matching is stable
        time_to_match: Time periods needed to find matching
    """
    matches: Dict[int, Set[int]]
    wages: Dict[int, float]
    unmatched_workers: Set[int]
    unmatched_companies: Set[int]
    is_stable: bool
    time_to_match: int


class StableMatching:
    """
    Implements stable matching algorithms for labor markets.

    Based on Gale-Shapley deferred acceptance algorithm, extended to handle:
    - Many-to-one matching (firms can hire multiple workers)
    - Wage determination
    - Beliefs about worker ability
    """

    def __init__(self, num_companies: int, num_workers: int):
        """
        Initialize stable matching mechanism.

        Args:
            num_companies: Number of firms
            num_workers: Number of workers
        """
        self.num_companies = num_companies
        self.num_workers = num_workers

    def compute_firm_preferences(
        self,
        company_id: int,
        worker_beliefs: np.ndarray,  # Expected ability for each worker
        worker_experience: np.ndarray,
        unemployed_workers: List[int]
    ) -> List[int]:
        """
        Compute firm's preference ranking over workers.

        Firms prefer workers with higher expected profit:
            E[profit] = E[σ_j | beliefs] + β*log(1 + exp_j)

        Args:
            company_id: Company index
            worker_beliefs: Firm's belief about each worker's ability (num_workers,)
            worker_experience: Experience levels (num_workers,)
            unemployed_workers: List of available worker IDs

        Returns:
            List of worker IDs sorted by preference (most preferred first)
        """
        beta = 0.5  # Experience return parameter

        # Compute expected profit for each unemployed worker
        expected_profits = []
        for worker_id in unemployed_workers:
            exp_profit = worker_beliefs[worker_id] + beta * np.log1p(worker_experience[worker_id])
            expected_profits.append((exp_profit, worker_id))

        # Sort by expected profit (descending)
        expected_profits.sort(reverse=True, key=lambda x: x[0])

        return [worker_id for _, worker_id in expected_profits]

    def compute_worker_preferences(
        self,
        worker_id: int,
        wage_offers: Dict[int, float]  # company_id -> wage
    ) -> List[int]:
        """
        Compute worker's preference ranking over firms.

        Workers prefer higher wages (perfectly greedy).

        Args:
            worker_id: Worker index
            wage_offers: Map of company_id -> wage offer

        Returns:
            List of company IDs sorted by preference (most preferred first)
        """
        # Sort companies by wage (descending)
        sorted_companies = sorted(wage_offers.items(), key=lambda x: x[1], reverse=True)
        return [company_id for company_id, _ in sorted_companies]

    def deferred_acceptance(
        self,
        firm_preferences: Dict[int, List[int]],  # company_id -> ranked list of workers
        worker_preferences: Dict[int, List[int]],  # worker_id -> ranked list of companies
        firm_capacities: Dict[int, int],  # company_id -> max workers
        wage_offers: Dict[Tuple[int, int], float]  # (company_id, worker_id) -> wage
    ) -> MatchingOutcome:
        """
        Firm-proposing deferred acceptance algorithm.

        Algorithm:
        1. Each firm proposes to its most preferred available worker
        2. Each worker tentatively accepts best offer, rejects others
        3. Rejected firms propose to next worker on their list
        4. Repeat until no firm wants to make new proposals

        Args:
            firm_preferences: Each firm's ranked list of workers
            worker_preferences: Each worker's ranked list of firms
            firm_capacities: Maximum workers per firm
            wage_offers: Wage that each firm offers to each worker

        Returns:
            MatchingOutcome with stable matching
        """
        # Track current tentative matches
        worker_to_firm: Dict[int, int] = {}  # worker -> current tentative firm
        firm_to_workers: Dict[int, Set[int]] = {c: set() for c in range(self.num_companies)}

        # Track which workers each firm has already proposed to
        proposed_to: Dict[int, Set[int]] = {c: set() for c in range(self.num_companies)}

        # Firms that still want to make proposals
        active_firms = set(range(self.num_companies))

        iterations = 0
        max_iterations = self.num_companies * self.num_workers  # Prevent infinite loops

        while active_firms and iterations < max_iterations:
            iterations += 1
            firms_to_deactivate = set()

            for company_id in list(active_firms):
                # Check if firm has capacity
                if len(firm_to_workers[company_id]) >= firm_capacities[company_id]:
                    firms_to_deactivate.add(company_id)
                    continue

                # Get firm's preference list
                pref_list = firm_preferences.get(company_id, [])

                # Find next worker to propose to (not yet proposed)
                next_worker = None
                for worker_id in pref_list:
                    if worker_id not in proposed_to[company_id]:
                        next_worker = worker_id
                        break

                if next_worker is None:
                    # Firm has proposed to all workers on its list
                    firms_to_deactivate.add(company_id)
                    continue

                # Make proposal
                proposed_to[company_id].add(next_worker)
                worker_id = next_worker

                # Worker evaluates proposal
                current_firm = worker_to_firm.get(worker_id, None)

                if current_firm is None:
                    # Worker is unmatched, accept proposal
                    worker_to_firm[worker_id] = company_id
                    firm_to_workers[company_id].add(worker_id)
                else:
                    # Worker compares current match with new proposal
                    current_wage = wage_offers.get((current_firm, worker_id), 0.0)
                    new_wage = wage_offers.get((company_id, worker_id), 0.0)

                    if new_wage > current_wage:
                        # Accept new offer, reject current
                        firm_to_workers[current_firm].remove(worker_id)
                        worker_to_firm[worker_id] = company_id
                        firm_to_workers[company_id].add(worker_id)

                        # Current firm becomes active again (lost a worker)
                        active_firms.add(current_firm)
                    # else: reject new offer, keep current

            # Deactivate firms that can't make more proposals
            active_firms -= firms_to_deactivate

        # Extract final matching
        final_wages = {
            worker_id: wage_offers.get((company_id, worker_id), 0.0)
            for worker_id, company_id in worker_to_firm.items()
        }

        matched_workers = set(worker_to_firm.keys())
        unmatched_workers = set(range(self.num_workers)) - matched_workers

        matched_companies = set(c for c in range(self.num_companies) if firm_to_workers[c])
        unmatched_companies = set(range(self.num_companies)) - matched_companies

        # Check stability
        is_stable = self._check_stability(
            firm_to_workers, worker_to_firm, firm_preferences,
            worker_preferences, wage_offers
        )

        return MatchingOutcome(
            matches=firm_to_workers,
            wages=final_wages,
            unmatched_workers=unmatched_workers,
            unmatched_companies=unmatched_companies,
            is_stable=is_stable,
            time_to_match=iterations
        )

    def _check_stability(
        self,
        firm_to_workers: Dict[int, Set[int]],
        worker_to_firm: Dict[int, int],
        firm_preferences: Dict[int, List[int]],
        worker_preferences: Dict[int, List[int]],
        wage_offers: Dict[Tuple[int, int], float]
    ) -> bool:
        """
        Check if matching is stable.

        A matching is stable if there are no blocking pairs:
        - Firm i and worker j form a blocking pair if:
          1. They are not currently matched
          2. Firm i prefers worker j to some current employee (or has capacity)
          3. Worker j prefers firm i to current employer (or is unemployed)

        Args:
            firm_to_workers: Current matching (firm -> workers)
            worker_to_firm: Current matching (worker -> firm)
            firm_preferences: Firm preference rankings
            worker_preferences: Worker preference rankings
            wage_offers: Wage structure

        Returns:
            True if matching is stable
        """
        # Check each potential (firm, worker) pair
        for company_id in range(self.num_companies):
            for worker_id in range(self.num_workers):
                # Skip if already matched
                if worker_id in firm_to_workers[company_id]:
                    continue

                # Check if firm prefers this worker to some current employee
                current_workers = firm_to_workers[company_id]
                if not current_workers:
                    firm_prefers = True  # Firm has capacity
                else:
                    firm_pref_list = firm_preferences.get(company_id, [])
                    if worker_id not in firm_pref_list:
                        continue

                    worker_rank = firm_pref_list.index(worker_id)
                    worst_current_rank = max(
                        firm_pref_list.index(w) for w in current_workers if w in firm_pref_list
                    )
                    firm_prefers = worker_rank < worst_current_rank

                if not firm_prefers:
                    continue

                # Check if worker prefers this firm to current employer
                current_firm = worker_to_firm.get(worker_id, None)
                new_wage = wage_offers.get((company_id, worker_id), 0.0)

                if current_firm is None:
                    worker_prefers = new_wage > 0  # Unemployed worker
                else:
                    current_wage = wage_offers.get((current_firm, worker_id), 0.0)
                    worker_prefers = new_wage > current_wage

                if worker_prefers:
                    # Found blocking pair!
                    return False

        return True  # No blocking pairs found

    def compute_competitive_equilibrium_wages(
        self,
        firm_to_workers: Dict[int, Set[int]],
        firm_preferences: Dict[int, List[int]],
        worker_abilities: np.ndarray,  # For computing match values
        worker_experience: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute competitive equilibrium wages given a matching.

        In equilibrium, wages are set such that:
        1. Firms are willing to pay (profit ≥ wage)
        2. Workers are willing to accept (wage ≥ reservation wage)
        3. No blocking pairs (stability)

        Simplified approach: wage = α * expected_profit
        where α ∈ [0, 1] is worker's bargaining power

        Args:
            firm_to_workers: Current matching
            firm_preferences: Firm preferences (to determine outside options)
            worker_abilities: Belief about abilities
            worker_experience: Experience levels

        Returns:
            Dictionary of worker_id -> equilibrium wage
        """
        beta = 0.5
        alpha = 0.6  # Workers capture 60% of surplus (Nash bargaining default)

        wages = {}

        for company_id, workers in firm_to_workers.items():
            for worker_id in workers:
                # Compute match value
                expected_profit = (
                    worker_abilities[worker_id] +
                    beta * np.log1p(worker_experience[worker_id])
                )

                # Worker captures fraction α of the surplus
                # (Remaining goes to firm)
                wage = alpha * expected_profit
                wages[worker_id] = max(0.0, wage)

        return wages


def compare_matching_mechanisms(
    greedy_outcome: MatchingOutcome,
    stable_outcome: MatchingOutcome
) -> Dict[str, float]:
    """
    Compare greedy vs stable matching outcomes.

    Metrics:
    - Time to match
    - Number of matches
    - Average wage
    - Stability

    Args:
        greedy_outcome: Result from greedy mechanism
        stable_outcome: Result from stable matching

    Returns:
        Dictionary of comparison metrics
    """
    greedy_matches = sum(len(workers) for workers in greedy_outcome.matches.values())
    stable_matches = sum(len(workers) for workers in stable_outcome.matches.values())

    greedy_avg_wage = np.mean(list(greedy_outcome.wages.values())) if greedy_outcome.wages else 0.0
    stable_avg_wage = np.mean(list(stable_outcome.wages.values())) if stable_outcome.wages else 0.0

    return {
        'greedy_time_to_match': greedy_outcome.time_to_match,
        'stable_time_to_match': stable_outcome.time_to_match,
        'greedy_num_matches': greedy_matches,
        'stable_num_matches': stable_matches,
        'greedy_avg_wage': greedy_avg_wage,
        'stable_avg_wage': stable_avg_wage,
        'greedy_is_stable': greedy_outcome.is_stable,
        'stable_is_stable': stable_outcome.is_stable,
        'time_difference': stable_outcome.time_to_match - greedy_outcome.time_to_match,
        'match_efficiency_ratio': stable_matches / max(greedy_matches, 1),
    }
