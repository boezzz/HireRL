from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Sequence

import numpy as np


@dataclass
class WageMatchingResult:
    """
    One-period matching result after interviews.

    firm_to_workers: firm i -> hired worker ids
    worker_to_firm: worker j -> accepted firm id (or None)
    worker_wage: worker j -> accepted wage
    """

    firm_to_workers: Dict[int, List[int]]
    worker_to_firm: Dict[int, Optional[int]]
    worker_wage: Dict[int, float]


def g(x: np.ndarray) -> np.ndarray:
    """Bounded, increasing mapping from signal to wage component."""
    alpha = 0.5
    return 0.5 * (1.0 + np.tanh(alpha * x))


def stable_matching_from_signals(
    sigma_tilde: np.ndarray,
    interviewed_mask: np.ndarray,
    capacities: Sequence[int],
    eligible_workers: Optional[Sequence[int]] = None,
    v_x: float = 0.0,
    g: Callable[[np.ndarray], np.ndarray] = g,
    firm_multipliers: Optional[Sequence[float]] = None,
) -> WageMatchingResult:
    """
    Gale-Shapley deferred acceptance algorithm for stable matching.

    Firms propose to workers in order of preference (based on sigma_tilde).
    Workers tentatively accept best offer and can trade up.

    Args:
        sigma_tilde: Firm beliefs about worker abilities (num_firms, num_workers)
        interviewed_mask: Which workers each firm interviewed (num_firms, num_workers)
        capacities: Remaining capacity for each firm
        eligible_workers: Workers available for hiring
        v_x: Experience weighting parameter
        g: Wage function mapping signals to wages
        firm_multipliers: Wage multipliers per firm

    Returns:
        WageMatchingResult with stable matching
    """
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    interviewed_mask = np.asarray(interviewed_mask, dtype=bool)
    num_firms, num_workers = sigma_tilde.shape

    if firm_multipliers is None:
        firm_multipliers = [1.0] * num_firms
    if len(firm_multipliers) != num_firms:
        raise ValueError("firm_multipliers length must equal number of firms")
    firm_multipliers = np.asarray(firm_multipliers, dtype=float)

    if eligible_workers is None:
        eligible_workers = list(range(num_workers))
    eligible_workers = set(int(j) for j in eligible_workers)

    # Initialize firm preferences (sorted by signal strength)
    firm_preferences: Dict[int, List[int]] = {}
    for firm_idx in range(num_firms):
        interviewed_workers = [
            int(j) for j in eligible_workers
            if interviewed_mask[firm_idx, j]
        ]
        if interviewed_workers:
            signals = sigma_tilde[firm_idx, interviewed_workers]
            sorted_indices = np.argsort(signals)[::-1]
            firm_preferences[firm_idx] = [interviewed_workers[i] for i in sorted_indices]
        else:
            firm_preferences[firm_idx] = []

    # Compute wages for all interviewed pairs
    wage_matrix = np.zeros((num_firms, num_workers), dtype=float)
    for firm_idx in range(num_firms):
        for worker_id in firm_preferences[firm_idx]:
            signal = sigma_tilde[firm_idx, worker_id]
            wage_matrix[firm_idx, worker_id] = float(g(np.array([signal]))[0] * firm_multipliers[firm_idx])

    # Track current matches
    worker_to_firm: Dict[int, Optional[int]] = {j: None for j in range(num_workers)}
    worker_wage: Dict[int, float] = {j: 0.0 for j in range(num_workers)}
    firm_next_proposal: Dict[int, int] = {i: 0 for i in range(num_firms)}
    firm_to_workers: Dict[int, List[int]] = {i: [] for i in range(num_firms)}

    # Firms propose in rounds
    max_rounds = num_firms * num_workers
    for _ in range(max_rounds):
        proposals_made = False

        for firm_idx in range(num_firms):
            # Check if firm has capacity
            current_size = len(firm_to_workers[firm_idx])
            cap = int(capacities[firm_idx]) if capacities is not None else 0

            if current_size >= cap:
                continue

            # Get next worker to propose to
            pref_list = firm_preferences[firm_idx]
            if firm_next_proposal[firm_idx] >= len(pref_list):
                continue

            worker_id = pref_list[firm_next_proposal[firm_idx]]
            firm_next_proposal[firm_idx] += 1
            proposals_made = True

            # Calculate wage offer
            wage_offer = wage_matrix[firm_idx, worker_id]

            # Worker decides
            current_firm = worker_to_firm[worker_id]
            if current_firm is None:
                # Worker is unmatched, accept
                worker_to_firm[worker_id] = firm_idx
                worker_wage[worker_id] = wage_offer
                firm_to_workers[firm_idx].append(worker_id)
            elif wage_offer > worker_wage[worker_id]:
                # Better offer, switch
                firm_to_workers[current_firm].remove(worker_id)
                worker_to_firm[worker_id] = firm_idx
                worker_wage[worker_id] = wage_offer
                firm_to_workers[firm_idx].append(worker_id)

        if not proposals_made:
            break

    return WageMatchingResult(
        firm_to_workers=firm_to_workers,
        worker_to_firm=worker_to_firm,
        worker_wage=worker_wage,
    )


def greedy_wage_matching_from_signals(
    sigma_tilde: np.ndarray,
    interviewed_mask: np.ndarray,
    capacities: Sequence[int],
    eligible_workers: Optional[Sequence[int]] = None,
    v_x: float = 0.0,
    g: Callable[[np.ndarray], np.ndarray] = g,
    firm_multipliers: Optional[Sequence[float]] = None,
) -> WageMatchingResult:
    """
    Step 2 (matching): firms make offers to 20% of the workers they interviewed.

    - Offers are limited to interviewed workers only.
    - Each firm i can extend up to ceil(0.2 * n_interviewed_i) offers,
      further capped by its remaining capacity.
    - Wage formula: w_{ij} = g(tilde_sigma_{ij}) scaled by an optional firm-specific multiplier.
    - Workers are greedy and accept the single highest wage offer.
    """
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    interviewed_mask = np.asarray(interviewed_mask, dtype=bool)
    num_firms, num_workers = sigma_tilde.shape

    if firm_multipliers is None:
        firm_multipliers = [1.0] * num_firms
    if len(firm_multipliers) != num_firms:
        raise ValueError("firm_multipliers length must equal number of firms")
    firm_multipliers = np.asarray(firm_multipliers, dtype=float)

    if eligible_workers is None:
        eligible_workers = list(range(num_workers))
    eligible_workers = np.asarray(eligible_workers, dtype=int)

    firm_to_workers: Dict[int, List[int]] = {i: [] for i in range(num_firms)}
    worker_to_firm: Dict[int, Optional[int]] = {j: None for j in range(num_workers)}
    worker_wage: Dict[int, float] = {}
    offers_by_worker: Dict[int, List[tuple[int, float]]] = {int(j): [] for j in eligible_workers}

    for firm_idx in range(num_firms):
        cap = int(capacities[firm_idx]) if capacities is not None else 0
        interviewed_workers = [
            int(j)
            for j in eligible_workers
            if interviewed_mask[firm_idx, j]
        ]
        if not interviewed_workers or cap <= 0:
            continue

        n_offers = min(
            cap,
            max(1, int(np.ceil(0.2 * len(interviewed_workers))))
        )
        signals = sigma_tilde[firm_idx, interviewed_workers]
        wages = g(signals) * firm_multipliers[firm_idx]
        top_indices = np.argsort(wages)[::-1][:n_offers]

        for idx in top_indices:
            worker_id = interviewed_workers[idx]
            wage_offer = float(wages[idx])
            offers_by_worker[worker_id].append((firm_idx, wage_offer))

    # Workers accept the single best wage (ties -> lower firm index)
    for worker_id, offers in offers_by_worker.items():
        if not offers:
            continue
        best_i, best_wage = max(offers, key=lambda pair: (pair[1], -pair[0]))
        worker_to_firm[worker_id] = best_i
        worker_wage[worker_id] = float(best_wage)
        firm_to_workers[best_i].append(worker_id)

    return WageMatchingResult(
        firm_to_workers=firm_to_workers,
        worker_to_firm=worker_to_firm,
        worker_wage=worker_wage,
    )
