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
