from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Sequence

import numpy as np


@dataclass
class WageMatchingResult:
    """Worker-side view of all offers and accepted wages."""

    firm_to_workers: Dict[int, List[int]]
    worker_to_firm: Dict[int, Optional[int]]
    worker_wage: Dict[int, float]


@dataclass
class FirmWageOffers:
    """
    Wage offers extended by a single firm based on its private signals.

    firm_id: identifier of the firm making offers
    offers: worker j -> wage offer (only includes workers that received an offer)
    wage_array: dense array aligned with sigma_tilde (0.0 for no offer)
    """

    firm_id: int
    offers: Dict[int, float]
    wage_array: np.ndarray


def g(x: np.ndarray) -> np.ndarray:
    """Bounded, increasing mapping from signal to wage component."""
    alpha = 0.5
    return 0.5 * (1.0 + np.tanh(alpha * x))

def firm_offer(
        sigma_tilde: np.ndarray,
        interviewed_mask: np.ndarray,
        capacity: int,
        eligible_workers: Optional[Sequence[int]] = None,
        g: Callable[[np.ndarray], np.ndarray] = g,
        firm_multiplier: float = 1.0,
        firm_id: int = 0,
) -> FirmWageOffers:

    """


    Args:
        sigma_tilde: 1d array
        interviewed_mask: 1d array, 1 = interviewed, 0 = not
        capacity: make 30% of offer to people that being interviewed.
        eligible_workers:
        g:
        firm_multiplier: it should be a number from sde.py

    Returns: an array of wage, same size as sigma_tilde,
    about how much amount of wage offered to interviewed worker
    and should be 0 if not offering an offer.

    """
    sigma_tilde = np.asarray(sigma_tilde, dtype=float).reshape(-1)
    interviewed_mask = np.asarray(interviewed_mask, dtype=bool).reshape(-1)
    if sigma_tilde.shape[0] != interviewed_mask.shape[0]:
        raise ValueError("sigma_tilde and interviewed_mask must have the same length")

    num_workers = sigma_tilde.shape[0]
    if eligible_workers is None:
        eligible_workers = list(range(num_workers))
    eligible_workers = np.asarray(eligible_workers, dtype=int)

    cap = int(capacity) if capacity is not None else 0
    interviewed_workers = [
        int(j)
        for j in eligible_workers
        if interviewed_mask[j]
    ]
    if not interviewed_workers or cap <= 0:
        return FirmWageOffers(firm_id=int(firm_id), offers={}, wage_array=np.zeros(num_workers, dtype=float))

    n_offers = min(
        cap,
        int(np.ceil(0.3 * len(interviewed_workers))) # 30% of interviewed workers can choose to offer no one.
    )
    signals = sigma_tilde[interviewed_workers]
    wages = g(signals) * float(firm_multiplier)
    top_indices = np.argsort(wages)[::-1][:n_offers]

    wage_array = np.zeros(num_workers, dtype=float)
    offers: Dict[int, float] = {}
    for idx in top_indices:
        worker_id = interviewed_workers[idx]
        wage_offer = float(wages[idx])
        wage_array[worker_id] = wage_offer
        offers[worker_id] = wage_offer

    return FirmWageOffers(
        firm_id=int(firm_id),
        offers=offers,
        wage_array=wage_array,
    )




def worker_wage(
        firm_offers: Sequence[FirmWageOffers],
        num_workers: int,
):
    """
    Given offers from multiple firms, let each worker accept the best wage.

    """
    firm_to_workers: Dict[int, List[int]] = {}
    worker_to_firm: Dict[int, Optional[int]] = {j: None for j in range(num_workers)}
    worker_wage: Dict[int, float] = {}

    offers_by_worker: Dict[int, List[tuple[int, float]]] = {j: [] for j in range(num_workers)}
    for firm_offer_obj in firm_offers:
        firm_id = firm_offer_obj.firm_id
        firm_to_workers.setdefault(firm_id, [])
        for worker_id, wage in firm_offer_obj.offers.items():
            offers_by_worker[worker_id].append((firm_id, float(wage)))

    for worker_id, offers in offers_by_worker.items():
        if not offers:
            continue
        best_firm, best_wage = max(offers, key=lambda pair: (pair[1], -pair[0]))
        worker_to_firm[worker_id] = best_firm
        worker_wage[worker_id] = float(best_wage)
        firm_to_workers.setdefault(best_firm, []).append(worker_id)

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
    g: Callable[[np.ndarray], np.ndarray] = g,
    firm_multipliers: Optional[Sequence[float]] = None,
) -> WageMatchingResult:
    """
    Convenience wrapper to produce firm offers then select best wages per worker.
    """
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    interviewed_mask = np.asarray(interviewed_mask, dtype=bool)
    if sigma_tilde.shape != interviewed_mask.shape:
        raise ValueError("sigma_tilde and interviewed_mask must have the same shape")

    num_firms, num_workers = sigma_tilde.shape
    if firm_multipliers is None:
        firm_multipliers = [1.0] * num_firms
    if len(firm_multipliers) != num_firms:
        raise ValueError("firm_multipliers length must equal number of firms")

    offers = []
    for firm_idx in range(num_firms):
        offers.append(
            firm_offer(
                sigma_tilde=sigma_tilde[firm_idx],
                interviewed_mask=interviewed_mask[firm_idx],
                capacity=int(capacities[firm_idx]) if capacities is not None else 0,
                eligible_workers=eligible_workers,
                g=g,
                firm_multiplier=firm_multipliers[firm_idx],
                firm_id=firm_idx,
            )
        )

    return worker_wage(
        firm_offers=offers,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Quick sanity check: 2 firms, 10 workers.
    sigma_tilde_demo = np.array(
        [
            np.linspace(-1.0, 1.0, 10),            # firm 0 signals
            np.linspace(0.5, -0.5, 10) + 0.2,      # firm 1 signals
        ],
        dtype=float,
    )
    interviewed_mask_demo = np.array(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],        # firm 0 interviewed first 5
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],        # firm 1 interviewed middle 5
        ],
        dtype=bool,
    )
    capacities_demo = [3, 2]
    firm_multipliers_demo = [1.0, 1.2]

    print("sigma_tilde:\n", sigma_tilde_demo)
    print("interviewed_mask:\n", interviewed_mask_demo.astype(int))

    result = greedy_wage_matching_from_signals(
        sigma_tilde=sigma_tilde_demo,
        interviewed_mask=interviewed_mask_demo,
        capacities=capacities_demo,
        firm_multipliers=firm_multipliers_demo,
    )

    # Collect dense wage offers per firm for display
    firm_offers_display = []
    for firm_idx in range(sigma_tilde_demo.shape[0]):
        offers = firm_offer(
            sigma_tilde=sigma_tilde_demo[firm_idx],
            interviewed_mask=interviewed_mask_demo[firm_idx],
            capacity=capacities_demo[firm_idx],
            eligible_workers=range(sigma_tilde_demo.shape[1]),
            firm_multiplier=firm_multipliers_demo[firm_idx],
        ).wage_array
        firm_offers_display.append(offers)
    wage_offers_demo = np.vstack(firm_offers_display)

    print("capacity looks like:\n", np.round(capacities_demo[firm_idx], 3))
    print("wage_offers (0 if no offer):\n", np.round(wage_offers_demo, 3))
    print("worker accepted firm:\n", result.worker_to_firm)
    print("worker accepted wage:\n", {k: round(v, 3) for k, v in result.worker_wage.items()})
