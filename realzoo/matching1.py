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
    wage_array: dense array aligned with sigma_tilde (0.0 for no offer)
    """

    firm_id: int
    wage_array: np.ndarray


def g(x: np.ndarray) -> np.ndarray:
    """Bounded, increasing mapping from signal to wage component."""
    alpha = 0.5
    return 0.5 * (1.0 + np.tanh(alpha * x))

def firm_offer(
        sigma_tilde: np.ndarray,
        interviewed_mask: np.ndarray,
        offer_rate: float = 0.3,
        g: Callable[[np.ndarray], np.ndarray] = g,
        firm_multiplier: float = 1.0,
        firm_id: int = 0,
        # TODO: pass in a policy to determine who to give an offer
) -> FirmWageOffers:

    """


    Args:
        sigma_tilde: 1d array
        interviewed_mask: 1d array, 1 = interviewed, 0 = not
        capacity: make offer to 30% of people that are being interviewed.
        g:
        firm_multiplier: it should be a number from sde.py

    Returns: an array of wage, same size as sigma_tilde,
    about how much amount of wage offered to interviewed worker
    and should be 0 if not offering an offer.

    """
    if sigma_tilde.shape[0] != interviewed_mask.shape[0]:
        raise ValueError("sigma_tilde and interviewed_mask must have the same length")

    num_offers = int(np.sum(interviewed_mask) * offer_rate) # floors

    signals = get_offer_mask(num_offers, sigma_tilde, interviewed_mask)
    wages = (g(sigma_tilde) * float(firm_multiplier)) * signals

    return FirmWageOffers(
        firm_id=int(firm_id),
        wage_array=wages,
    )

def get_offer_mask(num_offers: int, sigma_tilde: np.ndarray, interviewed_mask: np.ndarray):
    """
    Return nd.array of (0,1) for if we will give that employee an offers.
    There are only as many 1's as num_offers.
    The offered individuals are a subset of the interview_mask individuals

    Example input:
    1
    [0.2, 0,3, 0.4]
    [1, 0, 1]

    Output:
    [0, 0, 1]

    Args:
        num_offers:
        sigma_tilde:
        interviewed_mask:

    Returns: sigma_tilde shape array

    """
    masked_vals = np.where(interviewed_mask.astype(bool), sigma_tilde, -np.inf)
    top_idx = np.argpartition(masked_vals, -num_offers)[-num_offers:]
    top_idx = top_idx[np.argsort(masked_vals[top_idx])[::-1]]  # keep highest first
    out = np.zeros_like(interviewed_mask, dtype=np.int8)
    out[top_idx] = 1
    return out




def worker_wage(
        firm_offers: Sequence[FirmWageOffers],
        sigma_tilde:np.ndarray
):
    """
    Given offers from multiple firms, let each worker accept the best wage.
    returns a dictionary, worker_id, firm_id they chose, the wage they accept

    """

    offered_wage = {}
    for firm_offer_obj in firm_offers:
        firm_id = firm_offer_obj.firm_id
        wages = firm_offer_obj.wage_array
        for worker, wage in enumerate(wages):
            worker_offer_list = offered_wage.get(worker, [])
            if wage > 0:
                worker_offer_list.append((firm_id, wage))
                offered_wage[worker] = worker_offer_list

    # {0: [(1,3), (2,4)]
    # if a worker has no offers, wont be included
    wage_matching_result_for_worker = {}
    for key, tuples in offered_wage.items():
        # max(..., key=lambda x: x[1])  按 tuple 的第二个值比较
        wage_matching_result_for_worker[key] = max(tuples, key=lambda x: x[1])


    return wage_matching_result_for_worker

# if __name__ == "__main__":
#     print(get_offer_mask(1, np.array([0.2, 0.3, 0.4]), np.array([1, 0, 1])))
#     """
#     Example input:
#     1
#     [0.2, 0.3, 0.4]
#     [1, 0, 1]
#
#     Output:
#     [0, 0, 1]
#     """