from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Sequence

import numpy as np

@dataclass
class FinalOffers:
    firm_id: int
    employeed_by: np.ndarray  # a mask of final offer

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




def worker_wage_accepted(
        firm_offers: Sequence[FirmWageOffers],
        sigma_tilde:np.ndarray
) -> Sequence[FinalOffers]:
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

    # map of company to newly hired worker ids
    company_to_new_workers = {}
    for worker, worker_offers in offered_wage.items():
        # max(..., key=lambda x: x[1])  按 tuple 的第二个值比较
        best_offer_for_worker = max(worker_offers, key=lambda x: x[1])
        wage_matching_result_for_worker[worker] = best_offer_for_worker
        company_new_workers = company_to_new_workers.get(best_offer_for_worker[0], [])
        company_new_workers.append(worker)
        company_to_new_workers[best_offer_for_worker[0]] = company_new_workers

    offer_results = []
    for firm_id, new_employees in company_to_new_workers.items():
        employment_mask = np.zeros(sigma_tilde.shape)
        employment_mask[new_employees] = 1
        offer_results.append(FinalOffers(firm_id, employment_mask))

    # FinalOffers(firm_id) (mask of new employees)
    return offer_results

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

# ------------------------- Inline tests below ------------------------- #

def _test_get_offer_mask():
    # Ten workers; only those interviewed can get offers.
    sigma_tilde = np.array([0.9, 0.1, -0.3, 1.4, 0.2, 0.8, 1.1, -0.7, 0.5, 1.3])
    interviewed_mask = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=np.int8)
    num_offers = 3  # explicitly pick 3 offers for clarity
    offer_mask = get_offer_mask(num_offers, sigma_tilde, interviewed_mask)
    print("== test_get_offer_mask ==")
    print("sigma_tilde      :", sigma_tilde)
    print("interviewed_mask :", interviewed_mask)
    print("offer_mask       :", offer_mask)
    # The top interviewed workers are indices (3, 6, 9) given the signals above.
    expected = np.zeros_like(interviewed_mask)
    expected[[3, 6, 9]] = 1
    assert np.array_equal(offer_mask, expected), f"expected offers at {np.where(expected)[0]} but got {np.where(offer_mask)[0]}"


def _test_firm_offer():
    # Ten workers; firm makes offers to top 30% of interviewed (floor -> 2 offers).
    sigma_tilde = np.array([0.4, 1.2, -0.6, 0.9, 0.0, 1.5, -0.2, 0.7, 1.1, -0.1])
    interviewed_mask = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    offer_rate = 0.3
    firm_multiplier = 1.7
    offers = firm_offer(
        sigma_tilde=sigma_tilde,
        interviewed_mask=interviewed_mask,
        offer_rate=offer_rate,
        firm_multiplier=firm_multiplier,
        firm_id=2,
    )
    expected_mask = get_offer_mask(int(np.sum(interviewed_mask) * offer_rate), sigma_tilde, interviewed_mask)
    non_zero_mask = (offers.wage_array > 0).astype(np.int8)
    print("== test_firm_offer ==")
    print("sigma_tilde      :", sigma_tilde)
    print("interviewed_mask :", interviewed_mask)
    print("expected_mask    :", expected_mask)
    print("wage_array       :", offers.wage_array)
    assert offers.firm_id == 2
    assert np.array_equal(non_zero_mask, expected_mask), "wage mask does not match expected offers"
    np.testing.assert_allclose(offers.wage_array, g(sigma_tilde) * firm_multiplier * expected_mask)


def _test_worker_wage_accepted():
    # Two companies compete over ten workers with overlapping offers.
    sigma_tilde = np.linspace(-0.5, 1.5, 10)
    firm_a = FirmWageOffers(
        firm_id=0,
        wage_array=np.array([0, 1.2, 0, 0, 2.5, 0, 3.5, 0, 0, 4.0]),
    )
    firm_b = FirmWageOffers(
        firm_id=1,
        wage_array=np.array([0.5, 1.0, 0, 2.8, 0, 3.6, 2.0, 0, 4.2, 0]),
    )
    results = worker_wage_accepted([firm_a, firm_b], sigma_tilde)
    results = sorted(results, key=lambda r: r.firm_id)
    print("== test_worker_wage_accepted ==")
    for res in results:
        print(f"firm {res.firm_id} employeed_by mask:", res.employeed_by.astype(int))
    assert len(results) == 2, "Expected results for both firms"
    firm0_mask = results[0].employeed_by if results[0].firm_id == 0 else results[1].employeed_by
    firm1_mask = results[0].employeed_by if results[0].firm_id == 1 else results[1].employeed_by
    expected_firm0_mask = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=int)  # wins at wage ties? firm_b higher wins others
    expected_firm1_mask = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=int)
    assert np.array_equal(firm0_mask.astype(int), expected_firm0_mask), f"firm 0 hires {np.where(firm0_mask)[0]} but expected {np.where(expected_firm0_mask)[0]}"
    assert np.array_equal(firm1_mask.astype(int), expected_firm1_mask), f"firm 1 hires {np.where(firm1_mask)[0]} but expected {np.where(expected_firm1_mask)[0]}"


def _test_private_sigma_tilde_per_firm():
    # Each firm uses its own private sigma_tilde to generate offers, and we print the raw objects.
    sigma_firm_a = np.array([1.2, 1.4, 0.1, 1.6, -0.3, 0.9, 0.7, 1.8, 0.2, 1.0])
    sigma_firm_b = np.array([0.5, 1.5, 1.3, 0.6, 1.9, 0.2, 1.1, 0.0, 1.7, 0.8])
    interviewed_mask = np.ones(10, dtype=np.int8)  # all 10 workers interviewed by both

    offers_a = firm_offer(
        sigma_tilde=sigma_firm_a,
        interviewed_mask=interviewed_mask,
        offer_rate=0.3,
        firm_multiplier=1.0,
        firm_id=0,
    )
    offers_b = firm_offer(
        sigma_tilde=sigma_firm_b,
        interviewed_mask=interviewed_mask,
        offer_rate=0.3,
        firm_multiplier=1.5,
        firm_id=1,
    )

    results = worker_wage_accepted([offers_a, offers_b], sigma_firm_a)
    results = sorted(results, key=lambda r: r.firm_id)

    print("== test_private_sigma_tilde_per_firm ==")
    print("sigma_firm_a:", sigma_firm_a)
    print("sigma_firm_b:", sigma_firm_b)
    print("FirmWageOffers A:", offers_a)
    print("FirmWageOffers B:", offers_b)
    for res in results:
        print("FinalOffers:", res)

    firm0_mask = results[0].employeed_by if results[0].firm_id == 0 else results[1].employeed_by
    firm1_mask = results[0].employeed_by if results[0].firm_id == 1 else results[1].employeed_by

    expected_firm0_mask = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=int)
    expected_firm1_mask = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=int)

    assert np.array_equal(firm0_mask.astype(int), expected_firm0_mask), f"firm 0 hires {np.where(firm0_mask)[0]} but expected {np.where(expected_firm0_mask)[0]}"
    assert np.array_equal(firm1_mask.astype(int), expected_firm1_mask), f"firm 1 hires {np.where(firm1_mask)[0]} but expected {np.where(expected_firm1_mask)[0]}"


def run_tests():
    print("Running inline tests for matching1.py with 10 workers and 2 companies...")
    _test_get_offer_mask()
    _test_firm_offer()
    _test_worker_wage_accepted()
    _test_private_sigma_tilde_per_firm()
    print("All inline tests passed.")


if __name__ == "__main__":
    run_tests()
