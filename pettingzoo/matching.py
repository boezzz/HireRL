from dataclasses import dataclass
from typing import Dict, Optional, List, Callable
import numpy as np


@dataclass
class WageMatchingResult:
    """
    一步静态匹配结果（面试后、尚未产生 profit）：
      - 每个 firm 最多雇一个 worker
      - 工资只取决于面试信号 w_{ij} = (1 - v_x) g(tilde_sigma_{ij})
    """
    firm_to_worker: Dict[int, Optional[int]]   # firm i -> worker j or None
    worker_to_firm: Dict[int, Optional[int]]   # worker j -> firm i or None
    worker_wage: Dict[int, float]              # worker j -> accepted wage


def greedy_wage_matching_from_signals(
    tilde_sigma: np.ndarray,
    v_x: float,
    g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eligible_workers: Optional[List[int]] = None,
) -> WageMatchingResult:
    """
    3. Firms give the offer to the highest ability worker based on the
       updated tilde_sigma_{ij,t} and offer a wage:

           w_{j,t} = (1 - v_x) g(tilde_sigma_{ij,t=interview})

       Workers are perfectly greedy and accept the highest wage.

    参数
    ----
    tilde_sigma : np.ndarray
        形状为 (num_firms, num_workers)，
        tilde_sigma[i, j] 是 firm i 对 worker j 的面试信号。
    v_x : float
        权重 v_x ∈ [0, 1)，此处还未用到 profit，因此只出现 (1 - v_x)。
    g : callable, optional
        对面试信号做变换的函数 g(x)。若为 None，则 g(x) = x。
    eligible_workers : list[int], optional
        可以被雇佣的 worker 下标集合；若为 None，则所有 worker 均可。

    返回
    ----
    WageMatchingResult
    """
    tilde_sigma = np.asarray(tilde_sigma, dtype=float)
    if tilde_sigma.ndim != 2:
        raise ValueError("tilde_sigma must be a 2D array (num_firms, num_workers).")

    num_firms, num_workers = tilde_sigma.shape

    if eligible_workers is None:
        eligible_workers = list(range(num_workers))
    else:
        eligible_workers = list(eligible_workers)

    # 默认 g(x) 为有天花板、递减边际回报的单调函数：
    # g(x) = 0.5 * (1 + tanh(alpha * x)) ∈ (0, 1)，能力越高，工资信号越高，但逐渐趋于上界
    if g is None:
        def g(x: np.ndarray) -> np.ndarray:
            alpha = 0.5
            return 0.5 * (1.0 + np.tanh(alpha * x))

    # 计算工资矩阵 w_{ij} = (1 - v_x) * g(tilde_sigma_{ij})
    g_tilde = g(tilde_sigma)
    wages_ij = (1.0 - float(v_x)) * g_tilde

    firm_to_worker: Dict[int, Optional[int]] = {i: None for i in range(num_firms)}
    worker_to_firm: Dict[int, Optional[int]] = {j: None for j in range(num_workers)}
    worker_wage: Dict[int, float] = {}

    # 每个 firm：挑一个工资最高的 worker 发 offer
    offers_by_worker: Dict[int, List[tuple[int, float]]] = {j: [] for j in eligible_workers}

    for i in range(num_firms):
        best_j = None
        best_wage = -np.inf
        for j in eligible_workers:
            w_ij = wages_ij[i, j]
            if w_ij > best_wage:
                best_wage = w_ij
                best_j = j

        if best_j is not None and np.isfinite(best_wage):
            offers_by_worker[best_j].append((i, float(best_wage)))

    # Workers 完全贪心：接受最高工资
    for j in eligible_workers:
        if not offers_by_worker[j]:
            continue

        # 按工资排序；工资相同则 firm id 小的胜出
        best_i, best_wage = max(
            offers_by_worker[j],
            key=lambda pair: (pair[1], -pair[0])
        )

        firm_to_worker[best_i] = j
        worker_to_firm[j] = best_i
        worker_wage[j] = best_wage

    return WageMatchingResult(
        firm_to_worker=firm_to_worker,
        worker_to_firm=worker_to_firm,
        worker_wage=worker_wage,
    )