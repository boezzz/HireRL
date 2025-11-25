"""
目标：
- 用 Excel 里的 employee_count 抽象出现实中的公司规模分布
- 按这个分布初始化 MARL 环境里的公司规模（只用“员工数量”作为规模指标）

依赖：
- pandas
- numpy
"""

import pandas as pd
import numpy as np
import sys


def _coerce_numeric(series: pd.Series, col_name: str) -> pd.Series:
    """
    Lightly clean a numeric column: strip whitespace (incl. NBSP) and commas,
    coerce to numeric, and report rows that become NaN.
    """
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned)
    bad = numeric.isna() & series.notna()
    if bad.any():
        print(f"[clean] dropped {bad.sum()} non-numeric values in '{col_name}'")
    return numeric
# --------------------------------------------------------
# 数据清洗 & 经验方差目标提取
# --------------------------------------------------------

def load_wage_exp(excel_path: str = None,
                  exp_col: str = "year of experience",
                  wage_col: str = "salary") -> pd.DataFrame:
    """
    Load wage/experience data (light numeric coercion to handle stray strings).
    """
    path = excel_path or EXCEL_PATH
    df = pd.read_excel(path, usecols=[exp_col, wage_col])
    df[exp_col] = _coerce_numeric(df[exp_col], exp_col)
    df[wage_col] = _coerce_numeric(df[wage_col], wage_col)
    return df


def wage_variance_by_bin(df: pd.DataFrame,
                         exp_col: str = "year of experience",
                         wage_col: str = "salary",
                         max_bin: int | None = None) -> pd.DataFrame:
    """
    Bin wages by floor(experience) and return count/mean/variance per bin.
    """
    work = df.copy()
    work[exp_col] = _coerce_numeric(work[exp_col], exp_col)
    work[wage_col] = _coerce_numeric(work[wage_col], wage_col)
    work["exp_bin"] = work[exp_col].apply(lambda x: int(np.floor(x)))
    if max_bin is not None:
        work = work[work["exp_bin"] <= max_bin]
    stats = work.groupby("exp_bin")[wage_col].agg(["count", "mean", "var"]).reset_index()
    return stats


def wage_variance_ratio(df: pd.DataFrame,
                        exp_col: str = "year of experience",
                        wage_col: str = "salary",
                        bin_low: tuple = (0, 1),
                        bin_high: tuple = (3, 4),
                        max_bin: int | None = None) -> float | None:
    """
    Compute Var(wage | exp in bin_high) / Var(wage | exp in bin_low).

    Returns None if bins are missing or have zero variance.
    """
    stats = wage_variance_by_bin(df, exp_col=exp_col, wage_col=wage_col, max_bin=max_bin)
    bins = stats.set_index("exp_bin")
    if bin_low[0] not in bins.index or bin_high[0] not in bins.index:
        return None
    v_low = bins.loc[bin_low[0], "var"]
    v_high = bins.loc[bin_high[0], "var"]
    if pd.isna(v_low) or v_low <= 0:
        return None
    return float(v_high / v_low)


def wage_mean_by_bin(df: pd.DataFrame,
                     exp_col: str = "year of experience",
                     wage_col: str = "salary",
                     max_bin: int | None = None) -> pd.DataFrame:
    """
    Bin wages by floor(experience) and return count/mean per bin.
    This is similar to wage_variance_by_bin but keeps only count/mean.
    """
    work = df.copy()
    work["exp_bin"] = work[exp_col].apply(lambda x: int(np.floor(x)))
    if max_bin is not None:
        work = work[work["exp_bin"] <= max_bin]
    stats = work.groupby("exp_bin")[wage_col].agg(["count", "mean"]).reset_index()
    return stats


def load_wage_exp_with_size(excel_path: str = None,
                            exp_col: str = "year of experience",
                            wage_col: str = "salary",
                            emp_col: str = "employee_count") -> pd.DataFrame:
    """
    Load wage/experience data together with firm size, and attach firm_type
    (small / medium / large) using the globally computed q50 / q90 thresholds.

    注意：这依赖于模块顶部已经读取的 EXCEL_PATH、EMP_COL、q50、q90。
    """
    path = excel_path or EXCEL_PATH
    usecols = [exp_col, wage_col, emp_col]
    df = pd.read_excel(path, usecols=usecols)
    df[exp_col] = _coerce_numeric(df[exp_col], exp_col)
    df[wage_col] = _coerce_numeric(df[wage_col], wage_col)
    df[emp_col] = _coerce_numeric(df[emp_col], emp_col)

    # 使用与公司规模初始化相同的 q50 / q90 阈值划分 firm_type
    def _firm_type_from_emp(emp: float) -> str:
        if emp <= q50:
            return "small"
        elif emp < q90:
            return "medium"
        else:
            return "large"

    df["firm_type"] = df[emp_col].apply(_firm_type_from_emp)
    return df


def estimate_size_wage_premia(excel_path: str = None,
                              exp_col: str = "year of experience",
                              wage_col: str = "salary",
                              emp_col: str = "employee_count",
                              max_bin: int | None = None) -> dict:
    """
    Estimate simple firm-size-specific wage premia phi_type, such that

        phi_type * pooled_mean_wage(exp_bin)  ≈  type_mean_wage(exp_bin)

    in a least-squares / weighted-average sense over experience bins.

    Returns:
        {"small": phi_small, "medium": phi_medium, "large": phi_large}
    """
    df = load_wage_exp_with_size(excel_path=excel_path,
                                 exp_col=exp_col,
                                 wage_col=wage_col,
                                 emp_col=emp_col)

    # 全市场 pooled 的工资-经验均值曲线
    pooled_stats = wage_mean_by_bin(df, exp_col=exp_col, wage_col=wage_col, max_bin=max_bin)
    pooled_stats = pooled_stats.rename(columns={"mean": "mean_pooled", "count": "count_pooled"})

    # 各 firm_type 内的工资-经验均值曲线
    work = df.copy()
    work["exp_bin"] = work[exp_col].apply(lambda x: int(np.floor(x)))
    if max_bin is not None:
        work = work[work["exp_bin"] <= max_bin]
    type_stats = (
        work.groupby(["firm_type", "exp_bin"])[wage_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "count_type", "mean": "mean_type"})
    )

    premia: dict[str, float] = {}
    for t in ["small", "medium", "large"]:
        sub = type_stats[type_stats["firm_type"] == t]
        if sub.empty:
            continue
        # 和 pooled 进行 merge，只在两边都有数据的 exp_bin 上拟合
        merged = sub.merge(pooled_stats, on="exp_bin", how="inner")
        merged = merged[(merged["mean_pooled"] > 0) & (merged["mean_type"] > 0)]
        if merged.empty:
            continue

        # 计算每个经验桶的 ratio = mean_type / mean_pooled，权重取该类型在该桶的样本数
        ratio = merged["mean_type"] / merged["mean_pooled"]
        weights = merged["count_type"]
        phi = float(np.average(ratio, weights=weights))
        premia[t] = phi

    return premia

# --------------------------------------------------------
# 1. 读取 Excel 并提取 employee_count 列
# --------------------------------------------------------

# 这里填你的真实路径；如果在本地/服务器上运行，这个路径要改成你的实际位置
EXCEL_PATH = "/Users/joehisaishi/Library/CloudStorage/GoogleDrive-zhaijing@uw.edu/.shortcut-targets-by-id/1A8EblAG1p82E-7dXusgeL8h9ait30Ed9/Job Matching RL/sde_cleaned_Nov24.xlsx"

# 读入 Excel（默认读第一张表，如果你有多张表可以加 sheet_name 参数）
df = pd.read_excel(EXCEL_PATH)
# 假设列名就是 employee_count，如果不一样这里改一下
EMP_COL = "employee_count"

# 检查一下列是否存在（如果不存在就报错提醒）
if EMP_COL not in df.columns:
    raise ValueError(f"列 '{EMP_COL}' 不在数据里，请检查 Excel 的列名。现有列名：{df.columns.tolist()}")

# 提取这一列，并复制一份，避免直接改原 df
emp_counts = df[EMP_COL].copy()
print(f"员工数1是{emp_counts}")
emp_counts = _coerce_numeric(emp_counts, EMP_COL)
print(f"员工数2是{emp_counts}")

# --------------------------------------------------------
# 2. 基础清洗：去掉缺失、非正数、极端异常值
# --------------------------------------------------------



# 只保留 > 0 的值（0 或负数一般是脏数据或不想要的）
emp_counts = emp_counts[emp_counts > 0]

# （可选）去掉极端大值：例如 > 99 分位的当成 outlier
# 这一步视你的数据情况而定，可以先打印看看分布
upper_cap = emp_counts.quantile(0.99)
emp_counts = emp_counts[emp_counts <= upper_cap]


# 如果清洗完数据太少，可以打印检查
print(f"清洗后剩余公司数量: {len(emp_counts)}")

# --------------------------------------------------------
# 3. 计算公司规模分布：分位数 + 小/中/大公司划分
# --------------------------------------------------------

# 计算一些关键分位数：中位数（50%）、90% 分位
# Q50 / Q90 用于定义 small/medium/large 阈值。
q50 = emp_counts.quantile(0.5)
# 用 interpolation="higher" 保证 q90 取到实际数据点，避免 q90 等于 max 时出现“> q90”为空
q90 = emp_counts.quantile(0.9, interpolation="higher")

print(f"公司规模中位数 (Q50): {q50:.1f}")
print(f"公司规模 90 分位 (Q90): {q90:.1f}")

# 按分位数划分公司类型：
#   small: <= Q50
#   medium: Q50 ~ Q90
#   large: >= Q90（用 >= 避免 top 10% 因为重复值被丢光）
small_mask = emp_counts <= q50
medium_mask = (emp_counts > q50) & (emp_counts < q90)
large_mask = emp_counts >= q90

n_total = len(emp_counts)
n_small = small_mask.sum()
n_medium = medium_mask.sum()
n_large = large_mask.sum()

# 各类型占比（用于 MARL 初始化时抽样）
p_small = n_small / n_total
p_medium = n_medium / n_total
p_large = n_large / n_total

print("\n=== 真实数据中公司类型占比（清洗后） ===")
print(f"小公司 (<= Q50): {n_small} 家，占比 {p_small:.2%}")
print(f"中公司 (Q50 ~ Q90): {n_medium} 家，占比 {p_medium:.2%}")
print(f"大公司 (> Q90): {n_large} 家，占比 {p_large:.2%}")

# 为了初始化时给每类一个“代表性规模”，
# 取各类内部的中位数作为代表值（也可以用均值），用于后面抽样初始化。
rep_small = emp_counts[small_mask].median()
rep_medium = emp_counts[medium_mask].median()
rep_large = emp_counts[large_mask].median()

print("\n=== 各类型代表性公司规模（用中位数） ===")
print(f"小公司代表规模: {rep_small:.1f} 员工")
print(f"中公司代表规模: {rep_medium:.1f} 员工")
print(f"大公司代表规模: {rep_large:.1f} 员工")

# --------------------------------------------------------
# 4. 把上面的结果整理成一个“类型配置字典”，方便后面初始化 MARL
# --------------------------------------------------------

# firms_type_config 用来描述：
#   - 每种类型在现实中大概占多少比例（prob）
#   - 初始化时代表性规模是多少（rep_size）
firms_type_config = {
    "small": {
        "prob": p_small,
        "rep_size": int(round(rep_small))
    },
    "medium": {
        "prob": p_medium,
        "rep_size": int(round(rep_medium))
    },
    "large": {
        "prob": p_large,
        "rep_size": int(round(rep_large))
    }
}

print("\n=== 初始化用公司类型配置 ===")
for t, cfg in firms_type_config.items():
    print(f"{t}: prob={cfg['prob']:.2%}, rep_size={cfg['rep_size']}")

# --------------------------------------------------------
# 5. 一个工具函数：根据这些配置初始化 MARL 里的公司规模
# --------------------------------------------------------

def initialize_firms(num_firms: int,
                     type_config: dict,
                     random_state: int | None = None) -> pd.DataFrame:
    """
    根据现实数据估出来的公司规模分布，初始化 MARL 环境中的公司。

    参数：
    - num_firms: 你在 MARL 里想模拟多少家公司
    - type_config: 上面生成的 firms_type_config
    - random_state: 随机种子，保证可重复性

    返回：
    - 一个 DataFrame，每行是一家公司，包含：
        - firm_id: 公司编号
        - firm_type: small / medium / large
        - init_employee_count: 初始化时的员工数量
    """
    rng = np.random.default_rng(random_state)

    # 取出类型列表和对应概率，按真实占比抽样 firm_type
    types = list(type_config.keys())
    probs = np.array([type_config[t]["prob"] for t in types], dtype=float)

    # 防止概率和不为 1 的小数误差，归一化一下
    probs = probs / probs.sum()

    # 为每家公司随机抽一个类型（按真实比例）
    sampled_types_idx = rng.choice(len(types), size=num_firms, p=probs)
    sampled_types = [types[i] for i in sampled_types_idx]

    # 给每家公司一个代表性初始规模（按对应类型的 rep_size）
    init_sizes = [type_config[t]["rep_size"] for t in sampled_types]

    # 组织成一个 DataFrame，方便后面丢给 MARL 环境
    firms_df = pd.DataFrame({
        "firm_id": np.arange(num_firms),
        "firm_type": sampled_types,
        "init_employee_count": init_sizes,
    })

    return firms_df


def to_env_capacities(firms_df: pd.DataFrame,
                      employees_per_agent: float = 1_000.0,
                      min_capacity: int = 1) -> list[int]:
    """
    Map real employee counts to MARL environment capacities.

    employees_per_agent: 多少真实员工换算成 1 个“虚拟工人”槽位。
    较大的比例尺会压缩巨头容量，减少总 num_workers。
    """
    if employees_per_agent <= 0:
        raise ValueError("employees_per_agent must be positive")
    caps = (firms_df["init_employee_count"] / employees_per_agent).round().astype(int)
    caps = np.clip(caps, min_capacity, None)
    return caps.tolist()


# --------------------------------------------------------
# 7. 小仿真：校准信号/利润噪声，使后验方差下降速度贴近工资方差
# --------------------------------------------------------

def simulate_posterior_variances(delta_interview_sq: float,
                                 delta_eps_sq: float,
                                 g0: float = 0.1,
                                 g1: float = 0.05,
                                 theta: float = 0.05,
                                 periods: int = 4,
                                 n_workers: int = 10_000,
                                 seed: int | None = 0) -> list[float]:
    """
    Run a lightweight Monte Carlo that mirrors the paper's belief update.

    Returns a list of Var(tilde_sigma_t) for t=0..periods.

    关键公式：
      interview: tilde_sigma = sigma_true + N(0, delta_interview_sq)
      profit: profit = exp_t + growth + N(0, delta_eps_sq)
      K1 = delta_interview_sq / (delta_interview_sq + delta_eps_sq)
      v_x = (exp_t * K1) / (1 + (exp_t - 1) * K1)
      tilde_sigma_{t+1} = (1 - v_x) * tilde_sigma + v_x * profit
    """
    rng = np.random.default_rng(seed)
    sigma_true = rng.normal(0.0, 1.0, size=n_workers)

    # Interview signal
    interview_noise = rng.normal(0.0, np.sqrt(delta_interview_sq), size=n_workers)
    tilde_sigma = sigma_true + interview_noise

    var_history = [float(np.var(tilde_sigma))]

    exp_t = np.zeros(n_workers)
    for _ in range(periods):
        # Experience growth (always employed in this toy sim)
        growth = (g0 + g1 * sigma_true) * np.exp(-theta * exp_t)
        exp_t_plus = exp_t + growth

        # Profit signal with noise
        eps = rng.normal(0.0, np.sqrt(delta_eps_sq), size=n_workers)
        profit = exp_t + growth + eps

        # Posterior update weight
        denom = delta_interview_sq + delta_eps_sq
        K1 = delta_interview_sq / denom if denom > 0 else 0.0
        vx = (exp_t_plus * K1) / (1.0 + (exp_t_plus - 1.0) * K1)
        vx = np.clip(vx, 0.0, 1.0)

        tilde_sigma = (1.0 - vx) * tilde_sigma + vx * profit
        var_history.append(float(np.var(tilde_sigma)))

        exp_t = exp_t_plus

    return var_history


def calibrate_signal_noise(target_ratio: float,
                           half_life_periods: int = 3,
                           interview_grid: list[float] | None = None,
                           eps_grid: list[float] | None = None,
                           periods: int = 6,
                           **sim_kwargs):
    """
    Grid search (delta_interview_sq, delta_eps_sq) to match posterior variance drop.

    target_ratio: if provided, match Var(t=periods)/Var(t=0) to this value.
    half_life_periods: if target_ratio is None, enforce a half-life every
        `half_life_periods` (e.g., ratio at t=3 = 0.5, t=6 = 0.25).
    periods: number of post-hire periods to simulate; default 6 to check two half-lives.
    Returns:
        best (delta_interview_sq, delta_eps_sq, model_ratio, var_history)

    思路：遍历 (delta_interview_sq, delta_eps_sq) 网格，跑上面的仿真，
    把 Var(t)/Var(0) 与目标轨迹比较，找平均误差最小的组合。
    """
    if half_life_periods <= 0:
        raise ValueError("half_life_periods must be positive")
    interview_grid = interview_grid or [0.05, 0.1, 0.2, 0.4, 0.8]
    eps_grid = eps_grid or [0.02, 0.05, 0.1, 0.2, 0.4]

    best = None
    best_gap = float("inf")

    # Build target path: either single ratio or half-life trajectory
    if target_ratio is not None:
        target_path = {periods: target_ratio}
    else:
        target_path = {
            t: 0.5 ** (t / half_life_periods)
            for t in range(1, periods + 1)
        }

    for d_int in interview_grid:
        for d_eps in eps_grid:
            vh = simulate_posterior_variances(
                delta_interview_sq=d_int,
                delta_eps_sq=d_eps,
                periods=periods,
                **sim_kwargs,
            )
            ratios = {t: vh[t] / vh[0] if vh[0] > 0 else np.inf for t in range(1, periods + 1)}

            # gap = average absolute deviation over the target checkpoints
            gap = float(
                np.mean([abs(ratios[t] - target_path[t]) for t in target_path])
            )
            if gap < best_gap:
                best_gap = gap
                best = (d_int, d_eps, ratios, vh, target_path)

    return best

# --------------------------------------------------------
# 6. 示例：初始化 100 家公司，看一下规模分布
# --------------------------------------------------------


if __name__ == "__main__":
    num_firms = 100
    firms_init = initialize_firms(num_firms, firms_type_config, random_state=42)

    print("\n=== 示例：初始化 100 家公司 ===")
    print(firms_init.head())

    # 看看模拟出来的 small/medium/large 比例是否接近现实数据
    print("\n模拟环境中的公司类型占比：")
    print(firms_init["firm_type"].value_counts(normalize=True))

    # --- Empirical wage variance drop ---
    try:
        wage_df = load_wage_exp()
        stats = wage_variance_by_bin(wage_df)
        ratio = wage_variance_ratio(wage_df)
        print("\n=== 工资方差按经验分桶 ===")
        print(stats)
        print(f"\n经验方差比例 Var(exp≈3)/Var(exp≈0): {ratio:.3f}" if ratio is not None else "无法计算经验方差比例（数据缺失或方差为 0）")

        if ratio is not None:
            best = calibrate_signal_noise(target_ratio=ratio, periods=3, n_workers=5000, seed=0)
            if best:
                d_int, d_eps, ratios, vh, _ = best
                print("\n=== 建议的信号结构（贴合工资方差比例） ===")
                print(f"delta_interview_sq ≈ {d_int}, delta_eps_sq ≈ {d_eps}")
                print(f"模型方差比例 Var_model(t=3)/Var(t=0): {ratios[3]:.3f}")
                print(f"Var history: {vh}")

        # Half-life target: every 3 periods halves the variance
        best_half = calibrate_signal_noise(target_ratio=None, periods=6, half_life_periods=3, n_workers=5000, seed=0)
        if best_half:
            d_int, d_eps, ratios, vh, target_path = best_half
            print("\n=== 建议的信号结构（半衰期 3 年：每 3 年方差约减半） ===")
            print(f"delta_interview_sq ≈ {d_int}, delta_eps_sq ≈ {d_eps}")
            print("目标轨迹:", {t: round(target_path[t], 3) for t in target_path})
            print("模型轨迹:", {t: round(ratios[t], 3) for t in ratios})
            print(f"Var history: {vh}")

        # --- Firm-size-specific wage premia (phi_type) ---
        try:
            premia = estimate_size_wage_premia()
            print("\n=== 按公司规模估计的工资系数 phi_type ===")
            for t, phi in premia.items():
                print(f"{t}: phi_{t} ≈ {phi:.3f}")
        except Exception as e:
            print(f"估计公司规模工资系数时出错: {e}")
    except FileNotFoundError:
        print("找不到工资数据文件，跳过工资方差/信号校准示例。")
