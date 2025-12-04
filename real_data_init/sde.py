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

    • 举个简单的虚拟例子（用向下取整后的经验桶 0 和 3）：

  - 桶 0（新人，exp_bin=0）：工资样本 [10k, 14k, 12k, 9k]，方差 Var_low ≈ 4.2 (千元²)。
  - 桶 3（约 3 年经验，exp_bin=3）：工资样本 [18k, 20k, 19k, 17k]，方差 Var_high ≈ 1.0 (千元²)。

  比值：

  - Var_high / Var_low ≈ 1.0 / 4.2 ≈ 0.24

  含义：3 年经验组的工资方差是新人组的约 24%。如果你在真实数据算出的比值是 0.24，模型就会尝试把信号噪声调到让模拟中后验方差也下降到约 24% 的水平。
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


def wage_variance_ratio_path(df: pd.DataFrame,
                             exp_col: str = "year of experience",
                             wage_col: str = "salary",
                             bin_step: int = 1,
                             max_bin: int | None = None) -> dict[int, float]:
    """
    Compute Var(wage | exp_bin=t) / Var(wage | exp_bin=0) for t=bin_step,2*bin_step,...

    Returns a dict mapping exp_bin -> ratio. Bins missing data or with nonpositive
    variance are skipped. If bin 0 is missing or has nonpositive variance, returns {}.
    """
    stats = wage_variance_by_bin(df, exp_col=exp_col, wage_col=wage_col, max_bin=max_bin)
    bins = stats.set_index("exp_bin")
    if 0 not in bins.index:
        return {}
    v0 = bins.loc[0, "var"]
    if pd.isna(v0) or v0 <= 0:
        return {}
    ratios: dict[int, float] = {}
    max_t = max(bins.index)
    if max_bin is not None:
        max_t = min(max_t, max_bin)
    for t in range(bin_step, max_t + 1, bin_step):
        if t not in bins.index:
            continue
        vt = bins.loc[t, "var"]
        if pd.isna(vt) or vt <= 0:
            continue
        ratios[t] = float(vt / v0)
    return ratios


def wage_mean_by_bin(df: pd.DataFrame,
                     exp_col: str = "year of experience",
                     wage_col: str = "salary",
                     max_bin: int | None = None) -> pd.DataFrame:
    """
    Bin wages by floor(experience) and return count/mean per bin.
    This is similar to wage_variance_by_bin but keeps only count/mean.
      - 数据：[(0.4 年, 12k), (0.9 年, 13k), (1.2 年, 15k), (1.8 年, 14k), (3.1 年, 20k)]
  - 按 floor 取整分桶：
      - 桶 0：工资 [12k, 13k] → count=2, mean=12.5k
      - 桶 1：工资 [15k, 14k] → count=2, mean=14.5k
      - 桶 3：工资 [20k] → count=1, mean=20k

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
    df.to_csv("/Users/joehisaishi/Library/CloudStorage/GoogleDrive-zhaijing@uw.edu/.shortcut-targets-by-id/1A8EblAG1p82E-7dXusgeL8h9ait30Ed9/Job Matching RL/load_wage_with_size.csv", index = False)
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

          1. 先分桶并求均值

  - 桶 0（exp_bin=0）：全体均值=10；small 均值=9；small 样本数=100
  - 桶 1（exp_bin=1）：全体均值=12；small 均值=10.5；small 样本数=80

  2. 每个桶的 ratio

  - 桶 0：ratio = 9 / 10 = 0.90
  - 桶 1：ratio = 10.5 / 12 ≈ 0.875

  3. 用该类型在该桶的样本数做权重，求加权平均

  - 加权平均 = (0.90 * 100 + 0.875 * 80) / (100 + 80) ≈ 0.888

  结果：phi_small ≈ 0.888。解释：在相同经验下，小公司平均工资 ≈ 全体平均工资的 88.8%。
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
print(df.columns)

# 轻度数值化：去掉千分位/空格并转成数值，避免 agg 时字符串报错
for _col in ["year of experience", "salary", "employee_count"]:
    if _col in df.columns:
        df[_col] = _coerce_numeric(df[_col], _col)

stats = df.agg(
    {
        "year of experience": ["count", "mean", "std", "median", "max"],
        "salary": ["count", "mean", "std", "median", "max"],
        "employee_count": ["count", "mean", "std", "median", "max"],
    }
)

stats = stats.T
try:
    stats.to_csv("/Users/joehisaishi/Library/CloudStorage/GoogleDrive-zhaijing@uw.edu/.shortcut-targets-by-id/1A8EblAG1p82E-7dXusgeL8h9ait30Ed9/Job Matching RL/summary_statistics.csv")
except PermissionError as e:
    print(f"[warn] could not write summary_statistics.csv: {e}")

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
#   - samples: 该类型下清洗后的全部员工数样本，用于类型内再抽样
small_samples = emp_counts[small_mask].to_numpy()
medium_samples = emp_counts[medium_mask].to_numpy()
large_samples = emp_counts[large_mask].to_numpy()

firms_type_config = {
    "small": {
        "prob": p_small,
        "rep_size": int(round(rep_small)),
        "samples": small_samples
    },
    "medium": {
        "prob": p_medium,
        "rep_size": int(round(rep_medium)),
        "samples": medium_samples
    },
    "large": {
        "prob": p_large,
        "rep_size": int(round(rep_large)),
        "samples": large_samples
    }
}

print("\n=== 初始化用公司类型配置 ===")
for t, cfg in firms_type_config.items():
    print(f"{t}: prob={cfg['prob']:.2%}, rep_size={cfg['rep_size']}, sample_size={len(cfg['samples'])}")

# --------------------------------------------------------
# 5. 一个工具函数：根据这些配置初始化 MARL 里的公司规模
# --------------------------------------------------------

def initialize_firms(num_firms: int,
                     type_config: dict,
                     random_state: int | None = None,
                     sample_strategy: str = "representative") -> pd.DataFrame:
    """
    根据现实数据估出来的公司规模分布，初始化 MARL 环境中的公司。

    参数：
    - num_firms: 你在 MARL 里想模拟多少家公司
    - type_config: 上面生成的 firms_type_config
    - random_state: 随机种子，保证可重复性
    - sample_strategy: "representative" 使用类型中位数，"empirical" 在该类型样本内再随机抽样

    返回：
    - 一个 DataFrame，每行是一家公司，包含：
        - firm_id: 公司编号
        - firm_type: small / medium / large
        - init_employee_count: 初始化时的员工数量
    """
    if sample_strategy not in {"representative", "empirical"}:
        raise ValueError("sample_strategy must be 'representative' or 'empirical'")

    rng = np.random.default_rng(random_state)

    # 取出类型列表和对应概率，按真实占比抽样 firm_type
    types = list(type_config.keys())
    probs = np.array([type_config[t]["prob"] for t in types], dtype=float)

    # 防止概率和不为 1 的小数误差，归一化一下
    probs = probs / probs.sum()

    # 为每家公司随机抽一个类型（按真实比例）
    sampled_types_idx = rng.choice(len(types), size=num_firms, p=probs)
    sampled_types = [types[i] for i in sampled_types_idx]

    # 给每家公司一个初始规模
    init_sizes: list[int] = []
    for t in sampled_types:
        if sample_strategy == "representative":
            size = type_config[t]["rep_size"]
        else:
            samples = type_config[t].get("samples")
            if samples is not None and len(samples) > 0:
                size = float(rng.choice(samples))
            else:
                size = type_config[t]["rep_size"]
        init_sizes.append(int(round(size)))

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

def simulate_posterior_variances(delta_interview0_sq: float,
                                 delta_profit_sq: float,
                                 lambda_interview: float = 0.0,
                                 c_interview: float | np.ndarray = 0.0,
                                 g0: float = 0.1,
                                 g1: float = 0.05,
                                 theta: float = 0.05,
                                 periods: int = 4,
                                 n_workers: int = 10_000,
                                 seed: int | None = 0,
                                 profit_fn=None) -> list[float]:
    """
    Run a lightweight Monte Carlo that mirrors the updated on-the-job learning setup
    with interview noise that decays with interview cost.

    Returns a list of Var(tilde_sigma_t) for t=0..periods.

    关键公式（与论文更新对齐）：
      interview 信号:  tilde_sigma_interview = sigma_j + N(0, delta_interview_sq(c_interview)),
                       其中 delta_interview_sq = delta_interview0_sq * exp(-lambda_interview * c_interview)
      利润信号:        p_{t} = f( exp_{t-1} + (g0 + g1 * sigma_j) * 1{employed} * exp(-theta * exp_{t-1}) ) + N(0, delta_profit_sq)
      权重:            K1 = delta_interview_sq / (delta_interview_sq + delta_profit_sq)
                      v_t = (exp_t * K1) / (1 + (exp_t - 1) * K1)   （这里 exp_t 是更新后的经验状态，向量）
      后验更新:         tilde_sigma_{t+1} = (1 - v_t) * tilde_sigma_interview + v_t * sigma_j

    profit_fn: 可选的 f(·) 变换；默认 identity。
    """
    profit_fn = profit_fn or (lambda x: x)

    rng = np.random.default_rng(seed)
    sigma_true = rng.normal(0.0, 1.0, size=n_workers)

    # Cost-sensitive interview noise: delta_interview_sq = delta_interview0_sq * exp(-lambda_interview * c_interview)
    c_arr = np.asarray(c_interview)
    if c_arr.shape == ():
        c_arr = np.full(n_workers, float(c_arr))
    delta_interview_sq = delta_interview0_sq * np.exp(-lambda_interview * c_arr)
    delta_interview_sq = np.clip(delta_interview_sq, 1e-12, None)  # guard against numerical issues
    interview_noise = rng.normal(0.0, np.sqrt(delta_interview_sq), size=n_workers)
    tilde_sigma_interview = sigma_true + interview_noise
    tilde_sigma = tilde_sigma_interview.copy()

    var_history = [float(np.var(tilde_sigma))]

    exp_t = np.zeros(n_workers)
    for _ in range(periods):
        employed = np.ones(n_workers)  # 如果需要失业/空窗，可在此处修改
        # Experience growth uses exp_{t-1}
        growth = (g0 + g1 * sigma_true) * employed * np.exp(-theta * exp_t)
        exp_t_plus = exp_t + growth  # exp_t 更新后的状态

        # Profit signal per updated formula, with optional transform f(·)
        eps = rng.normal(0.0, np.sqrt(delta_profit_sq), size=n_workers)
        profit_mean = exp_t + growth
        profit_signal = profit_fn(profit_mean) + eps  # kept for completeness; not used in posterior per new formula

        # Posterior update weight
        denom = delta_interview_sq + delta_profit_sq
        denom = np.where(denom <= 0, np.inf, denom)
        K1 = delta_interview_sq / denom
        vx = (exp_t_plus * K1) / (1.0 + (exp_t_plus - 1.0) * K1)
        vx = np.clip(vx, 0.0, 1.0)

        tilde_sigma = (1.0 - vx) * tilde_sigma_interview + vx * sigma_true
        var_history.append(float(np.var(tilde_sigma)))

        exp_t = exp_t_plus

    return var_history


def calibrate_signal_noise(target_ratio: float,
                           target_path: dict[int, float] | None = None,
                           half_life_periods: int = 3,
                           interview_grid: list[float] | None = None,
                           eps_grid: list[float] | None = None,
                           periods: int = 6,
                           **sim_kwargs):
    """
    Grid search (delta_interview0_sq, delta_profit_sq) to match posterior variance drop.

    target_ratio: if provided, match Var(t=periods)/Var(t=0) to this value.
    half_life_periods: if target_ratio is None, enforce a half-life every
        `half_life_periods` (e.g., ratio at t=3 = 0.5, t=6 = 0.25).
    target_path: optional dict {timestep: target_ratio_t}; overrides target_ratio/half-life.
    periods: number of post-hire periods to simulate; default 6 to check two half-lives.
    Returns:
        best (delta_interview0_sq, delta_profit_sq, model_ratio, var_history)

    思路：遍历 (delta_interview0_sq, delta_eps_sq) 网格，跑上面的仿真，
    把 Var(t)/Var(0) 与目标轨迹比较，找平均误差最小的组合。
    """
    if half_life_periods <= 0:
        raise ValueError("half_life_periods must be positive")
    if target_ratio is not None and target_path is not None:
        raise ValueError("Provide either target_ratio or target_path, not both.")

    # 默认搜索网格；允许调用方自定义
    if interview_grid is None:
        interview_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
    if eps_grid is None:
        eps_grid = [0.02, 0.05, 0.1, 0.2, 0.4]
    if len(interview_grid) == 0 or len(eps_grid) == 0:
        raise ValueError("interview_grid and eps_grid must be non-empty")

    best = None
    best_gap = float("inf")

    # Build target path: either single ratio or half-life trajectory
    if target_path:
        target_path = dict(target_path)  # shallow copy
    elif target_ratio is not None:
        target_path = {periods: target_ratio}
    else:
        target_path = {
            t: 0.5 ** (t / half_life_periods)
            for t in range(1, periods + 1)
        }

    for d_int in interview_grid:
        for d_eps in eps_grid:
            vh = simulate_posterior_variances(
                delta_interview0_sq=d_int,
                delta_profit_sq=d_eps,
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

