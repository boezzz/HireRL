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

# --------------------------------------------------------
# 1. 读取 Excel 并提取 employee_count 列
# --------------------------------------------------------

# 这里填你的真实路径；如果在本地/服务器上运行，这个路径要改成你的实际位置
EXCEL_PATH = "/Users/joehisaishi/Library/CloudStorage/GoogleDrive-zhaijing@uw.edu/.shortcut-targets-by-id/1A8EblAG1p82E-7dXusgeL8h9ait30Ed9/Job Matching RL/sde_cleaned.xlsx"

# 读入 Excel（默认读第一张表，如果你有多张表可以加 sheet_name 参数）
df = pd.read_excel(EXCEL_PATH)

# 假设列名就是 employee_count，如果不一样这里改一下
EMP_COL = "employee_count"

# 检查一下列是否存在（如果不存在就报错提醒）
if EMP_COL not in df.columns:
    raise ValueError(f"列 '{EMP_COL}' 不在数据里，请检查 Excel 的列名。现有列名：{df.columns.tolist()}")

# 提取这一列，并复制一份，避免直接改原 df
emp_counts = df[EMP_COL].copy()

# --------------------------------------------------------
# 2. 基础清洗：去掉缺失、非正数、极端异常值
# --------------------------------------------------------

# 去掉缺失值
emp_counts = emp_counts.dropna()

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
# 我们取各类内部的中位数作为代表值（也可以用均值）
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

    # 取出类型列表和对应概率
    types = list(type_config.keys())
    probs = np.array([type_config[t]["prob"] for t in types], dtype=float)

    # 防止概率和不为 1 的小数误差，归一化一下
    probs = probs / probs.sum()

    # 为每家公司随机抽一个类型（按真实比例）
    sampled_types_idx = rng.choice(len(types), size=num_firms, p=probs)
    sampled_types = [types[i] for i in sampled_types_idx]

    # 给每家公司一个代表性初始规模
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

    employees_per_agent controls how many real employees correspond to one
    simulated worker slot. Increase it to compress very large firms.
    """
    if employees_per_agent <= 0:
        raise ValueError("employees_per_agent must be positive")
    caps = (firms_df["init_employee_count"] / employees_per_agent).round().astype(int)
    caps = np.clip(caps, min_capacity, None)
    return caps.tolist()


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
