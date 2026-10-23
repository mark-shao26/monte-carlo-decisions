# monte-carlo-decisions

**Zero-dependency Monte Carlo for business decisions.** Feed it ranges instead of point estimates, get a probability distribution of NPV: loss probability, P10/P50/P90, and which variable actually drives the outcome.

> 确定性测算告诉你「NPV = 40.9 万，可以干」。蒙特卡洛告诉你：**还有 5.7% 的概率亏钱，最差 10% 的情况 NPV 不到 7.8 万，而结果 62% 的波动来自同一个变量——年现金流。**
> 单点测算给的是幻觉，分布给的是决策依据。

一个约 200 行、只用 Python 标准库的决策模拟器，附带一套完整的 AI 技能（System Prompt + 方法论）。适合：开店/项目投资、预算、融资、任何「参数说不准但必须拍板」的场景。

---

## 为什么不用单点测算

| | 单点测算 | 蒙特卡洛 |
|---|---|---|
| 输入 | 一个猜的数 | 每个变量的区间与分布 |
| 输出 | 一个数 | 完整概率分布 |
| 能回答 | NPV 是多少 | **亏钱的概率是多少**、最坏 10% 有多坏 |
| 决策价值 | 掩盖不确定性 | 把不确定性变成可比较的风险 |

## 快速开始

只需要 Python 3.8+，**不需要安装任何依赖**。

```bash
git clone https://github.com/<mark-shao26>/monte-carlo-decisions.git
cd monte-carlo-decisions

# 用示例配置跑一次
python scripts/montecarlo.py examples/store_opening.json

# 加大样本、固定随机种子（结果可复现）
python scripts/montecarlo.py examples/store_opening.json --trials 20000 --seed 42

# 把结构化结果存成 JSON，方便接进报表
python scripts/montecarlo.py examples/store_opening.json --json out/result.json
```

## 输出示例

`examples/store_opening.json`：开店投资 80 万（±15 万），年现金流 30 万（σ=6 万），年增长 5%（σ=3%），贴现率 10%（8%–14%），5 年。

```
==============================================================
决策蒙特卡洛模拟 · 5,000 次
==============================================================
输入假设
  初始投资      三角 min=700000 mode=800000 max=950000
  贴现率        三角 min=0.08 mode=0.1 max=0.14
  年现金流      正态 μ=300000 σ=60000
  现金流增长率  正态 μ=0.05 σ=0.03
  年限          5 年

核心统计
  期望值            408,891
  标准差            262,890
  P10                78,481
  中位数            399,333
  P90               751,585

风险度量
  亏损概率 (NPV<0)     5.7%
  90% 置信区间       [-18,548 , 855,590]

NPV 分布
      -393,190 | █ 23
      -232,195 | ████ 132
       -71,199 | █████████████ 385
        89,797 | ████████████████████████████ 836
       250,792 | ████████████████████████████████████████ 1208
       411,788 | █████████████████████████████████████ 1114
       572,783 | ████████████████████████ 736
       733,779 | █████████████ 380
       894,775 | █████ 142
     1,055,770 | █ 33
     1,216,766 | █ 9
     1,377,761 | █ 2

敏感度（龙卷风）
  1. 年现金流      贡献  61.6% 方差  <- 最敏感
  2. 现金流增长率  贡献  15.9% 方差
  3. 初始投资      贡献  12.7% 方差
  4. 贴现率        贡献   9.8% 方差
```

## 输入格式

四个变量 + 年限。每个变量可以是固定值，也可以是分布：

```json
{
  "years": 5,
  "trials": 5000,
  "initial_investment": { "dist": "triangular", "min": 700000, "mode": 800000, "max": 950000 },
  "discount_rate":      { "dist": "triangular", "min": 0.08, "mode": 0.10, "max": 0.14 },
  "annual_cash_flow":   { "dist": "normal", "mu": 300000, "sigma": 60000 },
  "cash_flow_growth":   { "dist": "normal", "mu": 0.05, "sigma": 0.03 }
}
```

| 分布 | 参数 | 适用 |
|---|---|---|
| `triangular` | `min` / `mode` / `max` | 只有「最可能值 + 上下限」时的默认选择 |
| `normal` | `mu` / `sigma` | 有历史均值与波动率 |
| `uniform` | `min` / `max` | 只知道区间、无偏好 |
| `lognormal` | `mu` / `sigma` | 右偏变量（收入、估值、期限） |
| 数字 | —— | 确定不变的量 |

NPV 计算口径：`NPV = -I + Σ CF₀·(1+g)^(t-1)/(1+r)^t`

## 目录结构

```
monte-carlo-decisions/
├── scripts/montecarlo.py          # 模拟器（标准库，无依赖）
├── examples/store_opening.json    # 开店投资示例配置
├── SKILL.md                       # AI 技能：触发条件、工作流、输出模板
├── references/                    # 方法论：分布选择、模拟设置、结果解读
└── LICENSE
```

## 作为 AI 技能使用

`SKILL.md` 是一份可直接使用的 System Prompt：它规定了什么时候触发（"这个项目有多大可能亏钱"）、工作流（定变量 → 定分布 → 5000 次模拟 → 解读）、以及固定输出模板（模拟设置 / 核心统计 / 风险度量 / 龙卷风 / 决策启示）。

- **WorkBuddy / Claude Code 等支持技能的客户端**：把整个目录作为技能安装
- **任意大模型**：把 `SKILL.md` 全文作为 system prompt 贴进去，`references/` 一起丢进上下文
- **只想读方法论**：直接看 `references/`

## 方法论要点

1. **只模拟 3–5 个最不确定的变量**，其余按常数处理。变量越多，假设越脆
2. **分布比均值重要**：拿不到历史数据就用三角分布（min / mode / max），别用正态假装精确
3. **5000 次起步**，样本量只影响分位数稳定性，不影响结论方向
4. **读法固定**：亏损概率 → 主指标；P10/P90 → 分布宽度；龙卷风第一项 → 最该校准或对冲的变量

## 局限（请先读这一节）

- **GIGO**：分布假设错了，输出只是把错误量化得更精确
- **不建模相关性**：现实中现金流与贴现率可能同向恶化，本工具按独立抽样处理
- **不是预测**：输出是「在你的假设下」的概率，不是未来
- **不构成投资建议**

## License

MIT。商用、修改、分发都可以，保留版权声明即可。

## 作者

邵奇｜AICFO —— 20 年财务管理与 CFO 决策经验。这套模拟器是 AICFO 专家体系里的通用量化组件，被生存诊断、投资决策、预算偏差等场景调用。
