#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision-mc · 零依赖决策蒙特卡洛模拟器

输入：一份 JSON（投资额、贴现率、年现金流、增长率各自的分布）
输出：NPV 的概率分布——亏损概率、P10/P50/P90、90%/95% 置信区间、变量敏感度（龙卷风）

用法：
    python scripts/montecarlo.py examples/store_opening.json
    python scripts/montecarlo.py examples/store_opening.json --trials 20000 --seed 42
    python scripts/montecarlo.py examples/store_opening.json --json out/result.json
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import unicodedata
from typing import Dict, List

FIELDS = ("initial_investment", "discount_rate", "annual_cash_flow", "cash_flow_growth")


def pad(text: str, width: int) -> str:
    """按显示宽度补空格（中文按 2 列算），保证终端对齐。"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)


def sample(spec: Dict) -> float:
    """从一个分布规格中抽一个值；常数直接返回。"""
    if isinstance(spec, (int, float)):
        return float(spec)
    kind = str(spec.get("dist", "triangular")).lower()
    if kind == "normal":
        return random.gauss(float(spec.get("mu", 0.0)), float(spec.get("sigma", 0.0)))
    if kind == "triangular":
        return random.triangular(float(spec["min"]), float(spec["max"]), float(spec["mode"]))
    if kind == "uniform":
        return random.uniform(float(spec["min"]), float(spec["max"]))
    if kind == "lognormal":
        return random.lognormvariate(float(spec.get("mu", 0.0)), float(spec.get("sigma", 1.0)))
    raise ValueError(f"未知分布类型: {kind}")


def central(spec: Dict) -> float:
    """分布的中心值：normal 取均值，triangular 取众数，uniform 取中点。"""
    if isinstance(spec, (int, float)):
        return float(spec)
    kind = str(spec.get("dist", "triangular")).lower()
    if kind == "normal":
        return float(spec.get("mu", 0.0))
    if kind == "triangular":
        return float(spec["mode"])
    if kind == "uniform":
        return (float(spec["min"]) + float(spec["max"])) / 2
    if kind == "lognormal":
        mu, sigma = float(spec.get("mu", 0.0)), float(spec.get("sigma", 1.0))
        return math.exp(mu + sigma ** 2 / 2)
    raise ValueError(f"未知分布类型: {kind}")


def npv_once(cfg: Dict, draw: Dict[str, float]) -> float:
    """单次模拟的 NPV：-I + Σ CF0·(1+g)^(t-1)/(1+r)^t"""
    years = int(cfg["years"])
    value = -draw["initial_investment"]
    for t in range(1, years + 1):
        value += (
            draw["annual_cash_flow"]
            * (1 + draw["cash_flow_growth"]) ** (t - 1)
            / (1 + draw["discount_rate"]) ** t
        )
    return value


def quantile(sorted_values: List[float], q: float) -> float:
    """线性插值分位数，q ∈ [0,1]。"""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def simulate(cfg: Dict, trials: int, seed=None) -> List[float]:
    if seed is not None:
        random.seed(seed)
    return [npv_once(cfg, {k: sample(cfg[k]) for k in FIELDS}) for _ in range(trials)]


def tornado(cfg: Dict, trials: int = 800) -> List[tuple]:
    """单变量敏感性：只让一个变量波动、其余取中心值，比较各自造成的标准差。"""
    rows = []
    for field in FIELDS:
        base = {k: central(cfg[k]) for k in FIELDS}
        vals = []
        for _ in range(trials):
            draw = dict(base)
            draw[field] = sample(cfg[field])
            vals.append(npv_once(cfg, draw))
        rows.append((field, statistics.pstdev(vals)))
    total = sum(v for _, v in rows) or 1.0
    return sorted(((f, v / total) for f, v in rows), key=lambda x: -x[1])


def histogram(vals: List[float], bins: int = 12, width: int = 40) -> List[str]:
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / span * bins), bins - 1)
        counts[idx] += 1
    peak = max(counts)
    lines = []
    for i, c in enumerate(counts):
        left = lo + span * i / bins
        bar = "█" * max(1, round(c / peak * width)) if c else ""
        lines.append(f"{left:>12,.0f} | {bar} {c}")
    return lines


LABELS = {
    "initial_investment": "初始投资",
    "discount_rate": "贴现率",
    "annual_cash_flow": "年现金流",
    "cash_flow_growth": "现金流增长率",
}


def describe(spec: Dict) -> str:
    if isinstance(spec, (int, float)):
        return f"固定 {spec:,.6g}"
    kind = str(spec.get("dist", "triangular")).lower()
    if kind == "normal":
        return f"正态 μ={spec['mu']:g} σ={spec['sigma']:g}"
    if kind == "triangular":
        return f"三角 min={spec['min']:g} mode={spec['mode']:g} max={spec['max']:g}"
    if kind == "uniform":
        return f"均匀 [{spec['min']:g}, {spec['max']:g}]"
    return kind


def main() -> int:
    ap = argparse.ArgumentParser(description="决策蒙特卡洛模拟器（零依赖）")
    ap.add_argument("config", help="JSON 配置文件")
    ap.add_argument("--trials", type=int, default=None, help="模拟次数（默认取配置，未设则 5000）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    ap.add_argument("--json", dest="json_out", default=None, help="把结果另存为 JSON")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    missing = [k for k in FIELDS + ("years",) if k not in cfg]
    if missing:
        print(f"配置缺少字段: {', '.join(missing)}", file=sys.stderr)
        return 2

    trials = args.trials or int(cfg.get("trials", 5000))
    values = simulate(cfg, trials, args.seed)
    ordered = sorted(values)

    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    p10, p50, p90 = (quantile(ordered, q) for q in (0.10, 0.50, 0.90))
    lo90 = quantile(ordered, 0.05)
    hi90 = quantile(ordered, 0.95)
    loss_prob = sum(1 for v in values if v < 0) / len(values)

    print("=" * 62)
    print(f"决策蒙特卡洛模拟 · {trials:,} 次")
    print("=" * 62)
    print("输入假设")
    for field in FIELDS:
        print(f"  {pad(LABELS[field], 14)}{describe(cfg[field])}")
    print(f"  年限      {cfg['years']} 年")
    print()
    print("核心统计")
    print(f"  期望值    {mean:>15,.0f}")
    print(f"  标准差    {sd:>15,.0f}")
    print(f"  P10       {p10:>15,.0f}")
    print(f"  中位数    {p50:>15,.0f}")
    print(f"  P90       {p90:>15,.0f}")
    print()
    print("风险度量")
    print(f"  亏损概率 (NPV<0)   {loss_prob:>6.1%}")
    print(f"  90% 置信区间       [{lo90:,.0f} , {hi90:,.0f}]")
    print()
    print("NPV 分布")
    for line in histogram(values):
        print("  " + line)
    print()
    print("敏感度（龙卷风）")
    for i, (field, share) in enumerate(tornado(cfg), 1):
        print(f"  {i}. {pad(LABELS[field], 14)}贡献 {share:>6.1%} 方差" + ("  <- 最敏感" if i == 1 else ""))
    print()
    print("解读：亏损概率是决策的主指标；P10/P90 说明结果的分布宽度；")
    print("      龙卷风第一项就是最该去校准或对冲的变量。")

    if args.json_out:
        payload = {
            "config": cfg,
            "trials": trials,
            "mean": mean,
            "std": sd,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "ci90": [lo90, hi90],
            "loss_probability": loss_prob,
            "sensitivity": [{"field": f, "variance_share": s} for f, s in tornado(cfg)],
        }
        parent = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
