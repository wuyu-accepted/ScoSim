# 维度一：论坛生态真实性

本报告只读复算冻结的隐私安全充分统计量，不执行模拟、网络访问或模型调用。

## 指标定义

成熟线程满足 `final_tick-created_tick>=2`。按回复数分为 zero（0）、small（1–3）、medium（4–12）和 large（至少13）。有回复比例为成熟线程中回复数大于0的比例。四档分布以 base-2 Jensen–Shannon 散度比较：

\[
JS_2(P,Q)=\tfrac12 KL_2(P\|M)+\tfrac12 KL_2(Q\|M),\quad M=(P+Q)/2.
\]

## 结果

| 指标 | 模拟成熟线程 | 131604条聚合真实参照 | 比较量 |
|---|---:|---:|---:|
| 有回复线程占比 | 0.770833 | 0.762386 | 绝对误差 0.008447 |
| zero | 0.229167 | 0.237614 | — |
| small | 0.416667 | 0.481619 | — |
| medium | 0.291667 | 0.240707 | — |
| large | 0.062500 | 0.040060 | — |
| 四档规模 JS 散度（base 2） | — | — | 0.005298 |

3254条审阅线程的精确分位数属于另一参照面板，只作长尾诊断，不与上述聚合参照混算。

## 代码路径

- `campuspulse_core/scripts/recompute_dimensions_1_4_readonly.py`
- `campuspulse_core/src/campuspulse_core/forum_realism.py`
- `campuspulse_core/legacy_evaluation/analyze_forum_ecology_fidelity.py`
