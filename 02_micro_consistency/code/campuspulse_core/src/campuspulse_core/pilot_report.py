"""Standalone aggregate-only HTML reporting for pilot evaluation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .policies import POLICIES


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _number(value: float | int | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _arm_rows(result: dict[str, Any]) -> str:
    rows = []
    for arm_id in result["pilot"]["arms"]:
        summary = result["arm_summary"][arm_id]
        label = POLICIES[arm_id].label
        if summary["suppressed"]:
            rows.append(
                "<tr>"
                f"<th>{_escape(label)}</th>"
                '<td colspan="7"><span class="suppressed">低于隐私阈值，统计量已抑制</span></td>'
                "</tr>"
            )
            continue
        interval = summary["conservative_itt_wilson_95"]
        rows.append(
            "<tr>"
            f"<th>{_escape(label)}</th>"
            f"<td>{summary['eligible_cases']}</td>"
            f"<td>{_percent(summary['allocation_share'])}</td>"
            f"<td>{_percent(summary['delivery_rate'])}</td>"
            f"<td>{_percent(summary['outcome_completeness'])}</td>"
            f"<td>{_percent(summary['conservative_itt_resolution_rate'])} "
            f"<small>[{_percent(interval[0])}, {_percent(interval[1])}]</small></td>"
            f"<td>{_percent(summary['complete_case_resolution_rate'])}</td>"
            f"<td>{_number(summary['mean_governance_cost'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _comparison_rows(result: dict[str, Any]) -> str:
    rows = []
    for arm_id, comparison in result["comparisons_vs_baseline"].items():
        label = POLICIES[arm_id].label
        if comparison["suppressed"]:
            rows.append(
                "<tr>"
                f"<th>{_escape(label)}</th>"
                '<td colspan="6"><span class="suppressed">一个或多个研究臂低于隐私阈值</span></td>'
                "</tr>"
            )
            continue
        interval = comparison["risk_difference_95_pp"]
        balance_suppressed = comparison["balance_diagnostics_suppressed"]
        need_balance = (
            "小单元抑制"
            if balance_suppressed["need_type"]
            else f"{_number(comparison['maximum_need_type_balance_gap_pp'])} pp"
        )
        urgency_balance = (
            "小单元抑制"
            if balance_suppressed["urgency"]
            else f"{_number(comparison['maximum_urgency_balance_gap_pp'])} pp"
        )
        rows.append(
            "<tr>"
            f"<th>{_escape(label)}</th>"
            f"<td>{_number(comparison['conservative_resolution_risk_difference_pp'])} pp</td>"
            f"<td>[{_number(interval[0])}, {_number(interval[1])}] pp</td>"
            f"<td>{_number(comparison['relative_risk'], 2)}</td>"
            f"<td>{_number(comparison['mean_governance_cost_difference'])}</td>"
            f"<td>{need_balance}</td>"
            f"<td>{urgency_balance}</td>"
            "</tr>"
        )
    return "".join(rows)


def _gate_rows(result: dict[str, Any]) -> str:
    rows = []
    for gate, value in result["readiness"]["criteria"].items():
        if isinstance(value, dict):
            for arm_id, passed in value.items():
                rows.append(
                    "<li>"
                    f"<span>{_escape(gate)} · {_escape(arm_id)}</span>"
                    f"<strong class=\"{'pass' if passed else 'fail'}\">{'通过' if passed else '未通过'}</strong>"
                    "</li>"
                )
        else:
            rows.append(
                "<li>"
                f"<span>{_escape(gate)}</span>"
                f"<strong class=\"{'pass' if value else 'fail'}\">{'通过' if value else '未通过'}</strong>"
                "</li>"
            )
    return "".join(rows)


def generate_pilot_report(result: dict[str, Any], output_path: str | Path) -> str:
    """Write one self-contained report that contains aggregate data only."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ready = result["readiness"]["all_automated_gates_passed"]
    status = "自动门槛通过，等待独立复核" if ready else "仅作描述性试点结果"
    audit = result["data_audit"]
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CampusPulse 试点评估 · {_escape(result['pilot']['pilot_id'])}</title>
<style>
:root {{ color-scheme: light; --ink:#172019; --muted:#667168; --line:#dfe4df; --paper:#fbfaf6; --accent:#d45d32; --green:#1d704d; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif; }}
main {{ width:min(1160px,calc(100% - 36px)); margin:0 auto; padding:56px 0 64px; }}
.eyebrow {{ margin:0 0 10px; color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.14em; }}
h1 {{ max-width:850px; margin:0; font-family:Georgia,"Songti SC",serif; font-size:clamp(34px,5vw,66px); line-height:1.08; letter-spacing:-.035em; }}
.lede {{ max-width:760px; margin:20px 0 34px; color:var(--muted); line-height:1.75; }}
.status {{ display:flex; justify-content:space-between; gap:20px; align-items:center; padding:18px 20px; border:1px solid var(--line); border-left:4px solid {'var(--green)' if ready else 'var(--accent)'}; background:#fff; }}
.status strong {{ font-size:18px; }} .status span {{ color:var(--muted); font-size:13px; }}
.audit {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin:28px 0 56px; border:1px solid var(--line); background:var(--line); }}
.audit div {{ padding:18px; background:#fff; }} .audit dt {{ color:var(--muted); font-size:12px; }} .audit dd {{ margin:6px 0 0; font-family:Georgia,serif; font-size:28px; }}
section {{ margin-top:52px; }} h2 {{ margin:0 0 8px; font-size:24px; }} section>p {{ margin:0 0 22px; color:var(--muted); line-height:1.65; font-size:14px; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; min-width:850px; border-collapse:collapse; background:#fff; }} th,td {{ padding:14px 12px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; }} thead th {{ color:var(--muted); font-size:11px; }} tbody th {{ font-weight:700; }} small {{ color:var(--muted); }}
.suppressed {{ color:var(--accent); font-weight:650; }}
.gates {{ display:grid; grid-template-columns:repeat(2,1fr); gap:0 32px; padding:0; list-style:none; }} .gates li {{ display:flex; justify-content:space-between; gap:16px; padding:11px 0; border-bottom:1px solid var(--line); font-size:12px; }} .pass {{ color:var(--green); }} .fail {{ color:var(--accent); }}
.boundary {{ display:grid; grid-template-columns:1fr 1fr; gap:26px; padding:26px; border:1px solid var(--line); background:#fff; }} .boundary h3 {{ margin:0 0 8px; font-size:14px; }} .boundary p {{ margin:0; color:var(--muted); font-size:13px; line-height:1.7; }}
footer {{ margin-top:54px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:11px; line-height:1.7; }}
@media(max-width:720px) {{ .audit,.gates,.boundary {{ grid-template-columns:1fr; }} .status {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body><main>
<p class="eyebrow">CAMPUSPULSE · PILOT EVIDENCE</p>
<h1>校园治理试点的聚合证据检查</h1>
<p class="lede">本报告按照预注册研究臂进行 eligible intention-to-treat 分析。未观察到的主要结局按未解决处理；所有小于隐私阈值的研究臂均被抑制。</p>
<div class="status"><strong>{status}</strong><span>{_escape(result['pilot']['scenario_id'])} · {_escape(result['pilot']['pilot_id'])}</span></div>
<dl class="audit">
<div><dt>接收记录</dt><dd>{audit['accepted_rows']}</dd></div>
<div><dt>符合条件</dt><dd>{audit['eligible_rows']}</dd></div>
<div><dt>已观察结局</dt><dd>{audit['observed_primary_outcomes']}</dd></div>
<div><dt>被拒绝记录</dt><dd>{audit['rejected_rows']}</dd></div>
</dl>
<section><h2>研究臂汇总</h2><p>主要比例为保守 ITT 结果；Wilson 区间未进行多重比较或聚类调整。</p>
<div class="table-wrap"><table><thead><tr><th>研究臂</th><th>eligible N</th><th>分配占比</th><th>实际送达</th><th>结局完整度</th><th>保守 ITT 解决率</th><th>完整病例</th><th>平均成本</th></tr></thead><tbody>{_arm_rows(result)}</tbody></table></div></section>
<section><h2>相对基线</h2><p>差值来自预注册分组的聚合比较；平衡差距用于提示实施审计重点，不是人口公平指标。</p>
<div class="table-wrap"><table><thead><tr><th>处理臂</th><th>绝对差</th><th>95% 区间</th><th>相对风险</th><th>成本差</th><th>需求平衡差</th><th>紧急度平衡差</th></tr></thead><tbody>{_comparison_rows(result)}</tbody></table></div></section>
<section><h2>自动化就绪门槛</h2><p>通过只表示数据满足预声明的机器检查，不能替代随机化实施、污染、选择性记录和统计方案的独立复核。</p><ul class="gates">{_gate_rows(result)}</ul></section>
<section class="boundary"><div><h3>本报告可以支持</h3><p>描述研究臂样本、完整度、送达、运营解决率、成本和未调整组间差异，并判断是否值得进入独立复核。</p></div><div><h3>本报告不能直接支持</h3><p>真实因果优越性、学生福祉改善、长期社会影响，或将试点估计直接写回模拟效果参数。</p></div></section>
<footer>Aggregate-only artifact · case tokens and input paths are not persisted · generated {_escape(result['generated_at_utc'])}</footer>
</main></body></html>"""
    target.write_text(document, encoding="utf-8", newline="\n")
    return str(target)
