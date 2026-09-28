"""Standalone Chinese HTML report for empirical credibility evaluation."""

from __future__ import annotations

import html
from pathlib import Path


GRADE_LABELS = {
    "strong": "强",
    "moderate": "中等",
    "weak": "弱",
    "insufficient_data": "样本不足",
}

STATUS_LABELS = {
    "temporally_tested": "已做时间外检验",
    "repeated_window_tested": "已做多事件留一回测",
    "partially_supported_not_scored": "有数据但暂未评分",
    "not_supported": "现有数据不支持",
}

FIELD_GATE_LABELS = {
    "pass": "可直接使用",
    "supporting_only": "仅辅助",
    "proxy_only": "仅作代理",
    "auxiliary_only": "仅探索",
    "reject_temporal_missingness": "拒绝：时间缺失",
    "excluded": "不使用",
}


def _number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _grade_badge(grade: str) -> str:
    return f"<span class='badge {html.escape(grade)}'>{GRADE_LABELS.get(grade, grade)}</span>"


def generate_validation_report(result: dict, output_path: str | Path) -> str:
    scenario_rows = []
    detail_rows = []
    for scenario_id, scenario in sorted(result["scenario_results"].items()):
        temporal = scenario["temporal_stability"]
        generator = scenario["observable_generator_holdout_fidelity"]
        scenario_rows.append(
            "<tr>"
            f"<td>{html.escape(scenario['title'])}</td>"
            f"<td>{_grade_badge(temporal['grade'])}</td>"
            f"<td>{_grade_badge(generator['grade'])}</td>"
            f"<td>{_number(generator['response_rate_absolute_error_pp'])} pp</td>"
            f"<td>{_number(100 * generator['positive_mean_relative_error'] if generator['positive_mean_relative_error'] is not None else None, 1)}%</td>"
            f"<td>{_number(generator['comment_cdf_mae'], 3)}</td>"
            "</tr>"
        )
        for need_type, detail in sorted(
            result["need_type_details_by_scenario"][scenario_id].items()
        ):
            temporal_need = detail["train_persistence_to_holdout"]
            generator_need = detail["calibrated_generator_to_holdout"]
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(scenario['title'])}</td>"
                f"<td><code>{html.escape(need_type)}</code></td>"
                f"<td>{detail['train']['proxy_posts']:,}</td>"
                f"<td>{detail['holdout']['proxy_posts']:,}</td>"
                f"<td>{detail['train']['response_rate']:.1%}</td>"
                f"<td>{detail['holdout']['response_rate']:.1%}</td>"
                f"<td>{detail['calibrated_model']['response_rate']:.1%}</td>"
                f"<td>{_grade_badge(temporal_need['grade'])}</td>"
                f"<td>{_grade_badge(generator_need['grade'])}</td>"
                "</tr>"
            )

    claim_rows = []
    for item in result["claim_evidence_matrix"]:
        claim_rows.append(
            "<tr>"
            f"<td>{html.escape(item['claim'])}</td>"
            f"<td><span class='status {html.escape(item['status'])}'>{STATUS_LABELS[item['status']]}</span></td>"
            f"<td>{html.escape(item['reason'])}</td>"
            f"<td>{html.escape(item['next_evidence'])}</td>"
            "</tr>"
        )

    focus_rows = []
    focus_fold_rows = []
    focus_evidence = result.get("focus_evidence", {})
    for scenario_id, focus in sorted(
        focus_evidence.get("scenario_backtests", {}).items()
    ):
        mean_mae = focus.get("mean_normalized_arrival_profile_mae")
        focus_rows.append(
            "<tr>"
            f"<td>{html.escape(focus['title'])}</td>"
            f"<td>{focus['event_count']}</td>"
            f"<td>{_grade_badge(focus['overall_grade'])}</td>"
            f"<td>{focus['stable_fold_fraction']:.0%}</td>"
            f"<td>{_number(mean_mae, 3)}</td>"
            "<td>仅检验事件内代理帖占比形状</td>"
            "</tr>"
        )
        for fold in focus.get("folds", []):
            arrival = fold["arrival_profile"]
            focus_fold_rows.append(
                "<tr>"
                f"<td>{html.escape(focus['title'])}</td>"
                f"<td>{html.escape(fold['holdout_peak_date'])}</td>"
                f"<td>{fold['training_event_count']}</td>"
                f"<td>{fold['holdout_proxy_posts']:,}</td>"
                f"<td>{_grade_badge(fold['fold_grade'])}</td>"
                f"<td>{_number(arrival['normalized_profile_mae'], 3)}</td>"
                f"<td>{_number(100 * fold['unscored_volume_relative_error'] if fold['unscored_volume_relative_error'] is not None else None, 1)}%</td>"
                "</tr>"
            )

    field_rows = []
    field_audit = focus_evidence.get("field_audit", {})
    for dataset_key, dataset_label in (("posts", "帖子"), ("comments", "评论")):
        for item in field_audit.get(dataset_key, {}).get("fields", []):
            formats = ", ".join(
                f"{key}:{value:,}" for key, value in item.get("format_counts", {}).items()
            ) or "—"
            field_rows.append(
                "<tr>"
                f"<td>{dataset_label}</td>"
                f"<td><code>{html.escape(item['field'])}</code></td>"
                f"<td>{item['nonempty_rate']:.1%}</td>"
                f"<td>{html.escape(FIELD_GATE_LABELS.get(item['field_gate'], item['field_gate']))}</td>"
                f"<td>{html.escape(formats)}</td>"
                "</tr>"
            )

    methodology = result["methodology"]
    conclusion = result["overall_conclusion"]
    coverage = result["empirical_claim_coverage"]
    limitations = "".join(
        f"<li>{html.escape(item)}</li>" for item in result["limitations"]
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CampusPulse 数据可信度评估</title>
<style>
:root{{--ink:#172033;--muted:#667085;--paper:#fff;--line:#dfe5ee;--blue:#2458a6;--bg:#f3f6fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Microsoft YaHei",Segoe UI,sans-serif}}
main{{max-width:1240px;margin:auto;padding:34px 24px 70px}}h1{{margin:0 0 8px;font-size:32px}}h2{{margin:34px 0 14px}}
.subtitle{{color:var(--muted);margin:0}}.notice{{margin:24px 0;padding:16px 18px;border-left:5px solid #d68a00;background:#fff7dd;line-height:1.65}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:22px 0}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:18px;box-shadow:0 4px 18px #243b6410}}
.card small{{color:var(--muted)}}.card strong{{display:block;font-size:24px;margin-top:8px}}table{{width:100%;border-collapse:collapse;background:white;border:1px solid var(--line)}}
th,td{{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#eaf0f8;white-space:nowrap}}code{{font-size:12px}}
.scroll{{overflow-x:auto}}.badge,.status{{display:inline-block;border-radius:999px;padding:4px 9px;font-weight:700;font-size:12px;white-space:nowrap}}
.strong,.temporally_tested,.repeated_window_tested{{background:#dff5e8;color:#17603a}}.moderate,.partially_supported_not_scored{{background:#fff0c7;color:#855800}}
.weak,.not_supported{{background:#fde3e1;color:#982a20}}.insufficient_data{{background:#e9edf3;color:#4c5565}}li{{margin:7px 0;line-height:1.5}}
.foot{{color:var(--muted);font-size:13px;line-height:1.6}}
</style></head><body><main>
<h1>CampusPulse 数据可信度评估</h1>
<p class="subtitle">严格时间留出检验 · 只评估现有论坛数据真正可观测的部分</p>
<div class="notice"><strong>结论边界：</strong>当前数据可以检验“帖子是否得到评论”和“最终累计评论分布”；不能检验真实响应速度、问题是否解决、FAQ/同伴/扩容的因果效果。因此政策结果目前适合做相对压力测试，不适合作为真实效果预测。</div>
<div class="grid">
<div class="card"><small>时间稳定性</small><strong>{_grade_badge(conclusion['temporal_stability'])}</strong></div>
<div class="card"><small>生成器留出期拟合</small><strong>{_grade_badge(conclusion['observable_generator_holdout_fidelity'])}</strong></div>
<div class="card"><small>干预效果实证状态</small><strong style="font-size:18px">尚未实证验证</strong></div>
<div class="card"><small>已时间检验的主张族</small><strong>{coverage['temporally_tested_claim_families']} / {coverage['total_claim_families']}</strong></div>
<div class="card"><small>训练 / 留出窗口</small><strong style="font-size:15px">{methodology['train_window']['start_inclusive']}–{methodology['train_window']['end_exclusive']}<br>{methodology['holdout_window']['start_inclusive']}–{methodology['holdout_window']['end_exclusive']}</strong></div>
</div>
<h2>四场景可观测层评估</h2>
<p class="foot">场景配置1.1版只使用训练窗设定自然回应概率和正评论均值；所有生成器误差均来自未参与这两个参数估计的后期数据。</p>
<div class="scroll"><table><thead><tr><th>场景</th><th>时间稳定性</th><th>生成器留出拟合</th><th>回应率误差</th><th>正评论均值误差</th><th>评论CDF误差</th></tr></thead><tbody>{''.join(scenario_rows)}</tbody></table></div>
<h2>按诉求类型拆解</h2>
<div class="scroll"><table><thead><tr><th>场景</th><th>诉求类型</th><th>训练样本</th><th>留出样本</th><th>训练回应率</th><th>留出回应率</th><th>校准模型</th><th>时间稳定性</th><th>生成器留出拟合</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>
<h2>重点场景的历史事件留一回测</h2>
<p class="foot">事件由严格关键词代理、相对前序基线的提升和最小间隔确定。每次排除一个事件，仅用其余事件形成参考形状；绝对事件规模不评分，且此处不检验政策因果效果。</p>
<div class="scroll"><table><thead><tr><th>重点场景</th><th>合格事件窗</th><th>最差折等级</th><th>稳定折比例</th><th>形状 MAE</th><th>证据范围</th></tr></thead><tbody>{''.join(focus_rows)}</tbody></table></div>
<h2>逐事件折</h2>
<div class="scroll"><table><thead><tr><th>场景</th><th>留出峰值日</th><th>训练事件数</th><th>留出代理帖</th><th>等级</th><th>占比形状 MAE</th><th>绝对量误差（不评分）</th></tr></thead><tbody>{''.join(focus_fold_rows)}</tbody></table></div>
<h2>字段可用性闸门</h2>
<p class="foot">完整不等于有效。帖子标注仅可作机器辅助代理；评论 sentiment/tags/analysis 因后期非随机缺失被拒绝进入当前评分，评论日期因继承父帖日期也不能用于响应时延。</p>
<div class="scroll"><table><thead><tr><th>数据表</th><th>字段</th><th>非空率</th><th>当前闸门</th><th>结构格式计数</th></tr></thead><tbody>{''.join(field_rows)}</tbody></table></div>
<h2>哪些结论有数据支撑</h2>
<div class="scroll"><table><thead><tr><th>主张</th><th>状态</th><th>原因</th><th>补齐证据</th></tr></thead><tbody>{''.join(claim_rows)}</tbody></table></div>
<h2>限制</h2><ul>{limitations}</ul>
<p class="foot">评分规则使用事先声明的实用误差阈值，而不是利用大样本制造显著性。关键词类别可以重叠，且仍需人工标注集验证。</p>
</main></body></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8", newline="\n")
    return str(target)
