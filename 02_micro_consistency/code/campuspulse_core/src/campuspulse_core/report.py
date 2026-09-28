"""Dependency-free standalone HTML reporting."""

from __future__ import annotations

import html
from pathlib import Path

from .policies import POLICIES


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_html_report(result: dict, output_path: str | Path) -> str:
    aggregate = result["aggregate"]
    ranking = result["ranking"]
    max_resolution = max(
        (metrics["operational_resolution_rate_mean"] for metrics in aggregate.values()),
        default=1.0,
    ) or 1.0
    rows = []
    bars = []
    for rank, policy_id in enumerate(ranking, start=1):
        metrics = aggregate[policy_id]
        label = POLICIES.get(policy_id).label if policy_id in POLICIES else policy_id
        pareto = "Yes" if policy_id in result["pareto_policies"] else ""
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{html.escape(label)}</td>"
            f"<td>{_pct(metrics['operational_resolution_rate_mean'])}</td>"
            f"<td>{metrics['urgent_unresolved_demand_mean']:.1f}</td>"
            f"<td>{metrics['peak_backlog_mean']:.1f}</td>"
            f"<td>{metrics['coverage_fairness_gap_mean']:.3f}</td>"
            f"<td>{metrics['governance_cost_mean']:.1f}</td><td>{pareto}</td>"
            "</tr>"
        )
        width = 100 * metrics["operational_resolution_rate_mean"] / max_resolution
        bars.append(
            "<div class='bar-row'>"
            f"<span>{html.escape(label)}</span><div class='bar-track'><div class='bar' style='width:{width:.1f}%'></div></div>"
            f"<strong>{_pct(metrics['operational_resolution_rate_mean'])}</strong></div>"
        )

    limitations = "".join(f"<li>{html.escape(text)}</li>" for text in result["limitations"])
    sources = "".join(
        f"<li><code>{html.escape(kind)}</code>: {count} parameters</li>"
        for kind, count in sorted(result["parameter_source_counts"].items())
    )
    definitions = "".join(
        f"<li><code>{html.escape(name)}</code>: {html.escape(definition)}</li>"
        for name, definition in sorted(result["metric_definitions"].items())
    )
    lineage = result["data_lineage"]
    source_files = "".join(
        f"<li><code>{html.escape(name)}</code>: SHA-256 "
        f"<code>{html.escape(details['sha256'])}</code> · {details['bytes']} bytes</li>"
        for name, details in sorted(lineage["source_files"].items())
    )
    evidence = lineage["calibration_and_validation"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CampusPulse - {html.escape(result['scenario_title'])}</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#172033}}
main{{max-width:1180px;margin:auto;padding:36px 24px 64px}}h1{{margin-bottom:6px}}.subtitle{{color:#5b6475;margin-top:0}}
.notice{{background:#fff6df;border-left:5px solid #e4a11b;padding:14px 18px;margin:24px 0;line-height:1.5}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:22px 0}}
.card{{background:white;border:1px solid #e1e6ef;border-radius:12px;padding:18px;box-shadow:0 4px 18px #2030500d}}
.card small{{color:#687386}}.card strong{{display:block;font-size:26px;margin-top:7px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 12px;border-bottom:1px solid #e8ecf2;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#eaf0f8}}
.bar-row{{display:grid;grid-template-columns:220px 1fr 70px;gap:12px;align-items:center;margin:12px 0}}.bar-track{{height:16px;background:#e6ebf2;border-radius:9px;overflow:hidden}}.bar{{height:100%;background:linear-gradient(90deg,#2166d1,#23a7a5)}}
section{{margin-top:30px}}code{{background:#edf1f6;padding:2px 5px;border-radius:4px}}li{{margin:7px 0}}
</style></head><body><main>
<h1>CampusPulse Policy Stress Test</h1><p class="subtitle">{html.escape(result['scenario_title'])} · version {html.escape(result['scenario_version'])}</p>
<div class="notice"><strong>Interpretation boundary:</strong> {html.escape(result['limitations'][0])}</div>
<div class="grid">
<div class="card"><small>Replicates per policy</small><strong>{result['repetitions']}</strong></div>
<div class="card"><small>Common inputs verified</small><strong>{'YES' if result['common_inputs_verified'] else 'NO'}</strong></div>
<div class="card"><small>Top operational policy</small><strong style="font-size:17px">{html.escape(POLICIES[ranking[0]].label)}</strong></div>
<div class="card"><small>Pareto policies</small><strong>{len(result['pareto_policies'])}</strong></div>
</div>
<section><h2>Operational resolution comparison</h2>{''.join(bars)}</section>
<section><h2>Policy metrics</h2><table><thead><tr><th>Rank</th><th>Policy</th><th>Resolution</th><th>Urgent unresolved</th><th>Peak backlog</th><th>Fairness gap</th><th>Cost</th><th>Pareto</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Metric definitions</h2><ul>{definitions}</ul></section>
<section><h2>Parameter provenance</h2><ul>{sources}</ul></section>
<section><h2>Empirical data lineage</h2>
<p>Dataset snapshot: <code>{html.escape(lineage['dataset_snapshot_id'])}</code></p>
<ul>{source_files}</ul>
<p>Chronological validation: {html.escape(str(evidence['train_window']))} → {html.escape(str(evidence['holdout_window']))}; observable generator fidelity: <strong>{html.escape(str(evidence['observable_generator_holdout_fidelity']))}</strong>; intervention effects: <strong>{html.escape(str(evidence['intervention_effect_validity']))}</strong>.</p>
<p>{html.escape(lineage['interpretation'])}</p></section>
<section><h2>Limitations</h2><ul>{limitations}</ul></section>
</main></body></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return str(target)
