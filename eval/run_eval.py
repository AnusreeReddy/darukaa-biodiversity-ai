"""
Evaluation harness.

    python -m eval.run_eval            # run and print report
    python -m eval.run_eval --json     # machine-readable results
    python -m eval.run_eval --markdown eval/REPORT.md

What this measures: behaviour of the system against declared expectations —
whether the right rules fire, whether gates suppress the interventions they
should, whether every recommendation combines enough variables, and whether
every recommendation carries retrieved evidence.

What this does not measure: real ecological outcomes. No field trial is being
simulated and no accuracy claim about ecological prediction is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.conversation import Session  # noqa: E402
from core.evidence import evidence_summary  # noqa: E402
from core.schema import Metric  # noqa: E402

SCENARIOS_FILE = ROOT / "eval" / "scenarios.yaml"


def run_scenario(scenario: Dict) -> Dict:
    session = Session()
    response = None
    for turn in scenario["turns"]:
        response = session.process(turn)

    a = scenario.get("assertions", {})
    failures: List[str] = []
    checks = 0

    rec_ids = [r.intervention_id for r in response.recommendations]

    if "expect_mode" in a:
        checks += 1
        if response.mode != a["expect_mode"]:
            failures.append(f"mode: expected '{a['expect_mode']}', got '{response.mode}'")

    if "min_variables" in a:
        checks += 1
        n = len(response.state.known_core_metrics())
        if n < a["min_variables"]:
            failures.append(f"variables: expected >= {a['min_variables']}, got {n}")

    for metric_name, expected in (a.get("expect_classifications") or {}).items():
        checks += 1
        actual = response.state.classification(Metric(metric_name))
        if actual != expected:
            failures.append(f"classification {metric_name}: expected '{expected}', got '{actual}'")

    for rid in a.get("expect_recommendations", []):
        checks += 1
        if rid not in rec_ids:
            failures.append(f"missing recommendation '{rid}' (got {rec_ids})")

    for rid in a.get("forbid_recommendations", []):
        checks += 1
        if rid in rec_ids:
            failures.append(f"forbidden recommendation '{rid}' was present")

    if "expect_top" in a:
        checks += 1
        top = rec_ids[0] if rec_ids else None
        if top != a["expect_top"]:
            failures.append(f"top recommendation: expected '{a['expect_top']}', got '{top}'")

    if "min_multi_metric" in a:
        checks += 1
        bad = [(r.intervention_id, r.n_metrics()) for r in response.recommendations
               if r.n_metrics() < a["min_multi_metric"]]
        if bad:
            failures.append(f"single-variable reasoning found: {bad}")

    if a.get("require_evidence"):
        checks += 1
        unsupported = [r.intervention_id for r in response.recommendations if not r.evidence]
        if unsupported:
            failures.append(f"recommendations without evidence: {unsupported}")

    if "expect_questions_about" in a:
        checks += 1
        asked = {q.metric.value for q in response.questions}
        allowed = set(a["expect_questions_about"])
        if not asked or not asked.issubset(allowed):
            failures.append(f"clarifying questions targeted {sorted(asked)}, expected a subset of {sorted(allowed)}")

    n_sources, n_passages = evidence_summary(response.recommendations)
    metric_counts = [r.n_metrics() for r in response.recommendations]

    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "turns": len(scenario["turns"]),
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "mode": response.mode,
        "variables_known": len(response.state.known_core_metrics()),
        "recommendations": rec_ids,
        "distinct_sources_cited": n_sources,
        "evidence_passages": n_passages,
        "min_variables_per_recommendation": min(metric_counts) if metric_counts else 0,
        "mean_variables_per_recommendation": (
            round(sum(metric_counts) / len(metric_counts), 2) if metric_counts else 0
        ),
    }


def run_all() -> Dict:
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
        scenarios = yaml.safe_load(fh)["scenarios"]

    results = [run_scenario(s) for s in scenarios]
    passed = sum(1 for r in results if r["passed"])
    total_checks = sum(r["checks"] for r in results)
    total_failures = sum(len(r["failures"]) for r in results)
    with_recs = [r for r in results if r["recommendations"]]

    summary = {
        "scenarios_total": len(results),
        "scenarios_passed": passed,
        "scenarios_failed": len(results) - passed,
        "assertions_total": total_checks,
        "assertions_failed": total_failures,
        "mean_variables_per_recommendation": (
            round(sum(r["mean_variables_per_recommendation"] for r in with_recs) / len(with_recs), 2)
            if with_recs else 0
        ),
        "min_variables_per_recommendation_observed": (
            min(r["min_variables_per_recommendation"] for r in with_recs) if with_recs else 0
        ),
        "scenarios_with_full_evidence_coverage": sum(
            1 for r in with_recs
            if not any("without evidence" in f for f in r["failures"])
        ),
    }
    return {"summary": summary, "results": results}


def print_report(data: Dict) -> None:
    s = data["summary"]
    print("=" * 74)
    print("DARUKAA.EARTH BIODIVERSITY INTELLIGENCE — EVALUATION")
    print("=" * 74)
    for r in data["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{mark}] {r['id']}")
        print(f"       {r['description']}")
        print(f"       turns={r['turns']} mode={r['mode']} variables={r['variables_known']} "
              f"checks={r['checks']}")
        if r["recommendations"]:
            print(f"       fired: {', '.join(r['recommendations'])}")
            print(f"       evidence: {r['evidence_passages']} passages from "
                  f"{r['distinct_sources_cited']} distinct sources; "
                  f"variables/recommendation min={r['min_variables_per_recommendation']} "
                  f"mean={r['mean_variables_per_recommendation']}")
        for f in r["failures"]:
            print(f"       -> {f}")

    print("\n" + "=" * 74)
    print(f"Scenarios : {s['scenarios_passed']}/{s['scenarios_total']} passed")
    print(f"Assertions: {s['assertions_total'] - s['assertions_failed']}/{s['assertions_total']} passed")
    print(f"Mean variables reasoned over per recommendation: {s['mean_variables_per_recommendation']}")
    print(f"Minimum variables observed in any recommendation: {s['min_variables_per_recommendation_observed']}")
    print("=" * 74)


def to_markdown(data: Dict) -> str:
    s = data["summary"]
    L = ["# Evaluation report", "",
         "Generated by `python -m eval.run_eval`. These are behavioural assertions about the",
         "system (which rules fire, whether gates suppress the interventions they should,",
         "whether every recommendation is multi-variable and evidence-backed). They are not",
         "measurements of ecological outcomes.", "",
         "## Summary", "",
         "| Metric | Value |", "|---|---|",
         f"| Scenarios passed | {s['scenarios_passed']} / {s['scenarios_total']} |",
         f"| Assertions passed | {s['assertions_total'] - s['assertions_failed']} / {s['assertions_total']} |",
         f"| Mean variables per recommendation | {s['mean_variables_per_recommendation']} |",
         f"| Minimum variables in any recommendation | {s['min_variables_per_recommendation_observed']} |",
         "", "## Scenarios", ""]
    for r in data["results"]:
        L.append(f"### {'PASS' if r['passed'] else 'FAIL'} — `{r['id']}`")
        L.append("")
        L.append(r["description"])
        L.append("")
        L.append(f"- Turns: {r['turns']}; final mode: `{r['mode']}`; variables known: {r['variables_known']}")
        if r["recommendations"]:
            L.append(f"- Recommendations fired: {', '.join(f'`{x}`' for x in r['recommendations'])}")
            L.append(f"- Evidence: {r['evidence_passages']} passages from {r['distinct_sources_cited']} distinct sources")
            L.append(f"- Variables per recommendation: min {r['min_variables_per_recommendation']}, mean {r['mean_variables_per_recommendation']}")
        if r["failures"]:
            L.append("- Failures:")
            L.extend(f"  - {f}" for f in r["failures"])
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", type=str, default=None)
    args = parser.parse_args()

    data = run_all()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_report(data)
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(data), encoding="utf-8")
        print(f"\nMarkdown report written to {args.markdown}")

    sys.exit(0 if data["summary"]["scenarios_failed"] == 0 else 1)
