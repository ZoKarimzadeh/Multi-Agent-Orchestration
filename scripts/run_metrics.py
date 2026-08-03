"""Run metrics.py on evaluation_results.json files and print/save results.

Uses hybrid/evaluation/metrics.py for all JSON files.
Output is both printed to console and saved as console.out next to each JSON.
Run:  python3 code/scripts/run_metrics.py
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # code/scripts/
CODE_ROOT = SCRIPT_DIR.parent                  # code/

# ── CONFIG ───────────────────────────────────────────────────────────────────
METRICS_MODULE = CODE_ROOT / "hybrid/evaluation/metrics.py"

JSON_FILES = [
    CODE_ROOT / "hybrid/evaluation/results/gemma4-31b/evaluation_results.json",
    CODE_ROOT / "hybrid/evaluation/results/gpt-4o-mini/evaluation_results.json",
    CODE_ROOT / "hybrid/evaluation/results/gpt-5.4-mini/evaluation_results.json",
    CODE_ROOT / "hybrid/evaluation/results/qwen2.5-7b_local/evaluation_results.json",
    CODE_ROOT / "llm-base/evaluation/results/gemma4-31b/evaluation_results.json",
    CODE_ROOT / "llm-base/evaluation/results/gpt-4o-mini/evaluation_results.json",
    CODE_ROOT / "llm-base/evaluation/results/gpt-5.4-mini/evaluation_results.json",
    CODE_ROOT / "llm-base/evaluation/results/qwen2.5-7b_local/evaluation_results.json",
]
# ─────────────────────────────────────────────────────────────────────────────

# Add the project root (parent of evaluation/) to sys.path
sys.path.insert(0, str(CODE_ROOT))
from hybrid.evaluation.metrics import compute_all_metrics

try:
    from hybrid.evaluation.scenarios import build_scenarios
    scenarios = build_scenarios()
except Exception:
    scenarios = None


def format_metrics(results, results_path):
    lines = []
    lines.append(f"Loaded {len(results)} records from {results_path.name}")
    lines.append(f"Using metrics from {METRICS_MODULE.name}")
    lines.append("")

    by_cond = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    for cond in sorted(by_cond):
        group = by_cond[cond]
        m = compute_all_metrics(group, scenarios)

        lines.append(f"{'=' * 60}")
        lines.append(f"Condition: {cond}  (n={len(group)})")
        lines.append(f"{'=' * 60}")

        ct = m["completion_time"]
        lines.append(f"  Completion Time : mean={ct['mean']:.2f}s  std={ct['std']:.2f}  "
                      f"median={ct['median']:.2f}  min={ct['min']:.2f}  max={ct['max']:.2f}")
        lines.append(f"                   95% CI [{ct['ci_lower']:.2f}, {ct['ci_upper']:.2f}]")

        sr = m["success_rate"]
        lines.append(f"  Success Rate    : {sr['rate']:.1%}  ({sr['successes']}/{sr['n']})")
        lines.append(f"                   95% CI [{sr['ci_lower']:.1%}, {sr['ci_upper']:.1%}]")

        er = m["error_rate"]
        lines.append(f"  Error Rate      : {er['rate']:.1%}  ({er['errors']}/{er['n']})")

        esc = m["escalation_rate"]
        lines.append(f"  Escalation Rate : {esc['rate']:.1%}  ({esc['escalated']}/{esc['n']})")

        if "outcome_match_rate" in m:
            om = m["outcome_match_rate"]
            lines.append(f"  Outcome Match   : {om['rate']:.1%}  ({om['matches']}/{om['n']})")
            lines.append(f"                   95% CI [{om['ci_lower']:.1%}, {om['ci_upper']:.1%}]")

        if "tool_accuracy" in m:
            ta = m["tool_accuracy"]
            lines.append(f"  Tool Accuracy   : {ta['accuracy']:.1%}  ({ta['correct']}/{ta['n']})")

        if "llm_calls" in m:
            lc = m["llm_calls"]
            lines.append(f"  LLM Calls/run   : {lc['mean']:.2f}  (total={lc['total']}, runs={lc['n']})")

        lines.append("")

    return "\n".join(lines)


for json_path in JSON_FILES:
    if not json_path.exists():
        print(f"SKIP (not found): {json_path}")
        continue

    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    output = format_metrics(results, json_path)

    # Print to console
    print(output)

    # Save console.out next to the JSON file
    out_file = json_path.parent / "console.out"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print(f"Saved: {out_file}\n")
