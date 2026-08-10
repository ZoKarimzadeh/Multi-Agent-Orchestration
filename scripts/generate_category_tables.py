"""Print per-category outcome match rates for Chapter 8 (GPT-5.4-mini only).

Run:  python code/scripts/generate_category_tables.py
"""
import json
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CATEGORIES = ["simple", "multi_step", "cross_system", "error_prone"]


def load(filename):
    with open(os.path.join(BASE, filename)) as f:
        return json.load(f)


def actual_outcome(r):
    if r["escalated"]:  return "escalation"
    if r["success"]:    return "success"
    return "error"


def match_rate(rows, condition, category):
    filtered = [r for r in rows if r["condition"] == condition and r["scenario_category"] == category]
    if not filtered:
        return None
    return 100 * sum(actual_outcome(r) == r["expected_outcome"] for r in filtered) / len(filtered)


def escalation_rate(rows, condition, category):
    filtered = [r for r in rows if r["condition"] == condition and r["scenario_category"] == category]
    if not filtered:
        return None
    return 100 * sum(r["escalated"] for r in filtered) / len(filtered)


def latency(rows, condition, category):
    filtered = [r for r in rows if r["condition"] == condition and r["scenario_category"] == category]
    if not filtered:
        return None
    return sum(r["elapsed_time"] for r in filtered) / len(filtered)


def fmt(val, suffix="%"):
    if val is None:
        return "N/A"
    return f"{val:.1f}{suffix}"


def print_table(title, headers, rows):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)
    widths = []
    for i in range(len(headers)):
        col_vals = [str(row[i]) for row in rows] + [headers[i]]
        widths.append(max(len(v) for v in col_vals) + 2)
    for h, w in zip(headers, widths):
        print(f"{h:>{w}}", end="")
    print()
    print("-" * sum(widths))
    for row in rows:
        for v, w in zip(row, widths):
            print(f"{str(v):>{w}}", end="")
        print()


def main():
    llm = load("llm-base/evaluation/results/gpt-5.4-mini/evaluation_results.json")
    hyb = load("hybrid/evaluation/results/gpt-5.4-mini/evaluation_results.json")
    gemma_llm = load("llm-base/evaluation/results/gemma4-31b/evaluation_results.json")

    conds = ["baseline", "multi_agent_no_context", "multi_agent_with_context",
             "hybrid_no_context", "hybrid_with_context"]
    cond_labels = ["Baseline", "MAS_NoCtx", "MAS_WithCtx", "Hybrid_NoCtx", "Hybrid_WithCtx"]

    # Table 1: Outcome match by category (all conditions)
    headers = ["Category"] + cond_labels
    rows = []
    for cat in CATEGORIES:
        row = [cat]
        for cond in conds:
            source = hyb if "hybrid" in cond else llm
            row.append(fmt(match_rate(source, cond, cat)))
        rows.append(row)
    print_table("OUTCOME MATCH RATE BY CATEGORY (GPT-5.4-mini)", headers, rows)

    # Table 2: H1 - baseline vs MAS with context
    headers = ["Category", "Baseline", "MAS_WithCtx", "Delta"]
    rows = []
    for cat in CATEGORIES:
        bl = match_rate(llm, "baseline", cat)
        ma = match_rate(llm, "multi_agent_with_context", cat)
        delta = (ma - bl) if bl is not None and ma is not None else None
        rows.append([cat, fmt(bl), fmt(ma), f"{delta:+.1f}pp" if delta is not None else "N/A"])
    print_table("H1: BASELINE VS MAS WITH CONTEXT", headers, rows)

    # Table 3: H2 LLM - no context vs with context
    headers = ["Category", "NoCtx", "WithCtx", "Delta"]
    rows = []
    for cat in CATEGORIES:
        nc = match_rate(llm, "multi_agent_no_context", cat)
        wc = match_rate(llm, "multi_agent_with_context", cat)
        delta = (wc - nc) if nc is not None and wc is not None else None
        rows.append([cat, fmt(nc), fmt(wc), f"{delta:+.1f}pp" if delta is not None else "N/A"])
    print_table("H2 LLM: NO CONTEXT VS WITH CONTEXT", headers, rows)

    # Table 4: H2 Hybrid - no context vs with context
    rows = []
    for cat in CATEGORIES:
        nc = match_rate(hyb, "hybrid_no_context", cat)
        wc = match_rate(hyb, "hybrid_with_context", cat)
        delta = (wc - nc) if nc is not None and wc is not None else None
        rows.append([cat, fmt(nc), fmt(wc), f"{delta:+.1f}pp" if delta is not None else "N/A"])
    print_table("H2 HYBRID: NO CONTEXT VS WITH CONTEXT", headers, rows)

    # Table 5: H4 - best LLM vs hybrid
    headers = ["Category", "Best_LLM", "Hybrid", "Delta"]
    rows = []
    for cat in CATEGORIES:
        ll = match_rate(gemma_llm, "multi_agent_no_context", cat)
        hy = match_rate(hyb, "hybrid_with_context", cat)
        delta = (hy - ll) if ll is not None and hy is not None else None
        rows.append([cat, fmt(ll), fmt(hy), f"{delta:+.1f}pp" if delta is not None else "N/A"])
    print_table("H4: BEST LLM VS HYBRID", headers, rows)

    # Table 6: H3 escalation rates
    headers = ["Category", "Baseline", "MAS_NoCtx", "MAS_WithCtx"]
    rows = []
    for cat in CATEGORIES:
        bl = escalation_rate(llm, "baseline", cat)
        nc = escalation_rate(llm, "multi_agent_no_context", cat)
        wc = escalation_rate(llm, "multi_agent_with_context", cat)
        rows.append([cat, fmt(bl), fmt(nc), fmt(wc)])
    print_table("H3: ESCALATION RATES (GPT-5.4-mini)", headers, rows)

    # Table 7: Latency by category
    headers = ["Category", "Baseline", "MAS_NoCtx", "MAS_WithCtx", "Hybrid_NoCtx", "Hybrid_WithCtx"]
    rows = []
    for cat in CATEGORIES:
        row = [cat]
        for cond in conds:
            source = hyb if "hybrid" in cond else llm
            val = latency(source, cond, cat)
            row.append(f"{val:.2f}s" if val else "N/A")
        rows.append(row)
    print_table("LATENCY BY CATEGORY (GPT-5.4-mini)", headers, rows)


if __name__ == "__main__":
    main()
