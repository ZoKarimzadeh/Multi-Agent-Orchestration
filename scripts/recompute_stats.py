"""Recompute all Chapter 8 statistics from the raw evaluation JSONs.

This script is the single canonical source for every number reported in
Chapter 8 (Evaluation).

* Per-condition descriptive statistics (raw success, outcome match,
  escalation rate, LLM calls/run, elapsed time) are computed directly
  from the raw results and must match the corresponding entries in the
  console.out files produced by run_metrics.py.
* Hypothesis tests (Wilcoxon signed-rank, Cohen's d) use scipy.stats so
  that the implementation is the well-tested, canonical one rather than a
  hand-rolled approximation.  Per-scenario averages are rounded to a fixed
  precision before ranking so that genuinely equal differences (e.g. two
  scenarios that both moved by 20 pp) are treated as ties, matching the
  textbook definition of the Wilcoxon signed-rank test.

Run:  python code/scripts/recompute_stats.py

Outputs are printed to stdout; nothing is written to disk.
"""
import json
import math
import os
from collections import defaultdict

from scipy.stats import wilcoxon, rankdata

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ──────────────────────────────────────────────────────────────────────────────
# Data sources: the 8 evaluation_results.json files
# (4 LLM-based models + 4 hybrid models)
# ──────────────────────────────────────────────────────────────────────────────
FILES = {
    ("gemma",     "llm"):    "llm-base/evaluation/results/gemma4-31b/evaluation_results.json",
    ("gemma",     "hybrid"): "hybrid/evaluation/results/gemma4-31b/evaluation_results.json",
    ("gpt4omini", "llm"):    "llm-base/evaluation/results/gpt-4o-mini/evaluation_results.json",
    ("gpt4omini", "hybrid"): "hybrid/evaluation/results/gpt-4o-mini/evaluation_results.json",
    ("gpt54",     "llm"):    "llm-base/evaluation/results/gpt-5.4-mini/evaluation_results.json",
    ("gpt54",     "hybrid"): "hybrid/evaluation/results/gpt-5.4-mini/evaluation_results.json",
    ("qwen",      "llm"):    "llm-base/evaluation/results/qwen2.5-7b_local/evaluation_results.json",
    ("qwen",      "hybrid"): "hybrid/evaluation/results/qwen2.5-7b_local/evaluation_results.json",
}

# Precision used when rounding per-scenario averages so that equal percentage
# moves are detected as ties by the Wilcoxon rank computation.  Two decimals
# is enough to distinguish the granularity that occurs in this experiment
# (5-run averages produce steps of 20 pp).
ROUND_DP = 2


# ──────────────────────────────────────────────────────────────────────────────
# Outcome mapping (kept identical to metrics.py / generate_figures.py)
# ──────────────────────────────────────────────────────────────────────────────
def actual_outcome(r):
    if r["escalated"]:
        return "escalation"
    if r["success"]:
        return "success"
    return "error"


def is_match(r):
    return actual_outcome(r) == r["expected_outcome"]


def load(model, arch):
    with open(os.path.join(BASE, FILES[(model, arch)])) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Descriptive per-condition statistics
# ──────────────────────────────────────────────────────────────────────────────
def cond_stats(rows):
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["condition"]].append(r)
    out = {}
    for c, g in sorted(by_c.items()):
        n = len(g)
        out[c] = {
            "n":     n,
            "raw":   100 * sum(r["success"] for r in g) / n,
            "match": 100 * sum(is_match(r) for r in g) / n,
            "esc":   100 * sum(r["escalated"] for r in g) / n,
            "calls": sum(r["llm_calls_total"] for r in g) / n,
            "time":  sum(r["elapsed_time"] for r in g) / n,
        }
    return out


def per_scenario_rate(rows, condition, value_fn):
    """Return {scenario_id: mean over runs of value_fn(r)} rounded to ROUND_DP."""
    by_s = defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            by_s[r["scenario_id"]].append(value_fn(r))
    return {s: round(sum(v) / len(v), ROUND_DP) for s, v in sorted(by_s.items())}


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis tests
# ──────────────────────────────────────────────────────────────────────────────
def wilcoxon_signed_rank(diffs):
    """Wilcoxon signed-rank test using scipy.

    Zero differences are dropped (zero_method='wilcox').  Returns a dict with
    keys: W (sum of positive ranks), p (two-sided), r (rank-biserial
    correlation), n_nonzero.
    """
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return {"W": None, "p": None, "r": 0.0, "n_nonzero": 0}

    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    W = float(res.statistic)

    # Rank-biserial correlation: r = (W+ - W-) / (N(N+1)/2)
    ranks = rankdata([abs(x) for x in d], method="average")
    w_pos = sum(r for r, x in zip(ranks, d) if x > 0)
    w_neg = sum(r for r, x in zip(ranks, d) if x < 0)
    r_rb = (w_pos - w_neg) / (n * (n + 1) / 2)

    return {"W": W, "p": float(res.pvalue), "r": r_rb, "n_nonzero": n}


def cohens_d_pooled(a, b):
    """Independent-samples Cohen's d (pooled std).

    Used for the latency comparison because the two conditions are treated
    as independent samples of per-run elapsed time (n=250 each).
    """
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (mb - ma) / sp if sp else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Run the analyses
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("PER-CONDITION STATS  (must match console.out for each model/arch)")
print("=" * 72)
data = {}
for (model, arch) in FILES:
    rows = load(model, arch)
    data[(model, arch)] = rows
    print(f"\n--- {model} / {arch} ---")
    for c, s in cond_stats(rows).items():
        print(f"  {c:28s} raw={s['raw']:5.1f}%  match={s['match']:5.1f}%  "
              f"esc={s['esc']:4.1f}%  calls={s['calls']:.2f}  "
              f"time={s['time']:.2f}s  n={s['n']}")

print()
print("=" * 72)
print("H1 (GPT-5.4-mini, LLM):  baseline vs multi_agent_with_context")
print("  Outcome match rate, Wilcoxon signed-rank test (two-sided)")
print("=" * 72)
rows = data[("gpt54", "llm")]
a = per_scenario_rate(rows, "baseline", is_match)
b = per_scenario_rate(rows, "multi_agent_with_context", is_match)
diffs = [b[s] - a[s] for s in a]
res = wilcoxon_signed_rank(diffs)
print(f"  Baseline   outcome match = {100 * sum(a.values()) / len(a):.1f}%")
print(f"  Multi-agent outcome match = {100 * sum(b.values()) / len(b):.1f}%")
print(f"  Difference = {100 * (sum(b.values()) - sum(a.values())) / len(a):+.1f} pp")
print(f"  Non-zero paired differences = {res['n_nonzero']}")
print(f"  Wilcoxon W = {res['W']}")
print(f"  p-value    = {res['p']:.3f}")
print(f"  Rank-biserial r = {res['r']:.2f}")

print()
print("=" * 72)
print("Baseline vs Hybrid (GPT-5.4-mini):  baseline vs hybrid_with_context")
print("  Outcome match rate, Wilcoxon signed-rank test (two-sided)")
print("=" * 72)
base_rows = data[("gpt54", "llm")]
hyb_rows = data[("gpt54", "hybrid")]
a = per_scenario_rate(base_rows, "baseline", is_match)
b = per_scenario_rate(hyb_rows, "hybrid_with_context", is_match)
diffs = [b[s] - a[s] for s in a]
res = wilcoxon_signed_rank(diffs)
print(f"  Baseline (single LLM agent) outcome match = {100 * sum(a.values()) / len(a):.1f}%")
print(f"  Hybrid with context outcome match = {100 * sum(b.values()) / len(b):.1f}%")
print(f"  Difference = {100 * (sum(b.values()) - sum(a.values())) / len(a):+.1f} pp")
print(f"  Non-zero paired differences = {res['n_nonzero']}")
print(f"  Wilcoxon W = {res['W']}")
print(f"  p-value    = {res['p']:.3f}")
print(f"  Rank-biserial r = {res['r']:.2f}")

print()
print("=" * 72)
print("H1 latency (GPT-5.4-mini, LLM):  baseline vs multi_agent_with_context")
print("  Per-run elapsed time, Wilcoxon + Cohen's d (independent samples)")
print("=" * 72)
ta = [r_["elapsed_time"] for r_ in rows if r_["condition"] == "baseline"]
tb = [r_["elapsed_time"] for r_ in rows if r_["condition"] == "multi_agent_with_context"]
at = per_scenario_rate(rows, "baseline", lambda r_: r_["elapsed_time"])
bt = per_scenario_rate(rows, "multi_agent_with_context", lambda r_: r_["elapsed_time"])
tdiffs = [bt[s] - at[s] for s in at]
tres = wilcoxon_signed_rank(tdiffs)
print(f"  Baseline mean latency   = {sum(ta) / len(ta):.2f}s")
print(f"  Multi-agent mean latency = {sum(tb) / len(tb):.2f}s")
print(f"  Wilcoxon W = {tres['W']}, p = {tres['p']:.2e}")
print(f"  Cohen's d (pooled) = {cohens_d_pooled(ta, tb):.2f}")

print()
print("=" * 72)
print("H2 LLM (GPT-5.4-mini):  multi_agent_no_context vs multi_agent_with_context")
print("  Outcome match rate, Wilcoxon signed-rank test (two-sided)")
print("=" * 72)
a = per_scenario_rate(rows, "multi_agent_no_context", is_match)
b = per_scenario_rate(rows, "multi_agent_with_context", is_match)
diffs = [b[s] - a[s] for s in a]
res = wilcoxon_signed_rank(diffs)
print(f"  No-context  outcome match = {100 * sum(a.values()) / len(a):.1f}%")
print(f"  With-context outcome match = {100 * sum(b.values()) / len(b):.1f}%")
print(f"  Non-zero paired differences = {res['n_nonzero']}")
print(f"  Wilcoxon W = {res['W']}, p = {res['p']:.3f}, r = {res['r']:.2f}")

print()
print("=" * 72)
print("H2 HYBRID (GPT-5.4-mini):  hybrid_no_context vs hybrid_with_context")
print("  Outcome match rate, Wilcoxon signed-rank test (two-sided)")
print("=" * 72)
hrows = data[("gpt54", "hybrid")]
a = per_scenario_rate(hrows, "hybrid_no_context", is_match)
b = per_scenario_rate(hrows, "hybrid_with_context", is_match)
diffs = [b[s] - a[s] for s in a]
res = wilcoxon_signed_rank(diffs)
print(f"  Hybrid no-context  outcome match = {100 * sum(a.values()) / len(a):.1f}%")
print(f"  Hybrid with-context outcome match = {100 * sum(b.values()) / len(b):.1f}%")
print(f"  Difference = {100 * (sum(b.values()) - sum(a.values())) / len(a):+.1f} pp")
print(f"  Non-zero paired differences = {res['n_nonzero']}")
print(f"  Wilcoxon W = {res['W']}, p = {res['p']:.3f}, r = {res['r']:.2f}")

print()
print("=" * 72)
print("H1 percentage-point improvements per model (multi-agent with-context")
print("  vs baseline), used to phrase the H1 narrative in Chapter 8")
print("=" * 72)
for model in ["gpt54", "gpt4omini", "gemma", "qwen"]:
    rows = data[(model, "llm")]
    base = per_scenario_rate(rows, "baseline", is_match)
    with_ctx = per_scenario_rate(rows, "multi_agent_with_context", is_match)
    base_pct = 100 * sum(base.values()) / len(base)
    with_pct = 100 * sum(with_ctx.values()) / len(with_ctx)
    print(f"  {model:10s}: baseline={base_pct:5.1f}%  with-context={with_pct:5.1f}%  "
          f"delta={with_pct - base_pct:+.1f} pp")

print()
print("=" * 72)
print("H4 best-LLM vs best-hybrid comparison")
print("=" * 72)
best_llm = ("gemma", "multi_agent_no_context")
hyb_gpt54 = ("gpt54", "hybrid_with_context")

rows = data[(best_llm[0], "llm")]
a = per_scenario_rate(rows, best_llm[1], is_match)
llm_match = 100 * sum(a.values()) / len(a)
llm_calls = cond_stats(rows)[best_llm[1]]["calls"]
llm_time = cond_stats(rows)[best_llm[1]]["time"]

hrows = data[(hyb_gpt54[0], "hybrid")]
b = per_scenario_rate(hrows, hyb_gpt54[1], is_match)
hyb_match = 100 * sum(b.values()) / len(b)
hyb_calls = cond_stats(hrows)[hyb_gpt54[1]]["calls"]
hyb_time = cond_stats(hrows)[hyb_gpt54[1]]["time"]

print(f"  Best LLM-based  (Gemma 4 31B, multi-agent no-context):")
print(f"    Outcome match = {llm_match:.1f}%, LLM calls/run = {llm_calls:.2f}, "
      f"elapsed = {llm_time:.2f}s")
print(f"  Hybrid (GPT-5.4-mini, with-context):")
print(f"    Outcome match = {hyb_match:.1f}%, LLM calls/run = {hyb_calls:.2f}, "
      f"elapsed = {hyb_time:.2f}s")
print(f"  Outcome match improvement = {hyb_match - llm_match:+.1f} pp")
print(f"  Latency reduction         = {hyb_time - llm_time:+.2f}s "
      f"({(1 - hyb_time / llm_time) * 100:.0f}%)")

print()
print("=" * 72)
print("Escalation rates for H3 (LLM-based baseline vs multi-agent)")
print("=" * 72)
for model in ["gemma", "gpt4omini", "gpt54", "qwen"]:
    rows = data[(model, "llm")]
    stats = cond_stats(rows)
    base_esc = stats["baseline"]["esc"]
    multi_no = stats["multi_agent_no_context"]["esc"]
    multi_yes = stats["multi_agent_with_context"]["esc"]
    print(f"  {model:10s}: baseline={base_esc:4.1f}%  "
          f"multi-agent no-ctx={multi_no:4.1f}%  "
          f"multi-agent with-ctx={multi_yes:4.1f}%")

print()
print("=" * 72)
print("Hybrid escalation rates (range reported in H3)")
print("=" * 72)
hyb_escs = []
for model in ["gemma", "gpt4omini", "gpt54", "qwen"]:
    rows = data[(model, "hybrid")]
    stats = cond_stats(rows)
    hyb_escs.append(stats["hybrid_no_context"]["esc"])
    hyb_escs.append(stats["hybrid_with_context"]["esc"])
    print(f"  {model:10s}: no-ctx={stats['hybrid_no_context']['esc']:4.1f}%  "
          f"with-ctx={stats['hybrid_with_context']['esc']:4.1f}%")
print(f"  Hybrid escalation range = {min(hyb_escs):.1f}% -- {max(hyb_escs):.1f}%")

print()
print("=" * 72)
print("GPT-4o-mini vs Gemma 4 31B on multi-agent no-context outcome match")
print("  (the GPT-4o-mini 'paradox' reported in Section 'The GPT-4o-mini Paradox')")
print("=" * 72)
gpt_rows = data[("gpt4omini", "llm")]
gem_rows = data[("gemma", "llm")]
gpt_match = cond_stats(gpt_rows)["multi_agent_no_context"]["match"]
gem_match = cond_stats(gem_rows)["multi_agent_no_context"]["match"]
print(f"  GPT-4o-mini  multi-agent no-context outcome match = {gpt_match:.1f}%")
print(f"  Gemma 4 31B  multi-agent no-context outcome match = {gem_match:.1f}%")

print()
print("=" * 72)
print("Scenario S39 false-positive audit (cancel a delivered order)")
print("=" * 72)
print("FP = run marked success=True where expected_outcome='error' and the")
print("cancel tool (update_order_status) was never called.")
for arch in ["llm-base", "hybrid"]:
    for model_dir in ["gpt-4o-mini", "gemma4-31b", "gpt-5.4-mini", "qwen2.5-7b_local"]:
        path = os.path.join(BASE, arch, "evaluation", "results",
                            model_dir, "evaluation_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            runs = json.load(f)
        s39 = [r for r in runs if r.get("scenario_id") == "S39"]
        per_cond = defaultdict(int)
        for r in s39:
            cancel_called = any(
                tc.get("tool") == "update_order_status"
                for t in r.get("completed_tasks", []) for tc in t.get("tool_calls", [])
            )
            if r["success"] and r.get("expected_outcome") == "error" and not cancel_called:
                per_cond[r["condition"]] += 1
        for cond, n in sorted(per_cond.items()):
            print(f"  {arch:9s} {model_dir:18s} {cond:32s} FP runs = {n}/5")

print()
print("Done.")
