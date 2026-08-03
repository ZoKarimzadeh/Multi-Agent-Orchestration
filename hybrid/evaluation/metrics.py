"""
Statistical Metrics for Hypothesis Evaluation

This module implements all quantitative metrics used to evaluate the four
research hypotheses (H1–H4).  Each function takes a list of normalised result
records (produced by normalize_result) and returns a statistics dictionary.

All rate metrics (success rate, error rate, etc.) include 95% confidence
intervals so that the difference between experimental conditions can be
assessed with appropriate statistical rigour.  Two CI methods are used:

- Wilson score interval (for proportions): More accurate than the naive
  Wald interval when the sample size is small or the proportion is near 0 or 1.
  Used for success_rate, outcome_match_rate.
- Standard error interval (for means): Used for completion_time.

Primary vs secondary metrics
-----------------------------
outcome_match_rate is the primary evaluation metric for this thesis.
The older success_rate (all tasks returned success=True) is kept as a
secondary metric for comparability with prior work but is not used to draw
conclusions about hypothesis support.

The redefinition arose from a critical flaw discovered during log analysis:

- False positive: In S39 (cancel a delivered order), the fully LLM-based
  multi-agent system called get_order_details, confirmed the delivered
  status, then returned success=True without ever calling
  update_order_status.  The old metric counted this as a success.  The
  correct outcome is error (the request should have been refused).
- False negative: In the hybrid system, the same scenario is correctly
  refused.  The system returns error, which matches the expected outcome.
  The old metric counted this as a failure.

The outcome_match_rate fixes both cases: it compares the system's actual
behaviour (success, error, escalation) against the
ground-truth expected outcome annotated in each scenario.  Correct error
handling and correct refusal are rewarded equally to correct task completion.

This metric redefinition is also why the hybrid architecture's lower raw
success rate (68–78%) is not a contradiction of its better performance
(84–98.4% outcome match): the raw rate is lower precisely because hallucinated
completions are no longer being counted as successes.

"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_result(raw: dict, condition: str) -> dict:
    """Extract and normalise the fields needed for metric computation from a raw result.

    Raw results from run_workflow contain deeply nested structures (completed
    tasks, tool call logs, etc.).  This function flattens them into the compact
    representation needed by the metric functions.

    Parameters
    ----------
    raw : dict
        The raw WorkflowState dictionary returned by run_workflow.
    condition : str
        The experimental condition label ("hybrid_with_context").

    Returns
    -------
    dict
        A flat record with keys: condition, elapsed_time, success, escalated,
        error_count, tool_calls (list of tool names), agents_involved, final_response.
    """
    # Prefer the pre-computed elapsed_time; fall back to end - start
    elapsed = raw.get("elapsed_time", 0.0)
    if elapsed == 0.0 and raw.get("start_time") and raw.get("end_time"):
        elapsed = raw["end_time"] - raw["start_time"]

    # Flatten tool calls and agent names from the nested completed-tasks list
    tools_used = []
    agents = set()
    for task in raw.get("completed_tasks", []):
        agents.add(task.get("agent", "unknown"))
        for tc in task.get("tool_calls", []):
            tools_used.append(tc.get("tool", ""))

    escalated = raw.get("escalated", False)
    errors = raw.get("errors", [])

    # A run is successful only if ALL completed tasks succeeded and it was
    # not escalated.  A run with no completed tasks is not considered successful.
    completed = raw.get("completed_tasks", [])
    success = all(t.get("success", False) for t in completed) if completed else False
    if escalated:
        success = False

    return {
        "condition": condition,
        "elapsed_time": elapsed,
        "success": success,
        "escalated": escalated,
        "error_count": len(errors),
        "tool_calls": tools_used,
        "agents_involved": sorted(agents),
        "final_response": raw.get("final_response", ""),
        "llm_calls_total": raw.get("llm_calls_total", 0),
    }


def completion_time_stats(results: list[dict]) -> dict:
    """Compute descriptive statistics and a 95% confidence interval for elapsed time.

    Uses the standard error of the mean (SEM) with z=1.96 (normal approximation).
    This is appropriate because elapsed time is a continuous variable and the
    sample sizes are large enough for the CLT to apply.

    Parameters
    ----------
    results : list[dict]
        List of normalised result records.

    Returns
    -------
    dict
        Keys: mean, std, median, min, max, ci_lower, ci_upper, n.
    """
    times = [r["elapsed_time"] for r in results]
    if not times:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
                "ci_lower": 0, "ci_upper": 0, "n": 0}

    n = len(times)
    mean = sum(times) / n
    # Bessel-corrected sample variance (divides by n-1)
    var = sum((t - mean) ** 2 for t in times) / max(n - 1, 1)
    std = var ** 0.5

    st = sorted(times)
    median = st[n // 2] if n % 2 else (st[n // 2 - 1] + st[n // 2]) / 2

    # 95% CI using standard error of the mean
    se = std / math.sqrt(n)
    margin = 1.96 * se

    return {
        "mean": mean, "std": std, "median": median,
        "min": min(times), "max": max(times),
        "ci_lower": mean - margin, "ci_upper": mean + margin,
        "n": n,
    }


def success_rate(results: list[dict]) -> dict:
    """Compute the fraction of runs in which all tasks completed successfully.

    Relevant to Hypothesis H1 (multi-agent coordination improves task success).
    Uses the Wilson score interval for the confidence bounds.

    Note: for error-prone scenarios, a run that correctly escalates or returns
    an error may be marked as success=False even though the system behaved
    correctly.  See outcome_match_rate for a more semantically correct metric.
    """
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "successes": 0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    s = sum(1 for r in results if r.get("success"))
    rate = s / n

    # Wilson score 95% confidence interval (more accurate than Wald for small n)
    z = 1.96
    denom = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((rate * (1 - rate) + z ** 2 / (4 * n)) / n) / denom

    return {
        "rate": rate, "successes": s, "n": n,
        "ci_lower": max(0.0, centre - margin),
        "ci_upper": min(1.0, centre + margin),
    }


def error_rate(results: list[dict]) -> dict:
    """Compute the fraction of runs with at least one tool error.

    Relevant to Hypothesis H2 (shared context reduces error rates).
    A run has an error if error_count > 0 in the normalised record.
    """
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "errors": 0, "n": 0}
    with_err = sum(1 for r in results if r.get("error_count", 0) > 0)
    return {"rate": with_err / n, "errors": with_err, "n": n}


def escalation_rate(results: list[dict]) -> dict:
    """Compute the fraction of runs that were escalated to a human agent.

    Relevant to Hypothesis H3 (action-based automation reduces human intervention).
    Lower escalation rates indicate the system can handle more requests autonomously.
    """
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "escalated": 0, "n": 0}
    esc = sum(1 for r in results if r.get("escalated"))
    return {"rate": esc / n, "escalated": esc, "n": n}


def tool_accuracy(results: list[dict], scenarios: list[dict]) -> dict:
    """Compute the fraction of runs where all required tools were actually called.

    A scenario defines the minimum set of tools that a correct solution must use.
    For example, a cancellation scenario requires both get_order_details and
    update_order_status to be called.  A run is considered accurate only if
    the actual tool calls are a superset of the required tools.

    Scenarios with no required tools (pure escalation scenarios) are excluded
    from this calculation because there is nothing to compare against.

    Parameters
    ----------
    results : list[dict]
        Normalised result records; must include scenario_id and tool_calls.
    scenarios : list[dict]
        The full scenario definitions from scenarios.py.
    """
    by_id = {s["id"]: s for s in scenarios}
    n = correct = 0

    for r in results:
        sid = r.get("scenario_id")
        if sid not in by_id:
            continue
        required = set(by_id[sid].get("required_tools", []))
        if not required:
            continue  # Skip scenarios with no tool requirement
        called = set(r.get("tool_calls", []))
        n += 1
        if required.issubset(called):
            correct += 1

    return {"accuracy": correct / n if n else 0.0, "correct": correct, "n": n}


def _system_outcome(result: dict) -> str:
    """Infer the actual system outcome from a result record.

    Maps the raw result to one of three outcome labels that correspond to the
    expected_outcome field in the scenario definitions.  This allows direct
    comparison between what the system did and what was expected.

    Returns
    -------
    str
        One of: "success", "escalation", "error".
    """
    if result.get("escalated"):
        return "escalation"
    if result.get("success"):
        return "success"
    return "error"


def outcome_match_rate(results: list[dict]) -> dict:
    """Fraction of runs where the system outcome matches the expected_outcome.

    This is a more semantically correct primary metric than success_rate for
    evaluating the full scenario set, because it credits the system for correctly
    handling error scenarios (refusing an invalid cancellation, escalating an
    ambiguous request, detecting a double-refund attempt, etc.).

    Uses the Wilson score interval for confidence bounds, same as success_rate.
    """
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "matches": 0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    matches = sum(
        1 for r in results
        if _system_outcome(r) == r.get("expected_outcome", "")
    )
    rate = matches / n

    z = 1.96
    denom = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((rate * (1 - rate) + z ** 2 / (4 * n)) / n) / denom

    return {
        "rate": rate,
        "matches": matches,
        "n": n,
        "ci_lower": max(0.0, centre - margin),
        "ci_upper": min(1.0, centre + margin),
    }


def llm_calls_stats(results: list[dict]) -> dict:
    """Compute the average number of LLM API calls per run.

    Each run records llm_calls_total (the number of LLM calls made during
    that run).  The mean across all runs in a condition tells us the average
    API cost per customer request.

    Returns
    -------
    dict
        Keys: mean, total, n.
    """
    n = len(results)
    if n == 0:
        return {"mean": 0.0, "total": 0, "n": 0}
    total = sum(r.get("llm_calls_total", 0) for r in results)
    return {"mean": total / n, "total": total, "n": n}


def compute_all_metrics(results: list[dict], scenarios: list[dict] | None = None) -> dict:
    """Compute and return all metrics for a group of results.

    This is the convenience function used by the evaluation runner and by any
    post-processing scripts.  It bundles all metric functions into a single call.

    Parameters
    ----------
    results : list[dict]
        Normalised result records for one experimental condition.
    scenarios : list[dict] | None
        Full scenario definitions; required for tool_accuracy, optional otherwise.

    Returns
    -------
    dict
        Keys: completion_time, success_rate, error_rate, escalation_rate,
        outcome_match_rate, tool_accuracy (if scenarios provided).
    """
    m = {
        "completion_time": completion_time_stats(results),
        "success_rate": success_rate(results),
        "error_rate": error_rate(results),
        "escalation_rate": escalation_rate(results),
        "outcome_match_rate": outcome_match_rate(results),
        "llm_calls": llm_calls_stats(results),
    }
    if scenarios:
        m["tool_accuracy"] = tool_accuracy(results, scenarios)
    return m