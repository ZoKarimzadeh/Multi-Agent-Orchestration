import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_result(raw: dict, condition: str) -> dict:
    elapsed = raw.get("elapsed_time", 0.0)
    if elapsed == 0.0 and raw.get("start_time") and raw.get("end_time"):
        elapsed = raw["end_time"] - raw["start_time"]

    tools_used = []
    agents = set()
    for task in raw.get("completed_tasks", []):
        agents.add(task.get("agent", "unknown"))
        for tc in task.get("tool_calls", []):
            tools_used.append(tc.get("tool", ""))

    escalated = raw.get("escalated", False)
    errors = raw.get("errors", [])

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
    }


def completion_time_stats(results: list[dict]) -> dict:
    times = [r["elapsed_time"] for r in results]
    if not times:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
                "ci_lower": 0, "ci_upper": 0, "n": 0}

    n = len(times)
    mean = sum(times) / n
    var = sum((t - mean) ** 2 for t in times) / max(n - 1, 1)
    std = var ** 0.5

    st = sorted(times)
    median = st[n // 2] if n % 2 else (st[n // 2 - 1] + st[n // 2]) / 2

    se = std / math.sqrt(n)
    margin = 1.96 * se

    return {
        "mean": mean, "std": std, "median": median,
        "min": min(times), "max": max(times),
        "ci_lower": mean - margin, "ci_upper": mean + margin,
        "n": n,
    }


def success_rate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "successes": 0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    s = sum(1 for r in results if r.get("success"))
    rate = s / n

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
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "errors": 0, "n": 0}
    with_err = sum(1 for r in results if r.get("error_count", 0) > 0)
    return {"rate": with_err / n, "errors": with_err, "n": n}


def escalation_rate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"rate": 0.0, "escalated": 0, "n": 0}
    esc = sum(1 for r in results if r.get("escalated"))
    return {"rate": esc / n, "escalated": esc, "n": n}


def tool_accuracy(results: list[dict], scenarios: list[dict]) -> dict:
    by_id = {s["id"]: s for s in scenarios}
    n = correct = 0

    for r in results:
        sid = r.get("scenario_id")
        if sid not in by_id:
            continue
        required = set(by_id[sid].get("required_tools", []))
        if not required:
            continue
        called = set(r.get("tool_calls", []))
        n += 1
        if required.issubset(called):
            correct += 1

    return {"accuracy": correct / n if n else 0.0, "correct": correct, "n": n}


def compute_all_metrics(results: list[dict], scenarios: list[dict] | None = None) -> dict:
    m = {
        "completion_time": completion_time_stats(results),
        "success_rate": success_rate(results),
        "error_rate": error_rate(results),
        "escalation_rate": escalation_rate(results),
    }
    if scenarios:
        m["tool_accuracy"] = tool_accuracy(results, scenarios)
    return m