"""
Main Evaluation Harness

This module orchestrates the full empirical evaluation.
It executes all 50 scenarios under both experimental conditions,
records detailed per-run results, and writes them to JSON for later analysis.

Evaluation structure
--------------------
The experiment is a 2-condition × 50-scenario × N-run full factorial design:

  Conditions (CONDITIONS):
    - hybrid_no_context  — Multi-agent orchestration WITHOUT shared context
                               (H2 control condition).
    - hybrid_with_context — Multi-agent orchestration WITH shared context
                               (H2 treatment condition).

  Total executions = 50 scenarios × 2 conditions × SCENARIOS_PER_CONDITION
  runs per cell.  With the default value of 5 runs this gives 500 executions.

Database reset protocol
-----------------------
The SQLite database is reset to a clean, deterministically seeded state before
every single run (not just once per scenario).  This ensures:
  1. Mutations from one run (e.g. cancelling an order) cannot affect the next.
  2. Results are reproducible given the same model and API key.
  3. Order IDs remain stable across conditions, so cross-condition comparisons
     are valid (same scenario, same data, different orchestration logic).

Retry and error handling
------------------------
Each execution is wrapped in a retry loop with exponential back-off
(up to MAX_RETRIES = 3 attempts, delay = 2 s × attempt number).  If all
attempts fail, a synthetic "SYSTEM ERROR" result is injected so that the outer
loop can continue and the failed run is visible in the final report.

Progress saving
---------------
After completing all runs for each scenario, save_progress() writes the
accumulated results to two files:
  - A timestamped file (evaluation_results_YYYYMMDD_HHMMSS.json) for
    archival and reproducibility.
  - A fixed evaluation_results.json used as the "latest" pointer by the
    analysis notebooks.

Resume capability
-----------------
run_evaluation_resume() loads an existing results file and skips any
(scenario_id, condition, run_number) triple that is already present.  This
makes it safe to restart a crashed evaluation without duplicating results.

"""
import gc
import json
import logging
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import API_CALL_DELAY, RESULTS_DIR, SCENARIOS_PER_CONDITION
from database.setup_db import reset_database
from evaluation.metrics import compute_all_metrics, normalize_result
from evaluation.scenarios import build_scenarios
from orchestration.graph import run_workflow

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = RESULTS_DIR / f"evaluation_log_{ts}.log"
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")
log.addHandler(file_handler)

SEED = 42
"""Random seed used for all random.seed() calls in this module.

Seeding before every run ensures that any stochastic behaviour introduced by
random sampling (e.g. in scenario ordering or tie-breaking) is reproducible.
The LLM itself is not seeded — its temperature is controlled via config.py.
"""

CONDITIONS = [
    "hybrid_no_context",   # H2 control: hybrid architecture, shared context disabled
    "hybrid_with_context", # H2 treatment: hybrid architecture, shared context enabled
]
# Note: earlier evaluation runs also included "multi_agent_no_context" and
# "multi_agent_with_context" (fully LLM-based) across four models (Qwen2.5-7B,
# Gemma4-31B, GPT-4o-mini, GPT-5.4-mini).  Those runs are archived in the
# results directory and are compared against these hybrid conditions in Chapter 8.
# The current harness evaluates only the hybrid conditions because the hybrid
# architecture is the final design proposed in H4.


def execute_single(condition: str, customer_message: str) -> dict:
    """Run one scenario under the specified condition and return the raw result.

    Delegates to run_workflow() with the appropriate flags:
      - context_disabled=True  : hybrid_no_context  (H2 control)
      - context_disabled=False : hybrid_with_context (H2 treatment)

    Both conditions use hybrid=True, meaning the orchestration graph is
    active for both.  The only difference is whether the ContextManager
    accumulates and injects shared state between agent calls.
    """
    if condition == "hybrid_no_context":
        raw = run_workflow(customer_message, context_disabled=True, hybrid=True)
    elif condition == "hybrid_with_context":
        raw = run_workflow(customer_message, context_disabled=False, hybrid=True)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return raw


def _build_enhanced_result(raw: dict, sc: dict, condition: str, run_num: int) -> dict:
    """Merge raw workflow output with scenario metadata into a flat result record.

    The raw dict from run_workflow() is first normalised via
    normalize_result() (which extracts condition, success, error_count, etc.)
    and then enriched with scenario-level fields (id, category, expected_outcome)
    and run-level fields (run_number, tool_calls, llm_calls_total, latencies).

    The context_size_bytes field measures how much shared context the
    ContextManager had accumulated by the end of the run.  This is one of the
    secondary metrics for H2: if context improves accuracy, we expect larger
    context payloads to correlate with lower error rates.  For the no-context
    condition this field is always 0.
    """
    normalized = normalize_result(raw, condition)

    tools_used = []
    for task in raw.get("completed_tasks", []):
        for tc in task.get("tool_calls", []):
            tools_used.append(tc.get("tool", ""))

    latencies = [c.get("latency_ms", 0) for c in raw.get("llm_call_log", [])]

    result = dict(normalized)
    result["scenario_id"] = sc["id"]
    result["scenario_category"] = sc["category"]
    result["run_number"] = run_num
    result["expected_outcome"] = sc["expected_outcome"]
    result["tool_calls"] = tools_used
    result["llm_calls_total"] = len(raw.get("llm_call_log", []))
    result["per_call_latencies_ms"] = latencies
    result["customer_id"] = raw.get("customer_id")
    result["intent"] = raw.get("intent")
    result["entities"] = raw.get("entities", {})
    result["context_disabled"] = raw.get("context_disabled", False)
    result["request"] = raw.get("request", "")
    result["step_count"] = raw.get("step_count", 0)
    result["completed_tasks"] = raw.get("completed_tasks", [])
    result["errors"] = raw.get("errors", [])
    result["llm_call_log"] = raw.get("llm_call_log", [])
    result["conversation_log"] = raw.get("conversation_log", [])

    if not raw.get("context_disabled", False):
        from context.context_manager import ContextManager
        ctx_summary = ContextManager.get_context_summary(raw)
        result["context_size_bytes"] = len(ctx_summary.encode("utf-8"))
    else:
        result["context_size_bytes"] = 0

    return result


def run_evaluation(num_runs: int = SCENARIOS_PER_CONDITION) -> list[dict]:
    """Execute the full evaluation and return all result records.

    Steps:
      1. Reset the database once to establish a clean baseline.
      2. Build the 50 scenarios (with live order IDs from the database).
      3. For each scenario × condition × run:
         a. Reseed random for reproducibility.
         b. Reset the database (so mutations from previous runs don't carry over).
         c. Execute the scenario with up to MAX_RETRIES retry attempts.
         d. Build the enhanced result record.
         e. Append to all_results and save progress to disk.
      4. Print a quick summary table and return all records.

    Parameters
    ----------
    num_runs : int
        Number of repetitions per (scenario, condition) cell.  Defaults to
        SCENARIOS_PER_CONDITION from config.py.  Increasing this value
        improves statistical power but multiplies total API cost linearly.
    """
    random.seed(SEED)
    conn = reset_database()
    conn.close()

    scenarios = build_scenarios()
    total_runs = len(scenarios) * len(CONDITIONS) * num_runs
    log.info(
        "Starting: %d scenarios x %d conditions x %d runs = %d total",
        len(scenarios), len(CONDITIONS), num_runs, total_runs,
    )

    all_results = []
    out_file = RESULTS_DIR / f"evaluation_results_{ts}.json"
    latest = RESULTS_DIR / "evaluation_results.json"

    def save_progress():
        for p in (out_file, latest):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=str)

    for i, sc in enumerate(scenarios, 1):
        sid = sc["id"]
        log.info("[%d/%d] %s  (%s)  %s", i, len(scenarios), sid, sc["category"], sc["description"])

        for condition in CONDITIONS:
            for run_num in range(1, num_runs + 1):
                random.seed(SEED)
                gc.collect()
                conn = reset_database()
                conn.close()

                MAX_RETRIES = 3
                RETRY_DELAY = 2  # seconds

                raw = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        raw = execute_single(condition, sc["customer_message"])
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES:
                            log.error(
                                "  [%s] run %d CRASHED after %d attempts:\n%s",
                                condition, run_num, MAX_RETRIES, traceback.format_exc(),
                            )
                            raw = None
                            break
                        log.warning(
                            "  [%s] run %d attempt %d/%d failed (%s) — retrying in %ds...",
                            condition, run_num, attempt, MAX_RETRIES, exc, RETRY_DELAY * attempt,
                        )
                        time.sleep(RETRY_DELAY * attempt)

                if raw is not None:
                    result = _build_enhanced_result(raw, sc, condition, run_num)
                else:
                    result = {
                        "condition": condition,
                        "scenario_id": sid,
                        "scenario_category": sc["category"],
                        "run_number": run_num,
                        "expected_outcome": sc["expected_outcome"],
                        "elapsed_time": 0.0,
                        "success": False,
                        "escalated": True,
                        "error_count": 1,
                        "tool_calls": [],
                        "agents_involved": [],
                        "final_response": "SYSTEM ERROR",
                        "customer_id": None,
                        "intent": None,
                        "entities": {},
                        "context_disabled": condition == "hybrid_no_context",
                        "request": sc["customer_message"],
                        "step_count": 0,
                        "completed_tasks": [],
                        "errors": [{"agent": "system", "error_type": "crash", "message": traceback.format_exc()}],
                        "llm_calls_total": 0,
                        "per_call_latencies_ms": [],
                        "llm_call_log": [],
                        "conversation_log": [],
                        "context_size_bytes": 0,
                    }

                if API_CALL_DELAY > 0:
                    time.sleep(API_CALL_DELAY)

                all_results.append(result)

                tag = "OK" if result["success"] else ("ESC" if result["escalated"] else "FAIL")
                log.info(
                    "    [%s] run %d: %s  (%.2fs, llm_calls=%d, tools=%d)",
                    condition, run_num, tag,
                    result["elapsed_time"],
                    result["llm_calls_total"],
                    len(result["tool_calls"]),
                )

        save_progress()

    _print_quick_summary(all_results, scenarios)
    log.info("Results saved to %s", out_file)
    log.info("Console log saved to %s", log_file)
    return all_results


def _print_quick_summary(results: list[dict], scenarios: list[dict]) -> None:
    """Print a one-line-per-condition summary table to stdout after the run.

    Columns: success rate, outcome match rate, error rate, escalation rate,
    and mean completion time.  Full statistical analysis (Wilson CIs, t-tests,
    Cohen's d) is performed separately in the analysis notebooks.
    """
    by_cond: dict[str, list[dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<30s} {'Success':>8s} {'OutMatch':>9s} {'Error':>8s} {'Escaln':>8s} {'Time(s)':>8s}")
    print("-" * 80)

    for cond in CONDITIONS:
        group = by_cond.get(cond, [])
        if not group:
            continue
        m = compute_all_metrics(group, scenarios)
        print(
            f"{cond:<30s} "
            f"{m['success_rate']['rate']:>7.1%} "
            f"{m['outcome_match_rate']['rate']:>8.1%} "
            f"{m['error_rate']['rate']:>7.1%} "
            f"{m['escalation_rate']['rate']:>7.1%} "
            f"{m['completion_time']['mean']:>7.2f}"
        )

    print("=" * 80 + "\n")


def run_evaluation_resume(resume_file: str, num_runs: int = SCENARIOS_PER_CONDITION) -> list[dict]:
    """Resume an interrupted evaluation from an existing results file.

    Loads the specified JSON file, builds a set of already-completed
    (scenario_id, condition, run_number) triples, then iterates through all
    scenarios and conditions — skipping any triple already present.

    This is safe to call multiple times: if the file is already complete,
    the function logs the count of existing results and returns immediately
    after the loop (which skips everything).

    The --resume CLI argument (handled in __main__) is the normal way
    to invoke this function.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=resume_file, help="Resume from existing results file")
    args = parser.parse_args()

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            completed = set((r["scenario_id"], r["condition"], r["run_number"]) for r in all_results)
            log.info(f"Resuming with {len(all_results)} existing results, {len(completed)} completed runs")
        else:
            log.warning(f"Resume file {args.resume} not found, starting fresh")
            all_results = []
            completed = set()
    else:
        all_results = []
        completed = set()

    random.seed(SEED)
    conn = reset_database()
    conn.close()

    scenarios = build_scenarios()
    total_runs = len(scenarios) * len(CONDITIONS) * num_runs
    remaining = total_runs - len(completed)
    log.info(
        "Resuming: %d scenarios x %d conditions x %d runs = %d total, %d remaining",
        len(scenarios), len(CONDITIONS), num_runs, total_runs, remaining,
    )

    out_file = RESULTS_DIR / f"evaluation_results_{ts}.json"
    latest = RESULTS_DIR / "evaluation_results.json"

    def save_progress():
        with open(str(out_file), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)

    for i, sc in enumerate(scenarios, 1):
        sid = sc["id"]
        log.info("[%d/%d] %s  (%s)  %s", i, len(scenarios), sid, sc["category"], sc["description"])

        for condition in CONDITIONS:
            for run_num in range(1, num_runs + 1):
                if (sid, condition, run_num) in completed:
                    continue

                random.seed(SEED)
                gc.collect()
                conn = reset_database()
                conn.close()

                MAX_RETRIES = 3
                RETRY_DELAY = 2

                raw = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        raw = execute_single(condition, sc["customer_message"])
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES:
                            log.error(
                                "  [%s] run %d CRASHED after %d attempts:\n%s",
                                condition, run_num, MAX_RETRIES, traceback.format_exc(),
                            )
                            raw = None
                            break
                        log.warning(
                            "  [%s] run %d attempt %d/%d failed (%s) — retrying in %ds...",
                            condition, run_num, attempt, MAX_RETRIES, exc, RETRY_DELAY * attempt,
                        )
                        time.sleep(RETRY_DELAY * attempt)

                if raw is not None:
                    result = _build_enhanced_result(raw, sc, condition, run_num)
                else:
                    result = {
                        "condition": condition,
                        "scenario_id": sid,
                        "scenario_category": sc["category"],
                        "run_number": run_num,
                        "expected_outcome": sc["expected_outcome"],
                        "elapsed_time": 0.0,
                        "success": False,
                        "escalated": True,
                        "error_count": 1,
                        "tool_calls": [],
                        "agents_involved": [],
                        "final_response": "SYSTEM ERROR",
                        "customer_id": None,
                        "intent": None,
                        "entities": {},
                        "context_disabled": condition == "hybrid_no_context",
                        "request": sc["customer_message"],
                        "step_count": 0,
                        "completed_tasks": [],
                        "errors": [{"agent": "system", "error_type": "crash", "message": traceback.format_exc()}],
                        "llm_calls_total": 0,
                        "per_call_latencies_ms": [],
                        "llm_call_log": [],
                        "conversation_log": [],
                        "context_size_bytes": 0,
                    }

                if API_CALL_DELAY > 0:
                    time.sleep(API_CALL_DELAY)

                all_results.append(result)

                tag = "OK" if result["success"] else ("ESC" if result["escalated"] else "FAIL")
                log.info(
                    "    [%s] run %d: %s  (%.2fs, llm_calls=%d, tools=%d)",
                    condition, run_num, tag,
                    result["elapsed_time"],
                    result["llm_calls_total"],
                    len(result["tool_calls"]),
                )

        save_progress()

    _print_quick_summary(all_results, scenarios)
    import shutil
    shutil.copy(str(out_file), str(latest))
    log.info("Results saved to %s", out_file)
    log.info("Console log saved to %s", log_file)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing results file")
    args = parser.parse_args()

    if args.resume:
        run_evaluation_resume(args.resume)
    else:
        run_evaluation()