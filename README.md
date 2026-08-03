# Multi-Agent Orchestration for Action-Oriented Customer Service Automation

**Master Thesis — Proof-of-Concept Implementation**
**Zoha Karimzadeh · Berliner Hochschule für Technik (BHT) · M.Sc. Data Science**
**Supervisor: Prof. Dr. Stefan Edlich**

---

## Overview

This directory includes the complete Python source code to support the master thesis *"Multi-Agent Orchestration for Action-Oriented Customer Service Automation."* It demonstrates how a coordinated group of specialised AI agents can resolve concrete e-commerce customer-service requests (order tracking, account updates, billing queries, refund processing) in a manner more reliable than using a single monolithic LLM agent.

It relies on an SQLite database to reproduce the exact same experimental conditions on any machine.

---

## Research Hypotheses

The implementation is designed to test **four hypotheses** introduced in Chapter 4
(Analysis) of the thesis:

| ID | Hypothesis | Operationalised as | Status |
|----|------------|--------------------|--------|
| H1 | Coordinated multi-agent orchestration achieves a higher task-success rate and lower latency than a single-agent baseline | Outcome match rate and elapsed time: `hybrid_with_context` vs `baseline` | **Supported** for large models (GPT-5.4-mini, Gemma4-31B); exception: Qwen2.5-7B where multi-agent overhead exceeds benefit |
| H2 | Shared workflow context reduces error rates in multi-step tasks | Outcome match rate: `hybrid_with_context` vs `hybrid_no_context` | **Reframed**: Context-sharing is only reliable when the execution layer is deterministic. With a fully LLM-based model, the structure of the context (i.e., the input) is given to the LLM as an unstructured text string; in this case, the LLM will determine the relative importance assigned to each portion of the context with non-deterministic decision-making resulting in no consistent improvements over baseline. Within the hybrid model, the structured plan is sent to the rule-based agent(s), who cannot simply choose to disregard it; in this manner, hybrid-with-context outperforms hybrid-no-context by 2–3 pp. The LLM-only failure of H2 becomes a motivation for the hybrid design. |
| H3 | Action-based automation reduces the need for human intervention | Escalation rate and false-positive rate across conditions | **Supported**: hybrid conditions produce zero hallucinated completions (false positives) and more reliable rejection of genuinely invalid requests, reducing human workload compared to fully LLM-based conditions |
| H4 | A hybrid LLM-rule architecture achieves comparable (or better) task-success while reducing latency, API cost, and hallucination rate compared to the best fully LLM-based configuration | Outcome match rate, LLM call count, elapsed time, and cost: hybrid conditions vs best LLM-based condition | **Supported**: hybrid achieves 98.4% outcome match vs 86.8% for the best LLM-based condition (+11.6 pp); ~1.9 LLM calls/request vs 5.66–6.97; 55% latency reduction; 63% cost reduction; hallucination rate in execution layer = 0 by construction |

Two primary experimental conditions are evaluated (both use the hybrid architecture):

- **`hybrid_no_context`** — rule-based tool dispatch, shared context *disabled* (H2 control)
- **`hybrid_with_context`** — rule-based tool dispatch, shared context *enabled* (H2 treatment)

A third condition (`baseline`) is implemented in `evaluation/baseline.py` as a single
LLM agent that receives all tools at once, matching the architecture studied in prior
work (see Chapter 3, Related Work).

Earlier evaluation runs also tested two fully LLM-based multi-agent conditions
(`multi_agent_no_context`, `multi_agent_with_context`) across four models (Qwen2.5-7B,
Gemma4-31B, GPT-4o-mini, GPT-5.4-mini). Their results appear in the full comparison
table below and in Chapter 8.

---

## Directory Structure

```
code/
│
├── README.md                  ← This file
│
├── scripts/                   ← Utility scripts for figures, stats, and metrics
│   ├── generate_figures.py    ← Script to generate thesis figures from evaluation results
│   ├── recompute_stats.py     ← Script to recompute statistical metrics from saved results
│   ├── run_metrics.py         ← Script to run metrics analysis on evaluation output
│   └── find_failed_status_tool_errors.py  ← Script to find tool validation errors in results
│
└── hybrid/                    ← Primary implementation (hybrid LLM + rule-based architecture)
    ├── config.py              ← Central configuration (model, paths, API key)
    ├── requirements.txt       ← Python dependencies
    │
    ├── agents/                ← Specialist agent implementations
    │   ├── base_agent.py      ← Abstract base class shared by all agents
    │   ├── agent_manager.py   ← Orchestrator: intent classification + response generation
    │   ├── order_agent.py     ← Handles order status, tracking, cancellations, returns
    │   ├── account_agent.py   ← Handles customer profile, email, address, payment updates
    │   └── billing_agent.py   ← Handles payment status, refunds, billing history, duplicate detection
    │
    ├── orchestration/
    │   └── graph.py           ← LangGraph workflow: nodes, edges, conditional routing
    │
    ├── context/
    │   └── context_manager.py ← Shared workflow state: creation and summarisation
    │
    ├── database/
    │   ├── setup_db.py        ← Schema creation and deterministic sample-data seeding
    │   └── ecommerce.db       ← SQLite database (auto-generated; do not edit by hand)
    │
    ├── tools/
    │   └── db_tools.py        ← All database read/write functions used by agents as tools
    │
    └── evaluation/
        ├── scenarios.py       ← 50 test scenarios across four complexity categories
        ├── metrics.py         ← Statistical metrics (success rate, error rate, CI, etc.)
        ├── baseline.py        ← Single-agent baseline for H1 comparison
        ├── run_evaluation.py  ← Main evaluation runner; writes JSON results
        └── results/           ← Output directory (JSON + log files, auto-created)
```

---

## System Architecture

### Fully LLM-based multi-agent (earlier conditions)

```
Customer message
      │
      ▼  1 LLM call
┌───────────────┐
│  AgentManager │  classify intent, extract entities, produce task plan
└───────┬───────┘
        │  task_queue
        ▼  1+ LLM calls per agent
┌───────────────────────────────────────────┐
│  OrderAgent │ AccountAgent │ BillingAgent │  ← each agent uses its own LLM call(s)
└───────┬───────────────────────────────────┘
        ▼  1 LLM call
┌───────────────┐
│  LLM response │
└───────────────┘
```

### Hybrid architecture (primary evaluated system)

The hybrid design replaces the per-agent LLM calls with deterministic rule-based
dispatch.  The LLM is kept only where natural language understanding is genuinely
needed: intent planning at the start and response generation at the end.

```
Customer message
      │
      ▼  1 LLM call
┌───────────────┐
│  AgentManager │  classify intent, extract entities, produce structured task plan
└───────┬───────┘
        │  structured task plan (agent + tool + validated parameters)
        ▼  0 LLM calls per agent
┌────────────────────────────────────────────────────┐
│  OrderAgent │ AccountAgent │ BillingAgent          │
│  Rule-based: executes tool sequences from code     │
│  constants — no LLM-generated parameter values     │
└───────┬────────────────────────────────────────────┘
        ▼  1 LLM call
┌───────────────┐
│  LLM response │
└───────────────┘
Total: always 2 LLM calls per request (1.90–1.92 average due to 4 fully
       ambiguous escalation scenarios that skip the response generator)
```

**Why this matters for the hypotheses:**

- **H4 / cost**: ~1.9 calls vs 4.7–5.9 in LLM-based mode → ~63% cost reduction.
- **H4 / hallucinations**: parameter values (e.g. order status codes) come from
  code constants, not generated text → enum-mismatch hallucinations eliminated by
  construction.  The `amount=0` refund bug (where LLM guessed the refund amount
  instead of looking it up) is impossible in rule-based agents.
- **H2 / context**: the structured plan is *passed as input* to rule-based agents,
  not injected as unstructured text into another LLM prompt, so agents cannot
  non-deterministically ignore it.

**Key design decisions** (justified in Chapter 5, Design):

- **Separation of concerns**: each specialist agent has access only to the tools
  relevant to its domain; cross-domain contamination is prevented by design.
- **Shared context**: `ContextManager.get_context_summary()` builds a plain-text
  summary of everything the workflow has done so far and injects it into each
  subsequent agent's prompt, enabling context-aware decisions (H2).
- **Dynamic task expansion**: after a `list_orders` call, the orchestrator
  resolves any follow-up tasks whose `order_id` was unknown at plan time (e.g.
  "track all my shipped orders"), expanding the queue at runtime.

---

## Evaluation Design

### Scenario Set (50 scenarios)

Scenarios are defined in `evaluation/scenarios.py` and are grouped into four
complexity categories:

| Category | Count | Description |
|----------|-------|-------------|
| `simple` | 13 | Single-agent, single-tool requests |
| `multi_step` | 12 | Multi-tool requests within one agent's domain |
| `cross_system` | 13 | Requests spanning two or three agent domains |
| `error_prone` | 12 | Invalid inputs, impossible operations, deliberate escalations |

Each scenario specifies:
- The natural-language customer message (the only input to the system)
- The expected agent(s) and tool(s) that a correct solution requires
- An `expected_outcome` label (`success`, `error`, or `escalation`)

### Metrics

`evaluation/metrics.py` computes the following statistics for each experimental
condition, with Wilson score 95% confidence intervals where applicable:

| Metric | Description | Role |
|--------|-------------|------|
| `outcome_match_rate` | Fraction of runs where the system's actual outcome matches `expected_outcome` | **Primary metric** — rewards correct error handling and rejection of impossible requests, not just task completion |
| `success_rate` | Fraction of runs where all completed tasks returned `success: true` | **Secondary / legacy metric** — kept for comparability with prior work; inflated in fully LLM-based conditions by hallucinated completions |
| `error_rate` | Fraction of runs with at least one tool error | Diagnostic |
| `escalation_rate` | Fraction of runs escalated to human | H3 |
| `completion_time` | Wall-clock time per run (mean, std, median, 95% CI) | H1, H4 |
| `tool_accuracy` | Fraction of runs where all required tools were called | H1 |

> **Why is `outcome_match_rate` our most important metric?**: Raw Success Rate can produce false positives by indicating a successful completion of a task when the model has never even executed the request (S39) e.g., "Cancellation confirmed" but no call was made to `update_order_status`. Also, the raw success rate will produce false negatives where the system successfully rejects an impossible request (e.g., cancelling a delivered order) and therefore it counts the failed attempts in the raw rate. Outcome Match Rate addresses both of these issues; outcome match rate simply determines if the system did what was correct (including knowing when not to act).

### Full results across all models and conditions

| Model | Condition | Raw Success | Outcome Match | LLM Calls/run | Elapsed |
|-------|-----------|-------------|---------------|---------------|---------|
| gemma4-31b | Baseline | 68.4% | 76.8% | 2.21 | 5.90 s |
| gemma4-31b | Multi-agent no ctx | 85.6% | 86.8% | 5.66 | 13.83 s |
| gemma4-31b | Multi-agent with ctx | 84.0% | 83.2% | 5.69 | 13.44 s |
| gpt-4o-mini | Baseline | 61.2% | 70.4% | 2.18 | 3.74 s |
| gpt-4o-mini | Multi-agent no ctx | 71.6% | 74.0% | 6.34 | 8.86 s |
| gpt-4o-mini | Multi-agent with ctx | 72.8% | 73.6% | 6.18 | 8.72 s |
| gpt-5.4-mini | Baseline | 79.6% | 83.6% | 2.36 | 3.19 s |
| gpt-5.4-mini | Multi-agent no ctx | 86.4% | 86.0% | 6.00 | 8.02 s |
| gpt-5.4-mini | Multi-agent with ctx | 85.2% | 86.0% | 5.95 | 7.76 s |
| qwen2.5-7b | Baseline | 70.0% | 76.0% | 1.92 | 9.42 s |
| qwen2.5-7b | Multi-agent no ctx | 74.8% | 72.0% | 6.78 | 30.94 s |
| qwen2.5-7b | Multi-agent with ctx | 64.8% | 68.8% | 6.97 | 35.02 s |
| **hybrid gemma4-31b** | **Hybrid no ctx** | 76.0% | **96.0%** | **1.92** | 6.79 s |
| **hybrid gemma4-31b** | **Hybrid with ctx** | 78.0% | **98.0%** | **1.92** | 6.04 s |
| **hybrid gpt-4o-mini** | **Hybrid no ctx** | 74.0% | **94.0%** | **1.92** | 3.90 s |
| **hybrid gpt-4o-mini** | **Hybrid with ctx** | 76.0% | **96.0%** | **1.92** | 3.84 s |
| **hybrid gpt-5.4-mini** | **Hybrid no ctx** | 74.0% | **95.2%** | **1.90** | 4.19 s |
| **hybrid gpt-5.4-mini** | **Hybrid with ctx** | 76.8% | **98.4%** | **1.90** | 3.79 s |
| **hybrid qwen2.5-7b** | **Hybrid no ctx** | 68.0% | **84.0%** | **1.94** | 13.49 s |
| **hybrid qwen2.5-7b** | **Hybrid with ctx** | 70.0% | **86.0%** | **1.94** | 13.20 s |

**Notable findings from the model comparison** (discussed in Chapter 8):

- **GPT-4o-mini underperforms Gemma4-31B** despite costing more.  GPT-4o-mini is
  optimised for conversational quality; Gemma4-31B is a stronger instruction-follower.
  This system requires precise instruction following (e.g. the status string must be
  `"cancelled"` not `"canceled"`), not natural conversation quality — demonstrating
  that these are distinct and separable model capabilities.
- **Qwen2.5-7B with context performs *worse* than its own baseline** (64.8% raw, 68.8%
  outcome match).  The additional prompt length from the context summary likely
  exceeds the model's effective context window, a degradation not seen in larger models.
- **The hybrid architecture's lower *raw* success rate is a feature, not a bug**: it
  reflects that hallucinated completions are no longer being counted as successes.
  Under the correct primary metric (outcome match rate), hybrid is best in class.

### Running the Evaluation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Inspect the 50 scenarios
python -m evaluation.scenarios

# 3. Run the full evaluation (resets DB before each run for reproducibility)
python -m evaluation.run_evaluation
```

Results are written to `evaluation/results/evaluation_results_<timestamp>.json`
and a human-readable summary is printed to the console.

---

## Database Schema

The SQLite database (`database/ecommerce.db`) models a minimal e-commerce back-end
with six tables. It is **reset and reseeded before every evaluation run** to ensure
reproducible, order-independent results.

| Table | Purpose |
|-------|---------|
| `customers` | 10 customer profiles (name, email, address, payment method) |
| `products` | 10 product catalogue entries |
| `orders` | 30–50 synthetic orders with statuses across the full lifecycle |
| `billing_records` | One charge record per order; one duplicate charge for customer 5 (used in error-detection scenarios) |
| `support_tickets` | Three pre-seeded tickets representing real-world edge cases |

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `LLM_MODEL` | `gpt-4o-mini` | Active LLM backend (`gpt-4o-mini`, `gpt-5.4-mini`, `qwen2.5-7b_local`, `gemma4-31b`) |
| `TEMPERATURE` | `0.0` | Deterministic outputs (important for reproducibility) |
| `MAX_REASONING_STEPS` | `15` | Hard cap on graph iterations; prevents infinite loops |
| `SCENARIOS_PER_CONDITION` | `5` | Number of independent runs per scenario/condition pair |
| `API_CALL_DELAY` | `1.0 s` | Throttle between API calls to avoid rate-limit errors |

> **Note on the API key**: The key in `config.py` is a project-scoped key used
> exclusively for thesis experiments. Do not redistribute it. For your own runs,
> replace the value or set the `OPENAI_API_KEY` environment variable.

---

## Dependencies

| Package | Role |
|---------|------|
| `langchain` | LLM abstraction, message types, tool decorator |
| `langchain-openai` | `ChatOpenAI` wrapper for the OpenAI API |
| `langchain-ollama` | `ChatOllama` wrapper for local and remote Ollama models |
| `langgraph` | Stateful directed-graph workflow engine |
| `pandas` | Result post-processing and CSV export |
| `matplotlib` | Evaluation charts (Chapter 8 figures) |
| `scipy` | Statistical tests (Mann-Whitney U, t-test) |
| `pytest` | Unit test runner |
