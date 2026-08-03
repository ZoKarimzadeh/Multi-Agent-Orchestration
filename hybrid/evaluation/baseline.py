"""
Single-Agent Monolith (H1 Comparison Baseline)

This module implements the "baseline condition" used to evaluate Hypothesis 1
(H1): "A coordinated multi-agent system completes customer-service tasks faster
and with a higher success rate than independent, unspecialised agents."

The BaselineAgent is a single LLM that
receives all 13 tools at once with no domain specialisation, no shared
context, and no orchestration layer.  It mirrors the "standard" single-agent
approach described in prior work (Chapter 3, Related Work) and serves as the
H1 control condition.

H1 result summary
-----------------
H1 is supported for large models.  On the corrected primary metric (outcome
match rate), multi-agent conditions outperform the baseline by 3.6–4.4 percentage
points for Gemma4-31B and GPT-5.4-mini.  The exception is Qwen2.5-7B, where
the orchestration overhead increases latency without a corresponding accuracy
gain, suggesting a model-capability threshold below which multi-agent
coordination does not help.

Model comparison finding
------------------------
During evaluation it was observed that GPT-4o-mini performed worse than the
free Gemma4-31B model despite costing more.  Log analysis revealed that
GPT-4o-mini, optimised for natural conversation, frequently produced enum
value mismatches (e.g. "canceled" instead of "cancelled").  Gemma4-31B,
being a stronger instruction follower, reproduced the exact required string
constants more reliably.  This demonstrates that conversational quality and
instruction-following are distinct model capabilities, and that action-
oriented systems require the latter.

Design decisions
-----------------
- All tool descriptions are left empty so the LLM must infer usage from the
  function signature alone, just as a naive deployment would be configured.
- The system prompt is intentionally minimal (5 lines) — it does not provide
  any domain knowledge, priority rules, or escalation guidance.
- The success flag is deliberately conservative: any tool error or
  escalation phrase causes the run to be marked unsuccessful, matching the
  stricter success criterion applied to the multi-agent conditions.

Escalation detection uses a keyword heuristic (_ESCALATION_PHRASES).  This
is admittedly imperfect but mirrors the practical signal available in a
production system without human review.  The same heuristic is applied to the
multi-agent conditions, ensuring fair comparison.

"""
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_REASONING_STEPS, build_llm

from tools.db_tools import (
    check_duplicate_charges as _check_duplicate_charges,
    get_billing_history as _get_billing_history,
    get_customer_orders as _get_customer_orders,
    get_customer_profile as _get_customer_profile,
    get_order_details as _get_order_details,
    get_payment_status as _get_payment_status,
    get_tracking_info as _get_tracking_info,
    process_refund as _process_refund,
    update_customer_email as _update_customer_email,
    update_order_status as _update_order_status,
    update_payment_method as _update_payment_method,
    update_shipping_address as _update_shipping_address,
)


@tool(description="")
def get_customer_profile(customer_id: int) -> dict:
    return _get_customer_profile(customer_id)

@tool(description="")
def update_customer_email(customer_id: int, new_email: str) -> dict:
    return _update_customer_email(customer_id, new_email)

@tool(description="")
def update_shipping_address(customer_id: int, new_address: str) -> dict:
    return _update_shipping_address(customer_id, new_address)

@tool(description="")
def update_payment_method(customer_id: int, new_method: str) -> dict:
    return _update_payment_method(customer_id, new_method)

@tool(description="")
def get_order_details(order_id: int) -> dict:
    return _get_order_details(order_id)

@tool(description="")
def get_customer_orders(customer_id: int) -> dict:
    return _get_customer_orders(customer_id)

@tool(description="")
def update_order_status(order_id: int, new_status: str) -> dict:
    return _update_order_status(order_id, new_status)

@tool(description="")
def get_tracking_info(order_id: int) -> dict:
    return _get_tracking_info(order_id)

@tool(description="")
def get_payment_status(order_id: int) -> dict:
    return _get_payment_status(order_id)

@tool(description="")
def process_refund(order_id: int, amount: float, reason: str) -> dict:
    return _process_refund(order_id, amount, reason)

@tool(description="")
def get_billing_history(customer_id: int) -> dict:
    return _get_billing_history(customer_id)

@tool(description="")
def check_duplicate_charges(customer_id: int) -> dict:
    return _check_duplicate_charges(customer_id)

ALL_TOOLS = [
    get_customer_profile, update_customer_email, update_shipping_address,
    update_payment_method, get_order_details, get_customer_orders,
    update_order_status, get_tracking_info, get_payment_status,
    process_refund, get_billing_history, check_duplicate_charges,
]
"""All 12 domain tools bound to the single baseline LLM.

The multi-agent system distributes these across three specialists; the baseline
receives them all simultaneously.  This creates a larger tool-selection space
for the LLM, which is one of the hypothesised reasons for its lower performance.
"""

BASELINE_PROMPT = """\
You are a customer service agent for an e-commerce company.
Handle all customer requests: orders, accounts, billing, refunds.
Use the tools to look up and modify data as needed.
If you cannot complete something, say why clearly.
For multi-part requests, handle each part in order.
"""
"""Minimal system prompt — intentionally non-specialist.

The multi-agent system uses three domain-specific prompts (order, account,
billing), each with detailed tool guidance and business rules.  This single
generic prompt is the H1 control: it provides no domain knowledge, no
escalation rules, and no tool-selection hints.
"""

# Keyword heuristic for detecting when the agent has given up and is asking
# for human intervention.  Any of these phrases in the final LLM response
# causes the run to be marked as escalated = True.
_ESCALATION_PHRASES = [
    "unable to", "cannot help", "not possible", "human agent",
    "escalat", "beyond my capabilities", "contact support",
    "speak to a representative",
]


class BaselineAgent:
    """Single-agent monolith for the H1 comparison baseline.

    One instance is created per evaluation run and reused across all scenarios
    in that run.  The LLM is bound once at construction time; no re-binding
    occurs between scenarios (same as the specialist agents, for fair latency
    comparison).
    """

    def __init__(self):
        self.llm = build_llm(bind_tools=ALL_TOOLS)

    def run(self, customer_message: str) -> dict:
        """Execute a single customer message and return a standardised result dict.

        The ReAct loop runs for at most MAX_REASONING_STEPS iterations.
        Each iteration:
          1. Calls the LLM with the current message history.
          2. If the LLM returns tool calls, executes each tool and appends the
             results as ToolMessages.
          3. If the LLM returns a plain text response (no tool calls), the loop
             ends — the agent has finished reasoning.

        If the loop exhausts MAX_REASONING_STEPS without finishing, the run
        is marked as escalated (the agent could not resolve the request within
        the allowed reasoning budget).

        The success flag is True only when:
          - At least one tool was called (the agent took a concrete action), AND
          - No tool returned an error status, AND
          - The final response contains no escalation phrases.

        Returns a dict with the same top-level keys as run_workflow() in
        orchestration/graph.py, so that metrics.normalize_result() can
        process both conditions with the same code path.
        """
        t0 = time.time()

        msgs = [
            SystemMessage(content=BASELINE_PROMPT),
            HumanMessage(content=customer_message),
        ]

        llm_call_log = []
        tool_call_log = []
        errors = []
        escalated = False

        for _step in range(MAX_REASONING_STEPS):
            # Build a single string snapshot of the conversation for logging —
            # used only for debugging and offline analysis; not fed back to LLM.
            request_text = "\n".join(
                m.content if isinstance(m.content, str) else str(m.content)
                for m in msgs
            )

            t_call_start = time.perf_counter()
            resp = self.llm.invoke(msgs)
            t_call_end = time.perf_counter()
            msgs.append(resp)

            llm_call_log.append({
                "agent": "baseline_agent",
                "latency_ms": round((t_call_end - t_call_start) * 1000, 2),
                "request_text": request_text,
                "response_text": resp.content if hasattr(resp, "content") else str(resp),
                "has_tool_calls": bool(resp.tool_calls),
                "tools_called": [tc["name"] for tc in resp.tool_calls] if resp.tool_calls else [],
            })

            # No tool calls to the LLM has produced its final answer; exit loop.
            if not resp.tool_calls:
                break

            for tc in resp.tool_calls:
                res = self._call_tool(tc["name"], tc["args"])
                tool_call_log.append({"tool": tc["name"], "args": tc["args"], "result": res})

                # Any tool error is recorded; it will cause success = False.
                if isinstance(res, dict) and res.get("status") == "error":
                    errors.append({
                        "agent": "baseline_agent",
                        "error_type": "tool_error",
                        "message": res.get("message", "unknown error"),
                    })

                msgs.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
        else:
            # The for-else branch runs only when the loop exhausted all steps
            # without a break, the agent never stopped calling tools.
            escalated = True

        t1 = time.time()

        final_content = msgs[-1].content if msgs else ""
        final = str(final_content) if not isinstance(final_content, str) else final_content

        # Secondary escalation check: scan the text of the final response for
        # any phrase that indicates the agent gave up or deferred to a human.
        if any(p in final.lower() for p in _ESCALATION_PHRASES):
            escalated = True

        # Success requires: action taken + no errors + no escalation.
        # This mirrors the success definition used for the multi-agent conditions
        success = bool(tool_call_log) and not errors and not escalated

        conversation_log = []
        for m in msgs:
            content = getattr(m, "content", "")
            entry = {
                "role": m.type if hasattr(m, "type") else str(type(m).__name__),
                "content": str(content) if not isinstance(content, str) else content,
            }
            if hasattr(m, "tool_call_id"):
                entry["tool_call_id"] = str(m.tool_call_id)
            conversation_log.append(entry)

        return {
            "customer_id": None,
            "request": customer_message,
            "intent": None,
            "entities": {},
            "completed_tasks": [{
                "agent": "baseline_agent",
                "task_type": "full_request",
                "tool_calls": tool_call_log,
                "result": final,
                "success": success,
            }],
            "errors": errors,
            "escalated": escalated,
            "final_response": final,
            "start_time": t0,
            "end_time": t1,
            "elapsed_time": t1 - t0,
            "step_count": len(tool_call_log),
            "llm_call_log": llm_call_log,
            "conversation_log": conversation_log,
        }

    def _call_tool(self, name: str, args: dict) -> Any:
        """Dispatch a tool call by name, catching any runtime exception.

        Wraps exceptions in an error dict so the ReAct loop can continue and
        record the failure without crashing the entire run.  This matches the
        error-handling behaviour of the specialist agents in base_agent.py.
        """
        for t in ALL_TOOLS:
            if t.name == name:
                try:
                    return t.invoke(args)
                except Exception as exc:
                    return {"status": "error", "message": str(exc)}
        return {"status": "error", "message": f"unknown tool: {name}"}