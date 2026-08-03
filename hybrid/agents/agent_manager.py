"""
Orchestrator Agent: Intent Classification and Response Generation

The AgentManager sits at the top of the agent hierarchy and performs two distinct
roles that bookend every workflow execution:

1. Intent Classification and Planning (classify_and_plan):
   Given the raw customer message, the AgentManager uses an LLM call to parse the
   message into a structured plan: the customer's intent, extracted entities (order
   IDs, email addresses, etc.), and an ordered list of tasks for the specialist agents
   to execute.  This is the only LLM call that is always made regardless of the
   execution mode (LLM-based or hybrid).

2. Response Generation (generate_response):
   Once all tasks are complete, the AgentManager either assembles a structured
   plain-text summary from the completed-task records (fast, no API cost) or makes
   a second LLM call to produce a polished natural-language reply (used in hybrid
   mode for more human-readable output).

This separation of concerns (plan first, act second, summarise third) is a key
architectural decision discussed in Chapter 5 (Design).  It prevents the system
from interleaving reasoning with action in an unconstrained way, which would make
the control flow difficult to monitor and evaluate.

The CLASSIFICATION_PROMPT is the most critical piece of the system: it defines
the exact JSON schema the LLM must produce and provides two worked examples.  A
well-structured prompt is essential for reliable intent classification; any
ambiguity here propagates to every downstream agent.

"""

import json
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import build_llm

# This prompt instructs the LLM to act as a structured parser rather than a
# conversational assistant.  The strict JSON-only output requirement is
# intentional: it makes the LLM's output machine-readable and eliminates the
# need for fragile regex-based parsing of free text.
#
# The prompt lists every available agent, action, and required parameter
# explicitly so the LLM can generate fully-formed task plans without guessing.
# Providing two concrete examples has been shown to
# significantly improve structured-output reliability.

CLASSIFICATION_PROMPT = """You are the manager of a customer service multi-agent system.

Read the customer message and return a JSON object with these fields:
- customer_id: integer if mentioned, otherwise null
- intent: one of order_inquiry / order_modification / account_update / billing_inquiry / refund_request / error_report / escalation
- entities: key-value pairs extracted from the message (order_id, new_email, new_address, reason, etc.)
- task_sequence: list of steps, each with "agent", "task", "action", and "params"
  available agents: order_agent, account_agent, billing_agent
  available actions and their required params:
    order_agent:
      get_order_status  -> params: {"order_id": <int>}
      get_tracking      -> params: {"order_id": <int>}
      cancel_order      -> params: {"order_id": <int>}
      initiate_return   -> params: {"order_id": <int>}
      list_orders       -> params: {"customer_id": <int>}
    account_agent:
      get_profile    -> params: {"customer_id": <int>}
      update_email   -> params: {"customer_id": <int>, "new_email": "<string>"}
      update_address -> params: {"customer_id": <int>, "new_address": "<string>"}
      update_payment -> params: {"customer_id": <int>, "new_method": "<string>"}
    billing_agent:
      get_payment_status -> params: {"order_id": <int>}
      process_refund     -> params: {"order_id": <int>, "reason": "<string>"}
      get_billing_history -> params: {"customer_id": <int>}
      check_duplicates   -> params: {"customer_id": <int>}

Return ONLY valid JSON. No explanation, no markdown fences.

Example - "cancel order 1234":
{"customer_id": null, "intent": "order_modification", "entities": {"order_id": 1234}, "task_sequence": [{"agent": "order_agent", "task": "Cancel order 1234 if it has not shipped yet", "action": "cancel_order", "params": {"order_id": 1234}}]}

Example - "I want a refund for order 5678, item was broken":
{"customer_id": null, "intent": "refund_request", "entities": {"order_id": 5678, "reason": "broken item"}, "task_sequence": [{"agent": "order_agent", "task": "Set order 5678 to return_initiated", "action": "initiate_return", "params": {"order_id": 5678}}, {"agent": "billing_agent", "task": "Process refund for order 5678", "action": "process_refund", "params": {"order_id": 5678, "reason": "broken item"}}]}
"""


class AgentManager:
    """Orchestrator responsible for intent classification and final response generation.

    This is not a specialist agent in the domain sense, it does not call order,
    account, or billing tools directly.  Its role is purely coordinative: it
    translates the customer's natural-language request into a machine-executable
    plan, and then translates the plan's results back into a customer-facing reply.
    """

    def __init__(self):
        # Plain LLM without tools, the classification call uses structured JSON
        # output rather than tool calls, so no tool binding is needed here.
        self.llm = build_llm()

    def classify_and_plan(self, msg: str) -> dict:
        """Parse a customer message into an intent, entities, and a task plan.

        This method makes exactly one LLM API call.  The response is parsed as
        JSON; if parsing fails (which can happen when the LLM wraps the output
        in markdown code fences despite explicit instructions), the method falls
        back to an "escalation" intent so the workflow can degrade gracefully.

        Parameters
        ----------
        msg : str
            The raw customer message.

        Returns
        -------
        dict
            A dictionary with keys:
            - customer_id: int or None
            - intent: intent label string
            - entities: dict of extracted key-value pairs
            - task_sequence: list of {agent, task, action, params} dicts
            - _llm_call_log: internal logging metadata (stripped by the graph node)
        """
        msgs = [
            SystemMessage(content=CLASSIFICATION_PROMPT),
            HumanMessage(content=msg),
        ]
        request_text = "\n".join(m.content if hasattr(m, "content") else str(m) for m in msgs)

        t0 = time.perf_counter()
        resp = self.llm.invoke(msgs)
        t1 = time.perf_counter()
        response_text = resp.content if hasattr(resp, "content") else str(resp)

        # Log the API call for later latency analysis
        self._llm_call_log = [{
            "agent": "agent_manager",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "request_text": request_text,
            "response_text": response_text,
            "has_tool_calls": False,
            "tools_called": [],
        }]

        raw = str(response_text).strip()

        # Strip markdown code fences if the LLM added them despite instructions.
        # This defensive step handles a known LLM quirk when the model has been
        # fine-tuned to always format code output.
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1])

        try:
            result = json.loads(raw)
            result["_llm_call_log"] = self._llm_call_log
            return result
        except json.JSONDecodeError:
            # Graceful fallback: escalate to a human agent rather than crashing.
            # The orchestrator will see intent="escalation" and route accordingly.
            return {
                "customer_id": None,
                "intent": "escalation",
                "entities": {},
                "task_sequence": [],
                "_llm_call_log": self._llm_call_log,
            }

    def generate_response(self, state: dict, use_llm: bool = False) -> str:
        """Produce the final customer-facing response from the completed workflow state.

        Two modes are available:

        - Structured summary (use_llm=False, default for non-hybrid runs):
          Builds a plain-text report directly from the completed-task records.
          Fast, no API cost, fully deterministic — useful for evaluation where
          response style is less important than correctness.

        - LLM-polished response (use_llm=True, used in hybrid mode):
          Passes the structured summary to a second LLM call that rewrites it as
          a friendly customer-service reply.  Adds one API call but produces more
          natural output for demonstration purposes.

        Parameters
        ----------
        state : dict
            The completed WorkflowState dictionary.
        use_llm : bool
            Whether to make a polishing LLM call (True) or return the raw summary.

        Returns
        -------
        str
            The response string to be delivered to the customer.
        """
        done = state.get("completed_tasks", [])
        errs = state.get("errors", [])

        # Build a structured summary of what was accomplished
        summary_lines = [f"Customer request: {state.get('request', 'N/A')}", ""]
        for t in done:
            tag = "OK" if t.get("success") else "FAILED"
            summary_lines.append(f"[{tag}] {t['agent']}: {t['task_type']}")
            if t.get("result"):
                r = str(t["result"])
                # Truncate very long results to keep the summary readable
                summary_lines.append(f"  Result: {r[:300]}{'...' if len(r) > 300 else ''}")

        if errs:
            summary_lines.append("")
            for e in errs:
                summary_lines.append(f"[ERROR] {e['agent']}: {e['message']}")

        summary = "\n".join(summary_lines)

        self._response_llm_call_log = []  # Reset log for this call

        if not use_llm:
            # Return the raw structured summary without an additional API call
            return summary

        # Optional LLM polishing step
        msgs = [
            SystemMessage(content=(
                "You are a customer service assistant. Based on the completed task results below, "
                "write a clear, concise, and friendly response to the customer. "
                "Summarise what was done and include any relevant details (status, amounts, addresses)."
            )),
            HumanMessage(content=summary),
        ]

        t0 = time.perf_counter()
        resp = self.llm.invoke(msgs)
        t1 = time.perf_counter()

        raw_content = resp.content if hasattr(resp, "content") else ""
        response_text = raw_content if isinstance(raw_content, str) else str(raw_content)

        # Record the latency of this polishing call separately
        self._response_llm_call_log = [{
            "agent": "agent_manager_response",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "request_text": summary,
            "response_text": response_text,
            "has_tool_calls": False,
            "tools_called": [],
        }]

        return str(response_text)