import json
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import build_llm


CLASSIFICATION_PROMPT = """You are the manager of a customer service multi-agent system.

Read the customer message and return a JSON object with these fields:
- customer_id: integer if mentioned, otherwise null
- intent: one of order_inquiry / order_modification / account_update / billing_inquiry / refund_request / error_report / escalation
- entities: key-value pairs extracted from the message (order_id, new_email, new_address, reason, etc.)
- task_sequence: list of steps, each with "agent" and "task"
  available agents: order_agent, account_agent, billing_agent

Return ONLY valid JSON. No explanation, no markdown fences.

Example - "cancel order 1234":
{"customer_id": null, "intent": "order_modification", "entities": {"order_id": 1234}, "task_sequence": [{"agent": "order_agent", "task": "Cancel order 1234 if it has not shipped yet"}]}

Example - "I want a refund for order 5678, item was broken":
{"customer_id": null, "intent": "refund_request", "entities": {"order_id": 5678, "reason": "broken item"}, "task_sequence": [{"agent": "order_agent", "task": "Check status of order 5678"}, {"agent": "billing_agent", "task": "Process refund for order 5678"}, {"agent": "order_agent", "task": "Set order 5678 to return_initiated"}]}
"""


class AgentManager:

    def __init__(self):
        self.llm = build_llm()

    def classify_and_plan(self, msg: str) -> dict:
        msgs = [
            SystemMessage(content=CLASSIFICATION_PROMPT),
            HumanMessage(content=msg),
        ]
        request_text = "\n".join(m.content if hasattr(m, "content") else str(m) for m in msgs)

        t0 = time.perf_counter()
        resp = self.llm.invoke(msgs)
        t1 = time.perf_counter()
        response_text = resp.content if hasattr(resp, "content") else str(resp)

        self._llm_call_log = [{
            "agent": "agent_manager",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "request_text": request_text,
            "response_text": response_text,
            "has_tool_calls": False,
            "tools_called": [],
        }]

        raw = str(response_text).strip()

        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1])

        try:
            result = json.loads(raw)
            result["_llm_call_log"] = self._llm_call_log
            return result
        except json.JSONDecodeError:
            return {
                "customer_id": None,
                "intent": "escalation",
                "entities": {},
                "task_sequence": [],
                "_llm_call_log": self._llm_call_log,
            }

    def generate_response(self, state: dict, use_llm: bool = False) -> str:
        done = state.get("completed_tasks", [])
        errs = state.get("errors", [])

        lines = [f"Request: {state.get('request', 'N/A')}", ""]
        for t in done:
            tag = "OK" if t.get("success") else "FAILED"
            lines.append(f"[{tag}] {t['agent']}: {t['task_type']}")

        if errs:
            lines.append("")
            for e in errs:
                lines.append(f"[ERROR] {e['agent']}: {e['message']}")

        summary = "\n".join(lines)

        if not use_llm:
            return summary

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

        self._response_llm_call_log = [{
            "agent": "agent_manager_response",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "request_text": summary,
            "response_text": response_text,
            "has_tool_calls": False,
            "tools_called": [],
        }]

        return str(response_text)