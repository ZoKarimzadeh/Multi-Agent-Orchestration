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

BASELINE_PROMPT = """\
You are a customer service agent for an e-commerce company.
Handle all customer requests: orders, accounts, billing, refunds.
Use the tools to look up and modify data as needed.
If you cannot complete something, say why clearly.
For multi-part requests, handle each part in order.
"""

_ESCALATION_PHRASES = [
    "unable to", "cannot help", "not possible", "human agent",
    "escalat", "beyond my capabilities", "contact support",
    "speak to a representative",
]


class BaselineAgent:

    def __init__(self):
        self.llm = build_llm(bind_tools=ALL_TOOLS)

    def run(self, customer_message: str) -> dict:
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

            if not resp.tool_calls:
                break

            for tc in resp.tool_calls:
                res = self._call_tool(tc["name"], tc["args"])
                tool_call_log.append({"tool": tc["name"], "args": tc["args"], "result": res})

                if isinstance(res, dict) and res.get("status") == "error":
                    errors.append({
                        "agent": "baseline_agent",
                        "error_type": "tool_error",
                        "message": res.get("message", "unknown error"),
                    })

                msgs.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
        else:
            escalated = True

        t1 = time.time()

        final_content = msgs[-1].content if msgs else ""
        final = str(final_content) if not isinstance(final_content, str) else final_content

        if any(p in final.lower() for p in _ESCALATION_PHRASES):
            escalated = True

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
        for t in ALL_TOOLS:
            if t.name == name:
                try:
                    return t.invoke(args)
                except Exception as exc:
                    return {"status": "error", "message": str(exc)}
        return {"status": "error", "message": f"unknown tool: {name}"}