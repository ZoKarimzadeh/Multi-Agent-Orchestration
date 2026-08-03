import time
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_REASONING_STEPS
from agents.agent_manager import AgentManager
from agents.order_agent import OrderAgent
from agents.account_agent import AccountAgent
from agents.billing_agent import BillingAgent
from context.context_manager import ContextManager


class WorkflowState(TypedDict):
    customer_id: Optional[int]
    request: str
    intent: Optional[str]
    entities: dict
    task_queue: list[dict]
    current_task: Optional[str]
    completed_tasks: list[dict]
    errors: list[dict]
    escalated: bool
    step_count: int
    messages: Annotated[list[BaseMessage], add_messages]
    final_response: Optional[str]
    context_disabled: bool
    start_time: Optional[float]
    llm_call_log: list


_agent_manager = AgentManager()
_order_agent = OrderAgent()
_account_agent = AccountAgent()
_billing_agent = BillingAgent()

AGENT_MAP = {
    "order_agent": _order_agent,
    "account_agent": _account_agent,
    "billing_agent": _billing_agent,
}


def classify_intent(state: WorkflowState) -> dict:
    plan = _agent_manager.classify_and_plan(state["request"])
    llm_log = plan.pop("_llm_call_log", [])
    combined = list(state.get("llm_call_log", [])) + llm_log
    return {
        "customer_id": plan.get("customer_id") or state.get("customer_id"),
        "intent": plan.get("intent"),
        "entities": plan.get("entities", {}),
        "task_queue": plan.get("task_sequence", []),
        "step_count": state.get("step_count", 0) + 1,
        "start_time": state.get("start_time") or time.time(),
        "llm_call_log": combined,
    }


def execute_agent(state: WorkflowState) -> dict:
    queue = list(state.get("task_queue", []))
    if not queue:
        return {"task_queue": []}

    task_info = queue.pop(0)
    agent_name = task_info.get("agent", "")
    task_desc = task_info.get("task", "")

    agent = AGENT_MAP.get(agent_name)
    if agent is None:
        errs = list(state.get("errors", []))
        errs.append({
            "agent": agent_name,
            "error_type": "agent_not_found",
            "message": f"No agent registered with name '{agent_name}'.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recoverable": False,
        })
        return {
            "task_queue": queue,
            "errors": errs,
            "step_count": state.get("step_count", 0) + 1,
            "llm_call_log": list(state.get("llm_call_log", [])),
        }

    agent_state = dict(state)
    agent_state["current_task"] = task_desc

    if state.get("context_disabled", False):
        agent_state["completed_tasks"] = []

    result = agent.invoke(agent_state)

    done = list(state.get("completed_tasks", []))
    done.append({
        "agent": result.get("agent", agent_name),
        "task_type": task_desc,
        "tool_calls": result.get("tool_calls", []),
        "result": result.get("response", ""),
        "success": result.get("success", False),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    errs = list(state.get("errors", []))
    if not result.get("success", False):
        errs.append({
            "agent": agent_name,
            "error_type": "task_failure",
            "message": result.get("response", "task failed"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recoverable": True,
        })

    combined_llm_log = list(state.get("llm_call_log", [])) + list(result.get("llm_call_log", []))

    return {
        "task_queue": queue,
        "completed_tasks": done,
        "errors": errs,
        "step_count": state.get("step_count", 0) + 1,
        "llm_call_log": combined_llm_log,
    }


def generate_response(state: WorkflowState) -> dict:
    response = _agent_manager.generate_response(dict(state), use_llm=True)
    combined_log = list(state.get("llm_call_log", []))
    if hasattr(_agent_manager, "_response_llm_call_log"):
        combined_log = combined_log + _agent_manager._response_llm_call_log
    return {"final_response": response, "llm_call_log": combined_log}


def escalate(state: WorkflowState) -> dict:
    summary = ContextManager.get_context_summary(dict(state))
    msg = (
        "I was unable to fully resolve your request automatically and have flagged it "
        "for a human agent. Here is what I found so far:\n" + summary
    )
    return {"escalated": True, "final_response": msg}


def check_intent(state: WorkflowState) -> str:
    if state.get("intent") == "escalation" or not state.get("task_queue"):
        return "escalate"
    return "execute_agent"


def should_continue(state: WorkflowState) -> str:
    if state.get("step_count", 0) >= MAX_REASONING_STEPS:
        return "escalate"

    for e in state.get("errors", []):
        if not e.get("recoverable", False):
            return "escalate"

    if state.get("task_queue"):
        return "execute_agent"

    return "generate_response"


def build_workflow():
    g = StateGraph(WorkflowState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("execute_agent", execute_agent)
    g.add_node("generate_response", generate_response)
    g.add_node("escalate", escalate)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        check_intent,
        {"execute_agent": "execute_agent", "escalate": "escalate"},
    )
    g.add_conditional_edges(
        "execute_agent",
        should_continue,
        {
            "execute_agent": "execute_agent",
            "generate_response": "generate_response",
            "escalate": "escalate",
        },
    )
    g.add_edge("generate_response", END)
    g.add_edge("escalate", END)

    return g.compile()


def run_workflow(customer_message: str, context_disabled: bool = False) -> dict:
    init = ContextManager.create_initial_state(customer_message, context_disabled=context_disabled)
    init["current_task"] = None
    init["start_time"] = time.time()

    app = build_workflow()
    result = app.invoke(init)
    result["end_time"] = time.time()
    result["elapsed_time"] = result["end_time"] - result.get("start_time", result["end_time"])
    return result