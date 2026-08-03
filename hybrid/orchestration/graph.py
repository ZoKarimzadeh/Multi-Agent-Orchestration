"""
LangGraph Workflow: Nodes, Edges, and Conditional Routing

This module is the heart of the multi-agent orchestration system.  It implements
the directed state machine that coordinates all agents, handles errors, enforces
the step limit, and routes control flow based on the runtime state of the workflow.

The orchestration layer is implemented using LangGraph, a framework
that represents workflows as typed state graphs.  Each node in the
graph is a Python function that receives the current WorkflowState and returns a
partial update dictionary.  LangGraph merges the update into the shared state before
invoking the next node.  This design decouples the individual agents from the
routing logic: agents only need to produce a result dict; they do not need to know
which node comes next.

Conditional edge functions (check_intent, should_continue) determine which
branch to take at each decision point.  These implement the safety guards required
for a production-grade system:
- If the AgentManager cannot parse a plan, escalate immediately.
- If the step count exceeds MAX_REASONING_STEPS, escalate (prevents infinite loops).
- If an unrecoverable error is encountered, escalate.

The dynamic task expansion logic in execute_agent addresses a fundamental
limitation of LLM-based planning: the planner cannot know entity values that are
only discovered at execution time (e.g. the order_id of a customer's shipped order).
The expansion code fills in these unknowns after a list_orders call.

"""

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


# ---------------------------------------------------------------------------
# Workflow state schema
# ---------------------------------------------------------------------------
# WorkflowState is a TypedDict that defines every field that can appear in the
# shared state dictionary.  LangGraph uses this type to validate state updates
# and to generate documentation.  The Annotated field for messages uses
# LangGraph's built-in add_messages reducer, which appends new messages to
# the list rather than replacing the entire list on each update.
# ---------------------------------------------------------------------------
class WorkflowState(TypedDict):
    customer_id: Optional[int]              # Extracted by AgentManager; None until classification
    request: str                            # Original customer message (immutable throughout workflow)
    intent: Optional[str]                  # Classified intent label (e.g. "refund_request")
    entities: dict                          # Key-value pairs extracted from the message
    task_queue: list[dict]                  # Remaining tasks to execute
    current_task: Optional[str]            # Description of the task currently being executed
    completed_tasks: list[dict]             # History of all executed tasks with results
    errors: list[dict]                      # Structured error records from failed steps
    escalated: bool                         # True if workflow was handed off to human agent
    step_count: int                         # Number of graph node executions so far
    messages: Annotated[list[BaseMessage], add_messages]  # LangChain message history
    final_response: Optional[str]          # Customer-facing response (populated at the end)
    context_disabled: bool                 # H2 control condition: True = no shared context
    hybrid: bool                            # True = rule-based tool dispatch (no per-agent LLM call)
    start_time: Optional[float]            # Unix timestamp when workflow began
    llm_call_log: list                      # Accumulated log of all LLM API calls in this run


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
# Agents are instantiated once at import time rather than per-request.
# This avoids the overhead of re-creating LangChain objects on every call,
# which is particularly important during evaluation runs with many scenarios.
# ---------------------------------------------------------------------------
_agent_manager = AgentManager()
_order_agent = OrderAgent()
_account_agent = AccountAgent()
_billing_agent = BillingAgent()

# Named lookup table used to dispatch tasks to the correct agent instance
AGENT_MAP = {
    "order_agent": _order_agent,
    "account_agent": _account_agent,
    "billing_agent": _billing_agent,
}


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------
# Each function below is a LangGraph node.  The convention is:
# - Accept the full WorkflowState as input.
# - Return only the fields that changed (partial update dict).
# - Never mutate the input state directly (LangGraph handles merging).
# ---------------------------------------------------------------------------

def classify_intent(state: WorkflowState) -> dict:
    """Node 1: Parse the customer message and produce an executable task plan.

    Calls AgentManager.classify_and_plan which makes one LLM API call.
    The result is an intent label, extracted entities, and an ordered list of
    agent tasks (the task queue).  The LLM call log from this step is merged
    into the accumulated log for the entire workflow run.
    """
    plan = _agent_manager.classify_and_plan(state["request"])
    # The internal log entry is extracted from the plan dict before merging
    llm_log = plan.pop("_llm_call_log", [])
    combined = list(state.get("llm_call_log", [])) + llm_log
    return {
        # Preserve any customer_id already present in state (e.g. set by the caller)
        "customer_id": plan.get("customer_id") or state.get("customer_id"),
        "intent": plan.get("intent"),
        "entities": plan.get("entities", {}),
        "task_queue": plan.get("task_sequence", []),
        "step_count": state.get("step_count", 0) + 1,
        # Record the start time on the first step (subsequent steps do not update it)
        "start_time": state.get("start_time") or time.time(),
        "llm_call_log": combined,
    }


def execute_agent(state: WorkflowState) -> dict:
    """Node 2: Pop the next task from the queue and dispatch it to the named agent.

    This node is called repeatedly in a loop until the task queue is empty,
    the step limit is reached, or an unrecoverable error occurs.

    Two special behaviours are implemented here beyond simple task dispatch:

    1. Context clearing for H2 control condition: When context_disabled
       is True, the completed-task history is hidden from each agent invocation
       so the agent cannot benefit from prior results.  This simulates the
       independent-agent baseline used to test Hypothesis H2.

    2. Dynamic task expansion after list_orders: The AgentManager
       sometimes cannot supply a concrete order_id at plan time because it is only
       known after listing the customer's orders.  After a successful list_orders
       call, this node rewrites any follow-up tasks that have order_id=0, substituting
       the actual order IDs discovered at runtime.  For example, "track all shipped
       orders" expands into one tracking task per shipped order.
    """
    queue = list(state.get("task_queue", []))
    if not queue:
        return {"task_queue": []}

    # Dequeue the next task (FIFO — plan order is respected)
    task_info = queue.pop(0)
    agent_name = task_info.get("agent", "")
    task_desc = task_info.get("task", "")

    agent = AGENT_MAP.get(agent_name)
    if agent is None:
        # The planner named an agent that doesn't exist — this is unrecoverable
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

    # Build the agent-level state, adding the current task description
    agent_state = dict(state)
    agent_state["current_task"] = task_desc

    # H2 control condition: clear prior results so the agent is context-blind
    if state.get("context_disabled", False):
        agent_state["completed_tasks"] = []

    # Dispatch: hybrid = rule-based (no LLM call), otherwise full ReAct loop
    if state.get("hybrid", False):
        result = agent.invoke_rule_based(task_info, agent_state)
    else:
        result = agent.invoke(agent_state)

    # -----------------------------------------------------------------------
    # Dynamic task expansion (hybrid mode only)
    # -----------------------------------------------------------------------
    # After list_orders succeeds, resolve follow-up tasks whose order_id=0.
    # The value 0 is a sentinel inserted by the planner when it knows an
    # order_id will be needed but cannot determine it until runtime.
    # -----------------------------------------------------------------------
    if (
        state.get("hybrid", False)
        and task_info.get("action") == "list_orders"
        and result.get("success")
    ):
        # Extract the raw order list from the tool_calls result dict
        orders = []
        for tc in result.get("tool_calls", []):
            if tc.get("tool") == "get_customer_orders":
                orders = tc.get("result", {}).get("data", [])
                break

        if orders:
            expanded_queue = []
            for queued_task in queue:
                queued_action = queued_task.get("action", "")
                queued_params = dict(queued_task.get("params", {}))

                # Detect placeholder / missing order_id
                raw_oid = queued_params.get("order_id")
                try:
                    oid_val = int(raw_oid or 0)
                except (ValueError, TypeError):
                    oid_val = 0  # string placeholder like "TBD_FROM_LIST_ORDERS"

                if oid_val == 0:
                    if queued_action == "get_tracking":
                        # Expand: one get_tracking task per shipped order
                        # If no orders are shipped, the task is silently dropped
                        # (there is nothing to track).
                        shipped = [o for o in orders if o.get("status") == "shipped"]
                        for o in shipped:
                            new_task = dict(queued_task)
                            new_task["params"] = {**queued_params, "order_id": o["order_id"]}
                            new_task["task"] = f"Get tracking info for order {o['order_id']}"
                            expanded_queue.append(new_task)
                    elif queued_action in ("get_order_status", "get_order_details"):
                        # Expand: target the most recent order.
                        # get_customer_orders returns results ORDER BY placed_at DESC,
                        # so the first element is always the most recent.
                        most_recent = orders[0]
                        new_task = dict(queued_task)
                        new_task["params"] = {**queued_params, "order_id": most_recent["order_id"]}
                        new_task["task"] = f"Get details for order {most_recent['order_id']}"
                        expanded_queue.append(new_task)
                    else:
                        expanded_queue.append(queued_task)
                else:
                    expanded_queue.append(queued_task)
            queue = expanded_queue

    # Record the completed task (regardless of success/failure)
    done = list(state.get("completed_tasks", []))
    done.append({
        "agent": result.get("agent", agent_name),
        "task_type": task_desc,
        "tool_calls": result.get("tool_calls", []),
        "result": result.get("response", ""),
        "success": result.get("success", False),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    # Record a recoverable error if the task failed
    errs = list(state.get("errors", []))
    if not result.get("success", False):
        errs.append({
            "agent": agent_name,
            "error_type": "task_failure",
            "message": result.get("response", "task failed"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recoverable": True,  # Workflow continues; only unrecoverable errors trigger escalation
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
    """Node 3: Produce the final customer-facing response from completed task results.

    In hybrid mode (hybrid=True), an additional LLM call polishes the
    structured summary into a natural-language reply.  In non-hybrid mode,
    the structured summary is returned directly to avoid an extra API cost.
    """
    use_llm = state.get("hybrid", False)
    response = _agent_manager.generate_response(dict(state), use_llm=use_llm)
    combined_log = list(state.get("llm_call_log", []))
    if use_llm:
        # Include the latency record for the response-polishing LLM call
        combined_log = combined_log + getattr(_agent_manager, "_response_llm_call_log", [])
    return {"final_response": response, "llm_call_log": combined_log}


def escalate(state: WorkflowState) -> dict:
    """Node 4: Flag the request for human intervention and return a summary.

    Escalation is triggered by:
    - The AgentManager producing an "escalation" intent (ambiguous request).
    - The step counter reaching MAX_REASONING_STEPS (safeguard against loops).
    - An unrecoverable error in the error list.

    The context summary is included in the response so the human agent who
    picks up the ticket has full visibility of what the system already tried.
    """
    summary = ContextManager.get_context_summary(dict(state))
    msg = (
        "I was unable to fully resolve your request automatically and have flagged it "
        "for a human agent. Here is what I found so far:\n" + summary
    )
    return {"escalated": True, "final_response": msg}


# ---------------------------------------------------------------------------
# Conditional edge functions (routing logic)
# ---------------------------------------------------------------------------

def check_intent(state: WorkflowState) -> str:
    """Routing function after classify_intent.

    Returns "escalate" if the intent is escalation or the task queue is
    empty (nothing to do); otherwise routes to "execute_agent".
    """
    if state.get("intent") == "escalation" or not state.get("task_queue"):
        return "escalate"
    return "execute_agent"


def should_continue(state: WorkflowState) -> str:
    """Routing function after execute_agent (controls the main execution loop).

    Checks three conditions in priority order:
    1. Step limit reached : escalate (prevents runaway loops).
    2. Unrecoverable error present : escalate.
    3. More tasks in queue : loop back to execute_agent.
    4. Queue empty, no blocking errors : generate final response.
    """
    # Guard: hard step limit to prevent infinite execution
    if state.get("step_count", 0) >= MAX_REASONING_STEPS:
        return "escalate"

    # Guard: unrecoverable errors (e.g. agent not found) require human intervention
    for e in state.get("errors", []):
        if not e.get("recoverable", False):
            return "escalate"

    # Continue processing if there are tasks remaining
    if state.get("task_queue"):
        return "execute_agent"

    # All tasks are done — produce the final response
    return "generate_response"


# ---------------------------------------------------------------------------
# Workflow construction and entry point
# ---------------------------------------------------------------------------

def build_workflow():
    """Construct and compile the LangGraph state machine.

    Nodes and edges are registered here.  Conditional edges are registered
    with explicit branch dictionaries so LangGraph can validate that all
    possible return values of the routing function map to known nodes.

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph application ready to accept invoke calls.
    """
    g = StateGraph(WorkflowState)

    # Register nodes (each is a Python function defined above)
    g.add_node("classify_intent", classify_intent)
    g.add_node("execute_agent", execute_agent)
    g.add_node("generate_response", generate_response)
    g.add_node("escalate", escalate)

    # Entry point: every workflow starts with intent classification
    g.set_entry_point("classify_intent")

    # After classification: either start executing or escalate immediately
    g.add_conditional_edges(
        "classify_intent",
        check_intent,
        {"execute_agent": "execute_agent", "escalate": "escalate"},
    )
    # After each agent execution: loop, respond, or escalate
    g.add_conditional_edges(
        "execute_agent",
        should_continue,
        {
            "execute_agent": "execute_agent",
            "generate_response": "generate_response",
            "escalate": "escalate",
        },
    )
    # Terminal edges: both response and escalation end the workflow
    g.add_edge("generate_response", END)
    g.add_edge("escalate", END)

    return g.compile()


def run_workflow(customer_message: str, context_disabled: bool = False, hybrid: bool = False) -> dict:
    """Run the complete multi-agent workflow for a single customer request.

    This is the primary public interface used by the evaluation runner and by
    any external caller.  It creates a fresh initial state, compiles the graph,
    invokes it, and attaches timing metadata to the result.

    Parameters
    ----------
    customer_message : str
        The raw natural-language request from the customer.
    context_disabled : bool
        H2 control condition.  When True, agents do not receive the shared
        context summary; each agent sees only the bare customer request.
    hybrid : bool
        When True, specialist agents use rule-based (non-LLM) tool dispatch.

    Returns
    -------
    dict
        The final WorkflowState dictionary augmented with end_time and
        elapsed_time fields for performance measurement.
    """
    init = ContextManager.create_initial_state(customer_message, context_disabled=context_disabled, hybrid=hybrid)
    init["current_task"] = None
    init["start_time"] = time.time()

    app = build_workflow()
    result = app.invoke(init)
    result["end_time"] = time.time()
    result["elapsed_time"] = result["end_time"] - result.get("start_time", result["end_time"])
    return result