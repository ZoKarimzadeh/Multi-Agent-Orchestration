"""
Specialist Agent for Order-Related Operations

The OrderAgent is responsible for all customer requests that concern the
lifecycle of an order: checking its current status, retrieving tracking
information, cancelling a placed order, or initiating a return.

It is deliberately scoped to order data only.  It does not touch billing
records or customer profile information, those domains belong to the
BillingAgent and AccountAgent respectively.  This strict domain separation
is a core architectural principle of the multi-agent design: it reduces the
risk that a single agent will inadvertently corrupt data outside its remit,
and it makes the system easier to reason about and test (Chapter 5, Design).

The agent has two execution modes (inherited from BaseAgent):
- LLM mode (invoke): The LLM reads the context and decides which
  tools to call.  Flexible but slower and slightly unpredictable.
- Rule-based / hybrid mode (invoke_rule_based): A Python switch
  statement maps the AgentManager's planned action directly to a
  database call.  Deterministic, fast, and cheaper (no LLM call).

"""

from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from tools.db_tools import (
    get_order_details as _get_order_details,
    get_customer_orders as _get_customer_orders,
    update_order_status as _update_order_status,
    get_tracking_info as _get_tracking_info,
)

# Each tool wraps a database function and is decorated with @tool so that
# LangChain can generate the JSON schema the LLM needs to call it correctly.
# The description string is included in the LLM prompt; clear descriptions
# reduce the chance of the LLM choosing the wrong tool.


@tool(description="Get order details by order ID")
def get_order_details(order_id: int) -> dict:
    """Retrieve full order information (status, product, amounts, timestamps)."""
    return _get_order_details(order_id)


@tool(description="Get all orders for a customer")
def get_customer_orders(customer_id: int) -> dict:
    """List all orders belonging to a customer, sorted newest first."""
    return _get_customer_orders(customer_id)


@tool(description="Update order status")
def update_order_status(order_id: int, new_status: str) -> dict:
    """Change an order's lifecycle status (e.g. 'placed' to 'cancelled')."""
    return _update_order_status(order_id, new_status)


@tool(description="Get tracking info for an order")
def get_tracking_info(order_id: int) -> dict:
    """Return the tracking number and derived tracking status for an order."""
    return _get_tracking_info(order_id)


class OrderAgent(BaseAgent):
    """Specialist agent scoped to order management operations."""

    def __init__(self):
        super().__init__()
        self.name = "order_agent"
        self.system_prompt = (
            "You handle order-related requests: checking status, tracking, "
            "cancellations (placed orders only), and return initiations.\n"
            "Do not touch billing or account data — those belong to other agents.\n"
            "Check the workflow context first before calling tools; "
            "another agent may have already pulled what you need.\n"
            "If you cannot complete the task, explain why."
        )
        # Only order-related tools are exposed, the LLM cannot call billing
        # or account tools even if it tries.
        self.tools = [
            get_order_details,
            get_customer_orders,
            update_order_status,
            get_tracking_info,
        ]

    def invoke_rule_based(self, task_info: dict, state: dict) -> dict:
        """Execute the planned action deterministically without an LLM call.

        This method is used when the system runs in hybrid mode (hybrid=True).
        The AgentManager has already decided which action to perform and with what
        parameters; this method simply maps that action to a direct database call.

        Rule-based execution also applies business logic that would otherwise
        rely on the LLM to reason correctly:
        - Cancellation is blocked if the order has already shipped or been delivered.
        - Return initiation first verifies the order exists before updating status.

        Parameters
        ----------
        task_info : dict
            The task record from the AgentManager's plan, containing keys:
            agent, task, action, and params.
        state : dict
            The current WorkflowState dictionary (used to resolve entity values
            that may have been extracted into the top-level entities field).

        Returns
        -------
        dict
            Same structure as BaseAgent.invoke: agent, tool_calls, llm_call_log,
            response, success.  llm_call_log is always empty here since no
            LLM is involved.
        """
        action = task_info.get("action", "")
        params = task_info.get("params", {})
        entities = state.get("entities", {})
        # Merge entity-level params with task-level params; task params take precedence
        all_params = {**entities, **params}

        tool_calls = []  # Records of every database call made

        # Helper: extract and coerce order_id to int, defaulting to 0 if missing/invalid
        def _order_id() -> int:
            try:
                return int(all_params.get("order_id") or 0)
            except (ValueError, TypeError):
                return 0

        # Helper: extract customer_id, falling back to state-level value if not in params
        def _customer_id() -> int:
            try:
                return int(all_params.get("customer_id") or state.get("customer_id") or 0)
            except (ValueError, TypeError):
                return 0

        if action == "get_order_status":
            order_id = _order_id()
            result = _get_order_details(order_id)
            tool_calls.append({"tool": "get_order_details", "args": {"order_id": order_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "get_tracking":
            order_id = _order_id()
            result = _get_tracking_info(order_id)
            tool_calls.append({"tool": "get_tracking_info", "args": {"order_id": order_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "cancel_order":
            # Business rule: an order can only be cancelled if it is in 'placed'
            # or 'processing' status.  Shipped or delivered orders cannot be
            # cancelled — the customer must initiate a return instead.
            order_id = _order_id()
            details = _get_order_details(order_id)
            tool_calls.append({"tool": "get_order_details", "args": {"order_id": order_id}, "result": details})
            if details.get("status") == "success":
                current_status = details["data"]["status"]
                if current_status in ("placed", "processing"):
                    result = _update_order_status(order_id, "cancelled")
                    tool_calls.append({"tool": "update_order_status", "args": {"order_id": order_id, "new_status": "cancelled"}, "result": result})
                    success = result.get("status") == "success"
                    response = str(result)
                else:
                    # Cancellation is not possible — surface a meaningful error
                    result = {"status": "error", "message": f"Cannot cancel order with status '{current_status}'."}
                    tool_calls.append({"tool": "update_order_status", "args": {"order_id": order_id, "new_status": "cancelled"}, "result": result})
                    success = False
                    response = str(result)
            else:
                success = False
                response = str(details)

        elif action == "initiate_return":
            # Verify the order exists before attempting the status change
            order_id = _order_id()
            details = _get_order_details(order_id)
            tool_calls.append({"tool": "get_order_details", "args": {"order_id": order_id}, "result": details})
            if details.get("status") == "success":
                result = _update_order_status(order_id, "return_initiated")
                tool_calls.append({"tool": "update_order_status", "args": {"order_id": order_id, "new_status": "return_initiated"}, "result": result})
                success = result.get("status") == "success"
                response = str(result)
            else:
                success = False
                response = str(details)

        elif action == "list_orders":
            customer_id = _customer_id()
            result = _get_customer_orders(customer_id)
            tool_calls.append({"tool": "get_customer_orders", "args": {"customer_id": customer_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        else:
            # Unknown action — this should not occur in normal operation since the
            # AgentManager is constrained to the actions listed in the classification
            # prompt, but we handle it gracefully rather than raising an exception.
            success = False
            response = f"Unknown action for order_agent: '{action}'"

        return {
            "agent": self.name,
            "tool_calls": tool_calls,
            "llm_call_log": [],  # Empty: no LLM calls in rule-based mode
            "response": response,
            "success": success,
        }