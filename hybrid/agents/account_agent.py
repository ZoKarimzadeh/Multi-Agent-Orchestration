"""
Specialist Agent for Customer Profile Management

The AccountAgent manages all operations that touch a customer's personal
profile: viewing the profile, updating the email address, changing the
shipping address, and modifying the payment method.

A noteworthy design detail: updating a shipping address has a side effect on
pending orders.  The update_shipping_address database function (in
tools/db_tools.py) atomically updates both the customer profile and
any orders that are still in 'placed' or 'processing' status.  The agent's
system prompt instructs it to mention how many pending orders were affected,
so the customer is not surprised by implicit changes.

Like all specialist agents, the AccountAgent does not access billing records
or order data directly.  Cross-domain interactions are coordinated by the
orchestrator (orchestration/graph.py).

"""

from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from tools.db_tools import (
    get_customer_profile as _get_customer_profile,
    update_customer_email as _update_customer_email,
    update_shipping_address as _update_shipping_address,
    update_payment_method as _update_payment_method,
)


@tool(description="Get customer profile by ID")
def get_customer_profile(customer_id: int) -> dict:
    """Retrieve a customer's name, email, address, and payment method."""
    return _get_customer_profile(customer_id)


@tool(description="Update customer email address")
def update_customer_email(customer_id: int, new_email: str) -> dict:
    """Overwrite the customer's stored email with the new address."""
    return _update_customer_email(customer_id, new_email)


@tool(description="Update shipping address")
def update_shipping_address(customer_id: int, new_address: str) -> dict:
    """Update the customer's default address and propagate to pending orders."""
    return _update_shipping_address(customer_id, new_address)


@tool(description="Update payment method")
def update_payment_method(customer_id: int, new_method: str) -> dict:
    """Replace the customer's stored payment method string."""
    return _update_payment_method(customer_id, new_method)


class AccountAgent(BaseAgent):
    """Specialist agent scoped to customer profile data management."""

    def __init__(self):
        super().__init__()
        self.name = "account_agent"
        self.system_prompt = (
            "You manage customer profile data: viewing profiles, updating email, "
            "changing shipping address, updating payment methods.\n"
            "Do not touch orders or billing records directly.\n"
            "Note: updating a shipping address also updates any pending orders "
            "for that customer — mention how many were updated.\n"
            "Check the workflow context before calling tools."
        )
        self.tools = [
            get_customer_profile,
            update_customer_email,
            update_shipping_address,
            update_payment_method,
        ]

    def invoke_rule_based(self, task_info: dict, state: dict) -> dict:
        """Execute the planned account action deterministically without an LLM call.

        See OrderAgent.invoke_rule_based for the general pattern.  The payment
        update action deserves special attention: the LLM may use several
        different key names for the new payment value depending on context
        (new_method, payment_method, new_payment_email, etc.).
        The rule-based implementation tries each known alias, which makes it
        more robust than relying on the LLM to always use the canonical name.

        Parameters
        ----------
        task_info : dict
            Task record from the AgentManager plan.
        state : dict
            Current WorkflowState dictionary.

        Returns
        -------
        dict
            Result dictionary (see BaseAgent.invoke for structure).
        """
        action = task_info.get("action", "")
        params = task_info.get("params", {})
        entities = state.get("entities", {})
        all_params = {**entities, **params}

        tool_calls = []

        def _customer_id() -> int:
            try:
                return int(all_params.get("customer_id") or state.get("customer_id") or 0)
            except (ValueError, TypeError):
                return 0

        if action == "get_profile":
            customer_id = _customer_id()
            result = _get_customer_profile(customer_id)
            tool_calls.append({"tool": "get_customer_profile", "args": {"customer_id": customer_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "update_email":
            customer_id = _customer_id()
            new_email = str(all_params.get("new_email", ""))
            result = _update_customer_email(customer_id, new_email)
            tool_calls.append({"tool": "update_customer_email", "args": {"customer_id": customer_id, "new_email": new_email}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "update_address":
            customer_id = _customer_id()
            new_address = str(all_params.get("new_address", ""))
            result = _update_shipping_address(customer_id, new_address)
            tool_calls.append({"tool": "update_shipping_address", "args": {"customer_id": customer_id, "new_address": new_address}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "update_payment":
            customer_id = _customer_id()
            # The LLM planner may use any of several key names for the new
            # payment value depending on how the customer phrased the request.
            # We try each known alias in priority order to handle all variants.
            new_method = str(
                all_params.get("new_method")
                or all_params.get("payment_method")
                or all_params.get("new_payment_method")
                or all_params.get("new_payment_email")
                or all_params.get("paypal_email")
                or ""
            )
            result = _update_payment_method(customer_id, new_method)
            tool_calls.append({"tool": "update_payment_method", "args": {"customer_id": customer_id, "new_method": new_method}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        else:
            success = False
            response = f"Unknown action for account_agent: '{action}'"

        return {
            "agent": self.name,
            "tool_calls": tool_calls,
            "llm_call_log": [],
            "response": response,
            "success": success,
        }