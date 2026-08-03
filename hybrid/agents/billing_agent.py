"""
Specialist Agent for Billing and Payment Operations

The BillingAgent handles all financial aspects of the customer service workflow:
checking whether a payment was received, processing refunds, retrieving the full
billing history, and detecting duplicate charges.

The duplicate-charge detection scenario (Scenario S37 in the evaluation) is
particularly interesting from a research perspective: the AgentManager cannot
know the order_id for the duplicate charge at plan time because it is only
revealed after the check_duplicate_charges tool has been called.  The
rule-based implementation resolves this by reading the prior check_duplicates
result from the workflow's shared context (completed_tasks), demonstrating
concretely why shared context (Hypothesis H2) reduces errors, without it, the
refund step would fail because the order_id would default to 0.

The refund logic also contains a deliberate robustness improvement over the
LLM-based agent: rather than trusting the LLM to provide the correct refund
amount, the rule-based implementation always looks up the original charge amount
from the billing records.  This prevents LLM hallucination of amount=0, which
would create an incorrect (zero-value) refund record in the database.

"""

from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from tools.db_tools import (
    get_payment_status as _get_payment_status,
    process_refund as _process_refund,
    get_billing_history as _get_billing_history,
    check_duplicate_charges as _check_duplicate_charges,
)


@tool(description="Get payment status for an order")
def get_payment_status(order_id: int) -> dict:
    """Return all billing records associated with the given order."""
    return _get_payment_status(order_id)


@tool(description="Process a refund for an order")
def process_refund(order_id: int, amount: float, reason: str) -> dict:
    """Insert a refund record into billing_records for the given order."""
    return _process_refund(order_id, amount, reason)


@tool(description="Get billing history for a customer")
def get_billing_history(customer_id: int) -> dict:
    """Return all charge and refund records for a customer, newest first."""
    return _get_billing_history(customer_id)


@tool(description="Check for duplicate charges")
def check_duplicate_charges(customer_id: int) -> dict:
    """Detect pairs of identical charge records for the same order and customer."""
    return _check_duplicate_charges(customer_id)


class BillingAgent(BaseAgent):
    """Specialist agent scoped to payment, refund, and billing operations."""

    def __init__(self):
        super().__init__()
        self.name = "billing_agent"
        self.system_prompt = (
            "You handle billing: payment status, refunds, billing history, "
            "and duplicate charge detection.\n"
            "Do not modify orders or account data directly.\n"
            "Before processing a refund, check the workflow context — the order_agent "
            "should have already verified the order exists and its status.\n"
            "When you find duplicates, include the billing record IDs in your report."
        )
        self.tools = [
            get_payment_status,
            process_refund,
            get_billing_history,
            check_duplicate_charges,
        ]

    def invoke_rule_based(self, task_info: dict, state: dict) -> dict:
        """Execute the planned billing action deterministically without an LLM call.

        Key implementation details for the process_refund action:

        1. order_id resolution from context: If the planned task has
           no valid order_id (because the planner did not know it at plan time),
           _find_duplicate_order_id searches the completed-task records for
           a prior check_duplicate_charges result and extracts the order_id
           from it.  This is a concrete example of how shared context enables
           correct operation in scenarios where the task plan is necessarily
           incomplete.

        2. Amount lookup from billing records: Rather than using any amount
           supplied by the planner, the rule-based implementation always fetches
           the original charge amount from the database.  This prevents a class
           of errors where the LLM hallucinates an incorrect refund amount.

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

        def _order_id() -> int:
            try:
                return int(all_params.get("order_id") or 0)
            except (ValueError, TypeError):
                return 0

        def _customer_id() -> int:
            try:
                return int(all_params.get("customer_id") or state.get("customer_id") or 0)
            except (ValueError, TypeError):
                return 0

        def _find_duplicate_order_id() -> int:
            """Resolve the order_id for a refund from a prior duplicate-check result.

            This is invoked when the planned refund task has order_id=0, which
            happens in scenarios like S37 where the order_id is not known at
            plan time.  The function walks backwards through the completed-task
            records looking for a check_duplicate_charges tool call, then
            extracts the order_id of the first detected duplicate.

            Without shared context (context_disabled=True), the
            completed_tasks list is cleared before each agent call, so this
            function would return 0, causing the refund to fail, directly
            illustrating the mechanism behind Hypothesis H2.
            """
            for completed in state.get("completed_tasks", []):
                for tc in completed.get("tool_calls", []):
                    if tc.get("tool") == "check_duplicate_charges":
                        dup_result = tc.get("result", {})
                        if dup_result.get("duplicates_found") and dup_result.get("duplicates"):
                            try:
                                return int(dup_result["duplicates"][0]["order_id"])
                            except (KeyError, ValueError, TypeError):
                                pass
            return 0

        if action == "get_payment_status":
            order_id = _order_id()
            result = _get_payment_status(order_id)
            tool_calls.append({"tool": "get_payment_status", "args": {"order_id": order_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "process_refund":
            order_id = _order_id()
            # If order_id unknown, resolve from a prior check_duplicates
            # result stored in the shared workflow context
            if order_id == 0:
                order_id = _find_duplicate_order_id()
            reason = str(all_params.get("reason", "customer request"))

            # Always look up the actual charge amount from billing records rather
            # than using whatever amount the planner may have supplied.  This
            # prevents hallucinated amounts from creating incorrect refund entries.
            payment_info = _get_payment_status(order_id)
            tool_calls.append({"tool": "get_payment_status", "args": {"order_id": order_id}, "result": payment_info})

            if payment_info.get("status") == "success":
                records = payment_info.get("data", [])
                charge_records = [r for r in records if r.get("type") == "charge"]
                amount = float(charge_records[0]["amount"]) if charge_records else 0.0
            else:
                amount = 0.0

            result = _process_refund(order_id, amount, reason)
            tool_calls.append({"tool": "process_refund", "args": {"order_id": order_id, "amount": amount, "reason": reason}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "get_billing_history":
            customer_id = _customer_id()
            result = _get_billing_history(customer_id)
            tool_calls.append({"tool": "get_billing_history", "args": {"customer_id": customer_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        elif action == "check_duplicates":
            customer_id = _customer_id()
            result = _check_duplicate_charges(customer_id)
            tool_calls.append({"tool": "check_duplicate_charges", "args": {"customer_id": customer_id}, "result": result})
            success = result.get("status") == "success"
            response = str(result)

        else:
            success = False
            response = f"Unknown action for billing_agent: '{action}'"

        return {
            "agent": self.name,
            "tool_calls": tool_calls,
            "llm_call_log": [],
            "response": response,
            "success": success,
        }