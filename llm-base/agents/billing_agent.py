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
    return _get_payment_status(order_id)


@tool(description="Process a refund for an order")
def process_refund(order_id: int, amount: float, reason: str) -> dict:
    return _process_refund(order_id, amount, reason)


@tool(description="Get billing history for a customer")
def get_billing_history(customer_id: int) -> dict:
    return _get_billing_history(customer_id)


@tool(description="Check for duplicate charges")
def check_duplicate_charges(customer_id: int) -> dict:
    return _check_duplicate_charges(customer_id)


class BillingAgent(BaseAgent):

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