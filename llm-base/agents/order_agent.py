from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from tools.db_tools import (
    get_order_details as _get_order_details,
    get_customer_orders as _get_customer_orders,
    update_order_status as _update_order_status,
    get_tracking_info as _get_tracking_info,
)


@tool(description="Get order details by order ID")
def get_order_details(order_id: int) -> dict:
    return _get_order_details(order_id)


@tool(description="Get all orders for a customer")
def get_customer_orders(customer_id: int) -> dict:
    return _get_customer_orders(customer_id)


@tool(description="Update order status")
def update_order_status(order_id: int, new_status: str) -> dict:
    return _update_order_status(order_id, new_status)


@tool(description="Get tracking info for an order")
def get_tracking_info(order_id: int) -> dict:
    return _get_tracking_info(order_id)


class OrderAgent(BaseAgent):

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
        self.tools = [
            get_order_details,
            get_customer_orders,
            update_order_status,
            get_tracking_info,
        ]