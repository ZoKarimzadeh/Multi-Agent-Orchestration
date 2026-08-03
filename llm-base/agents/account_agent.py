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
    return _get_customer_profile(customer_id)


@tool(description="Update customer email address")
def update_customer_email(customer_id: int, new_email: str) -> dict:
    return _update_customer_email(customer_id, new_email)


@tool(description="Update shipping address")
def update_shipping_address(customer_id: int, new_address: str) -> dict:
    return _update_shipping_address(customer_id, new_address)


@tool(description="Update payment method")
def update_payment_method(customer_id: int, new_method: str) -> dict:
    return _update_payment_method(customer_id, new_method)


class AccountAgent(BaseAgent):

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