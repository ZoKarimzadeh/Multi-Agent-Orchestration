import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.db_tools import _get_conn


def _orders_for_customer(conn: sqlite3.Connection, cid: int) -> dict:
    rows = conn.execute(
        "SELECT order_id, status, total_amount "
        "FROM orders WHERE customer_id = ? ORDER BY placed_at DESC",
        (cid,),
    ).fetchall()

    grouped: dict[str, list[dict]] = {
        "placed": [], "shipped": [], "delivered": [],
        "cancelled": [], "return_initiated": [],
    }
    for r in rows:
        d = dict(r)
        if d["status"] in grouped:
            grouped[d["status"]].append(d)
    return grouped


def build_scenarios() -> list[dict]:
    conn = _get_conn()
    by_customer: dict[int, dict] = {}
    for cid in range(1, 11):
        by_customer[cid] = _orders_for_customer(conn, cid)
    conn.close()

    def pick(cid, status, fallback=9999):
        bucket = by_customer[cid].get(status, [])
        if bucket:
            return bucket[0]["order_id"], bucket[0]["total_amount"]
        return fallback, 0.0

    scenarios = []

    oid1, _ = pick(1, "shipped")
    scenarios.append({
        "id": "S01", "category": "simple",
        "customer_message": f"Hi, I am customer 1. What is the status of order {oid1}?",
        "customer_id": 1,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_order_details"],
        "expected_outcome": "success",
        "description": "Basic order status lookup.",
    })

    scenarios.append({
        "id": "S02", "category": "simple",
        "customer_message": "Can you show me my profile? My customer ID is 3.",
        "customer_id": 3,
        "expected_agents": ["account_agent"],
        "required_tools": ["get_customer_profile"],
        "expected_outcome": "success",
        "description": "Customer profile retrieval.",
    })

    scenarios.append({
        "id": "S03", "category": "simple",
        "customer_message": "I would like to see my billing history. Customer 2.",
        "customer_id": 2,
        "expected_agents": ["billing_agent"],
        "required_tools": ["get_billing_history"],
        "expected_outcome": "success",
        "description": "Billing history lookup.",
    })

    oid4, _ = pick(4, "shipped")
    scenarios.append({
        "id": "S04", "category": "simple",
        "customer_message": f"Where is my package? Order {oid4}. I am customer 4.",
        "customer_id": 4,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_tracking_info"],
        "expected_outcome": "success",
        "description": "Tracking info for a shipped order.",
    })

    oid5, _ = pick(6, "delivered")
    scenarios.append({
        "id": "S05", "category": "simple",
        "customer_message": f"What is the payment status for order {oid5}? Customer 6 here.",
        "customer_id": 6,
        "expected_agents": ["billing_agent"],
        "required_tools": ["get_payment_status"],
        "expected_outcome": "success",
        "description": "Payment status check.",
    })

    scenarios.append({
        "id": "S06", "category": "simple",
        "customer_message": "Please list all my orders. My customer ID is 7.",
        "customer_id": 7,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_customer_orders"],
        "expected_outcome": "success",
        "description": "List all orders for a customer.",
    })

    oid7, _ = pick(8, "delivered")
    scenarios.append({
        "id": "S07", "category": "simple",
        "customer_message": f"Can I see the details for order {oid7}?",
        "customer_id": 8,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_order_details"],
        "expected_outcome": "success",
        "description": "Order details retrieval.",
    })

    scenarios.append({
        "id": "S08", "category": "simple",
        "customer_message": "I would like to update my email to anna.new@example.com. I am customer 1.",
        "customer_id": 1,
        "expected_agents": ["account_agent"],
        "required_tools": ["update_customer_email"],
        "expected_outcome": "success",
        "description": "Email update.",
    })

    scenarios.append({
        "id": "S09", "category": "simple",
        "customer_message": "Please change my shipping address to Berliner Str. 10, 10115 Berlin. Customer ID 4.",
        "customer_id": 4,
        "expected_agents": ["account_agent"],
        "required_tools": ["update_shipping_address"],
        "expected_outcome": "success",
        "description": "Shipping address update.",
    })

    scenarios.append({
        "id": "S10", "category": "simple",
        "customer_message": "I want to update my payment method to Visa ending 9999. I am customer 6.",
        "customer_id": 6,
        "expected_agents": ["account_agent"],
        "required_tools": ["update_payment_method"],
        "expected_outcome": "success",
        "description": "Payment method update.",
    })

    scenarios.append({
        "id": "S11", "category": "simple",
        "customer_message": "Show me the profile for customer 8.",
        "customer_id": 8,
        "expected_agents": ["account_agent"],
        "required_tools": ["get_customer_profile"],
        "expected_outcome": "success",
        "description": "Profile retrieval.",
    })

    scenarios.append({
        "id": "S12", "category": "simple",
        "customer_message": "Are there any duplicate charges on my account? Customer 5.",
        "customer_id": 5,
        "expected_agents": ["billing_agent"],
        "required_tools": ["check_duplicate_charges"],
        "expected_outcome": "success",
        "description": "Duplicate charge check.",
    })

    oid13, _ = pick(9, "placed")
    scenarios.append({
        "id": "S13", "category": "simple",
        "customer_message": f"Please cancel order {oid13}. I am customer 9.",
        "customer_id": 9,
        "expected_agents": ["order_agent"],
        "required_tools": ["update_order_status"],
        "expected_outcome": "success",
        "description": "Cancel a placed order.",
    })

    scenarios.append({
        "id": "S14", "category": "multi_step",
        "customer_message": (
            "I am customer 1. Show me all my orders and then give me "
            "tracking info for any that are shipped."
        ),
        "customer_id": 1,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_customer_orders", "get_tracking_info"],
        "expected_outcome": "success",
        "description": "List orders then get tracking.",
    })

    oid15, _ = pick(3, "placed")
    scenarios.append({
        "id": "S15", "category": "multi_step",
        "customer_message": f"I am customer 3. Check the details for order {oid15} and then cancel it.",
        "customer_id": 3,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_order_details", "update_order_status"],
        "expected_outcome": "success",
        "description": "Inspect then cancel.",
    })

    scenarios.append({
        "id": "S16", "category": "multi_step",
        "customer_message": (
            "Customer 2. Show me my billing history and check if I was charged twice for anything."
        ),
        "customer_id": 2,
        "expected_agents": ["billing_agent"],
        "required_tools": ["get_billing_history", "check_duplicate_charges"],
        "expected_outcome": "success",
        "description": "Billing history then duplicate detection.",
    })

    scenarios.append({
        "id": "S17", "category": "multi_step",
        "customer_message": (
            "I am customer 5. Look up my profile then update my email to eva.new@example.com."
        ),
        "customer_id": 5,
        "expected_agents": ["account_agent"],
        "required_tools": ["get_customer_profile", "update_customer_email"],
        "expected_outcome": "success",
        "description": "Profile lookup then email update.",
    })

    oid18, _ = pick(4, "delivered")
    scenarios.append({
        "id": "S18", "category": "multi_step",
        "customer_message": (
            f"Customer 4. Check order {oid18}. If it is delivered, please initiate a return."
        ),
        "customer_id": 4,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_order_details", "update_order_status"],
        "expected_outcome": "success",
        "description": "Check status then start return.",
    })

    oid19, _ = pick(7, "shipped")
    scenarios.append({
        "id": "S19", "category": "multi_step",
        "customer_message": f"I am customer 7. List my orders, then track order {oid19}.",
        "customer_id": 7,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_customer_orders", "get_tracking_info"],
        "expected_outcome": "success",
        "description": "List orders then track.",
    })

    oid20, _ = pick(6, "shipped")
    scenarios.append({
        "id": "S20", "category": "multi_step",
        "customer_message": (
            f"Customer 6. Show me the payment status for order {oid20} "
            f"and also my full billing history."
        ),
        "customer_id": 6,
        "expected_agents": ["billing_agent"],
        "required_tools": ["get_payment_status", "get_billing_history"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S21", "category": "multi_step",
        "customer_message": (
            "I am customer 9. Update my address to Potsdamer Platz 1, "
            "10785 Berlin AND update my email to irene.new@example.com."
        ),
        "customer_id": 9,
        "expected_agents": ["account_agent"],
        "required_tools": ["update_shipping_address", "update_customer_email"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S22", "category": "multi_step",
        "customer_message": (
            "Customer 8 here. Check my profile and then update my "
            "payment method to PayPal: hans.pay@example.com."
        ),
        "customer_id": 8,
        "expected_agents": ["account_agent"],
        "required_tools": ["get_customer_profile", "update_payment_method"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S23", "category": "multi_step",
        "customer_message": (
            "I am customer 10. List all my orders and get details on "
            "the most recent one."
        ),
        "customer_id": 10,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_customer_orders", "get_order_details"],
        "expected_outcome": "success",
        "description": "",
    })

    oid24, _ = pick(2, "shipped")
    scenarios.append({
        "id": "S24", "category": "multi_step",
        "customer_message": (
            f"Customer 2. Get tracking for order {oid24} and also "
            f"show me its full order details."
        ),
        "customer_id": 2,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_tracking_info", "get_order_details"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S25", "category": "multi_step",
        "customer_message": (
            "I am customer 3. Please show my profile and then update "
            "my address to Teststr. 1, 80331 Munich."
        ),
        "customer_id": 3,
        "expected_agents": ["account_agent"],
        "required_tools": ["get_customer_profile", "update_shipping_address"],
        "expected_outcome": "success",
        "description": "",
    })

    oid26, amt26 = pick(1, "placed")
    scenarios.append({
        "id": "S26", "category": "cross_system",
        "customer_message": (
            f"I am customer 1. I want to cancel order {oid26} and get "
            f"a full refund."
        ),
        "customer_id": 1,
        "expected_agents": ["order_agent", "billing_agent"],
        "required_tools": ["update_order_status", "process_refund"],
        "expected_outcome": "success",
        "description": "",
    })

    oid27, _ = pick(3, "shipped")
    scenarios.append({
        "id": "S27", "category": "cross_system",
        "customer_message": (
            f"Customer 3. I moved to Gartenstr. 15, 50667 Cologne. "
            f"Also, where is order {oid27}?"
        ),
        "customer_id": 3,
        "expected_agents": ["account_agent", "order_agent"],
        "required_tools": ["update_shipping_address", "get_tracking_info"],
        "expected_outcome": "success",
        "description": "",
    })

    oid28, _ = pick(5, "delivered")
    scenarios.append({
        "id": "S28", "category": "cross_system",
        "customer_message": (
            f"Hi, customer 5 here. I think I was double-charged. "
            f"Check my duplicates and show me details for order {oid28}."
        ),
        "customer_id": 5,
        "expected_agents": ["billing_agent", "order_agent"],
        "required_tools": ["check_duplicate_charges", "get_order_details"],
        "expected_outcome": "success",
        "description": "",
    })

    oid29, _ = pick(2, "delivered")
    scenarios.append({
        "id": "S29", "category": "cross_system",
        "customer_message": (
            f"Customer 2. Update my email to ben.new@example.com "
            f"and also check the payment for order {oid29}."
        ),
        "customer_id": 2,
        "expected_agents": ["account_agent", "billing_agent"],
        "required_tools": ["update_customer_email", "get_payment_status"],
        "expected_outcome": "success",
        "description": "",
    })

    oid30, amt30 = pick(4, "delivered")
    scenarios.append({
        "id": "S30", "category": "cross_system",
        "customer_message": (
            f"I am customer 4. Return order {oid30} -- the item was "
            f"defective. Please also process a refund."
        ),
        "customer_id": 4,
        "expected_agents": ["order_agent", "billing_agent"],
        "required_tools": ["update_order_status", "process_refund"],
        "expected_outcome": "success",
        "description": "",
    })

    oid31, _ = pick(6, "placed")
    scenarios.append({
        "id": "S31", "category": "cross_system",
        "customer_message": (
            f"Customer 6. Change my address to Neue Str. 5, 10117 Berlin "
            f"and cancel order {oid31}."
        ),
        "customer_id": 6,
        "expected_agents": ["account_agent", "order_agent"],
        "required_tools": ["update_shipping_address", "update_order_status"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S32", "category": "cross_system",
        "customer_message": (
            "Customer 7. I need to see my billing history and also "
            "update my payment method to Visa ending 4444."
        ),
        "customer_id": 7,
        "expected_agents": ["billing_agent", "account_agent"],
        "required_tools": ["get_billing_history", "update_payment_method"],
        "expected_outcome": "success",
        "description": "",
    })

    oid33, amt33 = pick(8, "placed")
    scenarios.append({
        "id": "S33", "category": "cross_system",
        "customer_message": (
            f"Hi, customer 8. Cancel order {oid33}, refund it, and "
            f"change my email to hans.new@example.com."
        ),
        "customer_id": 8,
        "expected_agents": ["order_agent", "billing_agent", "account_agent"],
        "required_tools": ["update_order_status", "process_refund", "update_customer_email"],
        "expected_outcome": "success",
        "description": "",
    })

    oid34, _ = pick(9, "shipped")
    scenarios.append({
        "id": "S34", "category": "cross_system",
        "customer_message": (
            f"Customer 9. Track order {oid34} and check my billing "
            f"for duplicates."
        ),
        "customer_id": 9,
        "expected_agents": ["order_agent", "billing_agent"],
        "required_tools": ["get_tracking_info", "check_duplicate_charges"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S35", "category": "cross_system",
        "customer_message": (
            "I am customer 10. Update my address to Bahnhofstr. 20, "
            "20095 Hamburg, check my orders, and show my billing history."
        ),
        "customer_id": 10,
        "expected_agents": ["account_agent", "order_agent", "billing_agent"],
        "required_tools": [
            "update_shipping_address", "get_customer_orders", "get_billing_history"
        ],
        "expected_outcome": "success",
        "description": "",
    })

    oid36, amt36 = pick(1, "delivered")
    scenarios.append({
        "id": "S36", "category": "cross_system",
        "customer_message": (
            f"Customer 1. Return order {oid36}, process a refund, and "
            f"send me a confirmation notification."
        ),
        "customer_id": 1,
        "expected_agents": ["order_agent", "billing_agent"],
        "required_tools": ["update_order_status", "process_refund"],
        "expected_outcome": "success",
        "description": "",
    })

    scenarios.append({
        "id": "S37", "category": "cross_system",
        "customer_message": (
            "Customer 5. Check duplicate charges on my account, process "
            "a refund for the extra charge, and update my payment method "
            "to Mastercard ending 1111."
        ),
        "customer_id": 5,
        "expected_agents": ["billing_agent", "account_agent"],
        "required_tools": ["check_duplicate_charges", "update_payment_method"],
        "expected_outcome": "success",
        "description": "",
    })

    oid38, _ = pick(2, "shipped")
    scenarios.append({
        "id": "S38", "category": "cross_system",
        "customer_message": (
            f"Customer 2. Show my profile, list my orders, and check "
            f"billing for order {oid38}."
        ),
        "customer_id": 2,
        "expected_agents": ["account_agent", "order_agent", "billing_agent"],
        "required_tools": [
            "get_customer_profile", "get_customer_orders", "get_payment_status"
        ],
        "expected_outcome": "success",
        "description": "",
    })

    oid39, _ = pick(1, "delivered")
    scenarios.append({
        "id": "S39", "category": "error_prone",
        "customer_message": f"Cancel order {oid39} immediately. Customer 1.",
        "customer_id": 1,
        "expected_agents": ["order_agent"],
        "required_tools": ["update_order_status"],
        "expected_outcome": "error",
        "description": "",
    })

    scenarios.append({
        "id": "S40", "category": "error_prone",
        "customer_message": "I need a refund for order 9999.",
        "customer_id": None,
        "expected_agents": ["billing_agent"],
        "required_tools": [],
        "expected_outcome": "error",
        "description": "",
    })

    scenarios.append({
        "id": "S41", "category": "error_prone",
        "customer_message": "Update the email for customer 999 to test@test.com.",
        "customer_id": 999,
        "expected_agents": ["account_agent"],
        "required_tools": [],
        "expected_outcome": "error",
        "description": "",
    })

    scenarios.append({
        "id": "S42", "category": "error_prone",
        "customer_message": "Hello, I have a problem with my order.",
        "customer_id": None,
        "expected_agents": [],
        "required_tools": [],
        "expected_outcome": "escalation",
        "description": "",
    })

    scenarios.append({
        "id": "S43", "category": "error_prone",
        "customer_message": "Fix my order right now!!! This is unacceptable!!!",
        "customer_id": None,
        "expected_agents": [],
        "required_tools": [],
        "expected_outcome": "escalation",
        "description": "",
    })

    scenarios.append({
        "id": "S44", "category": "error_prone",
        "customer_message": (
            "I want to cancel my order and also change my address. "
            "My name is Sarah."
        ),
        "customer_id": None,
        "expected_agents": [],
        "required_tools": [],
        "expected_outcome": "escalation",
        "description": "",
    })

    oid45, amt45 = pick(10, "delivered")
    scenarios.append({
        "id": "S45", "category": "error_prone",
        "customer_message": (
            f"Customer 10. Process a refund for order {oid45} because "
            f"the item was defective. Actually, process the refund twice "
            f"to make sure."
        ),
        "customer_id": 10,
        "expected_agents": ["billing_agent"],
        "required_tools": ["process_refund"],
        "expected_outcome": "success",
        "description": "",
    })

    oid46, _ = pick(3, "cancelled")
    scenarios.append({
        "id": "S46", "category": "error_prone",
        "customer_message": f"Customer 3. Cancel order {oid46} please.",
        "customer_id": 3,
        "expected_agents": ["order_agent"],
        "required_tools": ["get_order_details"],
        "expected_outcome": "error",
        "description": "",
    })

    scenarios.append({
        "id": "S47", "category": "error_prone",
        "customer_message": (
            "Transfer my order to a different customer entirely. "
            "I am customer 7."
        ),
        "customer_id": 7,
        "expected_agents": [],
        "required_tools": [],
        "expected_outcome": "escalation",
        "description": "",
    })

    scenarios.append({
        "id": "S48", "category": "error_prone",
        "customer_message": "I want to speak to a manager immediately.",
        "customer_id": None,
        "expected_agents": [],
        "required_tools": [],
        "expected_outcome": "escalation",
        "description": "",
    })

    oid49, _ = pick(4, "shipped")
    scenarios.append({
        "id": "S49", "category": "error_prone",
        "customer_message": f"Customer 4. Cancel order {oid49} and refund it.",
        "customer_id": 4,
        "expected_agents": ["order_agent"],
        "required_tools": ["update_order_status"],
        "expected_outcome": "error",
        "description": "",
    })

    scenarios.append({
        "id": "S50", "category": "error_prone",
        "customer_message": (
            "Customer 8. Return order 9999 and also change the "
            "address on that order to somewhere in Japan. Also refund."
        ),
        "customer_id": 8,
        "expected_agents": ["order_agent", "billing_agent", "account_agent"],
        "required_tools": ["update_order_status"],
        "expected_outcome": "error",
        "description": "",
    })

    assert len(scenarios) == 50, f"Expected 50 scenarios, got {len(scenarios)}"
    return scenarios


def print_summary(scenarios=None):
    if scenarios is None:
        scenarios = build_scenarios()

    counts: dict[str, int] = {}
    for s in scenarios:
        counts[s["category"]] = counts.get(s["category"], 0) + 1

    print(f"Total: {len(scenarios)}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print()

    for s in scenarios:
        agents = ", ".join(s["expected_agents"]) or "(any)"
        print(f"  [{s['id']}] {s['category']:14s}  {s['expected_outcome']:7s}  [{agents}]")
        print(f"         {s['description']}")


if __name__ == "__main__":
    print_summary()