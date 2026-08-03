"""
Database Access Layer: All Read and Write Operations

This module is the sole point of contact between the agent layer and the
SQLite database.  Every database operation in the system is implemented here
as a standalone function; no agent or orchestration code contains raw SQL.

This separation follows the Repository pattern: the agents think in terms of
business operations ("get order details", "process a refund") rather than SQL
statements.  It makes the agent logic easier to read and makes the database
layer easy to swap out — for example, a production deployment would replace
these SQLite calls with REST API calls to actual e-commerce backend services.

All functions follow the same return convention:
- On success: {"status": "success", "data": <result> | "message": <string>}
- On error:   {"status": "error",   "message": <human-readable description>}

This consistent structure allows the agents and the orchestrator to check
result.get("status") == "success" without having to handle different error
shapes from different functions.

Every function opens a fresh connection, executes its query, and closes the
connection in a finally block.  This is appropriate for a single-process
evaluation runner; in a concurrent production system, a connection pool would
be preferable.

"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_PATH


def _get_conn() -> sqlite3.Connection:
    """Open and return a new SQLite connection with Row factory enabled.

    The Row factory makes column values accessible by name (e.g. row["email"])
    in addition to by index, which simplifies result processing in callers.
    """
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Customer / Account operations
# ---------------------------------------------------------------------------

def get_customer_profile(customer_id: int) -> dict:
    """Retrieve a customer's full profile record.

    Used by AccountAgent to display or verify customer information.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return {"status": "error", "message": f"Customer {customer_id} not found."}
        return {"status": "success", "data": dict(row)}
    finally:
        conn.close()


def update_customer_email(customer_id: int, new_email: str) -> dict:
    """Replace the customer's stored email address.

    Returns an error if the customer does not exist (rowcount == 0).
    Does not validate email format; validation is assumed to happen upstream
    (or in a production system, at the API gateway layer).
    """
    conn = _get_conn()
    try:
        cur = conn.execute(
            "UPDATE customers SET email = ? WHERE customer_id = ?",
            (new_email, customer_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            return {"status": "error", "message": f"Customer {customer_id} not found."}
        return {"status": "success", "message": f"Email updated to {new_email}."}
    finally:
        conn.close()


def update_shipping_address(customer_id: int, new_address: str) -> dict:
    """Update the customer's default shipping address and propagate to pending orders.

    This function performs two writes in a single transaction:
    1. Updates the customer's profile record.
    2. Updates the shipping_address on any orders whose status is 'placed' or
       'processing' (i.e. orders that have not yet been dispatched).

    The count of affected orders is returned in the success message so the
    AccountAgent can inform the customer, as required by its system prompt.
    Updating already-shipped or delivered orders would be incorrect because
    the parcels are already in transit.
    """
    conn = _get_conn()
    try:
        cur = conn.execute(
            "UPDATE customers SET shipping_address = ? WHERE customer_id = ?",
            (new_address, customer_id)
        )
        if cur.rowcount == 0:
            return {"status": "error", "message": f"Customer {customer_id} not found."}

        # Propagate to pending orders only (not shipped or delivered)
        pending = conn.execute(
            "UPDATE orders SET shipping_address = ? "
            "WHERE customer_id = ? AND status IN ('placed', 'processing')",
            (new_address, customer_id)
        )
        conn.commit()
        return {
            "status": "success",
            "message": f"Address updated. {pending.rowcount} pending order(s) also updated.",
        }
    finally:
        conn.close()


def update_payment_method(customer_id: int, new_method: str) -> dict:
    """Replace the customer's stored payment method string.

    The payment method is stored as a free-text string (e.g. "Visa ending 4242",
    "PayPal: user@example.com") rather than a structured type, which keeps the
    schema simple for this proof-of-concept.
    """
    conn = _get_conn()
    try:
        cur = conn.execute(
            "UPDATE customers SET payment_method = ? WHERE customer_id = ?",
            (new_method, customer_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            return {"status": "error", "message": f"Customer {customer_id} not found."}
        return {"status": "success", "message": f"Payment method updated to {new_method}."}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Order operations
# ---------------------------------------------------------------------------

def get_order_details(order_id: int) -> dict:
    """Retrieve full order information, including product details via a JOIN.

    The JOIN with the products table allows the agent to report the product name
    and unit price without requiring a separate tool call.
    """
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT o.*, p.name as product_name, p.category, p.price as unit_price
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE o.order_id = ?
        """, (order_id,)).fetchone()
        if row is None:
            return {"status": "error", "message": f"Order {order_id} not found."}
        return {"status": "success", "data": dict(row)}
    finally:
        conn.close()


def get_customer_orders(customer_id: int) -> dict:
    """Return a list of all orders for a customer, sorted newest first.

    Results are ordered by placed_at DESC so that the first element is
    always the most recent order.  The orchestrator's dynamic task expansion
    logic relies on this ordering when it needs to target "the most recent order".
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT o.order_id, o.status, o.total_amount, o.placed_at,
                   p.name as product_name
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE o.customer_id = ?
            ORDER BY o.placed_at DESC
        """, (customer_id,)).fetchall()
        return {"status": "success", "data": [dict(r) for r in rows], "count": len(rows)}
    finally:
        conn.close()


def update_order_status(order_id: int, new_status: str) -> dict:
    """Change an order's lifecycle status with business-rule validation.

    This function enforces the cancellation constraint: an order cannot be
    cancelled if it has already shipped or been delivered.  Other lifecycle
    transitions are not validated here — the calling agent is responsible for
    applying additional business logic (e.g. checking that a return is only
    initiated for delivered orders).
    """
    allowed = ["placed", "shipped", "delivered", "cancelled", "return_initiated"]
    if new_status not in allowed:
        return {"status": "error", "message": f"Status must be one of: {allowed}"}

    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return {"status": "error", "message": f"Order {order_id} not found."}

        current = row["status"]
        # Business rule: shipped and delivered orders cannot be cancelled
        if new_status == "cancelled" and current in ("shipped", "delivered"):
            return {"status": "error", "message": f"Cannot cancel order that is already {current}."}

        conn.execute(
            "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
            (new_status, datetime.now().isoformat(), order_id)
        )
        conn.commit()
        return {
            "status": "success",
            "message": f"Order {order_id} changed from '{current}' to '{new_status}'.",
        }
    finally:
        conn.close()


def get_tracking_info(order_id: int) -> dict:
    """Return tracking number and human-readable tracking status for an order.

    The tracking status is derived from the order status and tracking number:
    - No tracking number : "Not shipped yet"
    - Status is 'shipped' : "In transit"
    - Otherwise (delivered, return_initiated) : "Delivered"

    In a production system, the tracking number would be passed to a carrier API
    to retrieve real-time location data.  Here it is stored statically in the
    database for simulation purposes.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT order_id, status, tracking_number, shipping_address "
            "FROM orders WHERE order_id = ?",
            (order_id,)
        ).fetchone()
        if row is None:
            return {"status": "error", "message": f"Order {order_id} not found."}

        data = dict(row)
        if not data["tracking_number"]:
            data["tracking_status"] = "No tracking number yet (not shipped)."
        elif data["status"] == "shipped":
            data["tracking_status"] = "In transit"
        else:
            data["tracking_status"] = "Delivered"

        return {"status": "success", "data": data}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Billing operations
# ---------------------------------------------------------------------------

def get_payment_status(order_id: int) -> dict:
    """Return all billing records (charges and refunds) associated with an order.

    Multiple records can exist for one order — for example, the seed data
    intentionally includes a duplicate charge for customer 5 (used in the
    error-detection scenarios S12, S28, S34, S37).
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM billing_records WHERE order_id = ? ORDER BY created_at",
            (order_id,)
        ).fetchall()
        if not rows:
            return {"status": "error", "message": f"No billing records for order {order_id}."}
        return {"status": "success", "data": [dict(r) for r in rows]}
    finally:
        conn.close()


def process_refund(order_id: int, amount: float, reason: str) -> dict:
    """Insert a refund record and guard against double-refund.

    Two safety checks are applied before inserting:
    1. Verify the order exists (prevents orphan refund records).
    2. Check for an existing refund on the same order (idempotency guard).
       If a refund already exists, the operation is rejected to prevent
       the scenario tested in S45 (deliberate double-refund attempt).

    The refund amount is passed as a parameter rather than looked up here
    because the BillingAgent's rule-based method is responsible for fetching
    the correct amount from the billing records before calling this function.
    """
    conn = _get_conn()
    try:
        order = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if order is None:
            return {"status": "error", "message": f"Order {order_id} not found."}

        # Idempotency guard: refuse a second refund for the same order
        existing = conn.execute(
            "SELECT billing_id FROM billing_records WHERE order_id = ? AND type = 'refund'",
            (order_id,)
        ).fetchone()
        if existing:
            return {"status": "error", "message": f"Refund already processed for order {order_id}."}

        conn.execute(
            "INSERT INTO billing_records (order_id, customer_id, amount, type, status, created_at) "
            "VALUES (?, ?, ?, 'refund', 'completed', ?)",
            (order_id, order["customer_id"], amount, datetime.now().isoformat())
        )
        conn.commit()
        return {"status": "success", "message": f"Refund of {amount} processed. Reason: {reason}"}
    finally:
        conn.close()


def get_billing_history(customer_id: int) -> dict:
    """Return all billing records for a customer, newest first.

    The JOIN with the orders table adds the current order status to each
    billing record, giving the agent (and ultimately the customer) context
    about whether the associated order has been delivered, cancelled, etc.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT b.*, o.status as order_status
            FROM billing_records b
            JOIN orders o ON b.order_id = o.order_id
            WHERE b.customer_id = ? ORDER BY b.created_at DESC
        """, (customer_id,)).fetchall()
        return {"status": "success", "data": [dict(r) for r in rows], "count": len(rows)}
    finally:
        conn.close()


def check_duplicate_charges(customer_id: int) -> dict:
    """Detect pairs of identical charge records for the same order.

    A duplicate charge is defined as two 'charge' type records for the same
    order with the same amount.  The self-join approach compares every pair
    (b1, b2) where b1.billing_id < b2.billing_id to avoid reporting each
    duplicate twice.

    The seed data inserts exactly one duplicate for customer 5 (order placed
    twice within 5 minutes), which is used in scenarios S12, S28, S34, and S37
    to test the system's error-detection capabilities (Use Case 4 from the thesis:
    "Error Prevention and Correction").
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT b1.billing_id as id1, b2.billing_id as id2,
                   b1.order_id, b1.amount, b1.created_at as time1, b2.created_at as time2
            FROM billing_records b1
            JOIN billing_records b2
              ON b1.order_id = b2.order_id
              AND b1.billing_id < b2.billing_id
              AND b1.type = 'charge' AND b2.type = 'charge'
              AND b1.amount = b2.amount
            WHERE b1.customer_id = ?
        """, (customer_id,)).fetchall()

        if not rows:
            return {"status": "success", "duplicates_found": False, "message": "No duplicates found."}

        return {
            "status": "success",
            "duplicates_found": True,
            "duplicates": [dict(r) for r in rows],
            "message": f"Found {len(rows)} duplicate charge(s).",
        }
    finally:
        conn.close()