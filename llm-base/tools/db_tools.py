import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_PATH


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_customer_profile(customer_id: int) -> dict:
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
    conn = _get_conn()
    try:
        cur = conn.execute(
            "UPDATE customers SET shipping_address = ? WHERE customer_id = ?",
            (new_address, customer_id)
        )
        if cur.rowcount == 0:
            return {"status": "error", "message": f"Customer {customer_id} not found."}

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


def get_order_details(order_id: int) -> dict:
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


def get_payment_status(order_id: int) -> dict:
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
    conn = _get_conn()
    try:
        order = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if order is None:
            return {"status": "error", "message": f"Order {order_id} not found."}

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