"""
Database Schema Creation and Deterministic Data Seeding

This module creates the SQLite database and populates it with a realistic but
fully controlled set of sample data.  It is executed at the start of every
evaluation run (via reset_database()) to ensure a clean, identical initial
state, which is a prerequisite for reproducible experimental results.

The data set is designed to exercise all four thesis use cases:
1. Order Tracking : Orders in all lifecycle states (placed, shipped,
   delivered, cancelled, return_initiated) across 10 customers.
2. Multi-System Updates : Customers with existing pending orders so that
   address-update propagation can be tested.
3. Service Request Handling : Support tickets representing real-world
   scenarios (missing delivery, email change, duplicate charge).
4. Error Detection : Customer 5 has a deliberate duplicate charge record
   (two 'charge' entries for the same order), used in scenarios S12/S37.

The seed data is deterministic in structure but uses random for some
values (order quantities, statuses, dates).  The random seed is fixed in
the evaluation runner (SEED = 42) to ensure consistent order distributions
across all experimental runs.

"""

import gc
import random
import sqlite3
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_PATH


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all six tables if they do not already exist.

    Using "CREATE TABLE IF NOT EXISTS" means this function is safe to call
    on an existing database; it will not overwrite data.  In practice,
    reset_database() always deletes the file first so the tables are
    always freshly created.

    Schema notes:
    - customers.email has a UNIQUE constraint to prevent duplicate accounts.
    - orders uses foreign keys to both customers and products, enforcing
      referential integrity.
    - billing_records.type is either 'charge' or 'refund'; no constraint
      is enforced at the database level to keep the schema simple.
    - support_tickets.order_id is nullable (a ticket may not relate to a
      specific order).
    - notifications is append-only; no agent reads from it.
    """
    conn.cursor().executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            shipping_address TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'placed',
            shipping_address TEXT,
            tracking_number TEXT,
            placed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE IF NOT EXISTS billing_records (
            billing_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            type TEXT NOT NULL DEFAULT 'charge',
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_id INTEGER,
            subject TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE IF NOT EXISTS notifications (
            notification_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT 'email',
            sent_at TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)
    conn.commit()


def populate_sample_data(conn: sqlite3.Connection) -> None:
    """Insert the fixed set of customers, products, orders, billing records, and tickets.

    Design decisions for the sample data:

    Customers: 10 customers.
    The diversity of payment methods (Visa, Mastercard, PayPal) allows
    payment-update scenarios to test all paths.

    Products: 10 products across four categories.  The variety ensures
    that order detail responses contain meaningful, readable information.

    Orders: Each customer receives 2–5 randomly generated orders.
    Status weights are tuned so that the dataset contains a realistic
    distribution: more delivered orders than placed ones, with a small
    fraction cancelled or in return.

    Billing records: One charge record per order.  Customer 5's first
    order additionally receives a second, duplicate charge record (inserted
    5 minutes after the first) to seed the duplicate-detection test case.

    Support tickets: Three manually crafted tickets representing the
    three support scenarios.
    """
    cur = conn.cursor()
    now = datetime.now()

    # Customers
    # Fixed IDs (1–10) ensure that scenarios can reference customer_id directly.
    customers = [
        (1,  "Anna Mueller",   "anna.mueller@example.com",   "Friedrichstr. 42, 10117 Berlin",      "Visa ending 4242"),
        (2,  "Ben Schmidt",    "ben.schmidt@example.com",    "Hauptstr. 15, 80331 Munich",           "Mastercard ending 5555"),
        (3,  "Clara Weber",    "clara.weber@example.com",    "Koenigsallee 78, 40212 Duesseldorf",  "PayPal: clara.w@example.com"),
        (4,  "David Fischer",  "david.fischer@example.com",  "Moenckebergstr. 3, 20095 Hamburg",    "Visa ending 1234"),
        (5,  "Eva Braun",      "eva.braun@example.com",      "Zeil 106, 60313 Frankfurt",           "Mastercard ending 9876"),
        (6,  "Frank Wagner",   "frank.wagner@example.com",   "Schlossstr. 22, 70173 Stuttgart",     "Visa ending 3333"),
        (7,  "Greta Hoffmann", "greta.hoffmann@example.com", "Marktplatz 8, 04109 Leipzig",         "PayPal: greta.h@example.com"),
        (8,  "Hans Becker",    "hans.becker@example.com",    "Breite Str. 50, 50667 Cologne",       "Mastercard ending 7777"),
        (9,  "Irene Koch",     "irene.koch@example.com",     "Kaiserstr. 12, 76131 Karlsruhe",      "Visa ending 8888"),
        (10, "Jan Richter",    "jan.richter@example.com",    "Poststr. 5, 01067 Dresden",           "Visa ending 6666"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?)",
        [
            (c[0], c[1], c[2], c[3], c[4],
             (now - timedelta(days=random.randint(30, 365))).isoformat())
            for c in customers
        ]
    )

    # Products
    # Fixed IDs (101–110) and prices ensure that total_amount calculations
    # in the order generation below are deterministic given a fixed random seed.
    products = [
        (101, "Wireless Headphones", "Electronics",   79.99, 150),
        (102, "USB-C Hub Adapter",   "Electronics",   34.99, 300),
        (103, "Mechanical Keyboard", "Electronics",  129.99,  80),
        (104, "Running Shoes",       "Sports",        89.99, 200),
        (105, "Yoga Mat",            "Sports",        29.99, 500),
        (106, "Coffee Maker",        "Home",          59.99, 120),
        (107, "Desk Lamp",           "Home",          39.99, 250),
        (108, "Backpack",            "Accessories",   49.99, 180),
        (109, "Water Bottle",        "Accessories",   19.99, 400),
        (110, "Bluetooth Speaker",   "Electronics",   69.99, 100),
    ]
    cur.executemany("INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, ?)", products)

    # Orders and billing records
    # Status weights (15:25:40:10:10) are chosen to produce a realistic distribution:
    # most orders are delivered (40%), a good portion are in transit (25%),
    # some are freshly placed (15%), and a small fraction are cancelled or returned.
    statuses = ["placed", "shipped", "delivered", "cancelled", "return_initiated"]
    orders = []
    billing = []
    oid = 1001  # Starting order_id (avoids overlap with customer/product IDs)

    for cid in range(1, 11):
        for _ in range(random.randint(2, 5)):
            prod = random.choice(products)
            qty = random.randint(1, 3)
            total = round(prod[3] * qty, 2)
            status = random.choices(statuses, weights=[15, 25, 40, 10, 10], k=1)[0]
            days_ago = random.randint(1, 60)
            placed = now - timedelta(days=days_ago)
            # updated_at is set to a time between placement and today
            updated = placed + timedelta(days=random.randint(0, min(days_ago, 7)))
            # Tracking numbers are only assigned to shipped/delivered orders
            tracking = (
                f"TRK{random.randint(100000, 999999)}"
                if status in ("shipped", "delivered") else None
            )

            orders.append((
                oid, cid, prod[0], qty, total, status,
                customers[cid - 1][3], tracking,
                placed.isoformat(), updated.isoformat()
            ))
            # Every order has exactly one 'charge' billing record
            billing.append((
                oid * 10, oid, cid, total, "charge", "completed", placed.isoformat()
            ))

            # Deliberate duplicate charge for customer 5's first order.
            # This seeds the error-detection test cases (scenarios S12, S28, S34, S37).
            # The duplicate is timestamped 5 minutes after the original to simulate
            # a payment processing error (e.g. a double-click on "Pay").
            if cid == 5 and len([o for o in orders if o[1] == 5]) == 1:
                billing.append((
                    oid * 10 + 1, oid, cid, total, "charge", "completed",
                    (placed + timedelta(minutes=5)).isoformat()
                ))

            oid += 1

    cur.executemany(
        "INSERT OR IGNORE INTO orders (order_id, customer_id, product_id, quantity, total_amount, status, shipping_address, tracking_number, placed_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        orders
    )
    cur.executemany("INSERT OR IGNORE INTO billing_records VALUES (?, ?, ?, ?, ?, ?, ?)", billing)

    # Support tickets
    # Three pre-seeded tickets covering the scenarios.
    # Ticket 1: Customer 1 — missing delivery (tests order inquiry + escalation)
    # Ticket 2: Customer 3 — email change request (tests account update workflow)
    # Ticket 3: Customer 5 — duplicate charge complaint (tests billing error detection)
    tickets = [
        (1, 1, 1001, "Order not received",
         "My order shows delivered but nothing arrived.", "open",
         (now - timedelta(days=2)).isoformat(), None),
        (2, 3, None, "Update email",
         "Please change my email to clara.new@example.com.", "open",
         (now - timedelta(days=1)).isoformat(), None),
        (3, 5, None, "Duplicate charge",
         "I was charged twice for my last order.", "open",
         now.isoformat(), None),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tickets
    )

    # --- Post-seeding overrides for scenario synchronization ---
    # Scenario S27 requires customer 3 to have a shipped order
    cur.execute(
        "UPDATE orders SET status='shipped', tracking_number='TRK738720' "
        "WHERE order_id=1012"
    )

    # Scenario S07 requires customer 8 to have a delivered order
    cur.execute(
        "UPDATE orders SET status='delivered', tracking_number='TRK324643' "
        "WHERE order_id=1034"
    )

    # Scenario S15 requires customer 3 to have a placed order
    cur.execute(
        "UPDATE orders SET status='placed', tracking_number=NULL "
        "WHERE order_id=1013"
    )

    # Scenario S19 requires customer 7 to have a shipped order
    cur.execute(
        "UPDATE orders SET status='shipped', tracking_number='TRK555100' "
        "WHERE order_id=1031"
    )

    # Scenarios S24 and S38 require customer 2 to have a shipped order
    cur.execute(
        "INSERT INTO orders (order_id, customer_id, product_id, quantity, total_amount, "
        "status, shipping_address, tracking_number, placed_at, updated_at) "
        "VALUES (1040, 2, 109, 2, 39.98, 'shipped', "
        "'Hauptstr. 15, 80331 Munich', 'TRK888801', "
        "'2026-07-20T10:00:00', '2026-07-22T10:00:00')"
    )
    cur.execute(
        "INSERT INTO billing_records VALUES (10400, 1040, 2, 39.98, 'charge', 'completed', '2026-07-20T10:00:00')"
    )

    # Scenario S31 requires customer 6 to have a placed order
    cur.execute(
        "UPDATE orders SET status='placed', tracking_number=NULL "
        "WHERE order_id=1024"
    )

    # Scenario S34 requires customer 9 to have a shipped order
    cur.execute(
        "INSERT INTO orders (order_id, customer_id, product_id, quantity, total_amount, "
        "status, shipping_address, tracking_number, placed_at, updated_at) "
        "VALUES (1041, 9, 102, 1, 34.99, 'shipped', "
        "'Kaiserstr. 12, 76131 Karlsruhe', 'TRK888802', "
        "'2026-07-19T10:00:00', '2026-07-21T10:00:00')"
    )
    cur.execute(
        "INSERT INTO billing_records VALUES (10410, 1041, 9, 34.99, 'charge', 'completed', '2026-07-19T10:00:00')"
    )

    conn.commit()


def reset_database() -> sqlite3.Connection:
    """Delete the existing database file, recreate it, and return an open connection.

    This function is the primary entry point used by the evaluation runner.
    It guarantees a clean slate before each experimental run by:
    1. Collecting any Python garbage to release unreferenced SQLite connections
       (which would prevent the file from being deleted on Windows).
    2. Waiting up to 10 seconds for any stubborn file locks to clear.
    3. Creating the parent directory if needed.
    4. Creating a fresh database file with the full schema and sample data.

    Returns
    -------
    sqlite3.Connection
        An open connection to the newly created database.  The caller is
        responsible for closing it.
    """
    random.seed(42)
    if DATABASE_PATH.exists():
        gc.collect()  # Release any Python-held file handles before attempting deletion
        for attempt in range(10):
            try:
                DATABASE_PATH.unlink()
                break
            except PermissionError:
                if attempt == 9:
                    raise  # Give up after 10 attempts (10 seconds total)
                _time.sleep(1.0)  # Wait 1 second for the lock to be released

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    populate_sample_data(conn)
    return conn


if __name__ == "__main__":
    """reset the database and print row counts for each table."""
    conn = reset_database()
    c = conn.cursor()
    for table in ("customers", "products", "orders", "billing_records", "support_tickets"):
        c.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"{table}: {c.fetchone()[0]}")
    conn.close()
    print(f"\nDatabase: {DATABASE_PATH}")