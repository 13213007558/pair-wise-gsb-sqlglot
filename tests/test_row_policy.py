import sqlite3
import unittest

import sqlglot
from sqlglot import exp
from sqlglot.errors import UnsupportedError
from sqlglot.row_policy import apply_row_policies


class TestRowPolicy(unittest.TestCase):
    def assert_rewrite(self, sql, policies, expected, dialect=None):
        self.assertEqual(apply_row_policies(sql, policies, dialect=dialect).sql(), expected)

    def test_inner_join(self):
        self.assert_rewrite(
            "SELECT c.id, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id",
            {"orders": "tenant_id = 1"},
            "SELECT c.id, o.amount FROM customers AS c JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS o ON c.id = o.customer_id",
        )

    def test_left_join(self):
        self.assert_rewrite(
            "SELECT * FROM customers c LEFT JOIN orders o ON c.id = o.customer_id",
            {"orders": "tenant_id = 1"},
            "SELECT * FROM customers AS c LEFT JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS o ON c.id = o.customer_id",
        )

    def test_table_without_alias(self):
        self.assert_rewrite(
            "SELECT orders.id FROM orders",
            {"orders": "tenant_id = 1"},
            "SELECT orders.id FROM (SELECT * FROM orders WHERE tenant_id = 1) AS orders",
        )

    def test_self_join(self):
        self.assert_rewrite(
            "SELECT * FROM orders o1 JOIN orders o2 ON o1.parent_id = o2.id",
            {"orders": "tenant_id = 1"},
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS o1 JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS o2 ON o1.parent_id = o2.id",
        )

    def test_schema_qualified(self):
        self.assert_rewrite(
            "SELECT * FROM main.orders",
            {"main.orders": "tenant_id = 1"},
            "SELECT * FROM (SELECT * FROM main.orders WHERE tenant_id = 1) AS orders",
        )
        # A policy for a different qualification must not match
        self.assert_rewrite(
            "SELECT * FROM main.orders",
            {"orders": "tenant_id = 1"},
            "SELECT * FROM main.orders",
        )

    def test_cte_shadowing(self):
        # A CTE named like a physical table is not a physical table reference
        self.assert_rewrite(
            "WITH orders AS (SELECT 1 AS id) SELECT * FROM orders",
            {"orders": "tenant_id = 1"},
            "WITH orders AS (SELECT 1 AS id) SELECT * FROM orders",
        )

    def test_table_inside_cte(self):
        self.assert_rewrite(
            "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
            {"orders": "tenant_id = 1"},
            "WITH recent AS (SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS orders) SELECT * FROM recent",
        )

    def test_correlated_subquery(self):
        self.assert_rewrite(
            "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
            {"orders": "tenant_id = 1"},
            "SELECT * FROM customers AS c WHERE EXISTS(SELECT 1 FROM (SELECT * FROM orders WHERE tenant_id = 1) AS o WHERE o.customer_id = c.id)",
        )

    def test_set_operation_branches(self):
        self.assert_rewrite(
            "SELECT id FROM orders UNION ALL SELECT id FROM archived_orders",
            {"orders": "tenant_id = 1", "archived_orders": "tenant_id = 1"},
            "SELECT id FROM (SELECT * FROM orders WHERE tenant_id = 1) AS orders UNION ALL SELECT id FROM (SELECT * FROM archived_orders WHERE tenant_id = 1) AS archived_orders",
        )

    def test_using(self):
        self.assert_rewrite(
            "SELECT * FROM customers LEFT JOIN orders USING (id)",
            {"orders": "tenant_id = 1"},
            "SELECT * FROM customers LEFT JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS orders USING (id)",
        )

    def test_policy_qualified_with_table_name(self):
        self.assert_rewrite(
            "SELECT * FROM orders o",
            {"orders": "orders.tenant_id = 1"},
            "SELECT * FROM (SELECT * FROM orders WHERE orders.tenant_id = 1) AS o",
        )

    def test_policy_expression_and_placeholders(self):
        result = apply_row_policies(
            "SELECT * FROM orders",
            {"orders": exp.condition("tenant_id = :tenant")},
        )
        self.assertEqual(
            result.sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = :tenant) AS orders",
        )
        self.assertIsNotNone(result.find(exp.Placeholder))

    def test_where_clause_policy(self):
        result = apply_row_policies("SELECT * FROM orders", {"orders": exp.Where(this=exp.true())})
        self.assertEqual(result.sql(), "SELECT * FROM (SELECT * FROM orders WHERE TRUE) AS orders")

    def test_rejects_non_select(self):
        for sql in ("DELETE FROM orders", "UPDATE orders SET id = 1", "CREATE TABLE t (i INT)"):
            with self.assertRaises(UnsupportedError):
                apply_row_policies(sql, {"orders": "tenant_id = 1"})

    def test_rejects_pivoted_table(self):
        with self.assertRaises(UnsupportedError):
            apply_row_policies(
                "SELECT * FROM orders PIVOT (SUM(amount) FOR status IN ('a' AS x))",
                {"orders": "tenant_id = 1"},
            )

    def test_trees_are_independent(self):
        query = sqlglot.parse_one("SELECT * FROM orders o")
        policy = exp.condition("tenant_id = 1")

        result = apply_row_policies(query, {"orders": policy})

        self.assertEqual(query.sql(), "SELECT * FROM orders AS o")
        self.assertEqual(policy.sql(), "tenant_id = 1")
        self.assertEqual(
            result.sql(), "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS o"
        )

        # Mutating the result must not leak into the input tree or the policy tree
        for table in result.find_all(exp.Table):
            table.set("this", exp.to_identifier("other"))
        self.assertEqual(query.sql(), "SELECT * FROM orders AS o")
        self.assertEqual(policy.sql(), "tenant_id = 1")

    def test_sqlite_left_join_semantics(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER, name TEXT);
            CREATE TABLE orders (id INTEGER, customer_id INTEGER, tenant_id INTEGER, amount REAL);
            INSERT INTO customers VALUES (1, 'a'), (2, 'b');
            INSERT INTO orders VALUES (1, 1, 1, 10.0), (2, 1, 2, 20.0), (3, 2, 2, 30.0);
            """
        )

        # Tenant 1: customer 'a' has an order, customer 'b' only has orders of
        # another tenant and must survive the LEFT JOIN with NULLs on the right
        hand_written = """
            SELECT c.name, o.id, o.amount
            FROM customers c
            LEFT JOIN (SELECT * FROM orders WHERE tenant_id = ?) o ON c.id = o.customer_id
            ORDER BY c.name
        """
        rewritten = apply_row_policies(
            "SELECT c.name, o.id, o.amount FROM customers c LEFT JOIN orders o ON c.id = o.customer_id ORDER BY c.name",
            {"orders": "tenant_id = ?"},
        ).sql()

        expected = conn.execute(hand_written, (1,)).fetchall()
        actual = conn.execute(rewritten, (1,)).fetchall()

        self.assertEqual(actual, expected)
        self.assertEqual(actual, [("a", 1, 10.0), ("b", None, None)])

    def test_sqlite_star_and_using_columns(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER, name TEXT);
            CREATE TABLE orders (id INTEGER, tenant_id INTEGER, amount REAL);
            INSERT INTO customers VALUES (1, 'a'), (2, 'b');
            INSERT INTO orders VALUES (1, 1, 10.0), (2, 2, 20.0);
            """
        )

        for sql in (
            "SELECT * FROM customers c LEFT JOIN orders o ON c.id = o.id ORDER BY c.id",
            "SELECT * FROM customers LEFT JOIN orders USING (id) ORDER BY id",
        ):
            rewritten = apply_row_policies(sql, {"orders": "tenant_id = ?"}).sql()

            expected_cursor = conn.execute(
                sql.replace("orders", "(SELECT * FROM orders WHERE tenant_id = 1)", 1)
            )
            expected_columns = [d[0] for d in expected_cursor.description]
            expected_rows = expected_cursor.fetchall()

            actual_cursor = conn.execute(rewritten, (1,))
            self.assertEqual([d[0] for d in actual_cursor.description], expected_columns)
            self.assertEqual(actual_cursor.fetchall(), expected_rows)
