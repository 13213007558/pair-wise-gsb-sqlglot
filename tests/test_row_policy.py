import sqlite3
import unittest

from sqlglot import exp, parse_one
from sqlglot.errors import OptimizeError
from sqlglot.optimizer import apply_row_policies


class TestRowPolicy(unittest.TestCase):
    def test_wraps_table_reference(self):
        self.assertEqual(
            apply_row_policies("SELECT * FROM orders", {"orders": "tenant_id = 1"}).sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS orders",
        )

    def test_left_join_not_converted_to_inner(self):
        self.assertEqual(
            apply_row_policies(
                "SELECT * FROM customers c LEFT JOIN orders o ON c.id = o.customer_id",
                {"orders": "tenant_id = 1"},
            ).sql(),
            "SELECT * FROM customers AS c LEFT JOIN "
            "(SELECT * FROM orders WHERE tenant_id = 1) AS o ON c.id = o.customer_id",
        )

    def test_self_join(self):
        self.assertEqual(
            apply_row_policies(
                "SELECT * FROM orders a JOIN orders b ON a.id = b.id",
                {"orders": "tenant_id = 1"},
            ).sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS a "
            "JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS b ON a.id = b.id",
        )

    def test_cte_shadowing_physical_table(self):
        # The physical reference inside the CTE body is wrapped, but the outer
        # reference resolves to the CTE and must be left alone
        self.assertEqual(
            apply_row_policies(
                "WITH orders AS (SELECT * FROM orders) SELECT * FROM orders",
                {"orders": "tenant_id = 1"},
            ).sql(),
            "WITH orders AS (SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1) AS orders) "
            "SELECT * FROM orders",
        )

    def test_correlated_subquery(self):
        self.assertEqual(
            apply_row_policies(
                "SELECT * FROM customers c "
                "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
                {"orders": "tenant_id = 1"},
            ).sql(),
            "SELECT * FROM customers AS c WHERE EXISTS("
            "SELECT 1 FROM (SELECT * FROM orders WHERE tenant_id = 1) AS o "
            "WHERE o.customer_id = c.id)",
        )

    def test_set_operation_branches(self):
        self.assertEqual(
            apply_row_policies(
                "SELECT id FROM orders o UNION ALL SELECT id FROM orders p",
                {"orders": "tenant_id = 1"},
            ).sql(),
            "SELECT id FROM (SELECT * FROM orders WHERE tenant_id = 1) AS o UNION ALL "
            "SELECT id FROM (SELECT * FROM orders WHERE tenant_id = 1) AS p",
        )

    def test_schema_qualified_reference(self):
        # Policies are keyed by full physical table name: "public.orders" matches
        # the qualified reference, but a policy on "orders" must not
        expression = apply_row_policies(
            "SELECT * FROM public.orders o",
            {"public.orders": "tenant_id = 1", "orders": "tenant_id = 2"},
        )
        self.assertEqual(
            expression.sql(),
            "SELECT * FROM (SELECT * FROM public.orders WHERE tenant_id = 1) AS o",
        )

    def test_policy_expression_and_ast_input(self):
        policy = exp.and_(exp.column("tenant_id").eq(1), exp.column("status").neq("closed"))
        query = parse_one("SELECT * FROM orders o")

        self.assertEqual(
            apply_row_policies(query, {"orders": policy}).sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = 1 AND status <> 'closed') AS o",
        )
        # The caller's trees are left untouched
        self.assertEqual(query.sql(), "SELECT * FROM orders AS o")
        self.assertEqual(policy.sql(), "tenant_id = 1 AND status <> 'closed'")

    def test_parameters_are_left_unbound(self):
        expression = apply_row_policies(
            "SELECT * FROM orders",
            {"orders": "tenant_id = @tenant_id AND status = :status"},
        )
        self.assertEqual(
            expression.sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE tenant_id = @tenant_id AND status = :status) "
            "AS orders",
        )
        self.assertIsNotNone(expression.find(exp.Parameter))
        self.assertIsNotNone(expression.find(exp.Placeholder))

    def test_empty_policies(self):
        self.assertEqual(
            apply_row_policies("SELECT * FROM orders", {}).sql(), "SELECT * FROM orders"
        )

    def test_rejects_non_select(self):
        for sql in (
            "DELETE FROM orders",
            "UPDATE orders SET id = 1",
            "INSERT INTO orders VALUES (1)",
        ):
            with self.assertRaises(OptimizeError):
                apply_row_policies(sql, {"orders": "tenant_id = 1"})

    def test_rejects_unsupported_table_constructs(self):
        with self.assertRaises(OptimizeError):
            apply_row_policies("SELECT * FROM orders TABLESAMPLE (10)", {"orders": "tenant_id = 1"})

    def test_rejects_duplicate_aliases(self):
        with self.assertRaises(OptimizeError):
            apply_row_policies("SELECT * FROM orders, orders", {"orders": "tenant_id = 1"})

    def test_trees_are_independent(self):
        query = parse_one("SELECT * FROM orders o")
        transformed = apply_row_policies(query, {"orders": "tenant_id = 1"})

        transformed.find(exp.Table).set("this", exp.to_identifier("mutated"))
        self.assertEqual(query.sql(), "SELECT * FROM orders AS o")


class TestRowPolicySQLite(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(
            """
            CREATE TABLE customers (id INTEGER, name TEXT, tenant_id INTEGER);
            CREATE TABLE orders (id INTEGER, customer_id INTEGER, tenant_id INTEGER, amount REAL);
            INSERT INTO customers VALUES (1, 'ann', 1), (2, 'bob', 1), (3, 'cat', 2);
            -- only the other tenant (2) has an order, and it belongs to customer 3
            INSERT INTO orders VALUES (10, 3, 2, 100.0);
            """
        )

    def tearDown(self):
        self.db.close()

    def run_query(self, sql, *params):
        cursor = self.db.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        return columns, cursor.fetchall()

    def test_left_join_semantics_against_handwritten_query(self):
        policies = {"customers": "tenant_id = ?", "orders": "tenant_id = ?"}
        sql = (
            "SELECT c.id, c.name, o.amount FROM customers c "
            "LEFT JOIN orders o ON c.id = o.customer_id"
        )
        handwritten = (
            "SELECT c.id, c.name, o.amount FROM (SELECT * FROM customers WHERE tenant_id = ?) AS c "
            "LEFT JOIN (SELECT * FROM orders WHERE tenant_id = ?) AS o ON c.id = o.customer_id"
        )

        transformed = apply_row_policies(sql, policies).sql()
        expected_columns, expected_rows = self.run_query(handwritten, 1, 1)
        actual_columns, actual_rows = self.run_query(transformed, 1, 1)

        self.assertEqual(actual_columns, expected_columns)
        self.assertEqual(actual_rows, expected_rows)
        # customers of the current tenant are kept even though no order matches
        self.assertEqual(actual_rows, [(1, "ann", None), (2, "bob", None)])

    def test_select_star_and_using_preserve_output(self):
        # every row belongs to the current tenant, so the rewrite must not
        # change the result of SELECT * with a USING merge column
        self.db.execute("DELETE FROM customers WHERE tenant_id <> 1")
        self.db.execute("DELETE FROM orders WHERE tenant_id <> 1")
        self.db.execute("INSERT INTO orders VALUES (11, 1, 1, 42.0)")

        sql = "SELECT * FROM customers JOIN orders USING (tenant_id)"
        policies = {"customers": "tenant_id = ?", "orders": "tenant_id = ?"}

        expected_columns, expected_rows = self.run_query(sql)
        actual_columns, actual_rows = self.run_query(apply_row_policies(sql, policies).sql(), 1, 1)

        self.assertEqual(actual_columns, expected_columns)
        self.assertEqual(actual_rows, expected_rows)
