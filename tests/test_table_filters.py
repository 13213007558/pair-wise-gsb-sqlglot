import unittest

from sqlglot import apply_table_filters, exp, parse_one
from sqlglot.errors import UnsupportedError

try:
    import duckdb
except ImportError:
    duckdb = None


class TestApplyTableFilters(unittest.TestCase):
    def test_simple_select(self):
        rewritten = apply_table_filters(
            parse_one("SELECT id FROM orders"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT id FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS orders",
        )

    def test_alias_preserved(self):
        rewritten = apply_table_filters(
            parse_one("SELECT o.id FROM orders o"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT o.id FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS o",
        )

    def test_outer_join_filter_applies_per_scan(self):
        rewritten = apply_table_filters(
            parse_one("SELECT o.id, c.name FROM orders o LEFT JOIN customers c ON c.id = o.cid"),
            {"orders": "archived_at <= :cutoff", "customers": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT o.id, c.name "
            "FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS o "
            "LEFT JOIN (SELECT * FROM customers WHERE archived_at <= :cutoff) AS c "
            "ON c.id = o.cid",
        )
        # the predicate must not leak into the outer WHERE, which would drop unmatched rows
        self.assertIsNone(rewritten.args.get("where"))

    def test_same_table_multiple_aliases(self):
        rewritten = apply_table_filters(
            parse_one("SELECT a.id, b.id FROM orders a JOIN orders b ON a.cid = b.cid"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT a.id, b.id "
            "FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS a "
            "JOIN (SELECT * FROM orders WHERE archived_at <= :cutoff) AS b "
            "ON a.cid = b.cid",
        )

    def test_cte_shadowing(self):
        rewritten = apply_table_filters(
            parse_one("WITH orders AS (SELECT * FROM other) SELECT * FROM orders"),
            {"orders": "archived_at <= :cutoff", "other": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "WITH orders AS (SELECT * FROM (SELECT * FROM other WHERE archived_at <= :cutoff) AS other) "
            "SELECT * FROM orders",
        )

    def test_base_table_inside_cte_definition(self):
        rewritten = apply_table_filters(
            parse_one("WITH recent AS (SELECT * FROM orders WHERE id > 10) SELECT * FROM recent"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "WITH recent AS "
            "(SELECT * FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS orders WHERE id > 10) "
            "SELECT * FROM recent",
        )

    def test_union(self):
        rewritten = apply_table_filters(
            parse_one("SELECT id FROM orders UNION ALL SELECT id FROM orders"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT id FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS orders "
            "UNION ALL "
            "SELECT id FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS orders",
        )

    def test_correlated_subquery(self):
        rewritten = apply_table_filters(
            parse_one(
                "SELECT * FROM orders o WHERE EXISTS "
                "(SELECT 1 FROM customers c WHERE c.id = o.cid)"
            ),
            {"orders": "archived_at <= :cutoff", "customers": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT * FROM (SELECT * FROM orders WHERE archived_at <= :cutoff) AS o "
            "WHERE EXISTS("
            "SELECT 1 FROM (SELECT * FROM customers WHERE archived_at <= :cutoff) AS c "
            "WHERE c.id = o.cid"
            ")",
        )

    def test_full_identity_matching(self):
        rewritten = apply_table_filters(
            parse_one("SELECT * FROM db.orders, other.orders"),
            {"db.orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT * FROM (SELECT * FROM db.orders WHERE archived_at <= :cutoff) AS orders, "
            "other.orders",
        )

    def test_unqualified_filter_matches_qualified_table(self):
        rewritten = apply_table_filters(
            parse_one("SELECT * FROM db.orders"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual(
            rewritten.sql(),
            "SELECT * FROM (SELECT * FROM db.orders WHERE archived_at <= :cutoff) AS orders",
        )

    def test_input_not_mutated(self):
        expression = parse_one("SELECT id FROM orders o")
        before = expression.sql()
        apply_table_filters(expression, {"orders": "archived_at <= :cutoff"})
        self.assertEqual(expression.sql(), before)

    def test_placeholders_preserved(self):
        rewritten = apply_table_filters(
            parse_one("SELECT id FROM orders"),
            {"orders": "archived_at <= :cutoff"},
        )
        self.assertEqual([p.name for p in rewritten.find_all(exp.Placeholder)], ["cutoff"])

    def test_expression_condition(self):
        condition = exp.condition("archived_at <= :cutoff").and_("id > 0")
        rewritten = apply_table_filters(parse_one("SELECT id FROM orders"), {"orders": condition})
        self.assertEqual(
            rewritten.sql(),
            "SELECT id FROM (SELECT * FROM orders WHERE archived_at <= :cutoff AND id > 0) AS orders",
        )

    def test_empty_filters(self):
        expression = parse_one("SELECT id FROM orders")
        self.assertEqual(apply_table_filters(expression, {}).sql(), expression.sql())

    def test_rejects_dml(self):
        for sql in (
            "DELETE FROM orders WHERE id = 1",
            "UPDATE orders SET id = 2",
            "INSERT INTO orders SELECT * FROM other",
            "MERGE INTO orders USING other ON orders.id = other.id WHEN MATCHED THEN DELETE",
        ):
            with self.assertRaises(UnsupportedError, msg=sql):
                apply_table_filters(parse_one(sql), {"orders": "id > 0"})

    def test_rejects_recursive_cte(self):
        with self.assertRaises(UnsupportedError):
            apply_table_filters(
                parse_one("WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r"),
                {"orders": "id > 0"},
            )

    def test_rejects_table_functions(self):
        with self.assertRaises(UnsupportedError):
            apply_table_filters(
                parse_one("SELECT * FROM read_csv('x.csv')", read="duckdb"),
                {"orders": "id > 0"},
                dialect="duckdb",
            )
        with self.assertRaises(UnsupportedError):
            apply_table_filters(
                parse_one("SELECT * FROM UNNEST([1, 2]) AS u", read="duckdb"),
                {"orders": "id > 0"},
                dialect="duckdb",
            )


@unittest.skipUnless(duckdb, "duckdb is not installed")
class TestApplyTableFiltersDuckDB(unittest.TestCase):
    CUTOFF = "2024-06-01"

    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, archived_at DATE)")
        self.con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, archived_at DATE)")
        self.con.execute(
            "CREATE TABLE events (id INTEGER, order_id INTEGER, customer_id INTEGER, "
            "kind VARCHAR, value INTEGER, archived_at DATE)"
        )
        self.con.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [
                (1, 10, "2024-01-01"),
                (2, 20, "2024-01-01"),
                (3, 30, "2024-07-01"),
                (4, 40, "2024-01-01"),
            ],
        )
        self.con.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [
                (10, "alice", "2024-01-01"),
                (20, "bob", "2024-07-01"),
                (50, "carol", "2024-01-01"),
            ],
        )
        self.con.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 10, "purchase", 100, "2024-01-01"),
                (2, 1, 10, "refund", 5, "2024-01-01"),
                (3, 2, 20, "purchase", 200, "2024-07-01"),
                (4, 9, 50, "purchase", 300, "2024-01-01"),
            ],
        )
        for table in ("orders", "customers", "events"):
            self.con.execute(
                f"CREATE VIEW {table}_v AS "
                f"SELECT * FROM {table} WHERE archived_at <= DATE '{self.CUTOFF}'"
            )

    def tearDown(self):
        self.con.close()

    def _expected_sql(self, sql):
        def rename(node):
            if isinstance(node, exp.Table) and node.name + "_v" in VIEWS:
                node.set("this", exp.to_identifier(node.name + "_v"))
            return node

        return parse_one(sql, dialect="duckdb").transform(rename).sql("duckdb")

    def _assert_equivalent(self, sql):
        filters = {table: "archived_at <= ?" for table in ("orders", "customers", "events")}
        rewritten_ast = apply_table_filters(
            parse_one(sql, dialect="duckdb"), filters, dialect="duckdb"
        )
        rewritten = rewritten_ast.sql("duckdb")
        params = [self.CUTOFF] * len(list(rewritten_ast.find_all(exp.Placeholder)))

        expected = self.con.execute(self._expected_sql(sql)).fetchall()
        actual = self.con.execute(rewritten, params).fetchall()
        self.assertEqual(sorted(expected, key=str), sorted(actual, key=str), msg=rewritten)

    def test_outer_joins_and_alias_shadowing(self):
        self._assert_equivalent(
            """
            WITH recent AS (
                SELECT * FROM events WHERE kind = 'purchase'
            )
            SELECT o.id, c.name, r.value
            FROM orders o
            LEFT JOIN customers c ON c.id = o.customer_id
            FULL JOIN recent r ON r.order_id = o.id
            WHERE EXISTS (
                SELECT 1 FROM events o
                WHERE o.customer_id = c.id AND o.kind = 'refund'
            )
            """
        )

    def test_self_join_and_union(self):
        self._assert_equivalent(
            """
            SELECT a.id FROM orders a
            JOIN orders b ON a.customer_id = b.customer_id AND a.id <> b.id
            UNION ALL
            SELECT id FROM orders
            """
        )


VIEWS = {"orders_v", "customers_v", "events_v"}

if __name__ == "__main__":
    unittest.main()
