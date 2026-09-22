from __future__ import annotations

import unittest

import sqlglot
from sqlglot import exp
from sqlglot.archive import ArchiveFilterError, apply_archive_filters

try:
    import duckdb
except ImportError:
    duckdb = None


class TestApplyArchiveFilters(unittest.TestCase):
    maxDiff = None

    def rewrite(self, sql, filters, dialect=None):
        return apply_archive_filters(
            sqlglot.parse_one(sql, read=dialect), filters, dialect=dialect
        ).sql()

    def test_outer_join_filters_stay_at_scan_site(self):
        self.assertEqual(
            self.rewrite(
                "SELECT * FROM db.orders o LEFT JOIN db.customers c ON c.id = o.cid",
                {
                    "db.orders": "archived_at <= :cutoff",
                    "db.customers": "archived_at <= :cutoff",
                },
            ),
            "SELECT * FROM "
            "(SELECT * FROM db.orders WHERE archived_at <= :cutoff) AS o "
            "LEFT JOIN (SELECT * FROM db.customers WHERE archived_at <= :cutoff) AS c "
            "ON c.id = o.cid",
        )

    def test_scan_without_alias_uses_table_name(self):
        self.assertEqual(
            self.rewrite("SELECT * FROM db.orders", {"db.orders": "id > 0"}),
            "SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS orders",
        )

    def test_self_join_with_distinct_aliases(self):
        self.assertEqual(
            self.rewrite(
                "SELECT * FROM db.orders a JOIN db.orders b ON a.id = b.id",
                {"db.orders": "id > 0"},
            ),
            "SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS a "
            "JOIN (SELECT * FROM db.orders WHERE id > 0) AS b ON a.id = b.id",
        )

    def test_full_identity_matching(self):
        condition = {"db.orders": "id > 0"}
        self.assertEqual(self.rewrite("SELECT * FROM orders", condition), "SELECT * FROM orders")
        self.assertEqual(
            self.rewrite("SELECT * FROM other.orders", condition),
            "SELECT * FROM other.orders",
        )
        self.assertEqual(
            self.rewrite("SELECT * FROM DB.ORDERS", condition),
            "SELECT * FROM (SELECT * FROM DB.ORDERS WHERE id > 0) AS ORDERS",
        )
        self.assertEqual(
            self.rewrite('SELECT * FROM "DB".orders', condition),
            'SELECT * FROM "DB".orders',
        )
        self.assertEqual(
            self.rewrite("SELECT * FROM catalog.db.orders", {"catalog.db.orders": "id > 0"}),
            "SELECT * FROM (SELECT * FROM catalog.db.orders WHERE id > 0) AS orders",
        )

    def test_cte_shadowing_base_table(self):
        self.assertEqual(
            self.rewrite(
                "WITH customers AS (SELECT * FROM db.customers) "
                "SELECT * FROM customers c JOIN db.orders o ON o.cid = c.id",
                {"db.customers": "id > 0", "db.orders": "id > 0"},
            ),
            "WITH customers AS (SELECT * FROM (SELECT * FROM db.customers WHERE id > 0) AS customers) "
            "SELECT * FROM customers AS c "
            "JOIN (SELECT * FROM db.orders WHERE id > 0) AS o ON o.cid = c.id",
        )

    def test_derived_table_shadowing_base_table(self):
        self.assertEqual(
            self.rewrite(
                "SELECT * FROM (SELECT * FROM db.orders) orders",
                {"db.orders": "id > 0"},
            ),
            "SELECT * FROM (SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS orders) AS orders",
        )

    def test_union_branches(self):
        self.assertEqual(
            self.rewrite(
                "SELECT id FROM db.a UNION ALL SELECT id FROM db.b",
                {"db.a": "id > 0", "db.b": "id > 1"},
            ),
            "SELECT id FROM (SELECT * FROM db.a WHERE id > 0) AS a "
            "UNION ALL SELECT id FROM (SELECT * FROM db.b WHERE id > 1) AS b",
        )

    def test_correlated_subquery(self):
        self.assertEqual(
            self.rewrite(
                "SELECT * FROM db.orders o "
                "WHERE EXISTS (SELECT 1 FROM db.shipments s WHERE s.order_id = o.id)",
                {"db.orders": "id > 0", "db.shipments": "id > 1"},
            ),
            "SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS o "
            "WHERE EXISTS(SELECT 1 FROM (SELECT * FROM db.shipments WHERE id > 1) AS s "
            "WHERE s.order_id = o.id)",
        )

    def test_expression_filters_and_table_keys(self):
        self.assertEqual(
            self.rewrite(
                "SELECT * FROM db.orders o",
                {exp.to_table("db.orders"): exp.condition("id > 0")},
            ),
            "SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS o",
        )

    def test_placeholder_preserved(self):
        rewritten = apply_archive_filters(
            sqlglot.parse_one("SELECT * FROM db.orders", read="duckdb"),
            {"db.orders": "archived_at <= :cutoff"},
            dialect="duckdb",
        )
        self.assertEqual(
            rewritten.sql("duckdb"),
            "SELECT * FROM (SELECT * FROM db.orders WHERE archived_at <= $cutoff) AS orders",
        )

    def test_input_expression_not_mutated(self):
        condition = exp.condition("id > 0")
        query = sqlglot.parse_one("SELECT * FROM db.orders a JOIN db.orders b ON a.id = b.id")
        sql_before = query.sql()

        rewritten = apply_archive_filters(query, {"db.orders": condition})

        self.assertEqual(query.sql(), sql_before)
        self.assertEqual(condition.sql(), "id > 0")
        self.assertIsNone(condition.parent)
        self.assertNotEqual(rewritten.sql(), sql_before)

    def test_copy_false_rewrites_in_place(self):
        query = sqlglot.parse_one("SELECT * FROM db.orders")
        rewritten = apply_archive_filters(query, {"db.orders": "id > 0"}, copy=False)
        self.assertIs(rewritten, query)
        self.assertEqual(
            query.sql(), "SELECT * FROM (SELECT * FROM db.orders WHERE id > 0) AS orders"
        )

    def test_known_table_function_is_skipped(self):
        self.assertEqual(
            self.rewrite("SELECT * FROM read_csv('data.csv')", {}, dialect="duckdb"),
            "SELECT * FROM READ_CSV('data.csv')",
        )

    def test_rejects_dml(self):
        for sql in (
            "INSERT INTO db.t SELECT * FROM db.s",
            "UPDATE db.t SET x = 1",
            "DELETE FROM db.t WHERE x = 1",
            "MERGE INTO db.t USING db.s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = 1",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ArchiveFilterError):
                    apply_archive_filters(sqlglot.parse_one(sql), {"db.t": "id > 0"})

    def test_rejects_recursive_cte(self):
        with self.assertRaises(ArchiveFilterError):
            apply_archive_filters(
                sqlglot.parse_one(
                    "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) "
                    "SELECT * FROM t"
                ),
                {"db.t": "id > 0"},
            )

    def test_rejects_unknown_table_function(self):
        with self.assertRaises(ArchiveFilterError):
            apply_archive_filters(
                sqlglot.parse_one("SELECT * FROM mystery(1)", read="duckdb"),
                {},
                dialect="duckdb",
            )

    def test_rejects_non_query(self):
        with self.assertRaises(ArchiveFilterError):
            apply_archive_filters(sqlglot.parse_one("CREATE TABLE t AS SELECT 1"), {})


@unittest.skipUnless(duckdb, "duckdb is not installed")
class TestApplyArchiveFiltersDuckDB(unittest.TestCase):
    CUTOFF = "2024-06-01"

    @classmethod
    def setUpClass(cls):
        cls.conn = duckdb.connect()
        cls.conn.execute("CREATE SCHEMA raw")
        cls.conn.execute("CREATE SCHEMA filt")
        cls.conn.execute(
            "CREATE TABLE raw.orders (order_id INTEGER, cust_id INTEGER, archived_at DATE)"
        )
        cls.conn.execute(
            "CREATE TABLE raw.customers (cust_id INTEGER, name VARCHAR, archived_at DATE)"
        )
        cls.conn.execute(
            "CREATE TABLE raw.shipments "
            "(ship_id INTEGER, order_id INTEGER, status VARCHAR, archived_at DATE)"
        )
        cls.conn.execute(
            "INSERT INTO raw.orders VALUES "
            "(1, 10, '2024-01-01'), (2, 20, '2024-01-01'), (3, 30, '2024-07-01'), "
            "(4, NULL, '2024-01-01'), (5, 10, '2024-01-01')"
        )
        cls.conn.execute(
            "INSERT INTO raw.customers VALUES "
            "(10, 'alice', '2024-01-01'), (20, 'bob', '2024-07-01'), "
            "(40, 'carol', '2024-01-01')"
        )
        cls.conn.execute(
            "INSERT INTO raw.shipments VALUES "
            "(100, 1, 'shipped', '2024-01-01'), (101, 1, 'lost', '2024-07-01'), "
            "(102, 2, 'shipped', '2024-01-01'), (103, 999, 'shipped', '2024-01-01'), "
            "(104, 3, 'shipped', '2024-01-01')"
        )
        for table in ("orders", "customers", "shipments"):
            cls.conn.execute(
                f"CREATE VIEW filt.{table} AS "
                f"SELECT * FROM raw.{table} WHERE archived_at <= DATE '{cls.CUTOFF}'"
            )

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def assert_same_as_filtered_views(self, query):
        expected = self.conn.execute(query.format(schema="filt")).fetchall()
        rewritten = apply_archive_filters(
            sqlglot.parse_one(query.format(schema="raw"), read="duckdb"),
            {
                "raw.orders": "archived_at <= :cutoff",
                "raw.customers": "archived_at <= :cutoff",
                "raw.shipments": "archived_at <= :cutoff",
            },
            dialect="duckdb",
        )
        actual = self.conn.execute(rewritten.sql("duckdb"), {"cutoff": self.CUTOFF}).fetchall()

        def sort_key(row):
            return tuple((value is None, str(value)) for value in row)

        self.assertEqual(sorted(expected, key=sort_key), sorted(actual, key=sort_key))

    def test_outer_joins_and_alias_shadowing(self):
        self.assert_same_as_filtered_views(
            """
            WITH customers AS (
                SELECT cust_id, name FROM {schema}.customers
            )
            SELECT o.order_id, c.name, s.status
            FROM {schema}.orders AS o
            LEFT JOIN customers AS c ON c.cust_id = o.cust_id
            FULL JOIN {schema}.shipments AS s ON s.order_id = o.order_id
            """
        )

    def test_union_correlated_subquery_and_self_join(self):
        self.assert_same_as_filtered_views(
            """
            SELECT o1.order_id, o2.order_id AS other_id
            FROM {schema}.orders AS o1
            JOIN {schema}.orders AS o2 ON o2.order_id = o1.order_id + 100
            WHERE EXISTS (
                SELECT 1 FROM {schema}.shipments AS s WHERE s.order_id = o1.order_id
            )
            UNION ALL
            SELECT order_id, NULL AS other_id FROM {schema}.orders WHERE cust_id = 10
            """
        )


if __name__ == "__main__":
    unittest.main()
