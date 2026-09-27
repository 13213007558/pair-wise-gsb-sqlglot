import sqlite3
import unittest

from sqlglot.optimizer.nullability import (
    ColumnNullability,
    Nullability,
    infer_nullability,
)


NN = Nullability.NOT_NULL
N = Nullability.NULLABLE
U = Nullability.UNKNOWN


class TestNullabilityInference(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "users": {"id": False, "name": True, "age": "unknown"},
            "orders": {"id": False, "user_id": False, "note": True},
        }

    def infer(self, sql, **kwargs):
        return infer_nullability(sql, self.schema, dialect="sqlite", **kwargs)

    def assert_status(self, results, expected):
        self.assertEqual(
            [(column.name, column.nullability) for column in results],
            [(name, status) for name, status in expected],
        )

    def assert_reasons(self, results):
        for column in results:
            self.assertTrue(column.reason, f"{column.name} has no reason")
            self.assertIsInstance(column, ColumnNullability)

    def test_plain_select(self):
        results = self.infer("SELECT id, name, age FROM users")
        self.assert_status(results, [("id", NN), ("name", N), ("age", U)])
        self.assert_reasons(results)

    def test_missing_info_is_unknown(self):
        results = self.infer("SELECT id, mystery FROM users")
        self.assertEqual(results[1].nullability, U)

        results = infer_nullability("SELECT id FROM unlisted_table", self.schema, dialect="sqlite")
        self.assertEqual([(c.name, c.nullability) for c in results], [("id", U)])

    def test_star_expansion_preserves_order(self):
        results = self.infer("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
        self.assertEqual(
            [column.name for column in results],
            ["id", "name", "age", "id", "user_id", "note"],
        )

        results = self.infer("SELECT o.*, u.name FROM users u JOIN orders o ON u.id = o.user_id")
        self.assertEqual([column.name for column in results], ["id", "user_id", "note", "name"])

    def test_inner_join_keeps_not_null(self):
        results = self.infer(
            "SELECT u.id AS uid, o.id AS oid FROM users u INNER JOIN orders o ON u.id = o.user_id"
        )
        self.assert_status(results, [("uid", NN), ("oid", NN)])

    def test_outer_joins_nullify_sources(self):
        left = self.infer(
            "SELECT u.id AS uid, o.id AS oid FROM users u LEFT JOIN orders o ON u.id = o.user_id"
        )
        self.assert_status(left, [("uid", NN), ("oid", N)])
        self.assertIn("outer join", left[1].reason)

        right = self.infer(
            "SELECT u.id AS uid, o.id AS oid FROM users u RIGHT JOIN orders o ON u.id = o.user_id"
        )
        self.assert_status(right, [("uid", N), ("oid", NN)])

        full = self.infer(
            "SELECT u.id AS uid, o.id AS oid FROM users u FULL JOIN orders o ON u.id = o.user_id"
        )
        self.assert_status(full, [("uid", N), ("oid", N)])

    def test_where_is_not_null_narrows(self):
        results = self.infer(
            "SELECT o.id AS oid FROM users u "
            "LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NOT NULL"
        )
        self.assert_status(results, [("oid", NN)])
        self.assertIn("IS NOT NULL", results[0].reason)

        results = self.infer("SELECT name FROM users WHERE name IS NOT NULL AND id > 0")
        self.assert_status(results, [("name", NN)])

    def test_case_coalesce_nullif(self):
        results = self.infer(
            "SELECT "
            "COALESCE(name, 'x') AS a, "
            "COALESCE(name, age) AS b, "
            "COALESCE(name, note) AS c, "
            "NULLIF(id, id) AS d, "
            "CASE WHEN id > 0 THEN name ELSE 'x' END AS e, "
            "CASE WHEN id > 0 THEN id END AS f "
            "FROM users"
        )
        self.assert_status(
            results,
            [("a", NN), ("b", U), ("c", U), ("d", N), ("e", N), ("f", N)],
        )
        self.assertIn("no ELSE", results[5].reason)

    def test_count_and_sum(self):
        no_group = self.infer("SELECT COUNT(id) AS c, COUNT(*) AS c2, SUM(id) AS s FROM users")
        self.assert_status(no_group, [("c", NN), ("c2", NN), ("s", N)])
        self.assertIn("empty input", no_group[2].reason)

        grouped = self.infer("SELECT id, COUNT(*) AS c, SUM(id) AS s FROM users GROUP BY id")
        self.assert_status(grouped, [("id", NN), ("c", NN), ("s", NN)])

        nullable_sum = self.infer("SELECT SUM(name) AS s FROM users")
        self.assert_status(nullable_sum, [("s", N)])

    def test_cte_and_derived_table(self):
        results = self.infer("WITH c AS (SELECT id, name FROM users) SELECT id, name FROM c")
        self.assert_status(results, [("id", NN), ("name", N)])

        results = self.infer("SELECT x.id FROM (SELECT id FROM users) AS x")
        self.assert_status(results, [("id", NN)])

        results = self.infer("SELECT c1 FROM (SELECT id FROM users) AS x(c1)")
        self.assert_status(results, [("c1", NN)])

    def test_union_all(self):
        not_null = self.infer("SELECT id FROM users UNION ALL SELECT id FROM orders")
        self.assert_status(not_null, [("id", NN)])

        nullable = self.infer("SELECT id FROM users UNION ALL SELECT note FROM orders")
        self.assert_status(nullable, [("id", N)])
        self.assertIn("branch 2", nullable[0].reason)

        three_branches = self.infer(
            "SELECT id FROM users UNION ALL SELECT id FROM orders UNION ALL SELECT age FROM users"
        )
        self.assert_status(three_branches, [("id", U)])
        self.assertIn("unknown", three_branches[0].reason)

    def test_function_contracts(self):
        results = self.infer("SELECT MY_FN(id) FROM users")
        self.assert_status(results, [("MY_FN", U)])

        results = self.infer(
            "SELECT MY_FN(id) AS x FROM users",
            function_contracts={"MY_FN": "not_null"},
        )
        self.assert_status(results, [("x", NN)])

        results = self.infer(
            "SELECT nullable_fn(id) AS x FROM users", function_contracts={"NULLABLE_FN": True}
        )
        self.assert_status(results, [("x", N)])

    def test_literals_and_arithmetic(self):
        results = self.infer("SELECT 1 AS a, NULL AS b, id + 1 AS c, name || 'x' AS d FROM users")
        self.assert_status(results, [("a", NN), ("b", N), ("c", NN), ("d", N)])

    def test_returns_column_nullability_objects(self):
        results = self.infer("SELECT id FROM users")
        self.assertEqual(
            results[0],
            ColumnNullability("id", NN, results[0].reason),
        )


class TestNullabilityAgainstSQLite(unittest.TestCase):
    """
    Adversarial fixtures (empty tables, all-NULL data, no-match outer joins):
    a NULLABLE prediction must actually observe a NULL result, and every NOT_NULL
    prediction must hold against the actual result set.
    """

    schema = {
        "users": {"id": False, "name": True, "age": True},
        "orders": {"id": False, "user_id": False, "note": True},
    }

    def execute(self, sql, setup):
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(setup)
            cursor = connection.execute(sql)
            names = [description[0] for description in cursor.description]
            rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        finally:
            connection.close()
        return names, rows

    def assert_matches_results(self, sql, setup):
        predictions = infer_nullability(sql, self.schema, dialect="sqlite")
        names, rows = self.execute(sql, setup)
        self.assertEqual(
            [column.name for column in predictions],
            names,
        )
        for column in predictions:
            values = [row[column.name] for row in rows]
            if column.nullability is NN:
                self.assertTrue(
                    all(value is not None for value in values),
                    f"{column.name} predicted NOT NULL but contained NULLs: {values}",
                )
            if column.nullability is N and rows:
                self.assertTrue(
                    any(value is None for value in values),
                    f"{column.name} predicted NULLABLE but no NULL observed: {values}",
                )

    def test_empty_input_aggregate_without_group_by(self):
        setup = "CREATE TABLE users (id INTEGER NOT NULL, name TEXT);\n"
        sql = "SELECT COUNT(id) AS c, SUM(id) AS s FROM users"
        names, rows = self.execute(sql, setup)
        self.assertEqual(rows, [{"c": 0, "s": None}])
        self.assert_matches_results(sql, setup)

    def test_group_by_on_empty_table_produces_no_rows(self):
        setup = "CREATE TABLE users (id INTEGER NOT NULL, name TEXT);\n"
        sql = "SELECT id, COUNT(*) AS c, SUM(id) AS s FROM users GROUP BY id"
        names, rows = self.execute(sql, setup)
        self.assertEqual(rows, [])
        predictions = infer_nullability(sql, self.schema, dialect="sqlite")
        self.assertEqual(
            [(c.name, c.nullability) for c in predictions],
            [("id", NN), ("c", NN), ("s", NN)],
        )

    def test_all_null_columns(self):
        setup = (
            "CREATE TABLE users (id INTEGER NOT NULL, name TEXT, age INTEGER);\n"
            "INSERT INTO users VALUES (1, NULL, NULL);\n"
        )
        sql = (
            "SELECT COALESCE(name, age) AS a, COALESCE(name, 1) AS b, "
            "NULLIF(id, id) AS c FROM users"
        )
        names, rows = self.execute(sql, setup)
        self.assertTrue(all(row["a"] is None for row in rows))
        self.assertTrue(all(row["b"] == 1 for row in rows))
        self.assertTrue(all(row["c"] is None for row in rows))
        self.assert_matches_results(sql, setup)

    def test_no_match_left_join(self):
        setup = (
            "CREATE TABLE users (id INTEGER NOT NULL, name TEXT);\n"
            "CREATE TABLE orders (id INTEGER NOT NULL, user_id INTEGER NOT NULL, note TEXT);\n"
            "INSERT INTO users VALUES (1, NULL);\n"
            "INSERT INTO users VALUES (2, 'bob');\n"
        )
        sql = (
            "SELECT u.id AS uid, o.id AS oid, o.note AS note "
            "FROM users u LEFT JOIN orders o ON u.id = o.user_id"
        )
        names, rows = self.execute(sql, setup)
        self.assertTrue(all(row["uid"] is not None for row in rows))
        self.assertTrue(all(row["oid"] is None for row in rows))
        self.assertTrue(all(row["note"] is None for row in rows))
        self.assert_matches_results(sql, setup)

    def test_no_match_right_join(self):
        setup = (
            "CREATE TABLE users (id INTEGER NOT NULL, name TEXT);\n"
            "CREATE TABLE orders (id INTEGER NOT NULL, user_id INTEGER NOT NULL, note TEXT);\n"
            "INSERT INTO orders VALUES (10, 777, 'x');\n"
        )
        sql = (
            "SELECT u.id AS uid, o.id AS oid, o.user_id AS user_id "
            "FROM users u RIGHT JOIN orders o ON u.id = o.user_id"
        )
        names, rows = self.execute(sql, setup)
        self.assertEqual(rows, [{"uid": None, "oid": 10, "user_id": 777}])
        self.assert_matches_results(sql, setup)

    def test_full_join_with_mutually_empty_sides(self):
        setup = (
            "CREATE TABLE users (id INTEGER NOT NULL);\n"
            "CREATE TABLE orders (id INTEGER NOT NULL, user_id INTEGER NOT NULL);\n"
            "INSERT INTO users VALUES (1);\n"
            "INSERT INTO orders VALUES (10, 2);\n"
        )
        sql = (
            "SELECT u.id AS uid, o.user_id AS user_id "
            "FROM users u FULL JOIN orders o ON u.id = o.user_id"
        )
        names, rows = self.execute(sql, setup)
        observed = {(row["uid"], row["user_id"]) for row in rows}
        self.assertEqual(observed, {(1, None), (None, 2)})
        self.assert_matches_results(sql, setup)


if __name__ == "__main__":
    unittest.main()
