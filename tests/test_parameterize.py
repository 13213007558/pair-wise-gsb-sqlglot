import sqlite3
import unittest

import sqlglot
from sqlglot import exp
from sqlglot.errors import UnsupportedError
from sqlglot.parameterize import parameterize


class TestParameterize(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE t (score REAL, name TEXT)")
        self.conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(1.5, "O'Brien"), (3.0, "a?b"), (5.0, "x'; DROP TABLE t; --")],
        )

    def tearDown(self):
        self.conn.close()

    def assert_same_results(self, sql, parameterized):
        """Executes the original and the parameterized SQL on the same SQLite database."""
        expected = self.conn.execute(sql).fetchall()
        cursor = self.conn.execute(parameterized.sql, parameterized.params)
        self.assertEqual(cursor.fetchall(), expected)
        self.assertEqual(
            [d[0] for d in cursor.description],
            [d[0] for d in self.conn.execute(sql).description],
        )

    def test_example_sqlite(self):
        sql = "SELECT 7 AS n, 'a''b' AS s FROM t WHERE score > -2 ORDER BY 1 LIMIT 5"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(
            result.sql, "SELECT ? AS n, ? AS s FROM t WHERE score > ? ORDER BY 1 LIMIT ?"
        )
        self.assertEqual(result.params, [7, "a'b", -2, 5])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].value, 1)
        self.assert_same_results(sql, result)

    def test_example_postgres(self):
        sql = "SELECT 7 AS n, 'a''b' AS s FROM t WHERE score > -2 ORDER BY 1 LIMIT 5"
        result = parameterize(sql, dialect="postgres")

        self.assertEqual(
            result.sql, "SELECT $1 AS n, $2 AS s FROM t WHERE score > $3 ORDER BY 1 LIMIT $4"
        )
        self.assertEqual(result.params, [7, "a'b", -2, 5])
        self.assertEqual(len(result.skipped), 1)

    def test_postgres_numbering_order(self):
        sql = """
            WITH cte AS (SELECT 1 AS x)
            SELECT (SELECT 2), SUM(x) OVER (ORDER BY x) FROM cte
            WHERE x > 3
            UNION ALL SELECT 4, 5
        """
        result = parameterize(sql, dialect="postgres")

        self.assertEqual(
            result.sql,
            "WITH cte AS (SELECT $1 AS x) "
            "SELECT (SELECT $2), SUM(x) OVER (ORDER BY x) FROM cte "
            "WHERE x > $3 "
            "UNION ALL SELECT $4, $5",
        )
        self.assertEqual(result.params, [1, 2, 3, 4, 5])

    def test_cte_union_window_execution(self):
        sql = """
            WITH cte AS (SELECT score FROM t WHERE score > 0)
            SELECT score, SUM(score) OVER (ORDER BY score) FROM cte
            UNION ALL SELECT 100, 200
        """
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, [0, 100, 200])
        self.assert_same_results(sql, result)

    def test_window_frame_bounds_kept(self):
        sql = (
            "SELECT SUM(score) OVER (ORDER BY score "
            "ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING) FROM t WHERE score > 0"
        )
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, [0])
        self.assertIn("ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING", result.sql)
        self.assertEqual([s.value for s in result.skipped], [1, 2])
        self.assert_same_results(sql, result)

    def test_group_by_ordinal_kept(self):
        sql = "SELECT name, COUNT(*) FROM t GROUP BY 1 HAVING COUNT(*) > 0"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, [0])
        self.assertIn("GROUP BY 1", result.sql)
        self.assertEqual(len(result.skipped), 1)
        self.assert_same_results(sql, result)

    def test_duplicate_constants_get_separate_slots(self):
        result = parameterize("SELECT 1, 1, 'a', 'a'", dialect="sqlite")

        self.assertEqual(result.sql, "SELECT ?, ?, ?, ?")
        self.assertEqual(result.params, [1, 1, "a", "a"])

    def test_negative_values_keep_type_and_value(self):
        sql = "SELECT * FROM t WHERE score > -2 AND score < -0.5"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, [-2, -0.5])
        self.assertIsInstance(result.params[0], int)
        self.assertIsInstance(result.params[1], float)
        self.assert_same_results(sql, result)

    def test_question_mark_in_string_is_data(self):
        sql = "SELECT * FROM t WHERE name = 'a?b'"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.sql, "SELECT * FROM t WHERE name = ?")
        self.assertEqual(result.params, ["a?b"])
        self.assert_same_results(sql, result)

    def test_malicious_quotes_are_data(self):
        sql = "SELECT * FROM t WHERE name = 'O''Brien' OR name = 'x''; DROP TABLE t; --'"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, ["O'Brien", "x'; DROP TABLE t; --"])
        self.assert_same_results(sql, result)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM t").fetchone()[0], 3)

    def test_comment_numbers_are_not_parameters(self):
        sql = "SELECT name FROM t /* keep 100 */ WHERE score > 3 -- and 200"
        result = parameterize(sql, dialect="sqlite")

        self.assertEqual(result.params, [3])
        self.assertIn("/* keep 100 */", result.sql)
        self.assert_same_results(sql, result)

    def test_type_declaration_kept(self):
        sql = "SELECT CAST(score AS DECIMAL(10, 2)) FROM t"
        result = parameterize(sql, dialect="postgres")

        self.assertEqual(result.sql, "SELECT CAST(score AS DECIMAL(10, 2)) FROM t")
        self.assertEqual(result.params, [])
        self.assertEqual([s.value for s in result.skipped], [10, 2])

    def test_interval_value_kept(self):
        sql = "SELECT * FROM t WHERE score > 1 + INTERVAL '2' DAY"
        result = parameterize(sql, dialect="postgres")

        self.assertEqual(result.sql, "SELECT * FROM t WHERE score > $1 + INTERVAL '2 DAY'")
        self.assertEqual(result.params, [1])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].value, "2")

    def test_existing_placeholders_rejected(self):
        for sql, dialect in (
            ("SELECT * FROM t WHERE score = ?", "sqlite"),
            ("SELECT * FROM t WHERE score = $1", "postgres"),
            ("SELECT * FROM t WHERE score = :score", "postgres"),
        ):
            with self.assertRaises(UnsupportedError):
                parameterize(sql, dialect=dialect)

    def test_unsupported_dialect_rejected(self):
        with self.assertRaises(UnsupportedError):
            parameterize("SELECT 1", dialect="mysql")

    def test_non_select_rejected(self):
        with self.assertRaises(UnsupportedError):
            parameterize("INSERT INTO t VALUES (1)", dialect="sqlite")

    def test_multiple_statements_rejected(self):
        with self.assertRaises(UnsupportedError):
            parameterize("SELECT 1; SELECT 2", dialect="sqlite")

    def test_input_ast_not_mutated(self):
        tree = sqlglot.parse_one("SELECT 7 AS n FROM t WHERE score > -2 ORDER BY 1", read="sqlite")
        before = tree.sql(dialect="sqlite")

        result = parameterize(tree, dialect="sqlite")

        self.assertEqual(tree.sql(dialect="sqlite"), before)
        self.assertIsNone(tree.find(exp.Placeholder))
        self.assertIsNotNone(result.expression.find(exp.Placeholder))
        self.assertEqual(result.params, [7, -2])


if __name__ == "__main__":
    unittest.main()
