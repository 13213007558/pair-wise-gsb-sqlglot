import sqlite3
import unittest

import duckdb

from sqlglot import ParamError, generate_params, parse_params, transpile_params


class TestParams(unittest.TestCase):
    maxDiff = None

    def test_named_to_qmark(self):
        sql, params = transpile_params(
            "SELECT * FROM t WHERE a = :x OR b = :x AND c = :y",
            read="sqlite",
            write="duckdb",
            params={"x": 1, "y": "v"},
            output_style="qmark",
        )
        self.assertEqual(sql, "SELECT * FROM t WHERE a = ? OR b = ? AND c = ?")
        self.assertEqual(params, [1, 1, "v"])

    def test_qmark_to_named(self):
        sql, params = transpile_params(
            "SELECT * FROM t WHERE a = ? AND b = ?",
            read="sqlite",
            write="sqlite",
            params=[10, 20],
            input_style="qmark",
            output_style="named",
        )
        self.assertEqual(sql, "SELECT * FROM t WHERE a = :p1 AND b = :p2")
        self.assertEqual(params, {"p1": 10, "p2": 20})

    def test_numeric_input(self):
        sql, params = transpile_params(
            "SELECT * FROM t WHERE a = :2 AND b = :1 AND c = :2",
            params=["a", "b"],
            input_style="numeric",
            output_style="qmark",
        )
        self.assertEqual(sql, "SELECT * FROM t WHERE a = ? AND b = ? AND c = ?")
        self.assertEqual(params, ["b", "a", "b"])

    def test_numeric_output(self):
        sql, params = transpile_params(
            "SELECT * FROM t WHERE a = :x OR b = :x",
            params={"x": 1},
            output_style="numeric",
        )
        self.assertEqual(sql, "SELECT * FROM t WHERE a = :1 OR b = :2")
        self.assertEqual(params, [1, 1])

    def test_named_output_reuses_source_names(self):
        sql, params = transpile_params("SELECT * FROM t WHERE a = :x OR b = :x", params={"x": 1})
        self.assertEqual(sql, "SELECT * FROM t WHERE a = :x OR b = :x")
        self.assertEqual(params, {"x": 1})

    def test_distinct_params_are_not_merged(self):
        sql, params = transpile_params(
            "SELECT * FROM t WHERE a = :x AND b = :y",
            params={"x": 5, "y": 5},
            output_style="qmark",
        )
        self.assertEqual(sql, "SELECT * FROM t WHERE a = ? AND b = ?")
        self.assertEqual(params, [5, 5])

        _, named = transpile_params(
            "SELECT * FROM t WHERE a = :x AND b = :y", params={"x": 5, "y": 5}
        )
        self.assertEqual(named, {"x": 5, "y": 5})

    def test_literals_are_not_params(self):
        sql, params = transpile_params(
            "SELECT ':x' AS s, \"a:b\" AS i, '?' AS q FROM t WHERE c = :z -- :nope ?",
            read="sqlite",
            write="sqlite",
            params={"z": 5},
            output_style="qmark",
        )
        self.assertEqual(params, [5])
        self.assertIn("':x'", sql)
        self.assertIn("'?'", sql)
        self.assertNotIn("5", sql)

    def test_slices_are_not_params(self):
        sql, params = transpile_params(
            "SELECT arr[1:2] FROM t WHERE x = :1",
            read="duckdb",
            write="duckdb",
            params=[42],
            input_style="numeric",
            output_style="qmark",
        )
        self.assertEqual(sql, "SELECT arr[1:2] FROM t WHERE x = ?")
        self.assertEqual(params, [42])

    def test_mixed_styles_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :x AND b = ?", params={"x": 1})
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = ?", input_style="named", params=[])
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :1", input_style="qmark", params=[])
        with self.assertRaises(ParamError):
            transpile_params("SELECT @x", read="tsql", params={"x": 1})

    def test_missing_bindings_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :x", params={})
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :x", params=None)
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = ?", input_style="qmark", params=[])
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :2", input_style="numeric", params=[1])

    def test_unused_bindings_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = :x", params={"x": 1, "y": 2})
        with self.assertRaises(ParamError):
            transpile_params("SELECT * FROM t WHERE a = ?", input_style="qmark", params=[1, 2])
        with self.assertRaises(ParamError):
            transpile_params("SELECT :2", input_style="numeric", params=[1, 2])
        with self.assertRaises(ParamError):
            transpile_params("SELECT 1", params={"x": 1})

    def test_command_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("VACUUM t", read="sqlite")
        with self.assertRaises(ParamError):
            transpile_params("VACUUM t", read="sqlite", params={"x": 1})

    def test_multiple_statements_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("SELECT :x; SELECT :y", params={"x": 1, "y": 2})

    def test_invalid_style_rejected(self):
        with self.assertRaises(ParamError):
            transpile_params("SELECT 1", input_style="pyformat")
        with self.assertRaises(ParamError):
            transpile_params("SELECT 1", output_style="pyformat")

    def test_reordered_params(self):
        sql, params = transpile_params(
            "SELECT STR_POSITION(:a, :b)",
            read="spark",
            write="postgres",
            params={"a": "x", "b": "yz"},
            output_style="qmark",
        )
        self.assertEqual(sql, "SELECT POSITION(? IN ?)")
        self.assertEqual(params, ["yz", "x"])

    def test_expanded_expression_params(self):
        sql, params = transpile_params(
            "SELECT ARRAY_EXCEPT(:a, :b)",
            read="snowflake",
            write="duckdb",
            params={"a": [1, 2, 2, 3], "b": [2]},
            output_style="qmark",
        )
        self.assertNotIn(":a", sql)
        self.assertNotIn("[1, 2, 2, 3]", sql)
        self.assertEqual(sql.count("?"), len(params))
        self.assertEqual(params.count([1, 2, 2, 3]), 4)
        self.assertEqual(params.count([2]), 2)

    def test_expression_reuse_and_immutability(self):
        expression, keys = parse_params("SELECT * FROM t WHERE a = :x OR b = :x", read="sqlite")
        before = expression.sql()
        params = {"x": 9}

        sql, qmark = generate_params(expression, keys, params, write="sqlite", output_style="qmark")
        self.assertEqual((sql, qmark), ("SELECT * FROM t WHERE a = ? OR b = ?", [9, 9]))

        sql, named = generate_params(expression, keys, params, write="duckdb")
        self.assertEqual((sql, named), ("SELECT * FROM t WHERE a = :x OR b = :x", {"x": 9}))

        sql, numeric = generate_params(
            expression, keys, params, write="postgres", output_style="numeric"
        )
        self.assertEqual((sql, numeric), ("SELECT * FROM t WHERE a = :1 OR b = :2", [9, 9]))

        self.assertEqual(expression.sql(), before)
        self.assertEqual(params, {"x": 9})

    def test_no_params(self):
        self.assertEqual(
            transpile_params("SELECT 1", read="sqlite", write="duckdb"), ("SELECT 1", {})
        )
        self.assertEqual(transpile_params("SELECT 1", output_style="qmark"), ("SELECT 1", []))

    def test_sqlite_execution(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a INT, b INT)")
        conn.execute("CREATE TABLE u (b INT, c INT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(2, 1), (2, 2), (3, 2)])
        conn.executemany("INSERT INTO u VALUES (?, ?)", [(1, 2), (2, 3)])

        sql = "SELECT * FROM t WHERE a = :x AND b IN (SELECT b FROM u WHERE c = :x)"
        expected = conn.execute(sql, {"x": 2}).fetchall()

        out, params = transpile_params(
            sql, read="sqlite", write="sqlite", params={"x": 2}, output_style="qmark"
        )
        self.assertNotIn("2", out)
        self.assertEqual(conn.execute(out, params).fetchall(), expected)

        out, params = transpile_params(sql, read="sqlite", write="sqlite", params={"x": 2})
        self.assertEqual(conn.execute(out, params).fetchall(), expected)

    def test_duckdb_execution(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a INT, b VARCHAR)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y"), (3, "x")])

        sql = "SELECT * FROM t WHERE b = :v AND a < :n OR b = :v AND a > :n"
        expected = conn.execute(
            "SELECT * FROM t WHERE b = 'x' AND a < 2 OR b = 'x' AND a > 2"
        ).fetchall()

        out, params = transpile_params(
            sql,
            read="duckdb",
            write="duckdb",
            params={"v": "x", "n": 2},
            output_style="qmark",
        )
        self.assertNotIn("'x'", out)
        self.assertEqual(params, ["x", 2, "x", 2])
        self.assertEqual(conn.execute(out, params).fetchall(), expected)

        out, params = transpile_params(
            "SELECT ARRAY_EXCEPT(:a, :b)",
            read="snowflake",
            write="duckdb",
            params={"a": [1, 2, 2, 3], "b": [2]},
            output_style="qmark",
        )
        self.assertEqual(conn.execute(out, params).fetchall(), [([1, 2, 3],)])


if __name__ == "__main__":
    unittest.main()
