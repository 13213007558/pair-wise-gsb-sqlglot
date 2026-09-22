import sqlite3
import unittest

from sqlglot import parse_one, transpile_params
from sqlglot.errors import UnsupportedError
from sqlglot.executor import execute as execute_sqlglot

try:
    import duckdb
except ImportError:
    duckdb = None


def transpile_one(sql, params=None, **kwargs):
    results = transpile_params(sql, params, **kwargs)
    assert len(results) == 1
    return results[0]


class TestTranspileParams(unittest.TestCase):
    def test_qmark_identity(self):
        result = transpile_one("SELECT * FROM t WHERE a = ? AND b = ?", [1, "two"])
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = ? AND b = ?")
        self.assertEqual(result.params, [1, "two"])

    def test_named_shared_value(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = :x OR b = :x", {"x": 5}, output_style="named"
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = :x OR b = :x")
        self.assertEqual(result.params, {"x": 5})

    def test_named_to_qmark(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = :x OR b = :x AND c = :y",
            {"x": 5, "y": 6},
            output_style="qmark",
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = ? OR b = ? AND c = ?")
        self.assertEqual(result.params, [5, 5, 6])

    def test_qmark_to_named(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = ? AND b = ?", [1, 2], output_style="named"
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = :p1 AND b = :p2")
        self.assertEqual(result.params, {"p1": 1, "p2": 2})

    def test_numeric_input(self):
        result = transpile_one("SELECT * FROM t WHERE a = :1 AND b = :2 AND c = :1", [7, 8])
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = ? AND b = ? AND c = ?")
        self.assertEqual(result.params, [7, 8, 7])

    def test_numeric_output(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = :x OR b = :x", {"x": 5}, output_style="numeric"
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = :1 OR b = :2")
        self.assertEqual(result.params, [5, 5])

    def test_duckdb_dollar_numeric_input(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = $1 AND b = $2",
            [1, 2],
            read="duckdb",
            write="sqlite",
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = ? AND b = ?")
        self.assertEqual(result.params, [1, 2])

    def test_same_value_parameters_are_not_merged(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = ? AND b = ?", [5, 5], output_style="named"
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE a = :p1 AND b = :p2")
        self.assertEqual(result.params, {"p1": 5, "p2": 5})

    def test_expression_expansion(self):
        # XOR expands to (a AND NOT b) OR (NOT a AND b), duplicating each operand
        result = transpile_one(
            "SELECT * FROM t WHERE a XOR ?", [True], read="mysql", write="duckdb"
        )
        self.assertEqual(result.sql, "SELECT * FROM t WHERE (a AND (NOT ?)) OR ((NOT a) AND ?)")
        self.assertEqual(result.params, [True, True])

    def test_expression_reorder(self):
        # POSITION(x IN y) becomes INSTR(y, x) in SQLite, swapping the arguments
        result = transpile_one(
            "SELECT POSITION(:x IN 'hello world')", {"x": "o"}, read="mysql", write="sqlite"
        )
        self.assertEqual(result.sql, "SELECT INSTR('hello world', ?)")
        self.assertEqual(result.params, ["o"])

    def test_subquery(self):
        result = transpile_one(
            "SELECT * FROM t WHERE a = :x AND b IN (SELECT b FROM u WHERE c = :x)",
            {"x": 3},
            write="duckdb",
        )
        self.assertEqual(
            result.sql,
            "SELECT * FROM t WHERE a = ? AND b IN (SELECT b FROM u WHERE c = ?)",
        )
        self.assertEqual(result.params, [3, 3])

    def test_strings_comments_and_quoted_identifiers_are_not_parameters(self):
        result = transpile_one(
            "SELECT ':not', \":col\" FROM t /* :comment */ WHERE a = :real -- :trailing",
            {"real": 9},
        )
        self.assertEqual(result.params, [9])
        self.assertNotIn("9", result.sql)
        self.assertIn("':not'", result.sql)
        self.assertEqual(result.sql.count("?"), 1)

    def test_mixed_styles_rejected(self):
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = ? AND b = :x", [1])
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :1 AND b = :x", [1])

    def test_missing_bindings_rejected(self):
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :x", {"y": 1})
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :x")
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = ? AND b = ?", [1])
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = ?", [1, 2])
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :3", [1, 2])

    def test_wrong_params_type_rejected(self):
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :x", [1])
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = ?", {"x": 1})

    def test_extra_named_keys_ignored(self):
        result = transpile_one("SELECT * FROM t WHERE a = :x", {"x": 1, "y": 2})
        self.assertEqual(result.params, [1])

    def test_params_without_placeholders_rejected(self):
        with self.assertRaises(ValueError):
            transpile_params("SELECT 1", [1])

    def test_input_style_validation(self):
        with self.assertRaises(ValueError):
            transpile_params("SELECT * FROM t WHERE a = :x", {"x": 1}, input_style="qmark")
        result = transpile_one("SELECT * FROM t WHERE a = :x", {"x": 1}, input_style="named")
        self.assertEqual(result.params, [1])
        with self.assertRaises(ValueError):
            transpile_params("SELECT 1", input_style="bogus")
        with self.assertRaises(ValueError):
            transpile_params("SELECT 1", output_style="bogus")

    def test_command_rejected(self):
        with self.assertRaises(UnsupportedError):
            transpile_params("VACUUM t", [])
        with self.assertRaises(UnsupportedError):
            transpile_params("ALTER SESSION SET x = 1", [])

    def test_ast_reuse_and_immutability(self):
        ast = parse_one("SELECT * FROM t WHERE a = :x OR b = :x")
        ast_sql = ast.sql()
        params = {"x": 5}

        duckdb_result = transpile_one(ast, params, write="duckdb", output_style="qmark")
        sqlite_result = transpile_one(ast, params, write="sqlite", output_style="named")

        self.assertEqual(duckdb_result.sql, "SELECT * FROM t WHERE a = ? OR b = ?")
        self.assertEqual(duckdb_result.params, [5, 5])
        self.assertEqual(sqlite_result.sql, "SELECT * FROM t WHERE a = :x OR b = :x")
        self.assertEqual(sqlite_result.params, {"x": 5})
        self.assertEqual(ast.sql(), ast_sql)
        self.assertEqual(params, {"x": 5})

        positional = [1, 2]
        transpile_one("SELECT * FROM t WHERE a = ? AND b = ?", positional)
        self.assertEqual(positional, [1, 2])

    def test_multiple_statements(self):
        results = transpile_params(
            "SELECT * FROM t WHERE a = ?; SELECT * FROM u WHERE b = ?", [1, 2]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].params, [1])
        self.assertEqual(results[1].params, [2])

    def test_pretty(self):
        result = transpile_one("SELECT a FROM t WHERE b = :x", {"x": 1}, pretty=True)
        self.assertEqual(result.params, [1])
        self.assertEqual(result.sql.count("?"), 1)


class TestSQLiteExecution(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE t (a INTEGER, b INTEGER, name TEXT)")
        self.connection.execute("CREATE TABLE u (b INTEGER, c INTEGER)")
        self.connection.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            [(1, 2, "Alice"), (2, 2, "bob"), (3, 1, "CAROL"), (4, 5, "dave")],
        )
        self.connection.executemany("INSERT INTO u VALUES (?, ?)", [(2, 1), (1, 9)])

    def tearDown(self):
        self.connection.close()

    def run_query(self, sql, params):
        return self.connection.execute(sql, params).fetchall()

    def test_repeated_named_reference(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a = :x OR b = :x ORDER BY a",
            {"x": 1},
            write="sqlite",
            output_style="named",
        )
        self.assertNotIn("1", result.sql.replace(":1", ""))
        self.assertEqual(self.run_query(result.sql, result.params), [(1,), (3,)])

    def test_subquery_with_qmark(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a = :x AND b IN (SELECT b FROM u WHERE c = :x)",
            {"x": 1},
            write="sqlite",
        )
        self.assertEqual(result.params, [1, 1])
        self.assertEqual(self.run_query(result.sql, result.params), [(1,)])

    def test_expanded_expression_ilike(self):
        sentinel = "zzzSENTINELzzz"
        result = transpile_one(
            "SELECT name FROM t WHERE name ILIKE :pat",
            {"pat": sentinel},
            read="postgres",
            write="sqlite",
        )
        self.assertNotIn(sentinel, result.sql)
        self.assertEqual(result.sql, "SELECT name FROM t WHERE LOWER(name) LIKE LOWER(?)")
        self.assertEqual(self.run_query(result.sql, result.params), [])
        self.assertEqual(self.run_query(result.sql, ["a%"]), [("Alice",)])

    def test_reordered_expression(self):
        result = transpile_one(
            "SELECT POSITION(:x IN 'hello world')",
            {"x": "o"},
            read="mysql",
            write="sqlite",
        )
        self.assertNotIn("'o'", result.sql)
        self.assertEqual(self.run_query(result.sql, result.params), [(5,)])

    def test_numeric_output_executes(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a = :x OR b = :x ORDER BY a",
            {"x": 2},
            write="sqlite",
            output_style="numeric",
        )
        self.assertEqual(result.sql, "SELECT a FROM t WHERE a = :1 OR b = :2 ORDER BY a")
        params = {str(index + 1): value for index, value in enumerate(result.params)}
        self.assertEqual(self.run_query(result.sql, params), [(1,), (2,)])


class TestDuckDBExecution(unittest.TestCase):
    tables = {
        "t": [
            {"a": 1, "b": 2},
            {"a": 2, "b": 2},
            {"a": 3, "b": 1},
        ],
        "u": [
            {"b": 2, "c": 1},
            {"b": 1, "c": 9},
        ],
    }

    @staticmethod
    def inline(sql, params):
        # Test-only helper: substitute qmark placeholders positionally, mimicking
        # what a driver does when binding, so the SQL can run without parameters.
        parts = sql.split("?")
        assert len(parts) - 1 == len(params)
        out = parts[0]
        for value, part in zip(params, parts[1:]):
            if isinstance(value, str):
                out += f"'{value}'"
            elif isinstance(value, bool):
                out += str(value).upper()
            else:
                out += str(value)
            out += part
        return out

    def execute_duckdb(self, sql, params):
        if duckdb is not None:
            return duckdb.execute(sql, params).fetchall()
        inlined = self.inline(sql, params)
        return execute_sqlglot(inlined, tables=self.tables, dialect="duckdb").rows

    def test_expanded_expression_xor(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a XOR :flag ORDER BY a",
            {"flag": False},
            read="mysql",
            write="duckdb",
        )
        self.assertNotIn("False", result.sql)
        self.assertEqual(result.params, [False, False])
        self.assertEqual(self.execute_duckdb(result.sql, result.params), [(1,), (2,), (3,)])

        result = transpile_one(
            "SELECT a FROM t WHERE a XOR :flag ORDER BY a",
            {"flag": True},
            read="mysql",
            write="duckdb",
        )
        self.assertEqual(self.execute_duckdb(result.sql, result.params), [])

    def test_subquery_and_repeated_reference(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a = :x AND b IN (SELECT b FROM u WHERE c = :x)",
            {"x": 1},
            write="duckdb",
        )
        self.assertNotIn(" 1", result.sql)
        self.assertEqual(self.execute_duckdb(result.sql, result.params), [(1,)])

    def test_repeated_reference(self):
        result = transpile_one(
            "SELECT a FROM t WHERE a = :x OR b = :x ORDER BY a",
            {"x": 1},
            write="duckdb",
        )
        self.assertEqual(self.execute_duckdb(result.sql, result.params), [(1,), (3,)])


if __name__ == "__main__":
    unittest.main()
