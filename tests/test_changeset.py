import json
import os
import subprocess
import sys
import unittest

from sqlglot import exp, parse_one
from sqlglot.changeset import ChangesetConflict, apply_changeset, diff_changeset

SOURCE_SQL = """
WITH cte AS (SELECT x, y FROM t WHERE x > 1)
SELECT a, b, a + b, x FROM cte WHERE x > 1 AND y < 10
"""

TARGET_SQL = """
WITH cte AS (SELECT x, y FROM t WHERE x > 2)
SELECT x, a + b, b, a FROM cte WHERE x > 1 AND y < 20
"""


class TestChangeset(unittest.TestCase):
    maxDiff = None

    def roundtrip(self, source_sql, target_sql):
        source = parse_one(source_sql)
        target = parse_one(target_sql)
        changeset = diff_changeset(source, target)

        # the changeset must be JSON-serializable (no Python object references)
        changeset = json.loads(json.dumps(changeset))

        result = apply_changeset(parse_one(source_sql), changeset)
        self.assertEqual(result, target)
        self.assertEqual(result.sql(), target.sql())
        return changeset

    def test_insert(self):
        changeset = self.roundtrip("SELECT a", "SELECT a, b")
        self.assertIn("insert", {op["op"] for op in changeset["edits"]})

        self.roundtrip("SELECT a", "SELECT a, f(x, y)")
        self.roundtrip("SELECT a FROM t LIMIT 10", "SELECT a FROM t LIMIT 10 OFFSET 5")

    def test_remove(self):
        changeset = self.roundtrip("SELECT a, b", "SELECT a")
        self.assertIn("remove", {op["op"] for op in changeset["edits"]})

        self.roundtrip("SELECT f(x, y), a", "SELECT a")
        self.roundtrip("SELECT DISTINCT a", "SELECT a")

    def test_update(self):
        changeset = self.roundtrip("SELECT a FROM table_one", "SELECT a FROM table_two")
        self.assertIn("update", {op["op"] for op in changeset["edits"]})

        self.roundtrip("SELECT a FROM t", "SELECT a FROM u")
        self.roundtrip("SELECT a AS x", "SELECT a AS y")
        self.roundtrip("SELECT CAST(x AS INT)", "SELECT CAST(x AS BIGINT)")

    def test_replace(self):
        changeset = self.roundtrip("SELECT x + 1", "SELECT x + 2")
        self.assertIn("replace", {op["op"] for op in changeset["edits"]})

        # roots of different types are replaced as a whole
        self.roundtrip("SELECT a", "DELETE FROM t")

    def test_move(self):
        changeset = self.roundtrip("SELECT a, b, c", "SELECT c, a, b")
        self.assertIn("move", {op["op"] for op in changeset["edits"]})

        self.roundtrip("SELECT a, b, c, d", "SELECT d, c, b, a")
        self.roundtrip(
            "SELECT a FROM t1 JOIN t2 ON t1.x = t2.x JOIN t3 ON t2.y = t3.y",
            "SELECT a FROM t1 JOIN t3 ON t2.y = t3.y JOIN t2 ON t1.x = t2.x",
        )

    def test_move_with_duplicate_expressions(self):
        # identical expressions are distinct nodes: moves must not confuse them
        self.roundtrip("SELECT a, b, a", "SELECT a, a, b")
        self.roundtrip("SELECT a, a, b", "SELECT b, a, a")
        self.roundtrip("SELECT x, x, y, x", "SELECT x, y, x, x")

    def test_move_then_edit_same_list(self):
        # a move shifts list indexes; later edits must still hit the right nodes
        self.roundtrip(
            "SELECT a, b, c FROM t WHERE x = 1",
            "SELECT c, a, b + 1 FROM t WHERE x = 2",
        )

    def test_cte_projection_reorder_and_predicate_change(self):
        changeset = self.roundtrip(SOURCE_SQL, TARGET_SQL)
        ops = {op["op"] for op in changeset["edits"]}
        self.assertIn("move", ops)
        self.assertIn("replace", ops)

    def test_apply_in_separate_process(self):
        changeset = diff_changeset(parse_one(SOURCE_SQL), parse_one(TARGET_SQL))
        payload = json.dumps(changeset)

        program = (
            "import json, sys;"
            "from sqlglot import parse_one;"
            "from sqlglot.changeset import apply_changeset;"
            "sys.stdout.write(apply_changeset(parse_one(sys.argv[2]), json.loads(sys.argv[1])).sql())"
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        process = subprocess.run(
            [sys.executable, "-c", program, payload, SOURCE_SQL],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, parse_one(TARGET_SQL).sql())

    def test_conflict_on_modified_source(self):
        source_sql = "SELECT a, b FROM t WHERE x > 1 AND y < 5"
        target_sql = "SELECT b, a FROM t WHERE x > 2 AND y < 5"
        changeset = diff_changeset(parse_one(source_sql), parse_one(target_sql))

        tampered = parse_one(source_sql)
        tampered.find(exp.GT).set("expression", exp.Literal.number(9))
        tampered_sql = tampered.sql()

        with self.assertRaises(ChangesetConflict) as ctx:
            apply_changeset(tampered, changeset)

        # the conflict points at the modified expression, and the tree is untouched
        self.assertEqual(
            ctx.exception.path,
            [["where", None], ["this", None], ["this", None], ["expression", None]],
        )
        self.assertEqual(tampered.sql(), tampered_sql)

    def test_conflict_outside_edited_regions(self):
        source_sql = "SELECT a, b FROM t WHERE x > 1 AND y < 5"
        target_sql = "SELECT b, a FROM t WHERE x > 2 AND y < 5"
        changeset = diff_changeset(parse_one(source_sql), parse_one(target_sql))

        # modify an expression that no edit touches
        tampered = parse_one(source_sql)
        tampered.find(exp.LT).set("expression", exp.Literal.number(7))
        tampered_sql = tampered.sql()

        with self.assertRaises(ChangesetConflict) as ctx:
            apply_changeset(tampered, changeset)

        self.assertEqual(ctx.exception.path, [])
        self.assertEqual(tampered.sql(), tampered_sql)

    def test_apply_does_not_mutate_input(self):
        source_sql = "SELECT a, b FROM t WHERE x > 1"
        source = parse_one(source_sql)
        changeset = diff_changeset(source, parse_one("SELECT b, a FROM t WHERE x > 2"))

        result = apply_changeset(source, changeset)
        self.assertEqual(source.sql(), source_sql)
        self.assertEqual(result.sql(), "SELECT b, a FROM t WHERE x > 2")

    def test_apply_accepts_json(self):
        changeset = diff_changeset(parse_one("SELECT a"), parse_one("SELECT a, b"))
        result = apply_changeset(parse_one("SELECT a"), json.dumps(changeset))
        self.assertEqual(result.sql(), "SELECT a, b")

    def test_empty_changeset(self):
        source = parse_one("SELECT a FROM t")
        changeset = diff_changeset(source, parse_one("SELECT a FROM t"))
        self.assertEqual(changeset["edits"], [])
        self.assertEqual(apply_changeset(source, changeset), source)
