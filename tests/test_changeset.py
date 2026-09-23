import json
import os
import subprocess
import sys
import unittest

from sqlglot import exp, parse_one
from sqlglot.changeset import Changeset, ChangesetConflictError, changeset

SOURCE_SQL = """
WITH filtered AS (
    SELECT id, amount, amount AS amount_copy, region
    FROM orders
    WHERE region = 'us'
)
SELECT id, amount, amount, region
FROM filtered
JOIN customers ON customers.id = filtered.id
WHERE amount > 100 AND region = 'us'
"""

TARGET_SQL = """
WITH filtered AS (
    SELECT id, amount, amount AS amount_copy, region
    FROM orders
    WHERE region = 'eu'
)
SELECT region, amount, id, amount
FROM filtered
JOIN customers ON customers.id = filtered.id
WHERE amount > 200 AND region = 'eu'
"""


class TestChangeset(unittest.TestCase):
    def assert_replays(self, source_sql, target_sql):
        source = parse_one(source_sql)
        target = parse_one(target_sql)

        payload = changeset(source, target).to_json()
        json.loads(payload)

        replayed = parse_one(source_sql)
        Changeset.from_json(payload).apply(replayed)
        self.assertEqual(replayed, target)
        self.assertEqual(replayed.sql(), target.sql())

    def test_update(self):
        self.assert_replays("SELECT a, b FROM t WHERE x = 1", "SELECT a, b FROM t WHERE x = 2")

    def test_insert(self):
        self.assert_replays("SELECT a FROM t", "SELECT a, b, c FROM t WHERE x > 1")

    def test_remove(self):
        self.assert_replays("SELECT a, b, c FROM t WHERE x > 1", "SELECT a FROM t")

    def test_move_projection(self):
        self.assert_replays("SELECT a, b, c FROM t", "SELECT c, a, b FROM t")

    def test_move_duplicate_expressions(self):
        self.assert_replays("SELECT x, x, y FROM t", "SELECT y, x, x FROM t")
        self.assert_replays(
            "SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id",
            "SELECT * FROM a JOIN c ON b.id = c.id JOIN b ON a.id = b.id",
        )

    def test_cte_projection_reorder_and_predicate_update(self):
        self.assert_replays(SOURCE_SQL, TARGET_SQL)

    def test_root_replacement(self):
        self.assert_replays("SELECT a FROM t", "SELECT a, b FROM t WHERE x = 1")

    def test_noop(self):
        self.assert_replays("SELECT a, b FROM t", "SELECT a, b FROM t")

    def test_json_round_trip(self):
        cs = changeset(parse_one(SOURCE_SQL), parse_one(TARGET_SQL))
        payload = json.loads(json.dumps(cs.to_dict()))
        self.assertEqual(payload["version"], 1)
        self.assertTrue(payload["edits"])
        for edit in payload["edits"]:
            self.assertIn(edit["type"], ("insert", "remove", "update", "move"))
        self.assertEqual(Changeset.from_dict(payload).to_dict(), cs.to_dict())

    def test_conflict_is_atomic(self):
        cs = changeset(parse_one(SOURCE_SQL), parse_one(TARGET_SQL))

        tree = parse_one(SOURCE_SQL)
        tree.find(exp.Literal).replace(exp.Literal.string("xx"))
        modified_sql = tree.sql()

        with self.assertRaises(ChangesetConflictError) as ctx:
            cs.apply(tree)

        self.assertTrue(ctx.exception.conflicts)
        self.assertEqual(tree.sql(), modified_sql)

    def test_conflict_reports_all_positions(self):
        cs = changeset(parse_one("SELECT a, b FROM t"), parse_one("SELECT a, b, c FROM t"))

        tree = parse_one("SELECT a, b FROM t")
        tree.find(exp.Select).args["expressions"][0].replace(exp.column("z"))

        with self.assertRaises(ChangesetConflictError):
            cs.apply(tree)

    def test_cross_process_replay(self):
        cs = changeset(parse_one(SOURCE_SQL), parse_one(TARGET_SQL))
        script = (
            "import sys; "
            "from sqlglot import parse_one; "
            "from sqlglot.changeset import Changeset; "
            f"tree = parse_one({SOURCE_SQL!r}); "
            "Changeset.from_json(sys.stdin.read()).apply(tree); "
            "print(tree.sql())"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=cs.to_json(),
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        self.assertEqual(result.stdout.strip(), parse_one(TARGET_SQL).sql())


if __name__ == "__main__":
    unittest.main()
