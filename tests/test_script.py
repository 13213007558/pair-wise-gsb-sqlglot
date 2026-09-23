import unittest

from sqlglot import exp
from sqlglot.script import ScriptSession, analyze_script


def column_names(result):
    return [column.name for column in result.output_columns]


def column_types(result):
    return {column.name: column.type.this for column in result.output_columns}


def snapshot_types(snapshot, table):
    return {column: dtype.this for column, dtype in snapshot.tables[table].items()}


class TestScriptAnalysis(unittest.TestCase):
    def test_create_then_select(self):
        result = analyze_script(
            "CREATE TABLE users (id INT, name TEXT); SELECT * FROM users",
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.statements), 2)

        create, select = result.statements
        self.assertEqual(create.kind, "create_table")
        self.assertEqual(len(create.changes), 1)
        self.assertEqual(create.changes[0].action, "create_table")
        self.assertEqual(create.changes[0].table, "public.users")
        self.assertFalse(create.changes[0].temp)

        self.assertEqual(select.kind, "query")
        self.assertEqual(column_names(select), ["id", "name"])
        self.assertEqual(
            column_types(select),
            {"id": exp.DataType.Type.INT, "name": exp.DataType.Type.TEXT},
        )
        self.assertEqual(
            snapshot_types(result.catalog, "public.users"),
            {"id": exp.DataType.Type.INT, "name": exp.DataType.Type.TEXT},
        )

    def test_ctas_uses_query_columns(self):
        result = analyze_script(
            "CREATE TABLE t AS SELECT id, id + 1 AS next_id FROM users",
            schema={"users": {"id": "INT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        create = result.statements[0]
        self.assertEqual(create.kind, "create_table_as")
        self.assertEqual(
            snapshot_types(result.catalog, "public.t"),
            {"id": exp.DataType.Type.INT, "next_id": exp.DataType.Type.INT},
        )

    def test_ctas_cte_is_not_permanent(self):
        result = analyze_script(
            """
            CREATE TABLE t AS WITH c AS (SELECT id FROM users) SELECT id FROM c;
            SELECT * FROM c;
            SELECT * FROM t;
            """,
            schema={"users": {"id": "INT"}},
            dialect="postgres",
            on_error="continue",
        )
        create, cte_query, table_query = result.statements
        self.assertTrue(create.ok)
        self.assertFalse(cte_query.ok)
        self.assertTrue(table_query.ok)
        self.assertEqual(column_names(table_query), ["id"])
        self.assertNotIn("public.c", result.catalog.tables)

    def test_alter_add_drop_rename_column(self):
        result = analyze_script(
            """
            CREATE TABLE t (a INT);
            ALTER TABLE t ADD COLUMN b TEXT;
            ALTER TABLE t RENAME COLUMN b TO c;
            ALTER TABLE t DROP COLUMN a;
            SELECT * FROM t;
            """,
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        actions = [change.action for s in result.statements for change in s.changes]
        self.assertEqual(actions, ["create_table", "add_column", "rename_column", "drop_column"])
        rename = result.statements[2].changes[0]
        self.assertEqual(rename.column, "b")
        self.assertEqual(rename.new_name, "c")
        select = result.statements[4]
        self.assertEqual(column_names(select), ["c"])
        self.assertEqual(column_types(select), {"c": exp.DataType.Type.TEXT})

    def test_alter_failures_leave_catalog_untouched(self):
        result = analyze_script(
            """
            CREATE TABLE t (a INT);
            ALTER TABLE t ADD COLUMN a TEXT;
            ALTER TABLE t RENAME COLUMN missing TO b;
            ALTER TABLE t DROP COLUMN missing;
            ALTER TABLE missing ADD COLUMN b INT;
            SELECT * FROM t;
            """,
            dialect="postgres",
            on_error="continue",
        )
        self.assertFalse(result.ok)
        for statement in result.statements[1:5]:
            self.assertFalse(statement.ok)
            self.assertIsNotNone(statement.error)
            self.assertEqual(statement.changes, [])
            self.assertEqual(set(statement.catalog_after.tables["public.t"]), {"a"})
        self.assertTrue(result.statements[5].ok)
        self.assertEqual(column_names(result.statements[5]), ["a"])

    def test_temp_table_shadowing(self):
        result = analyze_script(
            """
            CREATE TEMP TABLE users AS SELECT id FROM users;
            SELECT * FROM users;
            DROP TABLE users;
            SELECT * FROM users;
            """,
            schema={"users": {"id": "INT", "name": "TEXT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        create, shadowed, drop, unshadowed = result.statements

        self.assertTrue(create.changes[0].temp)
        self.assertEqual(create.changes[0].table, "pg_temp.users")
        self.assertEqual(column_names(shadowed), ["id"])
        self.assertIn("users", shadowed.catalog_after.temp_tables)

        self.assertEqual(drop.changes[0].action, "drop_table")
        self.assertTrue(drop.changes[0].temp)
        self.assertEqual(drop.catalog_after.temp_tables, {})

        self.assertEqual(column_names(unshadowed), ["id", "name"])
        self.assertEqual(
            snapshot_types(result.catalog, "public.users"),
            {"id": exp.DataType.Type.INT, "name": exp.DataType.Type.TEXT},
        )

    def test_drop_and_recreate(self):
        result = analyze_script(
            """
            CREATE TABLE t (a INT);
            DROP TABLE t;
            CREATE TABLE t (b TEXT);
            SELECT * FROM t;
            """,
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(column_names(result.statements[3]), ["b"])
        self.assertEqual(snapshot_types(result.catalog, "public.t"), {"b": exp.DataType.Type.TEXT})

    def test_quoted_identifiers(self):
        result = analyze_script(
            """
            CREATE TABLE "MyTable" ("MyCol" INT);
            SELECT "MyCol" FROM "MyTable";
            SELECT mycol FROM mytable;
            """,
            dialect="postgres",
            on_error="continue",
        )
        create, quoted, unquoted = result.statements
        self.assertTrue(create.ok)
        self.assertIn("public.MyTable", create.catalog_after.tables)
        self.assertTrue(quoted.ok)
        self.assertEqual(column_names(quoted), ["MyCol"])
        self.assertFalse(unquoted.ok)

    def test_search_path(self):
        schema = {
            "public": {"a": {"x": "INT"}},
            "analytics": {"a": {"y": "INT"}},
        }
        result = analyze_script("SELECT * FROM a", schema=schema, dialect="postgres")
        self.assertEqual(column_names(result.statements[0]), ["x"])

        result = analyze_script(
            "SELECT * FROM a",
            schema=schema,
            dialect="postgres",
            search_path=("analytics", "public"),
        )
        self.assertEqual(column_names(result.statements[0]), ["y"])

        result = analyze_script(
            "SELECT * FROM analytics.a; SELECT * FROM public.a",
            schema=schema,
            dialect="postgres",
        )
        self.assertEqual(column_names(result.statements[0]), ["y"])
        self.assertEqual(column_names(result.statements[1]), ["x"])

    def test_unknown_types_stay_unknown(self):
        result = analyze_script(
            "SELECT mystery_fn(id) AS m, id AS i FROM users",
            schema={"users": {"id": "INT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            column_types(result.statements[0]),
            {"m": exp.DataType.Type.UNKNOWN, "i": exp.DataType.Type.INT},
        )

    def test_unknown_table_fails_without_modifying_catalog(self):
        result = analyze_script(
            "SELECT * FROM missing; SELECT * FROM users",
            schema={"users": {"id": "INT"}},
            dialect="postgres",
            on_error="continue",
        )
        failed, ok = result.statements
        self.assertFalse(failed.ok)
        self.assertEqual(set(failed.catalog_after.tables), {"public.users"})
        self.assertTrue(ok.ok)

    def test_stop_on_error(self):
        result = analyze_script(
            "CREATE TABLE t (a INT); SELECT * FROM missing; SELECT 1 AS x",
            dialect="postgres",
            on_error="stop",
        )
        self.assertEqual(len(result.statements), 2)
        self.assertTrue(result.statements[0].ok)
        self.assertFalse(result.statements[1].ok)
        self.assertEqual(set(result.catalog.tables), {"public.t"})

    def test_if_exists_noops(self):
        result = analyze_script(
            """
            CREATE TABLE IF NOT EXISTS users (a INT);
            ALTER TABLE IF EXISTS missing ADD COLUMN b INT;
            DROP TABLE IF EXISTS missing;
            """,
            schema={"users": {"id": "INT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        for statement in result.statements:
            self.assertEqual(statement.changes, [])
            self.assertTrue(statement.diagnostics)
        self.assertEqual(
            snapshot_types(result.catalog, "public.users"), {"id": exp.DataType.Type.INT}
        )

    def test_duplicate_create_fails(self):
        result = analyze_script(
            "CREATE TABLE t (a INT); CREATE TABLE t (b INT)",
            dialect="postgres",
            on_error="continue",
        )
        self.assertFalse(result.statements[1].ok)
        self.assertEqual(snapshot_types(result.catalog, "public.t"), {"a": exp.DataType.Type.INT})

    def test_create_or_replace(self):
        result = analyze_script(
            "CREATE TABLE t (a INT); CREATE OR REPLACE TABLE t (b TEXT); SELECT * FROM t",
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(column_names(result.statements[2]), ["b"])

    def test_sessions_are_independent(self):
        schema = {"users": {"id": "INT"}}
        first = ScriptSession(schema, dialect="postgres")
        second = ScriptSession(schema, dialect="postgres")

        created = first.analyze("CREATE TABLE extra (a INT)")
        self.assertTrue(created.ok)
        self.assertIn("public.extra", created.catalog.tables)

        other = second.analyze("SELECT * FROM extra", on_error="continue")
        self.assertFalse(other.ok)
        self.assertNotIn("public.extra", second.catalog.tables)
        self.assertNotIn("extra", schema)

    def test_unsupported_statements_are_noops(self):
        result = analyze_script(
            "INSERT INTO users VALUES (1); CREATE INDEX i ON users (id); SELECT * FROM users",
            schema={"users": {"id": "INT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.statements[0].kind, "other")
        self.assertEqual(result.statements[1].kind, "other")
        self.assertTrue(result.statements[2].ok)

    def test_cte_shadows_catalog_table(self):
        result = analyze_script(
            "WITH users AS (SELECT 1 AS x) SELECT * FROM users",
            schema={"users": {"id": "INT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        self.assertEqual(column_names(result.statements[0]), ["x"])

    def test_multi_table_drop_is_atomic(self):
        result = analyze_script(
            """
            CREATE TABLE a (x INT);
            CREATE TABLE b (y INT);
            DROP TABLE a, missing;
            SELECT * FROM a;
            """,
            dialect="postgres",
            on_error="continue",
        )
        drop = result.statements[2]
        self.assertFalse(drop.ok)
        self.assertEqual(drop.changes, [])
        self.assertEqual(set(drop.catalog_after.tables), {"public.a", "public.b"})
        self.assertTrue(result.statements[3].ok)

    def test_stepwise_script(self):
        result = analyze_script(
            """
            CREATE TABLE orders (id INT, user_id INT);
            ALTER TABLE orders RENAME COLUMN user_id TO customer_id;
            CREATE TEMP TABLE users AS SELECT id FROM users;
            SELECT * FROM users;
            DROP TABLE users;
            SELECT * FROM users;
            DROP TABLE orders;
            CREATE TABLE orders (id INT, total NUMERIC);
            SELECT * FROM orders;
            """,
            schema={"users": {"id": "INT", "name": "TEXT"}},
            dialect="postgres",
        )
        self.assertTrue(result.ok)
        statements = result.statements
        self.assertEqual(len(statements), 9)

        self.assertEqual(statements[0].changes[0].action, "create_table")
        self.assertEqual(
            set(statements[0].catalog_after.tables["public.orders"]), {"id", "user_id"}
        )

        self.assertEqual(statements[1].changes[0].action, "rename_column")
        self.assertEqual(
            set(statements[1].catalog_after.tables["public.orders"]), {"id", "customer_id"}
        )

        self.assertTrue(statements[2].changes[0].temp)
        self.assertEqual(set(statements[2].catalog_after.temp_tables["users"]), {"id"})

        self.assertEqual(column_names(statements[3]), ["id"])

        self.assertEqual(statements[4].changes[0].action, "drop_table")
        self.assertEqual(statements[4].catalog_after.temp_tables, {})

        self.assertEqual(column_names(statements[5]), ["id", "name"])

        self.assertNotIn("public.orders", statements[6].catalog_after.tables)

        self.assertEqual(set(statements[7].catalog_after.tables["public.orders"]), {"id", "total"})

        self.assertEqual(column_names(statements[8]), ["id", "total"])
        self.assertEqual(
            column_types(statements[8]),
            {"id": exp.DataType.Type.INT, "total": exp.DataType.Type.DECIMAL},
        )


if __name__ == "__main__":
    unittest.main()
