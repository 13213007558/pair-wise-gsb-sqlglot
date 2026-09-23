import unittest

from sqlglot.analysis import analyze_script
from sqlglot.errors import ErrorLevel, OptimizeError


class TestAnalysis(unittest.TestCase):
    def test_postgres_script_catalog_lifecycle(self):
        result = analyze_script(
            """
            CREATE TABLE seed (id INT);
            CREATE TABLE things (id INT, name TEXT);
            ALTER TABLE things ADD COLUMN note TEXT;
            ALTER TABLE things RENAME COLUMN name TO label;
            CREATE TEMP TABLE things AS SELECT id, label FROM things;
            ALTER TABLE things DROP COLUMN label;
            SELECT id FROM things;
            DROP TABLE things;
            SELECT id, label, note FROM things;
            DROP TABLE things;
            CREATE TABLE things AS SELECT id + 1 AS next_id FROM seed;
            """
        )

        self.assertEqual(len(result.statements), 11)
        self.assertEqual(
            [change.operation for change in result.statements[2].changes], ["ADD_COLUMN"]
        )
        self.assertEqual(
            [change.operation for change in result.statements[3].changes], ["RENAME_COLUMN"]
        )

        temp_create = result.statements[4]
        self.assertTrue(temp_create.changes[0].table.temporary)
        self.assertEqual([column.name for column in temp_create.output_columns], ["id", "label"])

        temp_drop = result.statements[7]
        self.assertTrue(temp_drop.changes[0].table.temporary)

        restored_select = result.statements[8]
        self.assertEqual(
            [
                (column.name, column.type.sql("postgres"))
                for column in restored_select.output_columns
            ],
            [("id", "INT"), ("label", "TEXT"), ("note", "TEXT")],
        )

        permanent_drop = result.statements[9]
        self.assertFalse(permanent_drop.changes[0].table.temporary)

        recreated = result.statements[10]
        self.assertFalse(recreated.changes[0].table.temporary)
        self.assertEqual(recreated.output_columns[0].name, "next_id")

    def test_ctas_scope_and_inferred_types(self):
        result = analyze_script(
            """
            CREATE TABLE source (id INT);
            CREATE TEMP TABLE target AS
            WITH source(id) AS (SELECT 1)
            SELECT id, some_function(id) AS inferred, NULL AS missing
            FROM source;
            SELECT * FROM target;
            """
        )

        ctas = result.statements[1]
        self.assertEqual(
            [column.type.sql("postgres") for column in ctas.output_columns],
            ["INT", "UNKNOWN", "UNKNOWN"],
        )

        select = result.statements[2]
        self.assertEqual(
            [column.name for column in select.output_columns], ["id", "inferred", "missing"]
        )

    def test_ctas_does_not_register_cte(self):
        result = analyze_script(
            """
            CREATE TABLE source (id INT, value TEXT);
            CREATE TEMP TABLE target AS
            WITH source(id) AS (SELECT 1)
            SELECT id FROM source;
            SELECT value FROM source;
            """
        )

        self.assertEqual(result.statements[2].output_columns[0].name, "value")
        self.assertEqual({key.name for key in result.catalog.tables()}, {"source", "target"})

        with self.assertRaises(OptimizeError):
            analyze_script(
                """
                CREATE TABLE source (id INT);
                CREATE TEMP TABLE target AS
                WITH source(id) AS (SELECT missing)
                SELECT id FROM source;
                """
            )

    def test_quoted_names_and_search_path(self):
        initial = {"app": {'"Existing"': {'"MixedColumn"': "INT"}}}
        result = analyze_script(
            'CREATE TABLE app."Mixed" ("Value" INT); SELECT "Value" FROM "Mixed";',
            schema=initial,
            search_path="app, public",
        )

        change = result.statements[0].changes[0]
        self.assertEqual(change.table.schema, "app")
        self.assertEqual(change.table.name, "Mixed")
        self.assertEqual(change.columns[0].name, "Value")
        self.assertEqual(result.statements[1].output_columns[0].name, "Value")
        self.assertIn("Existing", {key.name for key in result.catalog.tables()})
        existing = next(
            value for key, value in result.catalog.tables().items() if key.name == "Existing"
        )
        self.assertEqual(existing[0].name, "MixedColumn")

    def test_create_uses_first_schema_in_search_path(self):
        result = analyze_script(
            'CREATE TABLE "Later" (id INT);',
            schema={"public": {"Later": {"id": "INT"}}},
            search_path="app, public",
        )

        key = result.statements[0].changes[0].table
        self.assertEqual(key.schema, "app")

    def test_failed_statement_is_atomic_and_continues(self):
        result = analyze_script(
            """
            CREATE TABLE things (id INT, name TEXT);
            ALTER TABLE things ADD COLUMN id BIGINT, ADD COLUMN duplicate_name TEXT;
            SELECT * FROM things;
            """,
            error_level=ErrorLevel.IGNORE,
        )

        failed = result.statements[1]
        self.assertIsInstance(failed.error.error, OptimizeError)
        self.assertEqual(failed.changes, ())
        self.assertEqual(list(failed.catalog_after.tables()), list(failed.catalog_before.tables()))

        select = result.statements[2]
        self.assertEqual([column.name for column in select.output_columns], ["id", "name"])

    def test_sessions_are_independent(self):
        initial = {"base": {"id": "INT"}}
        first = analyze_script("CREATE TABLE first (id INT);", schema=initial)
        second = analyze_script("CREATE TABLE second (id INT);", schema=initial)

        first_tables = {key.name: value for key, value in first.catalog.tables().items()}
        second_tables = {key.name: value for key, value in second.catalog.tables().items()}
        self.assertEqual(set(first_tables), {"base", "first"})
        self.assertEqual(set(second_tables), {"base", "second"})
        self.assertNotIn("first", second_tables)
        self.assertNotIn("second", first_tables)

    def test_each_statement_records_catalog_change(self):
        result = analyze_script(
            """
            CREATE TABLE old_name (id INT);
            ALTER TABLE old_name RENAME COLUMN id TO renamed_id;
            CREATE TEMP TABLE old_name AS SELECT renamed_id FROM old_name;
            DROP TABLE old_name;
            """
        )

        for statement in result.statements:
            self.assertTrue(statement.changes)
            self.assertNotEqual(statement.catalog_before, statement.catalog_after)


if __name__ == "__main__":
    unittest.main()
