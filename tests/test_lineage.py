from __future__ import annotations

import unittest

import sqlglot
from sqlglot.errors import SqlglotError
from sqlglot.lineage import DMLLineage, WriteKind, dml_lineage, lineage
from sqlglot.schema import MappingSchema

try:
    import duckdb
except ImportError:
    duckdb = None


class TestLineage(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        sqlglot.schema = MappingSchema()

    def test_lineage(self) -> None:
        node = lineage(
            "a",
            "SELECT a FROM z",
            schema={"x": {"a": "int"}},
            sources={"y": "SELECT * FROM x", "z": "SELECT a FROM y"},
        )
        self.assertEqual(
            node.source.sql(),
            "SELECT z.a AS a FROM (SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y /* source: y */) AS z /* source: z */",
        )
        self.assertEqual(node.source_name, "")

        downstream = node.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y /* source: y */",
        )
        self.assertEqual(downstream.source_name, "z")

        downstream = downstream.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT x.a AS a FROM x AS x",
        )
        self.assertEqual(downstream.source_name, "y")

        graph_html = node.to_html()
        self.assertGreater(len(graph_html._repr_html_()), 1000)

        for edge in graph_html.edges:
            self.assertIn("from", edge)
            self.assertIn("to", edge)

        # test that sql is not modified
        sql = "SELECT a FROM x"
        ast = sqlglot.parse_one(sql)
        node = lineage("a", ast)
        self.assertEqual(ast.sql(), sql)

        # test that sources are not modified
        ast = sqlglot.parse_one(sql)

        source_sql = "SELECT a FROM y"
        source = sqlglot.parse_one(source_sql)

        node = lineage("a", ast, sources={"x": source})

        self.assertEqual(source.sql(), source_sql)

    def test_lineage_sql_with_cte(self) -> None:
        node = lineage(
            "a",
            "WITH z AS (SELECT a FROM y) SELECT a FROM z",
            schema={"x": {"a": "int"}},
            sources={"y": "SELECT * FROM x"},
        )
        self.assertEqual(
            node.source.sql(),
            "WITH z AS (SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y /* source: y */) SELECT z.a AS a FROM z AS z",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")

        # Node containing expanded CTE expression
        downstream = node.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y /* source: y */",
        )
        self.assertEqual(downstream.source_name, "")
        self.assertEqual(downstream.reference_node_name, "z")

        downstream = downstream.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT x.a AS a FROM x AS x",
        )
        self.assertEqual(downstream.source_name, "y")
        self.assertEqual(downstream.reference_node_name, "")

    def test_lineage_source_with_cte(self) -> None:
        node = lineage(
            "a",
            "SELECT a FROM z",
            schema={"x": {"a": "int"}},
            sources={"z": "WITH y AS (SELECT * FROM x) SELECT a FROM y"},
        )
        self.assertEqual(
            node.source.sql(),
            "SELECT z.a AS a FROM (WITH y AS (SELECT x.a AS a FROM x AS x) SELECT y.a AS a FROM y AS y) AS z /* source: z */",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")

        downstream = node.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "WITH y AS (SELECT x.a AS a FROM x AS x) SELECT y.a AS a FROM y AS y",
        )
        self.assertEqual(downstream.source_name, "z")
        self.assertEqual(downstream.reference_node_name, "")

        downstream = downstream.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT x.a AS a FROM x AS x",
        )
        self.assertEqual(downstream.source_name, "z")
        self.assertEqual(downstream.reference_node_name, "y")

    def test_lineage_source_with_star(self) -> None:
        node = lineage(
            "a",
            "WITH y AS (SELECT * FROM x) SELECT a FROM y",
        )
        self.assertEqual(
            node.source.sql(),
            "WITH y AS (SELECT * FROM x AS x) SELECT y.a AS a FROM y AS y",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")

        downstream = node.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "SELECT * FROM x AS x",
        )
        self.assertEqual(downstream.source_name, "")
        self.assertEqual(downstream.reference_node_name, "y")

    def test_lineage_join_with_star(self) -> None:
        node = lineage(
            "*",
            "SELECT * from x JOIN y USING (uid)",
        )
        self.assertEqual(
            node.source.sql(),
            "SELECT * FROM x AS x JOIN y AS y ON x.uid = y.uid",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")
        self.assertEqual(len(node.downstream), 2)

        downstream = node.downstream[0]
        self.assertEqual(downstream.expression.sql(), "x AS x")
        self.assertEqual(downstream.name, "*")

        downstream = node.downstream[1]
        self.assertEqual(downstream.expression.sql(), "y AS y")
        self.assertEqual(downstream.name, "*")

    def test_lineage_join_with_qualified_star(self) -> None:
        node = lineage(
            "*",
            "SELECT x.* from x JOIN y USING (uid)",
        )
        self.assertEqual(
            node.source.sql(),
            "SELECT x.* FROM x AS x JOIN y AS y ON x.uid = y.uid",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")
        self.assertEqual(len(node.downstream), 1)

        downstream = node.downstream[0]
        self.assertEqual(downstream.expression.sql(), "x AS x")
        self.assertEqual(downstream.name, "x.*")

    def test_lineage_external_col(self) -> None:
        node = lineage(
            "a",
            "WITH y AS (SELECT * FROM x) SELECT a FROM y JOIN z USING (uid)",
        )
        self.assertEqual(
            node.source.sql(),
            "WITH y AS (SELECT * FROM x AS x) SELECT a AS a FROM y AS y JOIN z AS z ON y.uid = z.uid",
        )
        self.assertEqual(node.source_name, "")
        self.assertEqual(node.reference_node_name, "")

        downstream = node.downstream[0]
        self.assertEqual(
            downstream.source.sql(),
            "?",
        )
        self.assertEqual(downstream.source_name, "")
        self.assertEqual(downstream.reference_node_name, "")

    def test_lineage_values(self) -> None:
        node = lineage(
            "a",
            "SELECT a FROM y",
            sources={"y": "SELECT a FROM (VALUES (1), (2)) AS t (a)"},
        )
        self.assertEqual(
            node.source.sql(),
            "SELECT y.a AS a FROM (SELECT t.a AS a FROM (VALUES (1), (2)) AS t(a)) AS y /* source: y */",
        )
        self.assertEqual(node.source_name, "")

        downstream = node.downstream[0]
        self.assertEqual(downstream.source.sql(), "SELECT t.a AS a FROM (VALUES (1), (2)) AS t(a)")
        self.assertEqual(downstream.expression.sql(), "t.a AS a")
        self.assertEqual(downstream.source_name, "y")

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.source.sql(), "(VALUES (1), (2)) AS t(a)")
        self.assertEqual(downstream.expression.sql(), "a")
        self.assertEqual(downstream.source_name, "y")

    def test_lineage_cte_name_appears_in_schema(self) -> None:
        schema = {"a": {"b": {"t1": {"c1": "int"}, "t2": {"c2": "int"}}}}

        node = lineage(
            "c2",
            "WITH t1 AS (SELECT * FROM a.b.t2), inter AS (SELECT * FROM t1) SELECT * FROM inter",
            schema=schema,
        )

        self.assertEqual(
            node.source.sql(),
            "WITH t1 AS (SELECT t2.c2 AS c2 FROM a.b.t2 AS t2), inter AS (SELECT t1.c2 AS c2 FROM t1 AS t1) SELECT inter.c2 AS c2 FROM inter AS inter",
        )
        self.assertEqual(node.source_name, "")

        downstream = node.downstream[0]
        self.assertEqual(downstream.source.sql(), "SELECT t1.c2 AS c2 FROM t1 AS t1")
        self.assertEqual(downstream.expression.sql(), "t1.c2 AS c2")
        self.assertEqual(downstream.source_name, "")

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.source.sql(), "SELECT t2.c2 AS c2 FROM a.b.t2 AS t2")
        self.assertEqual(downstream.expression.sql(), "t2.c2 AS c2")
        self.assertEqual(downstream.source_name, "")

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.source.sql(), "a.b.t2 AS t2")
        self.assertEqual(downstream.expression.sql(), "a.b.t2 AS t2")
        self.assertEqual(downstream.source_name, "")

        self.assertEqual(downstream.downstream, [])

    def test_lineage_union(self) -> None:
        node = lineage(
            "x",
            "SELECT ax AS x FROM a UNION SELECT bx FROM b UNION SELECT cx FROM c",
        )
        assert len(node.downstream) == 3

        node = lineage(
            "x",
            "SELECT x FROM (SELECT ax AS x FROM a UNION SELECT bx FROM b UNION SELECT cx FROM c)",
        )
        assert len(node.downstream) == 3

    def test_lineage_lateral_flatten(self) -> None:
        node = lineage(
            "VALUE",
            "SELECT FLATTENED.VALUE FROM TEST_TABLE, LATERAL FLATTEN(INPUT => RESULT, OUTER => TRUE) FLATTENED",
            dialect="snowflake",
        )
        self.assertEqual(node.name, "VALUE")

        downstream = node.downstream[0]
        self.assertEqual(downstream.name, "FLATTENED.VALUE")
        self.assertEqual(
            downstream.source.sql(dialect="snowflake"),
            "LATERAL FLATTEN(INPUT => TEST_TABLE.RESULT, OUTER => TRUE) AS FLATTENED(SEQ, KEY, PATH, INDEX, VALUE, THIS)",
        )
        self.assertEqual(downstream.expression.sql(dialect="snowflake"), "VALUE")
        self.assertEqual(len(downstream.downstream), 1)

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.name, "TEST_TABLE.RESULT")
        self.assertEqual(downstream.source.sql(dialect="snowflake"), "TEST_TABLE AS TEST_TABLE")

        node = lineage(
            "FIELD",
            "SELECT FLATTENED.VALUE:field::text AS FIELD FROM SNOWFLAKE.SCHEMA.MODEL AS MODEL_ALIAS, LATERAL FLATTEN(INPUT => MODEL_ALIAS.A) AS FLATTENED",
            schema={"SNOWFLAKE": {"SCHEMA": {"TABLE": {"A": "integer"}}}},
            sources={"SNOWFLAKE.SCHEMA.MODEL": "SELECT A FROM SNOWFLAKE.SCHEMA.TABLE"},
            dialect="snowflake",
        )
        self.assertEqual(node.name, "FIELD")

        downstream = node.downstream[0]
        self.assertEqual(downstream.name, "FLATTENED.VALUE")
        self.assertEqual(
            downstream.source.sql(dialect="snowflake"),
            "LATERAL FLATTEN(INPUT => MODEL_ALIAS.A) AS FLATTENED(SEQ, KEY, PATH, INDEX, VALUE, THIS)",
        )
        self.assertEqual(downstream.expression.sql(dialect="snowflake"), "VALUE")
        self.assertEqual(len(downstream.downstream), 1)

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.name, "MODEL_ALIAS.A")
        self.assertEqual(downstream.source_name, "SNOWFLAKE.SCHEMA.MODEL")
        self.assertEqual(
            downstream.source.sql(dialect="snowflake"),
            "SELECT TABLE.A AS A FROM SNOWFLAKE.SCHEMA.TABLE AS TABLE",
        )
        self.assertEqual(downstream.expression.sql(dialect="snowflake"), "TABLE.A AS A")
        self.assertEqual(len(downstream.downstream), 1)

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.name, "TABLE.A")
        self.assertEqual(
            downstream.source.sql(dialect="snowflake"),
            "SNOWFLAKE.SCHEMA.TABLE AS TABLE",
        )
        self.assertEqual(
            downstream.expression.sql(dialect="snowflake"),
            "SNOWFLAKE.SCHEMA.TABLE AS TABLE",
        )

    def test_subquery(self) -> None:
        node = lineage(
            "output",
            "SELECT (SELECT max(t3.my_column) my_column FROM foo t3) AS output FROM table3",
        )
        self.assertEqual(node.name, "output")
        node = node.downstream[0]
        self.assertEqual(node.name, "my_column")
        node = node.downstream[0]
        self.assertEqual(node.name, "t3.my_column")
        self.assertEqual(node.source.sql(), "foo AS t3")

        node = lineage(
            "y",
            "SELECT SUM((SELECT max(a) a from x) + (SELECT min(b) b from x) + c) AS y FROM x",
        )
        self.assertEqual(node.name, "y")
        self.assertEqual(len(node.downstream), 3)
        self.assertEqual(node.downstream[0].name, "a")
        self.assertEqual(node.downstream[1].name, "b")
        self.assertEqual(node.downstream[2].name, "x.c")

        node = lineage(
            "x",
            "WITH cte AS (SELECT a, b FROM z) SELECT sum(SELECT a FROM cte) AS x, (SELECT b FROM cte) as y FROM cte",
        )
        self.assertEqual(node.name, "x")
        self.assertEqual(len(node.downstream), 1)
        node = node.downstream[0]
        self.assertEqual(node.name, "a")
        node = node.downstream[0]
        self.assertEqual(node.name, "cte.a")
        self.assertEqual(node.reference_node_name, "cte")
        node = node.downstream[0]
        self.assertEqual(node.name, "z.a")

        node = lineage(
            "a",
            """
            WITH foo AS (
              SELECT
                1 AS a
            ), bar AS (
              (
                SELECT
                  a + 1 AS a
                FROM foo
              )
            )
            (
              SELECT
                a + b AS a
              FROM bar
              CROSS JOIN (
                SELECT
                  2 AS b
              ) AS baz
            )
            """,
        )
        self.assertEqual(node.name, "a")
        self.assertEqual(len(node.downstream), 2)
        a, b = sorted(node.downstream, key=lambda n: n.name)
        self.assertEqual(a.name, "bar.a")
        self.assertEqual(len(a.downstream), 1)
        self.assertEqual(b.name, "baz.b")
        self.assertEqual(b.downstream, [])

        node = a.downstream[0]
        self.assertEqual(node.name, "foo.a")

        # Select from derived table
        node = lineage(
            "a",
            "SELECT a FROM (SELECT a FROM x) subquery",
        )
        self.assertEqual(node.name, "a")
        self.assertEqual(len(node.downstream), 1)
        node = node.downstream[0]
        self.assertEqual(node.name, "subquery.a")
        self.assertEqual(node.reference_node_name, "subquery")

        node = lineage(
            "a",
            "SELECT a FROM (SELECT a FROM x)",
        )
        self.assertEqual(node.name, "a")
        self.assertEqual(len(node.downstream), 1)
        node = node.downstream[0]
        self.assertEqual(node.name, "_0.a")
        self.assertEqual(node.reference_node_name, "_0")

    def test_lineage_cte_union(self) -> None:
        query = """
        WITH dataset AS (
            SELECT *
            FROM catalog.db.table_a

            UNION

            SELECT *
            FROM catalog.db.table_b
        )

        SELECT x, created_at FROM dataset;
        """
        node = lineage("x", query)

        self.assertEqual(node.name, "x")

        downstream_a = node.downstream[0]
        self.assertEqual(downstream_a.name, "0")
        self.assertEqual(downstream_a.source.sql(), "SELECT * FROM catalog.db.table_a AS table_a")
        self.assertEqual(downstream_a.reference_node_name, "dataset")
        downstream_b = node.downstream[1]
        self.assertEqual(downstream_b.name, "0")
        self.assertEqual(downstream_b.source.sql(), "SELECT * FROM catalog.db.table_b AS table_b")
        self.assertEqual(downstream_b.reference_node_name, "dataset")

    def test_lineage_no_self_loops_with_multi_aliased_union_cte(self) -> None:
        # A 3+ branch UNION CTE parses to nested SetOps. Referencing it twice
        # via different aliases used to leave the upstream node cached at the
        # inner SetOp's key, so the second alias's traversal would find that
        # node on a cache hit and append it to its own downstream.
        query = """
        WITH three_way AS (
            SELECT a FROM t1
            UNION ALL
            SELECT a FROM t2
            UNION ALL
            SELECT a FROM t3
        ),
        attached AS (
            SELECT COALESCE(r1.a, r2.a) AS a
            FROM three_way AS r1
            LEFT JOIN three_way AS r2 ON r1.a = r2.a
        )
        SELECT a FROM attached
        """

        node = lineage("a", query)

        visited: set[int] = set()
        stack = [(node, frozenset([id(node)]))]
        while stack:
            current, path = stack.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            for child in current.downstream:
                self.assertNotIn(
                    id(child),
                    path,
                    f"cycle detected at node '{child.name}'",
                )
                stack.append((child, path | {id(child)}))

    def test_lineage_source_union(self) -> None:
        query = "SELECT x, created_at FROM dataset;"
        node = lineage(
            "x",
            query,
            sources={
                "dataset": """
                SELECT *
                FROM catalog.db.table_a

                UNION

                SELECT *
                FROM catalog.db.table_b
                """
            },
        )

        self.assertEqual(node.name, "x")

        downstream_a = node.downstream[0]
        self.assertEqual(downstream_a.name, "0")
        self.assertEqual(downstream_a.source_name, "dataset")
        self.assertEqual(downstream_a.source.sql(), "SELECT * FROM catalog.db.table_a AS table_a")
        self.assertEqual(downstream_a.reference_node_name, "")
        downstream_b = node.downstream[1]
        self.assertEqual(downstream_b.name, "0")
        self.assertEqual(downstream_b.source_name, "dataset")
        self.assertEqual(downstream_b.source.sql(), "SELECT * FROM catalog.db.table_b AS table_b")
        self.assertEqual(downstream_b.reference_node_name, "")

    def test_select_star(self) -> None:
        node = lineage("x", "SELECT x from (SELECT * from table_a)")

        self.assertEqual(node.name, "x")

        downstream = node.downstream[0]
        self.assertEqual(downstream.name, "_0.x")
        self.assertEqual(downstream.source.sql(), "SELECT * FROM table_a AS table_a")

        downstream = downstream.downstream[0]
        self.assertEqual(downstream.name, "*")
        self.assertEqual(downstream.source.sql(), "table_a AS table_a")

    def test_unnest(self) -> None:
        node = lineage(
            "b",
            "with _data as (select [struct(1 as a, 2 as b)] as col) select b from _data cross join unnest(col)",
        )
        self.assertEqual(node.name, "b")

    def test_lineage_normalize(self) -> None:
        node = lineage("a", "WITH x AS (SELECT 1 a) SELECT a FROM x", dialect="snowflake")
        self.assertEqual(node.name, "A")

        with self.assertRaises(sqlglot.errors.SqlglotError):
            lineage('"a"', "WITH x AS (SELECT 1 a) SELECT a FROM x", dialect="snowflake")

        with self.assertRaises(sqlglot.errors.SqlglotError):
            lineage(
                "b",
                "SELECT a,b FROM table1 UNION ALL BY NAME SELECT a FROM table2",
                dialect="duckdb",
            )

    def test_ddl_lineage(self) -> None:
        sql = """
        INSERT /*+ HINT1 */
        INTO target (x, y)
        SELECT subq.x, subq.y
        FROM (
          SELECT /*+ HINT2 */
            t.x AS x,
            TO_DATE('2023-12-19', 'YYYY-MM-DD') AS y
          FROM s.t t
          WHERE 1 = 1 AND y = TO_DATE('2023-12-19', 'YYYY-MM-DD')
        ) subq
        """

        node = lineage("y", sql, dialect="oracle")

        self.assertEqual(node.name, "Y")
        self.assertEqual(node.expression.sql(dialect="oracle"), "SUBQ.Y AS Y")

        downstream = node.downstream[0]
        self.assertEqual(downstream.name, "SUBQ.Y")
        self.assertEqual(
            downstream.expression.sql(dialect="oracle"),
            "TO_DATE('2023-12-19', 'YYYY-MM-DD') AS Y",
        )

    def test_trim(self) -> None:
        sql = """
            SELECT a, b, c
            FROM (select a, b, c from y) z
        """

        node = lineage("a", sql, trim_selects=False)

        self.assertEqual(node.name, "a")
        self.assertEqual(
            node.source.sql(),
            "SELECT z.a AS a, z.b AS b, z.c AS c FROM (SELECT y.a AS a, y.b AS b, y.c AS c FROM y AS y) AS z",
        )

        downstream = node.downstream[0]
        self.assertEqual(downstream.name, "z.a")
        self.assertEqual(downstream.source.sql(), "SELECT y.a AS a, y.b AS b, y.c AS c FROM y AS y")

    def test_node_name_doesnt_contain_comment(self) -> None:
        sql = "SELECT * FROM (SELECT x /* c */ FROM t1) AS t2"
        node = lineage("x", sql)

        self.assertEqual(len(node.downstream), 1)
        self.assertEqual(len(node.downstream[0].downstream), 1)
        self.assertEqual(node.downstream[0].downstream[0].name, "t1.x")

    def test_pivot_without_alias(self) -> None:
        sql = """
        SELECT
            a as other_a
        FROM (select value,category from sample_data)
        PIVOT (
            sum(value)
            FOR category IN ('a', 'b')
        );
        """
        node = lineage("other_a", sql)

        self.assertEqual(node.downstream[0].name, "_0.value")
        self.assertEqual(node.downstream[0].downstream[0].name, "sample_data.value")

    def test_pivot_with_alias(self) -> None:
        sql = """
            SELECT
                cat_a_s as other_as
            FROM sample_data
            PIVOT (
                sum(value) as s, max(price)
                FOR category IN ('a' as cat_a, 'b')
            )
        """
        node = lineage("other_as", sql)

        self.assertEqual(len(node.downstream), 1)
        self.assertEqual(node.downstream[0].name, "sample_data.value")

    def test_pivot_with_cte(self) -> None:
        sql = """
        WITH t as (
            SELECT
                a as other_a
            FROM sample_data
            PIVOT (
                sum(value)
                FOR category IN ('a', 'b')
            )
        )
        select other_a from t
        """
        node = lineage("other_a", sql)

        self.assertEqual(node.downstream[0].name, "t.other_a")
        self.assertEqual(node.downstream[0].reference_node_name, "t")
        self.assertEqual(node.downstream[0].downstream[0].name, "sample_data.value")

    def test_pivot_with_implicit_column_of_pivoted_source(self) -> None:
        sql = """
        SELECT empid
        FROM quarterly_sales
            PIVOT(SUM(amount) FOR quarter IN (
            '2023_Q1',
            '2023_Q2',
            '2023_Q3'))
        ORDER BY empid;
        """
        node = lineage("empid", sql)

        self.assertEqual(node.downstream[0].name, "quarterly_sales.empid")

    def test_pivot_with_implicit_column_of_pivoted_source_and_cte(self) -> None:
        sql = """
        WITH t as (
            SELECT empid
            FROM quarterly_sales
            PIVOT(SUM(amount) FOR quarter IN (
                '2023_Q1',
                '2023_Q2',
                '2023_Q3'))
        )
        select empid from t
        """
        node = lineage("empid", sql)

        self.assertEqual(node.downstream[0].name, "t.empid")
        self.assertEqual(node.downstream[0].reference_node_name, "t")
        self.assertEqual(node.downstream[0].downstream[0].name, "quarterly_sales.empid")

    def test_unpivot(self) -> None:
        sql = """
        SELECT id, metric_name, score
        FROM sales UNPIVOT (score FOR metric_name IN (jan, feb))
        """
        schema = {"sales": {"id": "int", "jan": "int", "feb": "int"}}

        node = lineage("score", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.JAN", "SALES.FEB"])

        node = lineage("metric_name", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.JAN", "SALES.FEB"])

        node = lineage("id", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.ID"])

    def test_unpivot_with_cte(self) -> None:
        sql = """
        WITH src AS (
          SELECT id, jan, feb FROM sales
        )
        SELECT id, metric_name, score
        FROM src UNPIVOT (score FOR metric_name IN (jan, feb))
        """
        schema = {"sales": {"id": "int", "jan": "int", "feb": "int"}}

        node = lineage("score", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SRC.JAN", "SRC.FEB"])
        self.assertEqual(node.downstream[0].downstream[0].name, "SALES.JAN")
        self.assertEqual(node.downstream[1].downstream[0].name, "SALES.FEB")

        node = lineage("id", sql, schema=schema, dialect="snowflake")
        self.assertEqual(node.downstream[0].name, "SRC.ID")
        self.assertEqual(node.downstream[0].downstream[0].name, "SALES.ID")

    def test_unpivot_multi_column(self) -> None:
        sql = """
        SELECT product, semesters, first_half_sales, second_half_sales
        FROM produce
        UNPIVOT((first_half_sales, second_half_sales) FOR semesters IN ((q1, q2) AS 'semester_1', (q3, q4) AS 'semester_2'))
        """
        schema = {
            "produce": {
                "product": "string",
                "q1": "int64",
                "q2": "int64",
                "q3": "int64",
                "q4": "int64",
            }
        }

        node = lineage("first_half_sales", sql, schema=schema, dialect="bigquery")
        self.assertEqual([d.name for d in node.downstream], ["produce.q1", "produce.q3"])

        node = lineage("second_half_sales", sql, schema=schema, dialect="bigquery")
        self.assertEqual([d.name for d in node.downstream], ["produce.q2", "produce.q4"])

        node = lineage("semesters", sql, schema=schema, dialect="bigquery")
        self.assertEqual(
            [d.name for d in node.downstream],
            ["produce.q1", "produce.q2", "produce.q3", "produce.q4"],
        )

        node = lineage("product", sql, schema=schema, dialect="bigquery")
        self.assertEqual([d.name for d in node.downstream], ["produce.product"])

    def test_unpivot_with_alias_columns(self) -> None:
        sql = """
        WITH src AS (
          SELECT empid, dept, jan, feb FROM monthly_sales
        )
        SELECT m, s, e FROM src UNPIVOT(sales FOR month IN (jan, feb)) AS t(e, d, m, s)
        """
        schema = {"monthly_sales": {"empid": "int", "dept": "text", "jan": "int", "feb": "int"}}

        for renamed in ("s", "m"):
            node = lineage(renamed, sql, schema=schema, dialect="snowflake")
            self.assertEqual([d.name for d in node.downstream], ["SRC.JAN", "SRC.FEB"])
            self.assertEqual(node.downstream[0].downstream[0].name, "MONTHLY_SALES.JAN")
            self.assertEqual(node.downstream[1].downstream[0].name, "MONTHLY_SALES.FEB")

        node = lineage("e", sql, schema=schema, dialect="snowflake")
        self.assertEqual(node.downstream[0].name, "SRC.EMPID")
        self.assertEqual(node.downstream[0].downstream[0].name, "MONTHLY_SALES.EMPID")

        # Physical table source: the pre-pivot columns come from the schema
        sql = "SELECT m, s, e FROM monthly_sales UNPIVOT(sales FOR month IN (jan, feb)) AS t(e, d, m, s)"

        for renamed in ("s", "m"):
            node = lineage(renamed, sql, schema=schema, dialect="snowflake")
            self.assertEqual(
                [d.name for d in node.downstream], ["MONTHLY_SALES.JAN", "MONTHLY_SALES.FEB"]
            )

        node = lineage("e", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["MONTHLY_SALES.EMPID"])

        # Without a schema the star can't be expanded, so the positional renames are
        # unknowable and must not be applied (else `d` would misalign to jan/feb)
        sql = """
        WITH src AS (SELECT * FROM monthly_sales)
        SELECT d FROM src UNPIVOT(sales FOR month IN (jan, feb)) AS t(e, d, m, s)
        """
        node = lineage("d", sql, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SRC.D"])

    def test_chained_pivots(self) -> None:
        """Each operator in a chain consumes the previous one's output, so the mappings
        have to be folded rather than read off a single operator."""
        schema = {
            "sales": {"id": "int", "jan": "int", "feb": "int", "north": "int", "south": "int"}
        }
        sql = """
        SELECT id, score, headcount
        FROM sales UNPIVOT(score FOR month IN (jan, feb)) UNPIVOT(headcount FOR region IN (north, south))
        """

        node = lineage("score", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.JAN", "SALES.FEB"])

        node = lineage("headcount", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.NORTH", "SALES.SOUTH"])

        # A column that survives every operator still traces back to the source
        node = lineage("id", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["SALES.ID"])

    def test_chained_pivots_through_cte(self) -> None:
        schema = {
            "sales": {"id": "int", "jan": "int", "feb": "int", "north": "int", "south": "int"}
        }
        sql = """
        WITH src AS (SELECT id, jan, feb, north, south FROM sales)
        SELECT score FROM src
        UNPIVOT(score FOR month IN (jan, feb)) UNPIVOT(headcount FOR region IN (north, south))
        """
        node = lineage("score", sql, schema=schema, dialect="snowflake")

        self.assertEqual([d.name for d in node.downstream], ["SRC.JAN", "SRC.FEB"])
        self.assertEqual(node.downstream[0].downstream[0].name, "SALES.JAN")
        self.assertEqual(node.downstream[1].downstream[0].name, "SALES.FEB")

    def test_chained_pivots_mixed(self) -> None:
        # The UNPIVOT consumes columns the PIVOT produced, so they resolve through it
        schema = {"t": {"id": "int", "cat": "text", "val": "int"}}
        sql = """
        SELECT id, c, v FROM t
        PIVOT(SUM(val) FOR cat IN ('a' AS a, 'b' AS b)) UNPIVOT(v FOR c IN (a, b))
        """

        node = lineage("v", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["T.VAL", "T.VAL"])

        node = lineage("id", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["T.ID"])

    def test_chained_pivots_with_alias_columns(self) -> None:
        # The alias column list renames the chain's *final* output positionally, so a
        # column an earlier operator produced is reached under its renamed name
        schema = {
            "m": {
                "empid": "int",
                "dept": "text",
                "jan": "int",
                "feb": "int",
                "n": "int",
                "s": "int",
            }
        }
        sql = """
        SELECT a, b, c, d, e, f FROM m
        UNPIVOT(sales FOR mon IN (jan, feb)) UNPIVOT(hc FOR reg IN (n, s)) AS t(a, b, c, d, e, f)
        """
        expected = {
            "a": ["M.EMPID"],
            "b": ["M.DEPT"],
            "c": ["M.JAN", "M.FEB"],
            "d": ["M.JAN", "M.FEB"],
            "e": ["M.N", "M.S"],
            "f": ["M.N", "M.S"],
        }
        for column, downstream in expected.items():
            with self.subTest(column):
                node = lineage(column, sql, schema=schema, dialect="snowflake")
                self.assertEqual([d.name for d in node.downstream], downstream)

    def test_chained_pivots_consuming_alias_columns(self) -> None:
        # An earlier operator's alias column list renames passthroughs (b -> north,
        # c -> south), and a later operator consumes them under the new names
        schema = {
            "sales": {"id": "int", "jan": "int", "feb": "int", "north": "int", "south": "int"}
        }
        sql = """
        SELECT hc, region FROM sales
        UNPIVOT(score FOR month IN (jan, feb)) AS u1(a, b, c, d)
        UNPIVOT(hc FOR region IN (b, c))
        """
        for column in ("hc", "region"):
            with self.subTest(column):
                node = lineage(column, sql, schema=schema, dialect="duckdb")
                self.assertEqual([d.name for d in node.downstream], ["sales.north", "sales.south"])

        # Multi-value form: the value columns are derived positionally from the entries
        sql = """
        SELECT hi, lo FROM sales
        UNPIVOT(score FOR month IN (jan, feb)) AS u1(a, b, c, d)
        UNPIVOT((hi, lo) FOR region IN ((b, c)))
        """
        node = lineage("hi", sql, schema=schema, dialect="duckdb")
        self.assertEqual([d.name for d in node.downstream], ["sales.north"])

        node = lineage("lo", sql, schema=schema, dialect="duckdb")
        self.assertEqual([d.name for d in node.downstream], ["sales.south"])

    def test_multiple_pivoted_sources(self) -> None:
        # Pivots over different sources don't form a chain, so folding them would trace
        # `hc` through the other source's `val` output; degrade to leaves instead
        schema = {
            "t1": {"id": "int", "jan": "int", "feb": "int"},
            "t2": {"id": "int", "val": "int", "other": "int"},
        }
        sql = """
        SELECT s1.val, s2.hc
        FROM t1 UNPIVOT(val FOR m IN (jan, feb)) AS s1
        JOIN t2 UNPIVOT(hc FOR r IN (val, other)) AS s2 ON s1.id = s2.id
        """
        node = lineage("hc", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["S2.HC"])

        node = lineage("val", sql, schema=schema, dialect="snowflake")
        self.assertEqual([d.name for d in node.downstream], ["S1.VAL"])

    def test_pivot_with_alias_columns(self) -> None:
        sql = """
        SELECT x FROM (SELECT value, category FROM sample_data) AS sd
        PIVOT (SUM(value) FOR category IN ('a', 'b')) AS p(x, y)
        """
        node = lineage("x", sql)

        self.assertEqual(node.downstream[0].name, "sd.value")
        self.assertEqual(node.downstream[0].downstream[0].name, "sample_data.value")

    def test_table_udtf_snowflake(self) -> None:
        lateral_flatten = """
        SELECT f.value:external_id::string AS external_id
        FROM database_name.schema_name.table_name AS raw,
        LATERAL FLATTEN(events) AS f
        """
        table_flatten = """
        SELECT f.value:external_id::string AS external_id
        FROM database_name.schema_name.table_name AS raw
        JOIN TABLE(FLATTEN(events)) AS f
        """

        lateral_node = lineage("external_id", lateral_flatten, dialect="snowflake")
        table_node = lineage("external_id", table_flatten, dialect="snowflake")

        self.assertEqual(lateral_node.name, "EXTERNAL_ID")
        self.assertEqual(table_node.name, "EXTERNAL_ID")

        lateral_node = lateral_node.downstream[0]
        table_node = table_node.downstream[0]

        self.assertEqual(lateral_node.name, "F.VALUE")
        self.assertEqual(
            lateral_node.source.sql("snowflake"),
            "LATERAL FLATTEN(RAW.EVENTS) AS F(SEQ, KEY, PATH, INDEX, VALUE, THIS)",
        )

        self.assertEqual(table_node.name, "F.VALUE")
        self.assertEqual(table_node.source.sql("snowflake"), "TABLE(FLATTEN(RAW.EVENTS)) AS F")

        lateral_node = lateral_node.downstream[0]
        table_node = table_node.downstream[0]

        self.assertEqual(lateral_node.name, "RAW.EVENTS")
        self.assertEqual(
            lateral_node.source.sql("snowflake"),
            "DATABASE_NAME.SCHEMA_NAME.TABLE_NAME AS RAW",
        )

        self.assertEqual(table_node.name, "RAW.EVENTS")
        self.assertEqual(
            table_node.source.sql("snowflake"),
            "DATABASE_NAME.SCHEMA_NAME.TABLE_NAME AS RAW",
        )

    def test_table_udtfs(self) -> None:
        sql = """
        WITH t AS (
          SELECT p.user_id, gs.n AS to_deposit_days
          FROM tbl1 AS p
          CROSS JOIN (
            SELECT generate_series AS n FROM TABLE(generate_series(0, 8000))
          ) AS gs
        ),
        t2 AS (
          SELECT t.* FROM t
        )
        SELECT user_id, to_deposit_days FROM t2
        """
        schema = {"tbl1": {"user_id": "BIGINT"}}

        node = lineage(
            "to_deposit_days",
            sql,
            schema=schema,
            dialect="starrocks",
            expand_stars=True,
            validate_qualify_columns=True,
        )

        node = node.downstream[0].downstream[0].downstream[0].downstream[0]
        self.assertEqual(node.name, "_0.generate_series")
        self.assertEqual(
            node.source.sql("starrocks"),
            "TABLE(GENERATE_SERIES(0, 8000)) AS _0(generate_series)",
        )

    def test_pivot_with_subquery(self) -> None:
        schema = {
            "loan_ledger": {
                "product_type": "varchar",
                "month": "date",
                "loan_id": "int",
            }
        }

        sql = """
        WITH cte AS (
            SELECT * FROM (
                SELECT product_type, month, loan_id
                FROM loan_ledger
            ) PIVOT (
                COUNT(loan_id) FOR month IN ('2024-10', '2024-11')
            )
        )
        SELECT
            cte.product_type AS product_type,
            cte."2024-10" AS "2024-10"
        FROM cte
        """

        node = lineage("product_type", sql, dialect="duckdb", schema=schema)
        self.assertEqual(node.downstream[0].name, "cte.product_type")
        self.assertEqual(node.downstream[0].downstream[0].name, "_0.product_type")
        self.assertEqual(
            node.downstream[0].downstream[0].downstream[0].name,
            "loan_ledger.product_type",
        )

        node = lineage('"2024-10"', sql, dialect="duckdb", schema=schema)
        self.assertEqual(node.downstream[0].name, "cte.2024-10")
        self.assertEqual(node.downstream[0].downstream[0].name, "_0.loan_id")
        self.assertEqual(node.downstream[0].downstream[0].downstream[0].name, "loan_ledger.loan_id")

    def test_copy_flag(self) -> None:
        schema = {
            "x": {
                "a": "int",
            },
        }

        query = sqlglot.parse_one("SELECT a FROM z")
        sources = {
            "y": sqlglot.parse_one("SELECT * FROM x"),
            "z": sqlglot.parse_one("SELECT * FROM y"),
        }

        lineage("a", query, schema=schema, sources=sources, copy=False)

        self.assertEqual(sources["y"].sql(), "SELECT * FROM x")
        self.assertEqual(sources["z"].sql(), "SELECT * FROM y")
        self.assertEqual(
            query.sql(),
            "SELECT z.a AS a FROM (SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y /* source: y */) AS z /* source: z */",
        )

        query = sqlglot.parse_one("SELECT a FROM z")
        sources = {
            "y": sqlglot.parse_one("SELECT * FROM x"),
            "z": sqlglot.parse_one("SELECT * FROM y"),
        }

        lineage("a", query, schema=schema, sources=sources, copy=True)

        self.assertEqual(sources["y"].sql(), "SELECT * FROM x")
        self.assertEqual(sources["z"].sql(), "SELECT * FROM y")
        self.assertEqual(query.sql(), "SELECT a FROM z")

        query = sqlglot.parse_one("SELECT a FROM z")
        sources = {
            "y": sqlglot.parse_one("SELECT * FROM x"),
            "z": sqlglot.parse_one("SELECT * FROM y"),
        }

        query = sqlglot.parse_one("SELECT a FROM x")
        lineage("a", query, schema=schema, copy=False)

        self.assertEqual(query.sql(), "SELECT x.a AS a FROM x AS x")

        query = sqlglot.parse_one("SELECT a FROM x")
        lineage("a", query, schema=schema, copy=True)

        self.assertEqual(query.sql(), "SELECT a FROM x")

    def test_lineage_shared_cte_performance(self) -> None:
        """Shared CTEs referenced from multiple places should not cause exponential expansion.

        Each cte_k joins cte_{k-1} with itself and references both sides
        (t1.a + t2.a), so without memoization to_node() is called 2^N times.
        With N=12 that's 4096 expansions.
        """
        n_levels = 12
        ctes = ["cte_0 AS (SELECT a FROM base_table)"]
        for k in range(1, n_levels):
            prev = f"cte_{k - 1}"
            ctes.append(
                f"cte_{k} AS (SELECT t1.a + t2.a AS a FROM {prev} t1 JOIN {prev} t2 ON t1.a = t2.a)"
            )
        sql = "WITH " + ",\n     ".join(ctes) + f"\nSELECT a FROM cte_{n_levels - 1}"

        node = lineage("a", sql, schema={"base_table": {"a": "int"}})

        # Walk the DAG and verify structure.
        all_nodes = list(node.walk())

        # shared references keep node count small (O(N), not O(2^N)).
        self.assertLess(
            len(all_nodes),
            200,
            f"got {len(all_nodes)} nodes -- DAG walk may be broken",
        )

        # walk() should yield each node exactly once.
        all_ids = [id(n) for n in all_nodes]
        self.assertEqual(len(all_ids), len(set(all_ids)))

        # Leaf nodes should reference base_table.
        leaves = [n for n in all_nodes if not n.downstream]
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all("base_table" in n.source.sql() for n in leaves))

    def test_lineage_cte_self_join_distinct_aliases(self) -> None:
        node = lineage(
            "combined",
            "WITH shared AS (SELECT a FROM x) SELECT s1.a + s2.a AS combined FROM shared s1, shared s2",
            schema={"x": {"a": "int"}},
        )
        downstream_names = sorted(d.name for d in node.downstream)
        self.assertEqual(downstream_names, ["s1.a", "s2.a"])

    def test_lineage_all_columns(self) -> None:
        # column=None returns dict[name, Node] for every top-level output column.
        result = lineage(
            None,
            "SELECT a, b + 1 AS bp FROM x",
            schema={"x": {"a": "int", "b": "int"}},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), {"a", "bp"})

        # Each entry is a Node; same shape as single-column lineage().
        single = lineage(
            "a", "SELECT a, b + 1 AS bp FROM x", schema={"x": {"a": "int", "b": "int"}}
        )
        self.assertEqual(result["a"].name, single.name)
        self.assertEqual(
            [d.name for d in result["a"].downstream], [d.name for d in single.downstream]
        )

    def test_lineage_on_node_hook(self) -> None:
        nodes_visited: list[str] = []

        def hook(node):
            nodes_visited.append(node.name)
            node.payload["seen"] = True

        result = lineage(
            None,
            "WITH t AS (SELECT a + 1 AS v FROM x) SELECT v FROM t",
            schema={"x": {"a": "int"}},
            on_node=hook,
        )
        # Top-level v + CTE-internal t.v + base leaf x.a => 3 fires minimum.
        self.assertGreaterEqual(len(nodes_visited), 3)
        self.assertTrue(result["v"].payload.get("seen"))

        # Walk the chain: every node in the DAG has the payload set.
        self.assertTrue(all(n.payload.get("seen") for n in result["v"].walk()))

    def test_lineage_all_columns_shares_nodes_across_outputs(self) -> None:
        # Different output columns referencing the same upstream column should
        # share the same downstream Node — the per-call cache deduplicates
        # repeated traversals of the same (scope, column, parent context).
        result = lineage(
            None,
            "WITH t AS (SELECT a, b FROM x) SELECT a, a + b AS ab FROM t",
            schema={"x": {"a": "int", "b": "int"}},
        )
        self.assertEqual(set(result), {"a", "ab"})

        # `a` flows directly through t.a; `ab` references t.a too. The Node
        # representing t.a should be the same Python object in both chains.
        a_via_a = result["a"].downstream[0]
        a_via_ab = next(d for d in result["ab"].downstream if d.name == "t.a")
        self.assertIs(a_via_a, a_via_ab)

        # And the deeper x.a leaf is shared as well.
        self.assertIs(a_via_a.downstream[0], a_via_ab.downstream[0])

    def test_lineage_all_columns_set_operation(self) -> None:
        # UNION CTE referenced from a top-level select. Each output column
        # should fan out to one downstream per branch of the UNION.
        result = lineage(
            None,
            """
            WITH u AS (SELECT a, b FROM x UNION ALL SELECT a, b FROM y)
            SELECT a, b FROM u
            """,
            schema={"x": {"a": "int", "b": "int"}, "y": {"a": "int", "b": "int"}},
        )
        self.assertEqual(set(result), {"a", "b"})

        for col in ("a", "b"):
            branches = result[col].downstream
            self.assertEqual(len(branches), 2)
            sources = sorted(
                leaf.expression.sql()
                for branch in branches
                for leaf in branch.walk()
                if leaf.expression.sql().endswith("AS x") or leaf.expression.sql().endswith("AS y")
            )
            self.assertEqual(sources, ["x AS x", "y AS y"])

    def test_lineage_all_columns_with_prebuilt_scope(self) -> None:
        # When a pre-qualified scope is passed, lineage must not re-qualify;
        # results should match the no-scope path.
        from sqlglot.optimizer import qualify
        from sqlglot.optimizer.scope import build_scope

        sql = "SELECT a, b + 1 AS bp FROM x"
        schema = {"x": {"a": "int", "b": "int"}}

        # Mirror lineage()'s internal qualify defaults so the two paths
        # produce structurally identical Node trees.
        qualified = qualify.qualify(
            sqlglot.parse_one(sql),
            schema=schema,
            validate_qualify_columns=False,
            identify=False,
        )
        scope = build_scope(qualified)

        from_scope = lineage(None, qualified, scope=scope)
        from_sql = lineage(None, sql, schema=schema)

        self.assertEqual(set(from_scope), set(from_sql))
        for name in from_scope:
            self.assertEqual(from_scope[name].name, from_sql[name].name)
            self.assertEqual(
                [d.name for d in from_scope[name].downstream],
                [d.name for d in from_sql[name].downstream],
            )

    def test_lineage_on_node_orders_children_before_parents(self) -> None:
        # The on_node callback must fire on a Node only after its downstream
        # has been fully populated — i.e. all of its children have already
        # received their on_node call. This is the invariant that lets
        # callers populate payload bottom-up from already-finalized children.
        order: list[int] = []

        def hook(node):
            order.append(id(node))

        result = lineage(
            None,
            "WITH t AS (SELECT a + 1 AS v FROM x) SELECT v FROM t",
            schema={"x": {"a": "int"}},
            on_node=hook,
        )

        position = {nid: i for i, nid in enumerate(order)}
        for node in result["v"].walk():
            for child in node.downstream:
                self.assertLess(
                    position[id(child)],
                    position[id(node)],
                    f"{child.name!r} fired after parent {node.name!r}",
                )

    def test_lineage_on_node_fires_once_per_node(self) -> None:
        # Caching means each Node is created exactly once even if reached
        # from multiple parents; the on_node callback must reflect that.
        counts: dict[int, int] = {}

        def hook(node):
            counts[id(node)] = counts.get(id(node), 0) + 1

        result = lineage(
            None,
            "WITH t AS (SELECT a, b FROM x) SELECT a, a + b AS ab FROM t",
            schema={"x": {"a": "int", "b": "int"}},
            on_node=hook,
        )

        # Walk every reachable Node from both output columns; each should
        # have been emitted to the hook exactly once.
        seen: set[int] = set()
        for top in result.values():
            for node in top.walk():
                seen.add(id(node))
        for nid in seen:
            self.assertEqual(counts.get(nid, 0), 1)


class TestDMLLineage(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        sqlglot.schema = MappingSchema()

    SCHEMA = {
        "t": {"id": "int", "a": "int", "b": "int"},
        "s": {"id": "int", "x": "int", "y": "int", "flag": "int"},
        "s2": {"id": "int", "x": "int"},
    }

    @staticmethod
    def leaf_names(node):
        return sorted({n.name for n in node.walk() if n is not node and not n.downstream})

    def test_insert_select(self) -> None:
        result = dml_lineage(
            "INSERT INTO t (id, a) SELECT id, x FROM s", schema=self.SCHEMA, dialect="postgres"
        )
        self.assertIsInstance(result, DMLLineage)
        self.assertEqual(result.target, "t")
        self.assertEqual(len(result.branches), 1)

        branch = result.branches[0]
        self.assertEqual(branch.action, "INSERT")
        self.assertIsNone(branch.condition)
        self.assertEqual(set(branch.columns), {"id", "a"})
        self.assertEqual(branch.columns["id"].kind, WriteKind.EXPRESSION)
        self.assertEqual(self.leaf_names(branch.columns["id"]), ["s.id"])
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["s.x"])

    def test_insert_select_schema_columns_and_star(self) -> None:
        # No column list: the target columns come from the schema, in order
        result = dml_lineage(
            "INSERT INTO t SELECT * FROM st",
            schema={**self.SCHEMA, "st": {"id": "int", "x": "int", "y": "int"}},
            dialect="postgres",
        )
        branch = result.branches[0]
        self.assertEqual(list(branch.columns), ["id", "a", "b"])
        self.assertEqual(self.leaf_names(branch.columns["id"]), ["st.id"])
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["st.x"])
        self.assertEqual(self.leaf_names(branch.columns["b"]), ["st.y"])

        # Constants in the source query are not expressions
        result = dml_lineage(
            "INSERT INTO t (id, a) SELECT 1, x FROM s", schema=self.SCHEMA, dialect="postgres"
        )
        branch = result.branches[0]
        self.assertEqual(branch.columns["id"].kind, WriteKind.CONSTANT)
        self.assertEqual(branch.columns["id"].downstream, [])
        self.assertEqual(branch.columns["a"].kind, WriteKind.EXPRESSION)

    def test_insert_values(self) -> None:
        result = dml_lineage(
            "INSERT INTO t (id, a, b) VALUES (1, DEFAULT, 2), (3, 4, 5)",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        branch = result.branches[0]
        self.assertEqual(set(branch.columns), {"id", "a", "b"})

        for node in branch.columns.values():
            self.assertEqual(node.kind, WriteKind.VALUES)
            self.assertEqual(len(node.downstream), 2)

        self.assertEqual(
            [n.kind for n in branch.columns["id"].downstream],
            [WriteKind.CONSTANT, WriteKind.CONSTANT],
        )
        self.assertEqual(
            [n.kind for n in branch.columns["a"].downstream],
            [WriteKind.DEFAULT, WriteKind.CONSTANT],
        )

    def test_insert_default_values(self) -> None:
        result = dml_lineage("INSERT INTO t DEFAULT VALUES", schema=self.SCHEMA, dialect="postgres")
        branch = result.branches[0]
        self.assertEqual(set(branch.columns), {"id", "a", "b"})
        for node in branch.columns.values():
            self.assertEqual(node.kind, WriteKind.DEFAULT)

    def test_insert_errors(self) -> None:
        # Column count mismatch between the target list and the source query
        with self.assertRaises(SqlglotError) as ctx:
            dml_lineage(
                "INSERT INTO t (id, a) SELECT id FROM s", schema=self.SCHEMA, dialect="postgres"
            )
        self.assertIn("Column count mismatch", str(ctx.exception))
        self.assertIn("t", str(ctx.exception))

        # Column count mismatch in a VALUES row
        with self.assertRaises(SqlglotError) as ctx:
            dml_lineage(
                "INSERT INTO t (id, a) VALUES (1, 2, 3)", schema=self.SCHEMA, dialect="postgres"
            )
        self.assertIn("Column count mismatch", str(ctx.exception))

        # No column list and the target table is not in the schema
        with self.assertRaises(SqlglotError) as ctx:
            dml_lineage(
                "INSERT INTO t SELECT * FROM s", schema={"s": self.SCHEMA["s"]}, dialect="postgres"
            )
        self.assertIn("target columns", str(ctx.exception))

        # The star cannot be expanded without a schema for the source
        with self.assertRaises(SqlglotError) as ctx:
            dml_lineage(
                "INSERT INTO t SELECT * FROM s", schema={"t": self.SCHEMA["t"]}, dialect="postgres"
            )
        self.assertIn("'*'", str(ctx.exception))

        # Not a DML statement
        with self.assertRaises(SqlglotError):
            dml_lineage("SELECT * FROM s", schema=self.SCHEMA, dialect="postgres")

    def test_update_swap_columns(self) -> None:
        # Each assignment reads the pre-update row: a <- old b, b <- old a.
        # The first SET must not become the source of the second.
        result = dml_lineage("UPDATE t SET a = b, b = a", schema=self.SCHEMA, dialect="postgres")
        branch = result.branches[0]
        self.assertEqual(branch.action, "UPDATE")
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["t.b"])
        self.assertEqual(self.leaf_names(branch.columns["b"]), ["t.a"])

        # Untouched target columns keep their old values
        self.assertEqual(branch.columns["id"].kind, WriteKind.RETAINED)

    def test_update_from(self) -> None:
        result = dml_lineage(
            "UPDATE t SET a = s.x, b = a + 1 FROM s WHERE t.id = s.id",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        branch = result.branches[0]
        self.assertEqual(branch.condition.sql(), "t.id = s.id")
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["s.x"])

        # The unqualified reference reads the pre-update target row, not the
        # column assigned by the previous SET
        self.assertEqual(self.leaf_names(branch.columns["b"]), ["t.a"])
        self.assertEqual(branch.columns["id"].kind, WriteKind.RETAINED)

        # Joins in the FROM clause
        result = dml_lineage(
            "UPDATE t SET a = s.x + s2.x FROM s JOIN s2 ON s.id = s2.id WHERE t.id = s.id",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        self.assertEqual(self.leaf_names(result.branches[0].columns["a"]), ["s.x", "s2.x"])

    def test_update_default_and_constant(self) -> None:
        result = dml_lineage(
            "UPDATE t SET a = DEFAULT, b = 0", schema=self.SCHEMA, dialect="postgres"
        )
        branch = result.branches[0]
        self.assertEqual(branch.columns["a"].kind, WriteKind.DEFAULT)
        self.assertEqual(branch.columns["b"].kind, WriteKind.CONSTANT)
        self.assertEqual(branch.columns["id"].kind, WriteKind.RETAINED)

    def test_merge(self) -> None:
        result = dml_lineage(
            "MERGE INTO t USING s ON t.id = s.id "
            "WHEN MATCHED AND s.flag = 1 THEN UPDATE SET a = s.x "
            "WHEN NOT MATCHED THEN INSERT (id, a, b) VALUES (s.id, s.y, 0) "
            "WHEN NOT MATCHED BY SOURCE THEN DELETE",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        self.assertEqual(result.target, "t")
        self.assertEqual(result.on.sql(), "t.id = s.id")
        self.assertEqual([b.action for b in result.branches], ["UPDATE", "INSERT", "DELETE"])
        self.assertEqual([b.matched for b in result.branches], [True, False, False])
        self.assertEqual([b.source for b in result.branches], [False, False, True])

        update, insert, delete = result.branches

        # The WHEN condition is a control dependency of the branch, kept apart
        # from the column value sources
        self.assertEqual(update.condition.sql(), "s.flag = 1")
        self.assertEqual(self.leaf_names(update.columns["a"]), ["s.x"])
        self.assertEqual(update.columns["id"].kind, WriteKind.RETAINED)
        self.assertEqual(update.columns["b"].kind, WriteKind.RETAINED)

        self.assertIsNone(insert.condition)
        self.assertEqual(self.leaf_names(insert.columns["id"]), ["s.id"])
        self.assertEqual(self.leaf_names(insert.columns["a"]), ["s.y"])
        self.assertEqual(insert.columns["b"].kind, WriteKind.CONSTANT)

        self.assertEqual(delete.columns, {})

        # Control dependencies never leak into the value sources
        for node in result.walk():
            self.assertNotIn("flag", node.name)

    def test_merge_insert_schema_columns(self) -> None:
        # No column list on the INSERT branch: columns come from the schema
        result = dml_lineage(
            "MERGE INTO t USING s ON t.id = s.id "
            "WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.x, 0)",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        branch = result.branches[0]
        self.assertEqual(list(branch.columns), ["id", "a", "b"])

        with self.assertRaises(SqlglotError) as ctx:
            dml_lineage(
                "MERGE INTO t USING s ON t.id = s.id "
                "WHEN NOT MATCHED THEN INSERT (id) VALUES (s.id, s.x)",
                schema=self.SCHEMA,
                dialect="postgres",
            )
        self.assertIn("Column count mismatch", str(ctx.exception))

    def test_dml_cte_union_subquery_and_shadowing(self) -> None:
        # CTEs and UNIONs are traced through; every UNION branch contributes
        result = dml_lineage(
            "WITH c AS (SELECT id, x FROM s2) INSERT INTO t (id, a) "
            "SELECT c.id, c.x FROM c UNION ALL SELECT id, x FROM s",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        branch = result.branches[0]
        self.assertEqual(self.leaf_names(branch.columns["id"]), ["s.id", "s2.id"])
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["s.x", "s2.x"])

        # A CTE shadows the physical table of the same name
        result = dml_lineage(
            "WITH s AS (SELECT id, x FROM s2) UPDATE t SET a = s.x FROM s WHERE t.id = s.id",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        self.assertEqual(self.leaf_names(result.branches[0].columns["a"]), ["s2.x"])

        # Derived tables and correlated subqueries
        result = dml_lineage(
            "UPDATE t SET a = q.x FROM (SELECT x, id FROM s) AS q WHERE t.id = q.id",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        self.assertEqual(self.leaf_names(result.branches[0].columns["a"]), ["s.x"])

        result = dml_lineage(
            "UPDATE t SET a = (SELECT MAX(x) FROM s WHERE s.id = t.id)",
            schema=self.SCHEMA,
            dialect="postgres",
        )
        self.assertEqual(self.leaf_names(result.branches[0].columns["a"]), ["s.x"])

    def test_dml_lineage_on_node(self) -> None:
        nodes = []
        dml_lineage(
            "UPDATE t SET a = b", schema=self.SCHEMA, dialect="postgres", on_node=nodes.append
        )
        self.assertGreater(len(nodes), 0)
        self.assertEqual(len({id(n) for n in nodes}), len(nodes))

    @unittest.skipUnless(duckdb, "duckdb is not installed")
    def test_dml_lineage_duckdb_writes(self) -> None:
        # The graph's target columns are checked against what DuckDB actually writes.
        # (DuckDB has no MERGE support, so the executable checks cover UPDATE and
        # INSERT ... SELECT; MERGE branches are verified structurally above.)
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (id INT, a INT, b INT)")
        conn.execute("INSERT INTO t VALUES (1, 10, 100), (2, 20, 200)")
        conn.execute("CREATE TABLE s (id INT, x INT, y INT, flag INT)")
        conn.execute("INSERT INTO s VALUES (1, 7, 70, 1), (3, 9, 90, 0)")

        # UPDATE swapping two columns: the graph says a <- old b and b <- old a
        sql = "UPDATE t SET a = b, b = a"
        result = dml_lineage(sql, schema=self.SCHEMA, dialect="duckdb")
        branch = result.branches[0]
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["t.b"])
        self.assertEqual(self.leaf_names(branch.columns["b"]), ["t.a"])
        self.assertEqual(branch.columns["id"].kind, WriteKind.RETAINED)

        before = {row[0]: row for row in conn.execute("SELECT id, a, b FROM t").fetchall()}
        conn.execute(sql)
        for id_, a, b in conn.execute("SELECT id, a, b FROM t").fetchall():
            self.assertEqual(a, before[id_][2])
            self.assertEqual(b, before[id_][1])

        # INSERT ... SELECT: the graph maps target columns to the source columns
        # whose values DuckDB writes
        sql = "INSERT INTO t (id, a) SELECT id, x FROM s WHERE flag = 1"
        result = dml_lineage(sql, schema=self.SCHEMA, dialect="duckdb")
        branch = result.branches[0]
        self.assertEqual(self.leaf_names(branch.columns["id"]), ["s.id"])
        self.assertEqual(self.leaf_names(branch.columns["a"]), ["s.x"])

        conn.execute(sql)
        self.assertEqual(
            conn.execute("SELECT id, a, b FROM t WHERE id = 1 AND a = 7").fetchall(),
            [(1, 7, None)],
        )
