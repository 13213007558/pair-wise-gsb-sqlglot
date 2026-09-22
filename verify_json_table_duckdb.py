"""Manual verification for MySQL JSON_TABLE -> DuckDB transpilation.

Builds a small table (empty arrays, missing fields, numeric strings, JSON null,
rows with different JSON shapes), transpiles a MySQL JSON_TABLE query with
sqlglot, executes the generated SQL in DuckDB and checks the result row by row.

Usage: pip install duckdb && python3 verify_json_table_duckdb.py
"""

import duckdb
import sqlglot

MYSQL_QUERY = """
SELECT s.id, jt.rid, jt.link, jt.nid, jt.nval
FROM source AS s,
JSON_TABLE(
    s.links,
    '$.org[*]' DEFAULT '0' ON ERROR DEFAULT '0' ON EMPTY
    COLUMNS(
        rid FOR ORDINALITY,
        link VARCHAR(10) PATH '$.link',
        NESTED PATH '$.items[*]' COLUMNS(nid FOR ORDINALITY, nval DOUBLE PATH '$.n')
    )
) AS jt
WHERE s.id > 0
ORDER BY s.id, jt.rid, jt.nid
"""

# (id, rid, link, nid, nval); ordinality is 1-based, nested misses left-join to NULL,
# JSON null stays SQL NULL, missing paths / failed casts fall back to DEFAULT '0'
EXPECTED = [
    (1, 1, "a", 1, 1.5),   # numeric string '1.5' -> 1.5
    (1, 1, "a", 2, 2.0),
    (1, 2, "b", None, None),  # empty nested array -> NULL nested columns
    (3, 1, "c", None, None),  # missing nested array -> NULL nested columns
    (4, 1, "0", 1, 0.0),   # missing link -> '0'; 'abc' fails DOUBLE cast -> 0.0
    (4, 1, "0", 2, 0.0),   # missing n -> ON EMPTY default 0.0
    (5, 1, None, 1, 0.0),  # JSON null link -> SQL NULL; null element -> empty default
]

duckdb_sql = sqlglot.parse_one(MYSQL_QUERY, read="mysql").sql("duckdb")
print("Generated DuckDB SQL:", duckdb_sql, "\n")

con = duckdb.connect()
con.execute("CREATE TABLE source (id INTEGER, links JSON)")
con.executemany(
    "INSERT INTO source VALUES (?, ?)",
    [
        (0, '{"org": [{"link": "z"}]}'),  # filtered out by WHERE s.id > 0
        (1, '{"org": [{"link": "a", "items": [{"n": "1.5"}, {"n": "2"}]}, {"link": "b", "items": []}]}'),
        (2, '{"org": []}'),  # empty parent array -> no rows
        (3, '{"org": [{"link": "c"}]}'),
        (4, '{"org": [{"items": [{"n": "abc"}, {}]}]}'),
        (5, '{"org": [{"link": null, "items": [null]}]}'),
    ],
)

rows = con.execute(duckdb_sql).fetchall()
for i, (row, expected) in enumerate(zip(rows, EXPECTED)):
    status = "OK" if row == expected else "MISMATCH"
    print(f"row {i}: {row} expected {expected} -> {status}")

assert rows == EXPECTED, f"row mismatch:\nactual   {rows}\nexpected {EXPECTED}"
print(f"All {len(rows)} rows match.")
