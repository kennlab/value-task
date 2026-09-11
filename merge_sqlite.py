#!/usr/bin/env python3

import argparse
import shutil
import sqlite3
from pathlib import Path


def get_tables(conn, schema="main"):
    cur = conn.execute(
        f"""
        SELECT name
        FROM {schema}.sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [row[0] for row in cur.fetchall()]


def get_columns(conn, table, schema="main"):
    cur = conn.execute(f'PRAGMA {schema}.table_info("{table}")')
    return [row[1] for row in cur.fetchall()]


def get_primary_key_columns(conn, table, schema="main"):
    cur = conn.execute(f'PRAGMA {schema}.table_info("{table}")')
    rows = cur.fetchall()
    return [row[1] for row in rows if row[5] > 0]


def table_has_column(conn, table, column, schema="main"):
    return column in get_columns(conn, table, schema)


def max_id_for_column(conn, column_name, schema="main"):
    max_id = None
    for table in get_tables(conn, schema):
        if table_has_column(conn, table, column_name, schema):
            row = conn.execute(
                f'SELECT MAX("{column_name}") FROM {schema}."{table}"'
            ).fetchone()
            value = row[0] if row else None
            if value is not None:
                max_id = value if max_id is None else max(max_id, value)
    return max_id


def merge_subjects(conn):
    if "subjects" not in get_tables(conn, "main") or "subjects" not in get_tables(conn, "src"):
        return

    main_cols = get_columns(conn, "subjects", "main")
    src_cols = get_columns(conn, "subjects", "src")
    cols = [c for c in main_cols if c in src_cols]
    if not cols:
        return

    # Fixed argument order here
    pk_cols = get_primary_key_columns(conn, "subjects", "main")

    col_list = ", ".join(f'"{c}"' for c in cols)

    if pk_cols:
        key_col = pk_cols[0]
        if key_col in cols:
            sql = f'''
                INSERT OR IGNORE INTO main."subjects" ({col_list})
                SELECT {col_list}
                FROM src."subjects"
            '''
            conn.execute(sql)
        else:
            sql = f'''
                INSERT OR IGNORE INTO main."subjects" ({col_list})
                SELECT {col_list}
                FROM src."subjects"
            '''
            conn.execute(sql)
    else:
        sql = f'''
            INSERT OR IGNORE INTO main."subjects" ({col_list})
            SELECT {col_list}
            FROM src."subjects"
        '''
        conn.execute(sql)

    print("Merged table: subjects")


def merge_other_tables(conn, session_offset, event_offset):
    main_tables = set(get_tables(conn, "main"))
    src_tables = get_tables(conn, "src")
    common_tables = [t for t in src_tables if t in main_tables and t != "subjects"]

    for table in common_tables:
        main_cols = get_columns(conn, table, "main")
        src_cols = get_columns(conn, table, "src")
        cols = [c for c in main_cols if c in src_cols]
        if not cols:
            continue

        insert_cols = ", ".join(f'"{c}"' for c in cols)

        select_exprs = []
        for col in cols:
            if col == "sessionid":
                select_exprs.append(f'src."{col}" + {session_offset} AS "{col}"')
            elif col == "eventid":
                select_exprs.append(f'src."{col}" + {event_offset} AS "{col}"')
            else:
                select_exprs.append(f'src."{col}"')

        select_sql = ", ".join(select_exprs)

        sql = f'''
            INSERT INTO main."{table}" ({insert_cols})
            SELECT {select_sql}
            FROM src."{table}" AS src
        '''
        conn.execute(sql)
        print(f"Merged table: {table}")


def merge_dbs(db1_path, db2_path, out_path):
    shutil.copyfile(db1_path, out_path)

    conn = sqlite3.connect(out_path)
    conn.execute("PRAGMA foreign_keys = OFF;")

    try:
        conn.execute("ATTACH DATABASE ? AS src", (str(db2_path),))

        base_max_sessionid = max_id_for_column(conn, "sessionid", "main")
        base_max_eventid = max_id_for_column(conn, "eventid", "main")

        session_offset = 0 if base_max_sessionid is None else base_max_sessionid + 1
        event_offset = 0 if base_max_eventid is None else base_max_eventid + 1

        with conn:
            merge_subjects(conn)
            merge_other_tables(conn, session_offset, event_offset)

        conn.execute("DETACH DATABASE src")

    finally:
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two SQLite databases with safe ID remapping."
    )
    parser.add_argument("db1", type=Path, help="Base SQLite database.")
    parser.add_argument("db2", type=Path, help="SQLite database to merge in.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output database.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.db1.exists():
        raise FileNotFoundError(args.db1)
    if not args.db2.exists():
        raise FileNotFoundError(args.db2)

    merge_dbs(args.db1, args.db2, args.output)
    print(f"\nMerged database written to '{args.output}'")


if __name__ == "__main__":
    main()