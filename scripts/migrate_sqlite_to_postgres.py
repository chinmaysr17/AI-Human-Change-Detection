
import sqlite3
from pathlib import Path

import psycopg
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQLITE_PATH = PROJECT_ROOT / "data" / "analysis_history.db"

config = dotenv_values(PROJECT_ROOT / ".env")

COLUMNS = (
    "id",
    "latitude",
    "longitude",
    "radius_km",
    "before_date",
    "after_date",
    "created_at",
    "status",
    "imagery_retrieved",
)

COLUMN_SQL = ", ".join(COLUMNS)
PLACEHOLDERS = ", ".join(["%s"] * len(COLUMNS))


def main():
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(SQLITE_PATH)

    source = sqlite3.connect(
        SQLITE_PATH.as_uri() + "?mode=ro",
        uri=True,
    )
    source.row_factory = sqlite3.Row

    try:
        integrity = source.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                f"SQLite integrity check failed: {integrity}"
            )

        source_rows = source.execute(
            f"SELECT {COLUMN_SQL} FROM analysis_history ORDER BY id"
        ).fetchall()

        records = [
            tuple(row[column] for column in COLUMNS)
            for row in source_rows
        ]

        print("SQLite integrity:", integrity)
        print("Source records:", len(records))

        if len(records) != 21:
            raise RuntimeError(
                f"Expected 21 records, found {len(records)}"
            )

        target = psycopg.connect(
            host=config["DB_HOST"],
            port=config["DB_PORT"],
            dbname=config["DB_NAME"],
            user=config["DB_USER"],
            password=config["DB_PASSWORD"],
            autocommit=True,
        )

        try:
            existing_count = target.execute(
                "SELECT COUNT(*) FROM analysis_history"
            ).fetchone()[0]

            if existing_count != 0:
                raise RuntimeError(
                    "PostgreSQL history is not empty. "
                    "Migration stopped."
                )

            with target.transaction():

                insert_sql = (
                    f"INSERT INTO analysis_history ({COLUMN_SQL}) "
                    f"VALUES ({PLACEHOLDERS})"
                )

                with target.cursor() as cursor:
                    cursor.executemany(insert_sql, records)

                destination_rows = target.execute(
                    f"SELECT {COLUMN_SQL} "
                    "FROM analysis_history ORDER BY id"
                ).fetchall()

                destination_records = [
                    tuple(row) for row in destination_rows
                ]

                if records != destination_records:
                    raise RuntimeError(
                        "Record verification failed. Rolling back."
                    )

                unassigned = target.execute(
                    "SELECT COUNT(*) FROM analysis_history "
                    "WHERE user_id IS NULL"
                ).fetchone()[0]

                if unassigned != 21:
                    raise RuntimeError(
                        "Legacy user verification failed."
                    )

                target.execute("""
                    SELECT setval(
                        pg_get_serial_sequence(
                            'analysis_history', 'id'
                        ),
                        COALESCE(MAX(id), 1),
                        COUNT(*) > 0
                    )
                    FROM analysis_history
                """)

            print("Migration: SUCCESS")
            print("PostgreSQL records:", len(destination_records))
            print("Records preserved exactly: YES")
            print("Legacy records unassigned: YES")
            print("Original IDs preserved: YES")
            print("Sequence synchronized: YES")

        finally:
            target.close()

    finally:
        source.close()


if __name__ == "__main__":
    main()
