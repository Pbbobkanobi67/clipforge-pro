"""Migration: Add layout_config column to reframe_configs table.

Idempotent -- safe to run multiple times.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "storage" / "video_extract_pro.db"


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def table_exists(cursor, table: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cursor.fetchone() is not None


def migrate():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run the server first to create it.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # --- Add layout_config column to reframe_configs ---
    if table_exists(cursor, "reframe_configs"):
        if not column_exists(cursor, "reframe_configs", "layout_config"):
            cursor.execute("ALTER TABLE reframe_configs ADD COLUMN layout_config TEXT DEFAULT '{}'")
            print("  Added column reframe_configs.layout_config")
        else:
            print("  Column reframe_configs.layout_config already exists, skipping")
    else:
        print("  Table reframe_configs does not exist yet. Run the server first to create it.")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
