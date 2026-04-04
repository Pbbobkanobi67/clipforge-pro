"""Migration: Add clip management columns and clip_exports table.

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

    # --- Add columns to clip_suggestions ---
    new_columns = [
        ("parent_clip_id", "VARCHAR(36)"),
        ("version_label", "VARCHAR(200)"),
        ("is_manual", "BOOLEAN DEFAULT 0"),
    ]
    for col_name, col_type in new_columns:
        if not column_exists(cursor, "clip_suggestions", col_name):
            cursor.execute(f"ALTER TABLE clip_suggestions ADD COLUMN {col_name} {col_type}")
            print(f"  Added column clip_suggestions.{col_name}")
        else:
            print(f"  Column clip_suggestions.{col_name} already exists, skipping")

    # --- Create clip_exports table ---
    if not table_exists(cursor, "clip_exports"):
        cursor.execute("""
            CREATE TABLE clip_exports (
                id VARCHAR(36) PRIMARY KEY,
                clip_id VARCHAR(36) NOT NULL REFERENCES clip_suggestions(id),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                export_path VARCHAR(1000) NOT NULL,
                format VARCHAR(10) DEFAULT 'mp4',
                file_size_bytes INTEGER,
                settings_json TEXT DEFAULT '{}',
                label VARCHAR(200)
            )
        """)
        cursor.execute("CREATE INDEX idx_clip_exports_clip_id ON clip_exports(clip_id)")
        print("  Created table clip_exports")
    else:
        print("  Table clip_exports already exists, skipping")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
