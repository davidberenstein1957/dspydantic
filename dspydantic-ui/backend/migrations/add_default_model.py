"""
Migration script to add default_model column to tasks table.
Run this script to update the database schema.
"""
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dspydantic_ui"
)

def run_migration():
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='tasks' AND column_name='default_model'
        """))
        existing = result.fetchone()

        # Add default_model if it doesn't exist
        if not existing:
            print("Adding default_model column...")
            conn.execute(text("ALTER TABLE tasks ADD COLUMN default_model VARCHAR"))
            conn.commit()
            print("✓ Added default_model column")
        else:
            print("✓ default_model column already exists")

        print("\nMigration completed successfully!")

if __name__ == "__main__":
    try:
        run_migration()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
