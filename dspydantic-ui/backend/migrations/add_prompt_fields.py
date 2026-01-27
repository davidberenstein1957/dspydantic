"""
Migration script to add system_prompt and instruction_prompt_template columns to tasks table.
Run this script to update the database schema.
"""
import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dspydantic_ui"
)

def run_migration():
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='tasks' AND column_name IN ('system_prompt', 'instruction_prompt_template')
        """))
        existing_columns = {row[0] for row in result}
        
        # Add system_prompt if it doesn't exist
        if 'system_prompt' not in existing_columns:
            print("Adding system_prompt column...")
            conn.execute(text("ALTER TABLE tasks ADD COLUMN system_prompt TEXT"))
            conn.commit()
            print("✓ Added system_prompt column")
        else:
            print("✓ system_prompt column already exists")
        
        # Add instruction_prompt_template if it doesn't exist
        if 'instruction_prompt_template' not in existing_columns:
            print("Adding instruction_prompt_template column...")
            conn.execute(text("ALTER TABLE tasks ADD COLUMN instruction_prompt_template TEXT"))
            conn.commit()
            print("✓ Added instruction_prompt_template column")
        else:
            print("✓ instruction_prompt_template column already exists")
        
        print("\nMigration completed successfully!")

if __name__ == "__main__":
    try:
        run_migration()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
