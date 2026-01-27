import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dspydantic_ui"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database: create tables and add missing columns."""
    Base.metadata.create_all(bind=engine)

    # Add missing columns if they don't exist (for existing databases)
    inspector = inspect(engine)
    if inspector.has_table("tasks"):
        with engine.connect() as conn:
            existing_columns = {col["name"] for col in inspector.get_columns("tasks")}

            if "system_prompt" not in existing_columns:
                conn.execute(text("ALTER TABLE tasks ADD COLUMN system_prompt TEXT"))
                conn.commit()

            if "instruction_prompt_template" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE tasks ADD COLUMN instruction_prompt_template TEXT")
                )
                conn.commit()

            if "input_schema" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE tasks ADD COLUMN input_schema JSON")
                )
                conn.commit()

            if "default_model" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE tasks ADD COLUMN default_model VARCHAR")
                )
                conn.commit()

    # Add missing columns to prompt_versions table
    if inspector.has_table("prompt_versions"):
        with engine.connect() as conn:
            existing_columns = {
                col["name"] for col in inspector.get_columns("prompt_versions")
            }

            if "parent_version_id" not in existing_columns:
                conn.execute(
                    text(
                        "ALTER TABLE prompt_versions "
                        "ADD COLUMN parent_version_id INTEGER "
                        "REFERENCES prompt_versions(id)"
                    )
                )
                conn.commit()

            if "system_prompt" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE prompt_versions ADD COLUMN system_prompt TEXT")
                )
                conn.commit()

            if "instruction_prompt" not in existing_columns:
                conn.execute(
                    text(
                        "ALTER TABLE prompt_versions ADD COLUMN instruction_prompt TEXT"
                    )
                )
                conn.commit()

            if "created_by" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE prompt_versions ADD COLUMN created_by VARCHAR")
                )
                conn.commit()

            if "output_schema_descriptions" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE prompt_versions ADD COLUMN output_schema_descriptions JSON")
                )
                conn.commit()

    # Add missing columns to labeled_examples table
    if inspector.has_table("labeled_examples"):
        with engine.connect() as conn:
            existing_columns = {
                col["name"] for col in inspector.get_columns("labeled_examples")
            }

            if "status" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE labeled_examples ADD COLUMN status VARCHAR")
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_labeled_examples_status "
                        "ON labeled_examples(status)"
                    )
                )
                conn.commit()

    # Create evaluation_example_results table if it doesn't exist
    if not inspector.has_table("evaluation_example_results"):
        Base.metadata.create_all(bind=engine, tables=[Base.metadata.tables.get("evaluation_example_results")])
    elif inspector.has_table("evaluation_example_results"):
        # Table exists, check for missing columns
        with engine.connect() as conn:
            existing_columns = {
                col["name"] for col in inspector.get_columns("evaluation_example_results")
            }

            if "differences" not in existing_columns:
                conn.execute(
                    text("ALTER TABLE evaluation_example_results ADD COLUMN differences JSON")
                )
                conn.commit()
