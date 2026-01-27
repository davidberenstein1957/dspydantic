from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    input_schema = Column(JSON, nullable=True)  # Schema for input fields (string, pdf, image)
    pydantic_schema = Column(JSON, nullable=False)  # Output schema
    system_prompt = Column(Text, nullable=True)
    instruction_prompt_template = Column(Text, nullable=True)
    default_model = Column(String, nullable=True)  # Default model for output generation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    examples = relationship(
        "LabeledExample", back_populates="task", cascade="all, delete-orphan"
    )
    optimization_runs = relationship(
        "OptimizationRun", back_populates="task", cascade="all, delete-orphan"
    )
    evaluation_runs = relationship(
        "EvaluationRun", back_populates="task", cascade="all, delete-orphan"
    )
    prompt_versions = relationship(
        "PromptVersion", back_populates="task", cascade="all, delete-orphan"
    )


class LabeledExample(Base):
    __tablename__ = "labeled_examples"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)
    # e.g., "approved", "rejected", "pending", "reviewed"
    status = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    task = relationship("Task", back_populates="examples")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    config = Column(JSON, nullable=False)  # optimization configuration
    metrics = Column(JSON)  # results metrics
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    progress = Column(Float, default=0.0)

    task = relationship("Task", back_populates="optimization_runs")
    prompt_versions = relationship("PromptVersion", back_populates="optimization_run")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    prompt_version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    config = Column(JSON, nullable=False)  # evaluation configuration
    metrics = Column(JSON)  # results metrics
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    progress = Column(Float, default=0.0)

    task = relationship("Task", back_populates="evaluation_runs")
    prompt_version = relationship("PromptVersion", back_populates="evaluation_runs")
    example_results = relationship(
        "EvaluationExampleResult",
        back_populates="evaluation_run",
        cascade="all, delete-orphan"
    )


class EvaluationExampleResult(Base):
    __tablename__ = "evaluation_example_results"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False)
    example_id = Column(Integer, ForeignKey("labeled_examples.id"), nullable=False)
    score = Column(Float, nullable=False)  # Score between 0.0 and 1.0
    extracted_output = Column(JSON, nullable=True)  # The extracted/actual output
    expected_output = Column(JSON, nullable=True)  # The expected output
    error_message = Column(Text, nullable=True)  # Error message if extraction failed
    differences = Column(JSON, nullable=True)  # Detailed differences between extracted and expected

    evaluation_run = relationship("EvaluationRun", back_populates="example_results")
    example = relationship("LabeledExample")


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    optimization_run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=True)
    parent_version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=True)
    version_number = Column(Integer, nullable=False)
    system_prompt = Column(Text, nullable=True)
    instruction_prompt = Column(Text, nullable=True)
    prompt_content = Column(Text, nullable=True)  # Legacy field, kept for backward compatibility
    # Optimized field descriptions for output schema
    output_schema_descriptions = Column(JSON, nullable=True)
    metrics = Column(JSON)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String, nullable=True)  # User or system identifier

    task = relationship("Task", back_populates="prompt_versions")
    optimization_run = relationship("OptimizationRun", back_populates="prompt_versions")
    parent_version = relationship("PromptVersion", remote_side=[id], backref="child_versions")
    evaluation_runs = relationship("EvaluationRun", back_populates="prompt_version")
