
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LabeledExample, OptimizationRun, PromptVersion, Task
from app.services.prompt_validation import validate_jinja2_template, validate_system_prompt
from app.services.schema_service import schema_to_json_schema, validate_data_against_schema

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def validate_schema_field_names(schema: dict, schema_name: str) -> None:
    """
    Validate that all field names in a schema are unique (case-insensitive).
    Raises HTTPException if duplicates are found.
    """
    if not schema or not isinstance(schema, dict):
        return

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return

    field_names = list(properties.keys())

    # Check for case-insensitive duplicates
    seen_lower = {}
    case_duplicates = []

    for name in field_names:
        name_lower = name.lower()
        if name_lower in seen_lower:
            case_duplicates.append(f"'{name}' and '{seen_lower[name_lower]}'")
        else:
            seen_lower[name_lower] = name

    if case_duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"{schema_name} contains duplicate field names (case-insensitive): {', '.join(case_duplicates)}. Field names must be unique."
        )

    # Recursively check nested objects
    for field_name, field_def in properties.items():
        if isinstance(field_def, dict):
            # Check nested object properties
            if field_def.get("type") == "object" and "properties" in field_def:
                nested_schema = {"properties": field_def["properties"]}
                validate_schema_field_names(nested_schema, f"{schema_name}.{field_name}")

            # Check array item properties if it's an object
            if field_def.get("type") == "array" and "items" in field_def:
                items = field_def["items"]
                if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
                    nested_schema = {"properties": items["properties"]}
                    validate_schema_field_names(nested_schema, f"{schema_name}.{field_name}[]")


class TaskCreate(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict | None = None
    pydantic_schema: dict
    system_prompt: str | None = None
    instruction_prompt_template: str | None = None
    default_model: str | None = None


class TaskDuplicateRequest(BaseModel):
    new_name: str
    copy_examples: bool = True


class TaskUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    input_schema: dict | None = None
    pydantic_schema: dict | None = None
    system_prompt: str | None = None
    instruction_prompt_template: str | None = None
    default_model: str | None = None


class TaskResponse(BaseModel):
    id: int
    name: str
    description: str | None
    input_schema: dict | None
    pydantic_schema: dict
    system_prompt: str | None
    instruction_prompt_template: str | None
    default_model: str | None = None
    created_at: str
    example_count: int = 0
    completed_examples_count: int = 0
    last_optimization_score: float | None = None
    last_prompt_update: str | None = None

    class Config:
        from_attributes = True


@router.post("", response_model=TaskResponse)
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    """Create a new task with a Pydantic schema."""
    # Validate field names are unique in input schema
    if task.input_schema:
        validate_schema_field_names(task.input_schema, "Input schema")

    # Filter out empty field names from schema
    if task.pydantic_schema and "properties" in task.pydantic_schema:
        task.pydantic_schema["properties"] = {
            k: v for k, v in task.pydantic_schema["properties"].items() if k and k.strip()
        }
        # Update required list to only include existing properties
        if "required" in task.pydantic_schema:
            task.pydantic_schema["required"] = [
                r for r in task.pydantic_schema["required"]
                if r in task.pydantic_schema["properties"]
            ]

    # Validate field names are unique in output schema
    validate_schema_field_names(task.pydantic_schema, "Output schema")

    # Validate schema can be parsed
    try:
        schema_to_json_schema(task.pydantic_schema)
    except (ValueError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema: {str(e)}")

    # Validate system prompt
    if task.system_prompt:
        is_valid, error_msg = validate_system_prompt(task.system_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    # Validate instruction prompt template (must be valid Jinja2)
    if task.instruction_prompt_template:
        is_valid, error_msg = validate_jinja2_template(task.instruction_prompt_template)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    db_task = Task(
        name=task.name,
        description=task.description,
        input_schema=task.input_schema,
        pydantic_schema=task.pydantic_schema,
        system_prompt=task.system_prompt,
        instruction_prompt_template=task.instruction_prompt_template,
        default_model=task.default_model
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # Create first prompt version if system/instruction prompts provided
    if task.system_prompt or task.instruction_prompt_template:
        prompt_version = PromptVersion(
            task_id=db_task.id,
            version_number=1,
            system_prompt=task.system_prompt,
            instruction_prompt=task.instruction_prompt_template,
            is_active=True,
            created_by="system"
        )
        db.add(prompt_version)
        db.commit()

    return _build_task_response(db_task, db)


def _build_task_response(task: Task, db: Session) -> TaskResponse:
    """Build TaskResponse with all calculated fields."""
    example_count = db.query(LabeledExample).filter(LabeledExample.task_id == task.id).count()

    # Count completed examples (approved or reviewed)
    completed_examples_count = db.query(LabeledExample).filter(
        LabeledExample.task_id == task.id,
        LabeledExample.status.in_(["approved", "reviewed"])
    ).count()

    # Get last optimization score from most recent completed run
    last_run = db.query(OptimizationRun).filter(
        OptimizationRun.task_id == task.id,
        OptimizationRun.status == "completed"
    ).order_by(desc(OptimizationRun.completed_at)).first()

    last_optimization_score = None
    if last_run and last_run.metrics:
        # Try to extract score from metrics (could be 'score', 'accuracy', 'f1', etc.)
        score_keys = ['score', 'accuracy', 'f1', 'f1_score', 'metric']
        for key in score_keys:
            if key in last_run.metrics:
                value = last_run.metrics[key]
                if isinstance(value, (int, float)):
                    last_optimization_score = float(value)
                    break

    # Get last prompt update
    last_prompt = db.query(PromptVersion).filter(
        PromptVersion.task_id == task.id
    ).order_by(desc(PromptVersion.created_at)).first()

    last_prompt_update = None
    if last_prompt and last_prompt.created_at:
        last_prompt_update = last_prompt.created_at.isoformat()

    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        input_schema=task.input_schema,
        pydantic_schema=task.pydantic_schema,
        system_prompt=task.system_prompt,
        instruction_prompt_template=task.instruction_prompt_template,
        default_model=task.default_model,
        created_at=task.created_at.isoformat() if task.created_at else "",
        example_count=example_count,
        completed_examples_count=completed_examples_count,
        last_optimization_score=last_optimization_score,
        last_prompt_update=last_prompt_update
    )


@router.get("", response_model=list[TaskResponse])
def list_tasks(db: Session = Depends(get_db)):
    """List all tasks."""
    tasks = db.query(Task).all()
    return [_build_task_response(task, db) for task in tasks]


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: int, db: Session = Depends(get_db)):
    """Get a specific task by ID."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return _build_task_response(task, db)


@router.put("/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db)):
    """Update a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_update.name is not None:
        task.name = task_update.name
    if task_update.description is not None:
        task.description = task_update.description
    if task_update.input_schema is not None:
        # Validate field names are unique in input schema
        validate_schema_field_names(task_update.input_schema, "Input schema")
        task.input_schema = task_update.input_schema
    if task_update.pydantic_schema is not None:
        # Validate field names are unique in output schema
        validate_schema_field_names(task_update.pydantic_schema, "Output schema")
        # Validate schema can be parsed
        try:
            schema_to_json_schema(task_update.pydantic_schema)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid schema: {str(e)}")
        task.pydantic_schema = task_update.pydantic_schema
    if task_update.system_prompt is not None:
        is_valid, error_msg = validate_system_prompt(task_update.system_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        task.system_prompt = task_update.system_prompt
    if task_update.instruction_prompt_template is not None:
        is_valid, error_msg = validate_jinja2_template(task_update.instruction_prompt_template)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        task.instruction_prompt_template = task_update.instruction_prompt_template
    if task_update.default_model is not None:
        task.default_model = task_update.default_model

    db.commit()
    db.refresh(task)

    return _build_task_response(task, db)


@router.delete("/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a task and all its examples."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return {"message": "Task deleted successfully"}


@router.post("/{task_id}/validate")
def validate_data(task_id: int, data: dict, db: Session = Depends(get_db)):
    """Validate data against a task's Pydantic schema."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    is_valid, error_message, validated_data = validate_data_against_schema(data, task.pydantic_schema)

    return {
        "is_valid": is_valid,
        "error_message": error_message,
        "validated_data": validated_data
    }


@router.get("/{task_id}/schema-validation")
def validate_examples_against_schema(task_id: int, db: Session = Depends(get_db)):
    """Validate all examples against current schema and return missing fields."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    examples = db.query(LabeledExample).filter(LabeledExample.task_id == task_id).all()

    validation_results = []
    required_fields = set(task.pydantic_schema.get("required", []))
    all_fields = set(task.pydantic_schema.get("properties", {}).keys())

    for example in examples:
        example_fields = set(example.output_data.keys())
        missing_required = required_fields - example_fields
        missing_optional = all_fields - example_fields - required_fields

        validation_results.append({
            "example_id": example.id,
            "is_valid": len(missing_required) == 0,
            "missing_required_fields": list(missing_required),
            "missing_optional_fields": list(missing_optional),
        })

    return {
        "task_id": task_id,
        "total_examples": len(examples),
        "valid_examples": sum(1 for r in validation_results if r["is_valid"]),
        "invalid_examples": sum(1 for r in validation_results if not r["is_valid"]),
        "validation_results": validation_results
    }


@router.post("/{task_id}/duplicate", response_model=TaskResponse)
def duplicate_task(task_id: int, request: TaskDuplicateRequest, db: Session = Depends(get_db)):
    """Duplicate a task with optional examples."""
    original_task = db.query(Task).filter(Task.id == task_id).first()
    if not original_task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if new name already exists
    existing_task = db.query(Task).filter(Task.name == request.new_name).first()
    if existing_task:
        raise HTTPException(status_code=400, detail=f"Task with name '{request.new_name}' already exists")

    # Create new task
    new_task = Task(
        name=request.new_name,
        description=original_task.description,
        input_schema=original_task.input_schema.copy() if original_task.input_schema else None,
        pydantic_schema=original_task.pydantic_schema.copy() if original_task.pydantic_schema else {},
        system_prompt=original_task.system_prompt,
        instruction_prompt_template=original_task.instruction_prompt_template,
        default_model=original_task.default_model
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    # Copy examples if requested
    if request.copy_examples:
        original_examples = db.query(LabeledExample).filter(LabeledExample.task_id == task_id).all()
        for example in original_examples:
            new_example = LabeledExample(
                task_id=new_task.id,
                input_data=example.input_data.copy() if example.input_data else {},
                output_data=example.output_data.copy() if example.output_data else {},
                status=example.status
            )
            db.add(new_example)
        db.commit()

    # Create first prompt version if system/instruction prompts exist
    if new_task.system_prompt or new_task.instruction_prompt_template:
        prompt_version = PromptVersion(
            task_id=new_task.id,
            version_number=1,
            system_prompt=new_task.system_prompt,
            instruction_prompt=new_task.instruction_prompt_template,
            is_active=True,
            created_by="system"
        )
        db.add(prompt_version)
        db.commit()

    return _build_task_response(new_task, db)
