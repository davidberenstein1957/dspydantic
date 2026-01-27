import base64
import io
import json
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from app.database import get_db
from app.models import LabeledExample, PromptVersion, Task
from app.services.prompt_validation import render_jinja2_template, validate_jinja2_template
from app.services.schema_service import (
    schema_to_json_schema,
    validate_data_against_schema,
)

router = APIRouter(prefix="/api", tags=["examples"])


class ExampleCreate(BaseModel):
    input_data: dict
    output_data: dict


class ExampleUpdate(BaseModel):
    input_data: dict | None = None
    output_data: dict | None = None
    status: str | None = None


class ExampleResponse(BaseModel):
    id: int
    task_id: int
    input_data: dict
    output_data: dict
    status: str | None = None
    created_at: str
    updated_at: str | None = None
    is_complete: bool = False
    input_complete: bool = False
    output_complete: bool = False

    class Config:
        from_attributes = True


class ExampleStatusUpdate(BaseModel):
    status: str | None = None


class ExampleDuplicateRequest(BaseModel):
    new_task_id: int


class GenerateOutputRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = None
    api_key: str | None = None


class BulkImportRequest(BaseModel):
    examples: list[ExampleCreate]


class BulkStatusUpdateRequest(BaseModel):
    example_ids: list[int]
    status: str | None = None


class BulkGenerateOutputRequest(BaseModel):
    example_ids: list[int]
    model_id: str | None = None
    api_key: str | None = None


def _is_output_complete(example: LabeledExample, task: Task) -> bool:
    """Check if an example has all required output fields."""
    required_fields = set(task.pydantic_schema.get("required", []))
    example_fields = set(example.output_data.keys())
    return len(required_fields - example_fields) == 0


def _is_input_complete(example: LabeledExample, task: Task) -> bool:
    """Check if an example has all required input fields."""
    if not task.input_schema or "required" not in task.input_schema:
        return True  # No input schema requirements means complete
    required_fields = set(task.input_schema.get("required", []))
    example_fields = set(example.input_data.keys() if example.input_data else [])
    return len(required_fields - example_fields) == 0


def _is_example_complete(example: LabeledExample, task: Task) -> bool:
    """Check if an example has all required fields (backward compatibility)."""
    return _is_output_complete(example, task)


def _build_example_response(example: LabeledExample, task: Task) -> ExampleResponse:
    """Build ExampleResponse with completeness checks."""
    input_complete = _is_input_complete(example, task)
    output_complete = _is_output_complete(example, task)
    is_complete = output_complete  # Backward compatibility
    return ExampleResponse(
        id=example.id,
        task_id=example.task_id,
        input_data=example.input_data,
        output_data=example.output_data,
        status=example.status,
        created_at=example.created_at.isoformat() if example.created_at else "",
        updated_at=example.updated_at.isoformat() if example.updated_at else None,
        is_complete=is_complete,
        input_complete=input_complete,
        output_complete=output_complete
    )


@router.post("/tasks/{task_id}/examples", response_model=ExampleResponse)
def create_example(task_id: int, example: ExampleCreate, db: Session = Depends(get_db)):
    """Create a labeled example for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Validate output_data against schema
    is_valid, error_message, _ = validate_data_against_schema(example.output_data, task.pydantic_schema)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Output data validation failed: {error_message}")

    db_example = LabeledExample(
        task_id=task_id,
        input_data=example.input_data,
        output_data=example.output_data
    )
    db.add(db_example)
    db.commit()
    db.refresh(db_example)

    return _build_example_response(db_example, task)


@router.get("/tasks/{task_id}/examples", response_model=list[ExampleResponse])
def list_examples(
    task_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_complete: bool | None = Query(None, description="Filter by output completeness (backward compatibility)"),
    input_complete: bool | None = Query(None, description="Filter by input completeness"),
    output_complete: bool | None = Query(None, description="Filter by output completeness"),
    status: str | None = Query(None, description="Filter by status"),
    created_after: str | None = Query(None, description="Filter by created date (ISO format)"),
    created_before: str | None = Query(None, description="Filter by created date (ISO format)"),
    db: Session = Depends(get_db)
):
    """List examples for a task with pagination and filtering."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    query = db.query(LabeledExample).filter(LabeledExample.task_id == task_id)

    # Apply filters
    if status is not None:
        if status == "":
            query = query.filter(LabeledExample.status.is_(None))
        else:
            query = query.filter(LabeledExample.status == status)

    if created_after:
        try:
            date_after = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
            query = query.filter(LabeledExample.created_at >= date_after)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid created_after date format")

    if created_before:
        try:
            date_before = datetime.fromisoformat(created_before.replace('Z', '+00:00'))
            query = query.filter(LabeledExample.created_at <= date_before)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid created_before date format")

    examples = query.order_by(LabeledExample.created_at.desc()).offset(offset).limit(limit).all()

    # Filter by completeness if requested
    filtered_examples = []
    for ex in examples:
        # Check input completeness
        if input_complete is not None:
            ex_input_complete = _is_input_complete(ex, task)
            if ex_input_complete != input_complete:
                continue

        # Check output completeness
        if output_complete is not None:
            ex_output_complete = _is_output_complete(ex, task)
            if ex_output_complete != output_complete:
                continue

        # Backward compatibility: is_complete filters by output completeness
        if is_complete is not None:
            ex_output_complete = _is_output_complete(ex, task)
            if ex_output_complete != is_complete:
                continue

        filtered_examples.append(ex)

    examples = filtered_examples

    return [_build_example_response(ex, task) for ex in examples]


@router.get("/examples/{example_id}", response_model=ExampleResponse)
def get_example(example_id: int, db: Session = Depends(get_db)):
    """Get a specific example by ID."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    task = db.query(Task).filter(Task.id == example.task_id).first()
    return _build_example_response(example, task)


@router.put("/examples/{example_id}", response_model=ExampleResponse)
def update_example(example_id: int, example_update: ExampleUpdate, db: Session = Depends(get_db)):
    """Update an example."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    task = db.query(Task).filter(Task.id == example.task_id).first()

    if example_update.output_data is not None:
        # Validate new output_data
        is_valid, error_message, _ = validate_data_against_schema(example_update.output_data, task.pydantic_schema)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Output data validation failed: {error_message}")
        example.output_data = example_update.output_data

    if example_update.input_data is not None:
        example.input_data = example_update.input_data

    if example_update.status is not None:
        example.status = example_update.status

    db.commit()
    db.refresh(example)

    return _build_example_response(example, task)


@router.delete("/examples/{example_id}")
def delete_example(example_id: int, db: Session = Depends(get_db)):
    """Delete an example."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    db.delete(example)
    db.commit()
    return {"message": "Example deleted successfully"}


@router.post("/tasks/{task_id}/examples/import")
def bulk_import_examples(task_id: int, request: BulkImportRequest, db: Session = Depends(get_db)):
    """Bulk import examples from JSON."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    created = []
    errors = []

    for idx, example in enumerate(request.examples):
        try:
            # Validate output_data
            is_valid, error_message, _ = validate_data_against_schema(example.output_data, task.pydantic_schema)
            if not is_valid:
                errors.append({"index": idx, "error": error_message})
                continue

            db_example = LabeledExample(
                task_id=task_id,
                input_data=example.input_data,
                output_data=example.output_data
            )
            db.add(db_example)
            created.append(example)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    db.commit()

    return {
        "created": len(created),
        "errors": errors,
        "total": len(request.examples)
    }


def _get_schema_fields(schema: dict) -> set[str]:
    """Extract all field names from a JSON schema."""
    fields = set()
    if "properties" in schema:
        fields.update(schema["properties"].keys())
    return fields


def _parse_file_to_examples(
    file_content: bytes,
    filename: str,
    task: Task,
) -> list[ExampleCreate]:
    """Parse CSV or Excel file and convert to ExampleCreate objects."""
    # Determine file type
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""

    try:
        if file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(BytesIO(file_content))
        elif file_ext == "csv":
            df = pd.read_csv(BytesIO(file_content))
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {file_ext}. "
                    "Supported: csv, xlsx, xls"
                )
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Get schema fields
    input_fields = _get_schema_fields(task.input_schema or {})
    output_fields = _get_schema_fields(task.pydantic_schema)

    # Map columns to input/output fields
    # Columns prefixed with "input_" or matching input schema fields
    # go to input_data. Other columns matching output schema fields
    # go to output_data. Special: "text" column goes to input_data["text"]

    examples = []
    for idx, row in df.iterrows():
        input_data = {}
        output_data = {}

        for col in df.columns:
            value = row[col]

            # Skip NaN values
            if pd.isna(value):
                continue

            # Convert value to appropriate type
            if isinstance(value, (int, float)) and pd.notna(value):
                if isinstance(value, float):
                    processed_value = float(value)
                else:
                    processed_value = int(value)
            elif isinstance(value, str):
                # Try to parse as JSON if it looks like JSON
                if value.strip().startswith(("{", "[")):
                    try:
                        processed_value = json.loads(value)
                    except json.JSONDecodeError:
                        processed_value = value
                else:
                    processed_value = value
            else:
                processed_value = value

            # Determine if column belongs to input or output
            col_lower = col.lower().strip()

            # Check if it's an input field
            if col_lower.startswith("input_") or col in input_fields:
                # Remove "input_" prefix if present
                if col_lower.startswith("input_"):
                    input_key = col[6:]
                else:
                    input_key = col
                input_data[input_key] = processed_value
            # Special case: "text" column goes to input_data["text"]
            elif col_lower == "text" and "text" not in input_data:
                input_data["text"] = processed_value
            # Check if it's an output field
            elif col in output_fields:
                output_data[col] = processed_value
            # Default: if column doesn't match anything, try to infer
            # If it matches a nested field path (e.g., "field.subfield"),
            # handle it
            else:
                # For now, if it doesn't match input schema, assume it's output
                # This allows flexibility for custom columns
                if col not in input_fields:
                    output_data[col] = processed_value

        # Ensure we have at least some data
        if not input_data and not output_data:
            continue

        # If no input_data, create empty dict
        if not input_data:
            input_data = {}

        # If no output_data, create empty dict (will be validated)
        if not output_data:
            output_data = {}

        examples.append(
            ExampleCreate(input_data=input_data, output_data=output_data)
        )

    return examples


@router.post("/tasks/{task_id}/examples/upload")
def upload_examples_file(
    task_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload CSV or Excel file to bulk import examples."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Read file content
    file_content = file.file.read()

    try:
        # Parse file to examples
        examples = _parse_file_to_examples(file_content, file.filename, task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse file: {str(e)}"
        )

    if not examples:
        raise HTTPException(
            status_code=400, detail="No valid examples found in file"
        )

    # Import examples using bulk import logic
    created = []
    errors = []

    for idx, example in enumerate(examples):
        try:
            # Validate output_data
            is_valid, error_message, _ = validate_data_against_schema(
                example.output_data, task.pydantic_schema
            )
            if not is_valid:
                errors.append({"index": idx, "error": error_message})
                continue

            db_example = LabeledExample(
                task_id=task_id,
                input_data=example.input_data,
                output_data=example.output_data
            )
            db.add(db_example)
            created.append(example)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    db.commit()

    return {
        "created": len(created),
        "errors": errors,
        "total": len(examples)
    }


@router.patch("/examples/{example_id}/status", response_model=ExampleResponse)
def update_example_status(example_id: int, status_update: ExampleStatusUpdate, db: Session = Depends(get_db)):
    """Update the status of an example."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    task = db.query(Task).filter(Task.id == example.task_id).first()
    example.status = status_update.status
    db.commit()
    db.refresh(example)

    return _build_example_response(example, task)


@router.post("/examples/{example_id}/duplicate", response_model=ExampleResponse)
def duplicate_example(example_id: int, request: ExampleDuplicateRequest, db: Session = Depends(get_db)):
    """Duplicate an example to a new task."""
    original_example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not original_example:
        raise HTTPException(status_code=404, detail="Example not found")

    # Verify target task exists
    target_task = db.query(Task).filter(Task.id == request.new_task_id).first()
    if not target_task:
        raise HTTPException(status_code=404, detail="Target task not found")

    # Validate output_data against target task schema
    is_valid, error_message, _ = validate_data_against_schema(original_example.output_data, target_task.pydantic_schema)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Output data validation failed for target task: {error_message}")

    # Create new example
    new_example = LabeledExample(
        task_id=request.new_task_id,
        input_data=original_example.input_data.copy() if original_example.input_data else {},
        output_data=original_example.output_data.copy() if original_example.output_data else {},
        status=None  # Reset status for new task
    )
    db.add(new_example)
    db.commit()
    db.refresh(new_example)

    return _build_example_response(new_example, target_task)


def _generate_output_for_example(example_id: int, model_id: str, api_key: str, db: Session):
    """Helper function to generate output for a single example."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        return

    task = db.query(Task).filter(Task.id == example.task_id).first()
    if not task:
        return

    try:
        import litellm

        # Get the most recent prompt version (active or latest)
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.task_id == task.id
        ).order_by(PromptVersion.created_at.desc()).first()

        # Use prompt version if available, otherwise fall back to task prompts
        system_prompt = prompt_version.system_prompt if prompt_version and prompt_version.system_prompt else task.system_prompt
        instruction_prompt_template = prompt_version.instruction_prompt if prompt_version and prompt_version.instruction_prompt else task.instruction_prompt_template

        # Validate instruction prompt template is valid Jinja2
        if instruction_prompt_template:
            is_valid, error_msg = validate_jinja2_template(instruction_prompt_template)
            if not is_valid:
                return

        # Get JSON schema for response_format
        json_schema = schema_to_json_schema(task.pydantic_schema)

        # Add additionalProperties: false to all object types (required by OpenAI)
        def add_additional_properties_false(schema: dict) -> dict:
            """Recursively add additionalProperties: false to all object types."""
            if isinstance(schema, dict):
                schema = schema.copy()
                if schema.get("type") == "object":
                    schema["additionalProperties"] = False
                    if "properties" in schema:
                        for prop_name, prop_schema in schema["properties"].items():
                            schema["properties"][prop_name] = add_additional_properties_false(prop_schema)
                    if "items" in schema:
                        schema["items"] = add_additional_properties_false(schema["items"])
                elif "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        schema["properties"][prop_name] = add_additional_properties_false(prop_schema)
                if "$defs" in schema:
                    for def_name, def_schema in schema["$defs"].items():
                        schema["$defs"][def_name] = add_additional_properties_false(def_schema)
            return schema

        json_schema = add_additional_properties_false(json_schema)

        # Prepare input data
        input_data = example.input_data or {}
        images = input_data.get("images", [])

        # Validate required PDF fields from input schema
        if task.input_schema and "required" in task.input_schema:
            required_fields = task.input_schema.get("required", [])
            input_schema_properties = task.input_schema.get("properties", {})

            for field_name in required_fields:
                field_def = input_schema_properties.get(field_name, {})
                if field_def.get("type") == "pdf":
                    # Check if PDF is provided in input_data
                    pdf_value = input_data.get(field_name) or input_data.get("pdf")
                    if not pdf_value or (isinstance(pdf_value, str) and not pdf_value.strip()):
                        return

        # Convert PDF to images at 150 DPI if present
        # Check both "pdf" field and any PDF fields from schema
        pdf_fields = []
        if "pdf" in input_data and input_data["pdf"]:
            pdf_fields.append(("pdf", input_data["pdf"]))

        # Check for PDF fields in input schema
        if task.input_schema and "properties" in task.input_schema:
            for field_name, field_def in task.input_schema.get("properties", {}).items():
                if field_def.get("type") == "pdf" and field_name in input_data and input_data[field_name]:
                    pdf_fields.append((field_name, input_data[field_name]))

        if pdf_fields:
            if not PDF_AVAILABLE:
                return

            try:
                # Process all PDF fields
                for field_name, pdf_value in pdf_fields:
                    # Handle base64 PDF
                    pdf_base64 = pdf_value
                    # Remove data URL prefix if present
                    if "," in pdf_base64:
                        pdf_base64 = pdf_base64.split(",", 1)[1]

                    # Decode base64 PDF to bytes
                    pdf_bytes = base64.b64decode(pdf_base64)

                    # Convert PDF to images at 150 DPI
                    pdf_images = convert_from_bytes(pdf_bytes, dpi=150)

                    # Convert each page to base64 and add to images list
                    for image in pdf_images:
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        base64_str = base64.b64encode(image_bytes).decode("utf-8")
                        images.append(base64_str)
            except Exception:
                return

        # Render instruction prompt with Jinja2 if template exists
        # The instruction prompt template should include all input data via Jinja2
        if not instruction_prompt_template:
            return

        try:
            user_message = render_jinja2_template(instruction_prompt_template, input_data)
        except Exception:
            return

        # Prepare messages for litellm
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Handle images if present
        if images:
            content = [{"type": "text", "text": user_message}]
            for img_base64 in images:
                # Remove data URL prefix if present
                if "," in img_base64:
                    img_base64 = img_base64.split(",", 1)[1]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})

        # Call litellm with response_format
        response = litellm.completion(
            model=model_id,
            messages=messages,
            api_key=api_key,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extracted_data",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        # Extract the JSON from the response
        output_data = json.loads(response.choices[0].message.content)

        # Validate output data against schema
        is_valid, error_message, validated_data = validate_data_against_schema(output_data, task.pydantic_schema)
        if not is_valid:
            return

        # Update example with generated output
        example.output_data = validated_data
        db.commit()
        db.refresh(example)

    except Exception:
        pass


@router.post("/examples/{example_id}/generate-output", response_model=ExampleResponse)
def generate_output(example_id: int, request: GenerateOutputRequest, db: Session = Depends(get_db)):
    """Generate output for an example using litellm with response_format."""
    example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    task = db.query(Task).filter(Task.id == example.task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Determine model to use
    model_id = request.model_id or task.default_model or os.getenv("DSPY_MODEL_ID", "gpt-4o")
    api_key = request.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required. Provide it in the request or set OPENAI_API_KEY environment variable.")

    try:
        import litellm

        # Get the most recent prompt version (active or latest)
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.task_id == task.id
        ).order_by(PromptVersion.created_at.desc()).first()

        # Use prompt version if available, otherwise fall back to task prompts
        system_prompt = prompt_version.system_prompt if prompt_version and prompt_version.system_prompt else task.system_prompt
        instruction_prompt_template = prompt_version.instruction_prompt if prompt_version and prompt_version.instruction_prompt else task.instruction_prompt_template

        # Validate instruction prompt template is valid Jinja2
        if instruction_prompt_template:
            is_valid, error_msg = validate_jinja2_template(instruction_prompt_template)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid instruction prompt template: {error_msg}")

        # Get JSON schema for response_format
        json_schema = schema_to_json_schema(task.pydantic_schema)

        # Add additionalProperties: false to all object types (required by OpenAI)
        def add_additional_properties_false(schema: dict) -> dict:
            """Recursively add additionalProperties: false to all object types."""
            if isinstance(schema, dict):
                schema = schema.copy()
                if schema.get("type") == "object":
                    schema["additionalProperties"] = False
                    if "properties" in schema:
                        for prop_name, prop_schema in schema["properties"].items():
                            schema["properties"][prop_name] = add_additional_properties_false(prop_schema)
                    if "items" in schema:
                        schema["items"] = add_additional_properties_false(schema["items"])
                elif "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        schema["properties"][prop_name] = add_additional_properties_false(prop_schema)
                if "$defs" in schema:
                    for def_name, def_schema in schema["$defs"].items():
                        schema["$defs"][def_name] = add_additional_properties_false(def_schema)
            return schema

        json_schema = add_additional_properties_false(json_schema)

        # Prepare input data
        input_data = example.input_data or {}
        images = input_data.get("images", [])

        # Validate required PDF fields from input schema
        if task.input_schema and "required" in task.input_schema:
            required_fields = task.input_schema.get("required", [])
            input_schema_properties = task.input_schema.get("properties", {})

            for field_name in required_fields:
                field_def = input_schema_properties.get(field_name, {})
                if field_def.get("type") == "pdf":
                    # Check if PDF is provided in input_data
                    pdf_value = input_data.get(field_name) or input_data.get("pdf")
                    if not pdf_value or (isinstance(pdf_value, str) and not pdf_value.strip()):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "I don't see a PDF attached. Please upload the PDF (or paste its text) "
                                "and tell me what information you want extracted "
                                "(e.g., full text, contact details, tables, dates, images, specific fields)."
                            )
                        )

        # Convert PDF to images at 150 DPI if present
        # Check both "pdf" field and any PDF fields from schema
        pdf_fields = []
        if "pdf" in input_data and input_data["pdf"]:
            pdf_fields.append(("pdf", input_data["pdf"]))

        # Check for PDF fields in input schema
        if task.input_schema and "properties" in task.input_schema:
            for field_name, field_def in task.input_schema.get("properties", {}).items():
                if field_def.get("type") == "pdf" and field_name in input_data and input_data[field_name]:
                    pdf_fields.append((field_name, input_data[field_name]))

        if pdf_fields:
            if not PDF_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="PDF processing requires pdf2image and pillow. Install with: pip install pdf2image pillow"
                )

            try:
                # Process all PDF fields
                for field_name, pdf_value in pdf_fields:
                    # Handle base64 PDF
                    pdf_base64 = pdf_value
                    # Remove data URL prefix if present
                    if "," in pdf_base64:
                        pdf_base64 = pdf_base64.split(",", 1)[1]

                    # Decode base64 PDF to bytes
                    pdf_bytes = base64.b64decode(pdf_base64)

                    # Convert PDF to images at 150 DPI
                    pdf_images = convert_from_bytes(pdf_bytes, dpi=150)

                    # Convert each page to base64 and add to images list
                    for image in pdf_images:
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        base64_str = base64.b64encode(image_bytes).decode("utf-8")
                        images.append(base64_str)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to convert PDF to images: {str(e)}"
                )

        # Render instruction prompt with Jinja2 if template exists
        # The instruction prompt template should include all input data via Jinja2
        if not instruction_prompt_template:
            raise HTTPException(status_code=400, detail="Instruction prompt template is required")

        try:
            user_message = render_jinja2_template(instruction_prompt_template, input_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to render instruction prompt template: {str(e)}")

        # Prepare messages for litellm
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Handle images if present
        if images:
            content = [{"type": "text", "text": user_message}]
            for img_base64 in images:
                # Remove data URL prefix if present
                if "," in img_base64:
                    img_base64 = img_base64.split(",", 1)[1]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})

        # Call litellm with response_format
        response = litellm.completion(
            model=model_id,
            messages=messages,
            api_key=api_key,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extracted_data",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        # Extract the JSON from the response
        output_data = json.loads(response.choices[0].message.content)

        # Validate output data against schema
        is_valid, error_message, validated_data = validate_data_against_schema(output_data, task.pydantic_schema)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Generated output validation failed: {error_message}")

        # Update example with generated output
        example.output_data = validated_data
        db.commit()
        db.refresh(example)

        return _build_example_response(example, task)

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Required dependencies not installed: {str(e)}")
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM output as JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Output generation failed: {str(e)}")


def _generate_outputs_background(example_ids: list[int], model_id: str, api_key: str):
    """Background task to generate outputs for multiple examples."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        for example_id in example_ids:
            _generate_output_for_example(example_id, model_id, api_key, db)
    finally:
        db.close()


@router.post("/examples/bulk/update-status")
def bulk_update_status(request: BulkStatusUpdateRequest, db: Session = Depends(get_db)):
    """Bulk update status for multiple examples."""
    if not request.example_ids:
        raise HTTPException(status_code=400, detail="example_ids cannot be empty")

    examples = db.query(LabeledExample).filter(LabeledExample.id.in_(request.example_ids)).all()
    if len(examples) != len(request.example_ids):
        raise HTTPException(status_code=404, detail="Some examples not found")

    for example in examples:
        example.status = request.status
    db.commit()

    return {"message": f"Updated status for {len(examples)} examples", "count": len(examples)}


@router.post("/examples/bulk/generate-output")
def bulk_generate_output(
    request: BulkGenerateOutputRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk generate outputs for multiple examples asynchronously."""
    if not request.example_ids:
        raise HTTPException(status_code=400, detail="example_ids cannot be empty")

    examples = db.query(LabeledExample).filter(LabeledExample.id.in_(request.example_ids)).all()
    if len(examples) != len(request.example_ids):
        raise HTTPException(status_code=404, detail="Some examples not found")

    # Get task to determine default model
    if examples:
        task = db.query(Task).filter(Task.id == examples[0].task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
    else:
        raise HTTPException(status_code=400, detail="No examples found")

    # Determine model to use
    model_id = request.model_id or task.default_model or os.getenv("DSPY_MODEL_ID", "gpt-4o")
    api_key = request.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required. Provide it in the request or set OPENAI_API_KEY environment variable.")

    # Start background task
    background_tasks.add_task(
        _generate_outputs_background,
        request.example_ids,
        model_id,
        api_key
    )

    return {"message": f"Started generation for {len(examples)} examples", "count": len(examples)}
