
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import EvaluationRun, OptimizationRun, PromptVersion, Task
from app.services.prompt_validation import validate_jinja2_template, validate_system_prompt

router = APIRouter(prefix="/api", tags=["prompts"])


class PromptVersionResponse(BaseModel):
    id: int
    task_id: int
    optimization_run_id: int | None
    parent_version_id: int | None
    version_number: int
    system_prompt: str | None
    instruction_prompt: str | None
    prompt_content: str | None  # Legacy field
    output_schema_descriptions: dict | None  # Optimized field descriptions
    metrics: dict | None
    is_active: bool
    created_at: str
    created_by: str | None

    class Config:
        from_attributes = True


class PromptVersionCreate(BaseModel):
    system_prompt: str | None = None
    instruction_prompt: str | None = None
    output_schema_descriptions: dict | None = None
    created_by: str | None = None


class PromptVersionUpdate(BaseModel):
    system_prompt: str | None = None
    instruction_prompt: str | None = None
    output_schema_descriptions: dict | None = None


class PromptComparisonResponse(BaseModel):
    version1: PromptVersionResponse
    version2: PromptVersionResponse
    metrics_comparison: dict


@router.get("/tasks/{task_id}/prompts", response_model=list[PromptVersionResponse])
def list_prompt_versions(task_id: int, db: Session = Depends(get_db)):
    """List all prompt versions for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    versions = db.query(PromptVersion).filter(
        PromptVersion.task_id == task_id
    ).order_by(PromptVersion.created_at.desc()).all()

    return [
        PromptVersionResponse(
            id=v.id,
            task_id=v.task_id,
            optimization_run_id=v.optimization_run_id,
            parent_version_id=v.parent_version_id,
            version_number=v.version_number,
            system_prompt=v.system_prompt,
            instruction_prompt=v.instruction_prompt,
            prompt_content=v.prompt_content,
            output_schema_descriptions=v.output_schema_descriptions,
            metrics=v.metrics,
            is_active=v.is_active,
            created_at=v.created_at.isoformat() if v.created_at else "",
            created_by=v.created_by
        )
        for v in versions
    ]


@router.get("/prompts/compare", response_model=PromptComparisonResponse)
def compare_prompts(
    version_id_1: int = Query(..., alias="version_id_1"),
    version_id_2: int = Query(..., alias="version_id_2"),
    db: Session = Depends(get_db)
):
    """Compare two prompt versions side-by-side."""
    v1 = db.query(PromptVersion).filter(PromptVersion.id == version_id_1).first()
    v2 = db.query(PromptVersion).filter(PromptVersion.id == version_id_2).first()

    if not v1 or not v2:
        raise HTTPException(status_code=404, detail="One or both prompt versions not found")

    # Compare metrics
    metrics_comparison = {}
    if v1.metrics and v2.metrics:
        all_keys = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for key in all_keys:
            val1 = v1.metrics.get(key)
            val2 = v2.metrics.get(key)
            metrics_comparison[key] = {
                "version1": val1,
                "version2": val2,
                "difference": (val2 - val1) if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
            }

    return PromptComparisonResponse(
        version1=PromptVersionResponse(
            id=v1.id,
            task_id=v1.task_id,
            optimization_run_id=v1.optimization_run_id,
            parent_version_id=v1.parent_version_id,
            version_number=v1.version_number,
            system_prompt=v1.system_prompt,
            instruction_prompt=v1.instruction_prompt,
            prompt_content=v1.prompt_content,
            output_schema_descriptions=v1.output_schema_descriptions,
            metrics=v1.metrics,
            is_active=v1.is_active,
            created_at=v1.created_at.isoformat() if v1.created_at else "",
            created_by=v1.created_by
        ),
        version2=PromptVersionResponse(
            id=v2.id,
            task_id=v2.task_id,
            optimization_run_id=v2.optimization_run_id,
            parent_version_id=v2.parent_version_id,
            version_number=v2.version_number,
            system_prompt=v2.system_prompt,
            instruction_prompt=v2.instruction_prompt,
            prompt_content=v2.prompt_content,
            output_schema_descriptions=v2.output_schema_descriptions,
            metrics=v2.metrics,
            is_active=v2.is_active,
            created_at=v2.created_at.isoformat() if v2.created_at else "",
            created_by=v2.created_by
        ),
        metrics_comparison=metrics_comparison
    )


@router.get("/prompts/{version_id}", response_model=PromptVersionResponse)
def get_prompt_version(version_id: int, db: Session = Depends(get_db)):
    """Get a specific prompt version."""
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    return PromptVersionResponse(
        id=version.id,
        task_id=version.task_id,
        optimization_run_id=version.optimization_run_id,
        parent_version_id=version.parent_version_id,
        version_number=version.version_number,
        system_prompt=version.system_prompt,
        instruction_prompt=version.instruction_prompt,
        prompt_content=version.prompt_content,
        output_schema_descriptions=version.output_schema_descriptions,
        metrics=version.metrics,
        is_active=version.is_active,
        created_at=version.created_at.isoformat() if version.created_at else "",
        created_by=version.created_by
    )


@router.post("/tasks/{task_id}/prompts", response_model=PromptVersionResponse)
def create_prompt_version(task_id: int, prompt_data: PromptVersionCreate, db: Session = Depends(get_db)):
    """Create a new prompt version for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Validate system prompt
    if prompt_data.system_prompt:
        is_valid, error_msg = validate_system_prompt(prompt_data.system_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    # Validate instruction prompt template (must be valid Jinja2)
    if prompt_data.instruction_prompt:
        is_valid, error_msg = validate_jinja2_template(prompt_data.instruction_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    # Get the next version number
    last_version = db.query(PromptVersion).filter(
        PromptVersion.task_id == task_id
    ).order_by(PromptVersion.version_number.desc()).first()

    next_version_number = (last_version.version_number + 1) if last_version else 1

    # Get the active version as parent if exists
    active_version = db.query(PromptVersion).filter(
        PromptVersion.task_id == task_id,
        PromptVersion.is_active
    ).first()

    new_version = PromptVersion(
        task_id=task_id,
        version_number=next_version_number,
        system_prompt=prompt_data.system_prompt,
        instruction_prompt=prompt_data.instruction_prompt,
        prompt_content="",  # Legacy field, set to empty string
        output_schema_descriptions=prompt_data.output_schema_descriptions,
        parent_version_id=active_version.id if active_version else None,
        created_by=prompt_data.created_by or "user"
    )

    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    return PromptVersionResponse(
        id=new_version.id,
        task_id=new_version.task_id,
        optimization_run_id=new_version.optimization_run_id,
        parent_version_id=new_version.parent_version_id,
        version_number=new_version.version_number,
        system_prompt=new_version.system_prompt,
        instruction_prompt=new_version.instruction_prompt,
        prompt_content=new_version.prompt_content,
        output_schema_descriptions=new_version.output_schema_descriptions,
        metrics=new_version.metrics,
        is_active=new_version.is_active,
        created_at=new_version.created_at.isoformat() if new_version.created_at else "",
        created_by=new_version.created_by
    )


@router.put("/prompts/{version_id}", response_model=PromptVersionResponse)
def update_prompt_version(version_id: int, prompt_data: PromptVersionUpdate, db: Session = Depends(get_db)):
    """Update a prompt version. Creates a new version if the version is active."""
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # Validate system prompt if provided
    if prompt_data.system_prompt is not None:
        is_valid, error_msg = validate_system_prompt(prompt_data.system_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    # Validate instruction prompt template if provided (must be valid Jinja2)
    if prompt_data.instruction_prompt is not None:
        is_valid, error_msg = validate_jinja2_template(prompt_data.instruction_prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

    # If version is active, create a new version instead of updating
    if version.is_active:
        last_version = db.query(PromptVersion).filter(
            PromptVersion.task_id == version.task_id
        ).order_by(PromptVersion.version_number.desc()).first()

        next_version_number = (last_version.version_number + 1) if last_version else 1

        new_version = PromptVersion(
            task_id=version.task_id,
            version_number=next_version_number,
            system_prompt=prompt_data.system_prompt or version.system_prompt,
            instruction_prompt=prompt_data.instruction_prompt or version.instruction_prompt,
            prompt_content="",  # Legacy field, set to empty string
            output_schema_descriptions=prompt_data.output_schema_descriptions if prompt_data.output_schema_descriptions is not None else version.output_schema_descriptions,
            parent_version_id=version.id,
            created_by="user"
        )

        db.add(new_version)
        db.commit()
        db.refresh(new_version)

        return PromptVersionResponse(
            id=new_version.id,
            task_id=new_version.task_id,
            optimization_run_id=new_version.optimization_run_id,
            parent_version_id=new_version.parent_version_id,
            version_number=new_version.version_number,
            system_prompt=new_version.system_prompt,
            instruction_prompt=new_version.instruction_prompt,
            prompt_content=new_version.prompt_content,
            output_schema_descriptions=new_version.output_schema_descriptions,
            metrics=new_version.metrics,
            is_active=new_version.is_active,
            created_at=new_version.created_at.isoformat() if new_version.created_at else "",
            created_by=new_version.created_by
        )
    else:
        # Update existing inactive version
        if prompt_data.system_prompt is not None:
            version.system_prompt = prompt_data.system_prompt
        if prompt_data.instruction_prompt is not None:
            version.instruction_prompt = prompt_data.instruction_prompt
        if prompt_data.output_schema_descriptions is not None:
            version.output_schema_descriptions = prompt_data.output_schema_descriptions

        db.commit()
        db.refresh(version)

        return PromptVersionResponse(
            id=version.id,
            task_id=version.task_id,
            optimization_run_id=version.optimization_run_id,
            parent_version_id=version.parent_version_id,
            version_number=version.version_number,
            system_prompt=version.system_prompt,
            instruction_prompt=version.instruction_prompt,
            prompt_content=version.prompt_content,
            output_schema_descriptions=version.output_schema_descriptions,
            metrics=version.metrics,
            is_active=version.is_active,
            created_at=version.created_at.isoformat() if version.created_at else "",
            created_by=version.created_by
        )


@router.post("/prompts/{version_id}/activate")
def activate_prompt_version(version_id: int, db: Session = Depends(get_db)):
    """Set a prompt version as active (deactivates others for the task)."""
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # Deactivate all other versions for this task
    db.query(PromptVersion).filter(
        PromptVersion.task_id == version.task_id,
        PromptVersion.id != version_id
    ).update({"is_active": False})

    # Activate this version
    version.is_active = True
    db.commit()

    return {"message": "Prompt version activated", "version_id": version_id}


@router.get("/prompts/{version_id}/delete-info")
def get_prompt_version_delete_info(version_id: int, db: Session = Depends(get_db)):
    """Get information about what will be deleted when deleting a prompt version."""
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # Count evaluation runs that reference this prompt version
    evaluation_runs_count = db.query(EvaluationRun).filter(
        EvaluationRun.prompt_version_id == version_id
    ).count()

    # Check if this prompt version was created by an optimization run
    optimization_run_id = version.optimization_run_id
    optimization_run = None
    will_delete_optimization_run = False
    if optimization_run_id:
        optimization_run = db.query(OptimizationRun).filter(
            OptimizationRun.id == optimization_run_id
        ).first()
        # Check if this is the only prompt version from this optimization run
        other_versions = db.query(PromptVersion).filter(
            PromptVersion.optimization_run_id == optimization_run_id,
            PromptVersion.id != version_id
        ).count()
        will_delete_optimization_run = (other_versions == 0)

    return {
        "evaluation_runs_count": evaluation_runs_count,
        "optimization_run_id": optimization_run_id if will_delete_optimization_run else None,
        "optimization_run_status": optimization_run.status if optimization_run and will_delete_optimization_run else None,
    }


@router.delete("/prompts/{version_id}")
def delete_prompt_version(version_id: int, db: Session = Depends(get_db)):
    """Delete a prompt version and all associated evaluation runs and optimization run."""
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # Prevent deletion of active version
    if version.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active prompt version. Please activate another version first."
        )

    # Delete all evaluation runs that reference this prompt version
    db.query(EvaluationRun).filter(
        EvaluationRun.prompt_version_id == version_id
    ).delete()

    # Delete the optimization run that created this prompt version (if any)
    optimization_run_id = version.optimization_run_id
    if optimization_run_id:
        # Check if this optimization run has other prompt versions
        other_versions = db.query(PromptVersion).filter(
            PromptVersion.optimization_run_id == optimization_run_id,
            PromptVersion.id != version_id
        ).count()

        # Only delete the optimization run if this is the only prompt version it created
        if other_versions == 0:
            db.query(OptimizationRun).filter(
                OptimizationRun.id == optimization_run_id
            ).delete()

    # Delete the prompt version
    db.delete(version)
    db.commit()

    return {"message": "Prompt version deleted", "version_id": version_id}
