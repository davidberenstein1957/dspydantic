from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import LabeledExample, OptimizationRun, PromptVersion, Task
from app.services.dspy_service import run_optimization

router = APIRouter(prefix="/api", tags=["optimization"])


class OptimizationConfig(BaseModel):
    metric: str = "exact"  # "exact", "levenshtein"
    optimizer: str | None = None  # None for auto, or "miprov2", "gepa", "copro", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", etc.
    train_split: float = 80.0  # Percentage (0-100) for train/test split
    model_id: str | None = None  # Override task default_model
    api_key: str | None = None  # Override default API key
    max_examples: int = 50  # Maximum number of approved examples to use for optimization
    prompt_version_id: int | None = None  # Use specific prompt version, or None for task defaults


class OptimizationRunResponse(BaseModel):
    id: int
    task_id: int
    prompt_version_number: int | None
    status: str
    config: dict
    metrics: dict | None
    error_message: str | None
    started_at: str
    completed_at: str | None
    progress: float

    class Config:
        from_attributes = True


def run_optimization_background(
    run_id: int,
    task_id: int,
    config: dict[str, Any]
):
    """Background task to run optimization."""
    db = SessionLocal()
    try:
        # Update status to running
        run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
        if run:
            run.status = "running"
            run.progress = 0.1
            db.commit()

        # Get task and examples
        task = db.query(Task).filter(Task.id == task_id).first()
        # Only use approved examples for optimization
        max_examples = config.get("max_examples", 50)
        examples = db.query(LabeledExample).filter(
            LabeledExample.task_id == task_id,
            LabeledExample.status == "approved"
        ).limit(max_examples).all()

        if not examples:
            raise ValueError("No approved examples found for task. Please approve some examples before optimizing.")

        # Get prompt version if specified, otherwise use task defaults
        prompt_version_id = config.get("prompt_version_id")
        prompt_version = None
        if prompt_version_id:
            prompt_version = db.query(PromptVersion).filter(
                PromptVersion.id == prompt_version_id,
                PromptVersion.task_id == task_id
            ).first()
            if not prompt_version:
                raise ValueError(f"Prompt version {prompt_version_id} not found")

        # Run optimization
        run.progress = 0.5
        db.commit()

        results = run_optimization(task, examples, config, prompt_version)

        # Update run with results
        run.status = "completed"
        run.metrics = results.get("metrics", {})
        run.completed_at = datetime.utcnow()
        run.progress = 1.0
        db.commit()

        # Create prompt version
        version_number = db.query(PromptVersion).filter(
            PromptVersion.task_id == task_id
        ).count() + 1

        prompt_version = PromptVersion(
            task_id=task_id,
            optimization_run_id=run_id,
            version_number=version_number,
            prompt_content=results.get("optimized_prompt", ""),  # Legacy field
            system_prompt=results.get("optimized_system_prompt"),
            instruction_prompt=results.get("optimized_instruction_prompt"),
            output_schema_descriptions=results.get("optimized_descriptions", {}),
            metrics=results.get("metrics", {})
        )
        db.add(prompt_version)
        db.commit()

    except Exception as e:
        run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/tasks/{task_id}/optimize", response_model=OptimizationRunResponse)
def trigger_optimization(
    task_id: int,
    config: OptimizationConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger an optimization run for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check for approved examples
    approved_examples = db.query(LabeledExample).filter(
        LabeledExample.task_id == task_id,
        LabeledExample.status == "approved"
    ).count()

    if approved_examples == 0:
        raise HTTPException(
            status_code=400,
            detail="No approved examples found. Please approve some examples before optimizing."
        )

    # Validate prompt version if specified
    if config.prompt_version_id:
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.id == config.prompt_version_id,
            PromptVersion.task_id == task_id
        ).first()
        if not prompt_version:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt version {config.prompt_version_id} not found"
            )

    # Create optimization run
    config_dict = config.model_dump()
    optimization_run = OptimizationRun(
        task_id=task_id,
        status="pending",
        config=config_dict
    )
    db.add(optimization_run)
    db.commit()
    db.refresh(optimization_run)

    # Start background task
    background_tasks.add_task(
        run_optimization_background,
        optimization_run.id,
        task_id,
        config_dict
    )

    # Get prompt version number if prompt_version_id exists in config
    prompt_version_number = None
    prompt_version_id = config_dict.get("prompt_version_id")
    if prompt_version_id:
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.id == prompt_version_id
        ).first()
        if prompt_version:
            prompt_version_number = prompt_version.version_number

    return OptimizationRunResponse(
        id=optimization_run.id,
        task_id=optimization_run.task_id,
        prompt_version_number=prompt_version_number,
        status=optimization_run.status,
        config=optimization_run.config,
        metrics=optimization_run.metrics,
        error_message=optimization_run.error_message,
        started_at=optimization_run.started_at.isoformat() if optimization_run.started_at else "",
        completed_at=optimization_run.completed_at.isoformat() if optimization_run.completed_at else None,
        progress=optimization_run.progress
    )


@router.get("/optimization-runs/{run_id}", response_model=OptimizationRunResponse)
def get_optimization_run(run_id: int, db: Session = Depends(get_db)):
    """Get optimization run status and results."""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    # Get prompt version number if prompt_version_id exists in config
    prompt_version_number = None
    prompt_version_id = run.config.get("prompt_version_id") if run.config else None
    if prompt_version_id:
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.id == prompt_version_id
        ).first()
        if prompt_version:
            prompt_version_number = prompt_version.version_number

    return OptimizationRunResponse(
        id=run.id,
        task_id=run.task_id,
        prompt_version_number=prompt_version_number,
        status=run.status,
        config=run.config,
        metrics=run.metrics,
        error_message=run.error_message,
        started_at=run.started_at.isoformat() if run.started_at else "",
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        progress=run.progress
    )


@router.get("/tasks/{task_id}/optimization-runs", response_model=list[OptimizationRunResponse])
def list_optimization_runs(task_id: int, db: Session = Depends(get_db)):
    """List all optimization runs for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    runs = db.query(OptimizationRun).filter(
        OptimizationRun.task_id == task_id
    ).order_by(OptimizationRun.started_at.desc()).all()

    # Get prompt version numbers for all runs
    prompt_version_ids = {run.config.get("prompt_version_id") for run in runs if run.config and run.config.get("prompt_version_id")}
    prompt_versions = {}
    if prompt_version_ids:
        versions = db.query(PromptVersion).filter(
            PromptVersion.id.in_(prompt_version_ids)
        ).all()
        prompt_versions = {v.id: v.version_number for v in versions}

    return [
        OptimizationRunResponse(
            id=run.id,
            task_id=run.task_id,
            prompt_version_number=prompt_versions.get(run.config.get("prompt_version_id")) if run.config and run.config.get("prompt_version_id") else None,
            status=run.status,
            config=run.config,
            metrics=run.metrics,
            error_message=run.error_message,
            started_at=run.started_at.isoformat() if run.started_at else "",
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            progress=run.progress
        )
        for run in runs
    ]


@router.get("/optimization-runs/{run_id}/delete-info")
def get_optimization_run_delete_info(run_id: int, db: Session = Depends(get_db)):
    """Get information about what will be deleted when deleting an optimization run."""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    # Count prompt versions created by this optimization run
    prompt_versions_count = db.query(PromptVersion).filter(
        PromptVersion.optimization_run_id == run_id
    ).count()

    # Get prompt version IDs to check for evaluation runs
    prompt_version_ids = [
        v.id for v in db.query(PromptVersion).filter(
            PromptVersion.optimization_run_id == run_id
        ).all()
    ]

    # Count evaluation runs that reference these prompt versions
    evaluation_runs_count = 0
    if prompt_version_ids:
        from app.models import EvaluationRun
        evaluation_runs_count = db.query(EvaluationRun).filter(
            EvaluationRun.prompt_version_id.in_(prompt_version_ids)
        ).count()

    return {
        "prompt_versions_count": prompt_versions_count,
        "evaluation_runs_count": evaluation_runs_count,
    }


@router.delete("/optimization-runs/{run_id}")
def delete_optimization_run(run_id: int, db: Session = Depends(get_db)):
    """Delete an optimization run and all associated prompt versions and evaluation runs."""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    # Get all prompt versions created by this optimization run
    prompt_version_ids = [
        v.id for v in db.query(PromptVersion).filter(
            PromptVersion.optimization_run_id == run_id
        ).all()
    ]

    # Delete all evaluation runs that reference these prompt versions
    if prompt_version_ids:
        from app.models import EvaluationRun
        db.query(EvaluationRun).filter(
            EvaluationRun.prompt_version_id.in_(prompt_version_ids)
        ).delete(synchronize_session=False)

    # Delete all prompt versions created by this optimization run
    db.query(PromptVersion).filter(
        PromptVersion.optimization_run_id == run_id
    ).delete(synchronize_session=False)

    # Delete the optimization run
    db.delete(run)
    db.commit()

    return {"message": "Optimization run deleted", "run_id": run_id}
