from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import EvaluationExampleResult, EvaluationRun, LabeledExample, PromptVersion, Task
from app.services.dspy_service import run_evaluation

router = APIRouter(prefix="/api", tags=["evaluation"])


class EvaluationConfig(BaseModel):
    metric: str = "exact"  # "exact", "levenshtein"
    model_id: str | None = None  # Override task default_model
    api_key: str | None = None  # Override default API key
    prompt_version_id: int | None = None  # Use specific prompt version, or None for latest/current
    max_examples: int | None = None  # Maximum number of approved examples to use (None = all)


class EvaluationRunResponse(BaseModel):
    id: int
    task_id: int
    prompt_version_id: int | None
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


def run_evaluation_background(
    run_id: int,
    task_id: int,
    config: dict[str, Any]
):
    """Background task to run evaluation."""
    db = SessionLocal()
    try:
        # Update status to running
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if run:
            run.status = "running"
            run.progress = 0.1
            db.commit()

        # Get task and examples
        task = db.query(Task).filter(Task.id == task_id).first()
        # Only use approved examples for evaluation
        max_examples = config.get("max_examples")
        query = db.query(LabeledExample).filter(
            LabeledExample.task_id == task_id,
            LabeledExample.status == "approved"
        )
        if max_examples:
            query = query.limit(max_examples)
        examples = query.all()

        if not examples:
            raise ValueError("No approved examples found for task. Please approve some examples before evaluating.")

        # Get prompt version if specified, otherwise use latest active or current task prompts
        prompt_version_id = config.get("prompt_version_id")
        prompt_version = None
        if prompt_version_id:
            prompt_version = db.query(PromptVersion).filter(
                PromptVersion.id == prompt_version_id,
                PromptVersion.task_id == task_id
            ).first()
            if not prompt_version:
                raise ValueError(f"Prompt version {prompt_version_id} not found")

        # Run evaluation
        run.progress = 0.5
        db.commit()

        results = run_evaluation(task, examples, config, prompt_version)

        # Update run with results
        run.status = "completed"
        run.metrics = results.get("metrics", {})
        run.completed_at = datetime.utcnow()
        run.progress = 1.0
        db.commit()

        # Store per-example results
        example_results = results.get("example_results", [])
        for result_data in example_results:
            example_result = EvaluationExampleResult(
                evaluation_run_id=run.id,
                example_id=result_data["example_id"],
                score=result_data["score"],
                extracted_output=result_data.get("extracted_output"),
                expected_output=result_data.get("expected_output"),
                error_message=result_data.get("error_message"),
                differences=result_data.get("differences")
            )
            db.add(example_result)
        db.commit()

    except Exception as e:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/tasks/{task_id}/evaluate", response_model=EvaluationRunResponse)
def trigger_evaluation(
    task_id: int,
    config: EvaluationConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger an evaluation run for a task."""
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
            detail="No approved examples found. Please approve some examples before evaluating."
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

    # Create evaluation run
    config_dict = config.model_dump()
    evaluation_run = EvaluationRun(
        task_id=task_id,
        prompt_version_id=config.prompt_version_id,
        status="pending",
        config=config_dict
    )
    db.add(evaluation_run)
    db.commit()
    db.refresh(evaluation_run)

    # Start background task
    background_tasks.add_task(
        run_evaluation_background,
        evaluation_run.id,
        task_id,
        config_dict
    )

    # Get prompt version number if prompt_version_id exists
    prompt_version_number = None
    if evaluation_run.prompt_version_id:
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.id == evaluation_run.prompt_version_id
        ).first()
        if prompt_version:
            prompt_version_number = prompt_version.version_number

    return EvaluationRunResponse(
        id=evaluation_run.id,
        task_id=evaluation_run.task_id,
        prompt_version_id=evaluation_run.prompt_version_id,
        prompt_version_number=prompt_version_number,
        status=evaluation_run.status,
        config=evaluation_run.config,
        metrics=evaluation_run.metrics,
        error_message=evaluation_run.error_message,
        started_at=evaluation_run.started_at.isoformat() if evaluation_run.started_at else "",
        completed_at=evaluation_run.completed_at.isoformat() if evaluation_run.completed_at else None,
        progress=evaluation_run.progress
    )


@router.get("/evaluation-runs/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run(run_id: int, db: Session = Depends(get_db)):
    """Get evaluation run status and results."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    # Get prompt version number if prompt_version_id exists
    prompt_version_number = None
    if run.prompt_version_id:
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.id == run.prompt_version_id
        ).first()
        if prompt_version:
            prompt_version_number = prompt_version.version_number

    return EvaluationRunResponse(
        id=run.id,
        task_id=run.task_id,
        prompt_version_id=run.prompt_version_id,
        prompt_version_number=prompt_version_number,
        status=run.status,
        config=run.config,
        metrics=run.metrics,
        error_message=run.error_message,
        started_at=run.started_at.isoformat() if run.started_at else "",
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        progress=run.progress
    )


@router.get("/tasks/{task_id}/evaluation-runs", response_model=list[EvaluationRunResponse])
def list_evaluation_runs(task_id: int, db: Session = Depends(get_db)):
    """List all evaluation runs for a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    runs = db.query(EvaluationRun).filter(
        EvaluationRun.task_id == task_id
    ).order_by(EvaluationRun.started_at.desc()).all()

    # Get prompt version numbers for all runs
    prompt_version_ids = {run.prompt_version_id for run in runs if run.prompt_version_id}
    prompt_versions = {}
    if prompt_version_ids:
        versions = db.query(PromptVersion).filter(
            PromptVersion.id.in_(prompt_version_ids)
        ).all()
        prompt_versions = {v.id: v.version_number for v in versions}

    return [
        EvaluationRunResponse(
            id=run.id,
            task_id=run.task_id,
            prompt_version_id=run.prompt_version_id,
            prompt_version_number=prompt_versions.get(run.prompt_version_id) if run.prompt_version_id else None,
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


class EvaluationExampleResultResponse(BaseModel):
    id: int
    evaluation_run_id: int
    example_id: int
    score: float
    extracted_output: dict | None
    expected_output: dict | None
    error_message: str | None
    differences: dict | None

    class Config:
        from_attributes = True


@router.get("/evaluation-runs/{run_id}/results", response_model=list[EvaluationExampleResultResponse])
def get_evaluation_run_results(run_id: int, db: Session = Depends(get_db)):
    """Get detailed per-example results for an evaluation run."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = db.query(EvaluationExampleResult).filter(
        EvaluationExampleResult.evaluation_run_id == run_id
    ).order_by(EvaluationExampleResult.id).all()

    return [
        EvaluationExampleResultResponse(
            id=result.id,
            evaluation_run_id=result.evaluation_run_id,
            example_id=result.example_id,
            score=result.score,
            extracted_output=result.extracted_output,
            expected_output=result.expected_output,
            error_message=result.error_message,
            differences=result.differences
        )
        for result in results
    ]


@router.post("/evaluation-runs/{run_id}/results/{result_id}/retry", response_model=EvaluationExampleResultResponse)
def retry_evaluation_example(
    run_id: int,
    result_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retry evaluation for a single example."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    result = db.query(EvaluationExampleResult).filter(
        EvaluationExampleResult.id == result_id,
        EvaluationExampleResult.evaluation_run_id == run_id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")

    # Start background task to retry
    background_tasks.add_task(
        retry_evaluation_example_background,
        run_id,
        result_id,
        result.example_id
    )

    return EvaluationExampleResultResponse(
        id=result.id,
        evaluation_run_id=result.evaluation_run_id,
        example_id=result.example_id,
        score=result.score,
        extracted_output=result.extracted_output,
        expected_output=result.expected_output,
        error_message=result.error_message,
        differences=result.differences
    )


def retry_evaluation_example_background(
    run_id: int,
    result_id: int,
    example_id: int
):
    """Background task to retry evaluation for a single example."""
    db = SessionLocal()
    try:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            return

        example = db.query(LabeledExample).filter(LabeledExample.id == example_id).first()
        if not example:
            return

        task = db.query(Task).filter(Task.id == run.task_id).first()
        if not task:
            return

        # Get prompt version if specified
        prompt_version = None
        if run.prompt_version_id:
            prompt_version = db.query(PromptVersion).filter(
                PromptVersion.id == run.prompt_version_id,
                PromptVersion.task_id == task.id
            ).first()

        # Re-run evaluation for this single example
        from app.services.dspy_service import run_evaluation
        results = run_evaluation(task, [example], run.config, prompt_version)

        # Update the result
        if results.get("example_results"):
            result_data = results["example_results"][0]
            result = db.query(EvaluationExampleResult).filter(
                EvaluationExampleResult.id == result_id
            ).first()
            if result:
                result.score = result_data["score"]
                result.extracted_output = result_data.get("extracted_output")
                result.expected_output = result_data.get("expected_output")
                result.error_message = result_data.get("error_message")
                result.differences = result_data.get("differences")
                db.commit()

                # Recalculate run metrics
                all_results = db.query(EvaluationExampleResult).filter(
                    EvaluationExampleResult.evaluation_run_id == run_id
                ).all()
                if all_results:
                    scores = [r.score for r in all_results]
                    run.metrics = {
                        "average_score": sum(scores) / len(scores) if scores else 0.0,
                        "exact_match_rate": sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0.0,
                        "exact_matches": sum(1 for s in scores if s == 1.0),
                        "total_examples": len(scores),
                        "metric": run.config.get("metric", "exact")
                    }
                    db.commit()
    except Exception as e:
        # Update result with error
        result = db.query(EvaluationExampleResult).filter(
            EvaluationExampleResult.id == result_id
        ).first()
        if result:
            result.error_message = str(e)
            result.score = 0.0
            db.commit()
    finally:
        db.close()


@router.delete("/evaluation-runs/{run_id}")
def delete_evaluation_run(run_id: int, db: Session = Depends(get_db)):
    """Delete an evaluation run and all associated example results."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    # Delete all example results (cascade should handle this, but being explicit)
    db.query(EvaluationExampleResult).filter(
        EvaluationExampleResult.evaluation_run_id == run_id
    ).delete(synchronize_session=False)

    # Delete the evaluation run
    db.delete(run)
    db.commit()

    return {"message": "Evaluation run deleted", "run_id": run_id}
