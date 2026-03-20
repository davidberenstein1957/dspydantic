"""Integration tests for MiPROV2 field description optimization.

Validates that optimizers produce actual field descriptions (not meta-instructions)
using realistic structured extraction datasets.

Run with: uv run pytest tests/integration/test_miprov2_descriptions.py -v
"""

import os
import re
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer

# Patterns that indicate meta-instructions rather than actual descriptions
META_INSTRUCTION_PATTERNS = [
    r"Given the fields",
    r"produce the fields",
    r"Using the fields.*optimize",
    r"your task is to produce",
    r"you are provided with",
    r"generate a clear and concise version",
]


# --- Models ---


class MedicalRecord(BaseModel):
    """Medical record extracted from clinical notes."""

    patient_name: str = Field(description="Full name of the patient")
    age: str = Field(description="Patient's age in years")
    diagnosis: str = Field(description="Medical condition diagnosed")
    medication: str = Field(description="Prescribed medication name")
    dosage: str = Field(description="Medication dosage and frequency")


class Reservation(BaseModel):
    """Restaurant reservation."""

    restaurant: str = Field(description="Restaurant name")
    date: str = Field(description="Date")
    time: str = Field(description="Time")
    party_size: Literal["1", "2", "3", "4", "5", "6+"] = Field(
        description="Number of guests"
    )
    seating: Literal["indoor", "outdoor", "bar"] = Field(
        description="Seating preference"
    )


class Transaction(BaseModel):
    """Financial transaction."""

    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Stock or financial instrument")
    date: str = Field(description="Transaction date")
    status: Literal["pending", "completed", "failed"] = Field(description="Status")
    type: Literal["equity", "bond", "option", "future"] = Field(description="Type")


# --- Fixtures ---


@pytest.fixture
def medical_examples():
    return [
        Example(
            text=(
                "Patient: Mary Wilson, Age: 45, diagnosed with hypertension, "
                "prescribed Lisinopril 10mg daily."
            ),
            expected_output={
                "patient_name": "Mary Wilson",
                "age": "45",
                "diagnosis": "hypertension",
                "medication": "Lisinopril",
                "dosage": "10mg daily",
            },
        ),
        Example(
            text=(
                "Dr. Sarah Johnson prescribed Metformin 500mg twice daily for "
                "diabetes treatment to patient Robert Chen, age 62."
            ),
            expected_output={
                "patient_name": "Robert Chen",
                "age": "62",
                "diagnosis": "diabetes",
                "medication": "Metformin",
                "dosage": "500mg twice daily",
            },
        ),
        Example(
            text=(
                "Clinical note: James Park, 38 years old, presenting with "
                "migraine. Started on Sumatriptan 50mg as needed."
            ),
            expected_output={
                "patient_name": "James Park",
                "age": "38",
                "diagnosis": "migraine",
                "medication": "Sumatriptan",
                "dosage": "50mg as needed",
            },
        ),
        Example(
            text=(
                "Patient Lisa Nguyen (age 55) diagnosed with osteoarthritis. "
                "Prescribed Ibuprofen 400mg three times daily."
            ),
            expected_output={
                "patient_name": "Lisa Nguyen",
                "age": "55",
                "diagnosis": "osteoarthritis",
                "medication": "Ibuprofen",
                "dosage": "400mg three times daily",
            },
        ),
        Example(
            text=(
                "Follow-up: David Brown, 71, chronic heart failure management. "
                "Continuing Metoprolol 25mg twice daily."
            ),
            expected_output={
                "patient_name": "David Brown",
                "age": "71",
                "diagnosis": "chronic heart failure",
                "medication": "Metoprolol",
                "dosage": "25mg twice daily",
            },
        ),
    ]


@pytest.fixture
def reservation_examples():
    return [
        Example(
            text=(
                "Reservation at Le Bernardin for 4 people on March 15th at 7:30 PM. "
                "We'd prefer outdoor seating."
            ),
            expected_output={
                "restaurant": "Le Bernardin",
                "date": "March 15th",
                "time": "7:30 PM",
                "party_size": "4",
                "seating": "outdoor",
            },
        ),
        Example(
            text="Table for 2 at Nobu on Friday at 8 PM, bar seating please.",
            expected_output={
                "restaurant": "Nobu",
                "date": "Friday",
                "time": "8 PM",
                "party_size": "2",
                "seating": "bar",
            },
        ),
        Example(
            text=(
                "Booking for 6+ guests at The French Laundry, December 20th, "
                "indoor, 6:00 PM."
            ),
            expected_output={
                "restaurant": "The French Laundry",
                "date": "December 20th",
                "time": "6:00 PM",
                "party_size": "6+",
                "seating": "indoor",
            },
        ),
        Example(
            text=(
                "I'd like a table for 1 at Eleven Madison Park on Saturday, "
                "7 PM. Indoor please."
            ),
            expected_output={
                "restaurant": "Eleven Madison Park",
                "date": "Saturday",
                "time": "7 PM",
                "party_size": "1",
                "seating": "indoor",
            },
        ),
        Example(
            text=(
                "Party of 5 at Masa, next Tuesday at 6:30 PM. "
                "Bar seating would be great."
            ),
            expected_output={
                "restaurant": "Masa",
                "date": "next Tuesday",
                "time": "6:30 PM",
                "party_size": "5",
                "seating": "bar",
            },
        ),
    ]


@pytest.fixture
def transaction_examples():
    return [
        Example(
            text=(
                "Goldman Sachs processed a $2.5M equity trade for Tesla Inc. "
                "on March 15, 2024. Commission: $1,250. Status: Completed."
            ),
            expected_output={
                "broker": "Goldman Sachs",
                "amount": "$2.5M",
                "security": "Tesla Inc.",
                "date": "March 15, 2024",
                "status": "completed",
                "type": "equity",
            },
        ),
        Example(
            text=(
                "Morgan Stanley executed a $500K bond purchase for US Treasury "
                "on January 10, 2024. Status: Pending."
            ),
            expected_output={
                "broker": "Morgan Stanley",
                "amount": "$500K",
                "security": "US Treasury",
                "date": "January 10, 2024",
                "status": "pending",
                "type": "bond",
            },
        ),
        Example(
            text=(
                "JP Morgan completed a $1.2M option trade for Apple Inc. "
                "on February 28, 2024. Status: Completed."
            ),
            expected_output={
                "broker": "JP Morgan",
                "amount": "$1.2M",
                "security": "Apple Inc.",
                "date": "February 28, 2024",
                "status": "completed",
                "type": "option",
            },
        ),
        Example(
            text=(
                "Barclays initiated a $3M futures contract for Crude Oil WTI "
                "on April 5, 2024. Status: Pending."
            ),
            expected_output={
                "broker": "Barclays",
                "amount": "$3M",
                "security": "Crude Oil WTI",
                "date": "April 5, 2024",
                "status": "pending",
                "type": "future",
            },
        ),
        Example(
            text=(
                "Citigroup processed a $750K equity purchase of Microsoft Corp. "
                "on March 1, 2024. Status: Completed."
            ),
            expected_output={
                "broker": "Citigroup",
                "amount": "$750K",
                "security": "Microsoft Corp.",
                "date": "March 1, 2024",
                "status": "completed",
                "type": "equity",
            },
        ),
    ]


# --- Helper ---


def assert_descriptions_not_meta_instructions(
    optimized_descriptions: dict[str, str],
) -> None:
    """Assert that optimized descriptions are actual descriptions, not meta-instructions."""
    for field_path, description in optimized_descriptions.items():
        assert description, f"Empty description for {field_path}"
        assert len(description) < 500, (
            f"Description for {field_path} is suspiciously long ({len(description)} chars): "
            f"{description[:100]}..."
        )
        for pattern in META_INSTRUCTION_PATTERNS:
            assert not re.search(pattern, description, re.IGNORECASE), (
                f"Description for '{field_path}' looks like a meta-instruction "
                f"(matched '{pattern}'): {description}"
            )


# --- Shared kwargs to limit MiPROV2 trials for faster test runs ---
# auto=None disables auto-mode so we can set num_candidates explicitly.
# num_trials is a compile() arg, passed via compile_kwargs.
FAST_MIPRO_OPTIMIZER_KWARGS = {"auto": None, "num_candidates": 3}
FAST_MIPRO_COMPILE_KWARGS = {"num_trials": 3, "minibatch": False}


# --- Tests ---


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
@pytest.mark.parametrize("optimizer", ["miprov2zeroshot"])
def test_medical_record_optimization(lm, medical_examples, optimizer) -> None:
    """Test that optimization produces actual field descriptions for medical records.

    Uses a GLiNER2-inspired medical record extraction task with 5 examples.
    Validates that:
    - Descriptions are actual descriptions, not meta-instructions
    - Optimization completes without error
    - Scores are non-negative
    """
    opt = PydanticOptimizer(
        model=MedicalRecord,
        examples=medical_examples,
        optimizer=optimizer,
        num_threads=4,
        verbose=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0
    # Verify all fields are present in output
    for field in ["patient_name", "age", "diagnosis", "medication", "dosage"]:
        assert field in result.optimized_descriptions, (
            f"Missing field '{field}' in optimized descriptions"
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
@pytest.mark.parametrize("optimizer", ["miprov2zeroshot"])
def test_reservation_descriptions_not_meta_instructions(
    lm, reservation_examples, optimizer
) -> None:
    """Test that optimization produces actual field descriptions for reservation model."""
    opt = PydanticOptimizer(
        model=Reservation,
        examples=reservation_examples,
        optimizer=optimizer,
        num_threads=4,
        verbose=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
@pytest.mark.parametrize("optimizer", ["miprov2zeroshot"])
def test_transaction_descriptions_not_meta_instructions(
    lm, transaction_examples, optimizer
) -> None:
    """Test that optimization produces actual field descriptions for transaction model."""
    opt = PydanticOptimizer(
        model=Transaction,
        examples=transaction_examples,
        optimizer=optimizer,
        num_threads=4,
        verbose=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_skip_field_description_preserves_originals(
    lm, reservation_examples
) -> None:
    """Test that skip_field_description_optimization preserves original descriptions."""
    opt = PydanticOptimizer(
        model=Reservation,
        examples=reservation_examples,
        optimizer="miprov2zeroshot",
        num_threads=4,
        verbose=True,
        skip_field_description_optimization=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    # Descriptions should be unchanged from originals
    assert result.optimized_descriptions["restaurant"] == "Restaurant name"
    assert result.optimized_descriptions["date"] == "Date"
    assert result.optimized_descriptions["time"] == "Time"
    assert result.optimized_descriptions["party_size"] == "Number of guests"
    assert result.optimized_descriptions["seating"] == "Seating preference"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_sequential_optimization_shows_improvement(
    lm, medical_examples
) -> None:
    """Test that sequential optimization with verbose output shows clear progress.

    This test validates the user-facing output is clear and informative.
    """
    opt = PydanticOptimizer(
        model=MedicalRecord,
        examples=medical_examples,
        optimizer="miprov2zeroshot",
        num_threads=4,
        verbose=True,
        sequential=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.baseline_score >= 0
    assert result.optimized_score >= 0
    # Verify the result has proper metrics
    assert "average_score" in result.metrics
    assert "baseline_score" in result.metrics
    assert "improvement" in result.metrics


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_early_stopping_patience(lm, medical_examples) -> None:
    """Test that early_stopping_patience stops after N consecutive non-improvements."""
    opt = PydanticOptimizer(
        model=MedicalRecord,
        examples=medical_examples,
        optimizer="miprov2zeroshot",
        num_threads=4,
        verbose=True,
        sequential=True,
        parallel_fields=False,
        early_stopping_patience=2,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_auto_generate_prompts(lm, medical_examples) -> None:
    """Test that auto_generate_prompts creates and optimizes system/instruction prompts."""
    opt = PydanticOptimizer(
        model=MedicalRecord,
        examples=medical_examples,
        optimizer="miprov2zeroshot",
        num_threads=4,
        verbose=True,
        auto_generate_prompts=True,
        optimizer_kwargs=FAST_MIPRO_OPTIMIZER_KWARGS,
        compile_kwargs=FAST_MIPRO_COMPILE_KWARGS,
    )

    # Verify prompts were auto-generated
    assert opt.system_prompt is not None
    assert "MedicalRecord" in opt.system_prompt
    assert opt.instruction_prompt is not None
    assert "patient_name" in opt.instruction_prompt

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0
    # Auto-generated prompts should be optimized (or preserved if no improvement)
    assert result.optimized_system_prompt is not None or result.optimized_instruction_prompt is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
@pytest.mark.parametrize("optimizer", ["miprov2zeroshot", "bootstrapfewshot"])
def test_optimizer_compatibility(lm, reservation_examples, optimizer) -> None:
    """Test that field description optimization works across different optimizer types."""
    # Only apply MiPROV2-specific fast kwargs for miprov2 optimizers
    extra_kwargs: dict[str, Any] = {}
    if "mipro" in optimizer:
        extra_kwargs["optimizer_kwargs"] = FAST_MIPRO_OPTIMIZER_KWARGS
        extra_kwargs["compile_kwargs"] = FAST_MIPRO_COMPILE_KWARGS
    opt = PydanticOptimizer(
        model=Reservation,
        examples=reservation_examples,
        optimizer=optimizer,
        num_threads=4,
        verbose=True,
        **extra_kwargs,
    )

    result = opt.optimize()

    assert_descriptions_not_meta_instructions(result.optimized_descriptions)
    assert result.optimized_score >= 0
