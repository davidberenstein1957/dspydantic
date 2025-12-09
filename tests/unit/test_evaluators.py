"""Tests for evaluators module."""

from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import BaseModel, Field

from dspydantic.evaluators import default_evaluate_fn, default_judge_fn
from dspydantic.types import Example


class Address(BaseModel):
    """Address model for testing."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")


class User(BaseModel):
    """User model for testing."""

    name: str = Field(description="User name")
    age: int = Field(description="User age")
    address: Address = Field(description="User address")


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str = Field(description="User name")
    age: int = Field(description="User age")


@pytest.fixture
def mock_lm() -> dspy.LM:
    """Create a mock LM for testing."""
    lm = MagicMock(spec=dspy.LM)
    return lm


def test_field_by_field_comparison_exact_match(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with exact matching for identical data."""
    example = Example(
        text="John Doe, 30, 123 Main St, NYC, 10001",
        expected_output={
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"},
        },
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=User,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    # Mock the extraction to return the same data
    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "age": 30, "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get perfect score for exact match
    assert score == 1.0


def test_field_by_field_comparison_partial_match(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with partial matches."""
    example = Example(
        text="John Doe, 30, 123 Main St, NYC, 10001",
        expected_output={
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"},
        },
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=User,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    # Mock extraction with one field wrong
    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "age": 30, "address": {"street": "123 Main St", "city": "LA", "zip_code": "10001"}}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get less than perfect score (some fields match, some don't)
    assert 0.0 <= score < 1.0


def test_field_by_field_comparison_simple_model(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with a simple model (no nesting)."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "age": 30}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    assert score == 1.0


def test_field_by_field_comparison_missing_field(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison when a field is missing."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # Missing age field
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get less than perfect score (one field missing)
    assert 0.0 <= score < 1.0


def test_field_by_field_comparison_levenshtein_metric(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with levenshtein metric."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="levenshtein",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # Slightly different name
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Do", "age": 30}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get a score between 0 and 1 (not perfect, but not zero)
    assert 0.0 < score < 1.0


def test_field_by_field_comparison_nested_structure(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with nested structures."""
    example = Example(
        text="John Doe, 30, 123 Main St, NYC, 10001",
        expected_output={
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"},
        },
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=User,
        system_prompt=None,
        instruction_prompt=None,
        metric="levenshtein",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # One nested field different
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "age": 30, "address": {"street": "123 Main St", "city": "LA", "zip_code": "10001"}}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get a score between 0 and 1
    # With 4 leaf fields (name, age, address.street, address.city, address.zip_code)
    # and one different, should be around 0.8
    assert 0.0 < score < 1.0


def test_field_by_field_comparison_list_fields(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with list fields."""

    class UserWithTags(BaseModel):
        """User model with list field."""

        name: str = Field(description="User name")
        tags: list[str] = Field(description="User tags")

    example = Example(
        text="John Doe, tags: python, testing",
        expected_output={"name": "John Doe", "tags": ["python", "testing"]},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=UserWithTags,
        system_prompt=None,
        instruction_prompt=None,
        metric="levenshtein",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "tags": ["python", "testing"]}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get perfect score for exact match
    assert score == 1.0


def test_field_by_field_comparison_list_fields_different(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison with different list values."""

    class UserWithTags(BaseModel):
        """User model with list field."""

        name: str = Field(description="User name")
        tags: list[str] = Field(description="User tags")

    example = Example(
        text="John Doe, tags: python, testing",
        expected_output={"name": "John Doe", "tags": ["python", "testing"]},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=UserWithTags,
        system_prompt=None,
        instruction_prompt=None,
        metric="levenshtein",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # Different tags
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "tags": ["java", "coding"]}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get less than perfect score
    assert 0.0 < score < 1.0


def test_default_judge_fn_signature(mock_lm: dspy.LM) -> None:
    """Test that default_judge_fn has the correct signature."""
    example = Example(
        text="John Doe, 30",
        expected_output=None,  # Judge is used when expected_output is None
    )

    # Mock the judge response
    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.evaluation = '{"score": 0.85, "reasoning": "Good extraction"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = default_judge_fn(
            lm=mock_lm,
            model=SimpleUser,
            example=example,
            extracted_data={"name": "John Doe", "age": 30},
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    assert 0.0 <= score <= 1.0


def test_field_by_field_comparison_all_fields_missing(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison when all fields are missing."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # Empty extraction
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = "{}"
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Should get zero score (all fields missing)
    assert score == 0.0


def test_field_by_field_comparison_extra_fields(mock_lm: dspy.LM) -> None:
    """Test field-by-field comparison when extra fields are present."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        # Extra field present
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Extra fields shouldn't affect the score (we only compare schema fields)
    assert score == 1.0
