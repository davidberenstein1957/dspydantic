"""Tests for evaluators module."""

from typing import Any
from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import BaseModel, Field

from dspydantic.evaluators import (
    LevenshteinEvaluator,
    PythonCodeEvaluator,
    ScoreJudge,
    StringCheckEvaluator,
)
from dspydantic.evaluators.config import EVALUATOR_REGISTRY, EvaluatorFactory, register_evaluator
from dspydantic.evaluators.functions import default_evaluate_fn, default_judge_fn
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


def test_multi_image_signature_single_image(mock_lm: dspy.LM) -> None:
    """Test that single image uses correct signature."""
    from unittest.mock import MagicMock, patch

    example = Example(
        text="Extract info",
        image_base64="base64_image1",
        expected_output={"name": "John Doe"},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with (
        patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class,
        patch("dspydantic.evaluators.convert_images_to_dspy_images") as mock_convert,
    ):
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

        # Verify signature was called with single image signature
        call_args = mock_chain_class.call_args[0][0]
        assert call_args == "prompt, image -> json_output"

        # Verify extractor was called with single image
        extractor_call_kwargs = mock_instance.call_args[1]
        assert "image" in extractor_call_kwargs
        assert extractor_call_kwargs["image"] == mock_image


def test_multi_image_signature_multiple_images(mock_lm: dspy.LM) -> None:
    """Test that multiple images use list[dspy.Image] signature."""
    from unittest.mock import MagicMock, patch

    # Create example with multiple images by manually setting input_data
    # since Example API doesn't directly support multiple images
    example = Example(
        text="Extract info",
        image_base64="base64_img1",  # Will be overridden
        expected_output={"name": "John Doe"},
    )
    # Manually set multiple images in input_data
    example.input_data = {
        "text": "Extract info",
        "images": ["base64_img1", "base64_img2", "base64_img3"],
    }

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with (
        patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class,
        patch("dspydantic.evaluators.convert_images_to_dspy_images") as mock_convert,
    ):
        mock_image1 = MagicMock()
        mock_image2 = MagicMock()
        mock_image3 = MagicMock()
        mock_convert.return_value = [mock_image1, mock_image2, mock_image3]

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "John Doe"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

        # Verify signature was called with list[dspy.Image]
        call_args = mock_chain_class.call_args[0][0]
        assert call_args == "prompt, images: list[dspy.Image] -> json_output"

        # Verify extractor was called with images list
        extractor_call_kwargs = mock_instance.call_args[1]
        assert "images" in extractor_call_kwargs
        assert extractor_call_kwargs["images"] == [mock_image1, mock_image2, mock_image3]


def test_string_output_evaluation(mock_lm: dspy.LM) -> None:
    """Test evaluation with string expected_output."""
    from dspydantic.types import create_output_model

    OutputModel = create_output_model()
    example = Example(
        text="Good response",
        expected_output="excellent",
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=OutputModel,
        system_prompt=None,
        instruction_prompt=None,
        metric="exact",
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"output": "excellent"}'
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    assert score == 1.0


def test_evaluator_config_string_name(mock_lm: dspy.LM) -> None:
    """Test evaluator config with string name."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        evaluator_config={"default": "exact", "field_overrides": {}},
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


def test_evaluator_config_with_field_overrides(mock_lm: dspy.LM) -> None:
    """Test evaluator config with field-specific overrides."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )

    evaluate = default_evaluate_fn(
        lm=mock_lm,
        model=SimpleUser,
        system_prompt=None,
        instruction_prompt=None,
        evaluator_config={
            "default": "exact",
            "field_overrides": {
                "name": {"type": "exact", "config": {"case_sensitive": False}},
            },
        },
    )

    with patch("dspydantic.evaluators.dspy.ChainOfThought") as mock_chain_class:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.json_output = '{"name": "john doe", "age": 30}'  # lowercase name
        mock_instance.return_value = mock_result
        mock_chain_class.return_value = mock_instance

        score = evaluate(
            example=example,
            optimized_descriptions={},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
        )

    # Name should match (case insensitive), age should match exactly
    assert score == 1.0


def test_string_check_evaluator() -> None:
    """Test StringCheckEvaluator."""
    evaluator = StringCheckEvaluator(config={"case_sensitive": True, "strip_whitespace": True})
    assert evaluator.evaluate("hello", "hello") == 1.0
    assert evaluator.evaluate("hello", "Hello") == 0.0
    assert evaluator.evaluate(" hello ", "hello") == 1.0  # strip_whitespace

    evaluator_case_insensitive = StringCheckEvaluator(
        config={"case_sensitive": False, "strip_whitespace": True}
    )
    assert evaluator_case_insensitive.evaluate("hello", "Hello") == 1.0


def test_levenshtein_evaluator() -> None:
    """Test LevenshteinEvaluator."""
    evaluator = LevenshteinEvaluator(config={})
    assert evaluator.evaluate("hello", "hello") == 1.0
    assert evaluator.evaluate("hello", "helo") > 0.0  # Similar but not exact
    assert evaluator.evaluate("hello", "helo") < 1.0


def test_evaluator_factory_string_name() -> None:
    """Test EvaluatorFactory with string name."""
    evaluator = EvaluatorFactory.create("exact")
    assert isinstance(evaluator, StringCheckEvaluator)


def test_evaluator_factory_score_judge() -> None:
    """Test EvaluatorFactory.create('score_judge') returns ScoreJudge."""
    evaluator = EvaluatorFactory.create("score_judge")
    assert isinstance(evaluator, ScoreJudge)


def test_evaluator_factory_score_model_grader_alias() -> None:
    """Test score_model_grader alias returns ScoreJudge for backward compat."""
    evaluator = EvaluatorFactory.create("score_model_grader")
    assert isinstance(evaluator, ScoreJudge)


def test_evaluator_factory_config_dict() -> None:
    """Test EvaluatorFactory with config dict."""
    evaluator = EvaluatorFactory.create(
        {"type": "exact", "config": {"case_sensitive": False}}
    )
    assert isinstance(evaluator, StringCheckEvaluator)
    assert evaluator.case_sensitive is False


def test_evaluator_factory_custom_class() -> None:
    """Test EvaluatorFactory with custom evaluator class."""

    class CustomEvaluator:
        def __init__(self, config: dict) -> None:
            self.threshold = config.get("threshold", 0.5)

        def evaluate(self, extracted: str, expected: str, **kwargs) -> float:
            return 1.0 if self.threshold > 0.0 else 0.0

    evaluator = EvaluatorFactory.create({"class": CustomEvaluator, "config": {"threshold": 0.8}})
    assert isinstance(evaluator, CustomEvaluator)
    assert evaluator.threshold == 0.8
    assert evaluator.evaluate("test", "test") == 1.0


def test_register_evaluator() -> None:
    """Test registering a custom evaluator."""

    class TestEvaluator:
        def __init__(self, config: dict) -> None:
            self.config = config

        def evaluate(self, extracted: str, expected: str, **kwargs) -> float:
            return 1.0

    register_evaluator("test", TestEvaluator)
    assert "test" in EVALUATOR_REGISTRY
    assert EVALUATOR_REGISTRY["test"] == TestEvaluator

    evaluator = EvaluatorFactory.create("test")
    assert isinstance(evaluator, TestEvaluator)


def test_python_code_evaluator_with_callable() -> None:
    """Test PythonCodeEvaluator with direct callable."""
    def custom_evaluate(
        extracted: Any,
        expected: Any,
        input_data: dict | None = None,
        field_path: str | None = None,
    ) -> float:
        """Custom evaluation function."""
        if extracted == expected:
            return 1.0
        return 0.5

    evaluator = PythonCodeEvaluator(config={"function": custom_evaluate})
    assert evaluator.evaluate("hello", "hello") == 1.0
    assert evaluator.evaluate("hello", "world") == 0.5


def test_python_code_evaluator_with_callable_field_path() -> None:
    """Test PythonCodeEvaluator callable with field_path parameter."""
    def custom_evaluate(
        extracted: Any,
        expected: Any,
        input_data: dict | None = None,
        field_path: str | None = None,
    ) -> float:
        """Custom evaluation function that uses field_path."""
        if field_path == "age":
            diff = abs(extracted - expected)
            if diff == 0:
                return 1.0
            elif diff <= 2:
                return 0.8
            return max(0.0, 1.0 - (diff / 10))
        return 1.0 if extracted == expected else 0.0

    evaluator = PythonCodeEvaluator(config={"function": custom_evaluate})
    assert evaluator.evaluate(30, 30, field_path="age") == 1.0
    assert evaluator.evaluate(30, 31, field_path="age") == 0.8
    assert evaluator.evaluate(30, 35, field_path="age") < 0.8


def test_python_code_evaluator_with_method() -> None:
    """Test PythonCodeEvaluator with a method."""
    class CustomEvaluator:
        def __init__(self, threshold: float):
            self.threshold = threshold

        def evaluate(
            self,
            extracted: float,
            expected: float,
            input_data: dict | None = None,
            field_path: str | None = None,
        ) -> float:
            diff = abs(extracted - expected)
            if diff <= self.threshold:
                return 1.0
            return max(0.0, 1.0 - (diff / expected))

    custom = CustomEvaluator(threshold=0.1)
    evaluator = PythonCodeEvaluator(config={"function": custom.evaluate})
    assert evaluator.evaluate(10.0, 10.05) == 1.0
    assert evaluator.evaluate(10.0, 11.0) < 1.0


def test_python_code_evaluator_callable_validation() -> None:
    """Test that non-callable function raises error."""
    with pytest.raises(ValueError, match="'function' must be a callable"):
        PythonCodeEvaluator(config={"function": "not a callable"})


def test_python_code_evaluator_no_function_error() -> None:
    """Test that providing no function raises error."""
    with pytest.raises(
        ValueError,
        match="'function' must be provided for PythonCodeEvaluator",
    ):
        PythonCodeEvaluator(config={})
