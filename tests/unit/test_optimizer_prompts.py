"""Tests for system prompt and instruction prompt optimization."""

from unittest.mock import MagicMock

import dspy
from pydantic import BaseModel, Field

from dspydantic import Example, OptimizationResult, PydanticOptimizer
from dspydantic.module import PydanticOptimizerModule


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")


def test_pydantic_optimizer_module_system_prompt() -> None:
    """Test that PydanticOptimizerModule optimizes system prompts."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
        has_instruction_prompt=False,
    )

    assert module.has_system_prompt
    assert not module.has_instruction_prompt
    assert hasattr(module, "system_prompt_optimizer")
    assert not hasattr(module, "instruction_prompt_optimizer")


def test_pydantic_optimizer_module_instruction_prompt() -> None:
    """Test that PydanticOptimizerModule optimizes instruction prompts."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=False,
        has_instruction_prompt=True,
    )

    assert not module.has_system_prompt
    assert module.has_instruction_prompt
    assert not hasattr(module, "system_prompt_optimizer")
    assert hasattr(module, "instruction_prompt_optimizer")


def test_pydantic_optimizer_module_both_prompts() -> None:
    """Test that PydanticOptimizerModule can optimize both prompts."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
        has_instruction_prompt=True,
    )

    assert module.has_system_prompt
    assert module.has_instruction_prompt
    assert hasattr(module, "system_prompt_optimizer")
    assert hasattr(module, "instruction_prompt_optimizer")


def test_pydantic_optimizer_module_forward_system_prompt() -> None:
    """Test forward pass with system prompt optimization."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
        has_instruction_prompt=False,
    )

    # Mock the optimizer to return a predictable result
    mock_result = MagicMock()
    mock_result.optimized_system_prompt = "Optimized system prompt"
    module.system_prompt_optimizer = MagicMock(return_value=mock_result)

    result = module.forward(system_prompt="Original system prompt")

    assert hasattr(result, "optimized_system_prompt")
    assert result.optimized_system_prompt == "Optimized system prompt"
    module.system_prompt_optimizer.assert_called_once_with(system_prompt="Original system prompt")


def test_pydantic_optimizer_module_forward_instruction_prompt() -> None:
    """Test forward pass with instruction prompt optimization."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=False,
        has_instruction_prompt=True,
    )

    # Mock the optimizer to return a predictable result
    mock_result = MagicMock()
    mock_result.optimized_instruction_prompt = "Optimized instruction prompt"
    module.instruction_prompt_optimizer = MagicMock(return_value=mock_result)

    result = module.forward(instruction_prompt="Original instruction prompt")

    assert hasattr(result, "optimized_instruction_prompt")
    assert result.optimized_instruction_prompt == "Optimized instruction prompt"
    module.instruction_prompt_optimizer.assert_called_once_with(
        instruction_prompt="Original instruction prompt"
    )


def test_pydantic_optimizer_module_forward_both_prompts() -> None:
    """Test forward pass with both prompts."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
        has_instruction_prompt=True,
    )

    # Mock the optimizers
    mock_system_result = MagicMock()
    mock_system_result.optimized_system_prompt = "Optimized system prompt"
    module.system_prompt_optimizer = MagicMock(return_value=mock_system_result)

    mock_instruction_result = MagicMock()
    mock_instruction_result.optimized_instruction_prompt = "Optimized instruction prompt"
    module.instruction_prompt_optimizer = MagicMock(return_value=mock_instruction_result)

    result = module.forward(
        system_prompt="Original system prompt",
        instruction_prompt="Original instruction prompt",
    )

    assert hasattr(result, "optimized_system_prompt")
    assert hasattr(result, "optimized_instruction_prompt")
    assert result.optimized_system_prompt == "Optimized system prompt"
    assert result.optimized_instruction_prompt == "Optimized instruction prompt"


def test_pydantic_optimizer_initialization_with_system_prompt() -> None:
    """Test PydanticOptimizer initialization with system prompt."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information",
    )

    assert optimizer.system_prompt == "Extract user information"
    assert optimizer.instruction_prompt is None


def test_pydantic_optimizer_initialization_with_instruction_prompt() -> None:
    """Test PydanticOptimizer initialization with instruction prompt."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        instruction_prompt="Parse the following text",
    )

    assert optimizer.system_prompt is None
    assert optimizer.instruction_prompt == "Parse the following text"


def test_pydantic_optimizer_initialization_with_both_prompts() -> None:
    """Test PydanticOptimizer initialization with both prompts."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information",
        instruction_prompt="Parse the following text",
    )

    assert optimizer.system_prompt == "Extract user information"
    assert optimizer.instruction_prompt == "Parse the following text"


def test_pydantic_optimizer_initialization_with_prompts_only() -> None:
    """Test PydanticOptimizer initialization with prompts but no field descriptions."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    class ModelWithoutDescriptions(BaseModel):
        """Model without field descriptions."""

        name: str
        age: int

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return 0.85

    optimizer = PydanticOptimizer(
        model=ModelWithoutDescriptions,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information",
        instruction_prompt="Parse the following text",
    )

    # Should work even without field descriptions
    assert optimizer.system_prompt == "Extract user information"
    assert optimizer.instruction_prompt == "Parse the following text"


def test_pydantic_optimizer_metric_function_extracts_prompts() -> None:
    """Test that metric function correctly extracts optimized prompts."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    captured_prompts: list[tuple[str | None, str | None]] = []

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        captured_prompts.append((optimized_system_prompt, optimized_instruction_prompt))
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Original system prompt",
        instruction_prompt="Original instruction prompt",
    )

    # Create a mock LM for the metric function
    mock_lm = MagicMock(spec=dspy.LM)

    # Create metric function
    metric = optimizer._create_metric_function(mock_lm)

    # Create a mock prediction with optimized prompts
    prediction = dspy.Prediction(
        optimized_system_prompt="Optimized system prompt",
        optimized_instruction_prompt="Optimized instruction prompt",
    )

    # Create a mock example
    example = dspy.Example(
        input_data={"text": "John Doe, 30"},
        expected_output={"name": "John Doe", "age": 30},
    )

    # Call metric function
    score = metric(example, prediction)

    assert score == 0.85
    assert len(captured_prompts) == 1
    assert captured_prompts[0][0] == "Optimized system prompt"
    assert captured_prompts[0][1] == "Optimized instruction prompt"


def test_pydantic_optimizer_metric_function_uses_original_prompts_when_none() -> None:
    """Test that metric function uses original prompts when optimized ones are None."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    captured_prompts: list[tuple[str | None, str | None]] = []

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        captured_prompts.append((optimized_system_prompt, optimized_instruction_prompt))
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Original system prompt",
        instruction_prompt="Original instruction prompt",
    )

    # Create a mock LM for the metric function
    mock_lm = MagicMock(spec=dspy.LM)

    # Create metric function
    metric = optimizer._create_metric_function(mock_lm)

    # Create a mock prediction without optimized prompts
    prediction = dspy.Prediction()

    # Create a mock example
    example = dspy.Example(
        input_data={"text": "John Doe, 30"},
        expected_output={"name": "John Doe", "age": 30},
    )

    # Call metric function
    score = metric(example, prediction)

    assert score == 0.85
    assert len(captured_prompts) == 1
    # Should use original prompts when optimized ones are None
    assert captured_prompts[0][0] == "Original system prompt"
    assert captured_prompts[0][1] == "Original instruction prompt"


def test_pydantic_optimizer_prepare_examples_includes_prompts() -> None:
    """Test that _prepare_dspy_examples includes prompts in examples."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        return 0.85

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information",
        instruction_prompt="Parse the following text",
    )

    dspy_examples = optimizer._prepare_dspy_examples()

    assert len(dspy_examples) == 1
    example = dspy_examples[0]
    assert hasattr(example, "system_prompt")
    assert hasattr(example, "instruction_prompt")
    assert example.system_prompt == "Extract user information"
    assert example.instruction_prompt == "Parse the following text"


def test_optimization_result_includes_prompts() -> None:
    """Test that OptimizationResult includes optimized prompts."""
    result = OptimizationResult(
        optimized_descriptions={"name": "Optimized name"},
        optimized_system_prompt="Optimized system prompt",
        optimized_instruction_prompt="Optimized instruction prompt",
        metrics={"average_score": 0.9},
        baseline_score=0.8,
        optimized_score=0.9,
    )

    assert result.optimized_system_prompt == "Optimized system prompt"
    assert result.optimized_instruction_prompt == "Optimized instruction prompt"


def test_optimization_result_with_none_prompts() -> None:
    """Test that OptimizationResult can have None prompts."""
    result = OptimizationResult(
        optimized_descriptions={"name": "Optimized name"},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        metrics={"average_score": 0.9},
        baseline_score=0.8,
        optimized_score=0.9,
    )

    assert result.optimized_system_prompt is None
    assert result.optimized_instruction_prompt is None


def test_pydantic_optimizer_module_forward_with_field_descriptions_and_prompts() -> None:
    """Test forward pass with both field descriptions and prompts."""
    module = PydanticOptimizerModule(
        field_descriptions={"name": "User name", "age": "User age"},
        has_system_prompt=True,
        has_instruction_prompt=True,
    )

    # Mock the optimizers
    mock_system_result = MagicMock()
    mock_system_result.optimized_system_prompt = "Optimized system prompt"
    module.system_prompt_optimizer = MagicMock(return_value=mock_system_result)

    mock_instruction_result = MagicMock()
    mock_instruction_result.optimized_instruction_prompt = "Optimized instruction prompt"
    module.instruction_prompt_optimizer = MagicMock(return_value=mock_instruction_result)

    mock_field_result_name = MagicMock()
    mock_field_result_name.optimized_field_description = "Optimized name description"
    mock_field_result_age = MagicMock()
    mock_field_result_age.optimized_field_description = "Optimized age description"
    module.field_optimizers["name"] = MagicMock(return_value=mock_field_result_name)
    module.field_optimizers["age"] = MagicMock(return_value=mock_field_result_age)

    result = module.forward(
        system_prompt="Original system prompt",
        instruction_prompt="Original instruction prompt",
        name="User name",
        age="User age",
    )

    assert hasattr(result, "optimized_system_prompt")
    assert hasattr(result, "optimized_instruction_prompt")
    assert hasattr(result, "optimized_name")
    assert hasattr(result, "optimized_age")
    assert result.optimized_system_prompt == "Optimized system prompt"
    assert result.optimized_instruction_prompt == "Optimized instruction prompt"
    assert result.optimized_name == "Optimized name description"
    assert result.optimized_age == "Optimized age description"
