"""Tests for Prompter class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import BaseModel, Field

from dspydantic import Prompter
from dspydantic.types import Example, OptimizationResult


class User(BaseModel):
    """User model for testing."""

    name: str = Field(description="User name")
    age: int = Field(description="User age")


class Product(BaseModel):
    """Product model for testing."""

    name: str = Field(description="Product name")
    price: float = Field(description="Product price")


def test_prompter_initialization():
    """Test Prompter initialization."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=User)

    assert prompter.model == User
    assert prompter.optimized_descriptions == {}


def test_prompter_initialization_with_prompts():
    """Test Prompter initialization with prompts."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(
        model=User,
        system_prompt="You are helpful",
        instruction_prompt="Extract information",
    )

    assert prompter.system_prompt == "You are helpful"
    assert prompter.instruction_prompt == "Extract information"


def test_prompter_from_optimization_result():
    """Test creating Prompter from OptimizationResult."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    result = OptimizationResult(
        optimized_descriptions={"name": "Optimized name description"},
        optimized_system_prompt="Optimized system",
        optimized_instruction_prompt="Optimized instruction",
        metrics={},
        baseline_score=0.5,
        optimized_score=0.8,
    )

    prompter = Prompter.from_optimization_result(
        model=User,
        optimization_result=result,
    )

    assert prompter.model == User
    assert prompter.optimized_descriptions == {"name": "Optimized name description"}
    assert prompter.optimized_system_prompt == "Optimized system"
    assert prompter.optimized_instruction_prompt == "Optimized instruction"


@patch("dspydantic.prompter.PydanticOptimizer")
def test_prompter_optimize(mock_optimizer_class):
    """Test Prompter.optimize() method."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    # Mock optimizer
    mock_optimizer = MagicMock()
    mock_result = OptimizationResult(
        optimized_descriptions={"name": "Optimized"},
        optimized_system_prompt="Optimized system",
        optimized_instruction_prompt="Optimized instruction",
        metrics={},
        baseline_score=0.5,
        optimized_score=0.8,
    )
    mock_optimizer.optimize.return_value = mock_result
    mock_optimizer_class.return_value = mock_optimizer

    prompter = Prompter(model=User)
    examples = [Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})]

    result = prompter.optimize(examples=examples)

    # Verify optimizer was called
    mock_optimizer_class.assert_called_once()
    mock_optimizer.optimize.assert_called_once()

    # Verify state was updated
    assert prompter.optimized_descriptions == {"name": "Optimized"}
    assert prompter.optimized_system_prompt == "Optimized system"
    assert prompter.optimized_instruction_prompt == "Optimized instruction"

    assert result == mock_result


@patch("dspydantic.prompter.PydanticOptimizer")
def test_prompter_optimize_without_model_creates_output_model(mock_optimizer_class):
    """Test that optimize auto-creates OutputModel when model is None and examples have string outputs."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    # Mock optimizer instance
    mock_optimizer_instance = MagicMock()
    mock_result = OptimizationResult(
        optimized_descriptions={"output": "Optimized output description"},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        metrics={},
        baseline_score=0.5,
        optimized_score=0.8,
    )
    mock_optimizer_instance.optimize.return_value = mock_result

    # Simulate model creation that happens in PydanticOptimizer.__init__
    from dspydantic.types import create_output_model

    created_model = create_output_model()
    mock_optimizer_instance.model = created_model

    mock_optimizer_class.return_value = mock_optimizer_instance

    prompter = Prompter(model=None)
    examples = [Example(text="Some text", expected_output="expected output string")]

    result = prompter.optimize(examples=examples)

    # Verify optimizer was called with None model
    mock_optimizer_class.assert_called_once()
    call_kwargs = mock_optimizer_class.call_args[1]
    assert call_kwargs["model"] is None

    # Verify model was auto-created and stored
    assert prompter.model is not None
    assert prompter.model.__name__ == "OutputModel"
    assert "output" in prompter.model.model_fields

    # Verify state was updated
    assert prompter.optimized_descriptions == {"output": "Optimized output description"}
    assert result == mock_result


@patch("dspydantic.prompter.dspy")
def test_prompter_predict_text(mock_dspy):
    """Test Prompter.predict() with text input."""
    # Configure DSPy
    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm
    mock_dspy.LM.return_value = mock_lm
    mock_dspy.configure = MagicMock()

    mock_result = MagicMock()
    mock_result.json_output = '{"name": "Jane Smith", "age": 25}'
    mock_extractor = MagicMock()
    mock_extractor.return_value = mock_result
    mock_dspy.ChainOfThought.return_value = mock_extractor

    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "User name", "age": "User age"},
    )

    result = prompter.predict(text="Jane Smith, 25")

    # Result is an instance of optimized model (new class), not original User
    assert isinstance(result, BaseModel)
    assert result.name == "Jane Smith"
    assert result.age == 25


def test_prompter_predict_requires_model():
    """Test that predict requires model to be set."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=None)

    with pytest.raises(ValueError, match="model is required"):
        prompter.predict(text="test")


def test_prompter_predict_no_input():
    """Test that predict requires at least one input."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=User)

    with pytest.raises(ValueError, match="No input provided"):
        prompter.predict()


def test_prompter_save():
    """Test Prompter.save() method."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "Optimized name", "age": "Optimized age"},
        optimized_system_prompt="Optimized system",
        optimized_instruction_prompt="Optimized instruction",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        prompter.save(save_path)

        # Verify files exist
        assert (save_path / "dspydantic_metadata.json").exists()
        assert (save_path / "optimized_state.json").exists()
        assert (save_path / "model_schema.json").exists()


def test_prompter_save_requires_model():
    """Test that save requires model to be set."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=None)

    with pytest.raises(ValueError, match="model is required"):
        prompter.save("./test")


def test_prompter_save_requires_optimization():
    """Test that save requires optimization."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=User)

    with pytest.raises(ValueError, match="must be optimized"):
        prompter.save("./test")


def test_prompter_load():
    """Test Prompter.load() method."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    # First save a prompter
    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "Optimized name", "age": "Optimized age"},
        optimized_system_prompt="Optimized system",
        optimized_instruction_prompt="Optimized instruction",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        prompter.save(save_path)

        # Load it (DSPy must be configured before loading)
        loaded = Prompter.load(save_path, model=User)

        assert loaded.model == User
        assert loaded.optimized_descriptions == {"name": "Optimized name", "age": "Optimized age"}
        assert loaded.optimized_system_prompt == "Optimized system"
        assert loaded.optimized_instruction_prompt == "Optimized instruction"


def test_prompter_load_without_model():
    """Test Prompter.load() without providing model."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "Optimized name"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        prompter.save(save_path)

        # Load without model (DSPy must be configured before loading)
        loaded = Prompter.load(save_path)

        assert loaded.model is None
        assert loaded.optimized_descriptions == {"name": "Optimized name"}

        # Predict should fail without model
        with pytest.raises(ValueError, match="model is required"):
            loaded.predict(text="test")


def test_prompter_round_trip():
    """Test complete round-trip: optimize -> save -> load -> predict."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=User)

    # Mock optimization
    with patch("dspydantic.prompter.PydanticOptimizer") as mock_optimizer_class:
        mock_optimizer = MagicMock()
        mock_result = OptimizationResult(
            optimized_descriptions={"name": "User's full name", "age": "User's age in years"},
            optimized_system_prompt="You are a helpful assistant",
            optimized_instruction_prompt="Extract user information",
            metrics={},
            baseline_score=0.5,
            optimized_score=0.9,
        )
        mock_optimizer.optimize.return_value = mock_result
        # Set model attribute so it can be stored after optimization
        mock_optimizer.model = User
        mock_optimizer_class.return_value = mock_optimizer

        examples = [Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})]
        prompter.optimize(examples=examples)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        prompter.save(save_path)

        # Load (DSPy must be configured before loading)
        loaded = Prompter.load(save_path, model=User)

        # Verify state
        assert loaded.optimized_descriptions == {"name": "User's full name", "age": "User's age in years"}
        assert loaded.optimized_system_prompt == "You are a helpful assistant"
        assert loaded.optimized_instruction_prompt == "Extract user information"


def test_prompter_predict_with_template_prompt():
    """Test Prompter.predict() with template prompt formatting."""
    prompter = Prompter(
        model=User,
        optimized_instruction_prompt="Extract from: {text}",
        optimized_descriptions={"name": "Name", "age": "Age"},
    )

    # Mock DSPy
    with patch("dspydantic.prompter.dspy") as mock_dspy:
        mock_lm = MagicMock()
        mock_dspy.settings.lm = mock_lm
        mock_dspy.LM.return_value = mock_lm
        mock_dspy.configure = MagicMock()

        mock_result = MagicMock()
        mock_result.json_output = '{"name": "Jane", "age": 25}'
        mock_extractor = MagicMock()
        mock_extractor.return_value = mock_result
        mock_dspy.ChainOfThought.return_value = mock_extractor

        # Predict with dict for template formatting
        result = prompter.predict(text={"text": "Jane, 25"})

        # Result is an instance of optimized model (new class), not original User
        assert isinstance(result, BaseModel)
        assert result.name == "Jane"
        assert result.age == 25


@patch("dspydantic.prompter.dspy")
def test_prompter_run_is_alias_for_predict(mock_dspy):
    """Test that Prompter.run() is an alias for predict()."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm
    mock_dspy.LM.return_value = mock_lm
    mock_dspy.configure = MagicMock()

    mock_result = MagicMock()
    mock_result.json_output = '{"name": "Alice", "age": 30}'
    mock_extractor = MagicMock()
    mock_extractor.return_value = mock_result
    mock_dspy.ChainOfThought.return_value = mock_extractor

    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "User name", "age": "User age"},
    )

    result = prompter.run(text="Alice, 30")

    assert isinstance(result, BaseModel)
    assert result.name == "Alice"
    assert result.age == 30


@patch("dspydantic.prompter.dspy")
def test_prompter_with_model_id_auto_configures_dspy(mock_dspy):
    """Test that model_id parameter auto-configures DSPy."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = None  # Start unconfigured
    mock_dspy.LM.return_value = mock_lm

    # Create prompter with model_id
    Prompter(model=User, model_id="openai/gpt-4o-mini", api_key="test-key")

    # Verify DSPy was configured (cache=False is default)
    mock_dspy.LM.assert_called_once_with("openai/gpt-4o-mini", api_key="test-key", cache=False)
    mock_dspy.configure.assert_called_once_with(lm=mock_lm)


@patch("dspydantic.prompter.dspy")
def test_prompter_model_id_skips_if_already_configured(mock_dspy):
    """Test that model_id doesn't reconfigure if DSPy already configured."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm  # Already configured

    # Create prompter with model_id
    Prompter(model=User, model_id="openai/gpt-4o-mini")

    # Verify DSPy was NOT reconfigured
    mock_dspy.LM.assert_not_called()
    mock_dspy.configure.assert_not_called()


@patch("dspydantic.prompter.dspy")
def test_prompter_predict_without_optimization(mock_dspy):
    """Test that predict() works without prior optimize() call."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm

    mock_result = MagicMock()
    mock_result.json_output = '{"name": "Bob", "age": 35}'
    mock_extractor = MagicMock()
    mock_extractor.return_value = mock_result
    mock_dspy.ChainOfThought.return_value = mock_extractor

    # Create prompter WITHOUT optimization
    prompter = Prompter(model=User)

    # Predict should work using original field descriptions
    result = prompter.predict(text="Bob, 35")

    assert isinstance(result, BaseModel)
    assert result.name == "Bob"
    assert result.age == 35


@patch("dspydantic.prompter.dspy")
def test_prompter_predict_batch(mock_dspy):
    """Test batch prediction with multiple inputs."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm

    call_count = [0]

    def mock_extractor_call(**kwargs):
        call_count[0] += 1
        mock_result = MagicMock()
        mock_result.json_output = f'{{"name": "User{call_count[0]}", "age": {20 + call_count[0]}}}'
        return mock_result

    mock_extractor = MagicMock()
    mock_extractor.side_effect = mock_extractor_call
    mock_dspy.ChainOfThought.return_value = mock_extractor

    prompter = Prompter(model=User)

    # Batch predict
    inputs = ["User1, 21", "User2, 22", "User3, 23"]
    results = prompter.predict_batch(inputs, max_workers=2)

    assert len(results) == 3
    assert all(isinstance(r, BaseModel) for r in results)


def test_prompter_error_message_no_model():
    """Test helpful error message when model is missing."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=None)

    with pytest.raises(ValueError) as exc_info:
        prompter.predict(text="test")

    assert "model is required" in str(exc_info.value)
    assert "Prompter(model=MyModel" in str(exc_info.value)


def test_prompter_error_message_no_lm():
    """Test helpful error message when LM is not configured."""
    # Reset DSPy config
    dspy.settings.lm = None

    prompter = Prompter(model=User)

    with pytest.raises(ValueError) as exc_info:
        prompter.predict(text="test")

    error_msg = str(exc_info.value)
    assert "No language model configured" in error_msg
    assert "model_id" in error_msg
    assert "dspy.configure" in error_msg


@patch("dspydantic.prompter.dspy")
def test_prompter_with_cache_enabled(mock_dspy):
    """Test that cache parameter configures caching."""
    mock_lm = MagicMock()
    mock_dspy.settings.lm = None
    mock_dspy.LM.return_value = mock_lm

    # Create prompter with cache enabled
    Prompter(model=User, model_id="openai/gpt-4o-mini", cache=True)

    # Verify LM was created with cache enabled
    mock_dspy.LM.assert_called_once()
    call_kwargs = mock_dspy.LM.call_args
    assert call_kwargs[1].get("cache") is True


@patch("dspydantic.prompter.dspy")
def test_prompter_predict_with_confidence(mock_dspy):
    """Test predict_with_confidence returns ExtractionResult."""
    from dspydantic.prompter import ExtractionResult

    mock_lm = MagicMock()
    mock_dspy.settings.lm = mock_lm

    mock_result = MagicMock()
    mock_result.json_output = '{"name": "John Doe", "age": 30}'
    mock_extractor = MagicMock()
    mock_extractor.return_value = mock_result
    mock_dspy.ChainOfThought.return_value = mock_extractor

    prompter = Prompter(
        model=User,
        optimized_descriptions={"name": "User name", "age": "User age"},
    )

    result = prompter.predict_with_confidence(text="John Doe is 30 years old")

    assert isinstance(result, ExtractionResult)
    assert result.data.name == "John Doe"
    assert result.data.age == 30
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
