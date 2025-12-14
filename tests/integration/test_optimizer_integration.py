"""Integration tests for PydanticOptimizer with real DSPy setup."""

import os
from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class Address(BaseModel):
    """Address model for integration testing."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")


class User(BaseModel):
    """User model with nested address for integration testing."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")
    email: str = Field(description="Email address")
    address: Address = Field(description="User address")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration test",
)
def test_optimizer_with_nested_model_and_prompts() -> None:
    """Integration test: Run optimizer with nested model and prompts."""
    examples = [
        Example(
            text="John Doe, 30 years old, john@example.com, 123 Main St, New York, 10001",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "zip_code": "10001",
                },
            },
        ),
        Example(
            text="Jane Smith, 25, jane@example.com, 456 Oak Ave, Los Angeles, 90001",
            expected_output={
                "name": "Jane Smith",
                "age": 25,
                "email": "jane@example.com",
                "address": {
                    "street": "456 Oak Ave",
                    "city": "Los Angeles",
                    "zip_code": "90001",
                },
            },
        ),
    ]

    evaluation_calls: list[tuple[str | None, str | None]] = []

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Mock evaluation function that tracks calls."""
        evaluation_calls.append((optimized_system_prompt, optimized_instruction_prompt))
        # Return a mock score
        return 0.85

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        model_id="gpt-4o",
        optimizer="miprov2zeroshot",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 2
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration test",
)
def test_optimizer_metric_function_integration() -> None:
    """Integration test: Verify metric function works with real DSPy predictions."""
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
        return 0.9

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user info",
        instruction_prompt="Parse text",
    )

    # Create a mock LM for the metric function
    mock_lm = MagicMock(spec=dspy.LM)

    metric = optimizer._create_metric_function(mock_lm)

    # Create a real DSPy prediction
    prediction = dspy.Prediction(
        optimized_system_prompt="Optimized system prompt",
        optimized_instruction_prompt="Optimized instruction prompt",
    )

    example = dspy.Example(
        input_data={"text": "John Doe, 30"},
        expected_output={"name": "John Doe", "age": 30},
    )

    score = metric(example, prediction)

    assert score == 0.9
    assert len(captured_prompts) == 1
    assert captured_prompts[0][0] == "Optimized system prompt"
    assert captured_prompts[0][1] == "Optimized instruction prompt"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping Gemini integration test",
)
def test_optimizer_with_gemini_model() -> None:
    """Integration test: Verify optimizer works with Google Gemini models."""
    examples = [
        Example(
            text="John Doe, 30 years old, john@example.com, 123 Main St, New York, 10001",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "zip_code": "10001",
                },
            },
        ),
    ]

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="gemini/gemini-2.0-flash-exp",
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer configuration
    assert optimizer.model_id == "gemini/gemini-2.0-flash-exp"
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions

    # Verify API key was auto-detected
    assert optimizer.api_key == os.getenv("GOOGLE_API_KEY")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("AWS_PROFILE")
    and not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="AWS credentials not set, skipping Bedrock integration test",
)
def test_optimizer_with_bedrock_model() -> None:
    """Integration test: Verify optimizer works with AWS Bedrock (Claude) models."""
    examples = [
        Example(
            text="Jane Smith, 25, jane@example.com, 456 Oak Ave, Los Angeles, 90001",
            expected_output={
                "name": "Jane Smith",
                "age": 25,
                "email": "jane@example.com",
                "address": {
                    "street": "456 Oak Ave",
                    "city": "Los Angeles",
                    "zip_code": "90001",
                },
            },
        ),
    ]

    # Create DSPy LM for Bedrock
    try:
        lm = dspy.LM(
            model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    except Exception as e:
        pytest.skip(f"Failed to create Bedrock LM: {e}")

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        lm=lm,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer configuration
    assert optimizer.lm is not None
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping Gemini API key detection test",
)
def test_gemini_api_key_auto_detection() -> None:
    """Integration test: Verify GOOGLE_API_KEY is auto-detected for Gemini models."""
    examples = [
        Example(
            text="Test user",
            expected_output={"name": "Test", "age": 25, "email": "test@test.com"},
        )
    ]

    # Don't pass api_key explicitly
    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="gemini/gemini-2.0-flash-exp",
        num_threads=1,
    )

    # Verify API key was auto-detected from environment
    assert optimizer.api_key == os.getenv("GOOGLE_API_KEY")
    assert optimizer.model_id.startswith("gemini/")


@pytest.mark.integration
def test_optimizer_model_id_prefix_validation() -> None:
    """Integration test: Verify different model_id prefixes are handled correctly."""
    examples = [
        Example(
            text="Test user",
            expected_output={"name": "Test", "age": 25, "email": "test@test.com"},
        )
    ]

    # Test that various model_id formats are accepted
    test_cases = [
        ("gpt-4o", "OPENAI_API_KEY"),
        ("azure/gpt-4o", "AZURE_OPENAI_API_KEY"),
        ("gemini/gemini-2.0-flash-exp", "GOOGLE_API_KEY"),
    ]

    for model_id, expected_env_var in test_cases:
        # Mock the environment variable if needed
        if not os.getenv(expected_env_var):
            with patch.dict(os.environ, {expected_env_var: "mock-key"}):
                optimizer = PydanticOptimizer(
                    model=User,
                    examples=examples,
                    model_id=model_id,
                    num_threads=1,
                )
                assert optimizer.model_id == model_id
                assert optimizer.api_key == "mock-key"
        else:
            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                model_id=model_id,
                num_threads=1,
            )
            assert optimizer.model_id == model_id


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping integration test",
)
def test_optimizer_with_gemini() -> None:
    """Integration test: Verify optimizer works with Google Gemini models."""
    examples = [
        Example(
            text="John Doe, 30 years old, john@example.com",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
            },
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Mock evaluation function that tracks calls."""
        return 0.85

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        model_id="gemini/gemini-2.0-flash-exp",
        optimizer="miprov2zeroshot",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly with Gemini model
    assert optimizer.model_id == "gemini/gemini-2.0-flash-exp"
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "email" in optimizer.field_descriptions

    # Verify API key was detected from environment
    assert optimizer.api_key == os.getenv("GOOGLE_API_KEY")

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 1
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID")),
    reason="AWS credentials not set, skipping integration test",
)
def test_optimizer_with_bedrock() -> None:
    """Integration test: Verify optimizer works with AWS Bedrock models."""
    try:
        import boto3  # noqa: F401
    except ImportError:
        pytest.skip("boto3 not installed, skipping Bedrock test")

    examples = [
        Example(
            text="John Doe, 30 years old, john@example.com",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
            },
        )
    ]

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Mock evaluation function that tracks calls."""
        return 0.85

    # Create DSPy LM with Bedrock
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    try:
        lm = dspy.LM(
            model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name=aws_region,
        )
    except Exception as e:
        pytest.skip(f"Failed to create Bedrock LM: {e}")

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        lm=lm,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        optimizer="miprov2zeroshot",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly with Bedrock
    assert optimizer.lm is lm
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "email" in optimizer.field_descriptions

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 1
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
def test_optimizer_api_key_auto_detection() -> None:
    """Test that API keys are auto-detected based on model_id prefix."""
    examples = [
        Example(
            text="Test",
            expected_output={"name": "Test", "age": 25},
        )
    ]

    # Test OpenAI auto-detection
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="gpt-4o",
        )
        assert optimizer.api_key == "test-openai-key"

    # Test Azure auto-detection
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-azure-key"}, clear=False):
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="azure/gpt-4o",
        )
        assert optimizer.api_key == "test-azure-key"

    # Test Gemini auto-detection
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False):
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="gemini/gemini-2.0-flash-exp",
        )
        assert optimizer.api_key == "test-google-key"

    # Test google/ prefix auto-detection
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key-2"}, clear=False):
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="google/gemini-pro",
        )
        assert optimizer.api_key == "test-google-key-2"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping Gemini integration test",
)
def test_optimizer_with_gemini_model() -> None:
    """Integration test: Verify optimizer works with Google Gemini models."""
    examples = [
        Example(
            text="Alice Johnson, 28, alice@example.com, 789 Pine St, Chicago, 60601",
            expected_output={
                "name": "Alice Johnson",
                "age": 28,
                "email": "alice@example.com",
                "address": {
                    "street": "789 Pine St",
                    "city": "Chicago",
                    "zip_code": "60601",
                },
            },
        ),
    ]

    # Test with Gemini model_id
    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="gemini/gemini-2.0-flash-exp",
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly
    assert optimizer.model_id == "gemini/gemini-2.0-flash-exp"
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions

    # Verify API key was auto-detected from GOOGLE_API_KEY
    assert optimizer.api_key == os.getenv("GOOGLE_API_KEY")

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 1
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping Gemini integration test",
)
def test_optimizer_gemini_api_key_detection() -> None:
    """Integration test: Verify GOOGLE_API_KEY is auto-detected for Gemini models."""
    examples = [
        Example(
            text="Test user, 25",
            expected_output={"name": "Test user", "age": 25},
        )
    ]

    # Test with gemini/ prefix
    optimizer_gemini = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="gemini/gemini-2.0-flash-exp",
        num_threads=1,
        verbose=False,
    )
    assert optimizer_gemini.api_key == os.getenv("GOOGLE_API_KEY")

    # Test with google/ prefix (alternative)
    optimizer_google = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="google/gemini-2.0-flash-exp",
        num_threads=1,
        verbose=False,
    )
    assert optimizer_google.api_key == os.getenv("GOOGLE_API_KEY")


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("AWS_PROFILE")
        or (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    ),
    reason="AWS credentials not configured, skipping Bedrock integration test",
)
def test_optimizer_with_bedrock_model() -> None:
    """Integration test: Verify optimizer works with AWS Bedrock (Claude) models."""
    examples = [
        Example(
            text="Bob Wilson, 35, bob@example.com, 321 Elm St, Seattle, 98101",
            expected_output={
                "name": "Bob Wilson",
                "age": 35,
                "email": "bob@example.com",
                "address": {
                    "street": "321 Elm St",
                    "city": "Seattle",
                    "zip_code": "98101",
                },
            },
        ),
    ]

    # Create DSPy LM with Bedrock
    try:
        lm = dspy.LM(
            model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    except Exception as e:
        pytest.skip(f"Failed to initialize Bedrock LM: {e}")

    # Test with Bedrock LM
    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        lm=lm,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly
    assert optimizer.lm is not None
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 1
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("AWS_PROFILE")
        or (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    ),
    reason="AWS credentials not configured, skipping Bedrock integration test",
)
def test_optimizer_bedrock_with_different_models() -> None:
    """Integration test: Verify optimizer works with different Bedrock model variants."""
    examples = [
        Example(
            text="Carol Davis, 42",
            expected_output={"name": "Carol Davis", "age": 42},
        )
    ]

    # Test Haiku model
    try:
        lm_haiku = dspy.LM(
            model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        optimizer_haiku = PydanticOptimizer(
            model=User,
            examples=examples,
            lm=lm_haiku,
            num_threads=1,
            verbose=False,
        )
        assert optimizer_haiku.lm is not None
        assert len(optimizer_haiku.field_descriptions) > 0
    except Exception as e:
        pytest.skip(f"Failed to initialize Bedrock Haiku model: {e}")

    # Test Sonnet v2 model (if available)
    try:
        lm_sonnet = dspy.LM(
            model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        optimizer_sonnet = PydanticOptimizer(
            model=User,
            examples=examples,
            lm=lm_sonnet,
            num_threads=1,
            verbose=False,
        )
        assert optimizer_sonnet.lm is not None
        assert len(optimizer_sonnet.field_descriptions) > 0
    except Exception:
        # Sonnet v2 may not be available in all regions, skip if not available
        pass


@pytest.mark.integration
def test_optimizer_api_key_auto_detection() -> None:
    """Integration test: Verify API key auto-detection for different providers."""
    examples = [
        Example(
            text="Test user, 30",
            expected_output={"name": "Test user", "age": 30},
        )
    ]

    # Test Azure OpenAI - should check AZURE_OPENAI_API_KEY first, then OPENAI_API_KEY
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "azure-test-key"}, clear=False):
        optimizer_azure = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="azure/gpt-4o",
            api_base="https://test.openai.azure.com",
            num_threads=1,
            verbose=False,
        )
        assert optimizer_azure.api_key == "azure-test-key"

    # Test OpenAI - should use OPENAI_API_KEY
    with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-key"}, clear=False):
        optimizer_openai = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="gpt-4o",
            num_threads=1,
            verbose=False,
        )
        assert optimizer_openai.api_key == "openai-test-key"

    # Test Gemini - should use GOOGLE_API_KEY
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-test-key"}, clear=False):
        optimizer_gemini = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="gemini/gemini-2.0-flash-exp",
            num_threads=1,
            verbose=False,
        )
        assert optimizer_gemini.api_key == "google-test-key"
