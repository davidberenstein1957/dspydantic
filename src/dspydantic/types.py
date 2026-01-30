"""Type definitions for dspydantic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model

from dspydantic.utils import prepare_input_data


@dataclass
class OptimizationResult:
    """Result of Pydantic model optimization.

    Attributes:
        optimized_descriptions: Dictionary mapping field paths to optimized descriptions.
        optimized_system_prompt: Optimized system prompt (if provided).
        optimized_instruction_prompt: Optimized instruction prompt (if provided).
        optimized_demos: Few-shot examples (input_data, expected_output) for the extraction prompt.
        metrics: Dictionary containing optimization metrics (score, improvement, etc.).
        baseline_score: Baseline score before optimization.
        optimized_score: Score after optimization.
        api_calls: Total number of API calls made during optimization.
        total_tokens: Total tokens used during optimization (if available).
        estimated_cost_usd: Estimated cost in USD (if available).
    """

    optimized_descriptions: dict[str, str]
    optimized_system_prompt: str | None
    optimized_instruction_prompt: str | None
    metrics: dict[str, Any]
    baseline_score: float
    optimized_score: float
    optimized_demos: list[dict[str, Any]] | None = None
    api_calls: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


@dataclass
class PrompterState:
    """State of a Prompter instance for serialization.

    This class contains all the information needed to save and restore a Prompter instance.

    Attributes:
        model_schema: JSON schema of the Pydantic model.
        optimized_descriptions: Dictionary of optimized field descriptions.
        optimized_system_prompt: Optimized system prompt (if any).
        optimized_instruction_prompt: Optimized instruction prompt (if any).
        model_id: LLM model identifier.
        model_config: Model configuration (API base, version, etc.).
        version: dspydantic version for compatibility checking.
        metadata: Additional metadata (timestamp, optimization metrics, etc.).
    """

    model_schema: dict[str, Any]
    optimized_descriptions: dict[str, str]
    optimized_system_prompt: str | None
    optimized_instruction_prompt: str | None
    model_id: str
    model_config: dict[str, Any]
    version: str
    metadata: dict[str, Any]
    optimized_demos: list[dict[str, Any]] | None = None


class Example:
    """Example data for optimization.

    This class automatically prepares input data from various input types:
    - Plain text
    - Images (from file path or base64 string)
    - PDFs (converted to images at specified DPI)

    Examples:
        ```python
        # Plain text
        Example(
            text="John Doe, 30 years old",
            expected_output={"name": "John Doe", "age": 30}
        )

        # Text dict for template formatting
        Example(
            text={"name": "John Doe", "location": "New York"},
            expected_output={"name": "John Doe", "age": 30}
        )

        # Image from file
        Example(
            image_path="document.png",
            expected_output={"name": "John Doe", "age": 30}
        )

        # PDF (converted to 300 DPI images)
        Example(
            pdf_path="document.pdf",
            pdf_dpi=300,
            expected_output={"name": "John Doe", "age": 30}
        )

        # Combined text and image
        Example(
            text="Extract information from this document",
            image_path="document.png",
            expected_output={"name": "John Doe", "age": 30}
        )

        # Image from base64 string
        Example(
            image_base64="iVBORw0KG...",
            expected_output={"name": "John Doe", "age": 30}
        )

        # Without expected_output (uses LLM judge for evaluation)
        Example(
            text="John Doe, 30 years old",
            expected_output=None
        )
        ```

    Attributes:
        input_data: Input data dictionary (automatically generated from input parameters).
        text_dict: Dictionary of text values for template formatting. Used to format
            instruction prompt templates with placeholders like "{key}".
            Set automatically when text parameter is a dict.
        expected_output: Expected output. Can be a str, dict, or Pydantic model matching
            the target schema.
            If a string, it will be wrapped in a single-field model with field name "output".
            If a Pydantic model, it will be converted to a dict for comparison.
            If None, evaluation will use an LLM judge or custom evaluation function instead of
            comparing against expected output.
    """

    def __init__(
        self,
        expected_output: str | dict[str, Any] | BaseModel | None = (None),
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> None:
        """Initialize an Example.

        Args:
            expected_output: Expected output. Can be a str, dict, or Pydantic model.
                If a string, it will be wrapped in a single-field model with field name "output".
                If None, evaluation will use an LLM judge or custom evaluation function.
            text: Plain text input (str) or dictionary of text values for template
                formatting (dict). If a dict, keys correspond to placeholders in
                instruction prompt templates (e.g., {"key": "value"}). If a string,
                it's used as the input text.
            image_path: Path to an image file to convert to base64.
            image_base64: Base64-encoded image string.
            pdf_path: Path to a PDF file to convert to images.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Raises:
            ValueError: If no input parameters are provided.
        """
        self.expected_output = expected_output

        # Store text_dict if text is a dict, otherwise store as text_string
        if isinstance(text, dict):
            self.text_dict = text
            # Extract text from dict if available (for input_data)
            # Check common keys: "text", "review", "content", etc.
            text_string = (
                text.get("text")
                or text.get("review")
                or text.get("content")
                or text.get("input")
                or None
            )
        else:
            self.text_dict = {}
            text_string = text

        # Use prepare_input_data to create input_data from parameters
        # If text_string is None and no other inputs, we'll set input_data manually later
        try:
            self.input_data = prepare_input_data(
                text=text_string,
                image_path=image_path,
                image_base64=image_base64,
                pdf_path=pdf_path,
                pdf_dpi=pdf_dpi,
            )
        except ValueError:
            # If no inputs provided and text is a dict, create empty input_data
            # It can be set manually later if needed
            if isinstance(text, dict):
                self.input_data = {}
            else:
                raise


def create_output_model() -> type[BaseModel]:
    """Create a Pydantic model with a single field 'output' of type str.
    
    This is used when string outputs are provided instead of structured Pydantic models.
    The model will have a single field called 'output' that accepts string values.
    
    Returns:
        A Pydantic model class with a single 'output' field of type str.
        
    Example:
        ```python
        OutputModel = create_output_model()
        instance = OutputModel(output="excellent")
        assert instance.output == "excellent"
        ```
    """
    return create_model(
        "OutputModel",
        output=(str, Field(description="The output value")),
    )

