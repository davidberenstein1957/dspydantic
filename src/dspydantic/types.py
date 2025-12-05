"""Type definitions for dspydantic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dspydantic.utils import prepare_input_data


@dataclass
class OptimizationResult:
    """Result of Pydantic model optimization.

    Attributes:
        optimized_descriptions: Dictionary mapping field paths to optimized descriptions.
        optimized_system_prompt: Optimized system prompt (if provided).
        optimized_instruction_prompt: Optimized instruction prompt (if provided).
        metrics: Dictionary containing optimization metrics (score, improvement, etc.).
        baseline_score: Baseline score before optimization.
        optimized_score: Score after optimization.
    """

    optimized_descriptions: dict[str, str]
    optimized_system_prompt: str | None
    optimized_instruction_prompt: str | None
    metrics: dict[str, Any]
    baseline_score: float
    optimized_score: float


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
        ```

    Attributes:
        input_data: Input data dictionary (automatically generated from input parameters).
        expected_output: Expected output. Can be a dict or Pydantic model matching the target schema.
            If a Pydantic model, it will be converted to a dict for comparison.
    """

    def __init__(
        self,
        expected_output: dict[str, Any] | BaseModel,
        text: str | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> None:
        """Initialize an Example.

        Args:
            expected_output: Expected output. Can be a dict or Pydantic model.
            text: Plain text input.
            image_path: Path to an image file to convert to base64.
            image_base64: Base64-encoded image string.
            pdf_path: Path to a PDF file to convert to images.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Raises:
            ValueError: If no input parameters are provided.
        """
        self.expected_output = expected_output

        # Use prepare_input_data to create input_data from parameters
        self.input_data = prepare_input_data(
            text=text,
            image_path=image_path,
            image_base64=image_base64,
            pdf_path=pdf_path,
            pdf_dpi=pdf_dpi,
        )

