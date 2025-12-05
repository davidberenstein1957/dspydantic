"""Utility functions for handling different input types."""

import base64
from pathlib import Path
from typing import Any

try:
    import dspy
except ImportError:
    dspy = None

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    convert_from_path = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


def pdf_to_base64_images(
    pdf_path: str | Path, dpi: int = 300
) -> list[str]:
    """Convert a PDF file to base64-encoded images at specified DPI.

    Args:
        pdf_path: Path to the PDF file.
        dpi: DPI for the converted images (default: 300).

    Returns:
        List of base64-encoded image strings.

    Raises:
        ImportError: If pdf2image or Pillow are not installed.
        FileNotFoundError: If the PDF file doesn't exist.
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install it with: uv pip install pdf2image pillow"
        )
    if Image is None:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install it with: uv pip install pillow"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Convert PDF to images
    images = convert_from_path(str(pdf_path), dpi=dpi)

    # Convert each image to base64
    base64_images = []
    for image in images:
        # Convert PIL Image to base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        base64_images.append(base64_str)

    return base64_images


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to base64-encoded string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded image string.

    Raises:
        ImportError: If Pillow is not installed.
        FileNotFoundError: If the image file doesn't exist.
    """
    if Image is None:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install it with: uv pip install pillow"
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Open and convert image to base64
    with Image.open(image_path) as img:
        import io

        buffer = io.BytesIO()
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = rgb_img
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

    return base64_str


def prepare_input_data(
    text: str | None = None,
    image_path: str | Path | None = None,
    image_base64: str | None = None,
    pdf_path: str | Path | None = None,
    pdf_dpi: int = 300,
) -> dict[str, Any]:
    """Prepare input data dictionary for different input types.

    This function creates a standardized input_data dictionary that can be used
    with the PydanticOptimizer. It supports:
    - Plain text
    - Images (from file path or base64 string)
    - PDFs (converted to images at specified DPI)

    Args:
        text: Plain text input.
        image_path: Path to an image file to convert to base64.
        image_base64: Base64-encoded image string.
        pdf_path: Path to a PDF file to convert to images.
        pdf_dpi: DPI for PDF conversion (default: 300).

    Returns:
        Dictionary with 'text' and/or 'images' keys:
        - 'text': Plain text string (if provided)
        - 'images': List of base64-encoded image strings (if images/PDF provided)

    Examples:
        ```python
        # Plain text
        input_data = prepare_input_data(text="John Doe, 30 years old")

        # Image from file
        input_data = prepare_input_data(image_path="document.png")

        # Image from base64
        input_data = prepare_input_data(image_base64="iVBORw0KG...")

        # PDF
        input_data = prepare_input_data(pdf_path="document.pdf", pdf_dpi=300)

        # Combined text and image
        input_data = prepare_input_data(
            text="Extract information from this document",
            image_path="document.png"
        )
        ```

    Raises:
        ValueError: If no input is provided or conflicting inputs are provided.
    """
    result: dict[str, Any] = {}

    # Handle text
    if text is not None:
        result["text"] = text

    # Handle images
    images: list[str] = []

    if image_path is not None:
        images.append(image_to_base64(image_path))

    if image_base64 is not None:
        images.append(image_base64)

    if pdf_path is not None:
        pdf_images = pdf_to_base64_images(pdf_path, dpi=pdf_dpi)
        images.extend(pdf_images)

    if images:
        result["images"] = images

    if not result:
        raise ValueError(
            "At least one input must be provided: text, image_path, image_base64, or pdf_path"
        )

    return result


def base64_to_dspy_image(base64_str: str) -> Any:
    """Convert a base64-encoded image string to a dspy.Image object.

    Args:
        base64_str: Base64-encoded image string.

    Returns:
        dspy.Image object.

    Raises:
        ImportError: If dspy is not installed.
    """
    if dspy is None:
        raise ImportError("dspy is required for image handling. Install it with: uv pip install dspy-ai")

    # Create a data URL from base64 string
    # DSPy's Image.from_url can handle data URLs
    data_url = f"data:image/png;base64,{base64_str}"

    # Use DSPy's Image.from_url to create the Image object
    return dspy.Image.from_url(data_url)


def convert_images_to_dspy_images(images: list[str] | None) -> list[Any] | None:
    """Convert a list of base64-encoded image strings to dspy.Image objects.

    Args:
        images: List of base64-encoded image strings, or None.

    Returns:
        List of dspy.Image objects, or None if input is None.

    Raises:
        ImportError: If dspy is not installed.
    """
    if images is None:
        return None

    return [base64_to_dspy_image(img) for img in images]

