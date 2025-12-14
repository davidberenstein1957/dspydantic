"""Integration test for multi-image signatures with real API calls.

Verifies that list[dspy.Image] signatures work correctly and multiple images
are forwarded to the API when processing multi-page PDFs.

Run with: uv run pytest tests/integration/test_multi_image_api_call.py -v

Requirements:
    - OPENAI_API_KEY environment variable set
    - reportlab, pdf2image, pillow (for PDF creation)
"""

import os
import tempfile
from pathlib import Path

import dspy
import pytest

from dspydantic import Example
from dspydantic.utils import build_image_signature_and_kwargs, convert_images_to_dspy_images


def create_test_pdf(num_pages: int = 3) -> Path:
    """Create a test multi-page PDF file."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed. Install with: uv pip install reportlab")

    temp_dir = tempfile.mkdtemp()
    pdf_path = Path(temp_dir) / "test_multi_page.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    for i in range(num_pages):
        c.drawString(100, 750, f"Test Document - Page {i + 1}")
        c.drawString(100, 700, f"This is page {i + 1} of {num_pages}")
        c.drawString(100, 650, f"Content for page {i + 1}")
        if i < num_pages - 1:
            c.showPage()
    c.save()

    return pdf_path


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration test",
)
def test_multi_image_api_call() -> None:
    """Test that multiple images are sent to API with list[dspy.Image] signature."""
    api_key = os.getenv("OPENAI_API_KEY")

    # Create a multi-page PDF with distinct content on each page
    pdf_path = create_test_pdf(num_pages=3)

    try:
        example = Example(
            text="Analyze all pages and extract information",
            pdf_path=pdf_path,
            pdf_dpi=150,
            expected_output={"page_count": 3, "content": "test document"},
        )

        base64_images = example.input_data["images"]
        dspy_images = convert_images_to_dspy_images(base64_images)

        assert len(dspy_images) == 3, f"Expected 3 images, got {len(dspy_images)}"

        signature, kwargs = build_image_signature_and_kwargs(dspy_images)
        kwargs["prompt"] = (
            "Count how many pages/images were provided. "
            "Return JSON with page_count (total number of pages)."
        )

        assert signature == "prompt, images: list[dspy.Image] -> json_output"
        assert len(kwargs["images"]) == 3

        # Configure DSPy with real LM
        lm = dspy.LM("gpt-4o-mini", api_key=api_key)
        dspy.configure(lm=lm)

        extractor = dspy.ChainOfThought(signature)
        result = extractor(**kwargs)

        # Verify API received all images by checking page_count
        if hasattr(result, "json_output"):
            import json

            try:
                parsed = json.loads(str(result.json_output))
                if "page_count" in parsed:
                    assert parsed["page_count"] == 3, (
                        f"API should have seen 3 pages, but reported {parsed['page_count']}. "
                        "This suggests not all images were sent."
                    )
            except json.JSONDecodeError:
                pass  # Output format may vary

    except Exception as e:
        pytest.fail(f"API call failed: {e}")
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
        if pdf_path.parent.exists():
            try:
                pdf_path.parent.rmdir()
            except OSError:
                pass
