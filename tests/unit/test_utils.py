"""Tests for utils module."""

from unittest.mock import MagicMock

from dspydantic.utils import build_image_signature_and_kwargs


def test_build_image_signature_no_images() -> None:
    """Test building signature with no images."""
    signature, kwargs = build_image_signature_and_kwargs(None)
    assert signature == "prompt -> json_output"
    assert kwargs == {}


def test_build_image_signature_empty_list() -> None:
    """Test building signature with empty image list."""
    signature, kwargs = build_image_signature_and_kwargs([])
    assert signature == "prompt -> json_output"
    assert kwargs == {}


def test_build_image_signature_single_image() -> None:
    """Test building signature with a single image."""
    mock_image = MagicMock()
    signature, kwargs = build_image_signature_and_kwargs([mock_image])

    assert signature == "prompt, image -> json_output"
    assert kwargs == {"prompt": None, "image": mock_image}


def test_build_image_signature_multiple_images() -> None:
    """Test building signature with multiple images using list[dspy.Image]."""
    mock_image1 = MagicMock()
    mock_image2 = MagicMock()
    mock_image3 = MagicMock()

    signature, kwargs = build_image_signature_and_kwargs([mock_image1, mock_image2, mock_image3])

    # Should use list[dspy.Image] type annotation
    assert signature == "prompt, images: list[dspy.Image] -> json_output"
    assert kwargs == {"prompt": None, "images": [mock_image1, mock_image2, mock_image3]}


def test_build_image_signature_two_images() -> None:
    """Test building signature with exactly two images."""
    mock_image1 = MagicMock()
    mock_image2 = MagicMock()

    signature, kwargs = build_image_signature_and_kwargs([mock_image1, mock_image2])

    assert signature == "prompt, images: list[dspy.Image] -> json_output"
    assert kwargs == {"prompt": None, "images": [mock_image1, mock_image2]}


def test_build_image_signature_kwargs_prompt_placeholder() -> None:
    """Test that kwargs contain prompt placeholder that can be set later."""
    mock_image = MagicMock()
    signature, kwargs = build_image_signature_and_kwargs([mock_image])

    # Prompt should be None initially, allowing it to be set later
    assert kwargs["prompt"] is None

    # Should be able to set prompt
    kwargs["prompt"] = "test prompt"
    assert kwargs["prompt"] == "test prompt"
    assert kwargs["image"] == mock_image
