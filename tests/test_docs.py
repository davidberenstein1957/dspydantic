"""Tests for documentation examples using mktestdocs."""

import pytest
from pathlib import Path

try:
    from mktestdocs import check_md_file
except ImportError:
    pytest.skip("mktestdocs not installed", allow_module_level=True)


def test_tutorials():
    """Test tutorial documentation files."""
    docs_dir = Path(__file__).parent.parent / "docs" / "tutorials"
    
    tutorial_files = [
        "getting-started.md",
        "your-first-optimization.md",
        "working-with-images.md",
    ]
    
    for tutorial_file in tutorial_files:
        file_path = docs_dir / tutorial_file
        if file_path.exists():
            check_md_file(file_path)


def test_how_to_guides():
    """Test how-to guide documentation files."""
    docs_dir = Path(__file__).parent.parent / "docs" / "how-to-guides"
    
    guide_files = [
        "optimize-text-extraction.md",
        "optimize-image-classification.md",
        "use-template-prompts.md",
        "configure-evaluators.md",
        "save-and-load-prompters.md",
        "exclude-fields-from-evaluation.md",
        "use-nested-models.md",
    ]
    
    for guide_file in guide_files:
        file_path = docs_dir / guide_file
        if file_path.exists():
            check_md_file(file_path)


def test_index():
    """Test index documentation file."""
    docs_dir = Path(__file__).parent.parent / "docs"
    index_file = docs_dir / "index.md"
    
    if index_file.exists():
        check_md_file(index_file)
