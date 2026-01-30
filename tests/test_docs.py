"""Tests for documentation Python code blocks.

This module tests that Python code blocks in documentation files are syntactically
correct using AST parsing (no execution required, no API keys needed).

For full integration testing with execution, use the examples/ directory or
run with `--run-docs-integration` flag.
"""

import ast
import re
import pytest
from pathlib import Path


DOCS_DIR = Path(__file__).parent.parent / "docs"


def _get_all_md_files(directory: Path) -> list[Path]:
    """Recursively find all markdown files in a directory."""
    if not directory.exists():
        return []
    return list(directory.rglob("*.md"))


def _extract_python_code_blocks(md_file: Path) -> list[tuple[int, str]]:
    """Extract Python code blocks from a markdown file.
    
    Args:
        md_file: Path to markdown file.
        
    Returns:
        List of (line_number, code_block) tuples.
    """
    content = md_file.read_text()
    pattern = r"```python\n(.*?)```"
    blocks = []
    
    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        start_pos = match.start()
        line_number = content[:start_pos].count("\n") + 1
        blocks.append((line_number, code.strip()))
    
    return blocks


def _check_syntax(md_file: Path) -> list[str]:
    """Check Python code blocks for syntax errors.
    
    Args:
        md_file: Path to markdown file.
        
    Returns:
        List of error messages (empty if all blocks pass).
    """
    errors = []
    blocks = _extract_python_code_blocks(md_file)
    
    for line_num, code in blocks:
        if not code or code.startswith("#"):
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(
                f"Line {line_num}: SyntaxError at code line {e.lineno}: {e.msg}\n"
                f"  Code: {e.text.strip() if e.text else code[:50]}..."
            )
    
    return errors


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "guides" / "optimization"),
    ids=lambda p: p.name,
)
def test_optimization_guides_syntax(md_file: Path) -> None:
    """Test optimization guide Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "guides" / "evaluators"),
    ids=lambda p: p.name,
)
def test_evaluator_guides_syntax(md_file: Path) -> None:
    """Test evaluator guide Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "guides" / "advanced"),
    ids=lambda p: p.name,
)
def test_advanced_guides_syntax(md_file: Path) -> None:
    """Test advanced guide Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "concepts"),
    ids=lambda p: p.name,
)
def test_concepts_syntax(md_file: Path) -> None:
    """Test concept documentation Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "use-cases"),
    ids=lambda p: p.name,
)
def test_use_cases_syntax(md_file: Path) -> None:
    """Test use-cases documentation Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "md_file",
    _get_all_md_files(DOCS_DIR / "reference" / "api"),
    ids=lambda p: p.name,
)
def test_api_reference_syntax(md_file: Path) -> None:
    """Test API reference documentation Python code blocks for syntax errors."""
    errors = _check_syntax(md_file)
    assert not errors, f"Syntax errors in {md_file.name}:\n" + "\n".join(errors)


def test_index_syntax() -> None:
    """Test index documentation file Python code blocks for syntax errors."""
    index_file = DOCS_DIR / "index.md"
    if index_file.exists():
        errors = _check_syntax(index_file)
        assert not errors, f"Syntax errors in index.md:\n" + "\n".join(errors)


def test_core_concepts_syntax() -> None:
    """Test core-concepts documentation Python code blocks for syntax errors."""
    core_concepts_file = DOCS_DIR / "core-concepts.md"
    if core_concepts_file.exists():
        errors = _check_syntax(core_concepts_file)
        assert not errors, f"Syntax errors in core-concepts.md:\n" + "\n".join(errors)


def test_cookbook_syntax() -> None:
    """Test cookbook documentation Python code blocks for syntax errors."""
    cookbook_file = DOCS_DIR / "cookbook.md"
    if cookbook_file.exists():
        errors = _check_syntax(cookbook_file)
        assert not errors, f"Syntax errors in cookbook.md:\n" + "\n".join(errors)
