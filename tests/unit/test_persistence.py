"""Tests for persistence system."""

import tempfile
from pathlib import Path

import pytest

from dspydantic import __version__
from dspydantic.persistence import PersistenceError, load_prompter_state, save_prompter_state
from dspydantic.types import PrompterState


def test_save_and_load_prompter_state():
    """Test saving and loading PrompterState."""
    state = PrompterState(
        model_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        optimized_descriptions={"name": "The user's full name"},
        optimized_system_prompt="You are a helpful assistant",
        optimized_instruction_prompt="Extract the name",
        model_id="gpt-4.1-mini",
        model_config={"api_base": None, "api_version": None},
        version=__version__,
        metadata={"timestamp": "2024-01-01"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        save_prompter_state(state, save_path)

        # Verify files exist
        assert (save_path / "dspydantic_metadata.json").exists()
        assert (save_path / "optimized_state.json").exists()
        assert (save_path / "model_schema.json").exists()

        # Load and verify
        loaded_state = load_prompter_state(save_path)

        assert loaded_state.model_schema == state.model_schema
        assert loaded_state.optimized_descriptions == state.optimized_descriptions
        assert loaded_state.optimized_system_prompt == state.optimized_system_prompt
        assert loaded_state.optimized_instruction_prompt == state.optimized_instruction_prompt
        assert loaded_state.model_id == state.model_id
        assert loaded_state.version == state.version


def test_load_nonexistent_path():
    """Test loading from nonexistent path raises error."""
    with pytest.raises(PersistenceError, match="does not exist"):
        load_prompter_state("/nonexistent/path")


def test_load_missing_files():
    """Test loading with missing files raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "incomplete"
        save_path.mkdir()

        with pytest.raises(PersistenceError, match="not found"):
            load_prompter_state(save_path)


def test_version_compatibility_major_mismatch():
    """Test version compatibility check for major version mismatch."""
    state = PrompterState(
        model_schema={"type": "object"},
        optimized_descriptions={},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        model_id="gpt-4.1-mini",
        model_config={},
        version="2.0.0",  # Different major version
        metadata={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        save_prompter_state(state, save_path)

        # Should raise error for major version mismatch
        with pytest.raises(PersistenceError, match="Incompatible version"):
            load_prompter_state(save_path)


def test_version_compatibility_minor_mismatch():
    """Test version compatibility check for minor version mismatch (warning only)."""
    state = PrompterState(
        model_schema={"type": "object"},
        optimized_descriptions={},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        model_id="gpt-4.1-mini",
        model_config={},
        version="0.1.0",  # Different minor version (0.1 vs 0.0)
        metadata={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        save_prompter_state(state, save_path)

        # Should warn but continue
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_state = load_prompter_state(save_path)
            assert len(w) > 0
            assert "Version mismatch" in str(w[0].message)

        assert loaded_state is not None


def test_save_creates_directory():
    """Test that save creates directory if it doesn't exist."""
    state = PrompterState(
        model_schema={"type": "object"},
        optimized_descriptions={},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        model_id="gpt-4.1-mini",
        model_config={},
        version=__version__,
        metadata={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "new" / "nested" / "prompter"
        save_prompter_state(state, save_path)

        assert save_path.exists()
        assert (save_path / "dspydantic_metadata.json").exists()


def test_save_with_none_prompts():
    """Test saving with None prompts."""
    state = PrompterState(
        model_schema={"type": "object"},
        optimized_descriptions={"field": "description"},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        model_id="gpt-4.1-mini",
        model_config={},
        version=__version__,
        metadata={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        save_prompter_state(state, save_path)

        loaded_state = load_prompter_state(save_path)
        assert loaded_state.optimized_system_prompt is None
        assert loaded_state.optimized_instruction_prompt is None
