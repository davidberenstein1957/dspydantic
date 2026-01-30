"""Persistence system for saving and loading Prompter instances.

This module handles serialization and deserialization of optimized Prompter state,
including optimized descriptions, prompts, and model configuration.

Files saved:

- `dspydantic_metadata.json` - Version, model configuration
- `optimized_state.json` - Optimized descriptions, prompts, demos
- `model_schema.json` - Pydantic model JSON schema

Example:
    >>> from dspydantic.persistence import save_prompter_state, load_prompter_state
    >>> from dspydantic.types import PrompterState
    >>> state = PrompterState(  # doctest: +SKIP
    ...     model_schema={"type": "object"},
    ...     optimized_descriptions={"name": "Full name"},
    ...     optimized_system_prompt="Extract data",
    ...     optimized_instruction_prompt=None,
    ...     model_id="gpt-4o",
    ...     model_config={},
    ...     version="0.1",
    ...     metadata={},
    ... )
    >>> save_prompter_state(state, "./my_prompter")  # doctest: +SKIP
    >>> loaded = load_prompter_state("./my_prompter")  # doctest: +SKIP
"""

import json
from pathlib import Path

from dspydantic.types import PrompterState

# Import version directly to avoid circular import
try:
    from importlib.metadata import version

    __version__ = version("dspydantic")
except Exception:
    # Fallback if package not installed
    __version__ = "0.0.7"


class PersistenceError(Exception):
    """Error raised when persistence operations fail."""

    pass


def save_prompter_state(
    state: PrompterState,
    save_path: str | Path,
) -> None:
    """Save Prompter state to disk.

    Creates a directory structure with metadata, optimized state, and model schema.

    Args:
        state: PrompterState to save.
        save_path: Path to save directory (will be created if doesn't exist).

    Raises:
        PersistenceError: If save operation fails.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save metadata
        metadata_path = save_path / "dspydantic_metadata.json"
        metadata = {
            "version": state.version,
            "model_id": state.model_id,
            "model_config": state.model_config,
            "metadata": state.metadata,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save optimized state
        optimized_state_path = save_path / "optimized_state.json"
        optimized_state = {
            "optimized_descriptions": state.optimized_descriptions,
            "optimized_system_prompt": state.optimized_system_prompt,
            "optimized_instruction_prompt": state.optimized_instruction_prompt,
            "optimized_demos": getattr(state, "optimized_demos", None) or [],
        }
        with open(optimized_state_path, "w") as f:
            json.dump(optimized_state, f, indent=2)

        # Save model schema
        schema_path = save_path / "model_schema.json"
        with open(schema_path, "w") as f:
            json.dump(state.model_schema, f, indent=2)

    except Exception as e:
        raise PersistenceError(f"Failed to save prompter state: {e}") from e


def load_prompter_state(
    load_path: str | Path,
) -> PrompterState:
    """Load Prompter state from disk.

    Args:
        load_path: Path to load directory.

    Returns:
        PrompterState loaded from disk.

    Raises:
        PersistenceError: If load operation fails or version is incompatible.
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise PersistenceError(f"Save directory does not exist: {load_path}")

    try:
        # Load metadata
        metadata_path = load_path / "dspydantic_metadata.json"
        if not metadata_path.exists():
            raise PersistenceError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check version compatibility
        saved_version = metadata.get("version", "0.0.0")
        current_version = __version__

        # Parse versions (assumes semantic versioning)
        saved_parts = saved_version.split(".")
        current_parts = current_version.split(".")

        if len(saved_parts) >= 1 and len(current_parts) >= 1:
            try:
                saved_major = int(saved_parts[0]) if saved_parts[0].isdigit() else 0
                current_major = int(current_parts[0]) if current_parts[0].isdigit() else 0

                if saved_major != current_major:
                    raise PersistenceError(
                        f"Incompatible version: saved version {saved_version} "
                        f"vs current version {current_version}. "
                        "Major version differences are not supported."
                    )

                # Warn on minor version differences but continue
                if len(saved_parts) >= 2 and len(current_parts) >= 2:
                    saved_minor = int(saved_parts[1]) if saved_parts[1].isdigit() else 0
                    current_minor = int(current_parts[1]) if current_parts[1].isdigit() else 0
                    if saved_minor != current_minor:
                        import warnings

                        warnings.warn(
                            f"Version mismatch: saved version {saved_version} "
                            f"vs current version {current_version}. "
                            "Minor version differences may cause issues.",
                            UserWarning,
                        )
            except (ValueError, IndexError):
                # If version parsing fails, skip compatibility check
                pass

        # Load optimized state
        optimized_state_path = load_path / "optimized_state.json"
        if not optimized_state_path.exists():
            raise PersistenceError(f"Optimized state file not found: {optimized_state_path}")

        with open(optimized_state_path) as f:
            optimized_state = json.load(f)

        # Load model schema
        schema_path = load_path / "model_schema.json"
        if not schema_path.exists():
            raise PersistenceError(f"Model schema file not found: {schema_path}")

        with open(schema_path) as f:
            model_schema = json.load(f)

        # Create PrompterState
        state = PrompterState(
            model_schema=model_schema,
            optimized_descriptions=optimized_state.get("optimized_descriptions", {}),
            optimized_system_prompt=optimized_state.get("optimized_system_prompt"),
            optimized_instruction_prompt=optimized_state.get("optimized_instruction_prompt"),
            model_id=metadata.get("model_id", "gpt-4o"),
            model_config=metadata.get("model_config", {}),
            version=metadata.get("version", __version__),
            metadata=metadata.get("metadata", {}),
            optimized_demos=optimized_state.get("optimized_demos"),
        )

        return state

    except PersistenceError:
        raise
    except Exception as e:
        raise PersistenceError(f"Failed to load prompter state: {e}") from e
