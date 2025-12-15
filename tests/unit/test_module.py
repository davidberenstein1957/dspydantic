"""Tests for module.py."""

from dspydantic.modules import PydanticOptimizerModule


def test_pydantic_optimizer_module_initialization() -> None:
    """Test initializing PydanticOptimizerModule."""
    field_descriptions = {"name": "User name", "age": "User age"}
    module = PydanticOptimizerModule(field_descriptions=field_descriptions)

    assert module.field_descriptions == field_descriptions
    assert len(module.field_optimizers) == 2
    assert "name" in module.field_optimizers
    assert "age" in module.field_optimizers
    assert not module.has_system_prompt
    assert not module.has_instruction_prompt


def test_pydantic_optimizer_module_with_prompts() -> None:
    """Test initializing PydanticOptimizerModule with prompts."""
    field_descriptions = {"name": "User name"}
    module = PydanticOptimizerModule(
        field_descriptions=field_descriptions,
        has_system_prompt=True,
        has_instruction_prompt=True,
    )

    assert module.has_system_prompt
    assert module.has_instruction_prompt
    assert hasattr(module, "system_prompt_optimizer")
    assert hasattr(module, "instruction_prompt_optimizer")

