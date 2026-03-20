"""Tests for module.py."""

import re
from unittest.mock import MagicMock

from dspydantic.module import PydanticOptimizerModule


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


def test_field_optimizer_uses_class_signature() -> None:
    """Test that field optimizers use OptimizeFieldDescription signature with a docstring."""
    module = PydanticOptimizerModule(
        field_descriptions={"name": "User name"},
    )
    predictor = module.field_optimizers["name"]
    # ChainOfThought wraps a Predict; access inner predict's signature
    sig = predictor.predict.signature
    # Verify the signature has a meaningful instruction (from the docstring)
    instructions = sig.__doc__ or getattr(sig, "instructions", "")
    assert "Rewrite" in instructions
    assert "field description" in instructions.lower()


def test_system_prompt_optimizer_uses_class_signature() -> None:
    """Test that system prompt optimizer uses OptimizeSystemPrompt signature."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
    )
    predictor = module.system_prompt_optimizer
    sig = predictor.predict.signature
    instructions = sig.__doc__ or getattr(sig, "instructions", "")
    assert "system prompt" in instructions.lower()


def test_instruction_prompt_optimizer_uses_class_signature() -> None:
    """Test that instruction prompt optimizer uses OptimizeInstructionPrompt signature."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_instruction_prompt=True,
    )
    predictor = module.instruction_prompt_optimizer
    sig = predictor.predict.signature
    instructions = sig.__doc__ or getattr(sig, "instructions", "")
    assert "instruction prompt" in instructions.lower()
    assert "placeholder" in instructions.lower()


# Patterns that indicate meta-instructions rather than actual descriptions
META_INSTRUCTION_PATTERNS = [
    r"Given the fields",
    r"produce the fields",
    r"Using the fields.*optimize",
    r"your task is to produce",
    r"you are provided with",
]


def test_forward_produces_descriptions_not_meta_instructions() -> None:
    """Test that forward() output contains actual descriptions, not meta-instructions."""
    module = PydanticOptimizerModule(
        field_descriptions={"name": "User name", "age": "User age"},
        has_system_prompt=True,
        has_instruction_prompt=True,
    )

    # Mock optimizers to return realistic field descriptions
    mock_name_result = MagicMock()
    mock_name_result.optimized_field_description = "The full legal name of the user"
    module.field_optimizers["name"] = MagicMock(return_value=mock_name_result)

    mock_age_result = MagicMock()
    mock_age_result.optimized_field_description = "The user's age in years as an integer"
    module.field_optimizers["age"] = MagicMock(return_value=mock_age_result)

    mock_sys_result = MagicMock()
    mock_sys_result.optimized_system_prompt = "Extract user information from text"
    module.system_prompt_optimizer = MagicMock(return_value=mock_sys_result)

    mock_instr_result = MagicMock()
    mock_instr_result.optimized_instruction_prompt = "Parse the text and extract data"
    module.instruction_prompt_optimizer = MagicMock(return_value=mock_instr_result)

    result = module.forward(
        name="User name",
        age="User age",
        system_prompt="Extract user info",
        instruction_prompt="Parse text",
    )

    # Verify output values are actual descriptions
    output_keys = [
        "optimized_name",
        "optimized_age",
        "optimized_system_prompt",
        "optimized_instruction_prompt",
    ]
    for key in output_keys:
        value = getattr(result, key)
        for pattern in META_INSTRUCTION_PATTERNS:
            assert not re.search(pattern, value, re.IGNORECASE), (
                f"Output '{key}' contains meta-instruction pattern '{pattern}': {value}"
            )


def test_contextual_field_signature_embeds_model_name() -> None:
    """Test that contextual signatures embed model name in docstring."""
    module = PydanticOptimizerModule(
        field_descriptions={"name": "User name"},
        model_name="MedicalRecord",
    )
    predictor = module.field_optimizers["name"]
    sig = predictor.predict.signature
    instructions = sig.__doc__ or ""
    assert "MedicalRecord" in instructions


def test_contextual_system_prompt_signature_embeds_model_name() -> None:
    """Test that contextual system prompt signature includes model name."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_system_prompt=True,
        model_name="Reservation",
    )
    sig = module.system_prompt_optimizer.predict.signature
    instructions = sig.__doc__ or ""
    assert "Reservation" in instructions


def test_contextual_instruction_prompt_signature_embeds_model_name() -> None:
    """Test that contextual instruction prompt signature includes model name."""
    module = PydanticOptimizerModule(
        field_descriptions={},
        has_instruction_prompt=True,
        model_name="Transaction",
    )
    sig = module.instruction_prompt_optimizer.predict.signature
    instructions = sig.__doc__ or ""
    assert "Transaction" in instructions


def test_falls_back_to_generic_without_model_name() -> None:
    """Test that without model_name, generic signatures are used."""
    module = PydanticOptimizerModule(
        field_descriptions={"name": "User name"},
    )
    sig = module.field_optimizers["name"].predict.signature
    instructions = sig.__doc__ or ""
    # Generic signature should NOT contain model-specific context
    assert "MedicalRecord" not in instructions
    # But should still have the base instruction
    assert "Rewrite" in instructions

