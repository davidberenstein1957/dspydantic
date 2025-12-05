"""DSPy module for optimizing Pydantic field descriptions and prompts."""

from typing import Any

import dspy


class PydanticOptimizerModule(dspy.Module):
    """DSPy module for optimizing field descriptions, system prompts, and instruction prompts."""

    def __init__(
        self,
        field_descriptions: dict[str, str] | None = None,
        has_system_prompt: bool = False,
        has_instruction_prompt: bool = False,
    ):
        """Initialize the optimizer module.

        Args:
            field_descriptions: Dictionary mapping field paths to their descriptions.
            has_system_prompt: Whether to optimize a system prompt.
            has_instruction_prompt: Whether to optimize an instruction prompt.
        """
        super().__init__()

        # Store field descriptions for optimization
        self.field_descriptions = field_descriptions or {}

        # Create optimizers for each field description
        self.field_optimizers: dict[str, dspy.ChainOfThought] = {}
        for field_path, description in self.field_descriptions.items():
            # Create a signature for optimizing this field's description
            signature = "field_description -> optimized_field_description"
            self.field_optimizers[field_path] = dspy.ChainOfThought(signature)

        # Create optimizers for prompts if needed
        self.has_system_prompt = has_system_prompt
        self.has_instruction_prompt = has_instruction_prompt

        if has_system_prompt:
            signature = "system_prompt -> optimized_system_prompt"
            self.system_prompt_optimizer = dspy.ChainOfThought(signature)

        if has_instruction_prompt:
            signature = "instruction_prompt -> optimized_instruction_prompt"
            self.instruction_prompt_optimizer = dspy.ChainOfThought(signature)

    def forward(
        self,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        **field_descriptions: str,
    ) -> dict[str, Any]:
        """Forward pass for optimization.

        Args:
            system_prompt: System prompt to optimize (if provided).
            instruction_prompt: Instruction prompt to optimize (if provided).
            **field_descriptions: Field descriptions to optimize (keyed by field path).

        Returns:
            Dictionary with optimized field descriptions and prompts.
        """
        optimized: dict[str, Any] = {}

        # Optimize field descriptions
        for field_path, description in field_descriptions.items():
            if field_path in self.field_optimizers:
                optimizer = self.field_optimizers[field_path]
                result = optimizer(field_description=description)
                optimized[f"optimized_{field_path}"] = (
                    result.optimized_field_description
                )

        # Optimize system prompt
        if self.has_system_prompt and system_prompt is not None:
            result = self.system_prompt_optimizer(system_prompt=system_prompt)
            optimized["optimized_system_prompt"] = result.optimized_system_prompt

        # Optimize instruction prompt
        if self.has_instruction_prompt and instruction_prompt is not None:
            result = self.instruction_prompt_optimizer(
                instruction_prompt=instruction_prompt
            )
            optimized["optimized_instruction_prompt"] = (
                result.optimized_instruction_prompt
            )

        return dspy.Prediction(**optimized)

