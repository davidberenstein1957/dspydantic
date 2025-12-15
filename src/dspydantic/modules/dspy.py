"""DSPy module for optimizing Pydantic field descriptions and prompts."""

import re
from typing import Any

import dspy


class PydanticOptimizerModule(dspy.Module):
    """DSPy module for optimizing field descriptions, system prompts, and instruction prompts."""

    def __init__(
        self,
        field_descriptions: dict[str, str] | None = None,
        field_types: dict[str, str] | None = None,
        has_system_prompt: bool = False,
        has_instruction_prompt: bool = False,
    ):
        """Initialize the optimizer module.

        Args:
            field_descriptions: Dictionary mapping field paths to their descriptions.
            field_types: Dictionary mapping field paths to their type names as strings.
            has_system_prompt: Whether to optimize a system prompt.
            has_instruction_prompt: Whether to optimize an instruction prompt.
        """
        super().__init__()

        # Store field descriptions and types for optimization
        self.field_descriptions = field_descriptions or {}
        self.field_types = field_types or {}

        # Create optimizers for each field description
        self.field_optimizers: dict[str, dspy.ChainOfThought] = {}
        for field_path, description in self.field_descriptions.items():
            # Create a signature for optimizing this field's description
            # Include field_type in the signature if available
            signature = "field_description, field_type -> optimized_field_description"
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
            # Skip field_type_ prefixed entries (they're metadata, not descriptions)
            if field_path.startswith("field_type_"):
                continue

            if field_path in self.field_optimizers:
                optimizer = self.field_optimizers[field_path]
                # Get field type from keyword arguments (passed as field_type_{field_path})
                # or fall back to stored field_types
                field_type_key = f"field_type_{field_path}"
                field_type = field_descriptions.get(field_type_key, "")
                if not field_type:
                    field_type = self.field_types.get(field_path, "")
                result = optimizer(field_description=description, field_type=field_type)
                optimized[f"optimized_{field_path}"] = (
                    result.optimized_field_description
                )

        # Optimize system prompt
        if self.has_system_prompt and system_prompt is not None:
            result = self.system_prompt_optimizer(system_prompt=system_prompt)
            optimized["optimized_system_prompt"] = result.optimized_system_prompt

        # Optimize instruction prompt
        if self.has_instruction_prompt and instruction_prompt is not None:
            # Check if instruction prompt contains template placeholders
            placeholder_pattern = r"\{([^}]+)\}"
            placeholders = re.findall(placeholder_pattern, instruction_prompt)

            if placeholders:
                # Extract unique placeholders
                unique_placeholders = list(dict.fromkeys(placeholders))
                placeholder_list = ", ".join([f"{{{p}}}" for p in unique_placeholders])

                # Create optimization prompt with clear instructions
                original_prompt_marker = "ORIGINAL_PROMPT_TEMPLATE"
                rewrite_instructions_marker = "PROMPT_TEMPLATE_REWRITE_INSTRUCTIONS"

                # Build field descriptions and types if available
                field_descriptions_text = ""
                if field_descriptions:
                    field_desc_lines = []
                    for key, value in field_descriptions.items():
                        # Skip field_type_ prefixed entries (they're metadata, not descriptions)
                        if key.startswith("field_type_"):
                            continue
                        # Get field type if available (from kwargs or stored types)
                        field_type_key = f"field_type_{key}"
                        field_type = field_descriptions.get(field_type_key, "")
                        if not field_type:
                            field_type = self.field_types.get(key, "")
                        if field_type:
                            field_desc_lines.append(f"- {key} ({field_type}): {value}")
                        else:
                            field_desc_lines.append(f"- {key}: {value}")
                    if field_desc_lines:
                        field_descriptions_text = (
                            "\n\nAdditionally, here are the field descriptions and types "
                            "for the structured information you are helping to extract:"
                            "\n" + "\n".join(field_desc_lines)
                        )

                optimization_prompt = (
                    f"{original_prompt_marker}:\n{instruction_prompt}\n\n"
                    f"{rewrite_instructions_marker}:\n"
                    f"You are provided with a {original_prompt_marker} which is used for structured data "
                    f"extraction.\n\n"
                    f"This template contains placeholders ({placeholder_list}) that will be "
                    f"replaced at runtime. {field_descriptions_text}\n\n"
                    f"You may optimize the template for clarity, logical flow, and effectiveness, "
                    f"including reordering, restructuring, or rephrasing text, while preserving the placeholders.\n\n"
                )

                # Optimize the prompt directly with placeholders intact
                result = self.instruction_prompt_optimizer(instruction_prompt=optimization_prompt)
                optimized_prompt = result.optimized_instruction_prompt

                # Remove the instruction markers and their content
                optimized_prompt = re.sub(
                    r"ORIGINAL_PROMPT_TEMPLATE:.*?PROMPT_TEMPLATE_REWRITE_INSTRUCTIONS:",
                    "",
                    optimized_prompt,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                optimized_prompt = re.sub(
                    r"PROMPT_TEMPLATE_REWRITE_INSTRUCTIONS:.*$",
                    "",
                    optimized_prompt,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                optimized_prompt = re.sub(
                    r"\n\n(?:ORIGINAL_PROMPT_TEMPLATE|PROMPT_TEMPLATE_REWRITE_INSTRUCTIONS):.*?\.?",
                    "",
                    optimized_prompt,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                optimized_prompt = optimized_prompt.strip()

                # Verify all placeholders are present exactly once
                for placeholder_name in unique_placeholders:
                    placeholder_str = f"{{{placeholder_name}}}"
                    count = optimized_prompt.count(placeholder_str)

                    if count == 0:
                        # Missing - restore from original
                        original_idx = instruction_prompt.find(placeholder_str)
                        if original_idx != -1:
                            before = instruction_prompt[max(0, original_idx - 30) : original_idx]
                            after_start = original_idx + len(placeholder_str)
                            after = instruction_prompt[after_start : after_start + 30]
                            context_words = (before + " " + after).split()[:5]
                            restored = False
                            for word in context_words:
                                if len(word) > 3 and word in optimized_prompt:
                                    optimized_prompt = optimized_prompt.replace(
                                        word, f"{word} {placeholder_str}", 1
                                    )
                                    restored = True
                                    break
                            if not restored:
                                optimized_prompt += f" {placeholder_str}"
                    elif count > 1:
                        # Duplicate - keep only the first occurrence
                        first_idx = optimized_prompt.find(placeholder_str)
                        if first_idx != -1:
                            before_first = optimized_prompt[: first_idx + len(placeholder_str)]
                            after_first = optimized_prompt[first_idx + len(placeholder_str) :]
                            after_first = after_first.replace(placeholder_str, "")
                            optimized_prompt = before_first + after_first

                # Final check: if any placeholders are still missing, use original prompt
                all_placeholders_present = all(
                    f"{{{p}}}" in optimized_prompt for p in unique_placeholders
                )
                if not all_placeholders_present:
                    optimized_prompt = instruction_prompt

                optimized["optimized_instruction_prompt"] = optimized_prompt.strip()
            else:
                result = self.instruction_prompt_optimizer(instruction_prompt=instruction_prompt)
                optimized["optimized_instruction_prompt"] = result.optimized_instruction_prompt

        return dspy.Prediction(**optimized)

