"""Default evaluation functions for Pydantic model optimization."""

import json
import re
from collections.abc import Callable
from typing import Any, cast

import dspy
from pydantic import BaseModel

from dspydantic.extractors.dspy import apply_optimized_descriptions, extract_field_descriptions
from dspydantic.types import Example
from dspydantic.utils import (
    build_image_signature_and_kwargs,
    convert_images_to_dspy_images,
    format_instruction_prompt_template,
)


def default_judge_fn(
    lm: dspy.LM,
    model: type[BaseModel],
    example: Example,
    extracted_data: dict[str, Any],
    optimized_descriptions: dict[str, str],
    optimized_system_prompt: str | None,
    optimized_instruction_prompt: str | None,
) -> float:
    """Default LLM judge function that evaluates extracted data quality.

    Args:
        lm: The DSPy language model to use for judging.
        model: The Pydantic model class.
        example: The example with input_data.
        extracted_data: The extracted structured data to evaluate.
        optimized_descriptions: Dictionary of optimized field descriptions.
        optimized_system_prompt: Optimized system prompt (if provided).
        optimized_instruction_prompt: Optimized instruction prompt (if provided).

    Returns:
        Score between 0.0 and 1.0 based on LLM judge evaluation.
    """
    # Get input data from example
    input_data = example.input_data
    if isinstance(input_data, BaseModel):
        input_data = input_data.model_dump()

    # Extract text and images from input_data
    input_text: str | None = None
    images: list[str] | None = None

    if isinstance(input_data, dict):
        input_text = input_data.get("text")
        images = input_data.get("images")
        if not input_text and images:
            input_text = "Extract structured data from the provided image(s)."
        elif not input_text:
            input_text = str(input_data)
    else:
        input_text = str(input_data)

    # Build judge prompt
    system_prompt = optimized_system_prompt or ""
    instruction_prompt = (
        format_instruction_prompt_template(optimized_instruction_prompt, example.text_dict) or ""
    )

    # Get model schema for context
    modified_schema = apply_optimized_descriptions(model, optimized_descriptions)

    judge_prompt_parts = []
    if system_prompt:
        judge_prompt_parts.append(f"System: {system_prompt}")
    if instruction_prompt:
        judge_prompt_parts.append(f"Instruction: {instruction_prompt}")

    judge_prompt_parts.append(
        f"\nJSON Schema (expected structure):\n{json.dumps(modified_schema, indent=2)}"
    )

    if input_text:
        judge_prompt_parts.append(f"\nInput text: {input_text}")
    if images:
        judge_prompt_parts.append(f"\nInput images: {len(images)} image(s) provided")

    judge_prompt_parts.append(f"\nExtracted data:\n{json.dumps(extracted_data, indent=2)}")

    judge_prompt_parts.append(
        "\nEvaluate the quality of the extracted data. Consider:\n"
        "- Does it match the expected JSON schema structure "
        "(including nested dictionaries and lists)?\n"
        "- Are the field values reasonable and accurate?\n"
        "- Is the data complete?\n"
        "- Are nested structures (lists, dictionaries) properly formatted "
        "and complete?\n"
        "- Are there any obvious errors or inconsistencies?\n\n"
        "Respond with a JSON object containing a 'score' field (float between 0.0 and 1.0) "
        "and optionally a 'reasoning' field explaining your evaluation."
    )

    judge_prompt = "\n\n".join(judge_prompt_parts)

    # Use DSPy's ChainOfThought to get judge evaluation
    signature = "prompt -> evaluation"
    judge = dspy.ChainOfThought(signature)
    result = judge(prompt=judge_prompt)

    # Extract evaluation from result
    evaluation_text = str(result.evaluation) if hasattr(result, "evaluation") else str(result)

    # Try to parse JSON from evaluation
    try:
        evaluation = json.loads(evaluation_text)
        score = float(evaluation.get("score", 0.5))
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Try to extract score from text using regex
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', evaluation_text)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                score = 0.5
        else:
            # Fallback: try to find a number between 0 and 1
            score_match = re.search(r"\b(0\.\d+|1\.0|1)\b", evaluation_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    score = 0.5
            else:
                score = 0.5

    return max(0.0, min(1.0, score))  # Ensure score is between 0.0 and 1.0


def default_evaluate_fn(
    lm: dspy.LM,
    model: type[BaseModel],
    system_prompt: str | None,
    instruction_prompt: str | None,
    metric: str = "exact",
    judge_lm: dspy.LM | None = None,
    custom_judge_fn: Callable[..., float] | None = None,
    exclude_fields: list[str] | None = None,
) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
    """Create a default evaluation function that uses the LLM for structured extraction.

    Args:
        lm: The DSPy language model to use for extraction.
        model: The Pydantic model class.
        system_prompt: Optional system prompt.
        instruction_prompt: Optional instruction prompt.
        metric: Comparison metric to use. Options:
            - "exact": Exact matching using DeepDiff with deep_distance
              for nested structures (default)
            - "levenshtein": Levenshtein distance-based matching for primitives,
              DeepDiff deep_distance for nested structures
        judge_lm: Optional separate LM to use as judge when expected_output is None.
        custom_judge_fn: Optional custom judge function to use when expected_output is None.
        exclude_fields: Optional list of field paths to exclude from evaluation.
            Field paths use dot notation for nested fields (e.g., ["address.street", "metadata"]).
            Fields matching these paths (or starting with them) will be excluded from scoring.

    Returns:
        An evaluation function that performs structured extraction and compares
        with expected output (or uses judge if expected_output is None).
    """
    from dspydantic.evaluators._scoring import (
        compare_values,
        get_nested_value,
    )

    def evaluate(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Default evaluation function using LLM for structured extraction.

        Args:
            example: The example with input_data and expected_output.
            optimized_descriptions: Dictionary of optimized field descriptions.
            optimized_system_prompt: Optimized system prompt (if provided).
            optimized_instruction_prompt: Optimized instruction prompt (if provided).

        Returns:
            Score between 0.0 and 1.0 based on extraction accuracy.
        """
        # Build the extraction prompt
        system_prompt_to_use = optimized_system_prompt or system_prompt or ""
        instruction_prompt_raw = optimized_instruction_prompt or instruction_prompt or ""
        instruction_prompt_to_use = (
            format_instruction_prompt_template(instruction_prompt_raw, example.text_dict) or ""
        )

        # Get input data from example
        input_data = example.input_data

        # Handle Pydantic models for input_data
        if isinstance(input_data, BaseModel):
            input_data = input_data.model_dump()

        # Extract text and images from input_data
        input_text: str | None = None
        images: list[str] | None = None
        dspy_images: list[Any] | None = None

        if isinstance(input_data, dict):
            input_text = input_data.get("text")
            images = input_data.get("images")
            # Convert base64 images to dspy.Image objects if present
            if images:
                try:
                    dspy_images = convert_images_to_dspy_images(images)
                except ImportError:
                    # If dspy is not available, fall back to base64 strings
                    dspy_images = None
            # If no text but images exist, create a placeholder text
            if not input_text and images:
                input_text = "Extract structured data from the provided image(s)."
            elif not input_text:
                input_text = str(input_data)
        else:
            input_text = str(input_data)

        # Apply optimized descriptions to the Pydantic model schema
        # This creates a JSON schema with the optimized descriptions embedded
        modified_schema = apply_optimized_descriptions(model, optimized_descriptions)

        # Create the full prompt for extraction
        prompt_parts = []
        if system_prompt_to_use:
            prompt_parts.append(f"System: {system_prompt_to_use}")
        if instruction_prompt_to_use:
            prompt_parts.append(f"Instruction: {instruction_prompt_to_use}")

        # Include the JSON schema with optimized descriptions
        # This provides the full structure, types, and optimized descriptions
        prompt_parts.append(
            f"\nJSON Schema (with optimized field descriptions):\n"
            f"{json.dumps(modified_schema, indent=2)}"
        )

        # Also include a summary of field descriptions for clarity
        if optimized_descriptions:
            prompt_parts.append("\nField descriptions summary:")
            for field_path, description in optimized_descriptions.items():
                prompt_parts.append(f"  - {field_path}: {description}")

        if input_text:
            prompt_parts.append(f"\nInput text: {input_text}")
        if images:
            prompt_parts.append(f"\nInput images: {len(images)} image(s) provided")
        prompt_parts.append(
            "\nExtract the structured data according to the JSON schema above "
            "(which includes optimized field descriptions) and return it as valid JSON."
        )

        full_prompt = "\n\n".join(prompt_parts)

        # Use DSPy's LM directly to generate structured output
        # DSPy is configured globally before optimization starts, so we can use it directly
        # Create a prompt that asks for JSON output
        json_prompt = f"{full_prompt}\n\nReturn only valid JSON, no other text."

        # For vision models, we need to pass images in the prompt
        # DSPy's LM can handle images if we format them as data URLs
        if images:
            # Format images as data URLs for vision models
            image_data_urls = [
                f"data:image/png;base64,{img}" for img in images
            ]
            # Add images to the prompt context
            # Note: DSPy's ChainOfThought may need special handling for images
            # For now, we'll include them in the prompt text
            image_context = "\n".join(
                [
                    f"Image {i+1} (base64): {url[:100]}..."
                    for i, url in enumerate(image_data_urls)
                ]
            )
            json_prompt = f"{json_prompt}\n\n{image_context}"

        # Use DSPy's ChainOfThought for extraction
        # This will use the globally configured LM from dspy.settings
        # If we have dspy.Image objects, we can pass them directly to the signature
        # Build signature with proper multi-image support using list[dspy.Image]
        signature, extractor_kwargs = build_image_signature_and_kwargs(dspy_images)
        extractor = dspy.ChainOfThought(signature)
        # Set the prompt and call extractor with image kwargs
        extractor_kwargs["prompt"] = json_prompt

        result = extractor(**extractor_kwargs)

        # Extract output text
        if hasattr(result, "json_output"):
            output_text = str(result.json_output)
        else:
            output_text = str(result)

        # Try to parse JSON directly
        extracted_data = None
        try:
            extracted_data = json.loads(output_text)
        except (json.JSONDecodeError, AttributeError):
            # Try to extract JSON from the text using regex (handles nested objects)
            # Match JSON objects including nested ones
            json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
            json_match = re.search(json_pattern, output_text, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try a more permissive pattern
                    json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted_data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            pass

        # If still no JSON found, return low score
        if extracted_data is None:
            return 0.0

        # Calculate accuracy score
        if not isinstance(extracted_data, dict):
            return 0.0

        # Handle None expected_output: use judge instead of comparison
        expected = example.expected_output
        if expected is None:
            # Check if custom_judge_fn is provided
            if custom_judge_fn is not None:
                # Try calling as judge function (with extracted_data)
                # Cast to Any to handle different function signatures
                judge_fn = cast(Any, custom_judge_fn)
                try:
                    return judge_fn(
                        example,
                        extracted_data,
                        optimized_descriptions,
                        optimized_system_prompt,
                        optimized_instruction_prompt,
                    )
                except TypeError:
                    # Fallback: try with old signature (without extracted_data)
                    # This handles backward compatibility
                    return judge_fn(
                        example,
                        optimized_descriptions,
                        optimized_system_prompt,
                        optimized_instruction_prompt,
                    )
            # Use judge_lm if provided, otherwise use default LM judge
            judge_to_use = judge_lm if judge_lm is not None else lm
            return default_judge_fn(
                judge_to_use,
                model,
                example,
                extracted_data,
                optimized_descriptions,
                optimized_system_prompt,
                optimized_instruction_prompt,
            )

        # Compare extracted data with expected output (existing logic)
        if isinstance(expected, BaseModel):
            expected = expected.model_dump()

        # Get all field paths from the model schema
        all_field_paths = list(extract_field_descriptions(model).keys())

        # Filter to only leaf fields (fields that don't have nested sub-fields)
        # This avoids double-counting when comparing both parent and child fields
        leaf_field_paths = []
        for field_path in all_field_paths:
            # Check if this field path is a prefix of any other field path
            is_leaf = True
            for other_path in all_field_paths:
                if other_path != field_path and other_path.startswith(f"{field_path}."):
                    is_leaf = False
                    break
            if is_leaf:
                leaf_field_paths.append(field_path)

        # Filter out excluded fields
        if exclude_fields:
            excluded_set = set(exclude_fields)
            filtered_leaf_field_paths = []
            for field_path in leaf_field_paths:
                # Check if this field path should be excluded
                # A field is excluded if:
                # 1. It exactly matches an excluded path, or
                # 2. It starts with an excluded path followed by a dot (nested field)
                should_exclude = False
                for excluded_path in excluded_set:
                    if field_path == excluded_path or field_path.startswith(f"{excluded_path}."):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_leaf_field_paths.append(field_path)
            leaf_field_paths = filtered_leaf_field_paths

        # Compute distance for each field and average
        if leaf_field_paths:
            field_scores = []
            for field_path in leaf_field_paths:
                extracted_value = get_nested_value(extracted_data, field_path)
                expected_value = get_nested_value(expected, field_path)

                # If both values are None, consider it a match (field not present)
                if extracted_value is None and expected_value is None:
                    field_scores.append(1.0)
                # If one is None and the other isn't, it's a mismatch
                elif extracted_value is None or expected_value is None:
                    field_scores.append(0.0)
                else:
                    # Compare the values for this field
                    field_score = compare_values(
                        extracted_value, expected_value, metric=metric
                    )
                    field_scores.append(field_score)

            # Average all field scores
            score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        else:
            # Fallback to comparing entire structures if no field paths found
            score = compare_values(extracted_data, expected, metric=metric)

        return max(0.0, min(1.0, score))  # Ensure score is between 0.0 and 1.0

    return evaluate
