"""Default evaluation functions for Pydantic model optimization."""

import json
import re
from collections.abc import Callable
from typing import Any, cast

import dspy
from pydantic import BaseModel

from dspydantic.extractor import apply_optimized_descriptions
from dspydantic.types import Example
from dspydantic.utils import convert_images_to_dspy_images


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
    instruction_prompt = optimized_instruction_prompt or ""

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
        "- Does it match the expected JSON schema structure?\n"
        "- Are the field values reasonable and accurate?\n"
        "- Is the data complete?\n"
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
) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
    """Create a default evaluation function that uses the LLM for structured extraction.

    Args:
        lm: The DSPy language model to use for extraction.
        model: The Pydantic model class.
        system_prompt: Optional system prompt.
        instruction_prompt: Optional instruction prompt.
        metric: Comparison metric to use. Options:
            - "exact": Exact string matching (default)
            - "levenshtein": Levenshtein distance-based matching
        judge_lm: Optional separate LM to use as judge when expected_output is None.
        custom_judge_fn: Optional custom judge function to use when expected_output is None.

    Returns:
        An evaluation function that performs structured extraction and compares
        with expected output (or uses judge if expected_output is None).
    """

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
        instruction_prompt_to_use = (
            optimized_instruction_prompt or instruction_prompt or ""
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
        if dspy_images and len(dspy_images) > 0:
            # For vision models, create a signature that accepts images
            # DSPy can handle Image objects directly in signatures
            # For multiple images, pass them as a list or use the first one
            if len(dspy_images) == 1:
                signature = "prompt, image -> json_output"
                extractor = dspy.ChainOfThought(signature)
                result = extractor(prompt=json_prompt, image=dspy_images[0])
            else:
                # For multiple images, pass as a list
                # Note: DSPy may handle this differently depending on the LM
                signature = "prompt, images -> json_output"
                extractor = dspy.ChainOfThought(signature)
                result = extractor(prompt=json_prompt, images=dspy_images)
        else:
            signature = "prompt -> json_output"
            extractor = dspy.ChainOfThought(signature)
            result = extractor(prompt=json_prompt)

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

        # Levenshtein distance function
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # Recursive comparison function
        def compare_dicts(
            extracted: dict[str, Any], expected: dict[str, Any]
        ) -> float:
            """Compare extracted dict with expected dict recursively."""
            if not isinstance(extracted, dict) or not isinstance(expected, dict):
                if metric == "exact":
                    return 1.0 if extracted == expected else 0.0
                else:  # levenshtein
                    str_extracted = str(extracted).strip()
                    str_expected = str(expected).strip()
                    if str_extracted == str_expected:
                        return 1.0
                    max_len = max(len(str_extracted), len(str_expected))
                    if max_len == 0:
                        return 1.0
                    distance = levenshtein_distance(str_extracted, str_expected)
                    similarity = 1.0 - (distance / max_len)
                    return max(0.0, similarity)

            total_fields = 0
            field_scores = []

            for key, expected_value in expected.items():
                total_fields += 1
                if key in extracted:
                    extracted_value = extracted[key]
                    if isinstance(expected_value, dict) and isinstance(
                        extracted_value, dict
                    ):
                        # Recursive comparison for nested dicts
                        nested_score = compare_dicts(extracted_value, expected_value)
                        field_scores.append(nested_score)
                    else:
                        # Compare values based on metric
                        str_extracted = str(extracted_value).strip()
                        str_expected = str(expected_value).strip()

                        if metric == "exact":
                            field_score = 1.0 if str_extracted == str_expected else 0.0
                        else:  # levenshtein
                            if str_extracted == str_expected:
                                field_score = 1.0
                            else:
                                max_len = max(len(str_extracted), len(str_expected))
                                if max_len == 0:
                                    field_score = 1.0
                                else:
                                    distance = levenshtein_distance(
                                        str_extracted, str_expected
                                    )
                                    similarity = 1.0 - (distance / max_len)
                                    field_score = max(0.0, similarity)
                        field_scores.append(field_score)
                else:
                    # Field missing
                    field_scores.append(0.0)

            # Return average score across all fields
            return (
                sum(field_scores) / total_fields if total_fields > 0 else 0.0
            )

        score = compare_dicts(extracted_data, expected)
        return max(0.0, min(1.0, score))  # Ensure score is between 0.0 and 1.0

    return evaluate

