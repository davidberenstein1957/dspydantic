"""Main optimizer class for Pydantic models using DSPy."""

import os
from collections.abc import Callable
from typing import Any

import dspy
from dspy.teleprompt import (
    GEPA,
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    MIPROv2,
)
from pydantic import BaseModel

from dspydantic.extractor import (
    apply_optimized_descriptions,
    extract_field_descriptions,
)
from dspydantic.module import PydanticOptimizerModule
from dspydantic.types import Example, OptimizationResult
from dspydantic.utils import convert_images_to_dspy_images


class PydanticOptimizer:
    """Optimizer that uses DSPy to optimize Pydantic model field descriptions.

    This class optimizes field descriptions in Pydantic models by using DSPy
    to iteratively improve descriptions based on example data and a custom
    evaluation function.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from dspydantic import PydanticOptimizer

        class User(BaseModel):
            name: str = Field(description="User name")
            age: int = Field(description="User age")

        # Define examples
        examples = [
            Example(
                input_data={"text": "John Doe, 30 years old"},
                expected_output={"name": "John Doe", "age": 30}
            )
        ]

        # Option 1: Create optimizer without evaluation function (uses default with "exact" metric)
        # The default evaluation function uses the same LLM for structured extraction
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            model_id="gpt-4o",
            api_key="your-key"
        )

        # Option 2: Create optimizer with "exact" metric (exact string matching)
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            evaluate_fn="exact",
            model_id="gpt-4o",
            api_key="your-key"
        )

        # Option 3: Create optimizer with "levenshtein" metric (fuzzy matching)
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            evaluate_fn="levenshtein",
            model_id="gpt-4o",
            api_key="your-key"
        )

        # Option 4: Create optimizer with custom evaluation function
        def evaluate(
            example,
            optimized_descriptions,
            optimized_system_prompt,
            optimized_instruction_prompt,
        ):
            # Your custom evaluation logic here
            # Return a score between 0.0 and 1.0
            return 0.85

        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            evaluate_fn=evaluate,
            model_id="gpt-4o",
            api_key="your-key"
        )

        # Option 5: Create optimizer with custom DSPy LM
        import dspy
        custom_lm = dspy.LM("gpt-4o", api_key="your-key")
        optimizer = PydanticOptimizer(
            model=User,
            examples=examples,
            lm=custom_lm  # Pass your custom LM (default eval will use this LM)
        )

        # Optimize
        result = optimizer.optimize()
        print(result.optimized_descriptions)
        ```
    """

    def __init__(
        self,
        model: type[BaseModel],
        examples: list[Example],
        evaluate_fn: Callable[
            [Example, dict[str, str], str | None, str | None], float
        ] | str | None = None,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        lm: dspy.LM | None = None,
        model_id: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        num_threads: int = 4,
        init_temperature: float = 1.0,
        verbose: bool = False,
        optimizer_type: str = "miprov2zeroshot",
        train_split: float = 0.8,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Pydantic optimizer.

        Args:
            model: The Pydantic model class to optimize.
            examples: List of examples to use for optimization.
            evaluate_fn: Optional function that evaluates the quality of optimized prompts.
                Takes (Example, optimized_descriptions dict, optimized_system_prompt,
                optimized_instruction_prompt), returns a float score (0.0-1.0).
                Can also be a string: "exact" for exact matching, "levenshtein" for
                Levenshtein distance-based matching, or None for default evaluation
                that performs structured extraction with the same LLM used for optimization.
            system_prompt: Optional initial system prompt to optimize.
            instruction_prompt: Optional initial instruction prompt to optimize.
            lm: Optional DSPy language model instance. If provided, this will be used
                instead of creating a new one from model_id/api_key/etc. If None,
                a new dspy.LM will be created.
            model_id: The model ID to use for optimization (e.g., "gpt-4o", "azure/gpt-4o").
                Only used if `lm` is None.
            api_key: Optional API key. If None, reads from OPENAI_API_KEY environment variable.
                Only used if `lm` is None.
            api_base: Optional API base URL (for Azure OpenAI or custom endpoints).
                Only used if `lm` is None.
            api_version: Optional API version (for Azure OpenAI). Only used if `lm` is None.
            num_threads: Number of threads for optimization.
            init_temperature: Initial temperature for optimization.
            verbose: If True, print detailed progress information.
            optimizer_type: Type of optimizer to use. Options:
                - "miprov2zeroshot" (default): MIPROv2 for 0-shot optimization
                - "miprov2": Full MIPROv2 optimization
                - "gepa": GEPA optimizer
                - "bootstrapfewshot": BootstrapFewShot optimizer
                - "bootstrapfewshotwithrandomsearch": BootstrapFewShotWithRandomSearch
            train_split: Fraction of examples to use for training (rest for validation).
            optimizer_kwargs: Optional dictionary of additional keyword arguments to pass
                to the optimizer constructor. These will override default parameters.
                For example: {"max_bootstrapped_demos": 8, "auto": "full"}.
        """
        if not examples:
            raise ValueError("At least one example must be provided")

        self.model = model
        self.examples = examples
        self.evaluate_fn = evaluate_fn
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt
        self.lm = lm
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.api_version = api_version
        self.num_threads = num_threads
        self.init_temperature = init_temperature
        self.verbose = verbose
        self.optimizer_type = optimizer_type.lower()
        self.train_split = train_split
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Validate optimizer type
        valid_optimizers = (
            "miprov2zeroshot",
            "miprov2",
            "gepa",
            "bootstrapfewshot",
            "bootstrapfewshotwithrandomsearch",
        )
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"optimizer_type must be one of {valid_optimizers}, got '{optimizer_type}'"
            )

        # Extract field descriptions from Pydantic model
        self.field_descriptions = extract_field_descriptions(self.model)

        # Check that we have something to optimize
        has_field_descriptions = bool(self.field_descriptions)
        has_system_prompt = system_prompt is not None
        has_instruction_prompt = instruction_prompt is not None

        if not (has_field_descriptions or has_system_prompt or has_instruction_prompt):
            raise ValueError(
                "At least one of the following must be provided: "
                "field descriptions (add Field(description=...) to model fields), "
                "system_prompt, or instruction_prompt"
            )

    def _default_evaluate_fn(
        self, lm: dspy.LM, metric: str = "exact"
    ) -> Callable[
        [Example, dict[str, str], str | None, str | None], float
    ]:
        """Create a default evaluation function that uses the LLM for structured extraction.

        Args:
            lm: The DSPy language model to use for extraction.
            metric: Comparison metric to use. Options:
                - "exact": Exact string matching (default)
                - "levenshtein": Levenshtein distance-based matching

        Returns:
            An evaluation function that performs structured extraction and compares
            with expected output.
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
            import json
            import re

            # Build the extraction prompt
            system_prompt = optimized_system_prompt or self.system_prompt or ""
            instruction_prompt = (
                optimized_instruction_prompt or self.instruction_prompt or ""
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
            modified_schema = apply_optimized_descriptions(
                self.model, optimized_descriptions
            )

            # Create the full prompt for extraction
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}")
            if instruction_prompt:
                prompt_parts.append(f"Instruction: {instruction_prompt}")

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

            # Compare extracted data with expected output
            expected = example.expected_output
            if isinstance(expected, BaseModel):
                expected = expected.model_dump()

            # Calculate accuracy score
            if not isinstance(extracted_data, dict):
                return 0.0

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

    def _create_metric_function(self, lm: dspy.LM) -> Callable[..., float]:
        """Create a metric function for DSPy optimization.

        Args:
            lm: The DSPy language model (needed for default evaluation function).

        Returns:
            A function that evaluates prompt performance.
        """
        # Use provided evaluate_fn or create default one
        evaluate_fn = self.evaluate_fn
        if evaluate_fn is None:
            evaluate_fn = self._default_evaluate_fn(lm)
        elif isinstance(evaluate_fn, str):
            # Handle string metrics: "exact" or "levenshtein"
            if evaluate_fn.lower() not in ("exact", "levenshtein"):
                raise ValueError(
                    f'evaluate_fn must be a callable, None, or one of ("exact", "levenshtein"), '
                    f'got "{evaluate_fn}"'
                )
            evaluate_fn = self._default_evaluate_fn(lm, metric=evaluate_fn.lower())

        def metric_function(
            example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
        ) -> float:
            """Evaluate the quality of optimized prompts and descriptions.

            Args:
                example: The DSPy example.
                prediction: The optimized field descriptions and prompts.
                trace: Optional trace from DSPy.

            Returns:
                A score between 0.0 and 1.0.
            """
            # Extract optimized field descriptions from prediction
            optimized_field_descriptions: dict[str, str] = {}
            optimized_system_prompt: str | None = None
            optimized_instruction_prompt: str | None = None

            for key, value in prediction.items():
                if key == "optimized_system_prompt":
                    optimized_system_prompt = value
                elif key == "optimized_instruction_prompt":
                    optimized_instruction_prompt = value
                elif key.startswith("optimized_"):
                    # Extract field path (remove "optimized_" prefix)
                    field_path = key.replace("optimized_", "")
                    optimized_field_descriptions[field_path] = value

            # If no optimized values provided (baseline evaluation), use original
            if not optimized_field_descriptions:
                optimized_field_descriptions = self.field_descriptions.copy()
            if optimized_system_prompt is None:
                optimized_system_prompt = self.system_prompt
            if optimized_instruction_prompt is None:
                optimized_instruction_prompt = self.instruction_prompt

            # Convert DSPy example to our Example type
            # Extract input_data and expected_output from DSPy example
            input_data = getattr(example, "input_data", {})
            expected_output = getattr(example, "expected_output", {})

            # Reconstruct Example from input_data dictionary
            # input_data can contain "text" and/or "images" keys
            # Images might be dspy.Image objects or base64 strings
            text = input_data.get("text") if isinstance(input_data, dict) else None
            images = input_data.get("images") if isinstance(input_data, dict) else None
            images_base64 = (
                input_data.get("images_base64")
                if isinstance(input_data, dict)
                else None
            )

            # Create Example - if we have images, use image_base64 (first image)
            # Prefer images_base64 (original base64) if available,
            # otherwise try to extract from images
            if images_base64 and isinstance(images_base64, list) and len(images_base64) > 0:
                example_obj = Example(
                    image_base64=images_base64[0],
                    expected_output=expected_output,
                )
            elif images and isinstance(images, list) and len(images) > 0:
                # If images are dspy.Image objects, we need to get base64 from them
                # For now, try to get base64 from the original input_data if available
                # Otherwise, create example with text
                if text:
                    example_obj = Example(
                        text=text,
                        expected_output=expected_output,
                    )
                else:
                    # Fallback: try to extract base64 from dspy.Image if possible
                    # dspy.Image objects have a url attribute that might be a data URL
                    first_image = images[0]
                    if hasattr(first_image, "url"):
                        # Extract base64 from data URL if it's a data URL
                        url = first_image.url
                        if url.startswith("data:image"):
                            # Extract base64 part from data URL
                            base64_part = url.split(",")[-1] if "," in url else None
                            if base64_part:
                                example_obj = Example(
                                    image_base64=base64_part,
                                    expected_output=expected_output,
                                )
                            else:
                                example_obj = Example(
                                    text="",
                                    expected_output=expected_output,
                                )
                        else:
                            example_obj = Example(
                                text="",
                                expected_output=expected_output,
                            )
                    else:
                        example_obj = Example(
                            text="",
                            expected_output=expected_output,
                        )
            elif text:
                example_obj = Example(
                    text=text,
                    expected_output=expected_output,
                )
            else:
                # Fallback: create a minimal example
                example_obj = Example(
                    text="",
                    expected_output=expected_output,
                )

            # Use the evaluation function
            score = evaluate_fn(
                example_obj,
                optimized_field_descriptions,
                optimized_system_prompt,
                optimized_instruction_prompt,
            )

            # Ensure score is valid (between 0.0 and 1.0)
            if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
                if self.verbose:
                    print(f"Warning: Invalid score {score}, defaulting to 0.0")
                return 0.0

            return float(score)

        return metric_function

    def _dspy_example_to_example(self, dspy_ex: dspy.Example) -> Example:
        """Convert a DSPy example to our Example object.

        Args:
            dspy_ex: DSPy example object.

        Returns:
            Example object.
        """
        # Extract input_data and expected_output from DSPy example
        input_data = getattr(dspy_ex, "input_data", {})
        expected_output = getattr(dspy_ex, "expected_output", {})

        # Reconstruct Example from input_data dictionary
        # input_data can contain "text" and/or "images" keys
        # Images might be dspy.Image objects or base64 strings
        text = input_data.get("text") if isinstance(input_data, dict) else None
        images = input_data.get("images") if isinstance(input_data, dict) else None
        images_base64 = (
            input_data.get("images_base64")
            if isinstance(input_data, dict)
            else None
        )

        # Create Example - if we have images, use image_base64 (first image)
        # Prefer images_base64 (original base64) if available,
        # otherwise try to extract from images
        if images_base64 and isinstance(images_base64, list) and len(images_base64) > 0:
            return Example(
                image_base64=images_base64[0],
                expected_output=expected_output,
            )
        elif images and isinstance(images, list) and len(images) > 0:
            # If images are dspy.Image objects, we need to get base64 from them
            # For now, try to get base64 from the original input_data if available
            # Otherwise, create example with text
            if text:
                return Example(
                    text=text,
                    expected_output=expected_output,
                )
            else:
                # Fallback: try to extract base64 from dspy.Image if possible
                # dspy.Image objects have a url attribute that might be a data URL
                first_image = images[0]
                if hasattr(first_image, "url"):
                    # Extract base64 from data URL if it's a data URL
                    url = first_image.url
                    if url.startswith("data:image"):
                        # Extract base64 part from data URL
                        base64_part = url.split(",")[-1] if "," in url else None
                        if base64_part:
                            return Example(
                                image_base64=base64_part,
                                expected_output=expected_output,
                            )
                        else:
                            return Example(
                                text="",
                                expected_output=expected_output,
                            )
                    else:
                        return Example(
                            text="",
                            expected_output=expected_output,
                        )
                else:
                    return Example(
                        text="",
                        expected_output=expected_output,
                    )
        elif text:
            return Example(
                text=text,
                expected_output=expected_output,
            )
        else:
            # Fallback: create a minimal example
            return Example(
                text="",
                expected_output=expected_output,
            )

    def _prepare_dspy_examples(self) -> list[dspy.Example]:
        """Prepare examples as DSPy examples.

        Returns:
            List of dspy.Example objects.
        """
        trainset = []
        input_keys = list(self.field_descriptions.keys())

        # Add prompts to input keys if they exist
        if self.system_prompt is not None:
            input_keys.append("system_prompt")
        if self.instruction_prompt is not None:
            input_keys.append("instruction_prompt")

        for ex in self.examples:
            # Convert input_data to dict if it's a Pydantic model
            input_data = ex.input_data
            if isinstance(input_data, BaseModel):
                input_data = input_data.model_dump()

            # Convert base64 images to dspy.Image objects if present
            # This allows DSPy to properly handle images in signatures
            if isinstance(input_data, dict) and "images" in input_data:
                base64_images = input_data.get("images")
                if base64_images:
                    try:
                        dspy_images = convert_images_to_dspy_images(base64_images)
                        # Replace base64 strings with dspy.Image objects
                        # Keep original base64 in a separate key for backward compatibility
                        input_data = input_data.copy()
                        input_data["images"] = dspy_images
                        input_data["images_base64"] = base64_images  # Keep original for reference
                    except ImportError:
                        # If dspy is not available, keep base64 strings
                        pass

            # Convert expected_output to dict if it's a Pydantic model
            expected_output = ex.expected_output
            if isinstance(expected_output, BaseModel):
                expected_output = expected_output.model_dump()

            example_dict: dict[str, Any] = {
                "input_data": input_data,
                "expected_output": expected_output,
            }
            # Add field descriptions as inputs
            example_dict.update(self.field_descriptions)

            # Add prompts as inputs if they exist
            if self.system_prompt is not None:
                example_dict["system_prompt"] = self.system_prompt
            if self.instruction_prompt is not None:
                example_dict["instruction_prompt"] = self.instruction_prompt

            trainset.append(dspy.Example(**example_dict).with_inputs(*input_keys))

        return trainset

    def optimize(self) -> OptimizationResult:
        """Optimize the Pydantic model field descriptions using DSPy.

        Returns:
            OptimizationResult containing optimized descriptions and metrics.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Starting DSPy Pydantic optimization")
            print(f"{'='*60}")
            print(f"Model: {self.model.__name__}")
            print(f"Optimizer: {self.optimizer_type.upper()}")
            print(f"Examples: {len(self.examples)}")
            print(f"Fields to optimize: {len(self.field_descriptions)}")
            print(f"Optimization threads: {self.num_threads}")
            print(f"{'='*60}\n")

        # Configure DSPy LM - use provided lm or create one
        if self.lm is not None:
            lm = self.lm
        elif self.api_base:
            lm = dspy.LM(
                self.model_id,
                api_key=self.api_key,
                api_base=self.api_base,
                api_version=self.api_version,
            )
        else:
            lm = dspy.LM(
                self.model_id,
                api_key=self.api_key,
            )

        # Configure DSPy LM in the main thread before optimization
        # This ensures the LM is available to all threads spawned by the optimizer
        # We configure it here so it's available when the optimizer spawns worker threads
        dspy.configure(lm=lm)

        # Ensure we have a valid evaluation function
        evaluate_fn = self.evaluate_fn
        if evaluate_fn is None:
            evaluate_fn = self._default_evaluate_fn(lm)
        elif isinstance(evaluate_fn, str):
            # Handle string metrics: "exact" or "levenshtein"
            if evaluate_fn.lower() not in ("exact", "levenshtein"):
                raise ValueError(
                    f'evaluate_fn must be a callable, None, or one of ("exact", "levenshtein"), '
                    f'got "{evaluate_fn}"'
                )
            evaluate_fn = self._default_evaluate_fn(lm, metric=evaluate_fn.lower())

        # Create DSPy program with field descriptions and prompts
        program = PydanticOptimizerModule(
            field_descriptions=self.field_descriptions,
            has_system_prompt=self.system_prompt is not None,
            has_instruction_prompt=self.instruction_prompt is not None,
        )

        # Prepare examples for DSPy
        trainset = self._prepare_dspy_examples()

        # Split into train and validation sets
        split_idx = int(len(trainset) * self.train_split)
        train_examples = trainset[:split_idx]
        val_examples = trainset[split_idx:] if split_idx < len(trainset) else trainset

        if self.verbose:
            print(f"Training examples: {len(train_examples)}")
            print(f"Validation examples: {len(val_examples)}")

        # Create metric function
        metric = self._create_metric_function(lm)

        # Initialize optimizer based on optimizer_type
        # Merge default kwargs with user-provided optimizer_kwargs
        if self.optimizer_type == "miprov2zeroshot":
            default_kwargs = {
                "metric": metric,
                "num_threads": self.num_threads,
                "init_temperature": self.init_temperature,
                "auto": "light",
                "max_bootstrapped_demos": 0,  # No few-shot examples
                "max_labeled_demos": 0,  # No labeled examples
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            optimizer = MIPROv2(**merged_kwargs)
        elif self.optimizer_type == "miprov2":
            default_kwargs = {
                "metric": metric,
                "num_threads": self.num_threads,
                "init_temperature": self.init_temperature,
                "auto": "light",
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            optimizer = MIPROv2(**merged_kwargs)
        elif self.optimizer_type == "gepa":
            default_kwargs = {
                "metric": metric,
                "num_threads": self.num_threads,
                "init_temperature": self.init_temperature,
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            optimizer = GEPA(**merged_kwargs)
        elif self.optimizer_type == "bootstrapfewshot":
            default_kwargs = {
                "metric": metric,
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 16,
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            optimizer = BootstrapFewShot(**merged_kwargs)
        elif self.optimizer_type == "bootstrapfewshotwithrandomsearch":
            default_kwargs = {
                "metric": metric,
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 16,
                "num_candidate_programs": 10,
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            optimizer = BootstrapFewShotWithRandomSearch(**merged_kwargs)
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        # Evaluate baseline (original prompts and descriptions) on validation set
        if self.verbose:
            print("\nEvaluating baseline configuration...")

        baseline_scores = []
        for val_ex in val_examples:
            # Convert DSPy example to our Example object
            example_obj = self._dspy_example_to_example(val_ex)
            # Use original prompts and descriptions (no optimization)
            baseline_score = evaluate_fn(
                example_obj,
                self.field_descriptions,
                self.system_prompt,
                self.instruction_prompt,
            )
            baseline_scores.append(baseline_score)

        baseline_avg = (
            sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        )
        if self.verbose:
            print(f"Baseline average score: {baseline_avg:.2%}")

        # Optimize
        if self.verbose:
            print("\nOptimizing prompts and field descriptions...")
            if self.system_prompt:
                print("  - System prompt")
            if self.instruction_prompt:
                print("  - Instruction prompt")
            if self.field_descriptions:
                print(f"  - {len(self.field_descriptions)} field descriptions")

        # Some optimizers don't support valset
        if self.optimizer_type in (
            "miprov2zeroshot",
            "miprov2",
            "gepa",
            "bootstrapfewshotwithrandomsearch",
        ):
            optimized_program = optimizer.compile(
                program,
                trainset=train_examples,
                valset=val_examples,
            )
        else:
            optimized_program = optimizer.compile(
                program,
                trainset=train_examples,
            )

        # Build arguments for optimized program (field descriptions and prompts)
        program_args: dict[str, Any] = {}
        program_args.update(self.field_descriptions)
        if self.system_prompt is not None:
            program_args["system_prompt"] = self.system_prompt
        if self.instruction_prompt is not None:
            program_args["instruction_prompt"] = self.instruction_prompt

        # Test the optimized program to get optimized values
        test_result = optimized_program(**program_args)

        # Extract optimized field descriptions
        optimized_field_descriptions: dict[str, str] = {}
        for field_path in self.field_descriptions.keys():
            attr_name = f"optimized_{field_path}"
            if hasattr(test_result, attr_name):
                optimized_field_descriptions[field_path] = getattr(
                    test_result, attr_name
                )

        # Extract optimized prompts
        optimized_system_prompt: str | None = None
        optimized_instruction_prompt: str | None = None
        if self.system_prompt is not None:
            if hasattr(test_result, "optimized_system_prompt"):
                optimized_system_prompt = getattr(test_result, "optimized_system_prompt")
        if self.instruction_prompt is not None:
            if hasattr(test_result, "optimized_instruction_prompt"):
                optimized_instruction_prompt = getattr(
                    test_result, "optimized_instruction_prompt"
                )

        # Evaluate optimized config on validation set
        if self.verbose:
            print("\nEvaluating optimized configuration...")

        evaluation_scores = []
        for val_ex in val_examples:
            # Get optimized descriptions and prompts for this example
            val_program_args: dict[str, Any] = {}
            for field_path in self.field_descriptions.keys():
                if hasattr(val_ex, field_path):
                    val_program_args[field_path] = getattr(val_ex, field_path)
                else:
                    val_program_args[field_path] = self.field_descriptions[field_path]

            if self.system_prompt is not None:
                val_program_args["system_prompt"] = self.system_prompt
            if self.instruction_prompt is not None:
                val_program_args["instruction_prompt"] = self.instruction_prompt

            prediction = optimized_program(**val_program_args)

            # Extract optimized descriptions and prompts from prediction
            pred_descriptions: dict[str, str] = {}
            pred_system_prompt: str | None = None
            pred_instruction_prompt: str | None = None

            for field_path in self.field_descriptions.keys():
                attr_name = f"optimized_{field_path}"
                if hasattr(prediction, attr_name):
                    pred_descriptions[field_path] = getattr(prediction, attr_name)

            if self.system_prompt is not None:
                if hasattr(prediction, "optimized_system_prompt"):
                    pred_system_prompt = getattr(prediction, "optimized_system_prompt")

            if self.instruction_prompt is not None:
                if hasattr(prediction, "optimized_instruction_prompt"):
                    pred_instruction_prompt = getattr(
                        prediction, "optimized_instruction_prompt"
                    )

            # Convert DSPy example to our Example object
            example_obj = self._dspy_example_to_example(val_ex)
            score = evaluate_fn(
                example_obj,
                pred_descriptions,
                pred_system_prompt,
                pred_instruction_prompt,
            )
            evaluation_scores.append(score)

        avg_score = (
            sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0.0
        )

        # Compare with baseline
        improvement = avg_score - baseline_avg
        improvement_pct = (
            (improvement / baseline_avg * 100) if baseline_avg > 0 else 0.0
        )

        # Only use optimized prompts/descriptions if they improve performance
        if improvement < 0:
            if self.verbose:
                print(
                    f"\n⚠️  WARNING: Optimization decreased performance by {abs(improvement):.2%}"
                )
                print("Keeping original prompts and descriptions instead of optimized ones.")
            optimized_field_descriptions = self.field_descriptions.copy()
            optimized_system_prompt = self.system_prompt
            optimized_instruction_prompt = self.instruction_prompt
            avg_score = baseline_avg
            improvement = 0.0
            improvement_pct = 0.0

        # Build result
        result = OptimizationResult(
            optimized_descriptions=optimized_field_descriptions,
            optimized_system_prompt=optimized_system_prompt,
            optimized_instruction_prompt=optimized_instruction_prompt,
            metrics={
                "average_score": avg_score,
                "baseline_score": baseline_avg,
                "improvement": improvement,
                "improvement_percent": improvement_pct,
                "validation_size": len(val_examples),
                "training_size": len(train_examples),
            },
            baseline_score=baseline_avg,
            optimized_score=avg_score,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print("Optimization complete")
            print(f"{'='*60}")
            print(f"Baseline score: {baseline_avg:.2%}")
            print(f"Final score: {avg_score:.2%}")
            if improvement > 0:
                print(f"Improvement: {improvement:+.2%} ({improvement_pct:+.1f}%)")
            elif improvement < 0:
                print(
                    f"⚠️  Optimization decreased performance by {abs(improvement):.2%}"
                )
                print("Using original field descriptions instead.")
            else:
                print("No change in performance.")
            print(f"{'='*60}\n")

        return result

