"""Unified Prompter class for optimization and extraction."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel

try:
    from importlib.metadata import version

    __version__ = version("dspydantic")
except Exception:
    __version__ = "0.0.7"

from dspydantic.extractor import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
)
from dspydantic.optimizer import PydanticOptimizer
from dspydantic.persistence import load_prompter_state, save_prompter_state
from dspydantic.types import Example, OptimizationResult, PrompterState
from dspydantic.utils import (
    build_image_signature_and_kwargs,
    convert_images_to_dspy_images,
    format_demo_input,
    format_instruction_prompt_template,
    prepare_input_data,
)


@dataclass
class ExtractionResult:
    """Result of extraction with optional metadata.

    Attributes:
        data: The extracted Pydantic model instance.
        confidence: Confidence score (0.0-1.0) if requested.
        raw_output: Raw LLM output text.
    """

    data: BaseModel
    confidence: float | None = None
    raw_output: str | None = None


def _configure_dspy_if_needed(
    model_id: str | None,
    api_key: str | None,
    cache_dir: str | None = None,
) -> None:
    """Configure DSPy with the given model_id if not already configured.

    Args:
        model_id: LiteLLM model identifier (e.g., "openai/gpt-4o-mini").
        api_key: API key for the model provider. If None, uses environment variable.
        cache_dir: Directory for caching LLM responses.
    """
    if model_id is None:
        return

    if dspy.settings.lm is not None:
        return

    lm = dspy.LM(model_id, api_key=api_key, cache=cache_dir is not None)
    dspy.configure(lm=lm)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)


class Prompter:
    """Unified class for optimizing and extracting with Pydantic models.

    This class combines optimization and extraction functionality in a single interface.
    It wraps PydanticOptimizer and adds extraction capabilities along with save/load.

    Examples:
        Simple usage with model_id (recommended):

            from dspydantic import Prompter
            from pydantic import BaseModel, Field

            class User(BaseModel):
                name: str = Field(description="User name")
                age: int = Field(description="User age")

            # Create prompter with model_id - auto-configures DSPy
            prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")

            # Extract directly (no optimization required)
            data = prompter.run("John Doe, 30 years old")
            print(data.name, data.age)  # John Doe 30

        With optimization:

            examples = [
                Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})
            ]
            result = prompter.optimize(examples=examples)

            # Extract with optimized prompts
            data = prompter.run("Jane Smith, 25")

        Manual DSPy configuration:

            import dspy
            lm = dspy.LM("openai/gpt-4o", api_key="your-key")
            dspy.configure(lm=lm)

            prompter = Prompter(model=User)  # Uses existing DSPy config
    """

    def __init__(
        self,
        model: type[BaseModel] | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        cache: bool | str = False,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        optimized_descriptions: dict[str, str] | None = None,
        optimized_system_prompt: str | None = None,
        optimized_instruction_prompt: str | None = None,
        optimized_demos: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize Prompter.

        Args:
            model: Pydantic model class for extraction schema.
            model_id: LiteLLM model identifier (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-sonnet").
                If provided, automatically configures DSPy. Supports all models via LiteLLM.
            api_key: API key for the model provider. If None, uses environment variable
                (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
            cache: Enable caching. True uses default ".dspydantic_cache", or provide path string.
            system_prompt: Initial system prompt for extraction.
            instruction_prompt: Initial instruction prompt for extraction.
            optimized_descriptions: Pre-optimized field descriptions (for loading).
            optimized_system_prompt: Pre-optimized system prompt (for loading).
            optimized_instruction_prompt: Pre-optimized instruction prompt (for loading).
            optimized_demos: Pre-optimized few-shot examples (for loading).

        Example:
            >>> prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")  # doctest: +SKIP
            >>> data = prompter.run("John Doe, 30")  # doctest: +SKIP
        """
        self.model = model
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt

        # Optimized state (set after optimization or loading)
        self.optimized_descriptions = optimized_descriptions or {}
        self.optimized_system_prompt = optimized_system_prompt
        self.optimized_instruction_prompt = optimized_instruction_prompt
        self.optimized_demos = optimized_demos

        # Internal state
        self._optimizer: PydanticOptimizer | None = None

        # Handle caching
        cache_dir = None
        if cache:
            cache_dir = cache if isinstance(cache, str) else ".dspydantic_cache"

        # Auto-configure DSPy if model_id provided
        _configure_dspy_if_needed(model_id, api_key, cache_dir)

    @classmethod
    def load(
        cls,
        load_path: str | Path,
        model: type[BaseModel] | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        cache: bool | str = False,
    ) -> Prompter:
        """Load Prompter from disk.

        Args:
            load_path: Path to saved prompter directory.
            model: Optional Pydantic model class. If provided, will be used for extraction.
            model_id: LiteLLM model identifier. If provided, auto-configures DSPy.
            api_key: API key for the model provider.
            cache: Enable caching. True uses default ".dspydantic_cache", or provide path.

        Returns:
            Loaded Prompter instance.

        Raises:
            PersistenceError: If load fails or version is incompatible.

        Example:
            >>> prompter = Prompter.load("./my_prompter", model=User, model_id="openai/gpt-4o-mini")  # doctest: +SKIP
            >>> result = prompter.run("John Doe, 30")  # doctest: +SKIP
        """
        state = load_prompter_state(load_path)

        prompter = cls(
            model=model,
            model_id=model_id,
            api_key=api_key,
            cache=cache,
            system_prompt=state.optimized_system_prompt,
            instruction_prompt=state.optimized_instruction_prompt,
            optimized_descriptions=state.optimized_descriptions,
            optimized_system_prompt=state.optimized_system_prompt,
            optimized_instruction_prompt=state.optimized_instruction_prompt,
            optimized_demos=getattr(state, "optimized_demos", None),
        )

        # Store schema for reference
        prompter._saved_schema = state.model_schema

        return prompter

    @classmethod
    def from_optimization_result(
        cls,
        model: type[BaseModel],
        optimization_result: OptimizationResult,
    ) -> Prompter:
        """Create Prompter from OptimizationResult.

        Useful for converting existing PydanticOptimizer results to Prompter.

        Args:
            model: Pydantic model class.
            optimization_result: Result from PydanticOptimizer.optimize().

        Returns:
            Prompter instance with optimized state.

        Note:
            DSPy must be configured with `dspy.configure(lm=dspy.LM(...))` before using
            the returned prompter.
        """
        return cls(
            model=model,
            optimized_descriptions=optimization_result.optimized_descriptions,
            optimized_system_prompt=optimization_result.optimized_system_prompt,
            optimized_instruction_prompt=optimization_result.optimized_instruction_prompt,
            optimized_demos=optimization_result.optimized_demos,
        )

    def optimize(
        self,
        examples: list[Example],
        evaluate_fn: Callable[[Example, dict[str, str], str | None, str | None], float]
        | Callable[[Example, dict[str, Any], dict[str, str], str | None, str | None], float]
        | dspy.LM
        | str
        | None = None,
        optimizer: str | Any | None = None,
        train_split: float = 0.8,
        num_threads: int = 4,
        verbose: bool = False,
        exclude_fields: list[str] | None = None,
        evaluator_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize prompts and field descriptions.

        Uses PydanticOptimizer internally to perform optimization.
        
        If model is None and examples have string expected_output values,
        a model with a single "output" field will be automatically created.

        Args:
            examples: List of examples for optimization.
            evaluate_fn: Evaluation function or string metric.
            optimizer: Optimizer name or instance (auto-selects if None).
            train_split: Training split fraction (default: 0.8).
            num_threads: Number of threads (default: 4).
            verbose: Print progress (default: False).
            exclude_fields: Field names to exclude from evaluation.
            evaluator_config: Evaluator configuration dict.
            **kwargs: Additional kwargs passed to PydanticOptimizer.

        Returns:
            OptimizationResult with optimized descriptions and prompts.
        """
        # Create optimizer (handles None model by auto-creating OutputModel if needed)
        # Uses dspy.settings.lm which should be configured via dspy.configure()
        optimizer_instance = PydanticOptimizer(
            model=self.model,
            examples=examples,
            evaluate_fn=evaluate_fn,
            system_prompt=self.system_prompt,
            instruction_prompt=self.instruction_prompt,
            num_threads=num_threads,
            verbose=verbose,
            optimizer=optimizer,
            train_split=train_split,
            exclude_fields=exclude_fields,
            evaluator_config=evaluator_config,
            **kwargs,
        )

        # Run optimization
        result = optimizer_instance.optimize()

        # Update internal state
        # Store the model from optimizer (may be auto-created OutputModel)
        self.model = optimizer_instance.model
        self.optimized_descriptions = result.optimized_descriptions
        self.optimized_system_prompt = result.optimized_system_prompt
        self.optimized_instruction_prompt = result.optimized_instruction_prompt
        self.optimized_demos = result.optimized_demos

        return result

    def _ensure_configured(self) -> None:
        """Ensure DSPy is configured, with helpful error message if not."""
        if dspy.settings.lm is not None:
            return

        # Build helpful error message
        error_msg = "No language model configured.\n\n"

        if self.model_id:
            error_msg += f"model_id='{self.model_id}' was provided but DSPy configuration failed.\n"
            error_msg += "Check that your API key is set correctly.\n\n"

        error_msg += "To configure, either:\n\n"
        error_msg += "1. Use model_id parameter (recommended):\n"
        error_msg += '   prompter = Prompter(model=MyModel, model_id="openai/gpt-4o-mini")\n\n'
        error_msg += "2. Or configure DSPy manually:\n"
        error_msg += "   import dspy\n"
        error_msg += '   lm = dspy.LM("openai/gpt-4o-mini", api_key="your-key")\n'
        error_msg += "   dspy.configure(lm=lm)\n\n"
        error_msg += "Supported models: openai/gpt-4o, anthropic/claude-3-sonnet, "
        error_msg += "gemini/gemini-pro, and many more via LiteLLM."

        raise ValueError(error_msg)

    def predict(
        self,
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> BaseModel:
        """Extract structured data from input.

        Works with or without prior optimization. If not optimized, uses the
        original field descriptions from the Pydantic model.

        Args:
            text: Input text (str) or dict for template formatting.
            image_path: Path to image file.
            image_base64: Base64-encoded image string.
            pdf_path: Path to PDF file.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Returns:
            Pydantic model instance with extracted data.

        Raises:
            ValueError: If model is not set, no input provided, or LLM not configured.
            ValidationError: If extracted data doesn't match model schema.

        Example:
            >>> prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")  # doctest: +SKIP
            >>> user = prompter.predict(text="John Doe, 30 years old")  # doctest: +SKIP
            >>> print(user.name, user.age)  # doctest: +SKIP
            John Doe 30
        """
        if self.model is None:
            raise ValueError(
                "model is required for extraction.\n\n"
                "Provide a Pydantic model when creating the Prompter:\n"
                "    prompter = Prompter(model=MyModel, model_id='openai/gpt-4o-mini')"
            )

        self._ensure_configured()

        # Prepare input data
        text_string = text if isinstance(text, str) else None
        text_dict = text if isinstance(text, dict) else None

        try:
            input_data = prepare_input_data(
                text=text_string,
                image_path=image_path,
                image_base64=image_base64,
                pdf_path=pdf_path,
                pdf_dpi=pdf_dpi,
            )
        except ValueError as e:
            if text_dict is not None:
                input_data = {}
            else:
                raise ValueError(
                    "No input provided. Provide at least one of:\n"
                    "  - text: str or dict\n"
                    "  - image_path: path to image file\n"
                    "  - image_base64: base64-encoded image\n"
                    "  - pdf_path: path to PDF file"
                ) from e

        # Get descriptions (optimized or original from model)
        descriptions = self.optimized_descriptions or extract_field_descriptions(self.model)

        # Get prompts
        system_prompt = self.optimized_system_prompt or self.system_prompt
        instruction_prompt = self.optimized_instruction_prompt or self.instruction_prompt

        # Format instruction prompt if template
        if instruction_prompt and text_dict:
            instruction_prompt = (
                format_instruction_prompt_template(instruction_prompt, text_dict)
                or instruction_prompt
            )

        # Build extraction prompt
        modified_schema = apply_optimized_descriptions(self.model, descriptions)

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        if instruction_prompt:
            prompt_parts.append(f"Instruction: {instruction_prompt}")

        prompt_parts.append(f"\nJSON Schema:\n{json.dumps(modified_schema, indent=2)}")

        # Few-shot examples
        if self.optimized_demos:
            prompt_parts.append("\nExamples:")
            for i, d in enumerate(self.optimized_demos, 1):
                inp = d.get("input_data") or {}
                out = d.get("expected_output")
                inp_desc = format_demo_input(inp)
                out_str = json.dumps(out) if out is not None else "{}"
                prompt_parts.append(f"  Example {i}:\n    Input: {inp_desc}\n    Output: {out_str}")

        # Add input data
        if isinstance(input_data, dict):
            if "text" in input_data:
                prompt_parts.append(f"\nInput text: {input_data['text']}")
            if "images" in input_data:
                prompt_parts.append(
                    f"\nInput images: {len(input_data['images'])} image(s) provided"
                )
        else:
            prompt_parts.append(f"\nInput: {str(input_data)}")

        prompt_parts.append(
            "\nExtract the structured data according to the JSON schema above "
            "and return it as valid JSON."
        )
        full_prompt = "\n\n".join(prompt_parts)
        json_prompt = f"{full_prompt}\n\nReturn only valid JSON, no other text."

        # Handle images
        images = input_data.get("images") if isinstance(input_data, dict) else None
        dspy_images = None
        if images:
            dspy_images = convert_images_to_dspy_images(images)

        # Build signature and run predictor
        signature, extractor_kwargs = build_image_signature_and_kwargs(dspy_images)
        extractor = dspy.ChainOfThought(signature)
        extractor_kwargs["prompt"] = json_prompt
        result = extractor(**extractor_kwargs)

        # Parse output
        output_text = str(result.json_output) if hasattr(result, "json_output") else str(result)

        # Try to parse JSON
        extracted_data = self._parse_json_output(output_text)

        if extracted_data is None:
            raise ValueError(
                f"Failed to extract valid JSON from LLM output.\n\n"
                f"Output received: {output_text[:300]}...\n\n"
                f"This may indicate the model struggled with the extraction task. "
                f"Try optimizing with examples to improve accuracy."
            )

        # Create optimized model and validate
        OptimizedModel = create_optimized_model(self.model, descriptions)
        return OptimizedModel.model_validate(extracted_data)

    def _parse_json_output(self, output_text: str) -> dict[str, Any] | None:
        """Parse JSON from LLM output, handling various formats."""
        try:
            return json.loads(output_text)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try regex extraction for nested JSON
        json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        json_match = re.search(json_pattern, output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try simpler pattern
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def predict_with_confidence(
        self,
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> ExtractionResult:
        """Extract structured data with confidence score.

        Uses a second LLM call to assess extraction confidence based on
        how well the input matches the extracted fields.

        Args:
            text: Input text (str) or dict for template formatting.
            image_path: Path to image file.
            image_base64: Base64-encoded image string.
            pdf_path: Path to PDF file.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Returns:
            ExtractionResult with data, confidence (0.0-1.0), and raw output.

        Example:
            >>> result = prompter.predict_with_confidence("John Doe, 30")  # doctest: +SKIP
            >>> print(f"{result.data.name}: {result.confidence:.0%} confident")  # doctest: +SKIP
            John Doe: 95% confident
        """
        data = self.predict(
            text=text,
            image_path=image_path,
            image_base64=image_base64,
            pdf_path=pdf_path,
            pdf_dpi=pdf_dpi,
        )

        confidence = self._assess_confidence(text, data)

        return ExtractionResult(data=data, confidence=confidence)

    def _assess_confidence(self, input_text: str | dict | None, extracted: BaseModel) -> float:
        """Assess extraction confidence using simple heuristics.

        Returns confidence 0.0-1.0 based on field population and input coverage.
        """
        if input_text is None:
            return 0.5

        input_str = str(input_text) if isinstance(input_text, dict) else input_text
        extracted_dict = extracted.model_dump()

        populated_fields = 0
        total_fields = len(extracted_dict)

        for field_name, value in extracted_dict.items():
            if value is not None and value != "" and value != []:
                populated_fields += 1

        if total_fields == 0:
            return 0.5

        field_coverage = populated_fields / total_fields

        input_words = set(input_str.lower().split())
        extracted_words = set()
        for value in extracted_dict.values():
            if isinstance(value, str):
                extracted_words.update(value.lower().split())

        if input_words:
            word_overlap = len(input_words & extracted_words) / len(input_words)
        else:
            word_overlap = 0.5

        confidence = (field_coverage * 0.6) + (word_overlap * 0.4)
        return min(1.0, max(0.0, confidence))

    def predict_batch(
        self,
        inputs: list[str | dict[str, str]],
        max_workers: int = 4,
        on_error: str = "raise",
    ) -> list[BaseModel | Exception]:
        """Extract structured data from multiple inputs in parallel.

        Args:
            inputs: List of input texts (str) or dicts for template formatting.
            max_workers: Maximum number of parallel workers (default: 4).
            on_error: Error handling strategy:
                - "raise": Raise first exception encountered
                - "return": Return exceptions in results list

        Returns:
            List of extracted Pydantic model instances (or exceptions if on_error="return").

        Example:
            >>> prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")  # doctest: +SKIP
            >>> texts = ["John Doe, 30", "Jane Smith, 25", "Bob Wilson, 40"]  # doctest: +SKIP
            >>> results = prompter.predict_batch(texts)  # doctest: +SKIP
            >>> for user in results:  # doctest: +SKIP
            ...     print(user.name, user.age)  # doctest: +SKIP
        """
        results: list[BaseModel | Exception] = [None] * len(inputs)  # type: ignore

        def process_item(index: int, item: str | dict[str, str]) -> tuple[int, Any]:
            try:
                result = self.predict(text=item)
                return (index, result)
            except Exception as e:
                if on_error == "raise":
                    raise
                return (index, e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, i, item): i for i, item in enumerate(inputs)}

            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

        return results

    async def apredict(
        self,
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> BaseModel:
        """Async version of predict() for concurrent extraction.

        Args:
            text: Input text (str) or dict for template formatting.
            image_path: Path to image file.
            image_base64: Base64-encoded image string.
            pdf_path: Path to PDF file.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Returns:
            Pydantic model instance with extracted data.

        Example:
            >>> async def main():  # doctest: +SKIP
            ...     prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")
            ...     user = await prompter.apredict(text="John Doe, 30")
            ...     print(user.name)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(
                text=text,
                image_path=image_path,
                image_base64=image_base64,
                pdf_path=pdf_path,
                pdf_dpi=pdf_dpi,
            ),
        )

    async def apredict_batch(
        self,
        inputs: list[str | dict[str, str]],
        max_concurrency: int = 4,
        on_error: str = "raise",
    ) -> list[BaseModel | Exception]:
        """Async batch extraction with controlled concurrency.

        Args:
            inputs: List of input texts (str) or dicts for template formatting.
            max_concurrency: Maximum concurrent requests (default: 4).
            on_error: Error handling strategy ("raise" or "return").

        Returns:
            List of extracted Pydantic model instances.

        Example:
            >>> async def main():  # doctest: +SKIP
            ...     prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")
            ...     texts = ["John Doe, 30", "Jane Smith, 25"]
            ...     results = await prompter.apredict_batch(texts)
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_semaphore(index: int, item: str | dict[str, str]) -> tuple[int, Any]:
            async with semaphore:
                try:
                    result = await self.apredict(text=item)
                    return (index, result)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    return (index, e)

        tasks = [process_with_semaphore(i, item) for i, item in enumerate(inputs)]
        completed = await asyncio.gather(*tasks, return_exceptions=(on_error == "return"))

        # Sort by original index
        results: list[BaseModel | Exception] = [None] * len(inputs)  # type: ignore
        for item in completed:
            if isinstance(item, Exception) and on_error == "raise":
                raise item
            if isinstance(item, tuple):
                index, result = item
                results[index] = result

        return results

    def run(
        self,
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> BaseModel:
        """Alias for predict() - extract structured data from input.

        Args:
            text: Input text (str) or dict for template formatting.
            image_path: Path to image file.
            image_base64: Base64-encoded image string.
            pdf_path: Path to PDF file.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Returns:
            Pydantic model instance with extracted data.
        """
        return self.predict(
            text=text,
            image_path=image_path,
            image_base64=image_base64,
            pdf_path=pdf_path,
            pdf_dpi=pdf_dpi,
        )

    def save(self, save_path: str | Path) -> None:
        """Save Prompter state to disk.

        Args:
            save_path: Path to save directory (will be created if doesn't exist).

        Raises:
            ValueError: If model is not set or not optimized.
            PersistenceError: If save fails.
        """
        if self.model is None:
            raise ValueError("model is required for saving")

        if not self.optimized_descriptions:
            raise ValueError("Prompter must be optimized before saving. Call optimize() first.")

        # Get model schema
        model_schema = self.model.model_json_schema()

        # Create state (model configuration not saved - user must configure DSPy separately)
        state = PrompterState(
            model_schema=model_schema,
            optimized_descriptions=self.optimized_descriptions,
            optimized_system_prompt=self.optimized_system_prompt,
            optimized_instruction_prompt=self.optimized_instruction_prompt,
            model_id="",  # Not used anymore, kept for backward compatibility
            model_config={},  # Not used anymore, kept for backward compatibility
            version=__version__,
            metadata={},
            optimized_demos=self.optimized_demos,
        )

        # Save
        save_prompter_state(state, save_path)
