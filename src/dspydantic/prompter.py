"""Unified Prompter class for optimization and extraction."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel

# Import version directly to avoid circular import
try:
    from importlib.metadata import version

    __version__ = version("dspydantic")
except Exception:
    # Fallback if package not installed
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


class Prompter:
    """Unified class for optimizing and extracting with Pydantic models.

    This class combines optimization and extraction functionality in a single interface,
    similar to LangStruct's unified approach. It wraps PydanticOptimizer and adds
    extraction capabilities along with save/load functionality.

    Examples:
        Basic usage:

            from dspydantic import Prompter, Example
            from pydantic import BaseModel, Field

            class User(BaseModel):
                name: str = Field(description="User name")
                age: int = Field(description="User age")

            # Configure DSPy first
            import dspy
            lm = dspy.LM("openai/gpt-4o", api_key="your-key")
            dspy.configure(lm=lm)

            # Create prompter
            prompter = Prompter(model=User)

            # Optimize
            result = prompter.optimize(
                examples=[Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})]
            )

            # Save
            prompter.save("./my_prompter")

            # Load (DSPy must be configured before loading)
            prompter = Prompter.load("./my_prompter")

            # Predict
            data = prompter.predict("Jane Smith, 25")
    """

    def __init__(
        self,
        model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        optimized_descriptions: dict[str, str] | None = None,
        optimized_system_prompt: str | None = None,
        optimized_instruction_prompt: str | None = None,
        optimized_demos: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize Prompter.

        Args:
            model: Pydantic model class. Required unless loading from disk.
                Can be None - DSPydantic will auto-create an OutputModel from examples.
            system_prompt: Initial system prompt (optional).
            instruction_prompt: Initial instruction prompt (optional).
            optimized_descriptions: Pre-optimized field descriptions (for loading).
            optimized_system_prompt: Pre-optimized system prompt (for loading).
            optimized_instruction_prompt: Pre-optimized instruction prompt (for loading).

        Note:
            DSPy must be configured with `dspy.configure(lm=dspy.LM(...))` before using
            this class. See DSPy documentation for model configuration options.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt

        # Optimized state (set after optimization or loading)
        self.optimized_descriptions = optimized_descriptions or {}
        self.optimized_system_prompt = optimized_system_prompt
        self.optimized_instruction_prompt = optimized_instruction_prompt
        self.optimized_demos = optimized_demos

        # Internal optimizer (created lazily when needed)
        self._optimizer: PydanticOptimizer | None = None

    @classmethod
    def load(
        cls,
        load_path: str | Path,
        model: type[BaseModel] | None = None,
    ) -> "Prompter":
        """Load Prompter from disk.

        Args:
            load_path: Path to saved prompter directory.
            model: Optional Pydantic model class. If provided, will be used for extraction.
                If not provided, extraction will require model to be set later.

        Returns:
            Loaded Prompter instance.

        Raises:
            PersistenceError: If load fails or version is incompatible.

        Note:
            DSPy must be configured with `dspy.configure(lm=dspy.LM(...))` before using
            the loaded prompter. Model configuration is not saved - configure DSPy separately.
        """
        state = load_prompter_state(load_path)

        prompter = cls(
            model=model,
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
    ) -> "Prompter":
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

    def predict(
        self,
        text: str | dict[str, str] | None = None,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_dpi: int = 300,
    ) -> BaseModel:
        """Predict structured data from input.

        Args:
            text: Input text (str) or dict for template formatting.
            image_path: Path to image file.
            image_base64: Base64-encoded image string.
            pdf_path: Path to PDF file.
            pdf_dpi: DPI for PDF conversion (default: 300).

        Returns:
            Pydantic model instance with predicted data.

        Raises:
            ValueError: If model is not set or no input provided.
            ValidationError: If predicted data doesn't match model schema.
        """
        if self.model is None:
            raise ValueError("model is required for extraction")

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
                raise ValueError("At least one input parameter must be provided") from e

        # Get optimized descriptions (use original if not optimized)
        descriptions = self.optimized_descriptions or extract_field_descriptions(self.model)

        # Get prompts
        system_prompt = self.optimized_system_prompt or self.system_prompt
        instruction_prompt = self.optimized_instruction_prompt or self.instruction_prompt

        # Format instruction prompt if template
        if instruction_prompt and text_dict:
            instruction_prompt = format_instruction_prompt_template(instruction_prompt, text_dict) or instruction_prompt

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
                prompt_parts.append(f"\nInput images: {len(input_data['images'])} image(s) provided")
        else:
            prompt_parts.append(f"\nInput: {str(input_data)}")

        prompt_parts.append("\nExtract the structured data according to the JSON schema above and return it as valid JSON.")
        full_prompt = "\n\n".join(prompt_parts)
        json_prompt = f"{full_prompt}\n\nReturn only valid JSON, no other text."

        # Use configured DSPy LM (should be set via dspy.configure())
        if dspy.settings.lm is None:
            raise ValueError(
                "DSPy must be configured before extraction. "
                "Call dspy.configure(lm=dspy.LM(...)) first."
            )

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
        extracted_data = None
        try:
            extracted_data = json.loads(output_text)
        except (json.JSONDecodeError, AttributeError):
            # Try regex extraction
            json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
            json_match = re.search(json_pattern, output_text, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted_data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            pass

        if extracted_data is None:
            raise ValueError(f"Failed to extract JSON from LLM output: {output_text[:200]}")

        # Create optimized model and validate
        OptimizedModel = create_optimized_model(self.model, descriptions)
        return OptimizedModel.model_validate(extracted_data)

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
