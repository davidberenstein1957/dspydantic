"""Main optimizer class for Pydantic models using DSPy."""

from collections.abc import Callable
from typing import Any

import dspy
from dspy.teleprompt import Teleprompter
from pydantic import BaseModel

from dspydantic.evaluators.dspy import default_evaluate_fn
from dspydantic.extractors.dspy import extract_field_descriptions, extract_field_types
from dspydantic.hitl import HitlManager
from dspydantic.modules.dspy import PydanticOptimizerModule
from dspydantic.optimizers.base import BaseOptimizer
from dspydantic.types import Example, OptimizationResult
from dspydantic.utils import convert_images_to_dspy_images, format_instruction_prompt_template


class PydanticOptimizer(BaseOptimizer):
    """Optimizer that uses DSPy to optimize Pydantic model field descriptions.

    This class optimizes field descriptions in Pydantic models by using DSPy
    to iteratively improve descriptions based on example data and a custom
    evaluation function.

    Examples:
        Basic usage without evaluation function (uses default with "exact" metric)::

            from pydantic import BaseModel, Field
            from dspydantic import PydanticOptimizer

            class User(BaseModel):
                name: str = Field(description="User name")
                age: int = Field(description="User age")

            examples = [
                Example(
                    input_data={"text": "John Doe, 30 years old"},
                    expected_output={"name": "John Doe", "age": 30}
                )
            ]

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using "exact" metric for exact string matching::

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                evaluate_fn="exact",
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using "levenshtein" metric for fuzzy matching::

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                evaluate_fn="levenshtein",
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using "exact-hitl" or "levenshtein-hitl" for human-in-the-loop evaluation::

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                evaluate_fn="exact-hitl",  # or "levenshtein-hitl"
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using a custom evaluation function::

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

        Using a custom DSPy LM::

            import dspy
            custom_lm = dspy.LM("gpt-4o", api_key="your-key")
            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                lm=custom_lm
            )

        Passing optimizer as a string::

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                optimizer="miprov2"
            )

        Passing a custom optimizer instance::

            from dspy.teleprompt import MIPROv2
            custom_optimizer = MIPROv2(
                metric=lambda x, y, trace=None: 0.9,
                num_threads=8,
                auto="full"
            )
            optimizer = PydanticOptimizer(
                model=User,
                examples=examples,
                optimizer=custom_optimizer
            )

        Using None expected_output with LLM judge::

            examples_without_expected = [
                Example(
                    text="John Doe, 30 years old",
                    expected_output=None
                )
            ]
            optimizer = PydanticOptimizer(
                model=User,
                examples=examples_without_expected,
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using None expected_output with custom judge LM::

            import dspy
            judge_lm = dspy.LM("gpt-4o", api_key="your-key")
            optimizer = PydanticOptimizer(
                model=User,
                examples=examples_without_expected,
                evaluate_fn=judge_lm,
                model_id="gpt-4o",
                api_key="your-key"
            )

        Using None expected_output with custom judge function::

            def custom_judge(example, extracted_data, optimized_descriptions,
                            optimized_system_prompt, optimized_instruction_prompt):
                # Your custom evaluation logic here
                # Return a score between 0.0 and 1.0
                return 0.85

            optimizer = PydanticOptimizer(
                model=User,
                examples=examples_without_expected,
                evaluate_fn=custom_judge,
                model_id="gpt-4o",
                api_key="your-key"
            )

        Running optimization::

            result = optimizer.optimize()
            print(result.optimized_descriptions)
    """

    def __init__(
        self,
        model: type[BaseModel],
        examples: list[Example],
        evaluate_fn: Callable[[Example, dict[str, str], str | None, str | None], float]
        | Callable[[Example, dict[str, Any], dict[str, str], str | None, str | None], float]
        | dspy.LM
        | str
        | None = None,
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
        optimizer: str | Teleprompter | None = None,
        train_split: float = 0.8,
        optimizer_kwargs: dict[str, Any] | None = None,
        exclude_fields: list[str] | None = None,
    ) -> None:
        """Initialize the Pydantic optimizer.

        Args:
            model: The Pydantic model class to optimize.
            examples: List of examples to use for optimization.
            evaluate_fn: Optional function that evaluates the quality of optimized prompts.

                When expected_output is provided:
                    - Takes (Example, optimized_descriptions dict, optimized_system_prompt,
                      optimized_instruction_prompt), returns a float score (0.0-1.0).
                    - Can also be a string: "exact" for exact matching, "levenshtein" for
                      Levenshtein distance-based matching, "exact-hitl" for human-in-the-loop
                      exact evaluation (shows GUI popup), "levenshtein-hitl" for
                      human-in-the-loop Levenshtein evaluation (shows GUI popup), or None
                      for default evaluation that performs structured extraction with the same
                      LLM used for optimization.

                When expected_output is None:
                    - Can be a dspy.LM instance to use as a judge.
                    - Can be a callable that takes (Example, extracted_data dict,
                      optimized_descriptions dict, optimized_system_prompt,
                      optimized_instruction_prompt) and returns a float score (0.0-1.0).
                    - If None, uses the default LLM judge (same LM as optimization).
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
            optimizer: Optimizer specification. Can be:

                - A string (optimizer type name): e.g., "miprov2", "gepa",
                  "bootstrapfewshot", etc. If None, optimizer will be auto-selected
                  based on dataset size.
                - A Teleprompter instance: Custom optimizer instance to use directly.

                  Available optimizer types include: "miprov2", "miprov2zeroshot", "gepa",
                  "bootstrapfewshot", "bootstrapfewshotwithrandomsearch", "knnfewshot",
                  "labeledfewshot", "copro", "simba", and all other Teleprompter subclasses.
            train_split: Fraction of examples to use for training (rest for validation).
            optimizer_kwargs: Optional dictionary of additional keyword arguments
                to pass to the optimizer constructor. These will override default
                parameters. For example: {"max_bootstrapped_demos": 8, "auto": "full"}.
                Only used if `optimizer` is a string or None.
            exclude_fields: Optional list of field paths to exclude from evaluation.
                Field paths use dot notation for nested fields
                (e.g., ["address.street", "metadata"]).
                Fields matching these paths (or starting with them) will be excluded
                from scoring. Only applies when using default evaluation functions
                (not custom evaluate_fn).

        Raises:
            ValueError: If at least one example is not provided, or if optimizer string
                is not a valid Teleprompter subclass name.
            TypeError: If optimizer is not a string, Teleprompter instance, or None.
        """
        # Initialize base class
        super().__init__(
            examples=examples,
            evaluate_fn=evaluate_fn,
            lm=lm,
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            num_threads=num_threads,
            init_temperature=init_temperature,
            verbose=verbose,
            optimizer=optimizer,
            train_split=train_split,
            optimizer_kwargs=optimizer_kwargs,
            exclude_fields=exclude_fields,
        )

        self.model = model
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt

        # Extract field descriptions from Pydantic model
        # Field descriptions are automatically set from field names if not provided
        self.field_descriptions = extract_field_descriptions(self.model)

        # Extract field types from Pydantic model
        self.field_types = extract_field_types(self.model)

        # Initialize HITL manager
        self._hitl_manager = HitlManager(self)

        # Check that we have something to optimize
        has_field_descriptions = bool(self.field_descriptions)
        has_system_prompt = self.system_prompt is not None
        has_instruction_prompt = self.instruction_prompt is not None

        if not (has_field_descriptions or has_system_prompt or has_instruction_prompt):
            raise ValueError(
                "At least one of the following must be provided: "
                "model fields (field descriptions are automatically set from field names if not provided), "
                "system_prompt, or instruction_prompt"
            )


    def _create_metric_function(self, lm: dspy.LM) -> Callable[..., float]:
        """Create a metric function for DSPy optimization.

        Args:
            lm: The DSPy language model (needed for default evaluation function).

        Returns:
            A function that evaluates prompt performance.
        """
        return self._create_pydantic_metric_function(lm)

    def _default_evaluate_fn(
        self, lm: dspy.LM, metric: str = "exact", judge_lm: dspy.LM | None = None
    ) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
        """Create a default evaluation function that uses the LLM for structured extraction.

        Args:
            lm: The DSPy language model to use for extraction.
            metric: Comparison metric to use. Options:
                - "exact": Exact string matching (default)
                - "levenshtein": Levenshtein distance-based matching
            judge_lm: Optional separate LM to use as judge when expected_output is None.

        Returns:
            An evaluation function that performs structured extraction and compares
            with expected output (or uses judge if expected_output is None).
        """
        # Check if original evaluate_fn is a custom judge callable
        custom_judge_fn = None
        if (
            callable(self.evaluate_fn)
            and not isinstance(self.evaluate_fn, str)
            and not isinstance(self.evaluate_fn, dspy.LM)
        ):
            custom_judge_fn = self.evaluate_fn

        return default_evaluate_fn(
            lm=lm,
            model=self.model,
            system_prompt=self.system_prompt,
            instruction_prompt=self.instruction_prompt,
            metric=metric,
            judge_lm=judge_lm,
            custom_judge_fn=custom_judge_fn,
            exclude_fields=self.exclude_fields,
        )

    def _show_loading_window(
        self, evaluation_num: int | None = None, total_evaluations: int | None = None
    ) -> None:
        """Show a loading window while processing the next evaluation.

        Args:
            evaluation_num: Current evaluation number (1-based).
            total_evaluations: Total number of evaluations.
        """
        self._hitl_manager.show_loading_window(evaluation_num, total_evaluations)

    def _close_loading_window(self) -> None:
        """Close the loading window if it exists."""
        self._hitl_manager.close_loading_window()

    def _show_hitl_popup(
        self,
        input_text: str | None,
        images: list[str] | None,
        proposed_output: dict[str, Any],
        evaluation_num: int | None = None,
        total_evaluations: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Show a GUI popup for human-in-the-loop evaluation.

        Reuses the same window across evaluations, updating content in place.

        Args:
            input_text: Input text to display.
            images: List of base64-encoded images to display.
            proposed_output: Proposed output JSON to display and allow editing.
            evaluation_num: Current evaluation number (1-based).
            total_evaluations: Total number of evaluations.

        Returns:
            Tuple of (edited_output, was_edited) where edited_output is the final JSON
            and was_edited indicates if the user made changes.
        """
        return self._hitl_manager.show_hitl_popup(
            input_text=input_text,
            images=images,
            proposed_output=proposed_output,
            evaluation_num=evaluation_num,
            total_evaluations=total_evaluations,
        )

    def _close_hitl_window(self) -> None:
        """Close the HITL window if it exists."""
        self._hitl_manager.close_hitl_window()

    def _hitl_evaluate_fn(
        self, lm: dspy.LM, metric: str = "exact"
    ) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
        """Create a human-in-the-loop evaluation function.

        Delegates to HitlManager.

        Args:
            lm: The DSPy language model to use for extraction.
            metric: Comparison metric to use. Options:
                - "exact-hitl": Score is 0 if edited, 1 if not edited
                - "levenshtein-hitl": Score is Levenshtein distance if edited, 1 if not edited

        Returns:
            An evaluation function that shows a GUI popup for human review.
        """
        return self._hitl_manager.create_hitl_evaluate_fn(lm, metric)

    def _create_program(self) -> Any:
        """Create DSPy program for optimization.

        Returns:
            DSPy program instance.
        """
        return PydanticOptimizerModule(
            field_descriptions=self.field_descriptions,
            field_types=self.field_types,
            has_system_prompt=self.system_prompt is not None,
            has_instruction_prompt=self.instruction_prompt is not None,
        )

    def _extract_optimized_descriptions(
        self, test_result: Any
    ) -> dict[str, str]:
        """Extract optimized descriptions from test result.

        Args:
            test_result: Result from optimized program.

        Returns:
            Dictionary of optimized descriptions.
        """
        optimized_field_descriptions: dict[str, str] = {}
        for field_path in self.field_descriptions.keys():
            attr_name = f"optimized_{field_path}"
            if hasattr(test_result, attr_name):
                optimized_field_descriptions[field_path] = getattr(
                    test_result, attr_name
                )
        return optimized_field_descriptions

    def _get_program_args(self) -> dict[str, Any]:
        """Get arguments for program execution.

        Returns:
            Dictionary of program arguments.
        """
        program_args: dict[str, Any] = {}
        program_args.update(self.field_descriptions)
        # Add field types with field_type_ prefix
        for field_path, field_type in self.field_types.items():
            program_args[f"field_type_{field_path}"] = field_type
        if self.system_prompt is not None:
            program_args["system_prompt"] = self.system_prompt
        if self.instruction_prompt is not None:
            program_args["instruction_prompt"] = self.instruction_prompt
        return program_args

    def _evaluate_baseline(
        self, val_examples: list[dspy.Example], evaluate_fn: Callable
    ) -> float:
        """Evaluate baseline configuration.

        Args:
            val_examples: Validation examples.
            evaluate_fn: Evaluation function.

        Returns:
            Baseline average score.
        """
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
        return baseline_avg

    def _evaluate_optimized(
        self,
        val_examples: list[dspy.Example],
        optimized_program: Any,
        evaluate_fn: Callable,
    ) -> float:
        """Evaluate optimized configuration.

        Args:
            val_examples: Validation examples.
            optimized_program: Optimized program.
            evaluate_fn: Evaluation function.

        Returns:
            Optimized average score.
        """
        evaluation_scores = []
        for val_ex in val_examples:
            # Get optimized descriptions and prompts for this example
            val_program_args: dict[str, Any] = {}
            for field_path in self.field_descriptions.keys():
                if hasattr(val_ex, field_path):
                    val_program_args[field_path] = getattr(val_ex, field_path)
                else:
                    val_program_args[field_path] = self.field_descriptions[field_path]

            # Add field types
            for field_path in self.field_types.keys():
                field_type_key = f"field_type_{field_path}"
                if hasattr(val_ex, field_type_key):
                    val_program_args[field_type_key] = getattr(val_ex, field_type_key)
                else:
                    val_program_args[field_type_key] = self.field_types[field_path]

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

        return (
            sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0.0
        )

    def _create_pydantic_metric_function(self, lm: dspy.LM) -> Callable[..., float]:
        """Create a metric function for DSPy optimization (Pydantic-specific).

        Args:
            lm: The DSPy language model (needed for default evaluation function).

        Returns:
            A function that evaluates prompt performance.
        """
        # Use provided evaluate_fn or create default one
        evaluate_fn = self.evaluate_fn
        judge_lm: dspy.LM | None = None

        if evaluate_fn is None:
            evaluate_fn = self._default_evaluate_fn(lm)
        elif isinstance(evaluate_fn, str):
            # Handle string metrics: "exact", "levenshtein", "exact-hitl", or "levenshtein-hitl"
            evaluate_fn_lower = evaluate_fn.lower()
            if evaluate_fn_lower in ("exact", "levenshtein"):
                evaluate_fn = self._default_evaluate_fn(lm, metric=evaluate_fn_lower)
            elif evaluate_fn_lower in ("exact-hitl", "levenshtein-hitl"):
                evaluate_fn = self._hitl_evaluate_fn(lm, metric=evaluate_fn_lower)
            else:
                valid_options = '"exact", "levenshtein", "exact-hitl", "levenshtein-hitl"'
                raise ValueError(
                    f"evaluate_fn must be a callable, dspy.LM, None, or "
                    f'one of ({valid_options}), got "{evaluate_fn}"'
                )
        elif isinstance(evaluate_fn, dspy.LM):
            # If evaluate_fn is a dspy.LM, use it as judge when expected_output is None
            judge_lm = evaluate_fn
            evaluate_fn = self._default_evaluate_fn(lm, judge_lm=judge_lm)
        elif callable(evaluate_fn):
            # Check if it's a judge function (takes extracted_data) or regular eval function
            # We'll handle this in the metric_function wrapper
            pass

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
            expected_output = getattr(example, "expected_output", None)
            # Only use {} as default if expected_output attribute doesn't exist
            # If it exists but is None, keep it as None
            if not hasattr(example, "expected_output"):
                expected_output = {}

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
        expected_output = getattr(dspy_ex, "expected_output", None)
        # Only use {} as default if expected_output attribute doesn't exist
        # If it exists but is None, keep it as None
        if not hasattr(dspy_ex, "expected_output"):
            expected_output = {}

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

        # Add field type keys to input keys
        for field_path in self.field_types.keys():
            input_keys.append(f"field_type_{field_path}")

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

            # Add field types as inputs (with field_type_ prefix to distinguish from descriptions)
            for field_path, field_type in self.field_types.items():
                example_dict[f"field_type_{field_path}"] = field_type

            # Add prompts as inputs if they exist
            if self.system_prompt is not None:
                example_dict["system_prompt"] = self.system_prompt
            if self.instruction_prompt is not None:
                # Format instruction prompt template with example's text_dict for optimization
                # The optimizer will see formatted versions in examples, but we'll preserve
                # the template structure when optimizing
                if ex.text_dict:
                    formatted_instruction = format_instruction_prompt_template(
                        self.instruction_prompt, ex.text_dict
                    )
                    example_dict["instruction_prompt"] = (
                        formatted_instruction or self.instruction_prompt
                    )
                else:
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
            if self.field_descriptions:
                print("\nInitial field descriptions (set during initialization):")
                for field_path, description in self.field_descriptions.items():
                    print(f"  {field_path}: {description}")
            print(f"Optimization threads: {self.num_threads}")
            print(f"{'='*60}\n")

        # Create and configure LM
        lm = self._create_lm()
        dspy.configure(lm=lm)

        # Ensure we have a valid evaluation function
        evaluate_fn_raw = self.evaluate_fn
        judge_lm: dspy.LM | None = None

        if evaluate_fn_raw is None:
            evaluate_fn = self._default_evaluate_fn(lm)
        elif isinstance(evaluate_fn_raw, str):
            # Handle string metrics: "exact", "levenshtein", "exact-hitl", or "levenshtein-hitl"
            evaluate_fn_lower = evaluate_fn_raw.lower()
            if evaluate_fn_lower in ("exact", "levenshtein"):
                evaluate_fn = self._default_evaluate_fn(lm, metric=evaluate_fn_lower)
            elif evaluate_fn_lower in ("exact-hitl", "levenshtein-hitl"):
                evaluate_fn = self._hitl_evaluate_fn(lm, metric=evaluate_fn_lower)
            else:
                valid_options = '"exact", "levenshtein", "exact-hitl", "levenshtein-hitl"'
                raise ValueError(
                    f"evaluate_fn must be a callable, dspy.LM, None, or "
                    f'one of ({valid_options}), got "{evaluate_fn_raw}"'
                )
        elif isinstance(evaluate_fn_raw, dspy.LM):
            # If evaluate_fn is a dspy.LM, use it as judge when expected_output is None
            judge_lm = evaluate_fn_raw
            evaluate_fn = self._default_evaluate_fn(lm, judge_lm=judge_lm)
        elif callable(evaluate_fn_raw):
            # Custom function - use default wrapper, it will handle judge functions internally
            evaluate_fn = self._default_evaluate_fn(lm)
        else:
            raise TypeError(f"Unexpected type for evaluate_fn: {type(evaluate_fn_raw)}")

        # Create program and prepare examples
        program = self._create_program()
        trainset = self._prepare_dspy_examples()
        train_examples, val_examples = self._split_examples(trainset)

        if self.verbose:
            print(f"Training examples: {len(train_examples)}")
            print(f"Validation examples: {len(val_examples)}")

        # Create metric and optimizer
        metric = self._create_metric_function(lm)
        optimizer = self._create_optimizer_instance(metric)

        if self.custom_optimizer is not None and self.verbose:
            print(f"Using custom optimizer: {type(optimizer).__name__}")

        # Evaluate baseline
        if self.verbose:
            print("\nEvaluating baseline configuration...")
        baseline_avg = self._evaluate_baseline(val_examples, evaluate_fn)

        # Optimize
        if self.verbose:
            print("\nOptimizing prompts and field descriptions...")
            if self.system_prompt:
                print("  - System prompt")
            if self.instruction_prompt:
                print("  - Instruction prompt")
            if self.field_descriptions:
                print(f"  - {len(self.field_descriptions)} field descriptions")

        optimized_program = self._compile_optimizer(optimizer, program, train_examples, val_examples)

        # Get optimized descriptions
        program_args = self._get_program_args()
        test_result = optimized_program(**program_args)
        optimized_field_descriptions = self._extract_optimized_descriptions(test_result)

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

        # Evaluate optimized config
        if self.verbose:
            print("\nEvaluating optimized configuration...")
        avg_score = self._evaluate_optimized(val_examples, optimized_program, evaluate_fn)

        # Calculate improvement
        improvement, improvement_pct = self._calculate_improvement(baseline_avg, avg_score)

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

        self._print_optimization_summary(baseline_avg, avg_score, improvement, improvement_pct)

        # Close HITL window if it exists (optimization complete)
        self._close_hitl_window()

        return result

