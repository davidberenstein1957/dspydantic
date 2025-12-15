"""Main optimizer class for GLiNER2 schemas using DSPy."""

from collections.abc import Callable
from typing import Any

import dspy
from dspy.teleprompt import Teleprompter

from dspydantic.evaluators.gliner import default_gliner_evaluate_fn
from dspydantic.extractors.gliner import (
    apply_optimized_gliner_descriptions,
    extract_gliner_descriptions,
)
from dspydantic.modules.gliner import GLiNER2OptimizerModule
from dspydantic.optimizers.base import BaseOptimizer
from dspydantic.types import Example, OptimizationResult


def _check_gliner_available() -> None:
    """Check if GLiNER2 is available, raise ImportError if not."""
    try:
        import gliner2  # noqa: F401
    except ImportError:
        raise ImportError(
            "GLiNER2 is required for GLiNER2SchemaOptimizer. "
            "Install it with: pip install dspydantic[gliner] or pip install gliner2"
        )


class GLiNER2SchemaOptimizer(BaseOptimizer):
    """Optimizer that uses DSPy to optimize GLiNER2 schema descriptions.

    This class optimizes descriptions in GLiNER2 schemas (entities, relations, classifications)
    by using DSPy to iteratively improve descriptions based on example data and a custom
    evaluation function.

    Examples:
        Basic usage::

            from gliner2 import GLiNER2
            from dspydantic import GLiNER2SchemaOptimizer, Example

            # Initialize GLiNER2 extractor
            extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

            # Define initial schema
            schema = {
                "entities": {
                    "person": "Names of people",
                    "location": "Places"
                },
                "relations": ["works_for", "located_in"]
            }

            # Create examples
            examples = [
                Example(
                    text="John works for Apple in Cupertino.",
                    expected_output={
                        "entities": {"person": ["John"], "location": ["Cupertino"]},
                        "relations": {"works_for": [("John", "Apple")]}
                    }
                )
            ]

            optimizer = GLiNER2SchemaOptimizer(
                extractor=extractor,
                schema=schema,
                examples=examples,
                model_id="gpt-4o",
                api_key="your-key"
            )

            result = optimizer.optimize()
            print(result.optimized_descriptions)
    """

    def __init__(
        self,
        extractor: Any,  # GLiNER2 extractor instance
        schema: dict[str, Any],
        examples: list[Example],
        evaluate_fn: Callable[[Example, dict[str, str], str | None, str | None], float]
        | Callable[[Example, dict[str, Any], dict[str, str], str | None, str | None], float]
        | dspy.LM
        | str
        | None = None,
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
        """Initialize the GLiNER2 schema optimizer.

        Args:
            extractor: GLiNER2 extractor instance (from GLiNER2.from_pretrained() or GLiNER2.from_api()).
            schema: GLiNER2 schema dictionary with entities, relations, and/or classifications.
            examples: List of examples to use for optimization.
            evaluate_fn: Optional function that evaluates the quality of optimized descriptions.
                When expected_output is provided:
                    - Takes (Example, optimized_descriptions dict, optimized_system_prompt,
                      optimized_instruction_prompt), returns a float score (0.0-1.0).
                    - Can also be a string: "exact" for exact matching, "levenshtein" for
                      Levenshtein distance-based matching, or None for default evaluation.
                When expected_output is None:
                    - Can be a dspy.LM instance to use as a judge.
                    - Can be a callable that takes (Example, extracted_data dict,
                      optimized_descriptions dict, optimized_system_prompt,
                      optimized_instruction_prompt) and returns a float score (0.0-1.0).
                    - If None, uses the default LLM judge (same LM as optimization).
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
            train_split: Fraction of examples to use for training (rest for validation).
            optimizer_kwargs: Optional dictionary of additional keyword arguments
                to pass to the optimizer constructor.
            exclude_fields: Optional list of field paths to exclude from evaluation.
                Field paths use dot notation (e.g., ["entities.person", "relations.works_for"]).

        Raises:
            ImportError: If GLiNER2 is not installed.
            ValueError: If at least one example is not provided, or if optimizer string
                is not a valid Teleprompter subclass name.
            TypeError: If optimizer is not a string, Teleprompter instance, or None.
        """
        _check_gliner_available()

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

        self.extractor = extractor
        self.schema = schema

        # Extract schema descriptions from GLiNER2 schema
        self.schema_descriptions = extract_gliner_descriptions(self.schema)

        # Check that we have something to optimize
        if not self.schema_descriptions:
            raise ValueError(
                "Schema must contain at least one entity, relation, or classification with descriptions"
            )


    def _create_program(self) -> Any:
        """Create DSPy program for optimization.

        Returns:
            DSPy program instance.
        """
        return GLiNER2OptimizerModule(schema_descriptions=self.schema_descriptions)

    def _extract_optimized_descriptions(
        self, test_result: Any
    ) -> dict[str, str]:
        """Extract optimized descriptions from test result.

        Args:
            test_result: Result from optimized program.

        Returns:
            Dictionary of optimized descriptions.
        """
        optimized_descriptions: dict[str, str] = {}
        for element_path in self.schema_descriptions.keys():
            attr_name = f"optimized_{element_path}"
            if hasattr(test_result, attr_name):
                optimized_descriptions[element_path] = getattr(test_result, attr_name)
        return optimized_descriptions

    def _get_program_args(self) -> dict[str, Any]:
        """Get arguments for program execution.

        Returns:
            Dictionary of program arguments.
        """
        program_args: dict[str, Any] = {}
        program_args.update(self.schema_descriptions)
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
            input_data = getattr(val_ex, "input_data", {})
            expected_output = getattr(val_ex, "expected_output", None)
            if not hasattr(val_ex, "expected_output"):
                expected_output = {}

            text = input_data.get("text") if isinstance(input_data, dict) else str(input_data)
            example_obj = Example(text=text, expected_output=expected_output)

            baseline_score = evaluate_fn(example_obj, self.schema_descriptions, None, None)
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
            val_program_args: dict[str, Any] = {}
            for element_path in self.schema_descriptions.keys():
                if hasattr(val_ex, element_path):
                    val_program_args[element_path] = getattr(val_ex, element_path)
                else:
                    val_program_args[element_path] = self.schema_descriptions[element_path]

            prediction = optimized_program(**val_program_args)

            pred_descriptions: dict[str, str] = {}
            for element_path in self.schema_descriptions.keys():
                attr_name = f"optimized_{element_path}"
                if hasattr(prediction, attr_name):
                    pred_descriptions[element_path] = getattr(prediction, attr_name)

            input_data = getattr(val_ex, "input_data", {})
            expected_output = getattr(val_ex, "expected_output", None)
            if not hasattr(val_ex, "expected_output"):
                expected_output = {}

            text = input_data.get("text") if isinstance(input_data, dict) else str(input_data)
            example_obj = Example(text=text, expected_output=expected_output)

            score = evaluate_fn(example_obj, pred_descriptions, None, None)
            evaluation_scores.append(score)

        return (
            sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0.0
        )

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
            evaluate_fn = default_gliner_evaluate_fn(
                self.extractor, self.schema, metric="exact", exclude_fields=self.exclude_fields
            )
        elif isinstance(evaluate_fn, str):
            # Handle string metrics: "exact" or "levenshtein"
            evaluate_fn_lower = evaluate_fn.lower()
            if evaluate_fn_lower in ("exact", "levenshtein"):
                evaluate_fn = default_gliner_evaluate_fn(
                    self.extractor,
                    self.schema,
                    metric=evaluate_fn_lower,
                    exclude_fields=self.exclude_fields,
                )
            else:
                valid_options = '"exact", "levenshtein"'
                raise ValueError(
                    f"evaluate_fn must be a callable, dspy.LM, None, or "
                    f'one of ({valid_options}), got "{evaluate_fn}"'
                )
        elif isinstance(evaluate_fn, dspy.LM):
            # If evaluate_fn is a dspy.LM, use it as judge when expected_output is None
            # For GLiNER2, we still need the extractor, so we'll use default with judge
            evaluate_fn = default_gliner_evaluate_fn(
                self.extractor, self.schema, metric="exact", exclude_fields=self.exclude_fields
            )
        elif callable(evaluate_fn):
            # Custom function - use as-is
            pass

        def metric_function(
            example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
        ) -> float:
            """Evaluate the quality of optimized descriptions.

            Args:
                example: The DSPy example.
                prediction: The optimized descriptions.
                trace: Optional trace from DSPy.

            Returns:
                A score between 0.0 and 1.0.
            """
            # Extract optimized descriptions from prediction
            optimized_descriptions: dict[str, str] = {}

            for key, value in prediction.items():
                if key.startswith("optimized_"):
                    # Extract element path (remove "optimized_" prefix)
                    element_path = key.replace("optimized_", "")
                    optimized_descriptions[element_path] = value

            # If no optimized values provided (baseline evaluation), use original
            if not optimized_descriptions:
                optimized_descriptions = self.schema_descriptions.copy()

            # Convert DSPy example to our Example type
            input_data = getattr(example, "input_data", {})
            expected_output = getattr(example, "expected_output", None)
            if not hasattr(example, "expected_output"):
                expected_output = {}

            # Reconstruct Example from input_data dictionary
            text = input_data.get("text") if isinstance(input_data, dict) else None
            if not text:
                text = str(input_data) if input_data else ""

            example_obj = Example(text=text, expected_output=expected_output)

            # Use the evaluation function
            score = evaluate_fn(example_obj, optimized_descriptions, None, None)

            # Ensure score is valid (between 0.0 and 1.0)
            if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
                if self.verbose:
                    print(f"Warning: Invalid score {score}, defaulting to 0.0")
                return 0.0

            return float(score)

        return metric_function

    def _prepare_dspy_examples(self) -> list[dspy.Example]:
        """Prepare examples as DSPy examples.

        Returns:
            List of dspy.Example objects.
        """
        trainset = []
        input_keys = list(self.schema_descriptions.keys())

        for ex in self.examples:
            # Convert input_data to dict if it's a Pydantic model
            input_data = ex.input_data
            if input_data is not None and hasattr(input_data, "model_dump"):
                input_data = input_data.model_dump()

            # Convert expected_output to dict if it's a Pydantic model
            expected_output_raw = ex.expected_output
            expected_output: dict[str, Any] | None = None
            if expected_output_raw is not None:
                if hasattr(expected_output_raw, "model_dump"):
                    expected_output = expected_output_raw.model_dump()  # type: ignore[union-attr]
                elif isinstance(expected_output_raw, dict):
                    expected_output = expected_output_raw

            example_dict: dict[str, Any] = {
                "input_data": input_data,
                "expected_output": expected_output if expected_output is not None else {},
            }
            # Add schema descriptions as inputs
            example_dict.update(self.schema_descriptions)

            trainset.append(dspy.Example(**example_dict).with_inputs(*input_keys))

        return trainset

    def optimize(self) -> OptimizationResult:
        """Optimize the GLiNER2 schema descriptions using DSPy.

        Returns:
            OptimizationResult containing optimized descriptions and metrics.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Starting DSPy GLiNER2 schema optimization")
            print(f"{'='*60}")
            print(f"Optimizer: {self.optimizer_type.upper()}")
            print(f"Examples: {len(self.examples)}")
            print(f"Schema elements to optimize: {len(self.schema_descriptions)}")
            if self.schema_descriptions:
                print("\nInitial schema descriptions:")
                for element_path, description in self.schema_descriptions.items():
                    print(f"  {element_path}: {description}")
            print(f"Optimization threads: {self.num_threads}")
            print(f"{'='*60}\n")

        # Create and configure LM
        lm = self._create_lm()
        dspy.configure(lm=lm)

        # Create evaluation function
        evaluate_fn = default_gliner_evaluate_fn(
            self.extractor, self.schema, metric="exact", exclude_fields=self.exclude_fields
        )

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
            print("\nOptimizing schema descriptions...")
            print(f"  - {len(self.schema_descriptions)} schema element descriptions")

        optimized_program = self._compile_optimizer(optimizer, program, train_examples, val_examples)

        # Get optimized descriptions
        program_args = self._get_program_args()
        test_result = optimized_program(**program_args)
        optimized_descriptions = self._extract_optimized_descriptions(test_result)

        # Evaluate optimized config
        if self.verbose:
            print("\nEvaluating optimized configuration...")
        avg_score = self._evaluate_optimized(val_examples, optimized_program, evaluate_fn)

        # Calculate improvement
        improvement, improvement_pct = self._calculate_improvement(baseline_avg, avg_score)

        # Only use optimized descriptions if they improve performance
        if improvement < 0:
            if self.verbose:
                print(
                    f"\n⚠️  WARNING: Optimization decreased performance by {abs(improvement):.2%}"
                )
                print("Keeping original descriptions instead of optimized ones.")
            optimized_descriptions = self.schema_descriptions.copy()
            avg_score = baseline_avg
            improvement = 0.0
            improvement_pct = 0.0

        # Build result
        result = OptimizationResult(
            optimized_descriptions=optimized_descriptions,
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
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

        return result

    def apply_optimized_schema(self) -> dict[str, Any]:
        """Apply optimized descriptions to the original schema and return updated schema.

        Returns:
            Updated GLiNER2 schema dictionary with optimized descriptions applied.
        """
        result = self.optimize()
        return apply_optimized_gliner_descriptions(self.schema, result.optimized_descriptions)
