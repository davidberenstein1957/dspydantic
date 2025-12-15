"""Base optimizer class with shared logic for all optimizers."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import dspy
from dspy.teleprompt import MIPROv2, Teleprompter

from dspydantic.types import Example, OptimizationResult


class BaseOptimizer(ABC):
    """Base class for optimizers with shared initialization and optimization logic."""

    def __init__(
        self,
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
        """Initialize the base optimizer.

        Args:
            examples: List of examples to use for optimization.
            evaluate_fn: Optional function that evaluates the quality of optimized prompts.
            lm: Optional DSPy language model instance.
            model_id: The model ID to use for optimization.
            api_key: Optional API key.
            api_base: Optional API base URL.
            api_version: Optional API version.
            num_threads: Number of threads for optimization.
            init_temperature: Initial temperature for optimization.
            verbose: If True, print detailed progress information.
            optimizer: Optimizer specification (string or Teleprompter instance).
            train_split: Fraction of examples to use for training.
            optimizer_kwargs: Optional dictionary of additional keyword arguments.
            exclude_fields: Optional list of field paths to exclude from evaluation.
        """
        if not examples:
            raise ValueError("At least one example must be provided")

        self.examples = examples
        self.evaluate_fn = evaluate_fn
        self.exclude_fields = exclude_fields
        self.lm = lm
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.api_version = api_version
        self.num_threads = num_threads
        self.init_temperature = init_temperature
        self.verbose = verbose
        self.train_split = train_split
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Handle optimizer parameter
        if optimizer is None:
            self.optimizer_type = self._auto_select_optimizer()
            self.custom_optimizer = None
        elif isinstance(optimizer, str):
            self.optimizer_type = optimizer.lower()
            teleprompter_classes = self._get_teleprompter_subclasses()
            if self.optimizer_type not in teleprompter_classes:
                valid_optimizers = sorted(teleprompter_classes.keys())
                raise ValueError(
                    f"optimizer '{optimizer}' is not a valid Teleprompter subclass. "
                    f"Valid options: {valid_optimizers}"
                )
            self.custom_optimizer = None
        elif isinstance(optimizer, Teleprompter):
            self.custom_optimizer = optimizer
            self.optimizer_type = "custom"
        else:
            raise TypeError(
                f"optimizer must be a string, Teleprompter instance, or None, "
                f"got {type(optimizer).__name__}"
            )

    @staticmethod
    def _get_teleprompter_subclasses() -> dict[str, type[Teleprompter]]:
        """Get all subclasses of Teleprompter and create a mapping by lowercase name.

        Returns:
            Dictionary mapping lowercase class names to Teleprompter subclasses.
        """
        # Get all subclasses recursively
        def get_all_subclasses(cls: type) -> set[type]:
            """Recursively get all subclasses of a class."""
            subclasses = set()
            for subclass in cls.__subclasses__():
                subclasses.add(subclass)
                subclasses.update(get_all_subclasses(subclass))
            return subclasses

        subclasses = get_all_subclasses(Teleprompter)
        # Create mapping: lowercase class name -> class
        mapping: dict[str, type[Teleprompter]] = {}
        for subclass in subclasses:
            # Skip abstract classes or classes that shouldn't be used directly
            if subclass.__name__ == "Teleprompter":
                continue
            # Map lowercase name to class
            mapping[subclass.__name__.lower()] = subclass

        # Add special case for miprov2zeroshot (which is MIPROv2 with zero-shot settings)
        if "miprov2" in mapping:
            mapping["miprov2zeroshot"] = mapping["miprov2"]

        return mapping

    def _auto_select_optimizer(self) -> str:
        """Auto-select the best optimizer based on the number of examples.

        Selection logic:
        - Very small datasets (1-2 examples): Use MIPROv2ZeroShot
        - Small datasets (3-19 examples): Use BootstrapFewShot
        - Larger datasets (>= 20 examples): Use BootstrapFewShotWithRandomSearch

        Returns:
            String name of the recommended optimizer type.
        """
        num_examples = len(self.examples)

        if num_examples <= 2:
            return "miprov2zeroshot"
        elif num_examples < 20:
            return "bootstrapfewshot"
        else:
            return "bootstrapfewshotwithrandomsearch"

    def _create_lm(self) -> dspy.LM:
        """Create and configure DSPy language model.

        Returns:
            Configured DSPy LM instance.
        """
        if self.lm is not None:
            return self.lm
        elif self.api_base:
            return dspy.LM(
                self.model_id,
                api_key=self.api_key,
                api_base=self.api_base,
                api_version=self.api_version,
            )
        else:
            return dspy.LM(self.model_id, api_key=self.api_key)

    def _create_optimizer_instance(
        self, metric: Callable[..., float]
    ) -> Teleprompter:
        """Create optimizer instance based on optimizer_type.

        Args:
            metric: Metric function for optimization.

        Returns:
            Teleprompter instance.
        """
        if self.custom_optimizer is not None:
            return self.custom_optimizer
        elif self.optimizer_type == "miprov2zeroshot":
            default_kwargs = {
                "metric": metric,
                "num_threads": self.num_threads,
                "init_temperature": self.init_temperature,
                "auto": "light",
                "max_bootstrapped_demos": 0,
                "max_labeled_demos": 0,
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            return MIPROv2(**merged_kwargs)
        elif self.optimizer_type == "miprov2":
            default_kwargs = {
                "metric": metric,
                "num_threads": self.num_threads,
                "init_temperature": self.init_temperature,
                "auto": "light",
            }
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            return MIPROv2(**merged_kwargs)
        else:
            teleprompter_classes = self._get_teleprompter_subclasses()
            if self.optimizer_type not in teleprompter_classes:
                valid_optimizers = sorted(teleprompter_classes.keys())
                raise ValueError(
                    f"Unknown optimizer_type: {self.optimizer_type}. "
                    f"Valid options: {valid_optimizers}"
                )

            optimizer_class = teleprompter_classes[self.optimizer_type]
            default_kwargs = {"metric": metric}
            # For BootstrapFewShot with small datasets, limit bootstrapped demos to avoid bugs
            if self.optimizer_type == "bootstrapfewshot" and len(self.examples) < 5:
                default_kwargs["max_bootstrapped_demos"] = min(2, len(self.examples) - 1)
            merged_kwargs = {**default_kwargs, **self.optimizer_kwargs}
            return optimizer_class(**merged_kwargs)

    def _split_examples(
        self, trainset: list[dspy.Example]
    ) -> tuple[list[dspy.Example], list[dspy.Example]]:
        """Split examples into training and validation sets.

        Args:
            trainset: List of DSPy examples.

        Returns:
            Tuple of (train_examples, val_examples).
        """
        split_idx = max(1, int(len(trainset) * self.train_split))
        train_examples = trainset[:split_idx]
        val_examples = trainset[split_idx:] if split_idx < len(trainset) else trainset
        return train_examples, val_examples

    def _compile_optimizer(
        self,
        optimizer: Teleprompter,
        program: Any,
        train_examples: list[dspy.Example],
        val_examples: list[dspy.Example],
    ) -> Any:
        """Compile optimizer with appropriate train/val sets.

        Args:
            optimizer: Teleprompter instance.
            program: DSPy program to optimize.
            train_examples: Training examples.
            val_examples: Validation examples.

        Returns:
            Optimized program.
        """
        optimizers_with_valset = (
            "miprov2zeroshot",
            "miprov2",
            "gepa",
            "bootstrapfewshotwithrandomsearch",
            "copro",
            "simba",
            "custom",
        )

        if self.optimizer_type in optimizers_with_valset:
            try:
                return optimizer.compile(
                    program, trainset=train_examples, valset=val_examples
                )
            except TypeError:
                if self.verbose:
                    print("Warning: Optimizer doesn't support valset, using trainset only")
                return optimizer.compile(program, trainset=train_examples)
        else:
            return optimizer.compile(program, trainset=train_examples)

    def _calculate_improvement(
        self, baseline_avg: float, avg_score: float
    ) -> tuple[float, float]:
        """Calculate improvement metrics.

        Args:
            baseline_avg: Baseline average score.
            avg_score: Optimized average score.

        Returns:
            Tuple of (improvement, improvement_pct).
        """
        improvement = avg_score - baseline_avg
        improvement_pct = (
            (improvement / baseline_avg * 100) if baseline_avg > 0 else 0.0
        )
        return improvement, improvement_pct

    def _print_optimization_summary(
        self,
        baseline_avg: float,
        avg_score: float,
        improvement: float,
        improvement_pct: float,
    ) -> None:
        """Print optimization summary.

        Args:
            baseline_avg: Baseline average score.
            avg_score: Optimized average score.
            improvement: Improvement value.
            improvement_pct: Improvement percentage.
        """
        if not self.verbose:
            return

        print(f"\n{'='*60}")
        print("Optimization complete")
        print(f"{'='*60}")
        print(f"Baseline score: {baseline_avg:.2%}")
        print(f"Final score: {avg_score:.2%}")
        if improvement > 0:
            print(f"Improvement: {improvement:+.2%} ({improvement_pct:+.1f}%)")
        elif improvement < 0:
            print(f"⚠️  Optimization decreased performance by {abs(improvement):.2%}")
            print("Using original descriptions instead.")
        else:
            print("No change in performance.")
        print(f"{'='*60}\n")

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _create_metric_function(self, lm: dspy.LM) -> Callable[..., float]:
        """Create a metric function for DSPy optimization.

        Args:
            lm: The DSPy language model.

        Returns:
            A function that evaluates prompt performance.
        """
        pass

    @abstractmethod
    def _prepare_dspy_examples(self) -> list[dspy.Example]:
        """Prepare examples as DSPy examples.

        Returns:
            List of dspy.Example objects.
        """
        pass

    @abstractmethod
    def _create_program(self) -> Any:
        """Create DSPy program for optimization.

        Returns:
            DSPy program instance.
        """
        pass

    @abstractmethod
    def _extract_optimized_descriptions(
        self, test_result: Any
    ) -> dict[str, str]:
        """Extract optimized descriptions from test result.

        Args:
            test_result: Result from optimized program.

        Returns:
            Dictionary of optimized descriptions.
        """
        pass

    @abstractmethod
    def _get_program_args(self) -> dict[str, Any]:
        """Get arguments for program execution.

        Returns:
            Dictionary of program arguments.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Optimize descriptions using DSPy.

        Returns:
            OptimizationResult containing optimized descriptions and metrics.
        """
        pass
