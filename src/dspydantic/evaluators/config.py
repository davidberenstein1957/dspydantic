"""Evaluator configuration system for dspydantic."""

from typing import Any, Protocol

import dspy


class BaseEvaluator(Protocol):
    """Protocol for all evaluators.

    All evaluators must implement the evaluate method that takes extracted and expected
    values and returns a score between 0.0 and 1.0.
    """

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate extracted value against expected value.

        Args:
            extracted: The extracted value to evaluate.
            expected: The expected value to compare against.
            input_data: Optional input data dictionary for context.
            field_path: Optional field path (e.g., "name", "address.street") for context.

        Returns:
            Score between 0.0 and 1.0, where 1.0 is a perfect match.
        """
        ...


# Evaluator registry - maps string names to evaluator classes
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_evaluator(name: str, evaluator_class: type[BaseEvaluator]) -> None:
    """Register a custom evaluator class.

    Args:
        name: String name to register the evaluator under.
        evaluator_class: The evaluator class that implements BaseEvaluator.
    """
    EVALUATOR_REGISTRY[name] = evaluator_class


# Type aliases for evaluator configuration
EvaluatorConfigValue = (
    str | dict[str, Any] | type[BaseEvaluator]
)  # String name, config dict, or class
EvaluatorConfig = dict[str, Any]  # Full evaluator configuration dict


class EvaluatorFactory:
    """Factory for creating evaluator instances from configuration."""

    @staticmethod
    def create(
        config: EvaluatorConfigValue,
        default_lm: dspy.LM | None = None,
    ) -> BaseEvaluator:
        """Create an evaluator instance from configuration.

        Args:
            config: Evaluator configuration. Can be:
                - String name (e.g., "exact", "text_similarity")
                - Dict with "type" and "config" keys
                - Dict with "class" and "config" keys (custom evaluator)
                - Evaluator class directly
            default_lm: Default LM to use if evaluator needs one and none is provided.

        Returns:
            Evaluator instance.

        Raises:
            ValueError: If config format is invalid or evaluator type not found.
        """
        # Handle string name
        if isinstance(config, str):
            if config not in EVALUATOR_REGISTRY:
                raise ValueError(
                    f"Unknown evaluator type: {config}. "
                    f"Available types: {list(EVALUATOR_REGISTRY.keys())}"
                )
            evaluator_class = EVALUATOR_REGISTRY[config]
            return evaluator_class(config={})

        # Handle evaluator class directly
        if isinstance(config, type):
            return config(config={})

        # Handle dict config
        if isinstance(config, dict):
            # Check for custom class
            if "class" in config:
                evaluator_class = config["class"]
                evaluator_config = config.get("config", {})
                # Inject default_lm if evaluator needs it and config doesn't have it
                if default_lm is not None and "lm" not in evaluator_config:
                    evaluator_config["lm"] = default_lm
                return evaluator_class(config=evaluator_config)

            # Check for type string
            if "type" in config:
                evaluator_type = config["type"]
                if evaluator_type not in EVALUATOR_REGISTRY:
                    raise ValueError(
                        f"Unknown evaluator type: {evaluator_type}. "
                        f"Available types: {list(EVALUATOR_REGISTRY.keys())}"
                    )
                evaluator_class = EVALUATOR_REGISTRY[evaluator_type]
                evaluator_config = config.get("config", {})
                # Inject default_lm if evaluator needs it and config doesn't have it
                if default_lm is not None and "lm" not in evaluator_config:
                    evaluator_config["lm"] = default_lm
                return evaluator_class(config=evaluator_config)

            raise ValueError(
                "Evaluator config dict must contain either 'type' or 'class' key"
            )

        raise ValueError(
            f"Invalid evaluator config type: {type(config)}. "
            "Must be str, dict, or evaluator class."
        )

    @staticmethod
    def parse_evaluator_config(
        config: EvaluatorConfigValue,
        default_lm: dspy.LM | None = None,
    ) -> tuple[str | type[BaseEvaluator], dict[str, Any]]:
        """Parse evaluator config and return type/class and config dict.

        Args:
            config: Evaluator configuration.
            default_lm: Default LM to use if evaluator needs one.

        Returns:
            Tuple of (evaluator_type_or_class, config_dict).
        """
        if isinstance(config, str):
            return config, {}
        if isinstance(config, type):
            return config, {}
        if isinstance(config, dict):
            if "class" in config:
                evaluator_config = config.get("config", {})
                if default_lm is not None and "lm" not in evaluator_config:
                    evaluator_config["lm"] = default_lm
                return config["class"], evaluator_config
            if "type" in config:
                evaluator_config = config.get("config", {})
                if default_lm is not None and "lm" not in evaluator_config:
                    evaluator_config["lm"] = default_lm
                return config["type"], evaluator_config
        raise ValueError(f"Invalid evaluator config: {config}")
