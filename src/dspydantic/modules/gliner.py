"""DSPy module for optimizing GLiNER2 schema descriptions."""

from typing import Any

import dspy


class GLiNER2OptimizerModule(dspy.Module):
    """DSPy module for optimizing GLiNER2 schema element descriptions."""

    def __init__(
        self,
        schema_descriptions: dict[str, str] | None = None,
    ):
        """Initialize the GLiNER2 optimizer module.

        Args:
            schema_descriptions: Dictionary mapping schema element paths to their descriptions.
                Paths use dot notation: "entities.{name}", "relations.{name}", "classifications.{label}"
        """
        super().__init__()

        # Store schema descriptions for optimization
        self.schema_descriptions = schema_descriptions or {}

        # Create optimizers for each schema element description
        self.description_optimizers: dict[str, dspy.ChainOfThought] = {}
        for element_path, description in self.schema_descriptions.items():
            # Create a signature for optimizing this element's description
            signature = "description -> optimized_description"
            self.description_optimizers[element_path] = dspy.ChainOfThought(signature)

    def forward(
        self,
        **descriptions: str,
    ) -> dict[str, Any]:
        """Forward pass for optimization.

        Args:
            **descriptions: Schema element descriptions to optimize (keyed by element path).

        Returns:
            Dictionary with optimized descriptions.
        """
        optimized: dict[str, Any] = {}

        # Optimize each description
        for element_path, description in descriptions.items():
            if element_path in self.description_optimizers:
                optimizer = self.description_optimizers[element_path]
                result = optimizer(description=description)
                optimized[f"optimized_{element_path}"] = result.optimized_description

        return dspy.Prediction(**optimized)
